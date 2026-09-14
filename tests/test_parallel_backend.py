"""Check process transport of cell arrays, rollback, and error cleanup."""

import copy
import multiprocessing as mp
import os
import time
import unittest

import numpy as np

from mf6pqc.backends import CheckedPhreeqcRM
from mf6pqc.exceptions import BackendError
from mf6pqc.parallel import ProcessBackendFactory, _ProcessPhreeqcRM


class Reactor:
    def __init__(self, n):
        self.n = n
        self.c = np.zeros((3, n))
        self.phi = np.ones(n)
        self.ic = np.ones((7, n))
        self.dt = 0.0
        self.temperature = np.full(n, 25.0)
        self.time = 0.0
        self.saved = {}

    def SetConcentrations(self, x):
        self.c = np.array(x).reshape(3, self.n)
        return 0

    def GetConcentrations(self):
        return self.c.ravel()

    def SetPorosity(self, x):
        self.phi = np.array(x)
        return 0

    def SetTimeStep(self, x):
        self.dt = x
        return 0

    def SetTime(self, value):
        self.time = value
        return 0

    def SetTemperature(self, values):
        self.temperature = np.array(values)
        return 0

    def GetTemperature(self):
        return self.temperature

    def GetWorkerThreadLimits(self):
        return os.environ.get("OPENBLAS_NUM_THREADS"), os.environ.get("OMP_NUM_THREADS")

    def Pause(self):
        time.sleep(5)
        return 0

    def GetTimeStep(self):
        return self.dt

    def InitialPhreeqc2Module(self, x):
        self.ic = np.array(x).reshape(7, self.n)
        return 0

    def InitialPhreeqc2Module_mix(self, a, b, f):
        return self.InitialPhreeqc2Module(a * f + b * (1 - f))

    def InitialPhreeqc2Concentrations(self, x):
        return np.array([np.array(x) + 10 * i for i in range(3)]).ravel()

    def RunCells(self):
        self.c += self.dt * self.ic[6] / self.phi
        return 0

    def GetSelectedOutput(self):
        return self.c[[0, 2]].ravel()

    def GetDensityCalculated(self):
        return 1 + 0.01 * self.c[1]

    def StateSave(self, k):
        self.saved[k] = copy.deepcopy((self.c, self.ic))
        return 0

    def StateApply(self, k):
        self.c, self.ic = copy.deepcopy(self.saved[k])
        return 0

    def StateDelete(self, k):
        del self.saved[k]
        return 0

    def Fail(self):
        raise ValueError("deliberate worker failure")

    def Crash(self):
        os._exit(17)

    def SetInvalid(self):
        return -3

    def GetErrorString(self):
        return "deliberate negative status"

    def CloseFiles(self):
        return 0


def broken_constructor(n):
    raise RuntimeError("deliberate initialization failure")


class ProcessBackendTests(unittest.TestCase):
    def make(self, n=5, p=3, **options):
        backend = _ProcessPhreeqcRM(n, p, constructor=Reactor, **options)
        self.addCleanup(backend.close)
        return backend

    def test_uneven_partitions_mixed_conditions_and_boundary_vector(self):
        b = self.make()
        c = np.arange(15, dtype=float).reshape(3, 5)
        phi = np.linspace(0.2, 0.6, 5)
        first = np.arange(35, dtype=np.int32)
        second = first + 3
        fraction = np.tile(np.linspace(0.1, 0.9, 5), 7)
        b.SetConcentrations(c.ravel())
        b.SetPorosity(phi)
        b.InitialPhreeqc2Module_mix(first, second, fraction)
        b.SetTimeStep(0.25)
        b.RunCells()
        rate = (first * fraction + second * (1 - fraction)).reshape(7, 5)[6]
        expected = c + 0.25 * rate / phi
        np.testing.assert_array_equal(b.GetConcentrations().reshape(3, 5), expected)
        np.testing.assert_array_equal(b.GetSelectedOutput().reshape(2, 5), expected[[0, 2]])
        np.testing.assert_array_equal(b.GetDensityCalculated(), 1 + 0.01 * expected[1])
        np.testing.assert_array_equal(
            b.InitialPhreeqc2Concentrations([2, 7]), [2, 7, 12, 17, 22, 27]
        )
        self.assertEqual(b.GetGridCellCount(), 5)
        self.assertEqual(b.GetThreadCount(), 1)

    def test_batched_reaction_preserves_arrays_temperature_and_rollback(self):
        b = self.make()
        checked = CheckedPhreeqcRM(b)
        initial = np.arange(15, dtype=float)
        temperature = np.linspace(15, 35, 5)
        checked.SetConcentrations(initial)
        checked.StateSave(1)
        checked.advance(initial, 10.0, 0.25, temperature)
        np.testing.assert_array_equal(checked.GetConcentrations(), initial + 0.25)
        np.testing.assert_array_equal(
            checked.GetSelectedOutput(), (initial.reshape(3, 5) + 0.25)[[0, 2]].ravel()
        )
        np.testing.assert_array_equal(checked.GetTemperature(), temperature)
        checked.advance(initial, 10.0, 0.5)
        checked.StateApply(1)
        np.testing.assert_array_equal(checked.GetConcentrations(), initial)
        checked.commit_porosity(np.full(5, 0.5))
        np.testing.assert_array_equal(checked.GetConcentrations(), initial)
        checked.RunCells()
        np.testing.assert_array_equal(checked.GetConcentrations(), initial + 1.0)

    def test_shared_memory_exchange_matches_pipe_and_releases_buffers(self):
        from multiprocessing.shared_memory import SharedMemory

        reference = self.make(shared_memory=False)
        shared = self.make(shared_memory=True)
        initial = np.arange(15, dtype=float)
        for backend in (reference, shared):
            backend.SetConcentrations(initial)
            backend.GetSelectedOutput()
        for dt in (0.25, 0.5):
            expected = reference.advance(initial, 12.0, dt, np.arange(5, dtype=float) + 25)
            actual = shared.advance(initial, 12.0, dt, np.arange(5, dtype=float) + 25)
            for first, second in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(
                shared.commit_porosity(np.full(5, 0.5)),
                reference.commit_porosity(np.full(5, 0.5)),
            )
        previous = actual[0].copy()
        shared.advance(initial * 2, 13.0, 0.5)
        np.testing.assert_array_equal(actual[0], previous)
        names = [exchange.memory.name for exchange in shared._exchange]
        self.assertEqual(len(names), 3)
        shared.close()
        for name in names:
            with self.assertRaises(FileNotFoundError):
                SharedMemory(name=name)

    def test_workers_limit_nested_threads_without_changing_parent_environment(self):
        previous = dict(os.environ)
        b = self.make(2, 2)
        self.assertEqual(b.GetWorkerThreadLimits(), ("1", "1"))
        self.assertEqual(dict(os.environ), previous)

    def test_timeout_terminates_all_workers(self):
        b = self.make(2, 2)
        b.timeout = 0.05
        with self.assertRaisesRegex(BackendError, "timed out during Pause"):
            b.Pause()
        self.assertTrue(b.closed)
        self.assertTrue(all(not p.is_alive() for p in b.processes))

    def test_state_rollback(self):
        b = self.make()
        initial = np.arange(15, dtype=float)
        b.SetConcentrations(initial)
        b.InitialPhreeqc2Module(np.arange(35, dtype=np.int32))
        b.StateSave(1)
        b.SetTimeStep(0.5)
        b.RunCells()
        first = b.GetConcentrations()
        b.StateApply(1)
        np.testing.assert_array_equal(b.GetConcentrations(), initial)
        b.RunCells()
        np.testing.assert_array_equal(b.GetConcentrations(), first)
        b.StateDelete(1)

    def test_invalid_array_leaves_streams_usable_and_negative_getters_are_values(self):
        b = self.make()
        with self.assertRaises(ValueError):
            b.SetPorosity([0.3] * 4)
        b.SetConcentrations(np.arange(15, dtype=float))
        np.testing.assert_array_equal(b.GetConcentrations(), np.arange(15, dtype=float))
        b.SetTimeStep(-0.5)
        self.assertEqual(b.GetTimeStep(), -0.5)

    def test_worker_exceptions_negative_status_and_crash_cleanup(self):
        for method, message in [
            ("Fail", "deliberate worker failure"),
            ("SetInvalid", "negative status"),
            ("Crash", "exited"),
        ]:
            with self.subTest(method=method):
                b = self.make(2, 2)
                with self.assertRaisesRegex(BackendError, message):
                    getattr(b, method)()
                self.assertTrue(b.closed)
                self.assertTrue(all(not p.is_alive() for p in b.processes))

    def test_receive_more_than_windows_wait_handle_limit(self):
        backend = _ProcessPhreeqcRM.__new__(_ProcessPhreeqcRM)
        backend.closed = True
        backend.timeout = 1.0
        backend.processes = [None] * 70
        pairs = [mp.Pipe() for _ in backend.processes]
        for pair in pairs:
            for connection in pair:
                self.addCleanup(connection.close)
        backend.connections = [pair[0] for pair in pairs]
        for index, (_, sender) in reversed(list(enumerate(pairs))):
            sender.send((True, index))
        self.assertEqual(backend._receive_all("many workers"), list(range(70)))

    def test_shared_memory_is_released_when_a_worker_crashes(self):
        from multiprocessing.shared_memory import SharedMemory

        backend = self.make(2, 2)
        backend.GetSelectedOutput()
        backend.advance(np.arange(6, dtype=float), 0.0, 1.0)
        names = [exchange.memory.name for exchange in backend._exchange]
        with self.assertRaisesRegex(BackendError, "exited"):
            backend.Crash()
        self.assertTrue(all(not process.is_alive() for process in backend.processes))
        for name in names:
            with self.assertRaises(FileNotFoundError):
                SharedMemory(name=name)

    def test_initialization_failure(self):
        with self.assertRaisesRegex(BackendError, "initialization failure"):
            _ProcessPhreeqcRM(2, 2, constructor=broken_constructor)

    def test_shutdown_count_limit_and_factory_validation(self):
        b = self.make(2, 8)
        self.assertEqual(len(b.processes), 2)
        b.MpiWorkerBreak()
        b.close()
        self.assertTrue(all(not p.is_alive() for p in b.processes))
        with self.assertRaisesRegex(BackendError, "closed"):
            b.GetConcentrations()
        for count in [0, -1, True, 1.5]:
            with self.subTest(count=count), self.assertRaises((TypeError, ValueError)):
                ProcessBackendFactory(processes=count)
        with self.assertRaisesRegex(ValueError, "nthreads=1"):
            ProcessBackendFactory().create_phreeqcrm(5, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
