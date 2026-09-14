"""Optional process parallelism for cell-local PhreeqcRM chemistry.

Each persistent worker owns an independent, single-threaded PhreeqcRM instance.
Cell arrays retain their global order at the MF6PQC boundary. PHREEQC input must
not depend on global CELL_NO values or on PUT/GET data shared between cells:
worker-local PHREEQC cell numbers differ from the transport grid numbers.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import threading
import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import wait

import numpy as np

from mf6pqc.backends import CheckedPhreeqcRM, NativeBackendFactory, advance_chemistry
from mf6pqc.chemistry_buffers import CellExchange
from mf6pqc.exceptions import BackendError
from mf6pqc.utils import require_integer

_SPAWN_LOCK = threading.Lock()
_WORKER_THREAD_LIMITS = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@contextlib.contextmanager
def _worker_environment():
    with _SPAWN_LOCK:
        previous = {key: os.environ.get(key) for key in _WORKER_THREAD_LIMITS}
        try:
            os.environ.update(dict.fromkeys(_WORKER_THREAD_LIMITS, "1"))
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _create_native_chemistry(count):
    return NativeBackendFactory().create_phreeqcrm(count, 1)


def _chemistry_worker(connection, count, constructor):
    chemistry = None
    exchange = None
    try:
        chemistry = CheckedPhreeqcRM(constructor(count))
        connection.send((True, None))
        while True:
            request = connection.recv()
            if request is None:
                break
            name, arguments = request
            try:
                if name == "attach_exchange":
                    attached = CellExchange.attach(arguments[0])
                    if exchange is not None:
                        exchange.close()
                    exchange = attached
                    result = None
                elif name == "advance_shared":
                    start_time, time_step, has_temperature = arguments
                    concentrations, selected = advance_chemistry(
                        chemistry,
                        exchange.concentrations.ravel(),
                        start_time,
                        time_step,
                        exchange.temperature if has_temperature else None,
                    )
                    if (
                        np.size(concentrations) != exchange.concentrations.size
                        or np.size(selected) != exchange.selected.size
                    ):
                        raise BackendError(
                            "Chemistry output size changed during shared-memory exchange"
                        )
                    exchange.concentrations.ravel()[:] = concentrations
                    exchange.selected.ravel()[:] = selected
                    result = None
                elif name == "commit_porosity_shared":
                    chemistry.SetPorosity(exchange.porosity)
                    exchange.concentrations.ravel()[:] = chemistry.GetConcentrations()
                    result = None
                elif name == "advance":
                    result = advance_chemistry(chemistry, *arguments)
                elif name == "commit_porosity":
                    chemistry.SetPorosity(arguments[0])
                    result = chemistry.GetConcentrations()
                else:
                    result = getattr(chemistry, name)(*arguments)
                connection.send((True, result))
            except Exception:
                connection.send((False, traceback.format_exc()))
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        with contextlib.suppress(BrokenPipeError, OSError):
            connection.send((False, traceback.format_exc()))
    finally:
        if chemistry is not None:
            with contextlib.suppress(Exception):
                chemistry.CloseFiles()
        if exchange is not None:
            exchange.close()
        connection.close()


class _ProcessPhreeqcRM:
    """PhreeqcRM surface used by MF6PQC, distributed over independent workers."""

    _CELL_SETTERS = frozenset(
        {
            "advance",
            "commit_porosity",
            "SetTemperature",
            "SetPressure",
            "SetPorosity",
            "SetSaturation",
            "SetSaturationUser",
            "SetDensityUser",
            "SetPrintChemistryMask",
            "SetRepresentativeVolume",
            "SetConcentrations",
        }
    )
    _CELL_GETTERS = frozenset(
        {
            "GetPorosity",
            "GetTemperature",
            "GetPressure",
            "GetConcentrations",
            "GetSelectedOutput",
            "GetDensityCalculated",
            "GetSaturationCalculated",
            "GetSolutionVolume",
        }
    )
    _MAPPING_SETTERS = frozenset({"InitialPhreeqc2Module", "InitialPhreeqc2Module_mix"})

    def __init__(
        self,
        count,
        processes,
        *,
        timeout=None,
        shared_memory=True,
        constructor=_create_native_chemistry,
    ):
        self.count = require_integer("nxyz", count)
        if timeout is not None and (not np.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be finite and positive")
        self.timeout = timeout
        self.shared_memory = shared_memory
        self._exchange = []
        self._selected_columns = None
        processes = min(self.count, require_integer("processes", processes))
        # Striding spreads a narrow reaction front over several workers.
        self.indices = tuple(np.arange(i, self.count, processes) for i in range(processes))
        self.slices = tuple(slice(i, None, processes) for i in range(processes))
        self.connections = []
        self.processes = []
        self.closed = False
        context = mp.get_context("spawn")
        try:
            for indices in self.indices:
                parent, child = context.Pipe()
                process = context.Process(
                    target=_chemistry_worker,
                    args=(child, indices.size, constructor),
                    daemon=True,
                )
                try:
                    with _worker_environment():
                        process.start()
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                self.connections.append(parent)
                self.processes.append(process)
            self._receive_all("initialization")
        except BaseException:
            self.close(terminate=True)
            raise

    def _receive_all(self, operation):
        remaining = set(range(len(self.processes)))
        values = [None] * len(self.processes)
        limit = self.timeout
        if limit is None and operation == "initialization":
            limit = 120.0
        deadline = None if limit is None else time.monotonic() + limit
        while remaining:
            watched = [self.connections[i] for i in remaining]
            timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
            if len(watched) <= 60:
                ready = wait(watched, timeout)
            else:
                ready = [
                    connection
                    for offset in range(0, len(watched), 60)
                    for connection in wait(watched[offset : offset + 60], 0)
                ]
                if not ready and (timeout is None or timeout > 0):
                    wait(watched[:60], 0.001 if timeout is None else min(timeout, 0.001))
                    continue
            if not ready:
                raise BackendError(
                    f"Chemistry workers {sorted(remaining)} timed out during {operation}"
                )
            for index in tuple(remaining):
                if self.connections[index] in ready:
                    try:
                        okay, value = self.connections[index].recv()
                    except (EOFError, OSError) as exc:
                        raise BackendError(
                            f"Chemistry worker {index} exited during {operation}"
                        ) from exc
                    if not okay:
                        raise BackendError(
                            f"Chemistry worker {index} failed during {operation}:\n{value}"
                        )
                    values[index] = value
                    remaining.remove(index)
        return values

    def _cell_matrix(self, value):
        array = np.asarray(value)
        if array.size == 0 or array.size % self.count:
            raise ValueError(f"Cell array size {array.size} is not a multiple of nxyz={self.count}")
        return array.reshape(-1, self.count)

    def _scatter(self, value, indices):
        return np.ascontiguousarray(self._cell_matrix(value)[:, indices].ravel())

    def _gather(self, values, name, out=None):
        columns, remainder = divmod(np.asarray(values[0]).size, self.indices[0].size)
        if remainder:
            raise BackendError(f"Invalid {name} array returned by chemistry worker 0")
        if out is None:
            result = np.empty((columns, self.count), dtype=np.asarray(values[0]).dtype)
        else:
            if out.size != columns * self.count:
                raise BackendError(f"Invalid destination shape for {name}")
            result = out.reshape(columns, self.count)
        for index, (indices, value) in enumerate(zip(self.indices, values, strict=True)):
            if np.asarray(value).size != columns * indices.size:
                raise BackendError(
                    f"Inconsistent {name} array returned by chemistry worker {index}"
                )
            result[:, self.slices[index]] = np.asarray(value).reshape(columns, indices.size)
        return result.ravel()

    def _call(self, name, arguments):
        if self.closed:
            raise BackendError("Process chemistry backend is closed")
        # Validate and prepare every message before sending any of them.
        messages = []
        for index, indices in enumerate(self.indices):
            args = list(arguments)
            if name == "attach_exchange":
                args = [arguments[0][index]]
            elif name in self._CELL_SETTERS:
                args[0] = self._scatter(args[0], indices)
                if name == "advance" and args[3] is not None:
                    args[3] = self._scatter(args[3], indices)
            elif name in self._MAPPING_SETTERS:
                args = [self._scatter(value, indices) for value in args]
            elif name == "SetFilePrefix":
                args[0] = f"{args[0]}_worker{index:02d}"
            messages.append((name, args))
        try:
            for connection, message in zip(self.connections, messages, strict=True):
                connection.send(message)
            values = self._receive_all(name)
            if name == "advance":
                return (
                    self._gather([value[0] for value in values], "GetConcentrations"),
                    self._gather([value[1] for value in values], "GetSelectedOutput"),
                )
            if name == "commit_porosity":
                return self._gather(values, "GetConcentrations")
            if name in self._CELL_GETTERS:
                result = self._gather(values, name)
                if name == "GetSelectedOutput":
                    self._selected_columns = result.size // self.count
                return result
            return values[0]
        except BaseException:
            self.close(terminate=True)
            raise

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*arguments):
            return self._call(name, arguments)

        setattr(self, name, call)
        return call

    def _ensure_exchange(self, ncomps):
        if (
            self._exchange
            and self._exchange[0].ncomps == ncomps
            and self._exchange[0].noutputs == self._selected_columns
        ):
            return
        exchanges = []
        try:
            for indices in self.indices:
                exchanges.append(CellExchange(indices.size, ncomps, self._selected_columns))
            self._call("attach_exchange", ([exchange.descriptor for exchange in exchanges],))
        except BaseException:
            for exchange in exchanges:
                exchange.close()
            raise
        for exchange in self._exchange:
            exchange.close()
        self._exchange = exchanges

    def _advance(self, concentrations, start_time, time_step, temperature=None, out=None):
        if self.closed:
            raise BackendError("Process chemistry backend is closed")
        if not self.shared_memory or not self._selected_columns:
            values = self._call("advance", (concentrations, start_time, time_step, temperature))
            if out is None:
                return values
            for destination, value in zip(out, values, strict=True):
                if destination.size != np.size(value):
                    raise BackendError("Chemistry output shape changed during reaction")
                np.copyto(destination.ravel(), np.asarray(value).ravel())
            return out
        values = self._cell_matrix(concentrations)
        temperatures = None if temperature is None else self._cell_matrix(temperature)
        if temperatures is not None and temperatures.shape[0] != 1:
            raise ValueError("Temperature must contain one value per cell")
        self._ensure_exchange(values.shape[0])
        for exchange, cells in zip(self._exchange, self.slices, strict=True):
            exchange.concentrations[:] = values[:, cells]
            if temperatures is not None:
                exchange.temperature[:] = temperatures[0, cells]
        self._call("advance_shared", (start_time, time_step, temperature is not None))
        return (
            self._gather(
                [exchange.concentrations for exchange in self._exchange],
                "GetConcentrations",
                None if out is None else out[0],
            ),
            self._gather(
                [exchange.selected for exchange in self._exchange],
                "GetSelectedOutput",
                None if out is None else out[1],
            ),
        )

    def advance(self, concentrations, start_time, time_step, temperature=None):
        return self._advance(concentrations, start_time, time_step, temperature)

    def advance_into(
        self, concentrations, start_time, time_step, reacted, selected, temperature=None
    ):
        self._advance(concentrations, start_time, time_step, temperature, (reacted, selected))

    def commit_porosity(self, porosity, *, out=None):
        if self.closed:
            raise BackendError("Process chemistry backend is closed")
        if not self._exchange:
            values = self._call("commit_porosity", (porosity,))
            if out is None:
                return values
            if out.shape != values.shape:
                raise BackendError("Chemistry concentration shape changed during porosity update")
            np.copyto(out, values)
            return out
        values = self._cell_matrix(porosity)
        if values.shape[0] != 1:
            raise ValueError("Porosity must contain one value per cell")
        for exchange, cells in zip(self._exchange, self.slices, strict=True):
            exchange.porosity[:] = values[0, cells]
        self._call("commit_porosity_shared", ())
        return self._gather(
            [exchange.concentrations for exchange in self._exchange], "GetConcentrations", out
        )

    def commit_porosity_into(self, porosity, concentrations):
        self.commit_porosity(porosity, out=concentrations)

    def GetThreadCount(self):
        """Each independent PhreeqcRM instance uses one native thread."""
        return 1

    def GetGridCellCount(self):
        return self.count

    def MpiWorkerBreak(self):
        self.close()
        return 0

    def close(self, *, terminate=False):
        if self.closed:
            return
        self.closed = True
        for connection, process in zip(self.connections, self.processes, strict=True):
            if terminate:
                if process.is_alive():
                    process.terminate()
            else:
                with contextlib.suppress(BrokenPipeError, EOFError, OSError):
                    connection.send(None)
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for connection in self.connections:
            connection.close()
        for exchange in self._exchange:
            exchange.close()
        self._exchange.clear()

    def __del__(self):
        if "closed" in self.__dict__:
            with contextlib.suppress(Exception):
                self.close(terminate=True)


@dataclass(frozen=True, slots=True)
class ProcessBackendFactory(NativeBackendFactory):
    """Opt-in process chemistry; MODFLOW still uses its normal native backend.

    Use ``nthreads=1`` and an executable script with a ``__main__`` guard.
    Reaction definitions must be cell-local (see this module's documentation).
    This is suitable for the independent kinetic cells of Xie B1-B3.
    """

    processes: int = 8
    timeout: float | None = None
    shared_memory: bool = True

    def __post_init__(self):
        require_integer("processes", self.processes)
        if self.timeout is not None and (not np.isfinite(self.timeout) or self.timeout <= 0):
            raise ValueError("timeout must be finite and positive")

    def create_phreeqcrm(self, nxyz, nthreads):
        if require_integer("nthreads", nthreads) != 1:
            raise ValueError("ProcessBackendFactory requires nthreads=1 per process")
        return _ProcessPhreeqcRM(
            nxyz, self.processes, timeout=self.timeout, shared_memory=self.shared_memory
        )
