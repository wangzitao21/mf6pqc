from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mf6pqc import MF6PQC, ProcessBackendFactory
from mf6pqc.backends import NativeBackendFactory

CASE = Path(__file__).resolve().parents[1] / "examples" / "ex011_PHT3D_11"


@unittest.skipUnless(
    os.environ.get("MF6PQC_NATIVE_TESTS") == "1" and CASE.is_dir(),
    "Set MF6PQC_NATIVE_TESTS=1 to exercise installed native chemistry",
)
class NativeCellIsolationTests(unittest.TestCase):
    def calculate(self, mapping, factory):
        with (
            tempfile.TemporaryDirectory() as temporary,
            MF6PQC(
                nxyz=len(mapping),
                nthreads=1,
                temperature=15.0,
                porosity=0.3,
                db_path=CASE / "input_data" / "database.dat",
                pqi_path=CASE / "input_data" / "input.pqi",
                output_dir=temporary,
                backend_factory=factory,
            ) as sim,
        ):
            sim.phreeqc_rm.SetUnitsKinetics(1)
            sim.phreeqc_rm.SetUnitsPPassemblage(1)
            sim.phreeqc_rm.SetRepresentativeVolume(np.full(len(mapping), 1.0 / 0.3))
            concentrations = sim.setup(
                {"solution": 0, "equilibrium_phases": 1, "kinetics": mapping}
            )
            outputs = []
            for step in range(3):
                sim.phreeqc_rm.advance(concentrations, step * 43200.0, 43200.0)
                concentrations = sim.phreeqc_rm.GetConcentrations()
                outputs.append(sim.phreeqc_rm.GetSelectedOutput().reshape(-1, len(mapping)))
            self.assertEqual(list(Path(temporary).glob("*_prm*")), [])
            return np.stack(outputs), concentrations.reshape(-1, len(mapping))

    def test_rates_are_independent_of_cell_order_and_process_partition(self):
        mapping = np.array([0, 1, 0, 1], dtype=np.int32)
        permutation = np.array([1, 3, 0, 2])
        expected = self.calculate(mapping, NativeBackendFactory())
        actual = self.calculate(mapping[permutation], ProcessBackendFactory(processes=2))
        for reference, values in zip(expected, actual, strict=True):
            np.testing.assert_allclose(
                values[..., np.argsort(permutation)], reference, rtol=1e-8, atol=1e-10
            )


if __name__ == "__main__":
    unittest.main()
