from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from mf6pqc.coupling.common import run_reaction_step
from mf6pqc.energy import (
    capture_flow_inputs,
    setup_energy_coupling,
    update_energy_porosity,
    write_reference_conductivity,
)


class FakeModflowApi:
    def __init__(self, arrays: dict[tuple[str, str, str | None], np.ndarray]):
        self.arrays = arrays

    def get_var_address(self, variable, model, package=None):
        key = (variable, model, package)
        if key not in self.arrays:
            raise KeyError(key)
        return key

    def get_value_ptr(self, address):
        return self.arrays[address]


class RecordingChemistry:
    def __init__(self):
        self.events = []
        self.concentrations = None

    def SetTemperature(self, values):
        self.events.append(("temperature", np.asarray(values).copy()))

    def SetConcentrations(self, values):
        self.concentrations = np.asarray(values).copy()
        self.events.append(("concentrations", self.concentrations.copy()))

    def SetTime(self, value):
        self.time = value

    def SetTimeStep(self, value):
        self.time_step = value

    def RunCells(self):
        self.events.append(("run", None))

    def GetConcentrations(self):
        return self.concentrations


def make_sim() -> SimpleNamespace:
    arrays = {
        ("X", "gwe_model", None): np.array([20.0, 30.0]),
        ("POROSITY", "gwe_model", "EST"): np.array([0.30, 0.35]),
        ("VISC", "gwf_model", "VSC"): np.array([1.0e-3, 8.0e-4]),
        ("K11INPUT", "gwf_model", "NPF"): np.array([1.0, 2.0]),
        ("K22INPUT", "gwf_model", "NPF"): np.array([0.5, 1.0]),
        ("K33INPUT", "gwf_model", "NPF"): np.array([0.25, 0.5]),
        ("K11", "gwf_model", "NPF"): np.array([0.9, 2.2]),
        ("K22", "gwf_model", "NPF"): np.array([0.45, 1.1]),
        ("K33", "gwf_model", "NPF"): np.array([0.225, 0.55]),
    }
    return SimpleNamespace(
        energy_enabled=True,
        vsc_enabled=True,
        nxyz=2,
        temperature=np.array([20.0, 30.0]),
        porosity=np.array([0.30, 0.35]),
        flow_model_name="gwf_model",
        energy_model_name="gwe_model",
        npf_package_name="NPF",
        vsc_package_name="VSC",
        est_package_name="EST",
        validate_initial_gwe_fields=True,
        initial_gwe_field_tolerance=1.0e-12,
        sync_gwe_temperature_to_phreeqc=True,
        modflow_api=FakeModflowApi(arrays),
    )


class EnergyBindingTests(unittest.TestCase):
    def test_reference_k_is_updated_without_overwriting_vsc_effective_k(self):
        sim = make_sim()
        binding = setup_energy_coupling(sim)
        effective_before = binding.effective_k11_ptr.copy()

        capture_flow_inputs(sim, np.array([3.0, 4.0]))
        write_reference_conductivity(sim, np.array([3.0, 4.0]))

        np.testing.assert_array_equal(binding.reference_k11_ptr, [3.0, 4.0])
        np.testing.assert_array_equal(binding.reference_k22_ptr, [1.5, 2.0])
        np.testing.assert_array_equal(binding.reference_k33_ptr, [0.75, 1.0])
        np.testing.assert_array_equal(binding.effective_k11_ptr, effective_before)
        np.testing.assert_array_equal(binding.reference_k11_for_flow, [3.0, 4.0])

    def test_reaction_receives_latest_gwe_temperature_before_run_cells(self):
        sim = make_sim()
        setup_energy_coupling(sim)
        sim.energy_binding.temperature_ptr[:] = [42.0, 55.0]
        sim.phreeqc_rm = RecordingChemistry()
        reacted = np.empty(2)

        run_reaction_step(sim, np.array([1.0, 2.0]), reacted, 3.0, 0.25)

        self.assertEqual([event[0] for event in sim.phreeqc_rm.events], [
            "temperature",
            "concentrations",
            "run",
        ])
        np.testing.assert_array_equal(sim.temperature, [42.0, 55.0])
        np.testing.assert_array_equal(reacted, [1.0, 2.0])
        self.assertEqual(sim.phreeqc_rm.time, 3.0 * 86_400.0)
        self.assertEqual(sim.phreeqc_rm.time_step, 0.25 * 86_400.0)

    def test_reaction_porosity_is_written_to_gwe_est(self):
        sim = make_sim()
        binding = setup_energy_coupling(sim)
        update_energy_porosity(sim, np.array([0.31, 0.36]))
        np.testing.assert_array_equal(binding.est_porosity_ptr, [0.31, 0.36])


if __name__ == "__main__":
    unittest.main()
