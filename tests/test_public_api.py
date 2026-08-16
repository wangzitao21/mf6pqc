from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

from mf6pqc import (
    BackendPaths,
    CouplingMethod,
    EnergyOptions,
    MF6PQC,
    SimulationConfig,
    mf6pqc,
)
from mf6pqc.coupling import get_coupling_runner
from mf6pqc.exceptions import ConfigurationError
from mf6pqc.input_processing import create_ic_array_from_map, setup_mixed_ic
from mf6pqc.output_processing import save_results
from mf6pqc.permeability import FluidAdjustedKozenyCarmanUpdater


class FakeChemistry:
    def __init__(self, nxyz: int) -> None:
        self.nxyz = nxyz
        self.closed = 0
        self.broken = 0

    def __getattr__(self, name):
        if name.startswith(("Set", "Use")) or name in {
            "LoadDatabase",
            "RunFile",
            "RunString",
        }:
            return lambda *args, **kwargs: None
        raise AttributeError(name)

    def OpenFiles(self) -> None:
        pass

    def FindComponents(self) -> int:
        return 1

    def GetComponents(self):
        return ["A"]

    def InitialPhreeqc2Module(self, values) -> None:
        self.initial_map = np.asarray(values).copy()

    def RunCells(self) -> None:
        pass

    def GetConcentrations(self):
        return np.full(self.nxyz, 0.25)

    def GetSelectedOutputHeadings(self):
        return ["A"]

    def GetSelectedOutput(self):
        return np.full(self.nxyz, 0.25)

    def InitialPhreeqc2Concentrations(self, values):
        return np.asarray(values, dtype=float)

    def CloseFiles(self) -> None:
        self.closed += 1

    def MpiWorkerBreak(self) -> None:
        self.broken += 1


class FakeFactory:
    def __init__(self) -> None:
        self.chemistry = None

    def create_phreeqcrm(self, nxyz: int, nthreads: int):
        self.chemistry = FakeChemistry(nxyz)
        return self.chemistry

    def create_modflow_api(self, dll_path: str, workspace: str):
        raise AssertionError("MODFLOW is not needed by this test")

    def load_modflow_simulation(self, modflow_api):
        raise AssertionError("MODFLOW is not needed by this test")


class PublicApiTests(unittest.TestCase):
    def test_pep8_and_historical_class_names_are_equivalent(self) -> None:
        self.assertIs(MF6PQC, mf6pqc)

    def test_coupling_method_aliases_are_normalized(self) -> None:
        method, runner = get_coupling_runner("standard")
        self.assertIs(method, CouplingMethod.SNIA)
        self.assertTrue(callable(runner))
        thermal_method, thermal_runner = get_coupling_runner("gwe_vsc")
        self.assertIs(thermal_method, CouplingMethod.THERMAL_SNIA)
        self.assertTrue(callable(thermal_runner))
        self.assertIs(
            get_coupling_runner("ThermalSNIA")[0], CouplingMethod.THERMAL_SNIA
        )
        with self.assertRaisesRegex(ValueError, "Unknown coupling method"):
            get_coupling_runner("invented")

    def test_reaction_schedule_rejects_unsupported_coupling_methods(self) -> None:
        simulator = object.__new__(mf6pqc)
        simulator._run_active = False
        simulator._run_completed = False
        simulator.reaction_steps = frozenset({1})
        simulator.energy_enabled = False
        with self.assertRaisesRegex(ConfigurationError, "only for SNIA"):
            simulator._run_coupling(lambda _: None, CouplingMethod.SIA)

    def test_structured_config_translates_paths_and_core_fields(self) -> None:
        config = SimulationConfig(
            case_name="case",
            nxyz=4,
            paths=BackendPaths("db", "input", "dll", "workspace", "output"),
        )
        values = config.to_legacy_kwargs()
        self.assertEqual(values["case_name"], "case")
        self.assertEqual(values["nxyz"], 4)
        self.assertEqual(values["db_path"], "db")
        self.assertEqual(values["porosity"], 0.35)
        self.assertFalse(values["energy_enabled"])
        self.assertIsNone(values["reaction_steps"])
        self.assertEqual(values["signed_components"], ("Charge",))

        thermal = SimulationConfig(
            case_name="thermal",
            nxyz=4,
            paths=BackendPaths("db", "input", "dll", "workspace", "output"),
            energy=EnergyOptions(enabled=True, viscosity_feedback=True),
        ).to_legacy_kwargs()
        self.assertTrue(thermal["energy_enabled"])
        self.assertTrue(thermal["vsc_enabled"])

    def test_fake_backend_supports_transactional_setup_and_idempotent_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / "database.dat"
            chemistry_input = root / "input.pqi"
            library = root / "libmf6.dll"
            database.touch()
            chemistry_input.touch()
            library.touch()
            workspace = root / "workspace"
            workspace.mkdir()
            factory = FakeFactory()
            simulator = MF6PQC(
                case_name="fake",
                nxyz=2,
                db_path=str(database),
                pqi_path=str(chemistry_input),
                modflow_dll_path=str(library),
                workspace=str(workspace),
                output_dir=str(root / "output"),
                backend_factory=factory,
            )
            initial = simulator.setup({"solution": 0})
            np.testing.assert_array_equal(initial, [0.25, 0.25])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                cached = simulator.setup({"solution": 1})
            np.testing.assert_array_equal(cached, initial)
            simulator.finalize()
            simulator.finalize()
            self.assertEqual(factory.chemistry.closed, 1)
            self.assertEqual(factory.chemistry.broken, 1)


class InputValidationTests(unittest.TestCase):
    def test_sia_rate_evaluator_requires_a_stateless_callable(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be callable"):
            MF6PQC(nxyz=1, sia_rate_evaluator=1)
        with self.assertRaisesRegex(
            ConfigurationError, "stateless aqueous-rate interface"
        ):
            MF6PQC(
                nxyz=1,
                if_update_porosity_K=True,
                sia_rate_evaluator=lambda *_: np.zeros((1, 1)),
            )

    def test_unknown_initial_condition_module_is_not_silently_ignored(self) -> None:
        with self.assertRaisesRegex(KeyError, "Unknown chemical module"):
            create_ic_array_from_map(2, {"equlibrium_phases": 1})

    def test_fraction_outside_unit_interval_is_rejected(self) -> None:
        class Chemistry:
            def InitialPhreeqc2Module_mix(self, *args):
                raise AssertionError("invalid fractions must fail before backend call")

        with self.assertRaisesRegex(ValueError, r"in \[0, 1\]"):
            setup_mixed_ic(
                Chemistry(), 2, {"solution": 0}, {"solution": 1}, [0.2, 1.1]
            )

    def test_vsc_rejects_competing_conductivity_owners(self) -> None:
        with self.assertRaisesRegex(ValueError, "boundary_conductance_updates"):
            MF6PQC(
                nxyz=1,
                energy_enabled=True,
                vsc_enabled=True,
                boundary_conductance_updates={
                    "GHB": {"cell_index": 0, "distance": 1.0}
                },
            )
        with self.assertRaisesRegex(ValueError, "apply viscosity.*twice"):
            MF6PQC(
                nxyz=1,
                energy_enabled=True,
                vsc_enabled=True,
                permeability_updater=FluidAdjustedKozenyCarmanUpdater(),
            )


class ResultSerializationTests(unittest.TestCase):
    def test_results_include_time_axis_and_machine_readable_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            save_results(
                temporary,
                "case",
                ["Ca"],
                np.array([[[1.0, 2.0]], [[3.0, 4.0]]]),
                [],
                [],
                [],
                False,
                False,
                result_times=[0.0, 1.5],
                metadata={"coupling_method": "SNIA"},
            )
            root = Path(temporary)
            np.testing.assert_array_equal(
                np.load(root / "results_times.npy"), [0.0, 1.5]
            )
            manifest = json.loads(
                (root / "results_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["run"]["coupling_method"], "SNIA")
            self.assertNotIn("has_energy", manifest)

    def test_thermal_results_are_validated_and_serialized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            frames = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
            thermal = {
                "temperature": [[20.0, 20.0], [40.0, 25.0]],
                "temperature_for_flow": [[20.0, 20.0], [20.0, 20.0]],
                "viscosity": [[1.0e-3, 1.0e-3], [7.0e-4, 9.0e-4]],
                "reference_K": [[1.0, 1.0], [1.1, 1.0]],
                "effective_K": [[1.0, 1.0], [1.5, 1.1]],
            }
            save_results(
                temporary,
                "thermal",
                ["A"],
                frames,
                [],
                [],
                [],
                False,
                False,
                result_times=[0.0, 1.0],
                energy_results=thermal,
            )
            root = Path(temporary)
            np.testing.assert_array_equal(
                np.load(root / "results_temperature.npy"), thermal["temperature"]
            )
            manifest = json.loads(
                (root / "results_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["has_energy"])
            self.assertIn("effective_K", manifest["energy"]["files"])


if __name__ == "__main__":
    unittest.main()
