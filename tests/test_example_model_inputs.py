"""Regression checks for example builder input contracts; no native solvers run."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mf6pqc.backends import NativeBackendFactory
from mf6pqc.utils import get_gwt_model_name

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
if not EXAMPLES.is_dir():
    raise unittest.SkipTest("Example sources are not included in this distribution")
if importlib.util.find_spec("flopy") is None:
    raise unittest.SkipTest("Example input checks require the optional flopy dependency")

import flopy  # noqa: E402


def load_builder(case: str):
    path = EXAMPLES / case / "modflow_model.py"
    spec = importlib.util.spec_from_file_location(f"test_inputs_{case}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(EXAMPLES), *sys.path]):
        spec.loader.exec_module(module)
    return module


class ExampleModelInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        for method in (
            "create_phreeqcrm",
            "create_modflow_api",
            "load_modflow_simulation",
        ):
            guard = patch.object(
                NativeBackendFactory,
                method,
                side_effect=AssertionError("These tests must not load native solvers"),
            )
            guard.start()
            self.addCleanup(guard.stop)
        guard = patch.object(
            flopy.mf6.MFSimulation,
            "run_simulation",
            side_effect=AssertionError("These tests must only write model inputs"),
        )
        guard.start()
        self.addCleanup(guard.stop)

    def test_pht3d03_keeps_outlet_active_and_comparison_times(self):
        with patch.object(sys, "path", [str(EXAMPLES), *sys.path]):
            runner = importlib.import_module("ex003_PHT3D_03.run")
        species = ["Ca", "Charge"]
        simulation = load_builder("ex003_PHT3D_03").build_model(
            workspace=self.workspace / "pht3d03",
            species=species,
            initial_concentrations=np.zeros(len(species) * runner.NXYZ),
            inflow_concentrations=np.array([0.01, -1e-5]),
            nlay=runner.NLAY,
            nrow=runner.NROW,
            ncol=runner.NCOL,
            delr=runner.DELR,
            delc=runner.DELC,
            top=runner.TOP,
            botm=runner.BOTM,
            perlen=runner.PERLEN,
            nstp=runner.NSTP,
            porosity=runner.POROSITY,
            hydraulic_conductivity=runner.HYDRAULIC_CONDUCTIVITY,
            initial_head=runner.INITIAL_HEAD,
            outlet_head=runner.OUTLET_HEAD,
            inflow_rate=runner.INFLOW_RATE,
            alh=runner.ALH,
            diffc=runner.DIFFC,
            mf6_executable="unused-mf6",
        )
        period = simulation.tdis.perioddata.get_data()[0]
        dt = period["perlen"] / period["nstp"]
        np.testing.assert_allclose(
            np.array([runner.NSTP // 4, runner.NSTP // 2, runner.NSTP]) * dt,
            [6.0, 12.0, 24.0],
        )
        gwf = simulation.get_model("gwf_model")
        outlet = gwf.get_package("CHD-OUTFLOW").stress_period_data.get_data(0)
        self.assertEqual(outlet["cellid"].tolist(), [(0, 0, 79)])
        np.testing.assert_array_equal(outlet["head"], [1.0])
        np.testing.assert_array_equal(outlet["CHARGE_OUT"], [0.0])
        inlet = gwf.get_package("WEL-1").stress_period_data.get_data(0)
        np.testing.assert_array_equal(inlet["q"], [0.007])
        np.testing.assert_array_equal(inlet["Charge"], [-1e-5])
        charge = simulation.get_model(get_gwt_model_name("Charge"))
        calcium = simulation.get_model(get_gwt_model_name("Ca"))
        self.assertEqual(
            [tuple(row) for row in charge.ssm.sources.get_data()],
            [("WEL-1", "AUX", "Charge"), ("CHD-OUTFLOW", "AUXMIXED", "CHARGE_OUT")],
        )
        self.assertEqual(
            [tuple(row) for row in calcium.ssm.sources.get_data()], [("WEL-1", "AUX", "Ca")]
        )
        for name in species:
            gwt = simulation.get_model(get_gwt_model_name(name))
            self.assertIsNone(gwt.get_package("CNC"))
            active = gwt.dis.idomain.array
            self.assertTrue(active is None or np.all(active == 1))
            np.testing.assert_array_equal(gwt.dis.delr.array, [0.005] * 80)
            np.testing.assert_allclose(gwt.mst.porosity.array, 0.35)
            np.testing.assert_allclose(gwt.dsp.alh.array, 0.005)

    def pht3d11_options(self, **overrides):
        options = dict(
            workspace=self.workspace / "pht3d11",
            species=["Cl"],
            initial_concentrations=np.full(18, 0.01),
            ambient_concentrations=np.array([0.02]),
            recharge_concentrations=np.array([0.03]),
            nlay=3,
            nrow=1,
            ncol=6,
            delr=0.5,
            delc=1.0,
            top=3.0,
            botm=[2.0, 1.0, 0.0],
            perlen=1.0,
            nstp=1,
            porosity=0.3,
            hydraulic_conductivity=1.0,
            outlet_head=3.0,
            recharge_rate=0.001,
            alh=0.1,
            alv=0.1,
            ath1=0.01,
            ath2=0.01,
            atv=0.01,
            diffc=0.0,
            boundary_conductivity=0.5,
            first_boundary_layer=1,
            left_rates=0.2,
            mf6_executable="unused-mf6",
        )
        return options | overrides

    def test_pht3d11_accepts_scalar_list_and_array_inputs(self):
        builder = load_builder("ex011_PHT3D_11").build_model
        variants = (
            ("scalar", 0.5, 0.0, 0.2),
            ("list", [0.5] * 6, [2.0, 1.0, 0.0], [0.2, 0.4]),
            (
                "array",
                np.arange(1, 7, dtype=float) / 10.0,
                np.array([2.0, 1.0, 0.0]),
                np.array([0.3, 0.6]),
            ),
        )
        for name, delr, botm, left_rates in variants:
            with self.subTest(form=name):
                workspace = self.workspace / name
                simulation = builder(
                    **self.pht3d11_options(
                        workspace=workspace, delr=delr, botm=botm, left_rates=left_rates
                    )
                )
                expected_widths = np.broadcast_to(delr, (6,))
                expected_bottoms = np.broadcast_to(np.asarray(botm).reshape(-1, 1, 1), (3, 1, 6))
                for model_name in ("gwf_model", get_gwt_model_name("Cl")):
                    model = simulation.get_model(model_name)
                    np.testing.assert_array_equal(model.dis.delr.array, expected_widths)
                    np.testing.assert_array_equal(model.dis.botm.array, expected_bottoms)
                wells = simulation.get_model("gwf_model").get_package("WEL-LEFT")
                records = wells.stress_period_data.get_data(0)
                np.testing.assert_array_equal(records["q"], np.broadcast_to(left_rates, (2,)))
                self.assertEqual(records["cellid"].tolist(), [(1, 0, 0), (2, 0, 0)])
                self.assertTrue((workspace / "mfsim.nam").is_file())

    def test_pht3d11_rejects_wrong_grid_lengths_before_creating_files(self):
        builder = load_builder("ex011_PHT3D_11").build_model
        for parameter, value, message in (
            ("delr", [0.5] * 5, "Cell widths"),
            ("botm", [1.0, 0.0], "Layer bottoms"),
        ):
            with self.subTest(parameter=parameter):
                options = self.pht3d11_options(**{parameter: value})
                with self.assertRaisesRegex(ValueError, message):
                    builder(**options)
                self.assertFalse(options["workspace"].exists())

    def test_pht3d11_rejects_wrong_boundary_rate_count_before_creating_files(self):
        builder = load_builder("ex011_PHT3D_11").build_model
        options = self.pht3d11_options(left_rates=[0.1, 0.2, 0.3])
        with self.assertRaises(ValueError):
            builder(**options)
        self.assertFalse(options["workspace"].exists())

    def xie_options(self, case: str, **overrides):
        options = dict(
            workspace=self.workspace / case,
            species=["Cl"],
            initial_concentrations=np.full(3, 0.01),
            inflow_concentrations=np.array([0.02]),
            nlay=1,
            nrow=1,
            ncol=3,
            delr=0.25,
            delc=1.0,
            top=1.0,
            botm=0.0,
            porosity=0.3,
            hydraulic_conductivity=1.0,
            vertical_conductivity_ratio=0.1,
            initial_head=1.0,
            alh=0.01,
            ath1=0.001,
            mf6_executable="unused-mf6",
        )
        if case == "ex017_Xie2015_B4":
            options.update(perlen=1.0, nstp=1, d0=0.001, boundary_head=1.0)
        else:
            options.update(
                period_data=[(1.0, 1, 1.0)],
                inlet_head=1.0,
                outlet_head=0.9,
                diffc=0.001,
                boundary_distance=0.1,
            )
        return options | overrides

    def test_xie_scalar_widths_are_shared_by_flow_and_transport(self):
        for number in range(14, 18):
            case = f"ex{number:03d}_Xie2015_B{number - 13}"
            with self.subTest(case=case):
                simulation = load_builder(case).build_model(**self.xie_options(case))
                for model_name in ("gwf_model", get_gwt_model_name("Cl")):
                    np.testing.assert_array_equal(
                        simulation.get_model(model_name).dis.delr.array, [0.25] * 3
                    )
                self.assertTrue((self.workspace / case / "mfsim.nam").is_file())

    def test_xie_rejects_wrong_width_count_before_creating_files(self):
        for number in range(14, 18):
            case = f"ex{number:03d}_Xie2015_B{number - 13}"
            with self.subTest(case=case):
                options = self.xie_options(case, delr=[0.25, 0.25])
                with self.assertRaisesRegex(ValueError, "Cell widths"):
                    load_builder(case).build_model(**options)
                self.assertFalse(options["workspace"].exists())

    def test_hamann_water_compensation_uses_the_evaporation_partition(self):
        with patch.object(sys, "path", [str(EXAMPLES), *sys.path]):
            runner = importlib.import_module("ex018_Hamann2015.run")
        grid = runner.Grid(delr=np.ones(100), delv=np.array([1.0, 1.0]), top=2.0)
        for split_x in (40.0, 50.0, 60.0):
            with self.subTest(split_x=split_x):
                evaporation = runner.evaporation_rates_mm_per_year(grid, split_x=split_x)
                sink = runner.water_only_sink_rates(grid, split_x=split_x).reshape(
                    grid.nlay, grid.nrow, grid.ncol
                )
                mask = grid.x_centres >= split_x
                np.testing.assert_array_equal(sink[0, 0, ~mask], 0.0)
                np.testing.assert_array_equal(sink[1:], 0.0)
                np.testing.assert_array_equal(
                    sink[0, 0, mask],
                    evaporation / 1000.0 / runner.DAYS_PER_YEAR * grid.delr[mask],
                )
                self.assertAlmostEqual(np.average(evaporation, weights=grid.delr[mask]), 80.0)

    def test_brine_rejects_invalid_components_before_creating_files(self):
        module = load_builder("ex021_Brine_Feedback2D")
        config = module.Config(
            nx=2,
            nz=2,
            length=2.0,
            height=2.0,
            width=1.0,
            days=1.0,
            dt=1.0,
            save_every=1.0,
            kv_ratio=0.1,
            injection_rate=0.1,
            injection_depth=2.0,
            extraction_depth=2.0,
            screen_conductance=1.0,
            outlet_head=2.0,
            alpha_l=0.1,
            alpha_t=0.01,
            diffusion=0.0,
        )
        for species, initial, message in (
            (["Cl", "Cl"], np.zeros(8), "Component names must be unique"),
            (["Cl", "Na"], np.zeros(7), "Concentrations have 7 entries; expected 8"),
        ):
            with self.subTest(message=message):
                workspace = self.workspace / "brine"
                with self.assertRaisesRegex(ValueError, message):
                    module.build_model(
                        workspace=workspace,
                        species=species,
                        initial_concentrations=initial,
                        inflow_concentrations=np.zeros(2),
                        config=config,
                        porosity=0.3,
                        hydraulic_conductivity=np.ones(config.nxyz),
                        reference_density=1000.0,
                        update_density=False,
                        mf6_executable="unused-mf6",
                    )
                self.assertFalse(workspace.exists())


if __name__ == "__main__":
    unittest.main()
