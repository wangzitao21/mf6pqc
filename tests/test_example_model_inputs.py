"""Validate common FloPy model contracts on small grids for every example."""

import importlib
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import flopy
import numpy as np

from mf6pqc.utils import get_gwt_model_name
from scripts.check_examples import discover_cases


def model_options(case, workspace):
    builder = importlib.import_module(f"examples.{case.name}.modflow_model")
    runner = importlib.import_module(f"examples.{case.name}.run")
    parameters = inspect.signature(builder.build_model).parameters
    options = dict(
        workspace=workspace,
        species=["Na", "Charge"],
        nlay=1,
        nrow=1,
        ncol=6,
        delr=1.0,
        delc=1.0,
        length=4.0,
        width=1.0,
        length_x=4.0,
        length_y=1.0,
        top=2.0,
        botm=0.0,
        perlen=1.0,
        nstp=2,
        tsmult=1.0,
        period_data=[(1.0, 2, 1.0), (1.0, 2, 1.0)],
        porosity=0.3,
        hydraulic_conductivity=1.0,
        vertical_conductivity_ratio=0.5,
        initial_head=2.0,
        inlet_head=2.1,
        outlet_head=2.0,
        inflow_rate=0.01,
        well_cell=(0, 0, 1),
        recharge_rate=0.001,
        boundary_conductivity=0.5,
        first_boundary_layer=0,
        left_rates=0.01,
        boundary_distance=0.5,
        alh=0.1,
        alv=0.1,
        ath1=0.01,
        ath2=0.01,
        atv=0.01,
        diffc=1e-5,
        pht3d_mobile_components=("Na", "Charge"),
        pulse_end=1.0,
        days_per_year=365.25,
        reference_density=1000.0,
        split_x=50.0,
        advection_scheme="UPSTREAM",
        boundary_node_species="Na",
        pore_velocity=0.1,
        pulse_duration=1.0,
        flush_duration=1.0,
        logical_steps_per_period=(2, 2),
        strang_half_steps=False,
        update_density=False,
        density_solid=2650.0,
        density_water=1000.0,
        heat_capacity_solid=800.0,
        heat_capacity_water=4184.0,
        initial_temperature=20.0,
        inflow_temperature=30.0,
        kts=2.0,
        ktw=0.6,
        thermal_a2=1.0,
        thermal_a3=1.0,
        thermal_a4=1.0,
        viscosity_reference=1.0,
    )
    nxyz = 6
    if "grid" in parameters:
        options["grid"] = grid = builder.Grid(delr=np.full(4, 25.0), delv=np.ones(2), top=2.0)
        options["time_config"] = runner.build_time_config()
        options["evaporation_rates"] = runner.evaporation_rates_mm_per_year(grid)
        nxyz = grid.nxyz
    if "config" in parameters:
        from dataclasses import replace

        options["config"] = config = replace(
            runner.DEFAULT_CONFIG,
            nx=4,
            nz=2,
            length=4.0,
            height=2.0,
            days=1.0,
            dt=0.5,
            save_every=0.5,
            injection_depth=2.0,
            extraction_depth=2.0,
        )
        options["hydraulic_conductivity"] = np.ones(config.nxyz)
        nxyz = config.nxyz
    options["initial_concentrations"] = np.r_[np.linspace(0.01, 0.02, nxyz), np.zeros(nxyz)]
    for name in parameters:
        if name.endswith("_concentrations") and name != "initial_concentrations":
            options[name] = np.array([0.02, 0.0])
    return builder.build_model, {k: v for k, v in options.items() if k in parameters}


class ExampleModelTests(unittest.TestCase):
    def test_flow_and_transport_share_grid_and_initial_fields(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(
                flopy.mf6.MFSimulation,
                "run_simulation",
                side_effect=AssertionError("Input generation must not run a solver"),
            ),
        ):
            for case in discover_cases():
                with self.subTest(case=case.name):
                    workspace = Path(temporary) / case.name
                    build, options = model_options(case, workspace)
                    simulation = build(**options)
                    flow = simulation.get_model("gwf_model")
                    nxyz = flow.modelgrid.nnodes
                    initial = options["initial_concentrations"].reshape(2, nxyz)
                    for i, component in enumerate(options["species"]):
                        transport = simulation.get_model(get_gwt_model_name(component))
                        self.assertEqual(transport.modelgrid.shape, flow.modelgrid.shape)
                        np.testing.assert_allclose(transport.dis.delr.array, flow.dis.delr.array)
                        np.testing.assert_allclose(transport.dis.botm.array, flow.dis.botm.array)
                        np.testing.assert_allclose(transport.ic.strt.array.ravel(), initial[i])
                        np.testing.assert_allclose(
                            transport.mst.porosity.array, options["porosity"]
                        )
                    self.assertTrue((workspace / "mfsim.nam").is_file())

    def test_invalid_concentration_size_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            for case in discover_cases():
                with self.subTest(case=case.name):
                    workspace = Path(temporary) / case.name
                    build, options = model_options(case, workspace)
                    options["initial_concentrations"] = np.zeros(3)
                    with self.assertRaises(ValueError):
                        build(**options)
                    self.assertFalse(workspace.exists())

    def test_duplicate_components_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            for case in discover_cases():
                with self.subTest(case=case.name):
                    workspace = Path(temporary) / case.name
                    build, options = model_options(case, workspace)
                    options["species"] = ["Na", "Na"]
                    with self.assertRaisesRegex(ValueError, "unique"):
                        build(**options)
                    self.assertFalse(workspace.exists())

    def test_adaptive_schedules_preserve_step_bounds_and_output_times(self):
        for case in discover_cases():
            runner = importlib.import_module(f"examples.{case.name}.run")
            if not hasattr(runner, "schedule"):
                continue
            with self.subTest(case=case.name):
                periods, saves, times = runner.schedule(20.0, 0.1, [1.0, 10.0, 20.0])
                dt = np.diff(np.r_[0, times]) / 365
                self.assertTrue(np.all(dt > 0))
                self.assertLessEqual(dt.max(), 0.1 * (1 + 1e-10))
                self.assertLessEqual(np.max(dt[1:] / dt[:-1]), 1.12 * (1 + 1e-9))
                self.assertEqual(sum(p[1] for p in periods), len(times))
                for target in (1.0, 10.0, 20.0):
                    self.assertLess(np.min(abs(times[np.array(saves) - 1] / 365 - target)), 1e-7)

    def test_solution_normalization_preserves_charge_and_element_ratios(self):
        names = ["H", "O", "Charge", "Ca", "C", "S", "Na"]
        expected = np.array([111, 55.5, -0.02, 0.0001, 0.01, 0.2, 0.39562])
        solution = {
            component: expected[names.index(component)] for component in ("Ca", "C", "S", "Na")
        }
        for case in discover_cases():
            runner = importlib.import_module(f"examples.{case.name}.run")
            if not hasattr(runner, "normalize_solution"):
                continue
            with self.subTest(case=case.name):
                actual = runner.normalize_solution(expected * 1.02673957, names, solution)
                np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=0)
                wrong = expected.copy()
                wrong[5] *= 2
                with self.assertRaisesRegex(ValueError, "total"):
                    runner.normalize_solution(wrong, names, solution)
