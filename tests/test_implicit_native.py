"""Small closed 3-D native model for diffusion / multiple-mineral regression."""

from __future__ import annotations

import os
import platform
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mf6pqc import (
    MF6PQC,
    BackendPaths,
    CellFields,
    ImplicitOptions,
    KineticReaction,
    OutputOptions,
    SimulationConfig,
)
from mf6pqc.utils import get_gwt_model_name

ROOT = Path(__file__).resolve().parents[1]
_LIBRARY_NAME = {"Windows": "libmf6.dll", "Darwin": "libmf6.dylib"}.get(
    platform.system(), "libmf6.so"
)
LIBRARY = Path(
    os.environ.get(
        "MF6PQC_LIBMF6", Path(os.environ.get("MF6PQC_BIN", ROOT / "bin/mf6.8.0")) / _LIBRARY_NAME
    )
)
INPUT = ROOT / "examples/ex015_Xie2015_B2/input_data"


def build_closed_model(workspace, components, initial, steps, days):
    import flopy

    from mf6pqc.utils import get_gwt_model_name

    shape = (2, 2, 3)
    model = flopy.mf6.MFSimulation(sim_ws=workspace, verbosity_level=0)
    model.simulation_data.float_precision = 16
    model.simulation_data.float_characters = 24
    flopy.mf6.ModflowTdis(model, time_units="DAYS", perioddata=[(days, steps, 1.0)])
    gwf = flopy.mf6.ModflowGwf(model, modelname="gwf_model", save_flows=True)
    ims = flopy.mf6.ModflowIms(model, linear_acceleration="CG")
    model.register_ims_package(ims, [gwf.name])
    grid = dict(nlay=2, nrow=2, ncol=3, delr=0.5, delc=0.7, top=2.0, botm=[1.0, 0.0])
    flopy.mf6.ModflowGwfdis(gwf, **grid)
    flopy.mf6.ModflowGwfnpf(gwf, k=1.0, icelltype=0)
    flopy.mf6.ModflowGwfic(gwf, strt=2.0)
    cells = list(np.ndindex(shape))
    flopy.mf6.ModflowGwfchd(
        gwf, save_flows=True, stress_period_data=[(cell, 2.0) for cell in cells]
    )
    for i, component in enumerate(components):
        name = get_gwt_model_name(component)
        gwt = flopy.mf6.ModflowGwt(model, modelname=name)
        ims = flopy.mf6.ModflowIms(
            model,
            filename=name + ".ims",
            linear_acceleration="BICGSTAB",
            outer_dvclose=1e-10,
            inner_dvclose=1e-10,
            rcloserecord=1e-10,
            outer_maximum=100,
            inner_maximum=100,
        )
        model.register_ims_package(ims, [name])
        flopy.mf6.ModflowGwtdis(gwt, **grid)
        flopy.mf6.ModflowGwtic(gwt, strt=initial.reshape(len(components), *shape)[i])
        flopy.mf6.ModflowGwtmst(gwt, porosity=0.35)
        flopy.mf6.ModflowGwtadv(gwt, scheme="UPSTREAM")
        flopy.mf6.ModflowGwtdsp(gwt, diffc=0.003, xt3d_off=True)
        flopy.mf6.ModflowGwtssm(gwt, sources=None)
        # Reverse SRC order to test explicit native NODELIST mapping.
        flopy.mf6.ModflowGwtsrc(
            gwt, pname="SRC", stress_period_data=[(cell, 0.0) for cell in cells[::-1]]
        )
        flopy.mf6.ModflowGwfgwt(
            model,
            exgtype="GWF6-GWT6",
            exgmnamea=gwf.name,
            exgmnameb=name,
            filename=name + ".gwfgwt",
        )
    model.write_simulation(silent=True)


def calculate(directory, method, steps, days=2.0, dense_limit=400):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    text = (INPUT / "input.pqi").read_text()
    text = (
        text.replace(
            "-headings Calcite d_Calcite Gypsum d_Gypsum",
            "-headings Calcite d_Calcite Gypsum d_Gypsum SR_Calcite SR_Gypsum",
        )
        .replace(
            '40 PUNCH KIN_DELTA("Gypsum")',
            '40 PUNCH KIN_DELTA("Gypsum")\n50 PUNCH SR("Calcite"), SR("Gypsum")',
        )
        .replace("-high_precision false", "-high_precision true")
    )
    text += "\nKNOBS\n-convergence_tolerance 1e-12\nEND\n"
    (directory / "input.pqi").write_text(text)
    config = SimulationConfig(
        "diffusion",
        12,
        BackendPaths(
            INPUT / "database.dat",
            directory / "input.pqi",
            LIBRARY,
            directory / "model",
            directory / "output",
        ),
        nthreads=1,
        fields=CellFields(porosity=0.35),
        output=OutputOptions(save_steps=[steps]),
        fail_on_modflow_nonconvergence=True,
        implicit=ImplicitOptions(
            reactions=(
                KineticReaction(
                    "Calcite", {"Ca": 1, "C": 1, "O": 3}, 0.00432, surface_exponent=2 / 3
                ),
                KineticReaction("Gypsum", {"Ca": 1, "S": 1, "O": 6, "H": 4}, 0.00432),
            ),
            absolute_tolerance=1e-10,
            dense_limit=dense_limit,
        ),
    )
    with MF6PQC.from_config(config) as sim:
        initial = sim.setup({"solution": [0, 0, 1] * 4, "kinetics": 1})
        build_closed_model(config.paths.workspace, sim.components, initial, steps, days)
        initial_m = sim.selected_output[
            [sim.headings.index(n) for n in ("Calcite", "Gypsum")]
        ].copy()
        sim.run(method)
        final_m = sim.results[-1, [sim.headings.index(n) for n in ("Calcite", "Gypsum")]]
        final_c = np.array(
            [
                sim.modflow_api.get_value_ptr(
                    sim.modflow_api.get_var_address("X", get_gwt_model_name(c))
                )
                for c in sim.components
            ]
        )
        nu = np.array(
            [
                [r.stoichiometry.get(c, 0.0) for r in config.implicit.reactions]
                for c in sim.components
            ]
        )
        # Closed equal-volume cells: aqueous + mineral component totals must agree.
        budget = (
            0.35 * (final_c - initial.reshape(len(sim.components), 12)) + nu @ (final_m - initial_m)
        ).sum(axis=1)
        return final_m.copy(), final_c.copy(), budget, sim.last_run_wall_time_seconds


@unittest.skipUnless(
    os.environ.get("MF6PQC_NATIVE_TESTS") == "1" and LIBRARY.exists(),
    "Set MF6PQC_NATIVE_TESTS=1 for native 3-D kinetics tests",
)
class NativeImplicitTests(unittest.TestCase):
    def test_cell_kernel_matches_main_without_solution_zero(self):
        from mf6pqc import ChemistryOptions
        from mf6pqc.coupling.common import update_selected_output
        from mf6pqc.coupling.speciation_tangent import CellSpeciation

        case = ROOT / "examples/ex016_Xie2015_B3/input_data"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = (case / "input.pqi").read_text().replace("SOLUTION 0", "SOLUTION 37")
            (root / "input.pqi").write_text(source)
            config = SimulationConfig(
                "cell_kernel",
                2,
                BackendPaths(
                    case / "database.dat", root / "input.pqi", LIBRARY, root, root / "output"
                ),
                nthreads=1,
                fields=CellFields(porosity=0.35, pressure_atm=1.0),
                chemistry=ChemistryOptions(
                    transport_water_component=False, use_solution_density_volume=False
                ),
            )
            with MF6PQC.from_config(config) as sim:
                c = sim.setup({"solution": [37, 1], "kinetics": 1}).reshape(sim.ncomps, 2)
                rows = [
                    sim.headings.index("SI_" + n)
                    for n in (
                        "Calcite",
                        "Gypsum",
                        "Ferrihydrite",
                        "Jarosite",
                        "Gibbsite",
                        "Siderite",
                    )
                ]
                nu = np.zeros((sim.ncomps, 6))
                sim.implicit_diagnostics.update(chemistry_calls=0, max_speciation_drift=0.0)
                kernel = CellSpeciation(sim, nu, rows)
                try:
                    for phi in (0.35, 0.2):
                        sim.phreeqc_rm.SetPorosity(np.full(2, phi))
                        sim.phreeqc_rm.SetConcentrations(c.ravel())
                        sim.phreeqc_rm.SetTimeStep(0)
                        sim.phreeqc_rm.RunCells()
                        update_selected_output(sim)
                        expected = sim.selected_output[rows] * np.log(10)
                        for i in range(2):
                            value = kernel.evaluate(c[:, i : i + 1], i, phi, 0.0)
                            np.testing.assert_allclose(
                                value[:, 0], expected[:, i], atol=2e-7, rtol=0
                            )
                finally:
                    kernel.close()

    def test_two_minerals_3d_diffusion_and_matrix_free_agree_with_native_kinetics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            implicit = calculate(root / "implicit", "Implicit", 100)
            matrix_free = calculate(root / "matrix_free", "Implicit", 100, dense_limit=1)
            reference = calculate(root / "native", "SNIA", 1000)
            for result in (implicit, matrix_free):
                np.testing.assert_allclose(result[0], reference[0], atol=2e-5, rtol=0)
                np.testing.assert_allclose(result[1], reference[1], atol=2e-4, rtol=0)
                self.assertLess(np.max(abs(result[2])), 1e-7)
            np.testing.assert_allclose(implicit[0], matrix_free[0], atol=1e-8, rtol=0)


if __name__ == "__main__":
    unittest.main()
