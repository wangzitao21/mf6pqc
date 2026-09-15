from __future__ import annotations

import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parents[1]))
import flopy
import numpy as np

from mf6pqc.utils import get_gwt_model_name


def build_model(
    *,
    workspace: str | Path,
    species: list[str],
    initial_concentrations: np.ndarray,
    inflow_concentrations: np.ndarray,
    nlay: int,
    nrow: int,
    ncol: int,
    delr: float | list[float] | np.ndarray,
    delc: float | list[float] | np.ndarray,
    top: float | list[float] | np.ndarray,
    botm: float | list[float] | np.ndarray,
    period_data: list[tuple[float, int, float]],
    porosity: float,
    hydraulic_conductivity: float | np.ndarray,
    vertical_conductivity_ratio: float,
    initial_head: float | np.ndarray | None,
    inlet_head: float,
    outlet_head: float,
    alh: float,
    ath1: float,
    diffc: float,
    boundary_distance: float,
    diffusion_boundary: bool = False,
    node_coordinates: np.ndarray | None = None,
    mf6_executable: str | Path | None = None,
) -> flopy.mf6.MFSimulation:
    """Build and write the MODFLOW 6 inputs and return the unrun simulation."""
    workspace = Path(workspace).expanduser().resolve()
    species = list(species)
    if len(set(species)) != len(species):
        raise ValueError("Component names must be unique")
    nxyz = nlay * nrow * ncol
    initial_fields = dict(
        zip(species, np.asarray(initial_concentrations).reshape(len(species), nxyz), strict=True)
    )
    if nlay != 1 or nrow != 1:
        raise ValueError("This example requires one layer and one row")
    inflow_concentrations = np.asarray(inflow_concentrations).reshape(len(species))
    if mf6_executable is None:
        mf6_executable = "mf6"
    gwf_name = "gwf_model"
    if np.ndim(delr) > 0 and np.size(delr) != ncol:
        raise ValueError("Cell widths must be a scalar or match ncol")
    nper = len(period_data)
    conductivity = np.broadcast_to(hydraulic_conductivity, (nlay, nrow, ncol))
    left_conductance = float(conductivity[0, 0, 0]) / boundary_distance
    right_conductance = float(conductivity[0, 0, -1]) / boundary_distance
    simulation = flopy.mf6.MFSimulation(
        sim_name="model", sim_ws=str(workspace), exe_name=mf6_executable, verbosity_level=0
    )
    simulation.simulation_data.float_precision = 16
    simulation.simulation_data.float_characters = 24
    flopy.mf6.ModflowTdis(
        simulation, pname="tdis", time_units="DAYS", nper=nper, perioddata=period_data
    )
    gwf = flopy.mf6.ModflowGwf(simulation, modelname=gwf_name, save_flows=False)
    flow_ims = flopy.mf6.ModflowIms(
        simulation,
        pname="ims",
        complexity="SIMPLE",
        outer_dvclose=1e-10,
        outer_maximum=50,
        under_relaxation="NONE",
        inner_maximum=500,
        inner_dvclose=1e-10,
        rcloserecord=1e-10,
        linear_acceleration="CG",
        scaling_method="DIAGONAL",
        reordering_method="RCM",
        relaxation_factor=0.97,
    )
    simulation.register_ims_package(flow_ims, [gwf.name])
    discretization = dict(nlay=nlay, nrow=nrow, ncol=ncol, delr=delr, delc=delc, top=top, botm=botm)
    if node_coordinates is not None:
        x = np.asarray(node_coordinates, dtype=float)
        if x.shape != (ncol,) or np.any(np.diff(x) <= 0):
            raise ValueError("Node coordinates must be strictly increasing and match ncol")
        iac, ja, ihc, cl12, hwva, angle = ([], [], [], [], [], [])
        for i in range(ncol):
            neighbours = [i] + [j for j in (i - 1, i + 1) if 0 <= j < ncol]
            iac.append(len(neighbours))
            for j in neighbours:
                ja.append(j)
                ihc.append(1 if i != j else 0)
                cl12.append(abs(x[j] - x[i]) / 2 if i != j else 0.0)
                hwva.append(float(np.asarray(delc).ravel()[0]))
                angle.append(0.0 if j >= i else 180.0)
        edges = np.r_[x[0], (x[:-1] + x[1:]) / 2, x[-1]]
        height = float(np.asarray(delc).ravel()[0])
        vertices = [(2 * i, float(edge), 0.0) for i, edge in enumerate(edges)]
        vertices += [(2 * i + 1, float(edge), height) for i, edge in enumerate(edges)]
        vertices.sort()
        cells = [
            (i, float(x[i]), height / 2, 4, 2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2)
            for i in range(ncol)
        ]
        discretization = dict(
            nvert=len(vertices),
            vertices=vertices,
            cell2d=cells,
            nodes=ncol,
            nja=len(ja),
            top=np.broadcast_to(top, (ncol,)),
            bot=np.broadcast_to(botm, (ncol,)),
            area=np.broadcast_to(delr, (ncol,)) * float(np.asarray(delc).ravel()[0]),
            iac=iac,
            ja=ja,
            ihc=ihc,
            cl12=cl12,
            hwva=hwva,
            angldegx=angle,
        )
        flopy.mf6.ModflowGwfdisu(gwf, pname="dis", **discretization)
    else:
        flopy.mf6.ModflowGwfdis(gwf, pname="dis", **discretization)

    def cell_id(column):
        return (column,) if node_coordinates is not None else (0, 0, column)

    flopy.mf6.ModflowGwfnpf(
        gwf,
        pname="npf",
        save_flows=True,
        save_specific_discharge=True,
        icelltype=0,
        k=hydraulic_conductivity,
        k33=hydraulic_conductivity * vertical_conductivity_ratio,
    )
    flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=initial_head)
    flopy.mf6.ModflowGwfsto(gwf, pname="sto", save_flows=False, iconvert=1, ss=0.0, sy=0.0)
    ghb_spd = [[cell_id(ncol - 1), outlet_head, right_conductance, 1.0]]
    flopy.mf6.ModflowGwfghb(
        gwf,
        pname="ghb_right",
        save_flows=True,
        maxbound=len(ghb_spd),
        auxiliary=["CHARGE_OUT_LIMIT"],
        stress_period_data={0: ghb_spd},
        filename=f"{gwf_name}.choushui.ghb",
    )
    ghb2_spd = [[cell_id(0), inlet_head, left_conductance, *inflow_concentrations]]
    flopy.mf6.ModflowGwfghb(
        gwf,
        pname="bushui",
        save_flows=True,
        maxbound=len(ghb2_spd),
        stress_period_data={0: ghb2_spd},
        auxiliary=species,
        filename=f"{gwf_name}.bushui.ghb",
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        pname="oc",
        budget_filerecord=f"{gwf_name}.bud",
        head_filerecord=f"{gwf_name}.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    src_data_list = [
        (cell_id(column), 0.0)
        for layer in range(nlay)
        for row in range(nrow)
        for column in range(ncol)
    ]
    src_maxbound = len(src_data_list)
    nouter, ninner = (50, 100)
    hclose, rclose, relax = (1e-12, 1e-12, 1.0)
    for species_name, concentration in initial_fields.items():
        gwt_name = get_gwt_model_name(species_name)
        gwt = flopy.mf6.ModflowGwt(
            simulation, modelname=gwt_name, save_flows=False, model_nam_file=f"{gwt_name}.nam"
        )
        transport_ims = flopy.mf6.ModflowIms(
            simulation,
            print_option="NONE",
            outer_dvclose=hclose,
            outer_maximum=nouter,
            under_relaxation="NONE",
            inner_maximum=ninner,
            inner_dvclose=hclose,
            rcloserecord=[rclose, "relative_rclose"],
            linear_acceleration="BICGSTAB",
            scaling_method="NONE",
            reordering_method="NONE",
            relaxation_factor=relax,
            filename=f"{gwt_name}.ims",
        )
        simulation.register_ims_package(transport_ims, [gwt.name])
        grid_class = (
            flopy.mf6.ModflowGwtdis if node_coordinates is None else flopy.mf6.ModflowGwtdisu
        )
        grid_class(gwt, pname="dis", idomain=1, filename=f"{gwt_name}.dis", **discretization)
        flopy.mf6.ModflowGwtic(gwt, strt=concentration, filename=f"{gwt_name}.ic")
        flopy.mf6.ModflowGwtadv(gwt, scheme="UPSTREAM", filename=f"{gwt_name}.adv")
        flopy.mf6.ModflowGwtdsp(
            gwt, xt3d_off=True, alh=alh, ath1=ath1, diffc=diffc, filename=f"{gwt_name}.dsp"
        )
        flopy.mf6.ModflowGwtmst(gwt, pname="mst", porosity=porosity, filename=f"{gwt_name}.mst")
        flopy.mf6.ModflowGwtsrc(
            gwt,
            pname="SRC",
            save_flows=True,
            maxbound=src_maxbound,
            stress_period_data={0: src_data_list},
            filename=f"{gwt_name}.src",
        )
        if diffusion_boundary:
            boundary_values_cnc = np.array(
                [inflow_concentrations[species.index(species_name)], concentration[-1]]
            )
            if species_name.casefold() == "charge":
                boundary_values_cnc[abs(boundary_values_cnc) < 1e-14] = 0.0
            if np.any(boundary_values_cnc < 0):
                raise ValueError("CNC requires nonnegative boundary values")
            flopy.mf6.ModflowGwtcnc(
                gwt,
                pname="fixed_cnc",
                maxbound=2,
                stress_period_data={
                    0: [
                        (cell_id(0), boundary_values_cnc[0]),
                        (cell_id(ncol - 1), boundary_values_cnc[1]),
                    ]
                },
                filename=f"{gwt_name}.cnc",
            )
        sourcerecarray = [("bushui", "AUX", species_name)]
        if species_name.casefold() == "charge":
            sourcerecarray.append(("ghb_right", "AUXMIXED", "CHARGE_OUT_LIMIT"))
        flopy.mf6.ModflowGwtssm(
            gwt, pname=f"{species_name}_ssm", sources=sourcerecarray, filename=f"{gwt_name}.ssm"
        )
        flopy.mf6.ModflowGwtoc(
            gwt,
            budget_filerecord=f"{gwt_name}.cbc",
            concentration_filerecord=f"{gwt_name}.ucn",
            saverecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
        )
        flopy.mf6.ModflowGwfgwt(
            simulation,
            exgtype="GWF6-GWT6",
            exgmnamea=gwf_name,
            exgmnameb=gwt_name,
            filename=f"{gwt_name}.gwfgwt",
        )
    simulation.write_simulation(silent=True)
    return simulation


__all__ = ["build_model"]
