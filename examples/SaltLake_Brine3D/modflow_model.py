"""Scientific configuration for the idealized three-dimensional salt-lake case.

Lengths are metres, time is days, water fluxes are m3/day, hydraulic
conductivity is m/day, and transported concentrations are mol/L.  Mineral
inventories use mol/L of representative bulk volume, matching MF6PQC's
PhreeqcRM units option 0 contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import dataclass
from typing import Final

import _example_support as _example_support
import numpy as np

DAYS_PER_YEAR: Final[float] = 365.0
INITIAL_POROSITY: Final[float] = 0.25
INERT_SOLID_FRACTION: Final[float] = 0.05
TOTAL_EVAPORITE_FRACTION: Final[float] = 0.70
INITIAL_HEAD_M: Final[float] = 35.0
CHANNEL_STAGE_M: Final[float] = 35.0
SPECIFIC_STORAGE_PER_M: Final[float] = 1.0e-5
K33_RATIO: Final[float] = 0.10

# The primary assemblage occupies 70% of the representative bulk volume.
# Its nominal elemental-K mass fraction is about 2.2%, representing a
# low-grade, high-tonnage solid potash resource.  Borax is included as a
# trace B-bearing evaporite; Li is represented by dissolved and exchangeable
# inventories because a site-specific Li mineral has not yet been identified.
BACKGROUND_MINERAL_VOLUME_FRACTIONS: Final[dict[str, float]] = {
    "Halite": 0.544,
    "Carnallite": 0.060,
    "Polyhalite": 0.040,
    "Sylvite": 0.005,
    "Gypsum": 0.045,
    "Borax": 0.006,
}

MINERAL_MOLAR_VOLUMES_L_PER_MOL: Final[dict[str, float]] = {
    "Halite": 0.0271,
    "Carnallite": 0.1737,
    "Polyhalite": 0.2180,
    "Sylvite": 0.0375,
    "Gypsum": 0.07421,
    "Borax": 0.2230,
    "Bischofite": 0.1271,
    "Syngenite": 0.1273,
    "Mirabilite": 0.2160,
}

MINERAL_MOLAR_MASSES_G_PER_MOL: Final[dict[str, float]] = {
    "Halite": 58.4428,
    "Carnallite": 277.853,
    "Polyhalite": 602.468,
    "Sylvite": 74.5513,
    "Gypsum": 172.171,
    "Borax": 381.37,
}

POTASSIUM_STOICHIOMETRY: Final[dict[str, float]] = {
    "Halite": 0.0,
    "Carnallite": 1.0,
    "Polyhalite": 2.0,
    "Sylvite": 1.0,
    "Gypsum": 0.0,
    "Borax": 0.0,
}


def _facies_volume_fractions(scale_k_minerals: float) -> dict[str, float]:
    """Return a 70%-evaporite facies while varying its potash grade."""
    values = dict(BACKGROUND_MINERAL_VOLUME_FRACTIONS)
    potassium_minerals = ("Carnallite", "Polyhalite", "Sylvite")
    old_k_fraction = sum(values[name] for name in potassium_minerals)
    for name in potassium_minerals:
        values[name] *= scale_k_minerals
    new_k_fraction = sum(values[name] for name in potassium_minerals)
    values["Halite"] -= new_k_fraction - old_k_fraction
    if not np.isclose(sum(values.values()), TOTAL_EVAPORITE_FRACTION):
        raise AssertionError("Facies mineral fractions no longer sum to 0.70")
    return values


FACIES_MINERAL_VOLUME_FRACTIONS: Final[dict[int, dict[str, float]]] = {
    1: _facies_volume_fractions(1.00),  # background, about 2.2 wt% K
    2: _facies_volume_fractions(1.30),  # locally K-richer material
    3: _facies_volume_fractions(0.70),  # locally K-poorer material
}


def mineral_moles_per_bulk_litre(
    volume_fractions: dict[str, float],
) -> dict[str, float]:
    """Convert solid volume fractions to PhreeqcRM kinetic inventories."""
    return {
        name: fraction / MINERAL_MOLAR_VOLUMES_L_PER_MOL[name]
        for name, fraction in volume_fractions.items()
    }


FACIES_MINERAL_MOLES: Final[dict[int, dict[str, float]]] = {
    facies: mineral_moles_per_bulk_litre(fractions)
    for facies, fractions in FACIES_MINERAL_VOLUME_FRACTIONS.items()
}


def potassium_grade_percent(volume_fractions: dict[str, float]) -> float:
    """Return elemental K mass divided by total modeled evaporite mass."""
    total_mass_g = 0.0
    potassium_mass_g = 0.0
    for mineral, fraction in volume_fractions.items():
        moles = fraction / MINERAL_MOLAR_VOLUMES_L_PER_MOL[mineral]
        total_mass_g += moles * MINERAL_MOLAR_MASSES_G_PER_MOL[mineral]
        potassium_mass_g += moles * POTASSIUM_STOICHIOMETRY[mineral] * 39.0983
    return 100.0 * potassium_mass_g / total_mass_g


@dataclass(frozen=True, slots=True)
class CaseProfile:
    """Grid and duration controls for one executable profile."""

    name: str
    nrow: int
    ncol: int
    delr: float
    delc: float
    years: int
    well_count: int
    well_rate_m3_per_day: float

    nlay: int = 3
    layer_thicknesses: tuple[float, float, float] = (10.0, 10.0, 10.0)
    steps_per_year: int = 12
    layer_geometric_mean_k_m_per_day: tuple[float, float, float] = (
        3.0,
        1.2,
        0.5,
    )
    log_k_standard_deviation: float = 0.45
    k_correlation_lengths_m: tuple[float, float, float] = (180.0, 55.0, 12.0)
    random_field_seed: int = 20260817
    permeability_exponent: float = 3.0
    channel_width_m: float = 40.0
    uniform_mineralogy: bool = False

    @property
    def nxyz(self) -> int:
        return self.nlay * self.nrow * self.ncol

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.nlay, self.nrow, self.ncol

    @property
    def top(self) -> float:
        return float(sum(self.layer_thicknesses))

    @property
    def botm(self) -> np.ndarray:
        return self.top - np.cumsum(self.layer_thicknesses)

    @property
    def total_steps(self) -> int:
        return self.years * self.steps_per_year

    @property
    def period_days(self) -> float:
        return self.years * DAYS_PER_YEAR

    @property
    def save_steps(self) -> list[int]:
        return list(range(self.steps_per_year, self.total_steps + 1, self.steps_per_year))

    @property
    def channel_column(self) -> int:
        return max(1, round(0.08 * (self.ncol - 1)))

    @property
    def well_column(self) -> int:
        return min(self.ncol - 2, round(0.92 * (self.ncol - 1)))

    @property
    def channel_rows(self) -> list[int]:
        return list(range(1, self.nrow - 1))

    @property
    def well_rows(self) -> list[int]:
        return [int(value) for value in np.linspace(1, self.nrow - 2, self.well_count)]

    @property
    def channel_cells(self) -> list[tuple[int, int, int]]:
        return [(0, row, self.channel_column) for row in self.channel_rows]

    @property
    def well_cells(self) -> list[tuple[int, int, int]]:
        return [(0, row, self.well_column) for row in self.well_rows]

    @property
    def channel_conductance_per_cell_m2_per_day(self) -> float:
        # C = K_bed A / b: 5e-3 m/d through a 0.5-m bed.
        # channel_width_m keeps the physical canal footprint independent of
        # horizontal grid refinement.
        return 0.005 * self.channel_width_m * self.delc / 0.5

    @property
    def cell_volumes_m3(self) -> np.ndarray:
        layer_values = np.asarray(self.layer_thicknesses) * self.delr * self.delc
        return np.broadcast_to(layer_values[:, None, None], self.shape).copy()


PROFILES: Final[dict[str, CaseProfile]] = {
    "smoke": CaseProfile(
        name="smoke",
        nrow=7,
        ncol=12,
        delr=50.0,
        delc=50.0,
        years=2,
        well_count=3,
        well_rate_m3_per_day=-60.0,
    ),
    "base": CaseProfile(
        name="base",
        nrow=15,
        ncol=30,
        delr=40.0,
        delc=40.0,
        years=30,
        well_count=5,
        well_rate_m3_per_day=-120.0,
    ),
    "highres": CaseProfile(
        name="highres",
        nrow=30,
        ncol=60,
        delr=20.0,
        delc=20.0,
        years=30,
        well_count=9,
        well_rate_m3_per_day=-(600.0 / 9.0),
        steps_per_year=4,
        log_k_standard_deviation=0.85,
        k_correlation_lengths_m=(180.0, 55.0, 12.0),
        random_field_seed=20260817,
        permeability_exponent=5.0,
        uniform_mineralogy=True,
    ),
}


def cell_index(profile: CaseProfile, cell: tuple[int, int, int]) -> int:
    """Return the C-order flat index used by MODFLOW and PhreeqcRM."""
    return int(np.ravel_multi_index(cell, profile.shape))


def initial_porosity(profile: CaseProfile) -> np.ndarray:
    """Return a cell-major porosity field."""
    return np.full(profile.nxyz, INITIAL_POROSITY, dtype=float)


def initial_hydraulic_conductivity(profile: CaseProfile) -> np.ndarray:
    """Return a reproducible anisotropic three-dimensional lognormal K field."""
    try:
        import gstools as gs
    except ImportError as error:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "GSTools is required for the SaltLake_Brine3D initial K field. "
            "Install the example dependencies with pip install -e .[examples]."
        ) from error

    x_centres = (np.arange(profile.ncol, dtype=float) + 0.5) * profile.delr
    y_centres = (np.arange(profile.nrow, dtype=float) + 0.5) * profile.delc
    layer_depth_centres = np.cumsum(
        np.asarray(profile.layer_thicknesses, dtype=float)
    ) - 0.5 * np.asarray(profile.layer_thicknesses, dtype=float)
    covariance = gs.Gaussian(
        dim=3,
        var=profile.log_k_standard_deviation**2,
        len_scale=list(profile.k_correlation_lengths_m),
    )
    log_k_generator = gs.SRF(covariance, seed=profile.random_field_seed)
    # GSTools returns (x, y, z); MODFLOW/PhreeqcRM use (layer, row, column).
    log_relative_k = log_k_generator.structured(
        [x_centres, y_centres, layer_depth_centres]
    ).transpose(2, 1, 0)
    # Enforce the requested geometric mean in every layer without changing
    # the spatial correlation pattern.
    log_relative_k -= np.mean(log_relative_k, axis=(1, 2), keepdims=True)
    layer_geometric_means = np.asarray(profile.layer_geometric_mean_k_m_per_day, dtype=float)
    field = layer_geometric_means[:, None, None] * np.exp(log_relative_k)
    return field.ravel()


def kinetic_facies(profile: CaseProfile) -> np.ndarray:
    """Return deterministic low/background/high potash facies identifiers."""
    if profile.uniform_mineralogy:
        return np.ones(profile.nxyz, dtype=int)

    layers, rows, columns = np.indices(profile.shape)
    x = columns / max(1, profile.ncol - 1)
    sinuous_centre = 0.50 * (profile.nrow - 1) + 0.12 * (profile.nrow - 1) * np.sin(2.0 * np.pi * x)
    distance = np.abs(rows - sinuous_centre)

    facies = np.ones(profile.shape, dtype=int)
    facies[distance <= 0.60] = 2
    poor_score = np.sin(5.0 * x + 1.7 * rows + 0.9 * layers)
    facies[(distance > 0.60) & (poor_score < -0.70)] = 3
    return facies.ravel()


def assert_scientific_configuration() -> None:
    """Fail early if the volume or grade bookkeeping is edited inconsistently."""
    if not np.isclose(
        INITIAL_POROSITY + INERT_SOLID_FRACTION + TOTAL_EVAPORITE_FRACTION,
        1.0,
    ):
        raise AssertionError("Bulk-volume fractions must sum to one")
    for facies, fractions in FACIES_MINERAL_VOLUME_FRACTIONS.items():
        if not np.isclose(sum(fractions.values()), TOTAL_EVAPORITE_FRACTION):
            raise AssertionError(f"Facies {facies} does not contain 70% evaporite")
        moles = FACIES_MINERAL_MOLES[facies]
        reconstructed = sum(moles[name] * MINERAL_MOLAR_VOLUMES_L_PER_MOL[name] for name in moles)
        if not np.isclose(reconstructed, TOTAL_EVAPORITE_FRACTION):
            raise AssertionError(f"Facies {facies} molar inventory is inconsistent")


assert_scientific_configuration()


from collections.abc import Iterable

import flopy

from mf6pqc.utils import get_gwt_model_name

FLOW_MODEL_NAME = "gwf_model"
NPF_PACKAGE_NAME = "NPF"
CHANNEL_PACKAGE_NAME = "RECHARGE_CHANNEL"
WELL_PACKAGE_NAME = "PRODUCTION_WELLS"


def _output_records(profile: CaseProfile, variable: str) -> dict[int, list[tuple]]:
    return {0: [(variable, "STEPS", *profile.save_steps)]}


def build_model(
    *,
    workspace: str | Path,
    mf6_executable: str | Path,
    profile: CaseProfile,
    species: Iterable[str],
    initial_concentrations: np.ndarray,
    channel_concentrations: np.ndarray,
    porosity: np.ndarray,
    hydraulic_conductivity: np.ndarray,
    density_feedback: bool = True,
) -> flopy.mf6.MFSimulation:
    """Build and write the confined GWF plus one GWT model per component."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    species = list(species)
    initial_concentrations = np.asarray(initial_concentrations, dtype=float).ravel()
    channel_concentrations = np.asarray(channel_concentrations, dtype=float).ravel()
    porosity = np.asarray(porosity, dtype=float).reshape(profile.shape)
    hydraulic_conductivity = np.asarray(hydraulic_conductivity, dtype=float).reshape(profile.shape)

    expected = len(species) * profile.nxyz
    if initial_concentrations.size != expected:
        raise ValueError(
            "Initial concentration vector has size "
            f"{initial_concentrations.size}; expected {expected}"
        )
    if channel_concentrations.size != len(species):
        raise ValueError("Channel concentration vector does not match components")
    if np.any(hydraulic_conductivity <= 0.0):
        raise ValueError("Hydraulic conductivity must be positive")

    simulation = flopy.mf6.MFSimulation(
        sim_name="salt_lake_brine_3d",
        sim_ws=str(workspace),
        exe_name=str(mf6_executable),
        verbosity_level=0,
    )
    flopy.mf6.ModflowTdis(
        simulation,
        pname="TDIS",
        time_units="DAYS",
        nper=1,
        perioddata=[(profile.period_days, profile.total_steps, 1.0)],
    )

    flow = flopy.mf6.ModflowGwf(
        simulation,
        modelname=FLOW_MODEL_NAME,
        save_flows=True,
    )
    flow_solver = flopy.mf6.ModflowIms(
        simulation,
        pname="flow_ims",
        print_option="SUMMARY",
        complexity="SIMPLE",
        outer_dvclose=1.0e-6,
        outer_maximum=100,
        under_relaxation="NONE",
        inner_maximum=500,
        inner_dvclose=1.0e-8,
        rcloserecord=1.0e-6,
        linear_acceleration="CG",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=0.99,
        filename="flow.ims",
    )
    simulation.register_ims_package(flow_solver, [flow.name])

    flopy.mf6.ModflowGwfdis(
        flow,
        pname="DIS",
        nlay=profile.nlay,
        nrow=profile.nrow,
        ncol=profile.ncol,
        delr=profile.delr,
        delc=profile.delc,
        top=profile.top,
        botm=profile.botm,
    )
    flopy.mf6.ModflowGwfic(flow, pname="IC", strt=INITIAL_HEAD_M)
    flopy.mf6.ModflowGwfnpf(
        flow,
        pname=NPF_PACKAGE_NAME,
        save_flows=True,
        save_specific_discharge=True,
        icelltype=0,
        k=hydraulic_conductivity,
        k33=hydraulic_conductivity * K33_RATIO,
    )
    flopy.mf6.ModflowGwfsto(
        flow,
        pname="STO",
        save_flows=True,
        iconvert=0,
        ss=SPECIFIC_STORAGE_PER_M,
        sy=0.0,
        transient={0: True},
    )

    channel_data = [
        (
            cell,
            CHANNEL_STAGE_M,
            profile.channel_conductance_per_cell_m2_per_day,
            *channel_concentrations,
        )
        for cell in profile.channel_cells
    ]
    flopy.mf6.ModflowGwfghb(
        flow,
        pname=CHANNEL_PACKAGE_NAME,
        filename="gwf_model.recharge_channel.ghb",
        save_flows=True,
        auxiliary=species,
        stress_period_data={0: channel_data},
    )

    well_data = [(cell, profile.well_rate_m3_per_day) for cell in profile.well_cells]
    flopy.mf6.ModflowGwfwel(
        flow,
        pname=WELL_PACKAGE_NAME,
        filename="gwf_model.production_wells.wel",
        save_flows=True,
        stress_period_data={0: well_data},
    )

    flopy.mf6.ModflowGwfoc(
        flow,
        pname="OC",
        budget_filerecord=f"{FLOW_MODEL_NAME}.bud",
        head_filerecord=f"{FLOW_MODEL_NAME}.hds",
        saverecord={
            0: [
                ("HEAD", "STEPS", *profile.save_steps),
                ("BUDGET", "STEPS", *profile.save_steps),
            ]
        },
    )

    component_fields = {
        name: initial_concentrations[index * profile.nxyz : (index + 1) * profile.nxyz].reshape(
            profile.shape
        )
        for index, name in enumerate(species)
    }

    for species_name, initial_field in component_fields.items():
        transport_name = get_gwt_model_name(species_name)
        transport = flopy.mf6.ModflowGwt(
            simulation,
            modelname=transport_name,
            model_nam_file=f"{transport_name}.nam",
            save_flows=True,
        )
        transport_solver = flopy.mf6.ModflowIms(
            simulation,
            print_option="SUMMARY",
            outer_dvclose=1.0e-7,
            outer_maximum=120,
            inner_maximum=250,
            inner_dvclose=1.0e-8,
            rcloserecord=1.0e-6,
            linear_acceleration="BICGSTAB",
            scaling_method="DIAGONAL",
            reordering_method="RCM",
            relaxation_factor=0.97,
            filename=f"{transport_name}.ims",
        )
        simulation.register_ims_package(transport_solver, [transport.name])

        flopy.mf6.ModflowGwtdis(
            transport,
            pname="DIS",
            nlay=profile.nlay,
            nrow=profile.nrow,
            ncol=profile.ncol,
            delr=profile.delr,
            delc=profile.delc,
            top=profile.top,
            botm=profile.botm,
            filename=f"{transport_name}.dis",
        )
        flopy.mf6.ModflowGwtic(
            transport,
            pname="IC",
            strt=initial_field,
            filename=f"{transport_name}.ic",
        )
        flopy.mf6.ModflowGwtadv(
            transport,
            pname="ADV",
            scheme="TVD",
            filename=f"{transport_name}.adv",
        )
        flopy.mf6.ModflowGwtdsp(
            transport,
            pname="DSP",
            xt3d_off=False,
            alh=20.0,
            alv=5.0,
            ath1=2.0,
            atv=0.5,
            diffc=1.0e-9 * 86_400.0,
            filename=f"{transport_name}.dsp",
        )
        flopy.mf6.ModflowGwtmst(
            transport,
            pname="MST",
            porosity=porosity,
            filename=f"{transport_name}.mst",
        )
        flopy.mf6.ModflowGwtssm(
            transport,
            pname=f"{species_name}_SSM",
            sources=[(CHANNEL_PACKAGE_NAME, "AUX", species_name)],
            filename=f"{transport_name}.ssm",
        )
        flopy.mf6.ModflowGwtoc(
            transport,
            pname="OC",
            budget_filerecord=f"{transport_name}.cbc",
            concentration_filerecord=f"{transport_name}.ucn",
            saverecord={
                0: [
                    ("CONCENTRATION", "STEPS", *profile.save_steps),
                    ("BUDGET", "STEPS", *profile.save_steps),
                ]
            },
        )
        flopy.mf6.ModflowGwfgwt(
            simulation,
            exgtype="GWF6-GWT6",
            exgmnamea=FLOW_MODEL_NAME,
            exgmnameb=transport_name,
            filename=f"{transport_name}.gwfgwt",
        )

    if density_feedback:
        chloride_model = get_gwt_model_name("Cl")
        if "Cl" not in species:
            raise ValueError("Density feedback requires the transported Cl component")
        # A zero-slope BUY entry allocates DENSE. MF6PQC then writes the full
        # Pitzer-calculated density field directly before each flow solve.
        flopy.mf6.ModflowGwfbuy(
            flow,
            pname="BUY",
            denseref=1000.0,
            nrhospecies=1,
            density_filerecord="model_density.bin",
            packagedata=[(0, 0.0, 0.0, chloride_model, "CONCENTRATION")],
        )

    simulation.write_simulation(silent=True)
    return simulation
