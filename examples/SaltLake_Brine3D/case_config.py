"""Scientific configuration for the idealized three-dimensional salt-lake case.

Lengths are metres, time is days, water fluxes are m3/day, hydraulic
conductivity is m/day, and transported concentrations are mol/L.  Mineral
inventories use mol/L of representative bulk volume, matching MF6PQC's
PhreeqcRM units option 0 contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

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
        potassium_mass_g += (
            moles * POTASSIUM_STOICHIOMETRY[mineral] * 39.0983
        )
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
        return list(
            range(self.steps_per_year, self.total_steps + 1, self.steps_per_year)
        )

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
        return [
            int(value)
            for value in np.linspace(1, self.nrow - 2, self.well_count)
        ]

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
        return np.broadcast_to(
            layer_values[:, None, None], self.shape
        ).copy()


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
    layer_depth_centres = (
        np.cumsum(np.asarray(profile.layer_thicknesses, dtype=float))
        - 0.5 * np.asarray(profile.layer_thicknesses, dtype=float)
    )
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
    layer_geometric_means = np.asarray(
        profile.layer_geometric_mean_k_m_per_day, dtype=float
    )
    field = layer_geometric_means[:, None, None] * np.exp(log_relative_k)
    return field.ravel()


def kinetic_facies(profile: CaseProfile) -> np.ndarray:
    """Return deterministic low/background/high potash facies identifiers."""
    if profile.uniform_mineralogy:
        return np.ones(profile.nxyz, dtype=int)

    layers, rows, columns = np.indices(profile.shape)
    x = columns / max(1, profile.ncol - 1)
    sinuous_centre = 0.50 * (profile.nrow - 1) + 0.12 * (
        profile.nrow - 1
    ) * np.sin(2.0 * np.pi * x)
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
        reconstructed = sum(
            moles[name] * MINERAL_MOLAR_VOLUMES_L_PER_MOL[name]
            for name in moles
        )
        if not np.isclose(reconstructed, TOTAL_EVAPORITE_FRACTION):
            raise AssertionError(f"Facies {facies} molar inventory is inconsistent")


assert_scientific_configuration()
