"""Structured configuration for scientist-facing MF6PQC workflows.

The legacy keyword constructor remains supported.  These dataclasses provide a
clearer path for new cases and isolate groups that will grow independently
(coupling methods, GWE/VSC, and learned property models).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mf6pqc.backends import BackendFactory
from mf6pqc.constants import SECONDS_PER_DAY
from mf6pqc.permeability import BasePermeabilityUpdater
from mf6pqc.types import ArrayLike


@dataclass(slots=True)
class BackendPaths:
    """Files and directories required by the two native solvers."""

    database: str | Path
    chemistry_input: str | Path
    modflow_library: str | Path
    workspace: str | Path
    output_directory: str | Path


@dataclass(slots=True)
class CellFields:
    """Cell-wise physical fields; scalars are expanded to all cells."""

    temperature_c: ArrayLike = 25.0
    pressure_atm: ArrayLike = 2.0
    porosity: ArrayLike = 0.35
    saturation: ArrayLike = 1.0
    density_kg_per_litre: ArrayLike = 1.0
    viscosity_relative: ArrayLike = 1.0
    free_water_diffusion_model_units: ArrayLike = 1.0e-9 * SECONDS_PER_DAY


@dataclass(slots=True)
class ChemistryOptions:
    """PhreeqcRM representation and diagnostic controls."""

    transport_water_component: bool = False
    use_solution_density_volume: bool = False
    print_chemistry_mask: ArrayLike = 1
    signed_components: tuple[str, ...] = ("Charge",)


@dataclass(slots=True)
class FeedbackOptions:
    """Chemistry-to-flow and chemistry-to-transport feedback controls."""

    update_porosity_and_k: bool = False
    update_density: bool = False
    update_diffusion: bool = False
    porosity_update_mask: ArrayLike = 1
    water_only_sink_rates: ArrayLike | None = None
    use_phreeqc_calculated_density: bool = False
    density_output_heading: str = "RHO"
    mineral_molar_volumes: dict[str, float] = field(default_factory=dict)
    permeability_updater: BasePermeabilityUpdater | None = None
    vertical_to_horizontal_k_ratio: float = 0.6
    boundary_conductance_updates: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class EnergyOptions:
    """Optional MODFLOW 6 GWE and VSC coupling controls.

    Energy coupling is deliberately opt-in.  When ``viscosity_feedback`` is
    enabled, MODFLOW VSC owns the effective NPF conductivity while MF6PQC
    updates only the reference (input) conductivity after reactions.
    """

    enabled: bool = False
    viscosity_feedback: bool = False
    flow_model_name: str = "gwf_model"
    energy_model_name: str = "gwe_model"
    npf_package_name: str = "NPF"
    vsc_package_name: str = "VSC"
    est_package_name: str = "EST"
    sync_temperature_to_chemistry: bool = True
    validate_initial_fields: bool = True
    initial_field_tolerance: float = 1.0e-8


@dataclass(slots=True)
class SIAOptions:
    """Convergence and relaxation controls for sequential iteration."""

    maximum_iterations: int = 2000
    relative_tolerance: float = 1.0e-4
    absolute_tolerance: float = 1.0e-9
    source_relaxation: float = 0.5
    density_relaxation: float = 0.5
    fail_on_nonconvergence: bool = False


@dataclass(slots=True)
class OutputOptions:
    """Result retention and progress-reporting controls."""

    save_interval: int = 1
    save_interval_offset: int = 0
    save_steps: list[int] | None = None
    progress_interval: int = 1000


@dataclass(slots=True)
class SimulationConfig:
    """Complete structured input for one MF6PQC simulator instance."""

    case_name: str
    nxyz: int
    paths: BackendPaths
    nthreads: int = 3
    fields: CellFields = field(default_factory=CellFields)
    chemistry: ChemistryOptions = field(default_factory=ChemistryOptions)
    feedback: FeedbackOptions = field(default_factory=FeedbackOptions)
    sia: SIAOptions = field(default_factory=SIAOptions)
    output: OutputOptions = field(default_factory=OutputOptions)
    fail_on_modflow_nonconvergence: bool = False
    backend_factory: BackendFactory | None = None
    # Appended after the original fields to preserve positional compatibility.
    energy: EnergyOptions = field(default_factory=EnergyOptions)
    reaction_steps: list[int] | None = None

    def to_legacy_kwargs(self) -> dict[str, Any]:
        """Translate structured settings to the stable constructor contract."""
        return {
            "case_name": self.case_name,
            "nxyz": self.nxyz,
            "nthreads": self.nthreads,
            "temperature": self.fields.temperature_c,
            "pressure": self.fields.pressure_atm,
            "porosity": self.fields.porosity,
            "saturation": self.fields.saturation,
            "density": self.fields.density_kg_per_litre,
            "viscosity": self.fields.viscosity_relative,
            "d0": self.fields.free_water_diffusion_model_units,
            "print_chemistry_mask": self.chemistry.print_chemistry_mask,
            "componentH2O": self.chemistry.transport_water_component,
            "solution_density_volume": self.chemistry.use_solution_density_volume,
            "signed_components": self.chemistry.signed_components,
            "db_path": str(self.paths.database),
            "pqi_path": str(self.paths.chemistry_input),
            "modflow_dll_path": str(self.paths.modflow_library),
            "workspace": str(self.paths.workspace),
            "output_dir": str(self.paths.output_directory),
            "if_update_porosity_K": self.feedback.update_porosity_and_k,
            "if_update_density": self.feedback.update_density,
            "if_update_diffc": self.feedback.update_diffusion,
            "save_interval": self.output.save_interval,
            "save_interval_offset": self.output.save_interval_offset,
            "save_steps": self.output.save_steps,
            "reaction_steps": self.reaction_steps,
            "progress_interval": self.output.progress_interval,
            "fail_on_nonconvergence": self.fail_on_modflow_nonconvergence,
            "boundary_conductance_updates": (
                self.feedback.boundary_conductance_updates
            ),
            "water_only_sink_rates": self.feedback.water_only_sink_rates,
            "use_phreeqc_calculated_density": (
                self.feedback.use_phreeqc_calculated_density
            ),
            "porosity_update_mask": self.feedback.porosity_update_mask,
            "sia_max_iterations": self.sia.maximum_iterations,
            "sia_rtol": self.sia.relative_tolerance,
            "sia_atol": self.sia.absolute_tolerance,
            "sia_source_relaxation": self.sia.source_relaxation,
            "sia_density_relaxation": self.sia.density_relaxation,
            "sia_fail_on_nonconvergence": self.sia.fail_on_nonconvergence,
            "permeability_updater": self.feedback.permeability_updater,
            "k33_ratio": self.feedback.vertical_to_horizontal_k_ratio,
            "density_output_heading": self.feedback.density_output_heading,
            "mineral_molar_volumes": self.feedback.mineral_molar_volumes,
            "energy_enabled": self.energy.enabled,
            "vsc_enabled": self.energy.viscosity_feedback,
            "flow_model_name": self.energy.flow_model_name,
            "energy_model_name": self.energy.energy_model_name,
            "npf_package_name": self.energy.npf_package_name,
            "vsc_package_name": self.energy.vsc_package_name,
            "est_package_name": self.energy.est_package_name,
            "sync_gwe_temperature_to_phreeqc": (
                self.energy.sync_temperature_to_chemistry
            ),
            "validate_initial_gwe_fields": self.energy.validate_initial_fields,
            "initial_gwe_field_tolerance": self.energy.initial_field_tolerance,
            "backend_factory": self.backend_factory,
        }


__all__ = [
    "BackendPaths",
    "CellFields",
    "ChemistryOptions",
    "EnergyOptions",
    "FeedbackOptions",
    "OutputOptions",
    "SIAOptions",
    "SimulationConfig",
]
