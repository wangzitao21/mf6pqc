from __future__ import annotations

from dataclasses import dataclass, field, replace
from operator import attrgetter
from pathlib import Path
from typing import Any

import numpy as np

from mf6pqc.backends import BackendFactory
from mf6pqc.constants import SECONDS_PER_DAY, VM_MINERALS
from mf6pqc.exceptions import ConfigurationError
from mf6pqc.kinetics import ImplicitOptions
from mf6pqc.permeability import (
    BasePermeabilityUpdater,
    FluidAdjustedKozenyCarmanUpdater,
    KozenyCarmanUpdater,
)
from mf6pqc.types import ArrayLike, SIARateEvaluator
from mf6pqc.utils import require_integer, step_numbers


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
    print_chemistry_mask: ArrayLike = 0
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
    boundary_conductance_updates: dict[str, dict[str, Any]] = field(default_factory=dict)
    fail_on_porosity_clipping: bool = False


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
    rate_evaluator: SIARateEvaluator | None = None


@dataclass(slots=True)
class OutputOptions:
    """Result retention and progress-reporting controls."""

    save_interval: int = 1
    save_interval_offset: int = 0
    save_steps: list[int] | None = None
    progress_interval: int = 1000
    storage: str = "memory"


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
    implicit: ImplicitOptions = field(default_factory=ImplicitOptions)

    @classmethod
    def from_legacy(cls, values):
        config = cls("temp_case", 80, BackendPaths(None, None, None, None, None))
        for name, value in values.items():
            path = LEGACY_FIELDS[name].split(".")
            owner = config
            for part in path[:-1]:
                owner = getattr(owner, part)
            setattr(owner, path[-1], value)
        return config

    def to_legacy_kwargs(self) -> dict[str, Any]:
        """Translate structured settings to the stable constructor contract."""
        values = {name: attrgetter(path)(self) for name, path in LEGACY_FIELDS.items()}
        for name, path in LEGACY_FIELDS.items():
            if path.startswith("paths.") and values[name] is not None:
                values[name] = str(values[name])
        return values

    def validated(self):
        config = replace(
            self,
            **{
                name: replace(getattr(self, name))
                for name in ("paths", "fields", "chemistry", "feedback", "sia", "output", "energy")
            },
        )
        config.implicit = (config.implicit or ImplicitOptions()).validated()
        if not isinstance(config.case_name, str) or not config.case_name.strip():
            raise ConfigurationError("case_name must be a non-empty string")
        if any(
            character in config.case_name for character in "/\\:\0"
        ) or config.case_name.strip() in {".", ".."}:
            raise ConfigurationError("case_name must be a filename label, without path separators")
        config.case_name = config.case_name.strip()
        config.nxyz = require_integer("nxyz", config.nxyz)
        config.nthreads = require_integer("nthreads", config.nthreads)
        output = config.output
        output.save_interval = require_integer("save_interval", output.save_interval)
        output.save_interval_offset = require_integer(
            "save_interval_offset", output.save_interval_offset, minimum=0
        )
        output.progress_interval = require_integer("progress_interval", output.progress_interval)
        output.save_steps = step_numbers("save_steps", output.save_steps)
        config.reaction_steps = step_numbers("reaction_steps", config.reaction_steps)
        if output.storage not in {"memory", "disk"}:
            raise ConfigurationError("result_storage must be 'memory' or 'disk'")
        signed = config.chemistry.signed_components
        if isinstance(signed, (str, bytes)):
            raise TypeError("signed_components must be a sequence of component names")
        try:
            signed = tuple(signed)
        except TypeError as exc:
            raise TypeError("signed_components must be a sequence of component names") from exc
        if any(not isinstance(name, str) or not name.strip() for name in signed):
            raise ValueError("signed_components must contain only non-empty component names")
        config.chemistry.signed_components = frozenset(name.strip().casefold() for name in signed)
        sia = config.sia
        sia.maximum_iterations = require_integer("sia_max_iterations", sia.maximum_iterations)
        if (
            not np.isfinite(sia.relative_tolerance)
            or not np.isfinite(sia.absolute_tolerance)
            or min(sia.relative_tolerance, sia.absolute_tolerance) < 0
        ):
            raise ConfigurationError("SIA convergence tolerances must be finite and nonnegative")
        if sia.relative_tolerance == sia.absolute_tolerance == 0:
            raise ConfigurationError("At least one SIA convergence tolerance must be positive")
        for name, value in (("source", sia.source_relaxation), ("density", sia.density_relaxation)):
            if not 0.0 < value <= 1.0:
                raise ConfigurationError(f"sia_{name}_relaxation must be in (0, 1]")
        if sia.rate_evaluator is not None and not callable(sia.rate_evaluator):
            raise TypeError("sia_rate_evaluator must be callable or None")
        energy = config.energy
        if energy.viscosity_feedback and not energy.enabled:
            raise ConfigurationError("vsc_enabled=True requires energy_enabled=True")
        for name in (
            "flow_model_name",
            "energy_model_name",
            "npf_package_name",
            "vsc_package_name",
            "est_package_name",
        ):
            value = getattr(energy, name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"{name} must be a non-empty string")
            setattr(energy, name, value.strip())
        if not np.isfinite(energy.initial_field_tolerance) or energy.initial_field_tolerance < 0:
            raise ConfigurationError("initial_gwe_field_tolerance must be finite and nonnegative")
        feedback = config.feedback
        feedback.boundary_conductance_updates = {
            name: dict(value)
            for name, value in (feedback.boundary_conductance_updates or {}).items()
        }
        if feedback.boundary_conductance_updates and not feedback.update_porosity_and_k:
            raise ConfigurationError(
                "boundary_conductance_updates requires if_update_porosity_K=True"
            )
        for name, entry in (feedback.boundary_conductance_updates or {}).items():
            if "cell_index" not in entry or "distance" not in entry:
                raise ConfigurationError(f"Invalid boundary_conductance_updates entry for {name!r}")
            index = require_integer("cell_index", entry["cell_index"], minimum=-config.nxyz)
            if index >= config.nxyz:
                raise ConfigurationError(
                    f"Boundary cell_index for {name!r} is outside the model: {index}"
                )
            entry["cell_index"] = index
        if sia.rate_evaluator is not None and (
            feedback.update_porosity_and_k or feedback.update_density or feedback.update_diffusion
        ):
            raise ConfigurationError(
                "sia_rate_evaluator is a stateless aqueous-rate interface and cannot update PHREEQC-owned density, porosity, conductivity, or diffusion feedback"
            )
        if feedback.permeability_updater is None:
            feedback.permeability_updater = KozenyCarmanUpdater()
        elif not isinstance(feedback.permeability_updater, BasePermeabilityUpdater):
            raise TypeError("permeability_updater must implement BasePermeabilityUpdater")
        if energy.viscosity_feedback:
            if feedback.boundary_conductance_updates:
                raise ConfigurationError("boundary_conductance_updates cannot be combined with VSC")
            if isinstance(feedback.permeability_updater, FluidAdjustedKozenyCarmanUpdater):
                raise ConfigurationError(
                    "FluidAdjustedKozenyCarmanUpdater cannot be combined with VSC; that would apply viscosity to hydraulic conductivity twice"
                )
        if (
            not np.isfinite(feedback.vertical_to_horizontal_k_ratio)
            or feedback.vertical_to_horizontal_k_ratio <= 0
        ):
            raise ConfigurationError("k33_ratio must be finite and positive")
        feedback.mineral_molar_volumes = {**VM_MINERALS, **(feedback.mineral_molar_volumes or {})}
        for name, value in feedback.mineral_molar_volumes.items():
            if not isinstance(name, str) or not name or not np.isfinite(value) or value <= 0:
                raise ConfigurationError(
                    "mineral_molar_volumes must map non-empty names to positive finite values in L/mol"
                )
        return config


LEGACY_FIELDS = {
    "implicit_options": "implicit",
    "case_name": "case_name",
    "nxyz": "nxyz",
    "nthreads": "nthreads",
    "temperature": "fields.temperature_c",
    "pressure": "fields.pressure_atm",
    "porosity": "fields.porosity",
    "saturation": "fields.saturation",
    "density": "fields.density_kg_per_litre",
    "viscosity": "fields.viscosity_relative",
    "d0": "fields.free_water_diffusion_model_units",
    "print_chemistry_mask": "chemistry.print_chemistry_mask",
    "componentH2O": "chemistry.transport_water_component",
    "solution_density_volume": "chemistry.use_solution_density_volume",
    "signed_components": "chemistry.signed_components",
    "db_path": "paths.database",
    "pqi_path": "paths.chemistry_input",
    "modflow_dll_path": "paths.modflow_library",
    "workspace": "paths.workspace",
    "output_dir": "paths.output_directory",
    "if_update_porosity_K": "feedback.update_porosity_and_k",
    "if_update_density": "feedback.update_density",
    "if_update_diffc": "feedback.update_diffusion",
    "save_interval": "output.save_interval",
    "save_interval_offset": "output.save_interval_offset",
    "save_steps": "output.save_steps",
    "reaction_steps": "reaction_steps",
    "progress_interval": "output.progress_interval",
    "fail_on_nonconvergence": "fail_on_modflow_nonconvergence",
    "boundary_conductance_updates": "feedback.boundary_conductance_updates",
    "water_only_sink_rates": "feedback.water_only_sink_rates",
    "use_phreeqc_calculated_density": "feedback.use_phreeqc_calculated_density",
    "porosity_update_mask": "feedback.porosity_update_mask",
    "sia_max_iterations": "sia.maximum_iterations",
    "sia_rtol": "sia.relative_tolerance",
    "sia_atol": "sia.absolute_tolerance",
    "sia_rate_evaluator": "sia.rate_evaluator",
    "sia_source_relaxation": "sia.source_relaxation",
    "sia_density_relaxation": "sia.density_relaxation",
    "sia_fail_on_nonconvergence": "sia.fail_on_nonconvergence",
    "permeability_updater": "feedback.permeability_updater",
    "k33_ratio": "feedback.vertical_to_horizontal_k_ratio",
    "density_output_heading": "feedback.density_output_heading",
    "mineral_molar_volumes": "feedback.mineral_molar_volumes",
    "energy_enabled": "energy.enabled",
    "vsc_enabled": "energy.viscosity_feedback",
    "flow_model_name": "energy.flow_model_name",
    "energy_model_name": "energy.energy_model_name",
    "npf_package_name": "energy.npf_package_name",
    "vsc_package_name": "energy.vsc_package_name",
    "est_package_name": "energy.est_package_name",
    "sync_gwe_temperature_to_phreeqc": "energy.sync_temperature_to_chemistry",
    "validate_initial_gwe_fields": "energy.validate_initial_fields",
    "initial_gwe_field_tolerance": "energy.initial_field_tolerance",
    "backend_factory": "backend_factory",
    "result_storage": "output.storage",
    "fail_on_porosity_clipping": "feedback.fail_on_porosity_clipping",
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
