from __future__ import annotations

from dataclasses import dataclass, field
from operator import attrgetter
from typing import Any

import numpy as np

from mf6pqc.exceptions import ConfigurationError
from mf6pqc.utils import ensure_array


@dataclass(slots=True)
class CellState:
    temperature: np.ndarray
    pressure: np.ndarray
    porosity: np.ndarray
    saturation: np.ndarray
    density: np.ndarray
    viscosity: np.ndarray
    d0: np.ndarray
    porosity_update_mask: np.ndarray
    print_chemistry_mask: np.ndarray
    water_only_sink_rates: np.ndarray
    has_water_only_sinks: bool

    @classmethod
    def from_config(cls, config):
        names = {
            "temperature": "temperature_c",
            "pressure": "pressure_atm",
            "porosity": "porosity",
            "saturation": "saturation",
            "density": "density_kg_per_litre",
            "viscosity": "viscosity_relative",
            "d0": "free_water_diffusion_model_units",
        }
        values = {
            name: ensure_array(config.nxyz, name, getattr(config.fields, field_name))
            for name, field_name in names.items()
        }
        for name, value in (
            ("porosity_update_mask", config.feedback.porosity_update_mask),
            ("print_chemistry_mask", config.chemistry.print_chemistry_mask),
        ):
            mask = ensure_array(config.nxyz, name, value)
            if not np.all(np.isin(mask, [0, 1])):
                raise ConfigurationError(f"{name} must contain only 0 or 1")
            values[name] = mask.astype(bool if name == "porosity_update_mask" else np.int32)
        sinks = config.feedback.water_only_sink_rates
        values["water_only_sink_rates"] = ensure_array(
            config.nxyz, "water_only_sink_rates", 0.0 if sinks is None else sinks
        )
        for name, value in values.items():
            if not np.all(np.isfinite(value)):
                raise ConfigurationError(f"{name} contains non-finite values")
        for name in ("pressure", "density", "viscosity"):
            if np.any(values[name] <= 0):
                raise ConfigurationError(f"{name} must be positive")
        if np.any(values["temperature"] <= -273.15):
            raise ConfigurationError("temperature must be above absolute zero")
        if np.any(values["porosity"] <= 0) or np.any(values["porosity"] > 1):
            raise ConfigurationError("porosity must be in (0, 1]")
        if np.any(values["saturation"] < 0) or np.any(values["saturation"] > 1):
            raise ConfigurationError("saturation must be in [0, 1]")
        for name in ("d0", "water_only_sink_rates"):
            if np.any(values[name] < 0):
                raise ConfigurationError(f"{name} must be nonnegative")
        values["has_water_only_sinks"] = bool(np.any(values["water_only_sink_rates"] > 0))
        return cls(**values)


@dataclass(slots=True)
class ChemistryState:
    backend: Any = None
    ncomps: int | None = None
    components: list[str] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    initial_concentrations: np.ndarray | None = None
    selected_output: np.ndarray | None = None
    output_indices: np.ndarray | None = None
    mineral_volumes: np.ndarray | None = None
    d_mineral_names: np.ndarray | None = None
    density_row: int = -1
    input_provenance: dict = field(default_factory=dict)


@dataclass(slots=True)
class TransportState:
    backend: Any = None
    simulation: Any = None
    energy_binding: Any = None
    head_addr: Any = None
    botm_arr: np.ndarray | None = None
    top_arr: np.ndarray | None = None
    cell_thick: np.ndarray | None = None
    K11_addr: Any = None
    K33_addr: Any = None
    K11_ptr: np.ndarray | None = None
    K33_ptr: np.ndarray | None = None
    tdis_kper_addr: Any = None
    tdis_kstp_addr: Any = None
    kchangeper_addr: Any = None
    kchangestp_addr: Any = None
    nodekchange_addr: Any = None
    kchange_ptrs: tuple = ()
    thetam_ptrs: dict = field(default_factory=dict)
    boundary_conductance_ptrs: list = field(default_factory=list)
    diffc_tags: dict = field(default_factory=dict)
    diffc_ptrs: tuple = ()
    current_diffusion: np.ndarray | None = None
    density_addr: Any = None
    density_ptr: np.ndarray | None = None
    k_update_density_prev: np.ndarray | None = None
    k_update_viscosity_prev: np.ndarray | None = None


@dataclass(slots=True)
class RunStatus:
    is_setup: bool = False
    final_time_step_index: int = 0
    last_run_wall_time_seconds: float | None = None
    last_coupling_method: str | None = None
    active: bool = False
    completed: bool = False
    chemistry_finalized: bool = False
    modflow_finalized: bool = False
    hooks: Any = None
    modflow_convergence_failures: list = field(default_factory=list)
    sia_iterations: list = field(default_factory=list)
    sia_convergence_failures: list = field(default_factory=list)
    sia_diagnostics: list = field(default_factory=list)
    porosity_clipping: dict = field(default_factory=dict)


def bind_aliases(cls, paths):
    for name, path in paths.items():
        owner, attribute = path.rsplit(".", 1)

        def setter(self, value, owner=owner, attribute=attribute):
            setattr(attrgetter(owner)(self), attribute, value)

        setattr(cls, name, property(attrgetter(path), setter))
