"""MODFLOW 6 GWE/VSC coupling and thermal-state ownership.

The central scientific contract is that reaction/porosity models own the
reference NPF conductivity (``KxxINPUT``), while MODFLOW VSC alone owns the
viscosity-adjusted effective conductivity (``Kxx``).  Keeping both fields
separate prevents viscosity from being applied twice or accumulated from one
time step to the next.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mf6pqc.exceptions import BackendError, ConfigurationError, CouplingError


@dataclass(slots=True)
class EnergyCouplingBinding:
    """Live MODFLOW arrays and snapshots used by thermal SNIA."""

    temperature_ptr: np.ndarray
    est_porosity_ptr: np.ndarray
    viscosity_ptr: np.ndarray | None = None
    reference_k11_ptr: np.ndarray | None = None
    reference_k22_ptr: np.ndarray | None = None
    reference_k33_ptr: np.ndarray | None = None
    effective_k11_ptr: np.ndarray | None = None
    effective_k22_ptr: np.ndarray | None = None
    effective_k33_ptr: np.ndarray | None = None
    k22_to_k11: np.ndarray | None = None
    k33_to_k11: np.ndarray | None = None
    temperature_for_flow: np.ndarray | None = None
    reference_k11_for_flow: np.ndarray | None = None
    viscosity_for_flow: np.ndarray | None = None
    effective_k11_for_flow: np.ndarray | None = None


def _get_pointer(
    api,
    variable: str,
    model: str,
    package: str | None,
    nxyz: int,
    label: str,
) -> np.ndarray:
    """Resolve one live MODFLOW array and validate its cell count."""
    try:
        if package is None:
            address = api.get_var_address(variable, model)
        else:
            address = api.get_var_address(variable, model, package)
        pointer = api.get_value_ptr(address)
    except Exception as exc:
        location = f"{model}/{package}" if package else model
        raise BackendError(
            f"Could not bind {label} ({variable}) at {location}. "
            "Check the configured MODFLOW model/package names and ensure the "
            "GWE/GWF-GWE/VSC packages are present."
        ) from exc
    if pointer.size != nxyz:
        raise BackendError(
            f"{label} has {pointer.size} cells; expected nxyz={nxyz}"
        )
    return pointer


def _validate_temperature(values: np.ndarray, label: str) -> np.ndarray:
    temperature = np.asarray(values, dtype=float).ravel()
    if not np.all(np.isfinite(temperature)) or np.any(temperature <= -273.15):
        raise CouplingError(f"{label} contains invalid Celsius temperatures")
    return temperature


def _validate_positive(values: np.ndarray, label: str) -> np.ndarray:
    field = np.asarray(values, dtype=float).ravel()
    if not np.all(np.isfinite(field)) or np.any(field <= 0.0):
        raise CouplingError(f"{label} must be finite and positive")
    return field


def setup_energy_coupling(sim) -> EnergyCouplingBinding:
    """Bind GWE, EST, VSC, and NPF arrays after MODFLOW initialization."""
    if not sim.energy_enabled:
        raise ConfigurationError(
            "Thermal coupling requires energy_enabled=True"
        )
    temperature_ptr = _get_pointer(
        sim.modflow_api,
        "X",
        sim.energy_model_name,
        None,
        sim.nxyz,
        "GWE temperature",
    )
    est_porosity_ptr = _get_pointer(
        sim.modflow_api,
        "POROSITY",
        sim.energy_model_name,
        sim.est_package_name,
        sim.nxyz,
        "GWE EST porosity",
    )
    temperature = _validate_temperature(temperature_ptr, "GWE temperature")
    est_porosity = np.asarray(est_porosity_ptr, dtype=float).ravel()
    if (
        not np.all(np.isfinite(est_porosity))
        or np.any(est_porosity <= 0.0)
        or np.any(est_porosity > 1.0)
    ):
        raise CouplingError("GWE EST porosity must be finite and in (0, 1]")

    tolerance = sim.initial_gwe_field_tolerance
    if sim.validate_initial_gwe_fields:
        if not np.allclose(
            temperature, sim.temperature, rtol=0.0, atol=tolerance
        ):
            maximum = float(np.max(np.abs(temperature - sim.temperature)))
            raise ConfigurationError(
                "Initial GWE and PhreeqcRM temperatures differ; maximum "
                f"absolute difference is {maximum:.6g} degC"
            )
        if not np.allclose(
            est_porosity, sim.porosity, rtol=0.0, atol=tolerance
        ):
            maximum = float(np.max(np.abs(est_porosity - sim.porosity)))
            raise ConfigurationError(
                "Initial GWE EST and MF6PQC porosities differ; maximum "
                f"absolute difference is {maximum:.6g}"
            )

    binding = EnergyCouplingBinding(
        temperature_ptr=temperature_ptr,
        est_porosity_ptr=est_porosity_ptr,
    )
    if sim.vsc_enabled:
        binding.viscosity_ptr = _get_pointer(
            sim.modflow_api,
            "VISC",
            sim.flow_model_name,
            sim.vsc_package_name,
            sim.nxyz,
            "VSC viscosity",
        )
        binding.reference_k11_ptr = _get_pointer(
            sim.modflow_api,
            "K11INPUT",
            sim.flow_model_name,
            sim.npf_package_name,
            sim.nxyz,
            "NPF reference K11",
        )
        binding.reference_k22_ptr = _get_pointer(
            sim.modflow_api,
            "K22INPUT",
            sim.flow_model_name,
            sim.npf_package_name,
            sim.nxyz,
            "NPF reference K22",
        )
        binding.reference_k33_ptr = _get_pointer(
            sim.modflow_api,
            "K33INPUT",
            sim.flow_model_name,
            sim.npf_package_name,
            sim.nxyz,
            "NPF reference K33",
        )
        binding.effective_k11_ptr = _get_pointer(
            sim.modflow_api,
            "K11",
            sim.flow_model_name,
            sim.npf_package_name,
            sim.nxyz,
            "NPF effective K11",
        )
        binding.effective_k22_ptr = _get_pointer(
            sim.modflow_api,
            "K22",
            sim.flow_model_name,
            sim.npf_package_name,
            sim.nxyz,
            "NPF effective K22",
        )
        binding.effective_k33_ptr = _get_pointer(
            sim.modflow_api,
            "K33",
            sim.flow_model_name,
            sim.npf_package_name,
            sim.nxyz,
            "NPF effective K33",
        )
        viscosity = _validate_positive(binding.viscosity_ptr, "VSC viscosity")
        reference_k11 = _validate_positive(
            binding.reference_k11_ptr, "NPF reference K11"
        )
        reference_k22 = _validate_positive(
            binding.reference_k22_ptr, "NPF reference K22"
        )
        reference_k33 = _validate_positive(
            binding.reference_k33_ptr, "NPF reference K33"
        )
        _validate_positive(binding.effective_k11_ptr, "NPF effective K11")
        _validate_positive(binding.effective_k22_ptr, "NPF effective K22")
        _validate_positive(binding.effective_k33_ptr, "NPF effective K33")
        binding.k22_to_k11 = reference_k22 / reference_k11
        binding.k33_to_k11 = reference_k33 / reference_k11
        binding.viscosity_for_flow = viscosity.copy()
        binding.reference_k11_for_flow = reference_k11.copy()
        binding.effective_k11_for_flow = np.asarray(
            binding.effective_k11_ptr, dtype=float
        ).copy()

    binding.temperature_for_flow = temperature.copy()
    sim.energy_binding = binding
    sim.results_temperature = [temperature.copy()]
    sim.results_temperature_for_flow = [temperature.copy()]
    if sim.vsc_enabled:
        sim.results_viscosity = [binding.viscosity_for_flow.copy()]
        sim.results_reference_K = [binding.reference_k11_for_flow.copy()]
        sim.results_effective_K = [binding.effective_k11_for_flow.copy()]
    return binding


def capture_flow_inputs(sim, current_reference_k11: np.ndarray | None) -> None:
    """Record the temperature and reference K seen by the upcoming flow solve."""
    binding = sim.energy_binding
    binding.temperature_for_flow = _validate_temperature(
        binding.temperature_ptr, "GWE temperature used for flow"
    ).copy()
    if not sim.vsc_enabled:
        return
    if current_reference_k11 is None:
        reference = binding.reference_k11_ptr
    else:
        reference = current_reference_k11
    binding.reference_k11_for_flow = _validate_positive(
        reference, "NPF reference K11 used for flow"
    ).copy()


def write_reference_conductivity(sim, reference_k11: np.ndarray) -> None:
    """Write reaction-updated base K without touching VSC-owned effective K."""
    binding = sim.energy_binding
    reference = _validate_positive(reference_k11, "Reaction-updated reference K11")
    binding.reference_k11_ptr[:] = reference
    binding.reference_k22_ptr[:] = reference * binding.k22_to_k11
    binding.reference_k33_ptr[:] = reference * binding.k33_to_k11


def capture_flow_response(sim) -> None:
    """Snapshot VSC viscosity and effective K immediately after MODFLOW solves."""
    if not sim.vsc_enabled:
        return
    binding = sim.energy_binding
    binding.viscosity_for_flow = _validate_positive(
        binding.viscosity_ptr, "VSC viscosity"
    ).copy()
    binding.effective_k11_for_flow = _validate_positive(
        binding.effective_k11_ptr, "NPF effective K11"
    ).copy()


def synchronize_temperature_to_chemistry(sim) -> np.ndarray:
    """Copy the latest GWE temperature into PhreeqcRM before ``RunCells``."""
    if not sim.energy_enabled:
        return sim.temperature
    binding = getattr(sim, "energy_binding", None)
    if binding is None:
        raise CouplingError("GWE temperature is not bound to the MODFLOW API")
    temperature = _validate_temperature(
        binding.temperature_ptr, "GWE temperature for chemistry"
    ).copy()
    sim.temperature = temperature
    if sim.sync_gwe_temperature_to_phreeqc:
        sim.phreeqc_rm.SetTemperature(temperature)
    return temperature


def update_energy_porosity(sim, porosity: np.ndarray) -> None:
    """Keep GWE EST thermal storage consistent with reaction-updated porosity."""
    if not sim.energy_enabled:
        return
    values = np.asarray(porosity, dtype=float).ravel()
    if (
        values.size != sim.nxyz
        or not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
        or np.any(values > 1.0)
    ):
        raise CouplingError("Updated GWE EST porosity must be in (0, 1]")
    sim.energy_binding.est_porosity_ptr[:] = values


def save_energy_time_step_results(sim, logical_step: int) -> None:
    """Store thermal and VSC fields on the same schedule as chemistry output."""
    from mf6pqc.coupling.common import should_save_time_step

    if not should_save_time_step(sim, logical_step):
        return
    binding = sim.energy_binding
    sim.results_temperature.append(
        _validate_temperature(binding.temperature_ptr, "GWE temperature").copy()
    )
    sim.results_temperature_for_flow.append(binding.temperature_for_flow.copy())
    if sim.vsc_enabled:
        sim.results_viscosity.append(binding.viscosity_for_flow.copy())
        sim.results_reference_K.append(binding.reference_k11_for_flow.copy())
        sim.results_effective_K.append(binding.effective_k11_for_flow.copy())


def finalize_energy_results(sim) -> None:
    """Freeze thermal result lists and verify alignment with selected output."""
    if not sim.energy_enabled:
        return
    sim.results_temperature = np.asarray(sim.results_temperature, dtype=float)
    sim.results_temperature_for_flow = np.asarray(
        sim.results_temperature_for_flow, dtype=float
    )
    expected = sim.results.shape[0]
    if (
        sim.results_temperature.shape != (expected, sim.nxyz)
        or sim.results_temperature_for_flow.shape != (expected, sim.nxyz)
    ):
        raise CouplingError("Thermal outputs do not align with chemistry frames")
    if sim.vsc_enabled:
        sim.results_viscosity = np.asarray(sim.results_viscosity, dtype=float)
        sim.results_reference_K = np.asarray(sim.results_reference_K, dtype=float)
        sim.results_effective_K = np.asarray(sim.results_effective_K, dtype=float)
        expected_shape = (expected, sim.nxyz)
        if any(
            values.shape != expected_shape
            for values in (
                sim.results_viscosity,
                sim.results_reference_K,
                sim.results_effective_K,
            )
        ):
            raise CouplingError("VSC/K outputs do not align with chemistry frames")


def energy_result_payload(sim) -> dict[str, Any] | None:
    """Return named arrays for durable serialization."""
    if not sim.energy_enabled:
        return None
    payload: dict[str, Any] = {
        "temperature": sim.results_temperature,
        "temperature_for_flow": sim.results_temperature_for_flow,
    }
    if sim.vsc_enabled:
        payload.update(
            {
                "viscosity": sim.results_viscosity,
                "reference_K": sim.results_reference_K,
                "effective_K": sim.results_effective_K,
            }
        )
    return payload


__all__ = [
    "EnergyCouplingBinding",
    "capture_flow_inputs",
    "capture_flow_response",
    "energy_result_payload",
    "finalize_energy_results",
    "save_energy_time_step_results",
    "setup_energy_coupling",
    "synchronize_temperature_to_chemistry",
    "update_energy_porosity",
    "write_reference_conductivity",
]
