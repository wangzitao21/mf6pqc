"""Hydrogeological-property feedback from chemistry to MODFLOW 6.

This module owns porosity, hydraulic-conductivity, diffusion, density-pointer,
and boundary-conductance updates. Coupling algorithms call this boundary at
their defined reaction-commit point (the midpoint for Strang); they do not
implement constitutive relationships themselves.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mf6pqc.constants import K33_RATIO
from mf6pqc.exceptions import BackendError, ConfigurationError, PropertyUpdateError
from mf6pqc.output_processing import update_diffc, update_porosity
from mf6pqc.utils import get_gwt_model_name


def cache_porosity_pointers(modflow_api, components: list[str]) -> dict[str, np.ndarray]:
    """Return live GWT MST porosity arrays for every transported component."""
    pointers: dict[str, np.ndarray] = {}
    for component in components:
        address = modflow_api.get_var_address(
            "THETAM", get_gwt_model_name(component), "MST"
        )
        pointers[component] = modflow_api.get_value_ptr(address)
    return pointers


def setup_porosity_and_conductivity(sim) -> np.ndarray | None:
    """Cache NPF update addresses and return the initial K11 field."""
    if not sim.if_update_porosity_K:
        return None
    if getattr(sim, "energy_enabled", False) and getattr(sim, "vsc_enabled", False):
        binding = getattr(sim, "energy_binding", None)
        if binding is None:
            raise BackendError("VSC coupling was not bound before K feedback setup")
        # VSC owns K11/K22/K33.  Chemistry updates only the reference arrays
        # from which VSC rebuilds effective conductivity every flow solve.
        sim.K11_ptr = binding.reference_k11_ptr
        sim.K33_ptr = binding.reference_k33_ptr
    else:
        sim.K11_addr = sim.modflow_api.get_var_address(
            "K11", sim.flow_model_name, sim.npf_package_name
        )
        sim.K33_addr = sim.modflow_api.get_var_address(
            "K33", sim.flow_model_name, sim.npf_package_name
        )
        sim.K11_ptr = sim.modflow_api.get_value_ptr(sim.K11_addr)
        sim.K33_ptr = sim.modflow_api.get_value_ptr(sim.K33_addr)
    if sim.K11_ptr.size != sim.nxyz or sim.K33_ptr.size != sim.nxyz:
        raise BackendError(
            "NPF K arrays do not match nxyz: "
            f"K11={sim.K11_ptr.size}, K33={sim.K33_ptr.size}, nxyz={sim.nxyz}"
        )
    current_k11 = np.asarray(sim.K11_ptr, dtype=float).copy()
    sim.tdis_kper_addr = sim.modflow_api.get_var_address("KPER", "TDIS")
    sim.tdis_kstp_addr = sim.modflow_api.get_var_address("KSTP", "TDIS")
    sim.kchangeper_addr = sim.modflow_api.get_var_address(
        "KCHANGEPER", sim.flow_model_name, sim.npf_package_name
    )
    sim.kchangestp_addr = sim.modflow_api.get_var_address(
        "KCHANGESTP", sim.flow_model_name, sim.npf_package_name
    )
    sim.nodekchange_addr = sim.modflow_api.get_var_address(
        "NODEKCHANGE", sim.flow_model_name, sim.npf_package_name
    )
    sim.modflow_api.set_value(
        sim.nodekchange_addr, np.ones(sim.nxyz, dtype=np.int32)
    )
    sim.results_porosity = [sim.porosity.copy()]
    sim.results_K = [current_k11.copy()]
    if sim.if_update_density:
        sim.k_update_density_prev = sim.selected_output[-1].copy()
        sim.k_update_viscosity_prev = sim.viscosity.copy()
    return current_k11


def setup_boundary_conductance_updates(sim) -> None:
    """Validate and cache GHB conductance pointers coupled to adjacent K."""
    sim.boundary_conductance_ptrs = []
    for package_name, raw_config in sim.boundary_conductance_updates.items():
        config: dict[str, Any] = dict(raw_config)
        try:
            cell_index = int(config["cell_index"])
            distance = float(config["distance"])
            area = float(config.get("area", 1.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"Invalid boundary_conductance_updates entry for {package_name!r}"
            ) from exc
        if not -sim.nxyz <= cell_index < sim.nxyz:
            raise ConfigurationError(
                f"Boundary cell_index for {package_name!r} is outside the model: "
                f"{cell_index}"
            )
        if distance <= 0.0 or area <= 0.0:
            raise ConfigurationError(
                f"Boundary distance and area for {package_name!r} must be positive"
            )
        address = sim.modflow_api.get_var_address(
            "COND", sim.flow_model_name, package_name
        )
        pointer = sim.modflow_api.get_value_ptr(address)
        if pointer.size != 1:
            raise BackendError(
                f"Expected one GHB bound for {package_name!r}, got {pointer.size}"
            )
        sim.boundary_conductance_ptrs.append(
            (pointer, cell_index, distance, area)
        )


def update_boundary_conductances(sim, current_k11: np.ndarray) -> None:
    """Set configured GHB conductance to ``K * face_area / distance``."""
    for pointer, cell_index, distance, area in sim.boundary_conductance_ptrs:
        pointer[0] = current_k11[cell_index] * area / distance


def setup_diffusion_updates(sim) -> None:
    """Cache GWT DSP diffusion addresses when diffusion feedback is enabled."""
    if not sim.if_update_diffc:
        return
    sim.diffc_tags = {
        component: sim.modflow_api.get_var_address(
            "DIFFC", get_gwt_model_name(component), "DSP"
        )
        for component in sim.components
    }


def setup_density_update(sim) -> None:
    """Cache the GWF BUY density array when density feedback is enabled."""
    if not sim.if_update_density:
        return
    sim.density_addr = sim.modflow_api.get_var_address(
        "DENSE", sim.flow_model_name, "BUY"
    )
    sim.density_ptr = sim.modflow_api.get_value_ptr(sim.density_addr)
    if sim.density_ptr.size != sim.nxyz:
        raise BackendError(
            f"BUY density array has {sim.density_ptr.size} cells; expected {sim.nxyz}"
        )


def prepare_feedback(sim) -> np.ndarray | None:
    """Prepare all enabled feedback mechanisms and return initial K11."""
    sim.thetam_ptrs = cache_porosity_pointers(sim.modflow_api, sim.components)
    current_k11 = setup_porosity_and_conductivity(sim)
    setup_boundary_conductance_updates(sim)
    setup_diffusion_updates(sim)
    setup_density_update(sim)
    return current_k11


def write_conductivity_for_step(
    sim,
    current_k11: np.ndarray | None,
    logical_step: int,
    *,
    force: bool = False,
) -> None:
    """Write K and its MODFLOW dirty flags before a step is solved.

    ``force`` is used by the second Strang half-step: chemistry has just
    changed K inside logical step zero, so the historical ``logical_step > 0``
    shortcut must not suppress that midpoint update.
    """
    if not sim.if_update_porosity_K or (logical_step <= 0 and not force):
        return
    if current_k11 is None:
        raise BackendError("K feedback is enabled but no current K11 field exists")
    if getattr(sim, "energy_enabled", False) and getattr(sim, "vsc_enabled", False):
        from mf6pqc.energy import write_reference_conductivity

        write_reference_conductivity(sim, current_k11)
        return
    current_kper = sim.modflow_api.get_value(sim.tdis_kper_addr)
    current_kstp = sim.modflow_api.get_value(sim.tdis_kstp_addr)
    sim.modflow_api.set_value(sim.kchangeper_addr, current_kper)
    sim.modflow_api.set_value(sim.kchangestp_addr, current_kstp)
    sim.K11_ptr[:] = current_k11
    sim.K33_ptr[:] = current_k11 * getattr(sim, "k33_ratio", K33_RATIO)
    update_boundary_conductances(sim, current_k11)


def update_medium_properties(
    sim, current_k11: np.ndarray | None, logical_step: int
) -> np.ndarray | None:
    """Apply feedback at the coupling algorithm's reaction-commit point."""
    from mf6pqc.coupling.common import should_save_time_step

    if sim.if_update_porosity_K:
        if current_k11 is None:
            raise BackendError("K feedback is enabled but no current K11 field exists")
        old_porosity = sim.porosity.copy()
        proposed_porosity = update_porosity(
            sim.selected_output,
            sim.output_indices,
            sim.mineral_volumes,
            old_porosity,
        )
        new_porosity = np.where(
            sim.porosity_update_mask, proposed_porosity, old_porosity
        )
        sim.porosity = new_porosity
        sim.phreeqc_rm.SetPorosity(new_porosity)
        for pointer in sim.thetam_ptrs.values():
            pointer[:] = new_porosity
        if getattr(sim, "energy_enabled", False):
            from mf6pqc.energy import update_energy_porosity

            update_energy_porosity(sim, new_porosity)

        if sim.if_update_density:
            density_new = sim.selected_output[-1].copy()
            viscosity_new = sim.viscosity.copy()
            current_k11 = sim._update_K(
                current_k11,
                old_porosity,
                new_porosity,
                density_old=sim.k_update_density_prev,
                density_new=density_new,
                viscosity_old=sim.k_update_viscosity_prev,
                viscosity_new=viscosity_new,
            )
            sim.k_update_density_prev = density_new
            sim.k_update_viscosity_prev = viscosity_new
        else:
            current_k11 = sim._update_K(
                current_k11, old_porosity, new_porosity
            )
        if not np.all(np.isfinite(current_k11)) or np.any(current_k11 < 0.0):
            raise PropertyUpdateError(
                "Permeability updater returned non-finite or negative K values"
            )
        if should_save_time_step(sim, logical_step):
            sim.results_porosity.append(new_porosity.copy())
            sim.results_K.append(current_k11.copy())

    if sim.if_update_diffc:
        new_diffc = update_diffc(sim.porosity, sim.d0)
        if not np.all(np.isfinite(new_diffc)) or np.any(new_diffc < 0.0):
            raise PropertyUpdateError(
                "Diffusion updater returned non-finite or negative values"
            )
        if should_save_time_step(sim, logical_step):
            sim.results_diffc.append(new_diffc.copy())
        for component, address in sim.diffc_tags.items():
            sim.modflow_api.set_value(address, new_diffc)
    return current_k11


# Historical internal names retained for downstream scripts.
cache_thetam_ptrs = cache_porosity_pointers
setup_porosity_k_updates = setup_porosity_and_conductivity
setup_diffc_updates = setup_diffusion_updates
update_k_for_time_step = write_conductivity_for_step
update_porosity_and_diffc = update_medium_properties
