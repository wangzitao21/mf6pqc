import os
import time
import numpy as np
import phreeqcrm
import modflowapi

from mf6pqc.constants import (
    PHREEQCRM_TIME_CONVERSION,
    PHREEQCRM_REBALANCE_FRACTION,
    PHREEQCRM_UNITS,
    SECONDS_PER_DAY,
    MIN_CONCENTRATION,
    MIN_TIME_STEP,
    K33_RATIO,
    SOURCE_RELAXATION,
    SIA_MAX_PICARD_ITER,
    SIA_RTOL,
    SIA_ATOL,
    DENSITY_SCALE,
    DENSITY_RELAXATION,
)
from mf6pqc.utils import get_species_slice
from mf6pqc.output_processing import update_porosity, update_diffc


def initialize_phreeqcrm(sim) -> None:
    """
    Initialize and configure the PhreeqcRM instance.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance containing configuration and state.
    Returns
    -------
    None
        Creates and configures the internal PhreeqcRM object.
    """
    print("--- Initializing PhreeqcRM ---")
    sim.phreeqc_rm = phreeqcrm.PhreeqcRM(sim.nxyz, sim.nthreads)
    os.makedirs(sim.output_dir, exist_ok=True)
    prefix = os.path.join(sim.output_dir, f"{sim.case_name}_prm")
    sim.phreeqc_rm.SetFilePrefix(prefix)
    sim.phreeqc_rm.OpenFiles()
    sim.phreeqc_rm.SetUnitsSolution(PHREEQCRM_UNITS["solution"])
    sim.phreeqc_rm.SetUnitsPPassemblage(PHREEQCRM_UNITS["ppassemblage"])
    sim.phreeqc_rm.SetUnitsExchange(PHREEQCRM_UNITS["exchange"])
    sim.phreeqc_rm.SetUnitsSurface(PHREEQCRM_UNITS["surface"])
    sim.phreeqc_rm.SetUnitsGasPhase(PHREEQCRM_UNITS["gas_phase"])
    sim.phreeqc_rm.SetUnitsSSassemblage(PHREEQCRM_UNITS["ssassemblage"])
    sim.phreeqc_rm.SetUnitsKinetics(PHREEQCRM_UNITS["kinetics"])
    sim.phreeqc_rm.SetTimeConversion(PHREEQCRM_TIME_CONVERSION)
    sim.phreeqc_rm.SetTemperature(sim.temperature)
    sim.phreeqc_rm.SetPressure(sim.pressure)
    sim.phreeqc_rm.SetPorosity(sim.porosity)
    sim.phreeqc_rm.SetSaturation(sim.saturation)
    sim.phreeqc_rm.SetComponentH2O(sim.componentH2O)
    sim.phreeqc_rm.UseSolutionDensityVolume(sim.solution_density_volume)
    sim.phreeqc_rm.SetRebalanceFraction(PHREEQCRM_REBALANCE_FRACTION)
    print(f"Loading Phreeqc database: {sim.db_path}")
    sim.phreeqc_rm.LoadDatabase(sim.db_path)
    sim.phreeqc_rm.SetPrintChemistryOn(True, False, False)
    print(f"Running chemistry definition file: {sim.pqi_path}")
    sim.phreeqc_rm.RunFile(True, True, True, sim.pqi_path)
    sim.phreeqc_rm.RunString(True, False, True, "DELETE; -all")
    sim.ncomps = sim.phreeqc_rm.FindComponents()
    sim.components = list(sim.phreeqc_rm.GetComponents())
    print(f"List of reactive chemical components: {sim.components}")
    sim.phreeqc_rm.SetScreenOn(False)
    sim.phreeqc_rm.SetSelectedOutputOn(True)


def initialize_modflow6(sim) -> None:
    """
    Initialize the MODFLOW 6 API instance.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance containing configuration and state.
    Returns
    -------
    None
        Creates and initializes the internal ModflowApi object.
    """
    print("--- Initializing MODFLOW 6 ---")
    print(f"Working directory: {sim.workspace}")
    try:
        sim.modflow_api = modflowapi.ModflowApi(
            sim.modflow_dll_path, working_directory=sim.workspace
        )
    except Exception as exc:
        print("Error: failed to load MODFLOW 6 DLL or set the working directory.")
        print(f"DLL path: {sim.modflow_dll_path}")
        print(f"Working directory: {sim.workspace}")
        raise exc
    sim.modflow_api.initialize()
    sim.sim = modflowapi.extensions.ApiSimulation.load(sim.modflow_api)


def validate_setup(sim) -> None:
    """
    Ensure setup has been executed before running the simulation.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance.
    Returns
    -------
    None
        Raises if setup has not been called.
    """
    if not sim.is_setup:
        raise RuntimeError("setup() must be called before running the simulation")


def cache_basic_geometry(sim) -> None:
    """
    Cache groundwater model geometry fields for potential saturation use.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance.
    Returns
    -------
    None
        Stores geometry arrays on the simulator.
    """
    sim.head_addr = sim.modflow_api.get_var_address("X", "gwf_model")
    sim.botm_arr = sim.sim.get_model("gwf_model").dis.bot.values.ravel()
    sim.top_arr = sim.sim.get_model("gwf_model").dis.top.values.ravel()
    sim.cell_thick = sim.top_arr - sim.botm_arr


def cache_concentration_variables(modflow_api, components: list) -> dict:
    """
    Cache MODFLOW 6 concentration pointers for each component.
    Parameters
    ----------
    modflow_api : object
        ModflowApi instance.
    components : list
        List of component names.
    Returns
    -------
    dict
        Mapping from component name to pointer information.
    """
    conc_var_info = {}
    for sp_name in components:
        gwt_model_name = f"gwt_{sp_name}_model"
        address = modflow_api.get_var_address("X", gwt_model_name)
        ptr = modflow_api.get_value_ptr(address)
        conc_var_info[sp_name] = {"address": address, "ptr": ptr, "shape": ptr.shape}
        print(f"  - solute '{sp_name}', shape={ptr.shape}")
    return conc_var_info


def cache_thetam_ptrs(modflow_api, components: list) -> dict:
    """
    Cache MODFLOW 6 porosity pointers for each component.
    Parameters
    ----------
    modflow_api : object
        ModflowApi instance.
    components : list
        List of component names.
    Returns
    -------
    dict
        Mapping from component name to porosity pointer.
    """
    thetam_ptrs = {}
    for sp in components:
        thetam_addr = modflow_api.get_var_address("thetam", f"gwt_{sp}_model", "MST")
        thetam_ptrs[sp] = modflow_api.get_value_ptr(thetam_addr)
    return thetam_ptrs


def setup_porosity_k_updates(sim) -> np.ndarray | None:
    """
    Prepare MODFLOW 6 pointers for porosity and permeability updates.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance.
    Returns
    -------
    np.ndarray | None
        Initial K11 array if updates are enabled.
    """
    if not sim.if_update_porosity_K:
        return None
    sim.K11_addr = sim.modflow_api.get_var_address("K11", "gwf_model", "NPF")
    sim.K33_addr = sim.modflow_api.get_var_address("K33", "gwf_model", "NPF")
    sim.K11_ptr = sim.modflow_api.get_value_ptr(sim.K11_addr)
    sim.K33_ptr = sim.modflow_api.get_value_ptr(sim.K33_addr)
    current_k11 = sim.K11_ptr.copy()
    sim.tdis_kper_addr = sim.modflow_api.get_var_address("KPER", "TDIS")
    sim.tdis_kstp_addr = sim.modflow_api.get_var_address("KSTP", "TDIS")
    sim.kchangeper_addr = sim.modflow_api.get_var_address("KCHANGEPER", "gwf_model", "NPF")
    sim.kchangestp_addr = sim.modflow_api.get_var_address("KCHANGESTP", "gwf_model", "NPF")
    sim.nodekchange_addr = sim.modflow_api.get_var_address("NODEKCHANGE", "gwf_model", "NPF")
    sim.modflow_api.set_value(sim.nodekchange_addr, np.ones(sim.nxyz, dtype=np.int32))
    sim.results_porosity = [sim.porosity.copy()]
    sim.results_K = [current_k11]
    return current_k11


def setup_diffc_updates(sim) -> None:
    """
    Prepare MODFLOW 6 diffusion coefficient addresses.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance.
    Returns
    -------
    None
        Stores diffusion coefficient addresses on the simulator.
    """
    if not sim.if_update_diffc:
        return
    sim.diffc_tags = {
        sp: sim.modflow_api.get_var_address("DIFFC", f"gwt_{sp}_model", "DSP")
        for sp in sim.components
    }


def setup_density_update(sim) -> None:
    """
    Prepare MODFLOW 6 density pointer if density feedback is enabled.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance.
    Returns
    -------
    None
        Stores density pointer on the simulator.
    """
    if not sim.if_update_density:
        return
    sim.density_addr = sim.modflow_api.get_var_address("DENSE", "gwf_model", "BUY")
    sim.density_ptr = sim.modflow_api.get_value_ptr(sim.density_addr)


def cache_solution_iterations(modflow_api) -> dict:
    """
    Cache IMS maximum iterations for each solution.
    Parameters
    ----------
    modflow_api : object
        ModflowApi instance.
    Returns
    -------
    dict
        Mapping from solution id to iteration pointer.
    """
    n_solutions = modflow_api.get_subcomponent_count()
    sol_iters_map = {}
    for solution_id in range(1, n_solutions + 1):
        sln_name = f"SLN_{solution_id}"
        ims_addr = modflow_api.get_var_address("MXITER", sln_name)
        sol_iters_map[solution_id] = modflow_api.get_value_ptr(ims_addr)
    return sol_iters_map


def allocate_concentration_buffers(nxyz: int, ncomps: int) -> tuple:
    """
    Allocate concentration buffers for transport and reaction.
    Parameters
    ----------
    nxyz : int
        Number of computational cells.
    ncomps : int
        Number of chemical components.
    Returns
    -------
    tuple
        conc_from_mf, conc_after_reaction arrays.
    """
    conc_from_mf = np.empty(nxyz * ncomps, dtype=float)
    conc_after_reaction = np.empty_like(conc_from_mf)
    return conc_from_mf, conc_after_reaction


def build_species_slices(nxyz: int, ncomps: int) -> list:
    """
    Build slices for component blocks in concentration vectors.
    Parameters
    ----------
    nxyz : int
        Number of computational cells.
    ncomps : int
        Number of chemical components.
    Returns
    -------
    list
        List of slice objects for each component.
    """
    return [get_species_slice(nxyz, i) for i in range(ncomps)]


def update_k_for_time_step(sim, current_k11: np.ndarray, time_step_index: int) -> None:
    """
    Update permeability arrays and dirty flags in MODFLOW 6.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance.
    current_k11 : np.ndarray
        Current K11 field.
    time_step_index : int
        Current time step index.
    Returns
    -------
    None
        Writes updated K values into MODFLOW 6.
    """
    if not sim.if_update_porosity_K or time_step_index <= 0:
        return
    current_kper = sim.modflow_api.get_value(sim.tdis_kper_addr)
    current_kstp = sim.modflow_api.get_value(sim.tdis_kstp_addr)
    sim.modflow_api.set_value(sim.kchangeper_addr, current_kper)
    sim.modflow_api.set_value(sim.kchangestp_addr, current_kstp)
    sim.K11_ptr[:] = current_k11
    sim.K33_ptr[:] = current_k11 * K33_RATIO


def solve_modflow_solutions(sim, sol_iters_map: dict, current_density: np.ndarray | None) -> None:
    """
    Solve MODFLOW 6 for all solutions in the current time step.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance.
    sol_iters_map : dict
        Mapping from solution id to iteration pointer.
    current_density : np.ndarray | None
        Current density field to apply to the flow model.
    Returns
    -------
    None
        Advances MODFLOW 6 to the end of the time step.
    """
    n_solutions = sim.modflow_api.get_subcomponent_count()
    for solution_id in range(1, n_solutions + 1):
        sim.modflow_api.prepare_solve(solution_id)
        if sim.if_update_density and solution_id == 1 and current_density is not None:
            sim.density_ptr[:] = current_density
        current_max_iter = sol_iters_map[solution_id][0]
        kiter = 0
        has_converged = False
        while kiter < current_max_iter:
            has_converged = sim.modflow_api.solve(solution_id)
            kiter += 1
            if has_converged:
                break
        sim.modflow_api.finalize_solve(solution_id)


def read_concentrations_from_modflow(
    conc_var_info: dict, species_slices: list, conc_from_mf: np.ndarray
) -> None:
    """
    Read concentrations from MODFLOW 6 into the transport buffer.
    Parameters
    ----------
    conc_var_info : dict
        Concentration pointer map.
    species_slices : list
        Slice list for each component.
    conc_from_mf : np.ndarray
        Output buffer for concentrations.
    Returns
    -------
    None
        Populates the concentration buffer.
    """
    for i, sp in enumerate(conc_var_info.keys()):
        ptr = conc_var_info[sp]["ptr"]
        sl = species_slices[i]
        conc_from_mf[sl] = ptr


def write_concentrations_to_modflow(
    conc_var_info: dict, species_slices: list, conc_after_reaction: np.ndarray
) -> None:
    """
    Write reacted concentrations back to MODFLOW 6.
    Parameters
    ----------
    conc_var_info : dict
        Concentration pointer map.
    species_slices : list
        Slice list for each component.
    conc_after_reaction : np.ndarray
        Reacted concentration buffer.
    Returns
    -------
    None
        Updates MODFLOW 6 concentration arrays in place.
    """
    for i, sp in enumerate(conc_var_info.keys()):
        ptr = conc_var_info[sp]["ptr"]
        sl = species_slices[i]
        ptr[:] = conc_after_reaction[sl].reshape(ptr.shape)


def run_reaction_step(
    sim,
    conc_from_mf: np.ndarray,
    conc_after_reaction: np.ndarray,
    current_time: float,
    dt: float,
) -> None:
    """
    Run PhreeqcRM reactions for a single time step.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance.
    conc_from_mf : np.ndarray
        Transport concentrations.
    conc_after_reaction : np.ndarray
        Output buffer for reacted concentrations.
    current_time : float
        Current simulation time in days.
    dt : float
        Time step length in days.
    Returns
    -------
    None
        Updates conc_after_reaction in place.
    """
    sim.phreeqc_rm.SetConcentrations(conc_from_mf)
    sim.phreeqc_rm.SetTime(current_time * SECONDS_PER_DAY)
    sim.phreeqc_rm.SetTimeStep(dt * SECONDS_PER_DAY)
    sim.phreeqc_rm.RunCells()
    conc_after_reaction[:] = sim.phreeqc_rm.GetConcentrations()


def update_porosity_and_diffc(sim, current_k11: np.ndarray, time_step_index: int) -> np.ndarray:
    """
    Update porosity, permeability, and diffusion coefficients.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance.
    current_k11 : np.ndarray
        Current K11 field.
    time_step_index : int
        Current time step index.
    Returns
    -------
    np.ndarray
        Updated K11 field.
    """
    if sim.if_update_porosity_K:
        old_porosity = sim.porosity.copy()
        new_porosity = update_porosity(
            sim.selected_output, sim.output_indices, sim.mineral_volumes, sim.porosity
        )
        sim.porosity = new_porosity
        sim.phreeqc_rm.SetPorosity(sim.porosity)
        for sp in sim.components:
            sim.thetam_ptrs[sp][:] = new_porosity
        current_k11 = sim._update_K(current_k11, old_porosity, new_porosity)
        if time_step_index % sim.save_interval == 0:
            sim.results_porosity.append(new_porosity)
            sim.results_K.append(current_k11)
    if sim.if_update_diffc:
        new_diffc = update_diffc(sim.porosity, sim.d0)
        if time_step_index % sim.save_interval == 0:
            sim.results_diffc.append(new_diffc)
        for sp in sim.components:
            sim.modflow_api.set_value(sim.diffc_tags[sp], new_diffc)
    return current_k11


def save_time_step_results(sim, time_step_index: int) -> None:
    """
    Append selected output to the results list.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance.
    time_step_index : int
        Current time step index.
    Returns
    -------
    None
        Stores selected output for later saving.
    """
    if time_step_index % sim.save_interval == 0:
        sim.results.append(sim.selected_output.copy())


def log_progress(current_time: float, end_time: float, time_step_index: int, suffix: str = "") -> None:
    """
    Print progress message every 20 steps.
    Parameters
    ----------
    current_time : float
        Current simulation time.
    end_time : float
        End simulation time.
    time_step_index : int
        Current time step index.
    suffix : str
        Optional suffix for progress line.
    Returns
    -------
    None
        Prints progress updates.
    """
    if time_step_index % 20 == 0:
        if suffix:
            print(f"  t = {current_time:.2f}/{end_time:.2f} days, step={time_step_index}, {suffix}")
        else:
            print(f"  t = {current_time:.2f}/{end_time:.2f} days, step={time_step_index}")


def build_standard_state(sim) -> dict:
    """
    Build the shared state for standard coupling.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance.
    Returns
    -------
    dict
        State dictionary for the standard loop.
    """
    conc_var_info = cache_concentration_variables(sim.modflow_api, sim.components)
    sim.thetam_ptrs = cache_thetam_ptrs(sim.modflow_api, sim.components)
    current_k11 = setup_porosity_k_updates(sim)
    setup_diffc_updates(sim)
    setup_density_update(sim)
    sol_iters_map = cache_solution_iterations(sim.modflow_api)
    conc_from_mf, conc_after_reaction = allocate_concentration_buffers(sim.nxyz, sim.ncomps)
    species_slices = build_species_slices(sim.nxyz, sim.ncomps)
    current_time = sim.modflow_api.get_current_time()
    end_time = sim.modflow_api.get_end_time()
    return {
        "conc_var_info": conc_var_info,
        "species_slices": species_slices,
        "conc_from_mf": conc_from_mf,
        "conc_after_reaction": conc_after_reaction,
        "sol_iters_map": sol_iters_map,
        "current_time": current_time,
        "end_time": end_time,
        "time_step_index": 0,
        "current_k11": current_k11,
    }


def standard_time_step(sim, state: dict) -> None:
    """
    Advance one time step for standard coupling.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance.
    state : dict
        Standard coupling state dictionary.
    Returns
    -------
    None
        Updates the state in place for the next time step.
    """
    dt = sim.modflow_api.get_time_step()
    sim.modflow_api.prepare_time_step(dt)
    update_k_for_time_step(sim, state["current_k11"], state["time_step_index"])
    current_density = None
    if sim.if_update_density:
        current_density = sim.phreeqc_rm.GetSelectedOutput()[-sim.nxyz:] * DENSITY_SCALE
    solve_modflow_solutions(sim, state["sol_iters_map"], current_density)
    sim.modflow_api.finalize_time_step()
    state["current_time"] = sim.modflow_api.get_current_time()
    read_concentrations_from_modflow(
        state["conc_var_info"], state["species_slices"], state["conc_from_mf"]
    )
    state["conc_from_mf"][state["conc_from_mf"] < MIN_CONCENTRATION] = MIN_CONCENTRATION
    run_reaction_step(
        sim,
        state["conc_from_mf"],
        state["conc_after_reaction"],
        state["current_time"],
        dt,
    )
    temp_selected = sim.phreeqc_rm.GetSelectedOutput()
    sim.selected_output = temp_selected.reshape(-1, sim.nxyz)
    write_concentrations_to_modflow(
        state["conc_var_info"], state["species_slices"], state["conc_after_reaction"]
    )
    state["current_k11"] = update_porosity_and_diffc(
        sim, state["current_k11"], state["time_step_index"]
    )
    save_time_step_results(sim, state["time_step_index"])
    state["time_step_index"] += 1
    log_progress(state["current_time"], state["end_time"], state["time_step_index"])


def finalize_standard(sim, state: dict, start_sim_time: float) -> None:
    """
    Finalize the standard coupling run.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance.
    state : dict
        Standard coupling state dictionary.
    start_sim_time : float
        Wall-clock start time.
    Returns
    -------
    None
        Writes final arrays to the simulator state.
    """
    sim.results = np.array(sim.results)
    if sim.if_update_porosity_K:
        sim.results_porosity = np.array(sim.results_porosity)
        sim.results_K = np.array(sim.results_K)
    if sim.if_update_diffc:
        sim.results_diffc = np.array(sim.results_diffc)
    sim.final_time_step_index = state["time_step_index"]
    elapsed = time.time() - start_sim_time
    print(f"--- Simulation finished, steps={state['time_step_index']}, time={elapsed:.2f} s ---")


def run_standard(sim) -> None:
    """
    Run the standard reactive transport coupling loop.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance.
    Returns
    -------
    None
        Advances the simulation and stores results on sim.
    """
    initialize_modflow6(sim)
    validate_setup(sim)
    print("\n--- Starting reactive transport simulation ---")
    start_sim_time = time.time()
    cache_basic_geometry(sim)
    state = build_standard_state(sim)
    while state["current_time"] < state["end_time"]:
        standard_time_step(sim, state)
    finalize_standard(sim, state, start_sim_time)


def cache_src_var_info(modflow_api, components: list) -> dict:
    """
    Cache MODFLOW 6 source term pointers for each component.
    Parameters
    ----------
    modflow_api : object
        ModflowApi instance.
    components : list
        List of component names.
    Returns
    -------
    dict
        Mapping from component name to source pointer.
    """
    src_var_info = {}
    for sp_name in components:
        gwt_model_name = f"gwt_{sp_name}_model"
        src_addr = modflow_api.get_var_address("SMASSRATE", gwt_model_name, "SRC")
        src_var_info[sp_name] = {"ptr": modflow_api.get_value_ptr(src_addr)}
    return src_var_info


def cache_cell_volume(sim, gwt_model_name: str) -> np.ndarray:
    """
    Compute cell volume for transport calculations.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance.
    gwt_model_name : str
        Name of the GWT model to query.
    Returns
    -------
    np.ndarray
        Cell volumes multiplied by porosity.
    """
    area = sim.modflow_api.get_value_ptr(sim.modflow_api.get_var_address("AREA", gwt_model_name, "DIS"))
    top = sim.modflow_api.get_value_ptr(sim.modflow_api.get_var_address("TOP", gwt_model_name, "DIS"))
    botm = sim.modflow_api.get_value_ptr(sim.modflow_api.get_var_address("BOT", gwt_model_name, "DIS"))
    volume = area * (top - botm)
    return volume * sim.porosity


def build_sia_state(sim) -> dict:
    """
    Build the shared state for SIA coupling.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance.
    Returns
    -------
    dict
        State dictionary for the SIA loop.
    """
    conc_var_info = cache_concentration_variables(sim.modflow_api, sim.components)
    sim.thetam_ptrs = cache_thetam_ptrs(sim.modflow_api, sim.components)
    current_k11 = setup_porosity_k_updates(sim)
    setup_diffc_updates(sim)
    setup_density_update(sim)
    sol_iters_map = cache_solution_iterations(sim.modflow_api)
    conc_from_mf, conc_after_reaction = allocate_concentration_buffers(sim.nxyz, sim.ncomps)
    conc_prev_time = np.zeros_like(conc_from_mf)
    conc_last_picard_iter = np.zeros_like(conc_from_mf)
    diff_buffer = np.empty_like(conc_from_mf)
    species_slices = build_species_slices(sim.nxyz, sim.ncomps)
    current_time = sim.modflow_api.get_current_time()
    end_time = sim.modflow_api.get_end_time()
    src_var_info = cache_src_var_info(sim.modflow_api, sim.components)
    gwt_model_name = f"gwt_{sim.components[0]}_model"
    cell_volume = cache_cell_volume(sim, gwt_model_name)
    prev_density = sim.density.copy() if sim.if_update_density else None
    return {
        "conc_var_info": conc_var_info,
        "species_slices": species_slices,
        "conc_from_mf": conc_from_mf,
        "conc_after_reaction": conc_after_reaction,
        "conc_prev_time": conc_prev_time,
        "conc_last_picard_iter": conc_last_picard_iter,
        "diff_buffer": diff_buffer,
        "sol_iters_map": sol_iters_map,
        "src_var_info": src_var_info,
        "cell_volume": cell_volume,
        "current_time": current_time,
        "end_time": end_time,
        "time_step_index": 0,
        "current_k11": current_k11,
        "prev_density": prev_density,
    }


def restore_concentrations(state: dict) -> None:
    """
    Restore MODFLOW 6 concentrations to the previous time level.
    Parameters
    ----------
    state : dict
        SIA state dictionary.
    Returns
    -------
    None
        Writes previous concentrations into MODFLOW 6 arrays.
    """
    for i, sp in enumerate(state["conc_var_info"].keys()):
        ptr = state["conc_var_info"][sp]["ptr"]
        sl = state["species_slices"][i]
        ptr[:] = state["conc_prev_time"][sl].reshape(ptr.shape)


def solve_modflow_picard(sim, state: dict, raw_new_density: np.ndarray | None) -> None:
    """
    Solve MODFLOW 6 within a Picard iteration.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance.
    state : dict
        SIA state dictionary.
    raw_new_density : np.ndarray | None
        Density from chemistry for relaxation.
    Returns
    -------
    None
        Updates MODFLOW 6 to the end of the Picard iteration.
    """
    n_solutions = sim.modflow_api.get_subcomponent_count()
    for solution_id in range(1, n_solutions + 1):
        sim.modflow_api.prepare_solve(solution_id)
        current_max_iter = state["sol_iters_map"][solution_id][0]
        if sim.if_update_density and solution_id == 1 and raw_new_density is not None:
            relaxed_density = (
                (1.0 - DENSITY_RELAXATION) * state["prev_density"]
                + DENSITY_RELAXATION * raw_new_density
            )
            sim.density_ptr[:] = relaxed_density
            state["prev_density"] = relaxed_density.copy()
        kiter = 0
        has_converged = False
        while kiter < current_max_iter:
            has_converged = sim.modflow_api.solve(solution_id)
            kiter += 1
            if has_converged:
                break
        sim.modflow_api.finalize_solve(solution_id)


def update_sources(sim, state: dict, dt: float) -> None:
    """
    Update source terms based on reaction correction.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance.
    state : dict
        SIA state dictionary.
    dt : float
        Time step length.
    Returns
    -------
    None
        Writes source term updates to MODFLOW 6.
    """
    for i, sp in enumerate(sim.components):
        src_ptr = state["src_var_info"][sp]["ptr"]
        sl = state["species_slices"][i]
        c_react = state["conc_after_reaction"][sl]
        c_trans = state["conc_from_mf"][sl]
        if dt > MIN_TIME_STEP:
            calculated_rate = (c_react - c_trans) * state["cell_volume"] / dt
        else:
            calculated_rate = 0.0
        src_ptr[:] = (1.0 - SOURCE_RELAXATION) * src_ptr[:] + SOURCE_RELAXATION * calculated_rate


def check_picard_convergence(state: dict) -> bool:
    """
    Check convergence of Picard iterations.
    Parameters
    ----------
    state : dict
        SIA state dictionary.
    Returns
    -------
    bool
        True if convergence criteria are satisfied.
    """
    np.subtract(state["conc_from_mf"], state["conc_last_picard_iter"], out=state["diff_buffer"])
    np.abs(state["diff_buffer"], out=state["diff_buffer"])
    tolerance_threshold = SIA_ATOL + (SIA_RTOL * np.abs(state["conc_from_mf"]))
    return np.all(state["diff_buffer"] <= tolerance_threshold)


def run_picard_iteration(sim, state: dict, current_time: float, dt: float) -> bool:
    """
    Run a single Picard iteration for SIA coupling.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance.
    state : dict
        SIA state dictionary.
    current_time : float
        Current simulation time in days.
    dt : float
        Time step length in days.
    Returns
    -------
    bool
        True if convergence is achieved in this iteration.
    """
    restore_concentrations(state)
    raw_new_density = None
    if sim.if_update_density:
        raw_new_density = sim.phreeqc_rm.GetSelectedOutput()[-sim.nxyz:] * DENSITY_SCALE
    solve_modflow_picard(sim, state, raw_new_density)
    read_concentrations_from_modflow(
        state["conc_var_info"], state["species_slices"], state["conc_from_mf"]
    )
    np.maximum(state["conc_from_mf"], 0.0, out=state["conc_from_mf"])
    sim.phreeqc_rm.StateApply(1)
    run_reaction_step(
        sim, state["conc_from_mf"], state["conc_after_reaction"], current_time, dt
    )
    update_sources(sim, state, dt)
    if check_picard_convergence(state):
        return True
    np.copyto(state["conc_last_picard_iter"], state["conc_from_mf"])
    return False


def run_picard_loop(sim, state: dict, current_time: float, dt: float) -> int:
    """
    Run Picard iterations until convergence or reaching max iterations.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance.
    state : dict
        SIA state dictionary.
    current_time : float
        Current simulation time in days.
    dt : float
        Time step length in days.
    Returns
    -------
    int
        Number of Picard iterations executed.
    """
    picard_k = 0
    while picard_k < SIA_MAX_PICARD_ITER:
        if run_picard_iteration(sim, state, current_time, dt):
            print(f"  -- SIA converged at iter {picard_k}")
            return picard_k
        picard_k += 1
    print("  -- SIA not converged")
    return picard_k


def update_sia_after_step(sim, state: dict, picard_k: int) -> None:
    """
    Update outputs and transport properties after an SIA step.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance.
    state : dict
        SIA state dictionary.
    picard_k : int
        Number of Picard iterations used in the step.
    Returns
    -------
    None
        Updates outputs and state for the next step.
    """
    temp_selected = sim.phreeqc_rm.GetSelectedOutput()
    sim.selected_output = temp_selected.reshape(-1, sim.nxyz)
    state["current_k11"] = update_porosity_and_diffc(
        sim, state["current_k11"], state["time_step_index"]
    )
    save_time_step_results(sim, state["time_step_index"])
    state["time_step_index"] += 1
    log_progress(state["current_time"], state["end_time"], state["time_step_index"], f"SIA iters={picard_k}")


def sia_time_step(sim, state: dict) -> None:
    """
    Advance one time step for SIA coupling.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance.
    state : dict
        SIA state dictionary.
    Returns
    -------
    None
        Updates the state in place for the next time step.
    """
    dt = sim.modflow_api.get_time_step()
    sim.modflow_api.prepare_time_step(dt)
    update_k_for_time_step(sim, state["current_k11"], state["time_step_index"])
    sim.phreeqc_rm.StateSave(1)
    read_concentrations_from_modflow(
        state["conc_var_info"], state["species_slices"], state["conc_prev_time"]
    )
    picard_k = run_picard_loop(sim, state, state["current_time"], dt)
    sim.phreeqc_rm.StateDelete(1)
    sim.modflow_api.finalize_time_step()
    state["current_time"] = sim.modflow_api.get_current_time()
    update_sia_after_step(sim, state, picard_k)


def finalize_sia(sim, state: dict, start_sim_time: float) -> None:
    """
    Finalize the SIA coupling run.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance.
    state : dict
        SIA coupling state dictionary.
    start_sim_time : float
        Wall-clock start time.
    Returns
    -------
    None
        Writes final arrays to the simulator state.
    """
    sim.results = np.array(sim.results)
    sim.final_time_step_index = state["time_step_index"]
    elapsed = time.time() - start_sim_time
    print(f"--- Simulation finished, steps={state['time_step_index']}, time={elapsed:.2f} s ---")


def run_sia(sim) -> None:
    """
    Run the SIA coupling loop with source feedback.
    Parameters
    ----------
    sim : mf6pqc
        Simulator instance.
    Returns
    -------
    None
        Advances the simulation and stores results on sim.
    """
    initialize_modflow6(sim)
    validate_setup(sim)
    print("\n--- Starting reactive transport simulation (SIA with Source Feedback) ---")
    start_sim_time = time.time()
    state = build_sia_state(sim)
    while state["current_time"] < state["end_time"]:
        sia_time_step(sim, state)
    finalize_sia(sim, state, start_sim_time)
