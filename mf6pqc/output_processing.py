import os
import numpy as np

from mf6pqc.constants import MIN_POROSITY, MAX_POROSITY


def extract_output_information(headings: list, vm_minerals: dict) -> tuple:
    """
    Extract mineral output indices and molar volumes from headings.
    Parameters
    ----------
    headings : list
        Selected output headings from PhreeqcRM.
    vm_minerals : dict
        Molar volume lookup table for minerals.
    Returns
    -------
    tuple
        output_indices, mineral_volumes, mineral_names arrays.
    """
    output_indices = []
    mineral_volumes = []
    mineral_names = []
    for idx, heading in enumerate(headings):
        if not (heading.startswith("d_") and len(heading) > 2):
            continue
        mineral_name = heading[2:]
        if mineral_name not in vm_minerals:
            raise ValueError(f"Cannot find molar volume for '{mineral_name}'")
        output_indices.append(idx)
        mineral_volumes.append(vm_minerals[mineral_name])
        mineral_names.append(mineral_name)
    return (
        np.array(output_indices, dtype=int),
        np.array(mineral_volumes, dtype=float).reshape(-1, 1),
        np.array(mineral_names),
    )


def update_porosity(
    selected_output: np.ndarray,
    output_indices: np.ndarray,
    mineral_volumes: np.ndarray,
    porosity: np.ndarray,
) -> np.ndarray:
    """
    Update porosity based on mineral volume changes.
    Parameters
    ----------
    selected_output : np.ndarray
        Selected output array with mineral mole changes.
    output_indices : np.ndarray
        Indices of mineral mole change rows.
    mineral_volumes : np.ndarray
        Molar volumes for minerals.
    porosity : np.ndarray
        Current porosity field.
    Returns
    -------
    np.ndarray
        Updated porosity field after reaction.
    """
    mineral_delta_moles = selected_output[output_indices, :]
    total_volume_change = np.sum(mineral_volumes * mineral_delta_moles, axis=0)
    new_porosity = porosity - total_volume_change
    new_porosity = np.maximum(MIN_POROSITY, new_porosity)
    new_porosity = np.minimum(MAX_POROSITY, new_porosity)
    return new_porosity


def update_diffc(new_porosity: np.ndarray, d0: np.ndarray) -> np.ndarray:
    """
    Compute effective diffusion coefficient from porosity.
    Parameters
    ----------
    new_porosity : np.ndarray
        Updated porosity field.
    d0 : np.ndarray
        Free-water diffusion coefficient.
    Returns
    -------
    np.ndarray
        Effective diffusion coefficient field.
    """
    tortuosity_factor = new_porosity ** (1.0 / 3.0)
    return tortuosity_factor * d0


def save_results(
    output_dir: str,
    case_name: str,
    headings: list,
    results: np.ndarray,
    results_porosity: list,
    results_k: list,
    results_diffc: list,
    if_update_porosity_k: bool,
    if_update_diffc: bool,
    filename: str | None = None,
) -> None:
    """
    Save selected outputs and transport properties to disk.
    Parameters
    ----------
    output_dir : str
        Output directory for result files.
    case_name : str
        Case name used for file naming.
    headings : list
        Selected output headings.
    results : np.ndarray
        Selected output time series.
    results_porosity : list
        Porosity time series.
    results_k : list
        Permeability time series.
    results_diffc : list
        Diffusion coefficient time series.
    if_update_porosity_k : bool
        Flag indicating whether porosity and permeability are updated.
    if_update_diffc : bool
        Flag indicating whether diffusion is updated.
    filename : str | None
        Optional base filename for results.
    Returns
    -------
    None
        Writes result files to disk.
    """
    if not headings:
        print("Error: Cannot obtain headings, result dimension unknown")
        return
    if filename is None:
        filename = os.path.join(output_dir, "results.npy")
    else:
        filename = os.path.join(output_dir, filename)
    base = os.path.splitext(filename)[0]
    np.save(filename, np.array(results))
    print(f"Results saved to: {filename}")
    header_file = base + "_headings.txt"
    with open(header_file, "w") as f:
        for heading in headings:
            f.write(f"{heading}\n")
    print(f"Headings saved to: {header_file}")
    if if_update_porosity_k:
        porosity_file = base + "_porosity.npy"
        np.save(porosity_file, np.array(results_porosity))
        print(f"Porosity results saved to: {porosity_file}")
        k_file = base + "_K.npy"
        np.save(k_file, np.array(results_k))
        print(f"K results saved to: {k_file}")
    if if_update_diffc:
        diffc_file = base + "_diffc.npy"
        np.save(diffc_file, np.array(results_diffc))
        print(f"DIFFC results saved to: {diffc_file}")
