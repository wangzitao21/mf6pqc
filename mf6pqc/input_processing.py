import numpy as np
import numbers

from mf6pqc.constants import MODULE_INDICES, IC_DEFAULT
from mf6pqc.types import ArrayLike
from mf6pqc.utils import ensure_array


def create_ic_array_from_map(nxyz: int, ic_map: dict) -> np.ndarray:
    """
    Build the PhreeqcRM initial-condition array from a module map.
    Parameters
    ----------
    nxyz : int
        Number of computational cells.
    ic_map : dict
        Mapping from module name to initial condition index or array.
    Returns
    -------
    np.ndarray
        Packed initial condition array for PhreeqcRM modules.
    """
    ic_array = np.full((nxyz * 7,), IC_DEFAULT, dtype=np.int32)
    for module_name, pqi_value in ic_map.items():
        if module_name not in MODULE_INDICES:
            print(f"Warning: unknown chemical module name '{module_name}' will be ignored.")
            continue
        idx = MODULE_INDICES[module_name]
        start, end = idx * nxyz, (idx + 1) * nxyz
        if isinstance(pqi_value, numbers.Number):
            ic_array[start:end] = int(pqi_value)
        elif isinstance(pqi_value, (list, tuple, np.ndarray)):
            arr = np.array(pqi_value, dtype=np.int32).ravel()
            if arr.shape != (nxyz,):
                raise ValueError(f"The provided array for '{module_name}' does not match nxyz.")
            ic_array[start:end] = arr
        else:
            raise TypeError(f"Unsupported parameter type: {module_name}.")
    return ic_array


def setup_single_ic(phreeqc_rm, nxyz: int, ic_map: dict) -> None:
    """
    Initialize single initial conditions for all cells.
    Parameters
    ----------
    phreeqc_rm : object
        PhreeqcRM instance.
    nxyz : int
        Number of computational cells.
    ic_map : dict
        Mapping from module name to initial condition index or array.
    Returns
    -------
    None
        Applies initial conditions to the PhreeqcRM instance.
    """
    print("--- Setting single initial chemical condition ---")
    ic_array = create_ic_array_from_map(nxyz, ic_map)
    phreeqc_rm.InitialPhreeqc2Module(ic_array)


def setup_mixed_ic(
    phreeqc_rm,
    nxyz: int,
    ic_map1: dict,
    ic_map2: dict,
    fractions: ArrayLike,
) -> None:
    """
    Initialize mixed initial conditions for all cells.
    Parameters
    ----------
    phreeqc_rm : object
        PhreeqcRM instance.
    nxyz : int
        Number of computational cells.
    ic_map1 : dict
        Mapping for the first initial condition set.
    ic_map2 : dict
        Mapping for the second initial condition set.
    fractions : ArrayLike
        Mixing fraction per cell for ic_map1.
    Returns
    -------
    None
        Applies mixed initial conditions to the PhreeqcRM instance.
    """
    print("--- Setting mixed initial chemical condition ---")
    ic_array1 = create_ic_array_from_map(nxyz, ic_map1)
    ic_array2 = create_ic_array_from_map(nxyz, ic_map2)
    fraction_array = ensure_array(nxyz, "mixing ratio", fractions)
    fractions_tiled = np.tile(fraction_array, 7)
    phreeqc_rm.InitialPhreeqc2Module_mix(ic_array1, ic_array2, fractions_tiled)
