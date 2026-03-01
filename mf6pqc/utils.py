import numbers
import numpy as np

from mf6pqc.types import ArrayLike


def ensure_array(nxyz: int, name: str, value: ArrayLike) -> np.ndarray:
    """
    Convert scalar or array-like input to a 1D array with length nxyz.
    Parameters
    ----------
    nxyz : int
        Number of computational cells.
    name : str
        Parameter name for error messages.
    value : ArrayLike
        Scalar or array input representing a cell-wise property.
    Returns
    -------
    np.ndarray
        Flattened array of length nxyz representing a physical field.
    """
    if isinstance(value, numbers.Number):
        return np.full((nxyz,), float(value))
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.array(value, dtype=float).ravel()
        if arr.shape != (nxyz,):
            raise ValueError(
                f"Parameter '{name}' length {arr.shape[0]} does not match nxyz ({nxyz})."
            )
        return arr
    raise TypeError(f"Unsupported type for {name}: {type(value).__name__}")


def get_species_slice(nxyz: int, ispecies: int) -> slice:
    """
    Get the slice for a species in a flattened concentration vector.
    Parameters
    ----------
    nxyz : int
        Number of computational cells.
    ispecies : int
        Index of the species in the component list.
    Returns
    -------
    slice
        Slice that targets the species block in a 1D vector.
    """
    start = ispecies * nxyz
    return slice(start, start + nxyz)
