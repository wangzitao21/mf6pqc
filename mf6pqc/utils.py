import hashlib
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
    if isinstance(nxyz, bool) or not isinstance(nxyz, numbers.Integral) or nxyz <= 0:
        raise ValueError("nxyz must be a positive integer")
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
    if nxyz <= 0:
        raise ValueError("nxyz must be positive")
    if isinstance(ispecies, bool) or not isinstance(ispecies, numbers.Integral):
        raise TypeError("ispecies must be an integer")
    if ispecies < 0:
        raise ValueError("ispecies must be nonnegative")
    start = int(ispecies) * nxyz
    return slice(start, start + nxyz)


def get_gwt_model_name(species_name: str) -> str:
    """Return a deterministic MODFLOW 6-safe GWT model name.

    Existing short component names keep the historical
    ``gwt_<component>_model`` form. MODFLOW 6 limits model names to 16
    characters, so longer component names use a compact form. A short digest
    prevents collisions when very long names share the same prefix.
    """
    legacy_name = f"gwt_{species_name}_model"
    if len(legacy_name) <= 16:
        return legacy_name
    compact_name = f"gwt_{species_name}"
    if len(compact_name) <= 16:
        return compact_name
    digest = hashlib.sha1(species_name.encode("utf-8")).hexdigest()[:4]
    return f"gwt_{species_name[:7]}_{digest}"
