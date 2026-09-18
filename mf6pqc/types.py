from collections.abc import Callable

import numpy as np

ArrayLike = int | float | list | tuple | np.ndarray

# The callback receives component names, a read-only (ncomp, ncell)
# concentration array, and the target time in model-time units.  It returns
# component rates in concentration per model-time unit with the same shape.
SIARateEvaluator = Callable[[tuple[str, ...], np.ndarray, float], np.ndarray]
