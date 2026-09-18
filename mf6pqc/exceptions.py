"""MF6PQC-specific exceptions.

The exception hierarchy lets applications distinguish configuration,
backend, and numerical-convergence failures without parsing console text.
"""


class MF6PQCError(Exception):
    """Base class for errors raised deliberately by MF6PQC."""


class ConfigurationError(MF6PQCError, ValueError):
    """A simulation configuration is incomplete or physically invalid."""


class BackendError(MF6PQCError, RuntimeError):
    """A MODFLOW 6 or PhreeqcRM backend could not be initialized or queried."""


class CouplingError(MF6PQCError, RuntimeError):
    """The coupling algorithm cannot advance consistently."""


class ConvergenceError(CouplingError):
    """A nonlinear or linear solve failed under strict convergence policy."""


class PropertyUpdateError(CouplingError):
    """A constitutive model produced an invalid medium property."""
