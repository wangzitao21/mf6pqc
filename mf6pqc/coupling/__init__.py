"""Coupling-algorithm registry exposed by MF6PQC."""

from __future__ import annotations

from enum import Enum
from typing import Callable

from mf6pqc.coupling.sia import run_sia
from mf6pqc.coupling.snia import run_standard
from mf6pqc.coupling.strang import run_strang
from mf6pqc.coupling.thermal_snia import run_thermal_snia


class CouplingMethod(str, Enum):
    """Built-in operator-coupling strategies."""

    SNIA = "SNIA"
    SIA = "SIA"
    STRANG = "Strang"
    THERMAL_SNIA = "ThermalSNIA"


_RUNNERS: dict[CouplingMethod, Callable] = {
    CouplingMethod.SNIA: run_standard,
    CouplingMethod.SIA: run_sia,
    CouplingMethod.STRANG: run_strang,
    CouplingMethod.THERMAL_SNIA: run_thermal_snia,
}


def get_coupling_runner(method: CouplingMethod | str) -> tuple[CouplingMethod, Callable]:
    """Normalize a user-facing method name and return its runner."""
    if isinstance(method, CouplingMethod):
        normalized = method
    else:
        aliases = {
            "snia": CouplingMethod.SNIA,
            "standard": CouplingMethod.SNIA,
            "sia": CouplingMethod.SIA,
            "strang": CouplingMethod.STRANG,
            "strang_splitting": CouplingMethod.STRANG,
            "thermal": CouplingMethod.THERMAL_SNIA,
            "thermalsnia": CouplingMethod.THERMAL_SNIA,
            "thermal_snia": CouplingMethod.THERMAL_SNIA,
            "gwe_vsc": CouplingMethod.THERMAL_SNIA,
        }
        try:
            normalized = aliases[str(method).strip().casefold()]
        except KeyError as exc:
            choices = ", ".join(item.value for item in CouplingMethod)
            raise ValueError(
                f"Unknown coupling method {method!r}; choose one of {choices}"
            ) from exc
    return normalized, _RUNNERS[normalized]


__all__ = [
    "CouplingMethod",
    "get_coupling_runner",
    "run_standard",
    "run_sia",
    "run_strang",
    "run_thermal_snia",
]
