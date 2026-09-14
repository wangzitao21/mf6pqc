from __future__ import annotations

import contextlib
from multiprocessing.shared_memory import SharedMemory

import numpy as np


class CellExchange:
    __slots__ = (
        "memory",
        "owner",
        "count",
        "ncomps",
        "noutputs",
        "concentrations",
        "selected",
        "temperature",
        "porosity",
    )

    def __init__(self, count: int, ncomps: int, noutputs: int, *, name: str | None = None):
        self.count = count
        self.ncomps = ncomps
        self.noutputs = noutputs
        self.owner = name is None
        self.memory = SharedMemory(
            name=name, create=self.owner, size=(ncomps + noutputs + 2) * count * 8
        )
        values = np.ndarray(
            (ncomps + noutputs + 2, count), dtype=np.float64, buffer=self.memory.buf
        )
        self.concentrations = values[:ncomps]
        self.selected = values[ncomps : ncomps + noutputs]
        self.temperature = values[-2]
        self.porosity = values[-1]

    @property
    def descriptor(self) -> tuple[str, int, int, int]:
        return self.memory.name, self.count, self.ncomps, self.noutputs

    @classmethod
    def attach(cls, descriptor) -> CellExchange:
        name, count, ncomps, noutputs = descriptor
        return cls(count, ncomps, noutputs, name=name)

    def close(self) -> None:
        if self.memory is None:
            return
        memory, self.memory = self.memory, None
        self.concentrations = self.selected = self.temperature = self.porosity = None
        memory.close()
        if self.owner:
            with contextlib.suppress(FileNotFoundError):
                memory.unlink()
