"""Approximate aqueous thermodynamic tangents from PHREEQC species inventories."""

from __future__ import annotations

import numpy as np

from mf6pqc.exceptions import CouplingError


class SpeciationTangent:
    """Fixed-activity-coefficient tangent; full PHREEQC residuals verify every step.

    For species composition B and concentrations s, dC = B diag(s) B.T dmu.
    Use an SVD of B sqrt(s), avoiding the squared conditioning of normal equations.
    This resolves trace redox pools without finite differences of large H/O totals.
    """

    def __init__(self, rm, components, mineral_stoichiometry):
        rm.SetSpeciesSaveOn(True)
        rm.FindComponents()
        self.rm = rm
        self.names = rm.GetSpeciesNames()
        composition = rm.GetSpeciesStoichiometry()
        self.b = np.array([[composition[s].get(c, 0) for s in self.names] for c in components])
        self.nu = np.asarray(mineral_stoichiometry)

    def derivative(self, directions, ncell, cells=None):
        species = np.asarray(self.rm.GetSpeciesConcentrations()).reshape(len(self.names), ncell)
        if cells is not None:
            species = species[:, cells]
        weighted = self.b[None, :, :] * np.sqrt(np.maximum(species.T[:, None, :], 0))
        u, singular, _ = np.linalg.svd(weighted, full_matrices=False)
        threshold = singular[:, :1] * 1e-12
        inverse = np.divide(1.0, singular, out=np.zeros_like(singular), where=singular > threshold)
        left = np.einsum("ir,nik->nrk", self.nu, u) * inverse[:, None, :]
        directions = np.array([d for _, _, d in directions]).T
        right = np.einsum("id,nik->ndk", directions, u) * inverse[:, None, :]
        return np.einsum("nrk,ndk->rdn", left, right)


class CellSpeciation:
    """Reusable one-cell PHREEQC kernel for block solves.

    The public solver permits only aqueous speciation and externally integrated
    kinetics here. No immobile equilibria are copied or silently omitted.
    """

    def __init__(self, sim, nu, rows):
        from mf6pqc.backends import CheckedPhreeqcRM

        self.sim, self.rows = sim, rows
        rm = self.rm = CheckedPhreeqcRM(sim.backend_factory.create_phreeqcrm(1, 1))
        rm.SetUnitsSolution(2)
        rm.SetComponentH2O(False)
        rm.UseSolutionDensityVolume(False)
        rm.SetRebalanceFraction(0)
        rm.SetScreenOn(False)
        rm.SetPrintChemistryOn(False, False, False)
        rm.SetSelectedOutputOn(True)
        rm.SetPorosity(sim.porosity[:1])
        rm.SetSaturation(np.ones(1))
        rm.SetDensityUser(np.ones(1))
        rm.LoadDatabase(str(sim.db_path))
        rm.RunFile(True, True, True, str(sim.pqi_path))
        rm.RunString(True, False, False, "DELETE; -all\nEND\n")
        # Create our own aqueous template. Do not assume the user's input
        # happens to contain SOLUTION 0 (or a specific initial-condition map).
        template = ["SOLUTION 0", "units mol/L", "pH 7"]
        template += [f"{name} 1e-20" for name in sim.components if name not in {"H", "O", "Charge"}]
        rm.RunString(False, True, False, "\n".join(template) + "\nEND\n")
        rm.FindComponents()
        if list(rm.GetComponents()) != sim.components:
            raise CouplingError("Cell speciation component order differs from the main backend")
        rm.InitialPhreeqc2Module(np.array([0, -1, -1, -1, -1, -1, -1], dtype=np.int32))
        rm.SetTimeStep(0)
        self.tangent = SpeciationTangent(rm, sim.components, nu)
        self.last = None

    def evaluate(self, c, cell, phi, time_days):
        key = (cell, float(phi), float(time_days), c.tobytes())
        if key != self.last:
            rm, sim = self.rm, self.sim
            rm.SetPorosity(np.array([phi]))
            rm.SetTemperature(sim.temperature[cell : cell + 1])
            rm.SetPressure(sim.pressure[cell : cell + 1])
            rm.SetDensityUser(sim.density[cell : cell + 1])
            rm.SetTime(time_days * 86400)
            rm.SetConcentrations(c.ravel())
            rm.RunCells()
            sim.implicit_diagnostics["chemistry_calls"] += 1
            back = np.asarray(rm.GetConcentrations())
            drift = float(np.max(abs(back - c.ravel())))
            sim.implicit_diagnostics["max_speciation_drift"] = max(
                sim.implicit_diagnostics["max_speciation_drift"], drift
            )
            if drift > sim.config.implicit.concentration_tolerance:
                raise CouplingError("Cell speciation changed a transported component inventory")
            self.values = np.asarray(rm.GetSelectedOutput())[self.rows, None] * np.log(10)
            self.last = key
        return self.values.copy()

    def derivative(self, directions):
        return self.tangent.derivative(directions, 1)

    def close(self):
        self.rm.MpiWorkerBreak()
