"""Quantitative checks for every coupling arrow in the thermal example."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import flopy
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO_DIR = ROOT.parent.parent
sys.path.insert(0, str(REPO_DIR))

from modflow_model import BASE_K, POROSITY, VISCOSITY_REFERENCE


OUTPUT = ROOT / "output"

results = np.load(OUTPUT / "results.npy")
headings = (OUTPUT / "results_headings.txt").read_text(encoding="utf-8").splitlines()
times = np.load(OUTPUT / "results_times.npy")
temperature = np.load(OUTPUT / "results_temperature.npy")
temperature_for_flow = np.load(OUTPUT / "results_temperature_for_flow.npy")
viscosity = np.load(OUTPUT / "results_viscosity.npy")
reference_k = np.load(OUTPUT / "results_reference_K.npy")
effective_k = np.load(OUTPUT / "results_effective_K.npy")
porosity = np.load(OUTPUT / "results_porosity.npy")
manifest = json.loads(
    (OUTPUT / "results_manifest.json").read_text(encoding="utf-8")
)


def field(name: str) -> np.ndarray:
    return results[:, headings.index(name), :]


assert np.all(np.diff(times) > 0.0)
assert temperature.shape == porosity.shape == reference_k.shape == effective_k.shape
assert float(np.max(temperature[-1])) > 50.0
assert float(np.ptp(temperature[-1])) > 20.0

# ThermalSNIA is explicit: the temperature computed in one frame drives flow
# in the next frame, while that newly computed temperature immediately drives
# the reaction in its own frame.
np.testing.assert_allclose(temperature_for_flow[1:], temperature[:-1], atol=1.0e-10)
arrhenius_expected = np.exp(
    -45000.0 / 8.314462618 * (1.0 / (temperature + 273.15) - 1.0 / 293.15)
)
np.testing.assert_allclose(
    field("ArrheniusFactor"), arrhenius_expected, rtol=2.0e-10, atol=1.0e-12
)

# VSC must be the sole owner of effective K.  The MODFLOW constitutive identity
# is exact for every stored flow snapshot.
np.testing.assert_allclose(
    effective_k,
    reference_k * VISCOSITY_REFERENCE / viscosity,
    rtol=2.0e-10,
    atol=1.0e-12,
)
assert float(np.ptp(viscosity[-1])) > 1.0e-4
assert float(np.ptp(effective_k[-1])) > 0.1

# Mineral dissolution changes porosity and the reference conductivity.  The K
# used by a completed flow step corresponds to the previous reaction endpoint.
assert float(np.max(np.abs(porosity[-1] - POROSITY))) > 1.0e-5
assert float(np.max(np.abs(reference_k[-1] - BASE_K))) > 1.0e-4
phi = porosity[-2]
expected_reference = BASE_K * (
    (phi**3 / (1.0 - phi) ** 2)
    / (POROSITY**3 / (1.0 - POROSITY) ** 2)
)
np.testing.assert_allclose(reference_k[-1], expected_reference, rtol=2.0e-9)

# Conservative tracer movement demonstrates GWT transport; temperature causes
# spatially different kinetic mineral loss and Product generation.
tracer = field("Tracer")
mineral = field("ThermalMineral")
product = field("Product")
assert tracer[-1, 0] > 0.9
assert np.count_nonzero(tracer[-1] > 1.0e-3) > 5
assert float(np.ptp(mineral[-1])) > 1.0e-4
assert float(np.max(product[-1])) > 1.0e-3

# A real nonzero seepage field was solved, not just transport/reaction arrays.
head_file = flopy.utils.HeadFile(ROOT / "simulation" / "gwf_model.hds")
heads = np.asarray(head_file.get_data()).ravel()
assert np.all(np.isfinite(heads))
assert heads[0] > heads[-1]
assert float(np.ptp(heads)) > 0.9

assert manifest["has_energy"]
assert manifest["has_porosity_and_k"]
assert manifest["run"]["modflow_convergence_failures"] == []

print("All GWE/VSC/reactive coupling checks passed.")
