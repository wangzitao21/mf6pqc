"""Quantitative checks for every coupling arrow in the thermal example."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import _example_support as _example_support
import flopy
import numpy as np
from _example_support import runtime_path
from modflow_model import BASE_K, POROSITY, VISCOSITY_REFERENCE


def main() -> None:

    OUTPUT = runtime_path(__file__, "output")

    results = np.load(OUTPUT / "results.npy")

    headings = (OUTPUT / "results_headings.txt").read_text(encoding="utf-8").splitlines()

    times = np.load(OUTPUT / "results_times.npy")

    temperature = np.load(OUTPUT / "results_temperature.npy")

    temperature_for_flow = np.load(OUTPUT / "results_temperature_for_flow.npy")

    viscosity = np.load(OUTPUT / "results_viscosity.npy")

    reference_k = np.load(OUTPUT / "results_reference_K.npy")

    effective_k = np.load(OUTPUT / "results_effective_K.npy")

    porosity = np.load(OUTPUT / "results_porosity.npy")

    manifest = json.loads((OUTPUT / "results_manifest.json").read_text(encoding="utf-8"))

    def field(name: str) -> np.ndarray:
        return results[:, headings.index(name), :]

    assert np.all(np.diff(times) > 0.0)

    assert temperature.shape == porosity.shape == reference_k.shape == effective_k.shape

    assert float(np.max(temperature[-1])) > 50.0

    assert float(np.ptp(temperature[-1])) > 20.0

    np.testing.assert_allclose(temperature_for_flow[1:], temperature[:-1], atol=1.0e-10)

    arrhenius_expected = np.exp(
        -45000.0 / 8.314462618 * (1.0 / (temperature + 273.15) - 1.0 / 293.15)
    )

    np.testing.assert_allclose(
        field("ArrheniusFactor"), arrhenius_expected, rtol=2.0e-10, atol=1.0e-12
    )

    np.testing.assert_allclose(
        effective_k,
        reference_k * VISCOSITY_REFERENCE / viscosity,
        rtol=2.0e-10,
        atol=1.0e-12,
    )

    assert float(np.ptp(viscosity[-1])) > 1.0e-4

    assert float(np.ptp(effective_k[-1])) > 0.1

    assert float(np.max(np.abs(porosity[-1] - POROSITY))) > 1.0e-5

    assert float(np.max(np.abs(reference_k[-1] - BASE_K))) > 1.0e-4

    phi = porosity[-2]

    expected_reference = BASE_K * (
        (phi**3 / (1.0 - phi) ** 2) / (POROSITY**3 / (1.0 - POROSITY) ** 2)
    )

    np.testing.assert_allclose(reference_k[-1], expected_reference, rtol=2.0e-9)

    tracer = field("Tracer")

    mineral = field("ThermalMineral")

    product = field("Product")

    assert tracer[-1, 0] > 0.9

    assert np.count_nonzero(tracer[-1] > 1.0e-3) > 5

    assert float(np.ptp(mineral[-1])) > 1.0e-4

    assert float(np.max(product[-1])) > 1.0e-3

    head_file = flopy.utils.HeadFile(runtime_path(__file__, "simulation") / "gwf_model.hds")

    heads = np.asarray(head_file.get_data()).ravel()

    assert np.all(np.isfinite(heads))

    assert heads[0] > heads[-1]

    assert float(np.ptp(heads)) > 0.9

    assert manifest["has_energy"]

    assert manifest["has_porosity_and_k"]

    assert manifest["run"]["modflow_convergence_failures"] == []

    print("All GWE/VSC/reactive coupling checks passed.")


if __name__ == "__main__":
    _example_support.configure_logging()
    main()
