"""Quantitative MIN3P comparison for the public Xie B1--B4 runners.

Examples:
  python scripts/validate_xie_implicit.py --results-root tmp/xie_validation/coarse
  python scripts/validate_xie_implicit.py --results-root tmp/xie_validation/coarse --refined-root tmp/xie_validation/refined
Reference files are read only. Comparisons require exact saved physical times.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = [row for row in reader if any((v or "").strip() for v in row.values())]
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def load_output(path):
    data = dict(
        raw=np.load(path / "results.npy"),
        phi=np.load(path / "results_porosity.npy"),
        k=np.load(path / "results_K.npy") / 86400,
        times=np.load(path / "results_times.npy") / 365,
        headings=(path / "results_headings.txt").read_text().splitlines(),
        manifest=json.loads((path / "results_manifest.json").read_text()),
    )
    run = data["manifest"]["run"]
    if not run["completed"] or not run["converged"]:
        raise ValueError("Only completed and converged results can be validated")
    if (
        data["raw"].ndim != 3
        or data["phi"].shape != data["k"].shape
        or data["raw"].shape[::2] != data["phi"].shape
        or len(data["times"]) != len(data["raw"])
        or not np.all(np.isfinite(data["times"]))
        or np.any(np.diff(data["times"]) <= 0)
    ):
        raise ValueError("Invalid output dimensions or physical time axis")
    return data


def comparison_years(data, targets, allow_partial=False):
    missing = [year for year in targets if not np.any(abs(data["times"] - year) <= 1e-6)]
    if missing and not allow_partial:
        raise ValueError(
            f"Missing reference times {missing}; all three comparison times are required"
        )
    return [year for year in targets if year not in missing]


def check_refinement(data, fine):
    for key in ("database", "chemistry_input"):
        if (
            data["manifest"]["run"]["inputs"][key]["sha256"]
            != fine["manifest"]["run"]["inputs"][key]["sha256"]
        ):
            raise ValueError("Step refinement requires identical chemistry inputs")

    def declared_kinetics(output):
        # Earlier manifests recorded MIN3P floors in the parameter snapshot
        # before they were also added to the general kinetics diagnostics.
        reactions = [dict(r) for r in output["manifest"]["run"]["implicit_diagnostics"]["kinetics"]]
        benchmark = output["manifest"].get("min3p_benchmark", {})
        entries = {m["name"]: m for m in benchmark.get("minerals", [])}
        for reaction in reactions:
            if "saturation_index_heading" not in reaction:
                logarithmic = output["manifest"]["run"]["implicit_diagnostics"][
                    "logarithmic_saturation"
                ]
                heading = "SI_" + reaction["name"] if logarithmic else None
                if heading is not None and heading not in output["headings"]:
                    raise ValueError("Refinement manifest lacks the logarithmic saturation heading")
                reaction["saturation_index_heading"] = heading
            if "minimum_amount_mol_bulk" not in reaction:
                if reaction["name"] not in entries:
                    raise ValueError("Refinement manifest does not record the minimum inventory")
                value = entries[reaction["name"]]["minimum_amount_mol_bulk"]
                minimum = np.full(output["phi"].shape[1], value)
                if benchmark["level"] == 4:
                    minimum[[0, -1]] = 0
                reaction["minimum_amount_mol_bulk"] = minimum.tolist()
        return reactions

    first, second = declared_kinetics(data), declared_kinetics(fine)
    if len(first) != len(second):
        raise ValueError("Step refinement requires identical kinetic declarations")
    for a, b in zip(first, second, strict=True):
        if set(a) != set(b):
            raise ValueError("Step refinement requires identical kinetic declarations")
        for key, value in a.items():
            if isinstance(value, (float, int, list)):
                equal = np.allclose(value, b[key], rtol=1e-12, atol=0)
            else:
                equal = value == b[key]
            if not equal:
                raise ValueError(f"Step refinement changed {a['name']} {key}")
    missing = []
    for key in ("predict_porosity", "porosity_coupling"):
        a = data["manifest"]["run"]["implicit_diagnostics"]["tolerances"]
        b = fine["manifest"]["run"]["implicit_diagnostics"]["tolerances"]
        if key not in a or key not in b:
            missing.append(key)
        elif a[key] != b[key]:
            raise ValueError("Step refinement requires identical property coupling")
    return missing


def index(data, year):
    i = int(np.argmin(abs(data["times"] - year)))
    if abs(data["times"][i] - year) > 1e-6:
        raise ValueError(f"Missing exact saved output at {year} years")
    return i


def difference(actual, expected):
    delta = np.asarray(actual) - np.asarray(expected)
    if not np.all(np.isfinite(delta)):
        raise ValueError("Nonfinite benchmark values")
    return dict(rmse=float(np.sqrt(np.mean(delta**2))), max_abs=float(np.max(abs(delta))))


def compare_case(level, root, refined_root=None, *, allow_partial=False):
    if level == 1:
        return compare_b1(root, refined_root, allow_partial=allow_partial)
    name = f"ex{level + 13:03d}_Xie2015_B{level}"
    inp = ROOT / "examples" / name / "input_data"
    path = root / name / "output"
    data = load_output(path)
    parameters = data["manifest"]["min3p_benchmark"]
    hydro = table(inp / ("MIN3P_hydro.csv" if level == 3 else "MIN3P_results.csv"))
    minerals = hydro if level == 2 else table(inp / "MIN3P_minerals.csv")
    fine = load_output(refined_root / name / "output") if refined_root else None
    missing_metadata = check_refinement(data, fine) if fine is not None else []
    years = comparison_years(data, parameters["reference_times_years"], allow_partial)
    refined_years = comparison_years(fine, years, allow_partial) if fine is not None else []
    from flopy.utils import HeadFile

    head_file = (
        HeadFile(root / name / "simulation/gwf_model.hds", precision="double")
        if level != 4
        else None
    )
    run = data["manifest"]["run"]
    diagnostics = {k: v for k, v in run["implicit_diagnostics"].items() if k != "kinetics"}
    report = dict(
        completed=run["completed"],
        results_directory=str(path.resolve()),
        wall_seconds=run["wall_time_seconds"],
        steps=run["logical_steps"],
        diagnostics=diagnostics,
        input_hashes=run["inputs"],
        parameters=parameters,
        profiles={},
        comparison_times_years=years,
        all_reference_times_present=len(years) == 3,
        refinement_metadata_missing=missing_metadata,
        max_step_years=parameters["max_step_years"],
    )
    if fine is not None:
        report["refinement_run"] = dict(
            max_step_years=fine["manifest"]["min3p_benchmark"]["max_step_years"],
            comparison_times_years=refined_years,
            all_reference_times_present=len(refined_years) == 3,
            steps=fine["manifest"]["run"]["logical_steps"],
            wall_seconds=fine["manifest"]["run"]["wall_time_seconds"],
        )
    for year in years:
        i = index(data, year)
        rows = {"porosity": difference(data["phi"][i], hydro[f"Porosity {year}years"])}
        if level == 4:
            pore = np.load(path / "results_diffc.npy")
            # Manifest uses completed_steps: no initial frame in diffusion output.
            de = data["phi"][i] * pore[i - 1] / 86400
            rows["effective_diffusion_m2_s"] = difference(
                de, hydro[f"Diffusion coefficient {year}years"]
            )
        else:
            rows["conductivity_m_s"] = difference(
                data["k"][i], hydro[f"Hydraulic conductivity {year}years"]
            )
            times = np.array(head_file.get_times())
            t = float(times[np.argmin(abs(times - year * 365))])
            if abs(t - year * 365) > 1e-5:
                raise ValueError("Missing head at requested time")
            h = head_file.get_data(totim=t).ravel()
            rows["head_m"] = difference(h, hydro[f"Hydraulic head {year}years"])
        for entry in parameters["minerals"]:
            mineral = entry["name"]
            values = (
                data["raw"][i, data["headings"].index(mineral)] * entry["molar_volume_l_per_mol"]
            )
            if level == 4:
                values = values[1:-1]
            rows[mineral + "_volume_fraction"] = difference(
                values, minerals[f"{mineral} {year}years"]
            )
        if fine is not None and year in refined_years:
            j = index(fine, year)
            refined = {"porosity": difference(data["phi"][i], fine["phi"][j])}
            if level == 4:
                fine_pore = np.load(refined_root / name / "output/results_diffc.npy")
                refined["effective_diffusion_m2_s"] = difference(
                    de, fine["phi"][j] * fine_pore[j - 1] / 86400
                )
            else:
                refined["conductivity_m_s"] = difference(data["k"][i], fine["k"][j])
                with HeadFile(
                    refined_root / name / "simulation/gwf_model.hds", precision="double"
                ) as fine_head:
                    refined["head_m"] = difference(h, fine_head.get_data(totim=year * 365).ravel())
            for entry in parameters["minerals"]:
                mineral = entry["name"]
                refined[mineral + "_volume_fraction"] = difference(
                    data["raw"][i, data["headings"].index(mineral)]
                    * entry["molar_volume_l_per_mol"],
                    fine["raw"][j, fine["headings"].index(mineral)]
                    * entry["molar_volume_l_per_mol"],
                )
            rows["step_refinement"] = refined
        report["profiles"][str(year)] = rows
    if head_file is not None:
        head_file.close()
    return report


def compare_b1(root, refined_root=None, *, allow_partial=False):
    import flopy

    name = "ex014_Xie2015_B1"
    data = load_output(root / name / "output")
    fine = load_output(refined_root / name / "output") if refined_root else None
    missing_metadata = check_refinement(data, fine) if fine is not None else []
    years = comparison_years(data, [10, 100, 120], allow_partial)
    reference = table(ROOT / "examples" / name / "input_data/MIN3P_results.csv")
    model = flopy.mf6.MFSimulation.load(sim_ws=root / name / "simulation", verbosity_level=0)
    gwf = model.get_model("gwf_model")
    widths = gwf.dis.delr.array.ravel()
    x = np.cumsum(widths) - widths / 2
    xr = np.linspace(0, 2, len(reference["Porosity 10years"]))
    heads = [
        float(gwf.get_package(p).stress_period_data.get_data(0)["bhead"][0])
        for p in ("bushui", "ghb_right")
    ]
    run = data["manifest"]["run"]
    report = dict(
        completed=True,
        results_directory=str((root / name / "output").resolve()),
        steps=run["logical_steps"],
        wall_seconds=run["wall_time_seconds"],
        diagnostics={k: v for k, v in run["implicit_diagnostics"].items() if k != "kinetics"},
        input_hashes=run["inputs"],
        comparison_times_years=years,
        all_reference_times_present=len(years) == 3,
        refinement_metadata_missing=missing_metadata,
        max_step_years=data["manifest"]["b1_implicit"]["max_step_years"],
        profiles={},
    )
    vm = 100.0894 / 2710
    with flopy.utils.HeadFile(root / name / "simulation/gwf_model.hds", precision="double") as hf:
        for year in years:
            i = index(data, year)
            h = hf.get_data(totim=year * 365).ravel()
            m = data["raw"][i, data["headings"].index("Calcite")] * vm
            rows = dict(
                porosity=difference(
                    np.interp(xr, x, data["phi"][i]), reference[f"Porosity {year}years"]
                ),
                Calcite_volume_fraction=difference(
                    np.interp(xr, x, m), reference[f"Calcite {year}years"]
                ),
                head_m=difference(
                    np.interp(xr, np.r_[0, x, 2], np.r_[heads[0], h, heads[1]]),
                    reference[f"Hydraulic head {year}years"],
                ),
            )
            if fine is not None:
                j = index(fine, year)
                rows["step_refinement"] = dict(
                    porosity=difference(data["phi"][i], fine["phi"][j]),
                    Calcite_volume_fraction=difference(
                        m, fine["raw"][j, fine["headings"].index("Calcite")] * vm
                    ),
                    conductivity_m_s=difference(data["k"][i], fine["k"][j]),
                )
            report["profiles"][str(year)] = rows
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--refined-root", type=Path)
    parser.add_argument("--cases", type=int, nargs="+", default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Explicitly allow short diagnostic runs with missing reference times",
    )
    args = parser.parse_args()
    report = {
        f"B{level}": compare_case(
            level, args.results_root, args.refined_root, allow_partial=args.allow_partial
        )
        for level in args.cases
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, value in report.items():
        print(name, "steps", value["steps"], "seconds", round(value["wall_seconds"], 3))
        for year, profiles in value["profiles"].items():
            print(
                year,
                "porosity",
                profiles["porosity"],
                "refinement",
                profiles.get("step_refinement", {}).get("porosity"),
            )


if __name__ == "__main__":
    main()
