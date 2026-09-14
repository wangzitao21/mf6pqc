import pandas as pd
from IPython.display import display

comparison_rows = []
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.dont_write_bytecode = True
CASE_DIR = next(
    p
    for root in (Path.cwd(), *Path.cwd().parents)
    for p in (root, root / "examples/ex016_Xie2015_B3")
    if p.name == "ex016_Xie2015_B3" and (p / "run.py").is_file()
)
sys.path.insert(0, str(CASE_DIR.parent))
from example_utils import load_results, read_headings, runtime_path

CASE_FILE = CASE_DIR / "run.py"
INPUT_DIR = CASE_DIR / "input_data"
OUTPUT_DIR = runtime_path(CASE_FILE, "output")
SIMULATION_DIR = runtime_path(CASE_FILE, "simulation")


def load_case_results(output_dir):
    if (output_dir / "results_times.npy").is_file():
        return load_results(output_dir)
    values = np.load(output_dir / "results.npy")
    headings = read_headings(output_dir)
    text = next(SIMULATION_DIR.glob("*.tdis")).read_text().lower()
    periods = np.array(
        [
            [float(v) for v in row.split()]
            for row in text.split("begin perioddata")[1].split("end perioddata")[0].splitlines()
            if row.strip()
        ]
    )
    assert np.all(periods[:, 2] == 1), "Legacy snapshots require uniform steps within each period"
    steps = periods[:, 1].astype(int)
    stride, remainder = divmod(int(steps.sum()), len(values) - 1)
    assert not remainder and stride > 0, "Legacy snapshot count does not match the TDIS schedule"
    times = np.r_[0, np.cumsum(np.repeat(periods[:, 0] / steps, steps))[stride - 1 :: stride]]
    return (values, headings, times)


import flopy
from matplotlib.lines import Line2D

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "mathtext.fontset": "stix",
    }
)
PROFILE_YEARS = (10, 100)
LENGTH = 2.0
EXPECTED_NSTP = 146000
EXPECTED_DT_DAYS = 0.25
SECONDS_PER_DAY = 86400.0
MINERAL_MOLAR_VOLUMES = {
    "Calcite": 0.03693,
    "Gypsum": 0.07421,
    "Ferrihydrite": 0.02399,
    "Gibbsite": 0.03319,
    "Siderite": 0.02926,
    "Jarosite": 0.15463,
}
required = [
    OUTPUT_DIR / name
    for name in ("results.npy", "results_porosity.npy", "results_K.npy", "results_headings.txt")
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError(
        "Run python examples/ex016_Xie2015_B3/run.py before plotting. Missing: "
        + ", ".join(missing)
    )
head_path = SIMULATION_DIR / "gwf_model.hds"
results, headings, result_times = load_case_results(OUTPUT_DIR)
porosity = np.load(OUTPUT_DIR / "results_porosity.npy")
permeability = np.load(OUTPUT_DIR / "results_K.npy")
reference_hydro = pd.read_csv(INPUT_DIR / "MIN3P_hydro.csv")
reference_minerals = pd.read_csv(INPUT_DIR / "MIN3P_minerals.csv")
sim = flopy.mf6.MFSimulation.load(sim_ws=str(SIMULATION_DIR), sim_name="model", verbosity_level=0)
gwf = sim.get_model("gwf_model")
delr = gwf.dis.delr.array.ravel()
x_mf6 = np.cumsum(delr) - 0.5 * delr
x_min3p = np.linspace(0.0, LENGTH, len(reference_hydro))
perioddata = sim.tdis.perioddata.get_data()
if sum(int(record[1]) for record in perioddata) != EXPECTED_NSTP:
    raise ValueError("Unexpected number of B3 time steps")
if any(
    not np.isclose(record[0] / record[1], EXPECTED_DT_DAYS) or not np.isclose(record[2], 1.0)
    for record in perioddata
):
    raise ValueError("Unexpected B3 time-step schedule")
years = result_times / 365.0
if results.shape[0] != porosity.shape[0] or permeability.shape[0] != porosity.shape[0]:
    raise ValueError("Saved B3 arrays have incompatible time dimensions")
if years[-1] < max(PROFILE_YEARS):
    raise ValueError("Saved B3 outputs do not reach every requested profile time.")


def state_index(year):
    index = int(np.abs(years - year).argmin())
    if not np.isclose(years[index], year):
        raise ValueError(f"No saved result at {year} years.")
    return index


def mineral_profile(year, mineral):
    return results[state_index(year), headings.index(mineral)] * MINERAL_MOLAR_VOLUMES[mineral]


head_file = flopy.utils.HeadFile(str(head_path))
head_times = np.asarray(head_file.get_times()) / 365.0
head_kstpkper = head_file.get_kstpkper()


def head_profile(year):
    index = int(np.abs(head_times - year).argmin())
    if not np.isclose(head_times[index], year):
        raise ValueError(f"No head output at {year} years.")
    return head_file.get_data(kstpkper=head_kstpkper[index])[0, 0]


styles = {10: "-", 100: "--"}


def resample(values):
    return np.interp(x_min3p, x_mf6, np.asarray(values))


fig, combined_axes = plt.subplots(5, 3, figsize=(13.8, 17.0), constrained_layout=True)
axes = np.array(combined_axes.flat[:4]).reshape(2, 2)
panels = [
    (
        axes[0, 0],
        "Porosity",
        lambda year: porosity[state_index(year)],
        "Porosity [-]",
        (0.001, 1.0),
        True,
    ),
    (axes[0, 1], "Hydraulic head", head_profile, "Hydraulic head [m]", (0.0, 0.008), False),
    (
        axes[1, 0],
        "Hydraulic conductivity",
        lambda year: permeability[state_index(year)] / SECONDS_PER_DAY,
        "Hydraulic conductivity [m s$^{-1}$]",
        (1e-12, 0.01),
        True,
    ),
]
for ax, key, getter, ylabel, ylim, log_scale in panels:
    for year, style in styles.items():
        error = resample(getter(year)) - reference_hydro[f"{key} {year}years"]
        comparison_rows.append(
            {"Variable": key, "Time (yr)": year, "RMSE": np.sqrt(np.mean(error**2))}
        )
        ax.plot(x_min3p, reference_hydro[f"{key} {year}years"], color="tab:green", ls=style, lw=1.7)
        ax.plot(
            x_min3p,
            resample(getter(year)),
            color="tab:red",
            ls=style,
            lw=1.1,
            marker="o",
            ms=2.2,
            markevery=4,
            zorder=3,
        )
    ax.set(xlim=(0, 1.0), ylim=ylim, xlabel="Distance [m]", ylabel=ylabel)
    if log_scale:
        ax.set_yscale("log")
right_ghb = gwf.get_package("ghb_right")
right_bhead = float(right_ghb.stress_period_data.get_data(key=0)["bhead"][0])
right_distance = 0.5 * delr[-1]
right_heads = np.empty_like(years)
right_heads[0] = 0.0
right_heads[1:] = [head_profile(year)[-1] for year in years[1:]]
outflow = permeability[:, -1] * (right_heads - right_bhead) / right_distance
outflow[0] = permeability[0, -1] * 0.007 / LENGTH
ax = axes[1, 1]
ax.plot(years, np.maximum(outflow, 1e-12), color="tab:red", lw=1.5)
for year, style in styles.items():
    ax.axvline(year, color="0.55", ls=style, lw=0.9)
ax.set(
    xlim=(0, 300),
    ylim=(1e-08, 0.1),
    xlabel="Time [years]",
    ylabel="Flux of outflow [m$^3$ d$^{-1}$]",
)
ax.set_yscale("log")
for ax, label in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)"), strict=False):
    ax.text(0.04, 0.95, label, transform=ax.transAxes, va="top", fontsize=14, fontweight="bold")
    ax.tick_params(top=True, right=True)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
legend = [
    item
    for year, style in styles.items()
    for item in (
        Line2D([], [], color="tab:green", ls=style, lw=1.7, label=f"MIN3P kinetic {year} years"),
        Line2D(
            [],
            [],
            color="tab:red",
            ls=style,
            marker="o",
            lw=1.1,
            ms=3,
            label=f"MF6PQC kinetic {year} years",
        ),
    )
]
axes[0, 1].legend(handles=legend, frameon=False, fontsize=8, loc="upper right")

mineral_panels = [
    ("Calcite", (0.0, 0.25), (0.0, 1.0)),
    ("Gypsum", (0.0, 0.5), (0.0, 1.0)),
    ("Ferrihydrite", (0.0, 0.25), (0.0, 1.0)),
    ("Gibbsite", (0.0, 0.25), (0.0, 1.0)),
    ("Siderite", (0.0, 0.25), (0.0, 1.0)),
    ("Jarosite", (0.0, 0.06), (0.0, 0.1)),
]
axes = np.array(combined_axes.flat[4:10]).reshape(3, 2)
for ax, (mineral, ylim, xlim), label in zip(
    axes.flat, mineral_panels, ("(e)", "(f)", "(g)", "(h)", "(i)", "(j)"), strict=False
):
    for year, style in styles.items():
        error = (
            resample(mineral_profile(year, mineral)) - reference_minerals[f"{mineral} {year}years"]
        )
        comparison_rows.append(
            {"Variable": mineral, "Time (yr)": year, "RMSE": np.sqrt(np.mean(error**2))}
        )
        ax.plot(
            x_min3p,
            reference_minerals[f"{mineral} {year}years"],
            color="tab:green",
            ls=style,
            lw=1.7,
        )
        ax.plot(
            x_min3p,
            resample(mineral_profile(year, mineral)),
            color="tab:red",
            ls=style,
            lw=1.1,
            marker="o",
            ms=2.2,
            markevery=4,
            zorder=3,
        )
    ax.set(
        xlim=xlim,
        ylim=ylim,
        xlabel="Distance [m]",
        ylabel=f"Volume fraction of {mineral.lower()} [m$^3$ m$^{-3}$]",
    )
    ax.text(0.04, 0.95, label, transform=ax.transAxes, va="top", fontsize=14, fontweight="bold")
    ax.tick_params(top=True, right=True)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
axes[0, 1].legend(handles=legend, frameon=False, fontsize=8, loc="upper right")

for ax in combined_axes.flat[10:]:
    ax.set_axis_off()
plt.show()
comparison = pd.DataFrame(comparison_rows)
comparison = comparison.pivot(index="Variable", columns="Time (yr)", values="RMSE").reindex(
    comparison["Variable"].drop_duplicates()
)
display(comparison.style.format("{:.6g}").format_index("{:.6g}", axis=1).set_uuid("ex016_1"))