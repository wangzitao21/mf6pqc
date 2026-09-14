import pandas as pd
from IPython.display import display

comparison_rows = []
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.dont_write_bytecode = True
CASE_DIR = next(
    p
    for root in (Path.cwd(), *Path.cwd().parents)
    for p in (root, root / "examples/ex017_Xie2015_B4")
    if p.name == "ex017_Xie2015_B4" and (p / "run.py").is_file()
)
sys.path.insert(0, str(CASE_DIR.parent))
from example_utils import load_results, read_headings, runtime_path

CASE_FILE = CASE_DIR / "run.py"
INPUT_DIR = CASE_DIR / "input_data"
OUTPUT_DIR = runtime_path(CASE_FILE, "output", override=os.environ.get("MF6PQC_B4_OUTPUT_DIR"))
SIMULATION_DIR = runtime_path(
    CASE_FILE, "simulation", override=os.environ.get("MF6PQC_B4_WORKSPACE")
)


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
PROFILE_YEARS = (100, 1000, 3000)
LENGTH = 2.0
SECONDS_PER_DAY = 86400.0
MINERAL_MOLAR_VOLUMES = {
    "Calcite": 0.03693,
    "Gypsum": 0.07421,
    "Ferrihydrite": 0.02399,
    "Jarosite": 0.15463,
    "Gibbsite": 0.03319,
    "Siderite": 0.02926,
}
required = [
    OUTPUT_DIR / name
    for name in (
        "results.npy",
        "results_porosity.npy",
        "results_K.npy",
        "results_diffc.npy",
        "results_headings.txt",
    )
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError(
        "Run python examples/ex017_Xie2015_B4/run.py before plotting. Missing: "
        + ", ".join(missing)
    )
results, headings, result_times = load_case_results(OUTPUT_DIR)
porosity = np.load(OUTPUT_DIR / "results_porosity.npy")
permeability = np.load(OUTPUT_DIR / "results_K.npy")
pore_diffusion = np.load(OUTPUT_DIR / "results_diffc.npy")
reference_hydro = pd.read_csv(INPUT_DIR / "MIN3P_results.csv")
reference_minerals = (
    pd.read_csv(INPUT_DIR / "MIN3P_minerals.csv").dropna(how="all").reset_index(drop=True)
)
sim = flopy.mf6.MFSimulation.load(sim_ws=str(SIMULATION_DIR), sim_name="model", verbosity_level=0)
gwf = sim.get_model("gwf_model")
delr = gwf.dis.delr.array.ravel()
x_mf6 = np.linspace(0.0, LENGTH, len(delr))
x_reference_hydro = np.linspace(0.0, LENGTH, len(reference_hydro))
dx = float(delr[1])
x_reference_minerals = dx * np.arange(1, len(reference_minerals) + 1)
if len(reference_minerals) != len(x_mf6) - 2:
    raise ValueError("Expected 79 interior MIN3P mineral profiles for the 81-node grid.")
state_years = result_times / 365.0
completed_years = state_years[1:]
if results.shape[0] != porosity.shape[0] or permeability.shape[0] != porosity.shape[0]:
    raise ValueError("Saved B4 arrays have incompatible time dimensions")
if len(completed_years) != pore_diffusion.shape[0]:
    raise ValueError("Diffusion snapshots must match completed output times")


def state_index(year):
    index = int(np.abs(state_years - year).argmin())
    if not np.isclose(state_years[index], year):
        raise ValueError(f"No saved chemistry/porosity state at {year} years.")
    return index


def diffusion_index(year):
    index = int(np.abs(completed_years - year).argmin())
    if not np.isclose(completed_years[index], year):
        raise ValueError(f"No saved diffusion state at {year} years.")
    return index


def mineral_profile(year, mineral):
    moles_per_litre_bulk = results[state_index(year), headings.index(mineral)]
    return moles_per_litre_bulk * MINERAL_MOLAR_VOLUMES[mineral]


def effective_diffusion(year):
    return porosity[state_index(year)] * pore_diffusion[diffusion_index(year)] / SECONDS_PER_DAY


styles = {100: "-", 1000: "--", 3000: ":"}
reference_color = "tab:blue"
model_color = "tab:red"
legend_handles = [
    item
    for year, style in styles.items()
    for item in (
        Line2D(
            [], [], color=reference_color, ls=style, lw=1.8, label=f"MIN3P kinetic {year} years"
        ),
        Line2D(
            [], [], color=model_color, ls=style, lw=1.5, label=f"MF6PQC equilibrium {year} years"
        ),
    )
]
for year in PROFILE_YEARS:
    for name, values in [
        ("Porosity", porosity[state_index(year)]),
        ("Diffusion coefficient", effective_diffusion(year)),
    ]:
        error = np.interp(x_reference_hydro, x_mf6, values) - reference_hydro[f"{name} {year}years"]
        comparison_rows.append(
            {"Variable": name, "Time (yr)": year, "RMSE": np.sqrt(np.mean(error**2))}
        )
    for mineral in MINERAL_MOLAR_VOLUMES:
        error = (
            np.interp(x_reference_minerals, x_mf6[1:-1], mineral_profile(year, mineral)[1:-1])
            - reference_minerals[f"{mineral} {year}years"]
        )
        comparison_rows.append(
            {"Variable": mineral, "Time (yr)": year, "RMSE": np.sqrt(np.mean(error**2))}
        )
fig, combined_axes = plt.subplots(3, 3, figsize=(13.8, 10.2), constrained_layout=True)
axes = combined_axes.flat[:2]
for year, style in styles.items():
    axes[0].plot(
        x_reference_hydro,
        reference_hydro[f"Porosity {year}years"],
        color=reference_color,
        ls=style,
        lw=1.8,
    )
    axes[0].plot(x_mf6, porosity[state_index(year)], color=model_color, ls=style, lw=1.5)
    axes[1].plot(
        x_reference_hydro,
        reference_hydro[f"Diffusion coefficient {year}years"],
        color=reference_color,
        ls=style,
        lw=1.8,
    )
    axes[1].plot(x_mf6, effective_diffusion(year), color=model_color, ls=style, lw=1.5)
axes[0].set(xlim=(0.0, 1.0), ylim=(0.001, 1.0), xlabel="Distance [m]", ylabel="Porosity [-]")
axes[1].set(
    xlim=(0.0, 1.0), ylim=(1e-12, 1e-09), xlabel="Distance [m]", ylabel="$D_e$ [m$^2$ s$^{-1}$]"
)
for ax, label in zip(axes, ("(a)", "(b)"), strict=False):
    ax.set_yscale("log")
    ax.set_xticks(np.arange(0.2, 1.01, 0.2))
    ax.text(0.04, 0.95, label, transform=ax.transAxes, va="top", fontsize=14, fontweight="bold")
    ax.tick_params(top=True, right=True)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
axes[1].legend(handles=legend_handles, frameon=False, fontsize=7.5, loc="lower right")

mineral_panels = [
    ("Calcite", (0.0, 0.25), (0.0, 1.0)),
    ("Gypsum", (0.0, 0.5), (0.0, 1.0)),
    ("Ferrihydrite", (0.0, 0.25), (0.0, 1.0)),
    ("Gibbsite", (0.0, 0.25), (0.0, 1.0)),
    ("Siderite", (0.0, 0.25), (0.0, 1.0)),
    ("Jarosite", (0.0, 0.6), (0.02, 0.1)),
]
axes = np.array(combined_axes.flat[2:8]).reshape(3, 2)
for ax, (mineral, ylim, xlim), label in zip(
    axes.flat, mineral_panels, ("(c)", "(d)", "(e)", "(f)", "(g)", "(h)"), strict=False
):
    for year, style in styles.items():
        ax.plot(
            x_reference_minerals,
            reference_minerals[f"{mineral} {year}years"],
            color=reference_color,
            ls=style,
            lw=1.8,
        )
        ax.plot(
            x_mf6[1:-1], mineral_profile(year, mineral)[1:-1], color=model_color, ls=style, lw=1.5
        )
    ax.set(
        xlim=xlim,
        ylim=ylim,
        xlabel="Distance [m]",
        ylabel=f"Volume fraction of {mineral.lower()} [m$^3$ m$^{{-3}}$]",
    )
    ax.text(0.04, 0.95, label, transform=ax.transAxes, va="top", fontsize=14, fontweight="bold")
    ax.tick_params(top=True, right=True)
    if mineral == "Jarosite":
        ax.set_xticks(np.arange(0.04, 0.101, 0.02))
    else:
        ax.set_xticks(np.arange(0.2, 1.01, 0.2))
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
axes[2, 1].legend(handles=legend_handles, frameon=False, fontsize=7.2, loc="upper right")

for ax in combined_axes.flat[8:]:
    ax.set_axis_off()
plt.show()
comparison = pd.DataFrame(comparison_rows)
comparison = comparison.pivot(index="Variable", columns="Time (yr)", values="RMSE").reindex(
    comparison["Variable"].drop_duplicates()
)
display(comparison.style.format("{:.6g}").format_index("{:.6g}", axis=1).set_uuid("ex017_1"))