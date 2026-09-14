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
    for p in (root, root / "examples/ex015_Xie2015_B2")
    if p.name == "ex015_Xie2015_B2" and (p / "run.py").is_file()
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
    {"font.family": "Arial", "font.size": 10, "xtick.direction": "in", "ytick.direction": "in"}
)
PROFILE_YEARS = (10, 100)
LENGTH = 2.0
SECONDS_PER_DAY = 86400.0
MINERAL_MOLAR_VOLUMES = {"Calcite": 0.03693, "Gypsum": 0.07421}
required = [OUTPUT_DIR / name for name in ("results.npy", "results_porosity.npy", "results_K.npy")]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("Run run.py before plotting. Missing: " + ", ".join(missing))
results, headings, result_times = load_case_results(OUTPUT_DIR)
porosity = np.load(OUTPUT_DIR / "results_porosity.npy", mmap_mode="r")
permeability = np.load(OUTPUT_DIR / "results_K.npy", mmap_mode="r")
reference = pd.read_csv(INPUT_DIR / "MIN3P_results.csv")
sim = flopy.mf6.MFSimulation.load(sim_ws=str(SIMULATION_DIR), sim_name="model", verbosity_level=0)
gwf = sim.get_model("gwf_model")
delr = gwf.dis.delr.array.ravel()
x_mf6 = np.cumsum(delr) - 0.5 * delr
x_min3p = np.linspace(0.0, LENGTH, len(reference))
years = result_times / 365.0
if (
    results.shape[0] != porosity.shape[0]
    or permeability.shape[0] != porosity.shape[0]
    or years[-1] < max(PROFILE_YEARS)
):
    raise ValueError("Saved outputs do not cover the required B2 profile times.")
head_file = flopy.utils.HeadFile(str(SIMULATION_DIR / "gwf_model.hds"))
head_times = np.asarray(head_file.get_times()) / 365.0
np.testing.assert_allclose(head_times[-1], years[-1], rtol=0.0, atol=1e-07)
head_kstpkper = head_file.get_kstpkper()


def nearest_index(values, target):
    return int(np.abs(np.asarray(values) - target).argmin())


def saved_index(year):
    index = nearest_index(years, year)
    if not np.isclose(years[index], year):
        raise ValueError(f"No saved state at {year} years.")
    return index


def mineral_profile(year, mineral):
    return results[saved_index(year), headings.index(mineral), :] * MINERAL_MOLAR_VOLUMES[mineral]


head_profiles = {
    year: head_file.get_data(kstpkper=head_kstpkper[nearest_index(head_times, year)])[0, 0]
    for year in PROFILE_YEARS
}
right_ghb = gwf.get_package("ghb_right")
right_bhead = float(right_ghb.stress_period_data.get_data(key=0)["bhead"][0])
right_distance = 0.5 * delr[-1]
saved_right_head = np.array(
    [
        head_file.get_data(kstpkper=head_kstpkper[nearest_index(head_times, year)])[0, 0, -1]
        for year in years
    ]
)
saved_flux = permeability[:, -1] * (saved_right_head - right_bhead) / right_distance
saved_flux[0] = permeability[0, -1] * 0.007 / LENGTH
styles = {10: "-", 100: "--"}
fig, axes = plt.subplots(3, 2, figsize=(9.2, 10.8), constrained_layout=True)


def min3p_grid_values(mf6_profile):
    return np.interp(x_min3p, x_mf6, np.asarray(mf6_profile))


def plot_profiles(ax, reference_key, mf6_values, ylabel, ylim=None, log=False, label=""):
    for year, style in styles.items():
        error = min3p_grid_values(mf6_values(year)) - reference[f"{reference_key} {year}years"]
        comparison_rows.append(
            {"Variable": reference_key, "Time (yr)": year, "RMSE": np.sqrt(np.mean(error**2))}
        )
        ax.plot(
            x_min3p, reference[f"{reference_key} {year}years"], color="tab:green", ls=style, lw=1.7
        )
        ax.plot(
            x_min3p,
            min3p_grid_values(mf6_values(year)),
            color="tab:red",
            ls=style,
            lw=1.1,
            marker="o",
            ms=2.2,
            markevery=4,
            zorder=3,
        )
    ax.set(xlim=(0, LENGTH), xlabel="Distance [m]", ylabel=ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if log:
        ax.set_yscale("log")


plot_profiles(
    axes[0, 0],
    "Porosity",
    lambda year: porosity[saved_index(year)],
    "Porosity [-]",
    (0.001, 1.0),
    log=True,
)
plot_profiles(
    axes[0, 1],
    "Hydraulic head",
    lambda year: head_profiles[year],
    "Hydraulic head [m]",
    (0.0, 0.008),
)
plot_profiles(
    axes[1, 0],
    "Gypsum",
    lambda year: mineral_profile(year, "Gypsum"),
    "Volume fraction of gypsum [m$^3$ m$^{-3}$]",
    (0.0, 0.7),
)
plot_profiles(
    axes[1, 1],
    "Hydraulic conductivity",
    lambda year: permeability[saved_index(year)] / SECONDS_PER_DAY,
    "Hydraulic conductivity [m s$^{-1}$]",
    (1e-12, 0.01),
    log=True,
)
plot_profiles(
    axes[2, 0],
    "Calcite",
    lambda year: mineral_profile(year, "Calcite"),
    "Volume fraction of calcite [m$^3$ m$^{-3}$]",
    (0.0, 0.7),
)
plot_flux = np.maximum(saved_flux, 1e-12)
axes[2, 1].plot(years, plot_flux, color="tab:red", lw=1.5, marker="o", ms=2.4, markevery=10)
for year, style in styles.items():
    axes[2, 1].axvline(year, color="0.55", ls=style, lw=0.9)
axes[2, 1].set(
    xlim=(0, 120),
    ylim=(1e-08, 0.1),
    xlabel="Time [years]",
    ylabel="Flux of outflow [m$^3$ d$^{-1}$]",
)
axes[2, 1].set_yscale("log")
for ax, label in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)"), strict=False):
    ax.text(0.04, 0.95, label, transform=ax.transAxes, va="top", fontsize=14, fontweight="bold")
    ax.tick_params(top=True, right=True)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
legend_handles = [
    item
    for year, style in styles.items()
    for item in (
        Line2D([], [], color="tab:green", ls=style, lw=1.7, label=f"MIN3P {year} years"),
        Line2D(
            [],
            [],
            color="tab:red",
            ls=style,
            marker="o",
            lw=1.1,
            ms=3,
            label=f"MF6PQC {year} years",
        ),
    )
]
axes[0, 1].legend(handles=legend_handles, frameon=False, fontsize=8, loc="upper right")
plt.show()
comparison = pd.DataFrame(comparison_rows)
comparison = comparison.pivot(index="Variable", columns="Time (yr)", values="RMSE").reindex(
    comparison["Variable"].drop_duplicates()
)
display(comparison.style.format("{:.6g}").format_index("{:.6g}", axis=1).set_uuid("ex015_1"))