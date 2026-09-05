"""Quantify and plot preferential-channel development from paired runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import flopy
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
import numpy as np

from analyze_results import summarize_run
from case_config import PROFILES


CASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="highres")
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    for path in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\msyhbd.ttc"),
    ):
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
    available = {font.name for font in font_manager.fontManager.ttflist}
    preferred = ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans")
    selected = next((name for name in preferred if name in available), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": [selected, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.size": 10,
        }
    )


def read_run(profile: str, scenario: str) -> dict[str, Any]:
    label = f"{profile}_{scenario}"
    output_dir = CASE_DIR / "output" / label
    simulation_dir = CASE_DIR / "simulation" / label
    required = (
        output_dir / "case_metadata.json",
        output_dir / "results.npy",
        output_dir / "results_times.npy",
        output_dir / "results_headings.txt",
        output_dir / "initial_K.npy",
        simulation_dir / "gwf_model.bud",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Paired run is incomplete:\n" + "\n".join(missing))

    metadata = json.loads(
        (output_dir / "case_metadata.json").read_text(encoding="utf-8")
    )
    results = np.load(output_dir / "results.npy")
    times_years = (
        np.load(output_dir / "results_times.npy")
        / metadata["time"]["days_per_year"]
    )
    headings = (output_dir / "results_headings.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    field_index = {name: index for index, name in enumerate(headings)}
    shape = (
        metadata["grid"]["nlay"],
        metadata["grid"]["nrow"],
        metadata["grid"]["ncol"],
    )
    initial_k = np.load(output_dir / "initial_K.npy").reshape(shape)
    if (output_dir / "results_K.npy").is_file():
        hydraulic_k = np.load(output_dir / "results_K.npy").reshape(
            (times_years.size, *shape)
        )
    else:
        hydraulic_k = np.broadcast_to(
            initial_k, (times_years.size, *shape)
        ).copy()

    initial_br = float(metadata["initial_selected_output"]["Br"])
    channel_br = float(metadata["channel_components_mol_per_litre"]["Br"])
    tracer = np.clip(
        (results[:, field_index["Br"], :] - initial_br)
        / (channel_br - initial_br),
        0.0,
        1.0,
    ).reshape((times_years.size, *shape))
    well_indices = np.asarray(
        metadata["boundaries"]["well_flat_indices"], dtype=int
    )
    well_tracer = np.mean(tracer.reshape(times_years.size, -1)[:, well_indices], axis=1)
    well_k = np.mean(
        results[:, field_index["K"], :][:, well_indices], axis=1
    )
    qx, qy, qz, flow_times_years = read_specific_discharge(
        simulation_dir / "gwf_model.bud",
        shape,
        metadata["time"]["days_per_year"],
    )
    return {
        "label": label,
        "output_dir": output_dir,
        "simulation_dir": simulation_dir,
        "metadata": metadata,
        "results": results,
        "times_years": times_years,
        "field_index": field_index,
        "shape": shape,
        "initial_k": initial_k,
        "hydraulic_k": hydraulic_k,
        "tracer": tracer,
        "well_tracer": well_tracer,
        "well_k": well_k,
        "qx": qx,
        "qy": qy,
        "qz": qz,
        "flow_times_years": flow_times_years,
        "summary": summarize_run(output_dir, simulation_dir),
    }


def read_specific_discharge(
    budget_path: Path,
    shape: tuple[int, int, int],
    days_per_year: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    budget = flopy.utils.CellBudgetFile(budget_path, precision="double")
    times_days = np.asarray(budget.get_times(), dtype=float)
    nxyz = int(np.prod(shape))
    qx = np.empty((times_days.size, *shape), dtype=float)
    qy = np.empty_like(qx)
    qz = np.empty_like(qx)
    for time_index, time_days in enumerate(times_days):
        record = budget.get_data(text="DATA-SPDIS", totim=float(time_days))[0]
        node_indices = np.asarray(record["node"], dtype=int) - 1
        if (
            node_indices.size != nxyz
            or node_indices.min() != 0
            or node_indices.max() != nxyz - 1
        ):
            raise ValueError("Unexpected DATA-SPDIS node coverage")
        for destination, name in ((qx, "qx"), (qy, "qy"), (qz, "qz")):
            flat = np.empty(nxyz, dtype=float)
            flat[node_indices] = np.asarray(record[name], dtype=float)
            destination[time_index] = flat.reshape(shape)
    return qx, qy, qz, times_days / days_per_year


def first_arrival_time(
    tracer: np.ndarray, times_years: np.ndarray, threshold: float
) -> np.ndarray:
    reached = tracer >= threshold
    any_reached = np.any(reached, axis=0)
    first_indices = np.argmax(reached, axis=0)
    arrival = np.full(tracer.shape[1:], np.nan, dtype=float)
    arrival[any_reached] = times_years[first_indices[any_reached]]
    return arrival


def effective_flow_width_m(
    qx: np.ndarray,
    layer_thicknesses_m: np.ndarray,
    delc_m: float,
    column: int,
) -> float:
    positive_x_flux_by_row = np.sum(
        np.maximum(qx[:, :, column], 0.0)
        * layer_thicknesses_m[:, None]
        * delc_m,
        axis=0,
    )
    total = float(np.sum(positive_x_flux_by_row))
    squared = float(np.sum(positive_x_flux_by_row**2))
    if total <= 0.0 or squared <= 0.0:
        return float("nan")
    effective_rows = total**2 / squared
    return float(effective_rows * delc_m)


def effective_flow_width_profile_m(
    qx: np.ndarray,
    layer_thicknesses_m: np.ndarray,
    delc_m: float,
) -> np.ndarray:
    return np.asarray(
        [
            effective_flow_width_m(
                qx, layer_thicknesses_m, delc_m, column
            )
            for column in range(qx.shape[2])
        ],
        dtype=float,
    )


def top_fraction_share(values: np.ndarray, fraction: float = 0.10) -> float:
    positive = np.maximum(np.asarray(values, dtype=float).ravel(), 0.0)
    if not np.any(positive > 0.0):
        return 0.0
    count = max(1, int(np.ceil(fraction * positive.size)))
    largest = np.partition(positive, positive.size - count)[-count:]
    return float(np.sum(largest) / np.sum(positive))


def boundary_overlay(axis: plt.Axes, metadata: dict[str, Any]) -> None:
    delr = metadata["grid"]["delr_m"]
    delc = metadata["grid"]["delc_m"]
    channel = metadata["boundaries"]["channel_cells"]
    wells = metadata["boundaries"]["well_cells"]
    axis.scatter(
        [(cell[2] + 0.5) * delr for cell in channel],
        [(cell[1] + 0.5) * delc for cell in channel],
        marker="s",
        s=9,
        facecolors="none",
        edgecolors="cyan",
        linewidths=0.7,
        label="补水渠",
    )
    axis.scatter(
        [(cell[2] + 0.5) * delr for cell in wells],
        [(cell[1] + 0.5) * delc for cell in wells],
        marker="v",
        s=24,
        color="white",
        edgecolors="black",
        linewidths=0.7,
        label="采卤井",
    )


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {path}")


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    feedback = read_run(args.profile, "feedback")
    fixed = read_run(args.profile, "fixed")
    metadata = feedback["metadata"]
    shape = feedback["shape"]
    if shape != fixed["shape"]:
        raise ValueError("Feedback and fixed grids differ")
    np.testing.assert_array_equal(feedback["initial_k"], fixed["initial_k"])
    np.testing.assert_allclose(
        feedback["times_years"], fixed["times_years"], rtol=0.0, atol=1.0e-10
    )

    comparison_dir = CASE_DIR / "output" / f"{args.profile}_channel_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    delr = float(metadata["grid"]["delr_m"])
    delc = float(metadata["grid"]["delc_m"])
    extent = (0.0, shape[2] * delr, 0.0, shape[1] * delc)
    layer_means = np.asarray(
        metadata["initial_hydraulic_conductivity"][
            "layer_geometric_means_m_per_day"
        ],
        dtype=float,
    )
    log_relative_k = np.log(
        feedback["initial_k"] / layer_means[:, None, None]
    )

    # Figure 1: the stochastic initial condition itself.
    limit = float(np.quantile(np.abs(log_relative_k), 0.995))
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    image = None
    for layer, axis in enumerate(axes.flat[:3]):
        image = axis.imshow(
            log_relative_k[layer],
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        boundary_overlay(axis, metadata)
        axis.set(
            title=f"第 {layer + 1} 层：ln(K_h/K_g)",
            xlabel="x（m）",
            ylabel="y（m）",
        )
        if layer == 0:
            axis.legend(loc="lower right", fontsize=8)
    if image is not None:
        figure.colorbar(
            image,
            ax=axes.flat[:3].tolist(),
            shrink=0.80,
            label="ln(K_h/K_g)",
        )
    histogram_axis = axes[1, 1]
    histogram_axis.hist(
        log_relative_k.ravel(),
        bins=50,
        density=True,
        color="0.35",
        alpha=0.75,
        label="生成场",
    )
    x_values = np.linspace(
        float(np.min(log_relative_k)), float(np.max(log_relative_k)), 300
    )
    fitted_std = float(np.std(log_relative_k))
    normal_density = np.exp(-0.5 * (x_values / fitted_std) ** 2) / (
        fitted_std * np.sqrt(2.0 * np.pi)
    )
    histogram_axis.plot(x_values, normal_density, color="tab:red", label="正态参考")
    histogram_axis.set(
        title=f"全域 ln(K_h/K_g)，标准差={fitted_std:.2f}",
        xlabel="ln(K_h/K_g)",
        ylabel="概率密度",
    )
    histogram_axis.legend()
    figure.suptitle(
        f"GSTools 三维对数高斯初始水力传导系数场（{np.prod(shape):,} 单元）",
        fontsize=15,
    )
    save_figure(figure, comparison_dir / "01_initial_log_gaussian_K.png")

    # Time-dependent flow-concentration metrics.
    layer_thicknesses = np.asarray(
        metadata["grid"]["layer_thicknesses_m"], dtype=float
    )
    channel_column = int(metadata["boundaries"]["channel_cells"][0][2])
    well_column = int(metadata["boundaries"]["well_cells"][0][2])
    # Diagnose focusing in the reactive zone, 10% of the domain length
    # downstream from the recharge canal. A mid-domain section would dilute
    # the signal because porosity changes occur first near the canal.
    diagnostic_column = min(
        well_column - 1,
        channel_column + max(1, round(0.10 * (shape[2] - 1))),
    )
    diagnostic_x_m = (diagnostic_column + 0.5) * delr
    metric_rows: list[dict[str, float]] = []
    for flow_index, time_years in enumerate(feedback["flow_times_years"]):
        row: dict[str, float] = {"time_years": float(time_years)}
        for name, run in (("feedback", feedback), ("fixed", fixed)):
            q_magnitude_layer1 = np.hypot(
                run["qx"][flow_index, 0], run["qy"][flow_index, 0]
            )
            row[f"effective_flow_width_{name}_m"] = effective_flow_width_m(
                run["qx"][flow_index],
                layer_thicknesses,
                delc,
                diagnostic_column,
            )
            row[f"top10_q_share_{name}"] = top_fraction_share(q_magnitude_layer1)
            result_index = min(flow_index + 1, run["well_tracer"].size - 1)
            row[f"well_tracer_{name}"] = float(run["well_tracer"][result_index])
            row[f"well_K_{name}_mol_per_L"] = float(run["well_k"][result_index])
        metric_rows.append(row)

    metric_csv = comparison_dir / "channel_metrics_timeseries.csv"
    with metric_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    flow_times = np.asarray([row["time_years"] for row in metric_rows])
    feedback_width = np.asarray(
        [row["effective_flow_width_feedback_m"] for row in metric_rows]
    )
    fixed_width = np.asarray(
        [row["effective_flow_width_fixed_m"] for row in metric_rows]
    )
    feedback_top10 = np.asarray(
        [row["top10_q_share_feedback"] for row in metric_rows]
    )
    fixed_top10 = np.asarray(
        [row["top10_q_share_fixed"] for row in metric_rows]
    )

    # Figure 2: evolution of permeability, tracer and flow focusing.
    target_years = (10.0, 20.0, 30.0)
    result_indices = [
        int(np.argmin(np.abs(feedback["times_years"] - target)))
        for target in target_years
    ]
    flow_indices = [
        int(np.argmin(np.abs(feedback["flow_times_years"] - target)))
        for target in target_years
    ]
    k_ratios = feedback["hydraulic_k"] / feedback["hydraulic_k"][0]
    k_max = max(
        1.01,
        float(np.quantile(k_ratios[result_indices, 0], 0.995)),
    )
    q_log_ratios = []
    for flow_index in flow_indices:
        q_feedback = np.hypot(
            feedback["qx"][flow_index, 0], feedback["qy"][flow_index, 0]
        )
        q_fixed = np.hypot(
            fixed["qx"][flow_index, 0], fixed["qy"][flow_index, 0]
        )
        floor = max(float(np.quantile(q_fixed, 0.02)), 1.0e-12)
        q_log_ratios.append(
            np.log10((q_feedback + floor) / (q_fixed + floor))
        )
    q_limit = max(
        0.05,
        float(np.quantile(np.abs(np.asarray(q_log_ratios)), 0.995)),
    )

    figure, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    for column, (target, result_index, flow_index) in enumerate(
        zip(target_years, result_indices, flow_indices, strict=True)
    ):
        k_image = axes[0, column].imshow(
            k_ratios[result_index, 0],
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="inferno",
            vmin=1.0,
            vmax=k_max,
            interpolation="nearest",
        )
        tracer_image = axes[1, column].imshow(
            feedback["tracer"][result_index, 0],
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        q_image = axes[2, column].imshow(
            q_log_ratios[column],
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="RdBu_r",
            norm=colors.TwoSlopeNorm(vmin=-q_limit, vcenter=0.0, vmax=q_limit),
            interpolation="nearest",
        )
        for row in range(3):
            boundary_overlay(axes[row, column], metadata)
            axes[row, column].set_xlabel("x（m）")
            if column == 0:
                axes[row, column].set_ylabel("y（m）")
        axes[0, column].set_title(f"{target:.0f} 年：K_h/K_h,0")
        axes[1, column].set_title(f"{target:.0f} 年：补给水比例")
        axes[2, column].set_title(f"{target:.0f} 年：log10(q_feedback/q_fixed)")
    figure.colorbar(
        k_image, ax=axes[0, :], shrink=0.75, label="水力传导系数倍率"
    )
    figure.colorbar(
        tracer_image, ax=axes[1, :], shrink=0.75, label="Br 补给水比例"
    )
    figure.colorbar(
        q_image, ax=axes[2, :], shrink=0.75, label="流速反馈放大（log10）"
    )
    figure.suptitle("介质反馈下 K_h、补给水与相对流速演化", fontsize=15)
    save_figure(figure, comparison_dir / "02_channel_evolution.png")

    # Figure 3: final feedback/fixed arrival and flow-field comparison.
    feedback_arrival = first_arrival_time(
        feedback["tracer"], feedback["times_years"], threshold=0.05
    )
    fixed_arrival = first_arrival_time(
        fixed["tracer"], fixed["times_years"], threshold=0.05
    )
    feedback_arrival_layer1 = feedback_arrival[0]
    fixed_arrival_layer1 = fixed_arrival[0]
    arrival_advance = fixed_arrival_layer1 - feedback_arrival_layer1
    final_flow_index = -1
    final_q_feedback = np.hypot(
        feedback["qx"][final_flow_index, 0], feedback["qy"][final_flow_index, 0]
    )
    final_q_fixed = np.hypot(
        fixed["qx"][final_flow_index, 0], fixed["qy"][final_flow_index, 0]
    )
    q_floor = max(float(np.quantile(final_q_fixed, 0.02)), 1.0e-12)
    final_q_amplification = np.log10(
        (final_q_feedback + q_floor) / (final_q_fixed + q_floor)
    )

    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    panels = (
        (
            axes[0, 0],
            log_relative_k[0],
            "初始 ln(K_h/K_g)",
            "RdBu_r",
            -limit,
            limit,
            None,
        ),
        (
            axes[0, 1],
            k_ratios[-1, 0],
            "30 年 K_h/K_h,0（feedback）",
            "inferno",
            1.0,
            k_max,
            None,
        ),
        (
            axes[0, 2],
            final_q_amplification,
            "30 年流速反馈放大 log10(q_f/q_0)",
            "RdBu_r",
            -q_limit,
            q_limit,
            colors.TwoSlopeNorm(vmin=-q_limit, vcenter=0.0, vmax=q_limit),
        ),
        (
            axes[1, 0],
            fixed_arrival_layer1,
            "5% 补给水到达时间（fixed）",
            "viridis_r",
            0.0,
            float(feedback["times_years"][-1]),
            None,
        ),
        (
            axes[1, 1],
            feedback_arrival_layer1,
            "5% 补给水到达时间（feedback）",
            "viridis_r",
            0.0,
            float(feedback["times_years"][-1]),
            None,
        ),
        (
            axes[1, 2],
            arrival_advance,
            "反馈使到达提前的年数（fixed − feedback）",
            "RdBu_r",
            -10.0,
            10.0,
            colors.TwoSlopeNorm(vmin=-10.0, vcenter=0.0, vmax=10.0),
        ),
    )
    for axis, values, title, cmap, vmin, vmax, norm in panels:
        image_kwargs: dict[str, Any] = {"cmap": cmap}
        if norm is None:
            image_kwargs.update(vmin=vmin, vmax=vmax)
        else:
            image_kwargs["norm"] = norm
        image = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            **image_kwargs,
        )
        figure.colorbar(image, ax=axis, shrink=0.80)
        boundary_overlay(axis, metadata)
        axis.set(title=title, xlabel="x（m）", ylabel="y（m）")
    figure.suptitle("初始非均质性与 feedback/fixed 空间对照", fontsize=15)
    save_figure(figure, comparison_dir / "03_feedback_fixed_spatial_comparison.png")

    # Figure 4: concise hydrogeological diagnostics.
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    axes[0, 0].plot(
        feedback["times_years"], feedback["well_tracer"], label="feedback", lw=2
    )
    axes[0, 0].plot(
        fixed["times_years"], fixed["well_tracer"], "--", label="fixed", lw=2
    )
    axes[0, 0].axhline(0.05, color="grey", ls=":", label="5% 突破")
    axes[0, 0].axhline(0.50, color="black", ls=":", label="50% 突破")
    axes[0, 0].set(
        title="采卤井补给水突破",
        xlabel="时间（年）",
        ylabel="Br 补给水比例",
        ylim=(-0.02, 1.02),
    )
    axes[0, 0].legend()

    initial_k_concentration = float(
        metadata["initial_selected_output"]["K"]
    )
    axes[0, 1].plot(
        feedback["times_years"],
        feedback["well_k"] / initial_k_concentration,
        label="feedback",
        lw=2,
    )
    axes[0, 1].plot(
        fixed["times_years"],
        fixed["well_k"] / initial_k_concentration,
        "--",
        label="fixed",
        lw=2,
    )
    axes[0, 1].set(
        title="采出卤水钾浓度",
        xlabel="时间（年）",
        ylabel="C_K / 初始 C_K",
    )
    axes[0, 1].legend()

    axes[1, 0].plot(flow_times, feedback_width, label="feedback", lw=2)
    axes[1, 0].plot(flow_times, fixed_width, "--", label="fixed", lw=2)
    axes[1, 0].set(
        title=f"x={diagnostic_x_m:.0f} m 断面有效过流宽度（越小越集中）",
        xlabel="时间（年）",
        ylabel="有效宽度（m）",
    )
    axes[1, 0].legend()

    axes[1, 1].plot(
        flow_times, 100.0 * feedback_top10, label="feedback", lw=2
    )
    axes[1, 1].plot(
        flow_times, 100.0 * fixed_top10, "--", label="fixed", lw=2
    )
    axes[1, 1].set(
        title="第 1 层前 10% 单元承担的流速份额（越大越集中）",
        xlabel="时间（年）",
        ylabel="流速份额（%）",
    )
    axes[1, 1].legend()
    figure.suptitle("优势通道的四项诊断指标", fontsize=15)
    save_figure(figure, comparison_dir / "04_channel_metrics.png")

    final_width_profile_feedback = effective_flow_width_profile_m(
        feedback["qx"][-1], layer_thicknesses, delc
    )
    final_width_profile_fixed = effective_flow_width_profile_m(
        fixed["qx"][-1], layer_thicknesses, delc
    )
    x_centres = (np.arange(shape[2]) + 0.5) * delr
    width_reduction_percent = (
        100.0
        * (final_width_profile_fixed - final_width_profile_feedback)
        / final_width_profile_fixed
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    axes[0].plot(
        x_centres,
        final_width_profile_feedback,
        lw=2,
        label="feedback",
    )
    axes[0].plot(
        x_centres,
        final_width_profile_fixed,
        "--",
        lw=2,
        label="fixed",
    )
    axes[0].axvline(diagnostic_x_m, color="black", ls=":", label="诊断断面")
    axes[0].set(
        title="30 年各 x 断面的有效过流宽度",
        xlabel="x（m）",
        ylabel="有效宽度（m）",
    )
    axes[0].legend()
    axes[1].plot(x_centres, width_reduction_percent, color="tab:red", lw=2)
    axes[1].axhline(0.0, color="grey", ls="--")
    axes[1].axvline(diagnostic_x_m, color="black", ls=":")
    axes[1].set(
        title="feedback 相对 fixed 的过流宽度缩减",
        xlabel="x（m）",
        ylabel="缩减（%），正值表示更集中",
    )
    figure.suptitle("优势通道沿主流向的空间位置", fontsize=15)
    save_figure(figure, comparison_dir / "05_flow_width_profile.png")

    common_arrival = np.isfinite(feedback_arrival_layer1) & np.isfinite(
        fixed_arrival_layer1
    )
    feedback_breakthrough = feedback["summary"]["bromide_breakthrough_5pct_years"]
    fixed_breakthrough = fixed["summary"]["bromide_breakthrough_5pct_years"]
    breakthrough_advance = (
        None
        if feedback_breakthrough is None or fixed_breakthrough is None
        else float(fixed_breakthrough - feedback_breakthrough)
    )
    feedback_breakthrough_50 = feedback["summary"][
        "bromide_breakthrough_50pct_years"
    ]
    fixed_breakthrough_50 = fixed["summary"][
        "bromide_breakthrough_50pct_years"
    ]
    breakthrough_advance_50 = (
        None
        if feedback_breakthrough_50 is None or fixed_breakthrough_50 is None
        else float(fixed_breakthrough_50 - feedback_breakthrough_50)
    )
    final_width_reduction = float(
        100.0 * (fixed_width[-1] - feedback_width[-1]) / fixed_width[-1]
    )
    top10_flow_share_increase = float(
        feedback_top10[-1] - fixed_top10[-1]
    )
    preferential_channel_formed = bool(
        final_width_reduction >= 10.0
        and top10_flow_share_increase >= 0.02
        and feedback["summary"]["maximum_K_multiplier"] >= 2.0
    )
    summary = {
        "profile": args.profile,
        "grid_cells": int(np.prod(shape)),
        "grid_shape": list(shape),
        "initial_fields_identical": True,
        "initial_log_relative_K_std": float(np.std(log_relative_k)),
        "feedback_5pct_breakthrough_years": feedback_breakthrough,
        "fixed_5pct_breakthrough_years": fixed_breakthrough,
        "feedback_advances_5pct_breakthrough_years": breakthrough_advance,
        "feedback_50pct_breakthrough_years": feedback_breakthrough_50,
        "fixed_50pct_breakthrough_years": fixed_breakthrough_50,
        "feedback_advances_50pct_breakthrough_years": breakthrough_advance_50,
        "final_effective_flow_width_feedback_m": float(feedback_width[-1]),
        "final_effective_flow_width_fixed_m": float(fixed_width[-1]),
        "flow_width_diagnostic_section_x_m": float(diagnostic_x_m),
        "feedback_flow_width_reduction_percent": final_width_reduction,
        "maximum_flow_width_reduction_percent_between_channel_and_wells": float(
            np.nanmax(
                width_reduction_percent[channel_column : well_column + 1]
            )
        ),
        "final_top10_flow_share_feedback": float(feedback_top10[-1]),
        "final_top10_flow_share_fixed": float(fixed_top10[-1]),
        "final_top10_flow_share_increase": top10_flow_share_increase,
        "maximum_K_multiplier_feedback": float(
            feedback["summary"]["maximum_K_multiplier"]
        ),
        "layer1_area_fraction_K_multiplier_gt_2": float(
            np.mean(k_ratios[-1, 0] > 2.0)
        ),
        "median_arrival_advance_years_where_both_arrive": (
            float(np.median(arrival_advance[common_arrival]))
            if np.any(common_arrival)
            else None
        ),
        "feedback_peak_well_K_mol_per_L": float(
            feedback["summary"]["peak_well_K_mol_per_L"]
        ),
        "fixed_peak_well_K_mol_per_L": float(
            fixed["summary"]["peak_well_K_mol_per_L"]
        ),
        "final_mean_well_head_feedback_m": float(
            feedback["summary"]["mean_final_well_head_m"]
        ),
        "final_mean_well_head_fixed_m": float(
            fixed["summary"]["mean_final_well_head_m"]
        ),
        "preferential_channel_formed": preferential_channel_formed,
        "interpretation": (
            "优势通道已形成，且使50%补给水井端突破提前"
            if preferential_channel_formed
            and breakthrough_advance_50 is not None
            and breakthrough_advance_50 > 0.0
            else (
                "存在流量集中，但尚无可分辨的50%井端提前突破"
                if preferential_channel_formed
                else "本组参数下未形成明确的反馈型优势通道"
            )
        ),
    }
    (comparison_dir / "channel_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
