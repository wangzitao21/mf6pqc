%config InlineBackend.figure_format = 'svg'
BASE = OUTPUT_DIR / "figures"
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

W = 183 / 25.4
COL = ["#477B92", "#C99659", "#986D91"]
TIME4 = ["#477B92", "#6B9B88", "#C99659", "#986D91"]
mpl.rcdefaults()
mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "savefig.facecolor": "white",
    }
)
raw, headings, result_times = load_case_results(OUTPUT_DIR)
with np.load(INPUT_DIR / "MIN3P_reference.npz", allow_pickle=False) as ref:
    D = {k: ref[k] for k in ref.files}
model = flopy.mf6.MFSimulation.load(sim_ws=str(SIMULATION_DIR), verbosity_level=0)
dx = model.get_model("gwf_model").dis.delr.array.ravel()
D["x_m"] = np.cumsum(dx) - dx / 2
D["phi"] = np.load(OUTPUT_DIR / "results_porosity.npy")
D["K_m_d"] = np.load(OUTPUT_DIR / "results_K.npy")
total_days = sum(float(r[0]) for r in model.tdis.perioddata.get_data())
D["time_yr"] = result_times / 365.0
D["mineral_names"] = np.array(
    ["Calcite", "Gypsum", "Ferrihydrite", "Gibbsite", "Siderite", "Jarosite"]
)
D["molar_volumes_L_mol"] = np.array([0.03693, 0.07421, 0.02399, 0.03319, 0.02926, 0.15463])
D["minerals"] = np.stack(
    [
        raw[:, headings.index(n)] * v
        for n, v in zip(D["mineral_names"], D["molar_volumes_L_mol"], strict=False)
    ],
    axis=1,
)
D["selected_yr"] = np.array([100, 1000, 3000])
completed = np.load(OUTPUT_DIR / "results_diffc.npy") / 86400
D["Dp_m2_s"] = np.vstack([np.full(len(dx), 1e-09 * 0.35 ** (1 / 3)), completed])
D["De_m2_s"] = D["phi"] * D["Dp_m2_s"]


def _manuscript_layout_09(fig):
    fig.canvas.draw()
    width_pt, height_pt = fig.get_size_inches() * 72
    offsets = [
        (0, 0),
        (0, 0),
        (0, 0),
        (0, -16.5),
        (0, -16.5),
        (0, -16.5),
        (88.747021, -33),
        (88.747021, -33),
    ]
    if len(fig.axes) != len(offsets):
        raise ValueError("Manuscript panel count changed")
    for ax, (dx, dy) in zip(fig.axes, offsets, strict=True):
        pos = ax.get_position()
        ax.set_position([pos.x0 + dx / width_pt, pos.y0 - dy / height_pt, pos.width, pos.height])
    for legend in fig.legends:
        box = legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
        legend.set_bbox_to_anchor(
            (box.x0 + 0 / width_pt, box.y0 - 4.5 / height_pt, box.width, box.height),
            transform=fig.transFigure,
        )


def save(n, fig, axes):
    _manuscript_layout_09(fig)
    plt.show()


def title(ax, letter, name):
    ax.set_title(f"{letter}  {name}", loc="left", pad=7, fontsize=9)


def grid(rows, cols, height, top=0.86, bottom=0.1, wspace=0.5, hspace=0.67):
    fig, axs = plt.subplots(rows, cols, figsize=(W, height), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=top, bottom=bottom, wspace=wspace, hspace=hspace)
    return (fig, list(axs.flat))


def legend(fig, times, unit="yr", ref="MIN3P", colors=None):
    colors = colors or (COL if len(times) == 3 else [COL[0], COL[-1]])
    handles = [
        Line2D([], [], color=c, lw=1.5, label=f"{t:,} {unit}")
        for t, c in zip(times, colors, strict=False)
    ]
    handles += [
        Line2D([], [], color=".3", lw=1.2, label="MF6PQC"),
        Line2D([], [], color=".3", ls="none", marker="o", mfc="white", ms=3, label=ref),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.991),
        ncol=len(handles),
        columnspacing=1.4,
        handlelength=1.8,
    )


def curves(ax, d, key, refkey, component=None, scale=1, refscale=1, interior=False, xmax=2):
    colors = COL if len(d["selected_yr"]) == 3 else [COL[0], COL[-1]]
    for j, (t, c) in enumerate(zip(d["selected_yr"], colors, strict=False)):
        idx = np.flatnonzero(np.isclose(d["time_yr"], t)).item()
        y = d[key][idx] if component is None else d[key][idx, component]
        cells = slice(1, -1) if interior else slice(None)
        ax.plot(d["x_m"][cells], y[cells] * scale, color=c, lw=1.35)
        rx = d["reference_mineral_x_m"] if refkey == "reference_minerals" else d["reference_x_m"]
        ry = d[refkey][j] if component is None else d[refkey][j, component]
        ax.plot(rx, ry * refscale, ls="none", marker="o", ms=2.3, mfc="white", mec=c, mew=0.5)
    ax.set(
        xlim=(0, xmax), xlabel="Distance (m)", xticks=[0, 0.4, 0.8] if xmax == 0.8 else [0, 1, 2]
    )
    ax.yaxis.set_major_locator(MaxNLocator(4))


def fig9():
    d = D
    fig, allaxes = grid(3, 3, 6.6, top=0.902, bottom=0.084, wspace=0.55, hspace=0.68)
    fig.delaxes(allaxes[-1])
    axs = allaxes[:8]
    legend(fig, [100, 1000, 3000])
    for i, name in enumerate(d["mineral_names"]):
        curves(axs[i], d, "minerals", "reference_minerals", component=i, interior=True, xmax=0.8)
        title(axs[i], chr(97 + i), str(name))
        axs[i].set_ylabel("Volume fraction", labelpad=3)
    curves(axs[6], d, "phi", "reference_phi", interior=True, xmax=0.8)
    title(axs[6], "g", "Porosity")
    axs[6].set_ylabel("Porosity")
    axs[6].axhline(0.35, color=".65", ls="--", lw=0.7)
    curves(
        axs[7],
        d,
        "De_m2_s",
        "reference_De_m2_s",
        scale=10000000000.0,
        refscale=10000000000.0,
        interior=True,
        xmax=0.8,
    )
    title(axs[7], "h", "Effective diffusion")
    axs[7].set_ylabel("$D_e$ ($10^{-10}$ m²/s)")
    for ax in axs:
        vals = np.concatenate(
            [
                line.get_ydata()[(line.get_xdata() >= 0) & (line.get_xdata() <= 0.8)]
                for line in ax.lines
                if len(line.get_xdata()) > 2
            ]
        )
        lo, hi = (float(vals.min()), float(vals.max()))
        margin = (hi - lo) * 0.06
        ax.set_ylim(max(0, lo - margin), hi + margin)
    save(9, fig, axs)


fig9()