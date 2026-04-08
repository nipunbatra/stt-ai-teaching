#!/usr/bin/env python3
"""Extra Week 11 figures: model size bars, batching curve, optimization stack,
profile loop, pruning visual, distillation visual, notebook map.

All diagrams use the IITGN-modern palette so they match the lecture theme.
Each function is generously padded so labels never overlap or get clipped.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# IITGN-modern palette
PRIMARY = "#1e3a5f"
PRIMARY_LIGHT = "#2e5a8f"
ACCENT = "#e85a4f"
ACCENT_SOFT = "#ff8c7f"
SUCCESS = "#2a9d8f"
WARNING = "#e9c46a"
INK = "#2d3748"
MUTED = "#718096"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.titleweight": "bold",
    }
)

OUT = Path(__file__).parent


# ----------------------------------------------------------------------------
# 1. Model size bar chart  (replaces the "how big are real models" table)
# ----------------------------------------------------------------------------
def model_size_bars(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=150)

    models = ["Spam\nclassifier", "BERT-base", "Llama-7B", "GPT-3"]
    sizes_gb = [0.05, 0.44, 28, 700]
    laptop_ram = 16

    x = np.arange(len(models))
    bars = ax.bar(
        x,
        sizes_gb,
        color=[SUCCESS, PRIMARY_LIGHT, ACCENT, "#7a1f2c"],
        edgecolor="white",
        linewidth=2,
        width=0.55,
    )

    ax.set_yscale("log")
    ax.set_ylabel("Model size in FP32 (GB, log scale)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0.02, 4000)
    ax.set_xlim(-0.7, len(models) - 0.3)

    # Reference line for laptop RAM
    ax.axhline(laptop_ram, color="#caa024", linestyle="--", linewidth=2)
    ax.text(
        -0.6,
        laptop_ram * 1.35,
        "16 GB laptop RAM",
        color="#a07700",
        fontsize=10,
        ha="left",
        fontweight="bold",
    )

    # Bar-top labels: a single line per bar so nothing overlaps on the log scale.
    labels = [
        ("50 MB · fits easily",       SUCCESS),
        ("440 MB · fits",             SUCCESS),
        ("28 GB · does not fit",      ACCENT),
        ("700 GB · needs ~44 laptops", ACCENT),
    ]
    for bar, (label, color) in zip(bars, labels):
        top = bar.get_height()
        cx = bar.get_x() + bar.get_width() / 2
        ax.text(
            cx,
            top * 1.55,
            label,
            ha="center",
            fontsize=11,
            fontweight="bold",
            color=color,
        )

    ax.set_title(
        "Real models grow fast — and laptops do not",
        fontsize=14,
        pad=16,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, which="major")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# 2. Batching: latency and throughput vs batch size  (matches notebook 04)
# ----------------------------------------------------------------------------
def batching_curve(out: Path) -> Path:
    batch_sizes = np.array([1, 8, 32, 128, 512])
    avg_batch_ms = np.array([0.6, 0.9, 1.4, 3.0, 9.5])  # stylised
    per_example_ms = avg_batch_ms / batch_sizes
    throughput = batch_sizes / (avg_batch_ms / 1000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.0), dpi=150)

    ax1.plot(
        batch_sizes,
        per_example_ms,
        "o-",
        color=ACCENT,
        linewidth=2.5,
        markersize=10,
        markeredgecolor="white",
        markeredgewidth=1.5,
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Batch size", fontsize=11)
    ax1.set_ylabel("Time per example (ms)", fontsize=11)
    ax1.set_title(
        "Per-example latency drops with batching",
        fontsize=12,
        color=ACCENT,
        pad=10,
    )
    ax1.grid(alpha=0.25, which="both")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    for x, y in zip(batch_sizes, per_example_ms):
        ax1.annotate(
            f"{y:.2f} ms",
            (x, y),
            textcoords="offset points",
            xytext=(10, 6),
            fontsize=9,
            color=INK,
        )
    ax1.set_xlim(0.7, 900)
    ax1.set_ylim(0.012, 1.2)

    ax2.plot(
        batch_sizes,
        throughput,
        "s-",
        color=SUCCESS,
        linewidth=2.5,
        markersize=10,
        markeredgecolor="white",
        markeredgewidth=1.5,
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("Batch size", fontsize=11)
    ax2.set_ylabel("Examples per second", fontsize=11)
    ax2.set_title(
        "Throughput grows with batching",
        fontsize=12,
        color=SUCCESS,
        pad=10,
    )
    ax2.grid(alpha=0.25)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for x, y in zip(batch_sizes, throughput):
        ax2.annotate(
            f"{y:,.0f}",
            (x, y),
            textcoords="offset points",
            xytext=(10, 4),
            fontsize=9,
            color=INK,
        )
    ax2.set_xlim(0.7, 900)
    ax2.set_ylim(0, max(throughput) * 1.18)

    fig.suptitle(
        "Why batching helps: same model, more work per call",
        fontsize=14,
        y=1.00,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# 3. Profile → fix → measure loop
# ----------------------------------------------------------------------------
def profile_loop(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=150)
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.7, 5.6)
    ax.axis("off")

    steps = [
        ("MEASURE", "time / profile\nthe code", PRIMARY),
        ("FIND", "the slowest\nfunction", "#caa024"),
        ("FIX", "only that\nbottleneck", ACCENT),
        ("MEASURE\nAGAIN", "did it actually\nget faster?", SUCCESS),
    ]

    box_w, box_h = 2.5, 2.4
    gap = 0.6
    total = 4 * box_w + 3 * gap  # = 11.8
    start_x = (13 - total) / 2  # centered
    centers = []
    for i, (title, sub, c) in enumerate(steps):
        x = start_x + i * (box_w + gap)
        rect = mpatches.FancyBboxPatch(
            (x, 1.6),
            box_w,
            box_h,
            boxstyle=mpatches.BoxStyle.Round(pad=0.04, rounding_size=0.18),
            facecolor=c,
            edgecolor="white",
            linewidth=2.5,
            alpha=0.96,
        )
        ax.add_patch(rect)
        ax.text(
            x + box_w / 2,
            3.25,
            title,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="white",
        )
        ax.text(
            x + box_w / 2,
            2.25,
            sub,
            ha="center",
            va="center",
            fontsize=10.5,
            color="white",
        )
        centers.append((x + box_w / 2, x + box_w))

    # Forward arrows between boxes
    for i in range(len(centers) - 1):
        x_start = centers[i][1] + 0.05
        x_end = centers[i + 1][0] - box_w / 2 - 0.05
        ax.annotate(
            "",
            xy=(x_end, 2.8),
            xytext=(x_start, 2.8),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=2),
        )

    # Loopback arrow under the boxes
    left_cx = centers[0][0]
    right_cx = centers[-1][0]
    ax.annotate(
        "",
        xy=(left_cx, 1.45),
        xytext=(right_cx, 1.45),
        arrowprops=dict(
            arrowstyle="-|>",
            color=MUTED,
            lw=1.8,
            connectionstyle="arc3,rad=-0.22",
        ),
    )
    ax.text(
        (left_cx + right_cx) / 2,
        -0.1,
        "if it's still not fast enough → loop again",
        ha="center",
        fontsize=10.5,
        color=MUTED,
        style="italic",
    )

    ax.set_title("The optimization loop", fontsize=15, pad=14)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# 4. Optimization stack (techniques you can combine)
# ----------------------------------------------------------------------------
def optimization_stack(out: Path) -> Path:
    """
    Layout strategy: each bar shows a SHORT title inside the bar.
    A SHORT effect tag (e.g. "440 MB · 90 ms") sits to the right of the bar
    in a fixed column that is well within xlim. Long descriptions are dropped.
    """
    fig, ax = plt.subplots(figsize=(12, 6.0), dpi=150)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    layers = [
        ("Big model (FP32)",         "440 MB · 90 ms",     "#7a1f2c",     8.4),
        ("After distillation",       "fewer layers",       ACCENT,        7.2),
        ("After pruning",            "drop tiny weights",  "#caa024",     6.0),
        ("After INT8 quantization",  "4x fewer bytes",     PRIMARY_LIGHT, 4.8),
        ("After ONNX export",        "fused ops",          SUCCESS,       3.6),
    ]

    base_x = 0.4
    y = 6.4
    bar_h = 0.75
    sub_x = 9.2  # fixed column for the short effect tag
    for label, sub, color, width in layers:
        rect = mpatches.FancyBboxPatch(
            (base_x, y),
            width,
            bar_h,
            boxstyle=mpatches.BoxStyle.Round(pad=0.02, rounding_size=0.12),
            facecolor=color,
            edgecolor="white",
            linewidth=2,
            alpha=0.96,
        )
        ax.add_patch(rect)
        ax.text(
            base_x + 0.30,
            y + bar_h / 2,
            label,
            va="center",
            ha="left",
            color="white",
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            sub_x,
            y + bar_h / 2,
            sub,
            va="center",
            ha="left",
            color=MUTED,
            fontsize=10,
        )
        y -= 1.20

    # Final tag, well below the last bar
    ax.text(
        6.5,
        0.55,
        "Final: ~60 MB  ·  ~8 ms per call  ·  still ~97% accuracy",
        ha="center",
        fontsize=13,
        fontweight="bold",
        color=SUCCESS,
    )

    ax.set_title(
        "Optimizations stack — pick the cheapest wins first",
        fontsize=14,
        pad=14,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# 5. Pruning visual: weight matrix before / after
# ----------------------------------------------------------------------------
def pruning_visual(out: Path) -> Path:
    rng = np.random.default_rng(42)
    W = rng.normal(0, 0.4, (10, 14))
    threshold = np.quantile(np.abs(W), 0.45)
    W_pruned = np.where(np.abs(W) < threshold, 0.0, W)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    fig.subplots_adjust(wspace=0.18)

    for ax, mat, title, color in zip(
        axes,
        [W, W_pruned],
        ["Before pruning", "After pruning"],
        [PRIMARY, SUCCESS],
    ):
        ax.imshow(mat, cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="equal")
        ax.set_title(title, color=color, fontsize=13, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
        if "After" in title:
            zeros = np.where(mat == 0)
            ax.scatter(
                zeros[1],
                zeros[0],
                color="white",
                s=22,
                edgecolor=MUTED,
                linewidths=0.8,
                zorder=3,
            )

    pct = (W_pruned == 0).mean() * 100
    fig.suptitle(
        f"Pruning: set the smallest weights to zero  ·  {pct:.0f}% removed here",
        fontsize=14,
        y=1.00,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# 6. Distillation visual: teacher → student
# ----------------------------------------------------------------------------
def distillation_visual(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5.0), dpi=150)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    # Teacher box
    teacher = mpatches.FancyBboxPatch(
        (0.5, 2.0),
        3.6,
        3.0,
        boxstyle=mpatches.BoxStyle.Round(pad=0.04, rounding_size=0.20),
        facecolor=PRIMARY,
        edgecolor="white",
        linewidth=2.5,
    )
    ax.add_patch(teacher)
    ax.text(2.3, 4.2, "TEACHER", ha="center", color="white", fontsize=14, fontweight="bold")
    ax.text(2.3, 3.55, "big model", ha="center", color="white", fontsize=11)
    ax.text(2.3, 3.05, "440 MB · slow", ha="center", color="#cdd9e8", fontsize=10)
    ax.text(2.3, 2.55, "but smart", ha="center", color="#cdd9e8", fontsize=10)

    # Student box
    student = mpatches.FancyBboxPatch(
        (8.7, 2.6),
        3.8,
        1.8,
        boxstyle=mpatches.BoxStyle.Round(pad=0.04, rounding_size=0.20),
        facecolor=SUCCESS,
        edgecolor="white",
        linewidth=2.5,
    )
    ax.add_patch(student)
    ax.text(10.6, 3.85, "STUDENT", ha="center", color="white", fontsize=13, fontweight="bold")
    ax.text(10.6, 3.30, "small model", ha="center", color="white", fontsize=10)
    ax.text(10.6, 2.85, "60 MB · fast", ha="center", color="#d6f0eb", fontsize=10)

    # Arrow + soft labels
    ax.annotate(
        "",
        xy=(8.65, 3.5),
        xytext=(4.15, 3.5),
        arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=3),
    )
    ax.text(
        6.4,
        4.20,
        '"90% cat,  8% lynx,  2% dog"',
        ha="center",
        color=ACCENT,
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        6.4,
        2.85,
        "soft predictions, not just labels",
        ha="center",
        color=MUTED,
        fontsize=10,
        style="italic",
    )

    ax.text(
        6.5,
        1.0,
        "The student learns the teacher's confidence, not just the answer.",
        ha="center",
        fontsize=11,
        color=INK,
    )

    ax.set_title(
        "Knowledge distillation: a small student copies a big teacher",
        fontsize=14,
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# 7. Notebook map: what each notebook covers (visual table-of-contents)
# ----------------------------------------------------------------------------
def notebook_map(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    # 9 cells in a 3x3 grid, color-coded by topic group
    rows = [
        ("01", "Floating point basics",        PRIMARY),
        ("02", "Parameters & memory",          PRIMARY),
        ("03", "Profiling basics",             "#caa024"),
        ("04", "Batching benchmark",           "#caa024"),
        ("05", "PyTorch dynamic quantization", ACCENT),
        ("06", "ONNX export & quantization",   ACCENT),
        ("07", "Pruning basics",               SUCCESS),
        ("08", "Distillation basics",          SUCCESS),
        ("09", "Comparison dashboard",         PRIMARY_LIGHT),
    ]

    cols = 3
    cell_w, cell_h = 3.6, 1.35
    gap_x, gap_y = 0.30, 0.40
    grid_w = cols * cell_w + (cols - 1) * gap_x
    base_x = (12 - grid_w) / 2
    base_y = 5.0

    for idx, (num, title, color) in enumerate(rows):
        c = idx % cols
        r = idx // cols
        x = base_x + c * (cell_w + gap_x)
        y = base_y - r * (cell_h + gap_y)

        rect = mpatches.FancyBboxPatch(
            (x, y),
            cell_w,
            cell_h,
            boxstyle=mpatches.BoxStyle.Round(pad=0.02, rounding_size=0.14),
            facecolor=color,
            edgecolor="white",
            linewidth=2,
            alpha=0.96,
        )
        ax.add_patch(rect)
        ax.text(
            x + 0.50,
            y + cell_h / 2,
            num,
            va="center",
            ha="left",
            color="white",
            fontsize=22,
            fontweight="bold",
            alpha=0.55,
        )
        ax.text(
            x + 1.40,
            y + cell_h / 2,
            title,
            va="center",
            ha="left",
            color="white",
            fontsize=11,
            fontweight="bold",
        )

    # Legend dots at the bottom for the colour groups, on a single centered row.
    legend_items = [
        ("Numbers", PRIMARY),
        ("Profiling", "#caa024"),
        ("Quantization", ACCENT),
        ("Pruning / distillation", SUCCESS),
        ("Wrap-up", PRIMARY_LIGHT),
    ]
    legend_y = 0.45
    # Pre-compute approximate widths so we can centre the row.
    item_w = [len(lbl) * 0.13 + 0.55 for lbl, _ in legend_items]
    spacing = 0.55
    total_w = sum(item_w) + spacing * (len(legend_items) - 1)
    legend_x = (12 - total_w) / 2
    for (label, c), w in zip(legend_items, item_w):
        ax.scatter(legend_x, legend_y, s=110, color=c, edgecolor="white", linewidth=1.5)
        ax.text(
            legend_x + 0.25,
            legend_y,
            label,
            va="center",
            ha="left",
            fontsize=9.5,
            color=INK,
        )
        legend_x += w + spacing

    ax.set_title(
        "Companion notebooks — one concept each, all CPU-friendly",
        fontsize=14,
        pad=14,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    print("Generated:", model_size_bars(OUT / "model_size_bars.png"))
    print("Generated:", batching_curve(OUT / "batching_curve.png"))
    print("Generated:", profile_loop(OUT / "profile_loop.png"))
    print("Generated:", optimization_stack(OUT / "optimization_stack.png"))
    print("Generated:", pruning_visual(OUT / "pruning_visual.png"))
    print("Generated:", distillation_visual(OUT / "distillation_visual.png"))
    print("Generated:", notebook_map(OUT / "notebook_map.png"))


if __name__ == "__main__":
    main()
