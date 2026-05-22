"""Generate the v3->v4->v4b->v4c cumulative QA cascade figure for the paper.

Outputs:
    cascade.png  (300 dpi, raster)
    cascade.pdf  (vector)

Run:
    python cascade.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    labels = ["v3 baseline", "+v4 extraction", "+v4b answer", "+v4c event-idx"]
    wins = [0, 1, 7, 10]
    total = 22
    pct = [100.0 * w / total for w in wins]

    base_color = "steelblue"
    highlight_color = "crimson"
    colors = [base_color] * (len(labels) - 1) + [highlight_color]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(labels, pct, color=colors, edgecolor="black", linewidth=0.6, width=0.65)

    # Annotate bars with percentage AND raw fraction
    for bar, p, w in zip(bars, pct, wins):
        height = bar.get_height()
        label = f"{p:.1f}%\n({w}/{total})"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1.2,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylim(0, max(pct) + 12)
    ax.set_ylabel("QA accuracy (%) on v3 retrieval-miss failures", fontsize=11)
    ax.set_title(
        "Multi-track iteration cascade: temporal-reasoning subset (n=22)",
        fontsize=12,
        pad=12,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout()

    here = Path(__file__).resolve().parent
    fig.savefig(here / "cascade.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(here / "cascade.pdf", bbox_inches="tight", facecolor="white")
    print(f"Wrote {here / 'cascade.png'} and {here / 'cascade.pdf'}")


if __name__ == "__main__":
    main()
