"""
visualize.py
Publication-ready figures for the VCF QC pipeline.
"""

from pathlib import Path
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from qc_stats import enrich_dataframe, compute_titv, load_variants

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PALETTE = {
    "Pathogenic": "#D85A30",
    "Benign":     "#1D9E75",
    "VUS":        "#EF9F27",
    "Other":      "#888780",
}

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "figure.dpi":         150,
})


def plot_variant_types(df: pd.DataFrame) -> None:
    counts = df["variant_type"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(counts.index, counts.values,
                   color=["#378ADD", "#1D9E75", "#D85A30", "#EF9F27"])
    ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=9)
    ax.set_xlabel("Variant count")
    ax.set_title("Variant type distribution", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "variant_types.png")
    plt.close(fig)
    print("Saved: outputs/variant_types.png")


def plot_clnsig_pie(df: pd.DataFrame) -> None:
    counts = df["clnsig_simple"].value_counts()
    colors = [PALETTE.get(k, "#888780") for k in counts.index]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(counts.values, labels=counts.index, colors=colors,
           autopct="%1.1f%%", startangle=140, pctdistance=0.82)
    ax.set_title("ClinVar clinical significance", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "clnsig_distribution.png")
    plt.close(fig)
    print("Saved: outputs/clnsig_distribution.png")


def plot_chromosome_counts(df: pd.DataFrame) -> None:
    chrom_order = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]
    present = [c for c in chrom_order if c in df["chrom"].unique()]
    counts = df["chrom"].value_counts().reindex(present).dropna()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(counts)), counts.values, color="#378ADD")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, fontsize=9)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Variant count")
    ax.set_title("Variant count by chromosome", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "variants_by_chrom.png")
    plt.close(fig)
    print("Saved: outputs/variants_by_chrom.png")


def plot_qual_histogram(df: pd.DataFrame) -> None:
    qual = df["qual"].dropna()
    if qual.empty:
        print("No QUAL scores to plot.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(qual, bins=50, color="#378ADD", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("QUAL score")
    ax.set_ylabel("Count")
    ax.set_title("QUAL score distribution", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "qual_distribution.png")
    plt.close(fig)
    print("Saved: outputs/qual_distribution.png")


def run_all(df: pd.DataFrame) -> None:
    df = enrich_dataframe(df)
    plot_variant_types(df)
    plot_clnsig_pie(df)
    plot_chromosome_counts(df)
    plot_qual_histogram(df)
    print(f"\nTi/Tv ratio: {compute_titv(df)}")


if __name__ == "__main__":
    df = load_variants()
    run_all(df)
