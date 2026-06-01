"""
qc_stats.py
Standard genomic QC metrics: Ti/Tv ratio, variant type breakdown,
allele frequency distributions, and ClinVar significance enrichment tests.
"""

import sqlite3
import pandas as pd
from scipy.stats import chi2_contingency


TRANSITIONS = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}


def load_variants(db_path: str = "data/variants.db") -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM variants", con)
    con.close()
    return df


def classify_variant_type(ref: str, alt: str) -> str:
    """Classify as SNP, insertion, deletion, or MNP."""
    alts = [a.strip() for a in alt.split(",") if a not in (".", "")]
    if not alts:
        return "unknown"
    a = alts[0]
    if len(ref) == 1 and len(a) == 1:
        return "SNP"
    if len(ref) < len(a):
        return "insertion"
    if len(ref) > len(a):
        return "deletion"
    return "MNP"


def is_transition(ref: str, alt: str):
    alts = [a.strip() for a in alt.split(",") if a not in (".", "")]
    if not alts or len(ref) != 1 or len(alts[0]) != 1:
        return None
    return (ref.upper(), alts[0].upper()) in TRANSITIONS


def compute_titv(df: pd.DataFrame) -> float:
    """Ti/Tv ratio for SNPs only. Expected ~2.0-2.1 for WGS."""
    snps = df[df["variant_type"] == "SNP"].copy()
    ti = snps["is_transition"].sum()
    tv = (~snps["is_transition"]).sum()
    return round(ti / tv, 3) if tv > 0 else float("nan")


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["variant_type"] = df.apply(lambda r: classify_variant_type(r["ref"], r["alt"]), axis=1)
    df["is_transition"] = df.apply(lambda r: is_transition(r["ref"], r["alt"]), axis=1)
    df["clnsig_simple"] = df["clnsig"].apply(_simplify_clnsig)
    return df


def _simplify_clnsig(raw: str) -> str:
    raw = str(raw).lower().replace("_", " ")
    if "pathogenic" in raw and "benign" not in raw:
        return "Pathogenic"
    if "benign" in raw:
        return "Benign"
    if "uncertain" in raw or "vus" in raw:
        return "VUS"
    return "Other"


def clnsig_by_chrom_test(df: pd.DataFrame):
    """
    Chi-square test: are Pathogenic variants enriched on specific chromosomes?
    Returns (chi2, p_value, contingency_table).
    """
    top_chroms = df["chrom"].value_counts().head(10).index.tolist()
    sub = df[df["chrom"].isin(top_chroms)].copy()
    ct = pd.crosstab(sub["chrom"], sub["clnsig_simple"])
    chi2, p, dof, expected = chi2_contingency(ct)
    return chi2, p, ct


def print_summary(df: pd.DataFrame) -> None:
    df = enrich_dataframe(df)
    titv = compute_titv(df)
    print(f"\n{'='*50}")
    print(f"  Total variants:    {len(df):>10,}")
    print(f"  Ti/Tv ratio:       {titv:>10.3f}  (expected ~2.0-2.1 for WGS)")
    print(f"\n  Variant types:")
    for vtype, n in df["variant_type"].value_counts().items():
        print(f"    {vtype:<12} {n:>8,}")
    print(f"\n  Clinical significance:")
    for sig, n in df["clnsig_simple"].value_counts().items():
        print(f"    {sig:<15} {n:>8,}")
    chi2, p, ct = clnsig_by_chrom_test(df)
    print(f"\n  Chi-square (pathogenic enrichment by chrom):")
    print(f"    chi2 = {chi2:.2f},  p = {p:.2e}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    df = load_variants()
    print_summary(df)
