"""
vcf_parser.py
Parse a ClinVar VCF file and load variants into a local SQLite database.
Handles both plain .vcf and gzipped .vcf.gz inputs.
"""

import sqlite3
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def vcf_to_dataframe(vcf_path: str, max_records: int = 100_000) -> pd.DataFrame:
    """
    Stream-parse a VCF file (plain or gzipped) using cyvcf2.
    Returns a DataFrame with one row per variant.

    Note: ClinVar VCFs may emit "[W::vcf_parse] Contig '1' is not defined in
    the header" warnings. These are harmless — index the file with tabix to
    suppress them:
        tabix -p vcf data/clinvar.vcf.gz
    """
    try:
        from cyvcf2 import VCF
    except ImportError:
        raise ImportError("Install cyvcf2: pip install cyvcf2")

    records = []
    vcf = VCF(vcf_path)

    log.info(f"Parsing VCF: {vcf_path}")
    for i, v in enumerate(vcf):
        if i >= max_records:
            log.info(f"Reached max_records limit ({max_records}). Stopping.")
            break

        # Use variant.INFO.get() directly — more robust than dict(v.INFO),
        # which can fail on multi-value or flag-type INFO fields.
        records.append({
            "chrom":      v.CHROM,
            "pos":        v.POS,        # 1-based, matches VCF spec
            "start":      v.start,      # 0-based, for BED/interval math
            "end":        v.end,
            "variant_id": v.ID or ".",
            "ref":        v.REF,
            "alt":        ",".join(v.ALT) if v.ALT else ".",
            "qual":       v.QUAL,
            "filter":     ";".join(v.FILTER) if v.FILTER else "PASS",
            # ClinVar-specific INFO fields via .get() — safe on missing keys
            "clnsig":     v.INFO.get("CLNSIG") or "Unknown",
            "clndn":      v.INFO.get("CLNDN")  or "Unknown",
            "mc":         v.INFO.get("MC")      or "",
            "af_esp":     _float(v.INFO.get("AF_ESP")),
            "af_exac":    _float(v.INFO.get("AF_EXAC")),
            "af_tgp":     _float(v.INFO.get("AF_TGP")),
        })

    vcf.close()
    df = pd.DataFrame(records)
    log.info(f"Parsed {len(df):,} variants.")
    return df


def _float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def load_to_sqlite(df: pd.DataFrame, db_path: str = "data/variants.db") -> None:
    """Write the variant DataFrame to a SQLite database."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    df.to_sql("variants", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS idx_chrom ON variants(chrom)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_clnsig ON variants(clnsig)")
    con.commit()
    con.close()
    log.info(f"Saved {len(df):,} rows to {db_path}")


if __name__ == "__main__":
    import sys

    # sys.argv is unreliable in Jupyter — notebook kernels populate it with
    # internal flags like "-f", which cyvcf2 tries to open as a file path.
    # Only use argv[1] when it looks like an actual file path.
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    vcf_path = args[0] if args else "data/clinvar.vcf.gz"

    df = vcf_to_dataframe(vcf_path)
    load_to_sqlite(df)
