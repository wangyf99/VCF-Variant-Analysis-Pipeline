"""
dicom_extractor.py
Extract structured metadata from DICOM files and store in SQLite,
alongside the existing variant data. Supports CT and MR modalities.

For demo purposes, uses pydicom's bundled test files.
To use real TCIA cancer imaging data, replace the paths in
load_dicom_files() with your downloaded DICOM directory.
"""

import sqlite3
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pydicom.data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# DICOM tags to extract — maps column name -> DICOM keyword
TAGS = {
    "patient_id":         "PatientID",
    "modality":           "Modality",
    "study_date":         "StudyDate",
    "study_description":  "StudyDescription",
    "series_description": "SeriesDescription",
    "manufacturer":       "Manufacturer",
    "rows":               "Rows",
    "columns":            "Columns",
    "slice_thickness":    "SliceThickness",
    "pixel_spacing":      "PixelSpacing",
    "bits_allocated":     "BitsAllocated",
    "photometric":        "PhotometricInterpretation",
    "sop_class_uid":      "SOPClassUID",
    "instance_uid":       "SOPInstanceUID",
}


def extract_metadata(dcm_path: str) -> dict:
    """Read one DICOM file and return a flat metadata dict."""
    ds = pydicom.dcmread(str(dcm_path))
    record = {"file_path": str(dcm_path)}

    for col, keyword in TAGS.items():
        val = ds.get(keyword)
        if val is None:
            record[col] = None
        elif keyword == "PixelSpacing":
            # PixelSpacing is a DSfloat sequence — store as "row_spacing,col_spacing"
            record[col] = ",".join(str(float(v)) for v in val)
        else:
            record[col] = str(val)

    # Pixel array stats — useful for imaging QC
    if hasattr(ds, "PixelData"):
        try:
            arr = ds.pixel_array.astype(float)
            record["pixel_mean"]   = round(float(np.mean(arr)), 2)
            record["pixel_std"]    = round(float(np.std(arr)), 2)
            record["pixel_min"]    = round(float(np.min(arr)), 2)
            record["pixel_max"]    = round(float(np.max(arr)), 2)
        except Exception:
            record.update({"pixel_mean": None, "pixel_std": None,
                           "pixel_min": None, "pixel_max": None})
    return record


def load_dicom_files(dicom_dir: str = None) -> list[str]:
    """
    Return a list of DICOM file paths to process.
    If dicom_dir is provided, recursively finds all .dcm files there.
    Otherwise falls back to pydicom bundled test files for demo purposes.
    """
    if dicom_dir:
        paths = list(Path(dicom_dir).rglob("*.dcm"))
        log.info(f"Found {len(paths)} DICOM files in {dicom_dir}")
        return [str(p) for p in paths]

    # Demo mode: use pydicom bundled test files
    demo_files = ["CT_small.dcm", "MR_small.dcm", "CT_small.dcm"]
    paths = []
    for fname in demo_files:
        p = pydicom.data.get_testdata_file(fname)
        if p:
            paths.append(str(p))
    log.info(f"Demo mode: loaded {len(paths)} bundled DICOM test files")
    return paths


def dicom_to_dataframe(dicom_dir: str = None) -> pd.DataFrame:
    """Parse all DICOM files and return a metadata DataFrame."""
    paths = load_dicom_files(dicom_dir)
    records = []
    for path in paths:
        try:
            records.append(extract_metadata(path))
        except Exception as e:
            log.warning(f"Skipping {path}: {e}")

    df = pd.DataFrame(records)
    log.info(f"Extracted metadata from {len(df)} DICOM files")
    return df


def load_to_sqlite(df: pd.DataFrame, db_path: str = "data/variants.db") -> None:
    """
    Store DICOM metadata in the same SQLite DB as variant data.
    This enables future JOIN queries linking imaging and genomic data
    by patient_id — a core pattern in cancer genomics research.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    df.to_sql("imaging_metadata", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS idx_patient ON imaging_metadata(patient_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_modality ON imaging_metadata(modality)")
    con.commit()
    con.close()
    log.info(f"Saved {len(df)} DICOM records to imaging_metadata table in {db_path}")


def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*50}")
    print(f"  DICOM files parsed:  {len(df):>6}")
    print(f"\n  Modalities:")
    for mod, n in df["modality"].value_counts().items():
        print(f"    {mod:<10} {n:>4}")
    print(f"\n  Pixel intensity stats (mean ± std):")
    for _, row in df.iterrows():
        print(f"    {row['modality']} | {row['file_path'].split('/')[-1]:<20} "
              f"mean={row['pixel_mean']:>8.1f}  std={row['pixel_std']:>7.1f}  "
              f"range=[{row['pixel_min']:.0f}, {row['pixel_max']:.0f}]")
    print(f"{'='*50}\n")


def plot_dicom_images(df: pd.DataFrame, output_dir: str = "outputs") -> None:
    """
    Render pixel arrays side-by-side for all parsed DICOM files.
    Saves to outputs/dicom_preview.png.
    """
    Path(output_dir).mkdir(exist_ok=True)
    n = len(df)
    fig = plt.figure(figsize=(5 * n, 5))
    gs = gridspec.GridSpec(1, n)

    for i, (_, row) in enumerate(df.iterrows()):
        ds = pydicom.dcmread(row["file_path"])
        ax = fig.add_subplot(gs[i])
        try:
            arr = ds.pixel_array
            ax.imshow(arr, cmap="gray", aspect="auto")
        except Exception:
            ax.text(0.5, 0.5, "No pixel data", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title(
            f"{row['modality']} | {row['rows']}x{row['columns']}px\n"
            f"mean={row['pixel_mean']:.0f}  std={row['pixel_std']:.0f}",
            fontsize=9
        )
        ax.axis("off")

    fig.suptitle("DICOM image preview — pixel array QC", fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = f"{output_dir}/dicom_preview.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info(f"Saved: {out_path}")


if __name__ == "__main__":
    import sys
    dicom_dir = next((a for a in sys.argv[1:] if not a.startswith("-")), None)
    df = dicom_to_dataframe(dicom_dir)
    print_summary(df)
    load_to_sqlite(df)
    plot_dicom_images(df)
