import json
import re
from pathlib import Path

import pandas as pd
import requests

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"


def normalize_tcga_case_id(raw: str) -> str | None:
    """
    Extract a case-level TCGA ID like TCGA-BP-5195 from strings such as:
    TCGA-BP-5195.25c0b433-5557-4165-922e-2c1eac9c26f0
    """
    if pd.isna(raw):
        return None

    raw = str(raw).strip()
    match = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", raw)
    return match.group(1) if match else None


def build_gdc_filters(projects: list[str]) -> dict:
    """
    GDC filters for diagnostic slide files in selected TCGA projects.
    """
    return {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": projects,
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "data_type",
                    "value": ["Slide Image"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "experimental_strategy",
                    "value": ["Diagnostic Slide"],
                },
            },
        ],
    }


def query_gdc_diagnostic_slides(projects: list[str], page_size: int = 10000) -> pd.DataFrame:
    """
    Query the GDC files endpoint for diagnostic slide files.
    """
    filters = build_gdc_filters(projects)

    fields = [
        "file_id",
        "file_name",
        "md5sum",
        "file_size",
        "data_type",
        "data_format",
        "experimental_strategy",
        "cases.case_id",
        "cases.submitter_id",
        "cases.project.project_id",
    ]

    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": str(page_size),
    }

    response = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()

    hits = payload["data"]["hits"]

    rows = []
    for hit in hits:
        for case in hit.get("cases", []):
            rows.append(
                {
                    "file_id": hit.get("file_id"),
                    "file_name": hit.get("file_name"),
                    "md5sum": hit.get("md5sum"),
                    "file_size": hit.get("file_size"),
                    "data_type": hit.get("data_type"),
                    "data_format": hit.get("data_format"),
                    "experimental_strategy": hit.get("experimental_strategy"),
                    "case_id": case.get("case_id"),
                    "case_submitter_id": case.get("submitter_id"),
                    "project_id": case.get("project", {}).get("project_id"),
                }
            )

    df = pd.DataFrame(rows).drop_duplicates()

    if df.empty:
        raise RuntimeError("GDC query returned no rows. Check filters or network access.")

    return df


def load_report_csv(report_csv_path: Path) -> pd.DataFrame:
    """
    Load the TCGA report CSV and extract case IDs from patient_filename.
    Expected columns:
      - patient_filename
      - text
    """
    df = pd.read_csv(report_csv_path)

    required_columns = {"patient_filename", "text"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    out = df[["patient_filename", "text"]].copy()
    out["case_submitter_id"] = out["patient_filename"].map(normalize_tcga_case_id)
    out = out.dropna(subset=["case_submitter_id"]).drop_duplicates()

    return out


def write_gdc_manifest(df: pd.DataFrame, manifest_path: Path) -> None:
    """
    Write a GDC manifest TSV suitable for gdc-client.
    """
    manifest = df[["file_id", "file_name", "md5sum", "file_size"]].drop_duplicates().copy()
    manifest = manifest.rename(columns={"file_id": "id"})
    manifest.to_csv(manifest_path, sep="\t", index=False)


def write_summary(df: pd.DataFrame, path: Path) -> None:
    """
    Save a small human-readable summary text file.
    """
    n_rows = len(df)
    n_cases = df["case_submitter_id"].nunique() if "case_submitter_id" in df.columns else 0
    n_projects = df["project_id"].nunique() if "project_id" in df.columns else 0

    lines = [
        f"Total matched slide rows: {n_rows}",
        f"Unique matched cases: {n_cases}",
        f"Unique projects: {n_projects}",
        "",
    ]

    if "project_id" in df.columns:
        counts = df.groupby("project_id")["case_submitter_id"].nunique().sort_index()
        lines.append("Matched unique cases by project:")
        for project, count in counts.items():
            lines.append(f"  {project}: {count}")

    path.write_text("\n".join(lines))


def main() -> None:
    # ---------------- USER SETTINGS ----------------
    projects = ["TCGA-LUAD", "TCGA-LUSC", "TCGA-BRCA"]

    output_dir = Path(
        "/Users/ethancaspi/Library/CloudStorage/Box-Box/Computer_Vision_Project"
    )

    # Update this if TCGA_Reports is actually a folder rather than the CSV file itself.
    report_csv_path = Path(
        "/Users/ethancaspi/Library/CloudStorage/Box-Box/Computer_Vision_Project/TCGA_Reports.csv"
    )
    # ------------------------------------------------

    output_dir.mkdir(parents=True, exist_ok=True)

    if not report_csv_path.exists():
        raise FileNotFoundError(f"Report CSV path does not exist: {report_csv_path}")
    if report_csv_path.is_dir():
        raise IsADirectoryError(
            f"Report CSV path points to a directory, not a CSV file: {report_csv_path}"
        )

    print("Querying GDC for TCGA diagnostic slide files...")
    gdc_df = query_gdc_diagnostic_slides(projects)
    print(
        f"Found {len(gdc_df)} slide rows across "
        f"{gdc_df['case_submitter_id'].nunique()} unique cases."
    )

    print("Loading pathology report CSV...")
    print(f"Using report CSV: {report_csv_path}")
    report_df = load_report_csv(report_csv_path)
    print(
        f"Loaded {len(report_df)} report rows across "
        f"{report_df['case_submitter_id'].nunique()} unique report cases."
    )

    print("Matching GDC slide cases to report cases...")
    matched_df = gdc_df.merge(report_df, on="case_submitter_id", how="inner").drop_duplicates()
    print(
        f"Matched {len(matched_df)} slide rows across "
        f"{matched_df['case_submitter_id'].nunique()} unique cases."
    )

    # Save all outputs into your Box Drive folder
    gdc_df.to_csv(output_dir / "all_gdc_diagnostic_slides.csv", index=False)
    report_df.to_csv(output_dir / "all_report_cases.csv", index=False)
    matched_df.to_csv(output_dir / "matched_case_slide_report_rows.csv", index=False)

    write_gdc_manifest(matched_df, output_dir / "gdc_manifest.tsv")

    matched_cases = matched_df[["case_submitter_id", "project_id"]].drop_duplicates()
    matched_cases.to_csv(output_dir / "matched_case_ids.csv", index=False)

    write_summary(matched_df, output_dir / "match_summary.txt")

    print("\nDone.")
    print(f"Outputs written to: {output_dir.resolve()}")
    print("\nNext step:")
    print("Use gdc_manifest.tsv with the GDC Data Transfer Tool, e.g.")
    print("gdc-client download -m /path/to/gdc_manifest.tsv -d /path/to/downloaded_slides")


if __name__ == "__main__":
    main()