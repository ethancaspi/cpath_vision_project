from __future__ import annotations

import argparse
from pathlib import Path
import math
import pandas as pd


def file_status(download_dir: Path, file_id: str, file_name: str) -> str:
    """
    Determine whether a manifest entry is complete, partial, or missing.

    Assumes gdc-client directory layout like:
      download_dir/<file_id>/<file_name>

    Returns one of:
      - "complete"
      - "partial"
      - "missing"
    """
    uuid_dir = download_dir / file_id
    if not uuid_dir.exists():
        return "missing"

    final_file = uuid_dir / file_name
    if final_file.exists():
        return "complete"

    partials = list(uuid_dir.glob("*.partial"))
    if partials:
        return "partial"

    return "missing"


def split_dataframe_evenly(df: pd.DataFrame, n_parts: int) -> list[pd.DataFrame]:
    """
    Split a dataframe into n nearly equal chunks.
    """
    if n_parts <= 0:
        raise ValueError("n_parts must be positive")

    if df.empty:
        return [df.copy() for _ in range(n_parts)]

    chunk_size = math.ceil(len(df) / n_parts)
    return [df.iloc[i:i + chunk_size].copy() for i in range(0, len(df), chunk_size)]


def build_remaining_manifests(
    manifest_path: Path,
    download_dir: Path,
    out_dir: Path,
    n_parts: int = 5,
) -> None:
    manifest = pd.read_csv(manifest_path, sep="\t")

    required_cols = {"id", "file_name", "md5sum", "file_size"}
    missing = required_cols - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    status_rows = []
    for _, row in manifest.iterrows():
        file_id = str(row["id"])
        file_name = str(row["file_name"])
        status = file_status(download_dir, file_id, file_name)

        status_rows.append(
            {
                "id": file_id,
                "file_name": file_name,
                "md5sum": row["md5sum"],
                "file_size": row["file_size"],
                "status": status,
            }
        )

    status_df = pd.DataFrame(status_rows)

    # Keep anything not fully complete
    remaining_df = status_df[status_df["status"] != "complete"][
        ["id", "file_name", "md5sum", "file_size"]
    ].copy()

    # Save overall remaining manifest and status report
    remaining_manifest_path = out_dir / "gdc_manifest_remaining.tsv"
    status_csv_path = out_dir / "download_status.csv"

    remaining_df.to_csv(remaining_manifest_path, sep="\t", index=False)
    status_df.to_csv(status_csv_path, index=False)

    # Split remaining into sub-manifests
    chunks = split_dataframe_evenly(remaining_df, n_parts=n_parts)

    written_parts = 0
    for idx, chunk in enumerate(chunks, start=1):
        if chunk.empty:
            continue
        chunk_path = out_dir / f"gdc_manifest_remaining_part_{idx:02d}.tsv"
        chunk.to_csv(chunk_path, sep="\t", index=False)
        written_parts += 1

    n_total = len(status_df)
    n_complete = (status_df["status"] == "complete").sum()
    n_partial = (status_df["status"] == "partial").sum()
    n_missing = (status_df["status"] == "missing").sum()
    n_remaining = len(remaining_df)

    print(f"Total manifest rows: {n_total}")
    print(f"Complete: {n_complete}")
    print(f"Partial:  {n_partial}")
    print(f"Missing:  {n_missing}")
    print(f"Remaining rows: {n_remaining}")
    print(f"Overall remaining manifest: {remaining_manifest_path}")
    print(f"Status report: {status_csv_path}")
    print(f"Wrote {written_parts} sub-manifest(s) to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exclude completed GDC downloads and split remaining files into sub-manifests."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to the original gdc_manifest.tsv",
    )
    parser.add_argument(
        "--download-dir",
        required=True,
        type=Path,
        help="Path to the gdc-client destination directory",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory to write the remaining manifest, status CSV, and sub-manifests",
    )
    parser.add_argument(
        "--n-parts",
        type=int,
        default=5,
        help="Number of sub-manifests to create (default: 5)",
    )

    args = parser.parse_args()

    build_remaining_manifests(
        manifest_path=args.manifest,
        download_dir=args.download_dir,
        out_dir=args.out_dir,
        n_parts=args.n_parts,
    )


if __name__ == "__main__":
    main()