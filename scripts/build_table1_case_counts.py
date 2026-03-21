#!/usr/bin/env python3
"""Build Table 1: number of cores per subtype."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


LABEL_ALIASES = {
    "wd-dd": "wd-ddlps",
}

EXCLUDED_LABELS = {
    "ris",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read the embedding tracking CSV and build a Table 1 summary of "
            "core counts per subtype."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to the tracking CSV, e.g. conch_embeddings.csv.",
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        help="Path to write the Table 1 CSV.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_csv = args.input_csv.resolve()
    output_csv = args.output_csv.resolve()

    if not input_csv.is_file():
        raise FileNotFoundError(f"--input-csv does not exist: {input_csv}")

    rows = list(read_tracking_rows(input_csv))
    if not rows:
        raise ValueError(f"No rows found in {input_csv}")

    subtype_counts: Counter[str] = Counter()
    for row in rows:
        subtype = normalize_label(row["label"])
        if should_exclude_label(subtype):
            continue
        subtype_counts[subtype] += 1

    write_table1(output_csv, subtype_counts)
    print(f"Wrote Table 1 to {output_csv}")
    return 0


def read_tracking_rows(input_csv: Path):
    with input_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"original_filename", "label"}
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {input_csv}: {', '.join(sorted(missing))}")
        yield from reader


def normalize_label(label: str) -> str:
    return LABEL_ALIASES.get(label, label)


def should_exclude_label(label: str) -> bool:
    return label in EXCLUDED_LABELS


def write_table1(output_csv: Path, counts: Counter[str]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subtype", "n_cores"])
        writer.writeheader()
        for subtype in sorted(counts):
            writer.writerow({"subtype": subtype, "n_cores": counts[subtype]})


if __name__ == "__main__":
    raise SystemExit(main())
