"""Merge per-method metric CSVs into method-indexed summary tables.

Each input CSV is expected to follow the project's table format:
- header: datasets + AVERAGE/AVARAGE
- body: one model row
- footer: AVERAGE/AVARAGE row

Output format:
- header: method-name + datasets + AVERAGE
- body: one row per method (folder name)
- footer: AVERAGE row recomputed across methods
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd

AVG_ALIASES = ("AVERAGE", "AVARAGE")


def _find_avg_column(columns: list[str]) -> str | None:
    for c in columns:
        if c.strip().upper() in AVG_ALIASES:
            return c
    return None


def _is_avg_label(label: object) -> bool:
    return str(label).strip().upper() in AVG_ALIASES


def merge_method_tables(root_dir: Path, table_name: str, output_path: Path | None = None) -> Path:
    files = sorted(root_dir.rglob(table_name))
    if not files:
        raise FileNotFoundError(f"No files named '{table_name}' found under {root_dir}")

    merged_rows: dict[str, pd.Series] = {}
    expected_columns: list[str] | None = None

    for csv_path in files:
        method_name = csv_path.parent.name

        if method_name in merged_rows:
            raise ValueError(
                f"Duplicate method name '{method_name}' from multiple files. "
                f"Conflicting file: {csv_path}"
            )

        df = pd.read_csv(csv_path, index_col=0)
        df.columns = [str(c).strip() for c in df.columns]

        avg_col_in_input = _find_avg_column(df.columns)
        if avg_col_in_input is None:
            raise ValueError(f"Missing AVERAGE/AVARAGE column in {csv_path}")

        body_df = df.loc[[idx for idx in df.index if not _is_avg_label(idx)]]
        if body_df.empty:
            raise ValueError(f"No model rows found in {csv_path}")

        # Each config is expected to contain one model row. We keep the first row by design.
        model_row = body_df.iloc[0].copy()

        if expected_columns is None:
            expected_columns = list(model_row.index)
        elif list(model_row.index) != expected_columns:
            raise ValueError(
                "Column mismatch across input tables.\n"
                f"Expected: {expected_columns}\n"
                f"Found in {csv_path}: {list(model_row.index)}"
            )

        merged_rows[method_name] = model_row

    merged_df = pd.DataFrame.from_dict(merged_rows, orient="index").sort_index()

    # Convert numeric values and recompute AVERAGE from dataset columns.
    merged_df = merged_df.apply(pd.to_numeric, errors="coerce")

    avg_col = _find_avg_column(list(merged_df.columns))
    if avg_col is None:
        raise ValueError("Internal error: missing average column after merge")

    dataset_cols = [c for c in merged_df.columns if c != avg_col]
    merged_df[avg_col] = merged_df[dataset_cols].mean(axis=1)

    footer = merged_df.mean(axis=0)
    merged_df.loc["AVERAGE"] = footer

    merged_df.index.name = "method-name"

    output = output_path or (root_dir / f"merged_{table_name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output)
    return output


def discover_table_names(root_dir: Path) -> list[str]:
    """Return sorted unique table CSV names found under root_dir."""
    names = {p.name for p in root_dir.rglob("table_*.csv")}
    return sorted(names)


def merge_all_tables(root_dir: Path, table_names: Iterable[str]) -> list[Path]:
    """Merge every table name in table_names and return output file paths."""
    outputs: list[Path] = []
    for table_name in table_names:
        outputs.append(merge_method_tables(root_dir, table_name))
    return outputs


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Root directory containing method subfolders",
    )
    parser.add_argument(
        "--table-name",
        default="table_followup_AUROC.csv",
        help="Input table filename to merge (default: table_followup_AUROC.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file path",
    )
    parser.add_argument(
        "--all-tables",
        action="store_true",
        help="Merge all discovered table_*.csv names under root_dir in one run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if args.all_tables:
        if args.output is not None:
            raise ValueError("--output cannot be used with --all-tables")

        table_names = discover_table_names(args.root_dir)
        if not table_names:
            raise FileNotFoundError(f"No table_*.csv files found under {args.root_dir}")

        outputs = merge_all_tables(args.root_dir, table_names)
        print(f"Merged {len(outputs)} tables:")
        for out in outputs:
            print(out)
        return 0

    output = merge_method_tables(args.root_dir, args.table_name, args.output)
    print(f"Merged table saved to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
