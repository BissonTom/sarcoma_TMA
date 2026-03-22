#!/usr/bin/env python3
"""Compute high-resolution UMAP plots for matched raw and normalized embedding sets."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


LABEL_ALIASES = {
    "wd-dd": "wd-ddlps",
}

EXCLUDED_LABELS = {
    "ris",
}

CORE_STEM_PATTERN = re.compile(r"^(?P<slide>.+)_r\d+_c\d+$")
VARIANT_NAMES = ("raw", "macenko", "vahadane")
ROW_MODES = ("uncolored", "diagnosis", "slide")


@dataclass(frozen=True)
class EmbeddingRecord:
    key: str
    original_filename: str
    original_path: Path
    label: str
    slide_name: str
    embedding_path: Path


@dataclass(frozen=True)
class MatchedRecord:
    key: str
    label: str
    slide_name: str
    records_by_variant: dict[str, EmbeddingRecord]


@dataclass(frozen=True)
class VariantPlotData:
    coords: object
    labels: list[str]
    slide_names: list[str]
    n_points: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read matched raw, Macenko, and Vahadane embedding CSVs, use all patch embeddings "
            "from the matched cores, and write high-resolution UMAP plots."
        )
    )
    parser.add_argument("--raw-csv", type=Path, required=True, help="Tracking CSV for raw embeddings.")
    parser.add_argument("--macenko-csv", type=Path, required=True, help="Tracking CSV for Macenko embeddings.")
    parser.add_argument("--vahadane-csv", type=Path, required=True, help="Tracking CSV for Vahadane embeddings.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for UMAP PNGs and metadata.")
    parser.add_argument(
        "--max-cores",
        type=int,
        default=None,
        help="Optional maximum number of matched cores to include before expanding to all patch embeddings.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=30,
        help="UMAP neighborhood size.",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP minimum distance parameter.",
    )
    parser.add_argument(
        "--metric",
        default="cosine",
        help="UMAP distance metric.",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=50,
        help="Number of PCA dimensions used before UMAP when possible.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for subsampling and UMAP.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Output DPI for saved PNGs.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=None,
        help="Scatter point size. Defaults to an automatic size based on the subsample count.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Point transparency for scatter plots.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()

    if args.max_cores is not None and args.max_cores <= 0:
        raise ValueError("--max-cores must be positive")
    if args.n_neighbors <= 1:
        raise ValueError("--n-neighbors must be greater than 1")
    if not 0.0 <= args.min_dist:
        raise ValueError("--min-dist must be non-negative")
    if args.pca_dim <= 0:
        raise ValueError("--pca-dim must be positive")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    if not 0.0 < args.alpha <= 1.0:
        raise ValueError("--alpha must be in the range (0, 1]")

    ensure_runtime_dependencies()
    output_dir.mkdir(parents=True, exist_ok=True)

    records_by_variant = {
        "raw": load_tracking_csv(args.raw_csv.resolve()),
        "macenko": load_tracking_csv(args.macenko_csv.resolve()),
        "vahadane": load_tracking_csv(args.vahadane_csv.resolve()),
    }
    matched_records = intersect_records(records_by_variant)
    if not matched_records:
        raise ValueError("No matched cores were found across the three embedding CSVs.")

    selected_records = (
        balanced_subsample(records=matched_records, max_cores=args.max_cores, seed=args.seed)
        if args.max_cores is not None
        else list(matched_records)
    )
    if len(selected_records) < 10:
        raise ValueError(f"Too few matched cores after filtering and subsampling: {len(selected_records)}")

    write_selected_metadata(output_dir / "selected_umap_cores.csv", selected_records)

    plot_data_by_variant: dict[str, VariantPlotData] = {}
    total_points_by_variant: dict[str, int] = {}
    for variant_name in VARIANT_NAMES:
        vectors, labels, slide_names = load_patch_feature_matrix(selected_records, variant_name=variant_name)
        coords = compute_umap_coordinates(
            vectors=vectors,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            pca_dim=args.pca_dim,
            seed=args.seed,
        )
        plot_data_by_variant[variant_name] = VariantPlotData(
            coords=coords,
            labels=labels,
            slide_names=slide_names,
            n_points=len(labels),
        )
        total_points_by_variant[variant_name] = len(labels)

    point_size = args.point_size or choose_point_size(max(total_points_by_variant.values()))
    diagnosis_colors = make_category_colors(
        [label for plot_data in plot_data_by_variant.values() for label in plot_data.labels]
    )
    slide_colors = make_category_colors(
        [slide_name for plot_data in plot_data_by_variant.values() for slide_name in plot_data.slide_names]
    )
    slide_legend_limit = 40
    separation_summary_rows = compute_separation_metrics(
        selected_records=selected_records,
        pca_dim=args.pca_dim,
        seed=args.seed,
    )

    write_individual_plots(
        output_dir=output_dir,
        plot_data_by_variant=plot_data_by_variant,
        diagnosis_colors=diagnosis_colors,
        slide_colors=slide_colors,
        point_size=point_size,
        alpha=args.alpha,
        dpi=args.dpi,
        slide_legend_limit=slide_legend_limit,
    )
    write_grid_figure(
        output_path=output_dir / "umap_grid.png",
        plot_data_by_variant=plot_data_by_variant,
        diagnosis_colors=diagnosis_colors,
        slide_colors=slide_colors,
        point_size=point_size,
        alpha=args.alpha,
        dpi=args.dpi,
        slide_legend_limit=slide_legend_limit,
    )
    write_color_key(output_dir / "diagnosis_colors.csv", diagnosis_colors)
    write_color_key(output_dir / "slide_colors.csv", slide_colors)
    write_separation_metric_outputs(output_dir=output_dir, summary_rows=separation_summary_rows)

    print(
        f"Wrote UMAP plots for {len(selected_records)} matched cores and "
        f"{max(total_points_by_variant.values())} patch embeddings into {output_dir}"
    )
    return 0


def ensure_runtime_dependencies() -> None:
    missing = []
    for module_name in ("h5py", "matplotlib", "numpy", "umap", "sklearn"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)

    if missing:
        raise RuntimeError(
            "Missing required Python dependencies: "
            + ", ".join(missing)
            + ". Install them in your environment and try again."
        )


def load_tracking_csv(csv_path: Path) -> dict[str, EmbeddingRecord]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Tracking CSV not found: {csv_path}")

    records: dict[str, EmbeddingRecord] = {}
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"original_filename", "original_path", "label", "embedding_path"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {', '.join(sorted(missing))}")

        for row in reader:
            label = normalize_label(row["label"])
            if should_exclude_label(label):
                continue

            embedding_path = Path(row["embedding_path"])
            if not embedding_path.is_file():
                continue

            key = derive_record_key(row)
            original_path = Path(row["original_path"])
            records[key] = EmbeddingRecord(
                key=key,
                original_filename=row["original_filename"],
                original_path=original_path,
                label=label,
                slide_name=derive_slide_name(row["original_filename"], original_path),
                embedding_path=embedding_path,
            )
    return records


def normalize_label(label: str) -> str:
    return LABEL_ALIASES.get(label, label)


def should_exclude_label(label: str) -> bool:
    return label in EXCLUDED_LABELS


def derive_record_key(row: dict[str, str]) -> str:
    return Path(row["original_filename"]).stem


def derive_slide_name(original_filename: str, original_path: Path) -> str:
    match = CORE_STEM_PATTERN.match(Path(original_filename).stem)
    if match is not None:
        return match.group("slide")
    return original_path.stem


def intersect_records(
    records_by_variant: dict[str, dict[str, EmbeddingRecord]]
) -> list[MatchedRecord]:
    common_keys = set.intersection(*(set(records.keys()) for records in records_by_variant.values()))
    matched_records: list[MatchedRecord] = []

    for key in sorted(common_keys):
        variant_records = {name: records_by_variant[name][key] for name in VARIANT_NAMES}
        labels = {record.label for record in variant_records.values()}
        slide_names = {record.slide_name for record in variant_records.values()}
        if len(labels) != 1 or len(slide_names) != 1:
            continue

        matched_records.append(
            MatchedRecord(
                key=key,
                label=next(iter(labels)),
                slide_name=next(iter(slide_names)),
                records_by_variant=variant_records,
            )
        )
    return matched_records


def balanced_subsample(
    records: Sequence[MatchedRecord],
    max_cores: int | None,
    seed: int,
) -> list[MatchedRecord]:
    if max_cores is None or len(records) <= max_cores:
        return list(records)

    import random

    rng = random.Random(seed)
    by_label: dict[str, list[MatchedRecord]] = defaultdict(list)
    for record in records:
        by_label[record.label].append(record)

    label_quota = allocate_evenly(
        items_by_group={label: len(group) for label, group in by_label.items()},
        total=max_cores,
    )

    selected: list[MatchedRecord] = []
    for label in sorted(by_label):
        label_records = list(by_label[label])
        rng.shuffle(label_records)
        by_slide: dict[str, list[MatchedRecord]] = defaultdict(list)
        for record in label_records:
            by_slide[record.slide_name].append(record)

        slide_quota = allocate_evenly(
            items_by_group={slide_name: len(group) for slide_name, group in by_slide.items()},
            total=label_quota[label],
        )
        label_selected: list[MatchedRecord] = []
        leftovers: list[MatchedRecord] = []

        for slide_name in sorted(by_slide):
            slide_records = list(by_slide[slide_name])
            rng.shuffle(slide_records)
            take_count = slide_quota[slide_name]
            label_selected.extend(slide_records[:take_count])
            leftovers.extend(slide_records[take_count:])

        remaining = label_quota[label] - len(label_selected)
        if remaining > 0:
            rng.shuffle(leftovers)
            label_selected.extend(leftovers[:remaining])

        selected.extend(label_selected)

    rng.shuffle(selected)
    return selected


def allocate_evenly(items_by_group: dict[str, int], total: int) -> dict[str, int]:
    allocation = {group: 0 for group in items_by_group}
    if total <= 0 or not items_by_group:
        return allocation

    active_groups = [group for group, count in items_by_group.items() if count > 0]
    remaining = min(total, sum(items_by_group.values()))

    while remaining > 0 and active_groups:
        share = max(1, remaining // len(active_groups))
        next_active_groups = []
        for group in active_groups:
            available = items_by_group[group] - allocation[group]
            if available <= 0:
                continue
            take = min(share, available, remaining)
            allocation[group] += take
            remaining -= take
            if items_by_group[group] - allocation[group] > 0:
                next_active_groups.append(group)
            if remaining == 0:
                break
        if active_groups == next_active_groups and remaining > 0:
            for group in active_groups:
                available = items_by_group[group] - allocation[group]
                if available <= 0:
                    continue
                allocation[group] += 1
                remaining -= 1
                if remaining == 0:
                    break
            next_active_groups = [
                group for group in active_groups if items_by_group[group] - allocation[group] > 0
            ]
        active_groups = next_active_groups

    return allocation


def write_selected_metadata(output_path: Path, records: Sequence[MatchedRecord]) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["key", "label", "slide_name", "raw_embedding_path", "macenko_embedding_path", "vahadane_embedding_path"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "key": record.key,
                    "label": record.label,
                    "slide_name": record.slide_name,
                    "raw_embedding_path": str(record.records_by_variant["raw"].embedding_path),
                    "macenko_embedding_path": str(record.records_by_variant["macenko"].embedding_path),
                    "vahadane_embedding_path": str(record.records_by_variant["vahadane"].embedding_path),
                }
            )


def load_patch_feature_matrix(records: Sequence[MatchedRecord], variant_name: str):
    import h5py
    import numpy as np

    vectors = []
    labels: list[str] = []
    slide_names: list[str] = []
    kept_keys: set[str] = set()
    for record in records:
        embedding_path = record.records_by_variant[variant_name].embedding_path
        with h5py.File(embedding_path, "r") as handle:
            features = np.asarray(handle["features"], dtype=np.float32)
        if features.ndim != 2 or features.shape[0] == 0:
            continue
        vectors.append(features)
        labels.extend([record.label] * int(features.shape[0]))
        slide_names.extend([record.slide_name] * int(features.shape[0]))
        kept_keys.add(record.key)

    if len(vectors) != len(records):
        missing_keys = sorted({record.key for record in records} - kept_keys)
        raise ValueError(
            "Some embeddings were empty and cannot be plotted. Missing feature matrices for: "
            + ", ".join(missing_keys[:10])
            + (" ..." if len(missing_keys) > 10 else "")
        )
    return np.vstack(vectors), labels, slide_names


def compute_umap_coordinates(
    vectors,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    pca_dim: int,
    seed: int,
):
    import numpy as np
    import umap
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    matrix = np.asarray(vectors, dtype=np.float32)
    matrix = StandardScaler().fit_transform(matrix)

    if matrix.shape[1] > 2 and matrix.shape[0] > 2:
        pca_components = min(pca_dim, matrix.shape[0] - 1, matrix.shape[1])
        if pca_components >= 2 and pca_components < matrix.shape[1]:
            matrix = PCA(n_components=pca_components, random_state=seed).fit_transform(matrix)

    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, max(2, matrix.shape[0] - 1)),
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
        transform_seed=seed,
    )
    return reducer.fit_transform(matrix)


def choose_point_size(n_points: int) -> float:
    if n_points <= 500:
        return 42.0
    if n_points <= 1500:
        return 27.0
    if n_points <= 3000:
        return 18.0
    if n_points <= 10000:
        return 8.0
    if n_points <= 30000:
        return 4.5
    return 2.8


def make_category_colors(categories: Sequence[str]) -> dict[str, tuple[float, float, float, float]]:
    import matplotlib.pyplot as plt

    unique_categories = sorted(set(categories))
    if not unique_categories:
        return {}

    if len(unique_categories) <= 10:
        cmap = plt.get_cmap("tab10", len(unique_categories))
    elif len(unique_categories) <= 20:
        cmap = plt.get_cmap("tab20", len(unique_categories))
    else:
        cmap = plt.get_cmap("gist_ncar", len(unique_categories))

    return {category: cmap(index) for index, category in enumerate(unique_categories)}


def write_individual_plots(
    output_dir: Path,
    plot_data_by_variant: dict[str, VariantPlotData],
    diagnosis_colors: dict[str, tuple[float, float, float, float]],
    slide_colors: dict[str, tuple[float, float, float, float]],
    point_size: float,
    alpha: float,
    dpi: int,
    slide_legend_limit: int,
) -> None:
    for variant_name in VARIANT_NAMES:
        for row_mode in ROW_MODES:
            output_path = output_dir / f"umap_{variant_name}_{row_mode}.png"
            fig, ax, legend_ax = create_single_panel_figure(figsize=(11.5, 8.5))
            legend_spec = draw_umap_panel(
                ax=ax,
                plot_data=plot_data_by_variant[variant_name],
                color_mode=row_mode,
                diagnosis_colors=diagnosis_colors,
                slide_colors=slide_colors,
                point_size=point_size,
                alpha=alpha,
                slide_legend_limit=slide_legend_limit,
            )
            render_legend_panel(legend_ax=legend_ax, legend_spec=legend_spec)
            if row_mode == "diagnosis":
                title_suffix = " (Colored by Diagnosis)"
            elif row_mode == "slide":
                title_suffix = " (Colored by Slide)"
            else:
                title_suffix = ""
            ax.set_title(f"{variant_name.title()} embeddings{title_suffix}", fontsize=16, pad=12)
            fig.savefig(output_path, dpi=dpi)
            close_figure(fig)


def write_grid_figure(
    output_path: Path,
    plot_data_by_variant: dict[str, VariantPlotData],
    diagnosis_colors: dict[str, tuple[float, float, float, float]],
    slide_colors: dict[str, tuple[float, float, float, float]],
    point_size: float,
    alpha: float,
    dpi: int,
    slide_legend_limit: int,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(26, 20))
    grid = fig.add_gridspec(3, 4, width_ratios=[1.0, 1.0, 1.0, 0.46])
    axes = [[fig.add_subplot(grid[row_idx, col_idx]) for col_idx in range(3)] for row_idx in range(3)]
    legend_axes = [fig.add_subplot(grid[row_idx, 3]) for row_idx in range(3)]
    fig.subplots_adjust(left=0.05, right=0.985, top=0.95, bottom=0.055, wspace=0.18, hspace=0.22)
    row_titles = {
        "uncolored": "Uncolored",
        "diagnosis": "Colored By Diagnosis",
        "slide": "Colored By TMA Slide",
    }
    row_legend_specs = {}

    for col_idx, variant_name in enumerate(VARIANT_NAMES):
        for row_idx, row_mode in enumerate(ROW_MODES):
            ax = axes[row_idx][col_idx]
            legend_spec = draw_umap_panel(
                ax=ax,
                plot_data=plot_data_by_variant[variant_name],
                color_mode=row_mode,
                diagnosis_colors=diagnosis_colors,
                slide_colors=slide_colors,
                point_size=point_size,
                alpha=alpha,
                slide_legend_limit=slide_legend_limit,
            )
            row_legend_specs.setdefault(row_mode, legend_spec)
            if row_idx == 0:
                if row_mode == "diagnosis":
                    title_suffix = " (Colored by Diagnosis)"
                elif row_mode == "slide":
                    title_suffix = " (Colored by Slide)"
                else:
                    title_suffix = ""
                ax.set_title(f"{variant_name.title()}{title_suffix}", fontsize=18, pad=12)
            if col_idx == 0:
                ax.set_ylabel(row_titles[row_mode], fontsize=16)

    for row_idx, row_mode in enumerate(ROW_MODES):
        render_legend_panel(legend_ax=legend_axes[row_idx], legend_spec=row_legend_specs.get(row_mode))

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def create_single_panel_figure(figsize: tuple[float, float]):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.42])
    ax = fig.add_subplot(grid[0, 0])
    legend_ax = fig.add_subplot(grid[0, 1])
    fig.subplots_adjust(left=0.08, right=0.985, top=0.92, bottom=0.09, wspace=0.2)
    return fig, ax, legend_ax


def close_figure(fig) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


def draw_umap_panel(
    ax,
    plot_data: VariantPlotData,
    color_mode: str,
    diagnosis_colors: dict[str, tuple[float, float, float, float]],
    slide_colors: dict[str, tuple[float, float, float, float]],
    point_size: float,
    alpha: float,
    slide_legend_limit: int,
) -> dict[str, object] | None:
    import numpy as np

    xy = np.asarray(plot_data.coords)
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    if color_mode == "uncolored":
        ax.scatter(xy[:, 0], xy[:, 1], s=point_size, c="#4d4d4d", alpha=alpha, linewidths=0)
        return None

    if color_mode == "diagnosis":
        categories = plot_data.labels
        color_lookup = diagnosis_colors
        legend_title = "Diagnosis"
        legend_limit = None
    elif color_mode == "slide":
        categories = plot_data.slide_names
        color_lookup = slide_colors
        legend_title = "TMA Slide"
        legend_limit = slide_legend_limit
    else:
        raise ValueError(f"Unsupported color mode: {color_mode}")

    colors = [color_lookup[category] for category in categories]
    ax.scatter(xy[:, 0], xy[:, 1], s=point_size, c=colors, alpha=alpha, linewidths=0)

    legend_categories = sorted(set(categories))
    if legend_limit is not None and len(legend_categories) > legend_limit:
        return {
            "title": legend_title,
            "note": f"{len(legend_categories)} slides\nSee slide_colors.csv",
            "entries": [],
        }
    return {
        "title": legend_title,
        "entries": [(category, color_lookup[category]) for category in legend_categories],
    }


def render_legend_panel(legend_ax, legend_spec: dict[str, object] | None) -> None:
    from matplotlib.lines import Line2D

    legend_ax.axis("off")
    if not legend_spec:
        return

    note = legend_spec.get("note")
    if note:
        legend_ax.text(
            0.02,
            0.98,
            str(note),
            transform=legend_ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#999999", "alpha": 0.95},
        )
        return

    entries = legend_spec.get("entries", [])
    if not entries:
        return

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=8,
            label=label,
        )
        for label, color in entries
    ]
    legend_ax.legend(
        handles=handles,
        title=str(legend_spec.get("title", "")),
        loc="upper left",
        frameon=False,
        fontsize=9,
        title_fontsize=10,
        ncol=1,
    )


def write_color_key(output_path: Path, color_lookup: dict[str, tuple[float, float, float, float]]) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "r", "g", "b", "a"])
        writer.writeheader()
        for category, rgba in sorted(color_lookup.items()):
            writer.writerow(
                {
                    "category": category,
                    "r": rgba[0],
                    "g": rgba[1],
                    "b": rgba[2],
                    "a": rgba[3],
                }
            )


def compute_separation_metrics(
    selected_records: Sequence[MatchedRecord],
    pca_dim: int,
    seed: int,
) -> list[dict[str, object]]:
    summary_rows = []
    for variant_name in VARIANT_NAMES:
        features, diagnosis_labels, slide_labels = load_patch_feature_matrix(
            selected_records,
            variant_name=variant_name,
        )
        transformed = transform_features(features, pca_dim=pca_dim, seed=seed)
        diagnosis_silhouette = compute_label_silhouette(
            transformed,
            diagnosis_labels,
            sample_size=10000,
            seed=seed,
        )
        slide_silhouette = compute_label_silhouette(
            transformed,
            slide_labels,
            sample_size=10000,
            seed=seed,
        )
        diagnosis_knn_purity = compute_knn_purity(transformed, diagnosis_labels, k=15)
        slide_knn_purity = compute_knn_purity(transformed, slide_labels, k=15)
        summary_rows.append(
            {
                "variant": variant_name,
                "n_cores": len(selected_records),
                "n_patches": len(diagnosis_labels),
                "n_diagnoses": len(set(diagnosis_labels)),
                "n_slides": len(set(slide_labels)),
                "diagnosis_silhouette": diagnosis_silhouette,
                "slide_silhouette": slide_silhouette,
                "diagnosis_knn_purity": diagnosis_knn_purity,
                "slide_knn_purity": slide_knn_purity,
            }
        )
    return summary_rows


def write_separation_metric_outputs(output_dir: Path, summary_rows: Sequence[dict[str, object]]) -> None:
    import json

    write_summary_csv(output_dir / "embedding_separation_metrics.csv", summary_rows)
    write_summary_plot(output_dir / "embedding_separation_metrics.png", summary_rows)
    summary_json = {
        "n_matched_cores": int(summary_rows[0]["n_cores"]) if summary_rows else 0,
        "variants": {str(row["variant"]): row for row in summary_rows},
    }
    (output_dir / "embedding_separation_metrics.json").write_text(json.dumps(summary_json, indent=2))


def transform_features(features, pca_dim: int, seed: int):
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    matrix = np.asarray(features, dtype=np.float32)
    matrix = StandardScaler().fit_transform(matrix)
    pca_components = min(pca_dim, matrix.shape[0] - 1, matrix.shape[1])
    if pca_components >= 2 and pca_components < matrix.shape[1]:
        matrix = PCA(n_components=pca_components, random_state=seed).fit_transform(matrix)
    return matrix


def compute_label_silhouette(features, labels: Sequence[str], sample_size: int, seed: int) -> float:
    import numpy as np
    import random
    from sklearn.metrics import silhouette_score

    valid_indices = indices_for_labels_with_at_least_two_members(labels)
    if len(valid_indices) < 2:
        return float("nan")

    filtered_labels = [labels[index] for index in valid_indices]
    if len(set(filtered_labels)) < 2:
        return float("nan")

    if len(valid_indices) > sample_size:
        rng = random.Random(seed)
        selected_indices = sorted(rng.sample(valid_indices, sample_size))
    else:
        selected_indices = valid_indices

    sampled_features = np.asarray(features)[selected_indices]
    sampled_labels = [labels[index] for index in selected_indices]
    return float(silhouette_score(sampled_features, sampled_labels, metric="euclidean"))


def indices_for_labels_with_at_least_two_members(labels: Sequence[str]) -> list[int]:
    counts: dict[str, int] = defaultdict(int)
    for label in labels:
        counts[label] += 1
    return [index for index, label in enumerate(labels) if counts[label] >= 2]


def compute_knn_purity(features, labels: Sequence[str], k: int) -> float:
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    if len(features) < 2:
        return float("nan")

    effective_k = min(k, len(features) - 1)
    if effective_k <= 0:
        return float("nan")

    nbrs = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
    nbrs.fit(features)
    neighbor_indices = nbrs.kneighbors(return_distance=False)[:, 1:]
    label_array = np.asarray(labels, dtype=object)
    neighbor_labels = label_array[neighbor_indices]
    match_fraction = (neighbor_labels == label_array[:, None]).mean(axis=1)
    return float(match_fraction.mean())


def write_summary_csv(output_path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = [
        "variant",
        "n_cores",
        "n_patches",
        "n_diagnoses",
        "n_slides",
        "diagnosis_silhouette",
        "slide_silhouette",
        "diagnosis_knn_purity",
        "slide_knn_purity",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_plot(output_path: Path, rows: Sequence[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    variant_order = [str(row["variant"]) for row in rows]
    x = np.arange(len(variant_order))
    width = 0.34

    diagnosis_color = "#2c7fb8"
    slide_color = "#d95f0e"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    silhouette_diag = [float(row["diagnosis_silhouette"]) for row in rows]
    silhouette_slide = [float(row["slide_silhouette"]) for row in rows]
    purity_diag = [float(row["diagnosis_knn_purity"]) for row in rows]
    purity_slide = [float(row["slide_knn_purity"]) for row in rows]

    draw_metric_panel(
        ax=axes[0],
        x=x,
        labels=variant_order,
        diagnosis_values=silhouette_diag,
        slide_values=silhouette_slide,
        width=width,
        title="Silhouette Separation",
        y_label="Score",
        diagnosis_color=diagnosis_color,
        slide_color=slide_color,
    )
    draw_metric_panel(
        ax=axes[1],
        x=x,
        labels=variant_order,
        diagnosis_values=purity_diag,
        slide_values=purity_slide,
        width=width,
        title="kNN Purity",
        y_label="Score",
        diagnosis_color=diagnosis_color,
        slide_color=slide_color,
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_metric_panel(
    ax,
    x,
    labels: Sequence[str],
    diagnosis_values: Sequence[float],
    slide_values: Sequence[float],
    width: float,
    title: str,
    y_label: str,
    diagnosis_color: str,
    slide_color: str,
) -> None:
    import math
    import numpy as np

    diagnosis_arr = np.asarray(diagnosis_values, dtype=float)
    slide_arr = np.asarray(slide_values, dtype=float)

    ax.bar(x - width / 2, diagnosis_arr, width=width, color=diagnosis_color, label="Diagnosis")
    ax.bar(x + width / 2, slide_arr, width=width, color=slide_color, label="Slide")
    ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False)

    finite_values = [value for value in np.concatenate([diagnosis_arr, slide_arr]) if math.isfinite(value)]
    if finite_values:
        y_min = min(finite_values)
        y_max = max(finite_values)
        pad = max(0.03, 0.1 * (y_max - y_min if y_max > y_min else 1.0))
        ax.set_ylim(y_min - pad, y_max + pad)


if __name__ == "__main__":
    raise SystemExit(main())
