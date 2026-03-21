#!/usr/bin/env python3
"""Run TRIDENT tissue segmentation and export TMA-core bounding boxes as QuPath GeoJSON."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence


SUPPORTED_EXTENSIONS = {
    ".svs",
    ".tif",
    ".tiff",
    ".ndpi",
    ".mrxs",
    ".scn",
    ".vms",
    ".vmu",
    ".bif",
}


@dataclass(frozen=True)
class CoreCandidate:
    slide_name: str
    source_geojson: Path
    bounds_x: tuple[float, float]
    bounds_y: tuple[float, float]
    centroid_x: float
    centroid_y: float
    area: float


@dataclass(frozen=True)
class RankedCore:
    candidate: CoreCandidate
    row: int
    col: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Segment tissue in a directory of WSIs with TRIDENT and export each "
            "detected TMA core as a QuPath-compatible GeoJSON bounding box."
        )
    )
    parser.add_argument("--slides-dir", type=Path, required=True, help="Directory containing WSI files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for per-slide GeoJSON bounding-box exports.",
    )
    parser.add_argument(
        "--trident-repo",
        type=Path,
        required=True,
        help="Path to a local TRIDENT checkout containing run_batch_of_slides.py.",
    )
    parser.add_argument(
        "--trident-job-dir",
        type=Path,
        default=None,
        help=(
            "Directory for TRIDENT outputs. Defaults to <output-dir>/trident_job. "
            "Existing contours_geojson files in this directory will be reused."
        ),
    )
    parser.add_argument("--segmenter", default="hest", choices=("hest", "grandqc", "otsu"))
    parser.add_argument("--gpu", type=int, default=0, help="GPU index passed to TRIDENT.")
    parser.add_argument("--search-nested", action="store_true", help="Recursively find slides under --slides-dir.")
    parser.add_argument(
        "--wsi-ext",
        nargs="+",
        default=None,
        help="Optional explicit list of WSI extensions to pass to TRIDENT, e.g. .svs .ndpi .tif",
    )
    parser.add_argument(
        "--skip-segmentation",
        action="store_true",
        help="Skip TRIDENT and only use existing GeoJSON files from --trident-job-dir/contours_geojson.",
    )
    parser.add_argument(
        "--min-core-area",
        type=float,
        default=0.0,
        help="Minimum polygon area in level-0 slide pixels for keeping a core candidate.",
    )
    parser.add_argument(
        "--max-core-area",
        type=float,
        default=None,
        help="Maximum polygon area in level-0 slide pixels for keeping a core candidate.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=64,
        help="Extra pixels added on each side of the detected core bounding box.",
    )
    parser.add_argument(
        "--row-tolerance-factor",
        type=float,
        default=0.6,
        help=(
            "Row grouping tolerance as a fraction of the median core height. "
            "Increase if nearby rows are being split too aggressively."
        ),
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional string prepended to every output GeoJSON filename.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing GeoJSON files instead of skipping them.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    slides_dir = args.slides_dir.resolve()
    output_dir = args.output_dir.resolve()
    trident_repo = args.trident_repo.resolve()
    trident_job_dir = (args.trident_job_dir or output_dir / "trident_job").resolve()

    validate_args(slides_dir, trident_repo)

    output_dir.mkdir(parents=True, exist_ok=True)
    trident_job_dir.mkdir(parents=True, exist_ok=True)
    normalize_slide_filenames(slides_dir, args.search_nested)

    if not args.skip_segmentation:
        run_trident_segmentation(
            trident_repo=trident_repo,
            slides_dir=slides_dir,
            job_dir=trident_job_dir,
            gpu=args.gpu,
            segmenter=args.segmenter,
            search_nested=args.search_nested,
            wsi_ext=args.wsi_ext,
        )

    contours_dir = trident_job_dir / "contours_geojson"
    if not contours_dir.exists():
        raise FileNotFoundError(
            f"TRIDENT contours directory not found: {contours_dir}. "
            "Run without --skip-segmentation or point --trident-job-dir at an existing TRIDENT output."
        )

    slide_paths = discover_slides(slides_dir, args.search_nested)
    slide_lookup = {path.stem: path for path in slide_paths}
    if not slide_lookup:
        raise FileNotFoundError(f"No slide files found under {slides_dir}")

    geojson_paths = sorted(contours_dir.glob("*.geojson"))
    if not geojson_paths:
        raise FileNotFoundError(f"No GeoJSON files found in {contours_dir}")

    exported = 0
    for geojson_path in iter_with_progress(geojson_paths):
        slide_name = geojson_path.stem
        slide_path = slide_lookup.get(slide_name)
        if slide_path is None:
            print(
                f"[warn] Skipping {geojson_path.name}: could not match it to a slide file in {slides_dir}",
                file=sys.stderr,
            )
            continue

        candidates = extract_candidates_from_geojson(
            slide_name=slide_name,
            geojson_path=geojson_path,
            min_core_area=args.min_core_area,
            max_core_area=args.max_core_area,
        )
        if not candidates:
            print(f"[warn] No core candidates found for {slide_path.name}", file=sys.stderr)
            continue

        ranked_cores = assign_grid_positions(candidates, row_tolerance_factor=args.row_tolerance_factor)
        exported += export_bounding_boxes_geojson(
            slide_stem=slide_path.stem,
            ranked_cores=ranked_cores,
            output_dir=output_dir,
            prefix=args.prefix,
            padding=args.padding,
            overwrite=args.overwrite,
        )

    print(f"Exported {exported} GeoJSON file(s) into {output_dir}")
    return 0


def validate_args(slides_dir: Path, trident_repo: Path) -> None:
    if not slides_dir.is_dir():
        raise NotADirectoryError(f"--slides-dir does not exist or is not a directory: {slides_dir}")
    if not trident_repo.is_dir():
        raise NotADirectoryError(f"--trident-repo does not exist or is not a directory: {trident_repo}")
    run_script = trident_repo / "run_batch_of_slides.py"
    if not run_script.exists():
        raise FileNotFoundError(f"Could not find TRIDENT entrypoint: {run_script}")


def iter_with_progress(items: Sequence[Path]) -> Iterable[Path]:
    try:
        from tqdm import tqdm

        return tqdm(items, desc="Processing slides", unit="slide")
    except ImportError:
        return items


def discover_slides(slides_dir: Path, search_nested: bool) -> list[Path]:
    pattern = "**/*" if search_nested else "*"
    return sorted(
        path
        for path in slides_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def normalize_slide_filenames(slides_dir: Path, search_nested: bool) -> None:
    slide_paths = discover_slides(slides_dir, search_nested)
    rename_pairs: list[tuple[Path, Path]] = []

    for slide_path in slide_paths:
        normalized_name = normalize_filename(slide_path.name)
        if normalized_name == slide_path.name:
            continue

        destination = slide_path.with_name(normalized_name)
        rename_pairs.append((slide_path, destination))

    if not rename_pairs:
        return

    seen_destinations: dict[Path, Path] = {}
    for source, destination in rename_pairs:
        existing_source = seen_destinations.get(destination)
        if existing_source is not None:
            raise FileExistsError(
                f"Cannot normalize slide filenames because both {existing_source.name} "
                f"and {source.name} map to {destination.name}"
            )
        seen_destinations[destination] = source

        if destination.exists() and destination != source:
            raise FileExistsError(
                f"Cannot rename {source.name} to {destination.name} because the destination already exists"
            )

    for source, destination in rename_pairs:
        print(f"[info] Renaming slide {source.name} -> {destination.name}")
        source.rename(destination)


def normalize_filename(filename: str) -> str:
    return filename.lower().replace(" ", "_")


def run_trident_segmentation(
    trident_repo: Path,
    slides_dir: Path,
    job_dir: Path,
    gpu: int,
    segmenter: str,
    search_nested: bool,
    wsi_ext: Sequence[str] | None,
) -> None:
    command = [
        sys.executable,
        str(trident_repo / "run_batch_of_slides.py"),
        "--task",
        "seg",
        "--wsi_dir",
        str(slides_dir),
        "--job_dir",
        str(job_dir),
        "--gpu",
        str(gpu),
        "--segmenter",
        segmenter,
    ]
    if search_nested:
        command.append("--search_nested")
    if wsi_ext:
        command.extend(["--wsi_ext", *wsi_ext])

    print("[info] Running TRIDENT segmentation:")
    print("       " + " ".join(command))
    subprocess.run(command, cwd=trident_repo, check=True)


def extract_candidates_from_geojson(
    slide_name: str,
    geojson_path: Path,
    min_core_area: float,
    max_core_area: float | None,
) -> list[CoreCandidate]:
    data = json.loads(geojson_path.read_text())
    features = data.get("features", [])
    candidates: list[CoreCandidate] = []

    for feature in features:
        geometry = feature.get("geometry") or {}
        geom_type = geometry.get("type")
        coordinates = geometry.get("coordinates", [])

        polygon_rings: list[list[list[float]]] = []
        if geom_type == "Polygon":
            polygon_rings = [coordinates]
        elif geom_type == "MultiPolygon":
            polygon_rings = list(coordinates)
        else:
            continue

        for rings in polygon_rings:
            if not rings:
                continue
            outer_ring = rings[0]
            if len(outer_ring) < 3:
                continue

            area = abs(polygon_area(outer_ring))
            if area < min_core_area:
                continue
            if max_core_area is not None and area > max_core_area:
                continue

            xs = [point[0] for point in outer_ring]
            ys = [point[1] for point in outer_ring]
            centroid_x, centroid_y = polygon_centroid(outer_ring)
            candidates.append(
                CoreCandidate(
                    slide_name=slide_name,
                    source_geojson=geojson_path,
                    bounds_x=(min(xs), max(xs)),
                    bounds_y=(min(ys), max(ys)),
                    centroid_x=centroid_x,
                    centroid_y=centroid_y,
                    area=area,
                )
            )

    return candidates


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    acc = 0.0
    for idx in range(len(points)):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % len(points)]
        acc += (x1 * y2) - (x2 * y1)
    return acc / 2.0


def polygon_centroid(points: Sequence[Sequence[float]]) -> tuple[float, float]:
    signed_area = polygon_area(points)
    if math.isclose(signed_area, 0.0):
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    cx = 0.0
    cy = 0.0
    for idx in range(len(points)):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % len(points)]
        cross = (x1 * y2) - (x2 * y1)
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross

    factor = 1.0 / (6.0 * signed_area)
    return cx * factor, cy * factor


def assign_grid_positions(
    candidates: Sequence[CoreCandidate],
    row_tolerance_factor: float,
) -> list[RankedCore]:
    if not candidates:
        return []

    heights = [candidate.bounds_y[1] - candidate.bounds_y[0] for candidate in candidates]
    median_height = median(heights)
    tolerance = max(1.0, median_height * row_tolerance_factor)

    sorted_candidates = sorted(candidates, key=lambda candidate: (candidate.centroid_y, candidate.centroid_x))
    rows: list[list[CoreCandidate]] = []
    row_centers: list[float] = []

    for candidate in sorted_candidates:
        if not rows:
            rows.append([candidate])
            row_centers.append(candidate.centroid_y)
            continue

        if abs(candidate.centroid_y - row_centers[-1]) <= tolerance:
            rows[-1].append(candidate)
            row_centers[-1] = sum(item.centroid_y for item in rows[-1]) / len(rows[-1])
        else:
            rows.append([candidate])
            row_centers.append(candidate.centroid_y)

    ranked: list[RankedCore] = []
    for row_idx, row in enumerate(rows, start=1):
        for col_idx, candidate in enumerate(sorted(row, key=lambda item: item.centroid_x), start=1):
            ranked.append(RankedCore(candidate=candidate, row=row_idx, col=col_idx))
    return ranked


def export_bounding_boxes_geojson(
    slide_stem: str,
    ranked_cores: Sequence[RankedCore],
    output_dir: Path,
    prefix: str,
    padding: int,
    overwrite: bool,
) -> int:
    output_path = output_dir / f"{prefix}{slide_stem}_tma_bboxes.geojson"
    if output_path.exists() and not overwrite:
        return 0

    features = []
    for ranked_core in ranked_cores:
        min_x = int(math.floor(ranked_core.candidate.bounds_x[0])) - padding
        min_y = int(math.floor(ranked_core.candidate.bounds_y[0])) - padding
        max_x = int(math.ceil(ranked_core.candidate.bounds_x[1])) + padding
        max_y = int(math.ceil(ranked_core.candidate.bounds_y[1])) + padding

        feature_id = f"{slide_stem}_r{ranked_core.row:02d}_c{ranked_core.col:02d}"
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        features.append(
            {
                "type": "Feature",
                "id": feature_id,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [min_x, min_y],
                        [min_x, max_y],
                        [max_x, max_y],
                        [max_x, min_y],
                        [min_x, min_y],
                    ]],
                },
                "properties": {
                    "objectType": "annotation",
                    "name": feature_id,
                    "classification": {
                        "name": "TMA Core",
                        "colorRGB": -16776961,
                    },
                    "measurements": {
                        "row": ranked_core.row,
                        "column": ranked_core.col,
                        "bbox_width_px": bbox_width,
                        "bbox_height_px": bbox_height,
                        "segmented_area_px": ranked_core.candidate.area,
                    },
                    "metadata": {
                        "slide_name": slide_stem,
                        "row": ranked_core.row,
                        "column": ranked_core.col,
                        "source_geojson": str(ranked_core.candidate.source_geojson),
                    },
                },
            }
        )

    output = {
        "type": "FeatureCollection",
        "features": features,
    }
    output_path.write_text(json.dumps(output, indent=2))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
