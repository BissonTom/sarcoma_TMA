#!/usr/bin/env python3
"""Run TRIDENT tissue segmentation and crop TMA cores from whole-slide images."""

from __future__ import annotations

import argparse
import json
import math
import shutil
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
    polygons: tuple[tuple[tuple[float, float], ...], ...]
    holes: tuple[tuple[tuple[float, float], ...], ...]
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
            "Segment tissue in a directory of WSIs with TRIDENT and crop each "
            "detected TMA core into an individual TIFF."
        )
    )
    parser.add_argument("--slides-dir", type=Path, required=True, help="Directory containing WSI files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for cropped core TIFFs.")
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
    parser.add_argument(
        "--trident-batch-size",
        type=int,
        default=64,
        help="Base TRIDENT batch size. Used as a fallback if no segmentation-specific batch size is provided.",
    )
    parser.add_argument(
        "--trident-seg-batch-size",
        type=int,
        default=512,
        help="TRIDENT segmentation batch size. Increase this first if your GPU has free memory.",
    )
    parser.add_argument(
        "--trident-max-workers",
        type=int,
        default=1,
        help="Optional number of TRIDENT worker processes for slide-level parallelism.",
    )
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
        help="Optional string prepended to every output filename.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output TIFFs instead of skipping them.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    slides_dir = args.slides_dir.resolve()
    output_dir = args.output_dir.resolve()
    trident_repo = args.trident_repo.resolve()
    trident_job_dir = (args.trident_job_dir or output_dir / "trident_job").resolve()

    validate_args(slides_dir, trident_repo)
    ensure_runtime_dependencies()

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
            trident_batch_size=args.trident_batch_size,
            trident_seg_batch_size=args.trident_seg_batch_size,
            trident_max_workers=args.trident_max_workers,
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

    extracted = 0
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
        written_count, core_paths = crop_and_save_cores(
            slide_path=slide_path,
            ranked_cores=ranked_cores,
            output_dir=output_dir,
            prefix=args.prefix,
            padding=args.padding,
            overwrite=args.overwrite,
        )
        extracted += written_count
        if core_paths:
            render_core_overview_pages(
                core_paths=core_paths,
                output_dir=output_dir,
                slide_stem=f"{args.prefix}{slide_path.stem}",
            )

    print(f"Extracted {extracted} core crops into {output_dir}")
    return 0


def validate_args(slides_dir: Path, trident_repo: Path) -> None:
    if not slides_dir.is_dir():
        raise NotADirectoryError(f"--slides-dir does not exist or is not a directory: {slides_dir}")
    if not trident_repo.is_dir():
        raise NotADirectoryError(f"--trident-repo does not exist or is not a directory: {trident_repo}")
    run_script = trident_repo / "run_batch_of_slides.py"
    if not run_script.exists():
        raise FileNotFoundError(f"Could not find TRIDENT entrypoint: {run_script}")


def ensure_runtime_dependencies() -> None:
    missing = []
    for module_name in ("numpy", "openslide", "tifffile", "PIL"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)

    if not shutil.which("openslide-show-properties"):
        print(
            "[warn] openslide CLI tools are not on PATH. That is fine as long as the "
            "OpenSlide shared library is available to the Python package.",
            file=sys.stderr,
        )

    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "Missing required Python packages: "
            f"{joined}. Install them in the environment running this script before use."
        )


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
    trident_batch_size: int,
    trident_seg_batch_size: int | None,
    trident_max_workers: int | None,
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
        "--batch_size",
        str(trident_batch_size),
    ]
    if trident_seg_batch_size is not None:
        command.extend(["--seg_batch_size", str(trident_seg_batch_size)])
    if trident_max_workers is not None:
        command.extend(["--max_workers", str(trident_max_workers)])
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
        annotation_shapes = geometry_to_annotation_shapes(geometry)
        if not annotation_shapes:
            continue

        polygons: list[tuple[tuple[float, float], ...]] = []
        holes: list[tuple[tuple[float, float], ...]] = []
        total_weighted_cx = 0.0
        total_weighted_cy = 0.0
        total_area = 0.0
        all_xs: list[float] = []
        all_ys: list[float] = []

        for outer_ring, inner_rings in annotation_shapes:
            normalized_outer = normalize_ring(outer_ring)
            if len(normalized_outer) < 3:
                continue

            outer_area = abs(polygon_area(normalized_outer))
            if math.isclose(outer_area, 0.0):
                continue

            outer_cx, outer_cy = polygon_centroid(normalized_outer)
            total_area += outer_area
            total_weighted_cx += outer_cx * outer_area
            total_weighted_cy += outer_cy * outer_area
            polygons.append(tuple(normalized_outer))
            all_xs.extend(point[0] for point in normalized_outer)
            all_ys.extend(point[1] for point in normalized_outer)

            for hole_ring in inner_rings:
                normalized_hole = normalize_ring(hole_ring)
                if len(normalized_hole) < 3:
                    continue
                hole_area = abs(polygon_area(normalized_hole))
                if math.isclose(hole_area, 0.0):
                    continue
                hole_cx, hole_cy = polygon_centroid(normalized_hole)
                total_area -= hole_area
                total_weighted_cx -= hole_cx * hole_area
                total_weighted_cy -= hole_cy * hole_area
                holes.append(tuple(normalized_hole))
                all_xs.extend(point[0] for point in normalized_hole)
                all_ys.extend(point[1] for point in normalized_hole)

        if total_area < min_core_area:
            continue
        if max_core_area is not None and total_area > max_core_area:
            continue
        if not polygons or not all_xs or not all_ys:
            continue

        if math.isclose(total_area, 0.0):
            centroid_x = sum(all_xs) / len(all_xs)
            centroid_y = sum(all_ys) / len(all_ys)
        else:
            centroid_x = total_weighted_cx / total_area
            centroid_y = total_weighted_cy / total_area

        candidates.append(
            CoreCandidate(
                slide_name=slide_name,
                source_geojson=geojson_path,
                polygons=tuple(polygons),
                holes=tuple(holes),
                bounds_x=(min(all_xs), max(all_xs)),
                bounds_y=(min(all_ys), max(all_ys)),
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                area=total_area,
            )
        )
    return candidates


def geometry_to_annotation_shapes(
    geometry: dict[str, object],
) -> list[tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]]:
    geom_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])

    if geom_type == "Polygon":
        polygon = parse_polygon_coordinates(coordinates)
        return [polygon] if polygon is not None else []

    if geom_type == "MultiPolygon":
        polygons = []
        for polygon_coords in coordinates:
            polygon = parse_polygon_coordinates(polygon_coords)
            if polygon is not None:
                polygons.append(polygon)
        return polygons

    return []


def parse_polygon_coordinates(
    coordinates: object,
) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]] | None:
    if not isinstance(coordinates, list) or not coordinates:
        return None

    outer_ring = parse_ring(coordinates[0])
    if outer_ring is None:
        return None

    holes = []
    for ring in coordinates[1:]:
        parsed_ring = parse_ring(ring)
        if parsed_ring is not None:
            holes.append(parsed_ring)
    return outer_ring, holes


def parse_ring(points: object) -> list[tuple[float, float]] | None:
    if not isinstance(points, (list, tuple)):
        return None

    parsed_points: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        parsed_points.append((float(point[0]), float(point[1])))

    normalized = normalize_ring(parsed_points)
    if len(normalized) < 3:
        return None
    return normalized


def normalize_ring(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    normalized = [(float(x), float(y)) for x, y in points]
    if len(normalized) >= 2 and normalized[0] == normalized[-1]:
        normalized = normalized[:-1]
    return normalized


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


def crop_and_save_cores(
    slide_path: Path,
    ranked_cores: Sequence[RankedCore],
    output_dir: Path,
    prefix: str,
    padding: int,
    overwrite: bool,
) -> tuple[int, list[Path]]:
    import numpy as np
    import openslide
    import tifffile
    from PIL import Image, ImageDraw

    slide = openslide.OpenSlide(str(slide_path))
    written = 0
    core_paths: list[Path] = []
    try:
        width, height = slide.dimensions
        stem = slide_path.stem

        for ranked_core in ranked_cores:
            min_x = max(0, int(math.floor(ranked_core.candidate.bounds_x[0])) - padding)
            min_y = max(0, int(math.floor(ranked_core.candidate.bounds_y[0])) - padding)
            max_x = min(width, int(math.ceil(ranked_core.candidate.bounds_x[1])) + padding)
            max_y = min(height, int(math.ceil(ranked_core.candidate.bounds_y[1])) + padding)

            crop_width = max_x - min_x
            crop_height = max_y - min_y
            if crop_width <= 0 or crop_height <= 0:
                continue

            output_name = (
                f"{prefix}{stem}_r{ranked_core.row:02d}_c{ranked_core.col:02d}.tiff"
            )
            output_path = output_dir / output_name
            if output_path.exists() and not overwrite:
                core_paths.append(output_path)
                continue

            region = slide.read_region((min_x, min_y), 0, (crop_width, crop_height)).convert("RGB")
            crop_rgb = np.asarray(region, dtype=np.uint8).copy()
            mask = Image.new("L", (crop_width, crop_height), 0)
            draw = ImageDraw.Draw(mask)

            for polygon in ranked_core.candidate.polygons:
                draw.polygon(
                    [(x - min_x, y - min_y) for x, y in polygon],
                    fill=255,
                )
            for hole in ranked_core.candidate.holes:
                draw.polygon(
                    [(x - min_x, y - min_y) for x, y in hole],
                    fill=0,
                )

            mask_array = np.asarray(mask, dtype=np.uint8)
            crop_rgb[mask_array == 0] = 255
            tifffile.imwrite(
                output_path,
                crop_rgb,
                photometric="rgb",
                compression="deflate",
            )
            written += 1
            core_paths.append(output_path)
    finally:
        slide.close()

    return written, core_paths


def render_core_overview_pages(
    core_paths: Sequence[Path],
    output_dir: Path,
    slide_stem: str,
    rows: int = 5,
    cols: int = 10,
) -> None:
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    if not core_paths:
        return

    cell_size = 320
    label_height = 28
    title_height = 40
    gutter = 16
    page_size = rows * cols
    font = ImageFont.load_default()

    for page_idx in range(0, len(core_paths), page_size):
        page_paths = core_paths[page_idx : page_idx + page_size]
        canvas_width = gutter + cols * (cell_size + gutter)
        canvas_height = title_height + gutter + rows * (cell_size + label_height + gutter)
        canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        current_page = (page_idx // page_size) + 1
        total_pages = math.ceil(len(core_paths) / page_size)
        title = f"{slide_stem} cores overview ({len(core_paths)} total, page {current_page}/{total_pages})"
        draw.text((gutter, 12), title, fill=(0, 0, 0), font=font)

        for idx, core_path in enumerate(page_paths):
            grid_row = idx // cols
            grid_col = idx % cols
            x0 = gutter + grid_col * (cell_size + gutter)
            y0 = title_height + gutter + grid_row * (cell_size + label_height + gutter)

            with Image.open(core_path) as image:
                image = image.convert("RGB")
                preview = ImageOps.contain(image, (cell_size, cell_size))

            preview_x = x0 + (cell_size - preview.width) // 2
            preview_y = y0 + (cell_size - preview.height) // 2
            canvas.paste(preview, (preview_x, preview_y))
            draw.rectangle(
                [(x0, y0), (x0 + cell_size - 1, y0 + cell_size - 1)],
                outline=(80, 80, 80),
                width=1,
            )
            draw.text(
                (x0, y0 + cell_size + 6),
                core_path.stem,
                fill=(0, 0, 0),
                font=font,
            )

        output_path = output_dir / f"{slide_stem}_core_overview_p{current_page:02d}.png"
        canvas.save(output_path)


if __name__ == "__main__":
    raise SystemExit(main())
