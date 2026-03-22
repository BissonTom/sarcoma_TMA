#!/usr/bin/env python3
"""Render paginated overview sheets for extracted TMA core images."""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
CORE_NAME_PATTERN = re.compile(r"^(?P<slide_stem>.+)_r(?P<row>\d+)_c(?P<col>\d+)$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create paginated overview images from extracted TMA core crops. "
            "Core files are grouped by slide name using the *_r##_c## filename pattern."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing extracted core image files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for overview PNGs. Defaults to --input-dir.",
    )
    parser.add_argument(
        "--search-nested",
        action="store_true",
        help="Recursively search for core images under --input-dir.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Number of rows per overview page.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=10,
        help="Number of columns per overview page.",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=320,
        help="Maximum preview size in pixels for each core image.",
    )
    parser.add_argument(
        "--label-height",
        type=int,
        default=28,
        help="Reserved label height in pixels under each preview.",
    )
    parser.add_argument(
        "--title-height",
        type=int,
        default=40,
        help="Reserved title height in pixels at the top of each page.",
    )
    parser.add_argument(
        "--gutter",
        type=int,
        default=16,
        help="Spacing in pixels between previews.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir).resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input-dir does not exist or is not a directory: {input_dir}")
    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("--rows and --cols must both be positive integers")
    if args.cell_size <= 0 or args.label_height < 0 or args.title_height < 0 or args.gutter < 0:
        raise ValueError("--cell-size must be positive and spacing dimensions cannot be negative")

    ensure_runtime_dependencies()
    output_dir.mkdir(parents=True, exist_ok=True)

    core_paths = discover_core_images(input_dir=input_dir, search_nested=args.search_nested)
    if not core_paths:
        raise FileNotFoundError(f"No supported core images found under {input_dir}")

    grouped_core_paths = group_core_images(core_paths)
    if not grouped_core_paths:
        raise ValueError(
            "No core images matched the expected naming pattern '*_r##_c##'. "
            "This script groups files using the row/column suffix written by extract_tma_cores.py."
        )

    total_pages = 0
    total_groups = 0
    for slide_stem, slide_core_paths in grouped_core_paths.items():
        total_groups += 1
        total_pages += render_core_overview_pages(
            core_paths=slide_core_paths,
            output_dir=output_dir,
            slide_stem=slide_stem,
            rows=args.rows,
            cols=args.cols,
            cell_size=args.cell_size,
            label_height=args.label_height,
            title_height=args.title_height,
            gutter=args.gutter,
        )

    print(
        f"Wrote {total_pages} overview page(s) for {total_groups} slide group(s) into {output_dir}"
    )
    return 0


def ensure_runtime_dependencies() -> None:
    missing = []
    for module_name in ("PIL",):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)

    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing required Python dependencies: {joined}. Install them in your environment and try again."
        )


def discover_core_images(input_dir: Path, search_nested: bool) -> list[Path]:
    iterator: Iterable[Path]
    if search_nested:
        iterator = input_dir.rglob("*")
    else:
        iterator = input_dir.iterdir()

    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def group_core_images(core_paths: Sequence[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[tuple[int, int, Path]]] = defaultdict(list)

    for core_path in core_paths:
        match = CORE_NAME_PATTERN.match(core_path.stem)
        if match is None:
            continue

        grouped[match.group("slide_stem")].append(
            (int(match.group("row")), int(match.group("col")), core_path)
        )

    ordered_groups: dict[str, list[Path]] = {}
    for slide_stem in sorted(grouped):
        ordered = sorted(grouped[slide_stem], key=lambda item: (item[0], item[1], item[2].name))
        ordered_groups[slide_stem] = [path for _, _, path in ordered]
    return ordered_groups


def render_core_overview_pages(
    core_paths: Sequence[Path],
    output_dir: Path,
    slide_stem: str,
    rows: int,
    cols: int,
    cell_size: int = 320,
    label_height: int = 28,
    title_height: int = 40,
    gutter: int = 16,
) -> int:
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    if not core_paths:
        return 0

    page_size = rows * cols
    font = ImageFont.load_default()
    total_pages = math.ceil(len(core_paths) / page_size)

    for page_idx in range(0, len(core_paths), page_size):
        page_paths = core_paths[page_idx : page_idx + page_size]
        canvas_width = gutter + cols * (cell_size + gutter)
        canvas_height = title_height + gutter + rows * (cell_size + label_height + gutter)
        canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        current_page = (page_idx // page_size) + 1
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

    return total_pages


if __name__ == "__main__":
    raise SystemExit(main())
