#!/usr/bin/env python3
"""Run TRIDENT CONCH feature extraction and write a tracking CSV."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


SUPPORTED_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".svs",
    ".ndpi",
    ".mrxs",
    ".scn",
    ".vms",
    ".vmu",
    ".bif",
}


@dataclass(frozen=True)
class EmbeddingJob:
    images_dir: Path
    output_dir: Path
    trident_job_dir: Path
    csv_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run TRIDENT CONCH patch feature extraction on a directory of images "
            "or WSIs, then write a CSV linking each original file to its embedding file."
        )
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        action="append",
        required=True,
        help=(
            "Directory containing cropped TMA cores or other image files to embed. "
            "Pass this flag multiple times to process multiple corresponding folders in one run. "
            "If you pass it once and it contains immediate child directories, each child directory "
            "will be treated as its own embedding job."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        action="append",
        required=True,
        help=(
            "Directory where the tracking CSV will be written for the corresponding --images-dir. "
            "Pass once per input directory. If a single parent --images-dir is expanded into child "
            "jobs automatically, each child job will write into a matching child directory here."
        ),
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
        action="append",
        default=None,
        help=(
            "Directory for TRIDENT outputs for the corresponding --images-dir. "
            "Defaults to <output-dir>/trident_job. Existing feature files in this directory will be reused. "
            "Pass once per input directory if you want explicit per-dataset job directories. "
            "If a single parent --images-dir is expanded into child jobs automatically, this may point "
            "to a parent job directory and matching child directories will be created underneath it."
        ),
    )
    parser.add_argument(
        "--patch-encoder",
        default="conch_v1",
        choices=("conch_v1", "conch_v15"),
        help="TRIDENT patch encoder to use.",
    )
    parser.add_argument("--mag", type=int, default=20, help="Magnification passed to TRIDENT.")
    parser.add_argument("--patch-size", type=int, default=512, help="Patch size in pixels passed to TRIDENT.")
    parser.add_argument("--overlap", type=int, default=0, help="Patch overlap in pixels passed to TRIDENT.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index passed to TRIDENT.")
    parser.add_argument(
        "--trident-batch-size",
        type=int,
        default=64,
        help="Base TRIDENT batch size. Used for segmentation and as a fallback for feature extraction.",
    )
    parser.add_argument(
        "--trident-feat-batch-size",
        type=int,
        default=256,
        help="TRIDENT feature-extraction batch size. Increase this first if your GPU has free memory.",
    )
    parser.add_argument(
        "--trident-max-workers",
        type=int,
        default=None,
        help="Optional number of TRIDENT worker processes for slide-level parallelism.",
    )
    parser.add_argument(
        "--search-nested",
        action="store_true",
        help="Recursively search for images under --images-dir.",
    )
    parser.add_argument(
        "--wsi-ext",
        nargs="+",
        default=None,
        help="Optional explicit list of image extensions to pass to TRIDENT.",
    )
    parser.add_argument(
        "--reader-type",
        default="auto",
        choices=("auto", "image", "openslide", "cucim", "sdpc"),
        help=(
            "Reader passed to TRIDENT. Use 'image' for regular TIFF/PNG/JPG core images. "
            "Use 'auto' to infer from file extensions."
        ),
    )
    parser.add_argument(
        "--mpp",
        type=float,
        default=None,
        help=(
            "Microns-per-pixel for regular image inputs. Required when TRIDENT uses "
            "the 'image' reader because TIFF/PNG/JPG cores do not store MPP reliably."
        ),
    )
    parser.add_argument(
        "--skip-feature-extraction",
        action="store_true",
        help="Skip TRIDENT and only build the CSV from existing feature files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the CSV and rerun TRIDENT even if outputs already exist.",
    )
    parser.add_argument(
        "--csv-name",
        default="conch_embeddings.csv",
        help="Filename for the tracking CSV written inside --output-dir.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    trident_repo = args.trident_repo.resolve()
    validate_common_args(trident_repo, args.patch_size, args.mag, args.overlap)
    jobs = build_embedding_jobs(
        images_dirs=args.images_dir,
        output_dirs=args.output_dir,
        trident_job_dirs=args.trident_job_dir,
        csv_name=args.csv_name,
    )

    total_rows = 0
    for job in jobs:
        total_rows += run_embedding_job(
            job=job,
            trident_repo=trident_repo,
            patch_encoder=args.patch_encoder,
            mag=args.mag,
            patch_size=args.patch_size,
            overlap=args.overlap,
            gpu=args.gpu,
            trident_batch_size=args.trident_batch_size,
            trident_feat_batch_size=args.trident_feat_batch_size,
            trident_max_workers=args.trident_max_workers,
            search_nested=args.search_nested,
            wsi_ext=args.wsi_ext,
            reader_type=args.reader_type,
            mpp=args.mpp,
            skip_feature_extraction=args.skip_feature_extraction,
        )

    print(f"Finished {len(jobs)} embedding job(s) with {total_rows} total CSV rows.")
    return 0


def validate_common_args(
    trident_repo: Path,
    patch_size: int,
    mag: int,
    overlap: int,
) -> None:
    if not trident_repo.is_dir():
        raise NotADirectoryError(f"--trident-repo does not exist or is not a directory: {trident_repo}")
    run_script = trident_repo / "run_batch_of_slides.py"
    if not run_script.exists():
        raise FileNotFoundError(f"Could not find TRIDENT entrypoint: {run_script}")
    if patch_size <= 0:
        raise ValueError("--patch-size must be positive")
    if mag <= 0:
        raise ValueError("--mag must be positive")
    if overlap < 0:
        raise ValueError("--overlap must be zero or positive")


def build_embedding_jobs(
    images_dirs: Sequence[Path],
    output_dirs: Sequence[Path],
    trident_job_dirs: Sequence[Path] | None,
    csv_name: str,
) -> list[EmbeddingJob]:
    if len(images_dirs) == 1 and len(output_dirs) == 1:
        return build_jobs_from_parent_directory(
            images_dir=images_dirs[0],
            output_dir=output_dirs[0],
            trident_job_dir=trident_job_dirs[0] if trident_job_dirs else None,
            csv_name=csv_name,
        )

    if len(images_dirs) != len(output_dirs):
        raise ValueError(
            f"Expected the same number of --images-dir and --output-dir values, got "
            f"{len(images_dirs)} and {len(output_dirs)}."
        )
    if trident_job_dirs is not None and len(trident_job_dirs) not in (0, len(images_dirs)):
        raise ValueError(
            f"Expected either zero or {len(images_dirs)} --trident-job-dir values, got {len(trident_job_dirs)}."
        )

    jobs: list[EmbeddingJob] = []
    for idx, (images_dir, output_dir) in enumerate(zip(images_dirs, output_dirs, strict=True), start=1):
        resolved_images_dir = images_dir.resolve()
        resolved_output_dir = output_dir.resolve()
        if not resolved_images_dir.is_dir():
            raise NotADirectoryError(
                f"--images-dir #{idx} does not exist or is not a directory: {resolved_images_dir}"
            )

        explicit_job_dir = None
        if trident_job_dirs:
            explicit_job_dir = trident_job_dirs[idx - 1]
        resolved_job_dir = (explicit_job_dir or resolved_output_dir / "trident_job").resolve()

        jobs.append(
            EmbeddingJob(
                images_dir=resolved_images_dir,
                output_dir=resolved_output_dir,
                trident_job_dir=resolved_job_dir,
                csv_path=resolved_output_dir / csv_name,
            )
        )
    return jobs


def build_jobs_from_parent_directory(
    images_dir: Path,
    output_dir: Path,
    trident_job_dir: Path | None,
    csv_name: str,
) -> list[EmbeddingJob]:
    resolved_images_dir = images_dir.resolve()
    resolved_output_dir = output_dir.resolve()
    if not resolved_images_dir.is_dir():
        raise NotADirectoryError(f"--images-dir does not exist or is not a directory: {resolved_images_dir}")

    child_dirs = sorted(path for path in resolved_images_dir.iterdir() if path.is_dir())
    if not child_dirs:
        resolved_job_dir = (trident_job_dir or resolved_output_dir / "trident_job").resolve()
        return [
            EmbeddingJob(
                images_dir=resolved_images_dir,
                output_dir=resolved_output_dir,
                trident_job_dir=resolved_job_dir,
                csv_path=resolved_output_dir / csv_name,
            )
        ]

    jobs = []
    resolved_job_parent = trident_job_dir.resolve() if trident_job_dir is not None else None
    for child_dir in child_dirs:
        child_output_dir = resolved_output_dir / child_dir.name
        child_job_dir = (
            resolved_job_parent / child_dir.name
            if resolved_job_parent is not None
            else child_output_dir / "trident_job"
        )
        jobs.append(
            EmbeddingJob(
                images_dir=child_dir.resolve(),
                output_dir=child_output_dir.resolve(),
                trident_job_dir=child_job_dir.resolve(),
                csv_path=(child_output_dir / csv_name).resolve(),
            )
        )
    return jobs


def run_embedding_job(
    job: EmbeddingJob,
    trident_repo: Path,
    patch_encoder: str,
    mag: int,
    patch_size: int,
    overlap: int,
    gpu: int,
    trident_batch_size: int,
    trident_feat_batch_size: int,
    trident_max_workers: int | None,
    search_nested: bool,
    wsi_ext: Sequence[str] | None,
    reader_type: str,
    mpp: float | None,
    skip_feature_extraction: bool,
) -> int:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.trident_job_dir.mkdir(parents=True, exist_ok=True)
    image_paths = discover_images(job.images_dir, search_nested)
    if not image_paths:
        raise FileNotFoundError(f"No supported image files found under {job.images_dir}")

    resolved_reader_type = resolve_reader_type(image_paths=image_paths, reader_type=reader_type)
    validate_reader_requirements(resolved_reader_type, mpp)

    print(f"[info] Processing images from {job.images_dir}")
    if not skip_feature_extraction:
        run_trident_feature_extraction(
            trident_repo=trident_repo,
            images_dir=job.images_dir,
            job_dir=job.trident_job_dir,
            patch_encoder=patch_encoder,
            mag=mag,
            patch_size=patch_size,
            overlap=overlap,
            gpu=gpu,
            trident_batch_size=trident_batch_size,
            trident_feat_batch_size=trident_feat_batch_size,
            trident_max_workers=trident_max_workers,
            search_nested=search_nested,
            wsi_ext=wsi_ext,
            reader_type=resolved_reader_type,
            image_paths=image_paths,
            mpp=mpp,
        )

    feature_dir = find_trident_feature_dir(
        job_dir=job.trident_job_dir,
        patch_encoder=patch_encoder,
        mag=mag,
        patch_size=patch_size,
        overlap=overlap,
    )
    if not feature_dir.exists():
        raise FileNotFoundError(
            f"Expected TRIDENT feature directory does not exist: {feature_dir}. "
            "Run without --skip-feature-extraction or point --trident-job-dir at an existing TRIDENT output."
        )

    rows = []
    for image_path in iter_with_progress(image_paths):
        feature_path = feature_dir / f"{image_path.stem}.h5"
        if not feature_path.exists():
            print(f"[warn] Missing feature file for {image_path.name}: {feature_path}", file=sys.stderr)
            continue

        rows.append(
            {
                "original_filename": image_path.name,
                "original_path": str(image_path),
                "label": derive_label(image_path.stem),
                "embedding_path": str(feature_path),
            }
        )

    write_tracking_csv(job.csv_path, rows)
    print(f"Wrote {len(rows)} CSV rows to {job.csv_path}")
    return len(rows)


def discover_images(images_dir: Path, search_nested: bool) -> list[Path]:
    pattern = "**/*" if search_nested else "*"
    return sorted(
        path
        for path in images_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def validate_reader_requirements(reader_type: str | None, mpp: float | None) -> None:
    if reader_type == "image" and mpp is None:
        raise ValueError(
            "--mpp is required when using regular image inputs with TRIDENT's image reader. "
            "For your cores, pass the known value such as --mpp 0.5013."
        )
    if mpp is not None and mpp <= 0:
        raise ValueError("--mpp must be positive")


def iter_with_progress(items: list[Path]) -> Iterable[Path]:
    try:
        from tqdm import tqdm

        return tqdm(items, desc="Building CSV", unit="image")
    except ImportError:
        return items


def run_trident_feature_extraction(
    trident_repo: Path,
    images_dir: Path,
    job_dir: Path,
    patch_encoder: str,
    mag: int,
    patch_size: int,
    overlap: int,
    gpu: int,
    trident_batch_size: int,
    trident_feat_batch_size: int,
    trident_max_workers: int | None,
    search_nested: bool,
    wsi_ext: Sequence[str] | None,
    reader_type: str | None,
    image_paths: Sequence[Path],
    mpp: float | None,
) -> None:
    custom_list_path = None
    if reader_type == "image":
        custom_list_path = write_custom_wsi_list(job_dir=job_dir, image_paths=image_paths, mpp=mpp)

    command = [
        sys.executable,
        str(trident_repo / "run_batch_of_slides.py"),
        "--task",
        "all",
        "--wsi_dir",
        str(images_dir),
        "--job_dir",
        str(job_dir),
        "--patch_encoder",
        patch_encoder,
        "--mag",
        str(mag),
        "--patch_size",
        str(patch_size),
        "--overlap",
        str(overlap),
        "--gpu",
        str(gpu),
        "--batch_size",
        str(trident_batch_size),
        "--feat_batch_size",
        str(trident_feat_batch_size),
    ]
    if trident_max_workers is not None:
        command.extend(["--max_workers", str(trident_max_workers)])
    if search_nested:
        command.append("--search_nested")
    if wsi_ext:
        command.extend(["--wsi_ext", *wsi_ext])
    if reader_type is not None:
        command.extend(["--reader_type", reader_type])
    if custom_list_path is not None:
        command.extend(["--custom_list_of_wsis", str(custom_list_path)])

    print("[info] Running TRIDENT segmentation, tiling, and feature extraction:")
    print("       " + " ".join(command))
    subprocess.run(command, cwd=trident_repo, check=True)


def trident_feature_dir_candidates(
    job_dir: Path,
    patch_encoder: str,
    mag: int,
    patch_size: int,
    overlap: int,
) -> list[Path]:
    mag_values = [str(mag), f"{float(mag):.1f}"]
    candidates = []
    for mag_value in mag_values:
        candidates.append(
            job_dir / f"{mag_value}x_{patch_size}px_{overlap}px_overlap" / f"features_{patch_encoder}"
        )
    return candidates


def find_trident_feature_dir(
    job_dir: Path,
    patch_encoder: str,
    mag: int,
    patch_size: int,
    overlap: int,
) -> Path:
    for candidate in trident_feature_dir_candidates(
        job_dir=job_dir,
        patch_encoder=patch_encoder,
        mag=mag,
        patch_size=patch_size,
        overlap=overlap,
    ):
        if candidate.exists():
            return candidate

    matches = sorted(job_dir.glob(f"*x_{patch_size}px_{overlap}px_overlap/features_{patch_encoder}"))
    if matches:
        return matches[0]

    expected = trident_feature_dir_candidates(
        job_dir=job_dir,
        patch_encoder=patch_encoder,
        mag=mag,
        patch_size=patch_size,
        overlap=overlap,
    )
    raise FileNotFoundError(
        "Expected TRIDENT feature directory does not exist. Checked: "
        + ", ".join(str(path) for path in expected)
    )


def derive_label(stem: str) -> str:
    return stem.split("_", 1)[0]


def resolve_reader_type(image_paths: Sequence[Path], reader_type: str) -> str | None:
    if reader_type != "auto":
        return reader_type

    if not image_paths:
        return None

    raster_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    if all(path.suffix.lower() in raster_exts for path in image_paths):
        return "image"
    return None


def write_custom_wsi_list(job_dir: Path, image_paths: Sequence[Path], mpp: float | None) -> Path:
    if mpp is None:
        raise ValueError("MPP is required to build a TRIDENT custom WSI list for image inputs")

    csv_path = job_dir / "custom_list_of_wsis.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["wsi", "mpp"])
        writer.writeheader()
        for image_path in image_paths:
            writer.writerow({"wsi": image_path.name, "mpp": mpp})
    return csv_path


def write_tracking_csv(csv_path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "original_filename",
        "original_path",
        "label",
        "embedding_path",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
