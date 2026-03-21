#!/usr/bin/env python3
"""Run Macenko and Vahadane stain normalization on core images and summarize OD shifts per TMA."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
CORE_NAME_PATTERN = re.compile(r"(.+)_r\d+_c\d+$")


@dataclass(frozen=True)
class MethodTarget:
    stain_matrix: "np.ndarray"
    max_concentrations: "np.ndarray"


@dataclass(frozen=True)
class NormalizationJob:
    image_path: Path
    macenko_output_path: Path
    vahadane_output_path: Path
    od_threshold: float
    method_sample_size: int
    macenko_alpha: float
    vahadane_lambda: float
    vahadane_max_iter: int
    seed: int
    fit_max_dimension: int | None
    macenko_target: MethodTarget
    vahadane_target: MethodTarget


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize extracted TMA core images with Macenko and Vahadane methods, "
            "write normalized cores into parallel directories, and generate per-TMA "
            "optical-density boxplots."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing extracted core images.")
    parser.add_argument("output_dir", type=Path, help="Parent directory for normalized outputs and QC plots.")
    parser.add_argument(
        "--target-image",
        type=Path,
        default=None,
        help="Optional target image used to fit the reference stain appearance. Defaults to the first sorted core.",
    )
    parser.add_argument(
        "--search-nested",
        action="store_true",
        help="Recursively search for images under --input-dir.",
    )
    parser.add_argument(
        "--method-sample-size",
        type=int,
        default=50000,
        help="Maximum number of tissue pixels sampled per image when fitting stain matrices.",
    )
    parser.add_argument(
        "--macenko-alpha",
        type=float,
        default=1.0,
        help="Percentile for the angular bounds used by Macenko stain estimation.",
    )
    parser.add_argument(
        "--od-threshold",
        type=float,
        default=0.15,
        help="Minimum optical density for treating a pixel as tissue during stain fitting.",
    )
    parser.add_argument(
        "--vahadane-lambda",
        type=float,
        default=0.1,
        help="Dictionary-learning sparsity strength used for Vahadane stain estimation.",
    )
    parser.add_argument(
        "--vahadane-max-iter",
        type=int,
        default=200,
        help="Maximum iterations for Vahadane dictionary learning.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed used for pixel sampling and Vahadane dictionary learning.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for per-core normalization. Use more than 1 to parallelize across CPU cores.",
    )
    parser.add_argument(
        "--fit-max-dimension",
        type=int,
        default=1024,
        help=(
            "Maximum width or height used when estimating stain matrices. "
            "Normalization is still applied to the full-resolution image."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ensure_runtime_dependencies()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    image_paths = discover_images(input_dir, args.search_nested)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found under {input_dir}")

    target_path = (args.target_image.resolve() if args.target_image is not None else image_paths[0])
    if not target_path.is_file():
        raise FileNotFoundError(f"Target image does not exist: {target_path}")

    macenko_dir = output_dir / "macenko"
    vahadane_dir = output_dir / "vahadane"
    macenko_dir.mkdir(parents=True, exist_ok=True)
    vahadane_dir.mkdir(parents=True, exist_ok=True)

    rng = make_rng(args.seed)
    target_rgb = load_rgb_image(target_path)
    target_fit_rgb = resize_for_fitting(target_rgb, max_dimension=args.fit_max_dimension)
    macenko_target = build_macenko_target(
        rgb=target_rgb,
        fit_rgb=target_fit_rgb,
        rng=rng,
        od_threshold=args.od_threshold,
        sample_size=args.method_sample_size,
        alpha=args.macenko_alpha,
    )
    vahadane_target = build_vahadane_target(
        rgb=target_rgb,
        fit_rgb=target_fit_rgb,
        rng=rng,
        od_threshold=args.od_threshold,
        sample_size=args.method_sample_size,
        lambda_value=args.vahadane_lambda,
        max_iter=args.vahadane_max_iter,
        seed=args.seed,
    )

    jobs = [
        NormalizationJob(
            image_path=image_path,
            macenko_output_path=macenko_dir / image_path.name,
            vahadane_output_path=vahadane_dir / image_path.name,
            od_threshold=args.od_threshold,
            method_sample_size=args.method_sample_size,
            macenko_alpha=args.macenko_alpha,
            vahadane_lambda=args.vahadane_lambda,
            vahadane_max_iter=args.vahadane_max_iter,
            seed=args.seed,
            fit_max_dimension=args.fit_max_dimension,
            macenko_target=macenko_target,
            vahadane_target=vahadane_target,
        )
        for image_path in image_paths
    ]

    print(
        f"Found {len(image_paths)} cores. "
        f"Using target {target_path.name}, workers={args.workers}, fit_max_dimension={args.fit_max_dimension}."
    )

    summary_rows = run_jobs(jobs=jobs, workers=args.workers)

    metrics_csv = output_dir / "stain_normalization_metrics.csv"
    write_summary_csv(metrics_csv, summary_rows)
    for method in ("original", "macenko", "vahadane"):
        plot_method_boxplots(
            summary_rows=summary_rows,
            method=method,
            output_path=output_dir / f"{method}_od_boxplots.png",
        )

    print(f"Target image: {target_path}")
    print(f"Wrote Macenko-normalized cores to {macenko_dir}")
    print(f"Wrote Vahadane-normalized cores to {vahadane_dir}")
    print(f"Wrote OD summary CSV to {metrics_csv}")
    return 0


def ensure_runtime_dependencies() -> None:
    missing = []
    for module_name in ("numpy", "PIL", "matplotlib", "sklearn", "scipy"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise RuntimeError(
            "Missing required Python packages: "
            + ", ".join(missing)
            + ". Run this script from an environment such as the TRIDENT venv where these are installed."
        )


def discover_images(input_dir: Path, search_nested: bool) -> list[Path]:
    pattern = "**/*" if search_nested else "*"
    return sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_rgb_image(path: Path) -> "np.ndarray":
    from PIL import Image
    import numpy as np

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def save_rgb_image(path: Path, rgb: "np.ndarray") -> None:
    from PIL import Image

    Image.fromarray(rgb, mode="RGB").save(path)


def resize_for_fitting(rgb: "np.ndarray", max_dimension: int | None) -> "np.ndarray":
    from PIL import Image

    if max_dimension is None or max_dimension <= 0:
        return rgb

    height, width = rgb.shape[:2]
    largest_dimension = max(height, width)
    if largest_dimension <= max_dimension:
        return rgb

    scale = max_dimension / float(largest_dimension)
    resized = Image.fromarray(rgb, mode="RGB").resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        resample=Image.Resampling.BILINEAR,
    )
    return load_rgb_from_pil(resized)


def load_rgb_from_pil(image) -> "np.ndarray":
    import numpy as np

    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def derive_tma_name(stem: str) -> str:
    match = CORE_NAME_PATTERN.match(stem)
    return match.group(1) if match else stem


def make_rng(seed: int):
    import numpy as np

    return np.random.default_rng(seed)


def rgb_to_od(rgb: "np.ndarray") -> "np.ndarray":
    import numpy as np

    rgb = np.asarray(rgb, dtype=np.float32)
    return -np.log((rgb + 1.0) / 255.0)


def od_to_rgb(od: "np.ndarray") -> "np.ndarray":
    import numpy as np

    rgb = 255.0 * np.exp(-np.asarray(od, dtype=np.float32))
    return np.clip(rgb, 0, 255).astype(np.uint8)


def make_tissue_mask(rgb: "np.ndarray", od_threshold: float) -> "np.ndarray":
    import numpy as np

    od = rgb_to_od(rgb)
    near_white = np.all(rgb > 235, axis=2)
    od_signal = np.any(od > od_threshold, axis=2)
    return ~near_white & od_signal


def sample_tissue_pixels(
    rgb: "np.ndarray",
    tissue_mask: "np.ndarray",
    rng,
    sample_size: int,
) -> "np.ndarray":
    import numpy as np

    tissue_od = rgb_to_od(rgb)[tissue_mask]
    if len(tissue_od) == 0:
        raise ValueError("No tissue pixels found for stain estimation")
    if len(tissue_od) <= sample_size:
        return tissue_od
    indices = rng.choice(len(tissue_od), size=sample_size, replace=False)
    return tissue_od[indices]


def normalize_rows(matrix: "np.ndarray") -> "np.ndarray":
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return matrix / norms


def sort_stain_matrix(stain_matrix: "np.ndarray") -> "np.ndarray":
    import numpy as np

    order = np.argsort(-stain_matrix[:, 0])
    matrix = stain_matrix[order]
    if matrix[0, 2] < matrix[1, 2]:
        matrix = matrix[::-1]
    return normalize_rows(matrix)


def fit_macenko_stain_matrix(
    rgb: "np.ndarray",
    rng,
    od_threshold: float,
    sample_size: int,
    alpha: float,
) -> "np.ndarray":
    import numpy as np

    tissue_mask = make_tissue_mask(rgb, od_threshold=od_threshold)
    sampled_od = sample_tissue_pixels(rgb, tissue_mask, rng=rng, sample_size=sample_size)
    sampled_od = sampled_od[np.any(sampled_od > od_threshold, axis=1)]
    if len(sampled_od) < 10:
        raise ValueError("Too few tissue pixels for Macenko estimation")

    cov = np.cov(sampled_od, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    principal = eigenvectors[:, :2]
    projections = sampled_od @ principal
    angles = np.arctan2(projections[:, 1], projections[:, 0])
    min_angle = np.percentile(angles, alpha)
    max_angle = np.percentile(angles, 100.0 - alpha)
    vector_1 = principal @ np.array([math.cos(min_angle), math.sin(min_angle)], dtype=np.float32)
    vector_2 = principal @ np.array([math.cos(max_angle), math.sin(max_angle)], dtype=np.float32)
    return sort_stain_matrix(np.vstack([vector_1, vector_2]))


def fit_vahadane_stain_matrix(
    rgb: "np.ndarray",
    rng,
    od_threshold: float,
    sample_size: int,
    lambda_value: float,
    max_iter: int,
    seed: int,
) -> "np.ndarray":
    from sklearn.decomposition import DictionaryLearning
    from sklearn.exceptions import ConvergenceWarning
    import numpy as np

    tissue_mask = make_tissue_mask(rgb, od_threshold=od_threshold)
    sampled_od = sample_tissue_pixels(rgb, tissue_mask, rng=rng, sample_size=sample_size)
    sampled_od = sampled_od[np.any(sampled_od > od_threshold, axis=1)]
    if len(sampled_od) < 10:
        raise ValueError("Too few tissue pixels for Vahadane estimation")

    learner = DictionaryLearning(
        n_components=2,
        alpha=lambda_value,
        max_iter=max_iter,
        fit_algorithm="cd",
        transform_algorithm="lasso_cd",
        transform_alpha=lambda_value,
        positive_code=True,
        positive_dict=True,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        learner.fit(sampled_od)
    return sort_stain_matrix(np.asarray(learner.components_, dtype=np.float32))


def solve_concentrations(od: "np.ndarray", stain_matrix: "np.ndarray") -> "np.ndarray":
    import numpy as np

    coeffs, _, _, _ = np.linalg.lstsq(stain_matrix.T, od.T, rcond=None)
    coeffs = np.clip(coeffs.T, 0.0, None)
    return coeffs


def build_target(
    rgb: "np.ndarray",
    stain_matrix: "np.ndarray",
    od_threshold: float,
) -> MethodTarget:
    import numpy as np

    tissue_mask = make_tissue_mask(rgb, od_threshold=od_threshold)
    od = rgb_to_od(rgb)[tissue_mask]
    concentrations = solve_concentrations(od, stain_matrix)
    max_concentrations = np.percentile(concentrations, 99, axis=0)
    max_concentrations = np.clip(max_concentrations, 1e-6, None)
    return MethodTarget(stain_matrix=stain_matrix, max_concentrations=max_concentrations)


def build_macenko_target(
    rgb: "np.ndarray",
    fit_rgb: "np.ndarray",
    rng,
    od_threshold: float,
    sample_size: int,
    alpha: float,
) -> MethodTarget:
    stain_matrix = fit_macenko_stain_matrix(
        rgb=fit_rgb,
        rng=rng,
        od_threshold=od_threshold,
        sample_size=sample_size,
        alpha=alpha,
    )
    return build_target(rgb=rgb, stain_matrix=stain_matrix, od_threshold=od_threshold)


def build_vahadane_target(
    rgb: "np.ndarray",
    fit_rgb: "np.ndarray",
    rng,
    od_threshold: float,
    sample_size: int,
    lambda_value: float,
    max_iter: int,
    seed: int,
) -> MethodTarget:
    stain_matrix = fit_vahadane_stain_matrix(
        rgb=fit_rgb,
        rng=rng,
        od_threshold=od_threshold,
        sample_size=sample_size,
        lambda_value=lambda_value,
        max_iter=max_iter,
        seed=seed,
    )
    return build_target(rgb=rgb, stain_matrix=stain_matrix, od_threshold=od_threshold)


def run_jobs(jobs: Sequence[NormalizationJob], workers: int) -> list[dict[str, object]]:
    if workers <= 1:
        results = []
        progress = make_progress(total=len(jobs))
        for index, job in enumerate(jobs, start=1):
            update_progress(progress, index=index, total=len(jobs), filename=job.image_path.name)
            results.extend(process_single_job(job))
        close_progress(progress)
        return results

    results = []
    progress = make_progress(total=len(jobs))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_job = {executor.submit(process_single_job, job): job for job in jobs}
        completed = 0
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            completed += 1
            update_progress(progress, index=completed, total=len(jobs), filename=job.image_path.name)
            results.extend(future.result())
    close_progress(progress)
    return results


def make_progress(total: int):
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc="Normalizing cores", unit="core")
    except ImportError:
        return None


def update_progress(progress, index: int, total: int, filename: str) -> None:
    if progress is None:
        print(f"[{index}/{total}] {filename}")
        return
    progress.update(1)
    progress.set_postfix_str(filename)


def close_progress(progress) -> None:
    if progress is not None:
        progress.close()


def process_single_job(job: NormalizationJob) -> list[dict[str, object]]:
    seed_offset = abs(hash(job.image_path.stem)) % (2**32)
    rng = make_rng(job.seed + seed_offset)
    image_rgb = load_rgb_image(job.image_path)
    fit_rgb = resize_for_fitting(image_rgb, max_dimension=job.fit_max_dimension)
    tissue_mask = make_tissue_mask(image_rgb, od_threshold=job.od_threshold)

    try:
        macenko_rgb = normalize_macenko(
            rgb=image_rgb,
            fit_rgb=fit_rgb,
            target=job.macenko_target,
            rng=rng,
            od_threshold=job.od_threshold,
            sample_size=job.method_sample_size,
            alpha=job.macenko_alpha,
        )
    except Exception as exc:
        print(f"[warn] Macenko normalization failed for {job.image_path.name}: {exc}", file=sys.stderr)
        macenko_rgb = image_rgb.copy()

    try:
        vahadane_rgb = normalize_vahadane(
            rgb=image_rgb,
            fit_rgb=fit_rgb,
            target=job.vahadane_target,
            rng=rng,
            od_threshold=job.od_threshold,
            sample_size=job.method_sample_size,
            lambda_value=job.vahadane_lambda,
            max_iter=job.vahadane_max_iter,
            seed=job.seed + seed_offset,
        )
    except Exception as exc:
        print(f"[warn] Vahadane normalization failed for {job.image_path.name}: {exc}", file=sys.stderr)
        vahadane_rgb = image_rgb.copy()

    save_rgb_image(job.macenko_output_path, macenko_rgb)
    save_rgb_image(job.vahadane_output_path, vahadane_rgb)

    tma_name = derive_tma_name(job.image_path.stem)
    return [
        build_summary_row(
            image_path=job.image_path,
            method="original",
            tma_name=tma_name,
            rgb=image_rgb,
            tissue_mask=tissue_mask,
        ),
        build_summary_row(
            image_path=job.image_path,
            method="macenko",
            tma_name=tma_name,
            rgb=macenko_rgb,
            tissue_mask=make_tissue_mask(macenko_rgb, od_threshold=job.od_threshold),
        ),
        build_summary_row(
            image_path=job.image_path,
            method="vahadane",
            tma_name=tma_name,
            rgb=vahadane_rgb,
            tissue_mask=make_tissue_mask(vahadane_rgb, od_threshold=job.od_threshold),
        ),
    ]


def normalize_with_target(
    rgb: "np.ndarray",
    source_matrix: "np.ndarray",
    target: MethodTarget,
    od_threshold: float,
) -> "np.ndarray":
    import numpy as np

    tissue_mask = make_tissue_mask(rgb, od_threshold=od_threshold)
    od = rgb_to_od(rgb)
    flat_od = od.reshape(-1, 3)
    concentrations = solve_concentrations(flat_od, source_matrix)
    tissue_concentrations = concentrations[tissue_mask.reshape(-1)]

    if len(tissue_concentrations) == 0:
        return rgb.copy()

    source_max = np.percentile(tissue_concentrations, 99, axis=0)
    source_max = np.clip(source_max, 1e-6, None)
    scaled = concentrations * (target.max_concentrations / source_max)
    normalized_od = scaled @ target.stain_matrix
    normalized_rgb = od_to_rgb(normalized_od).reshape(rgb.shape)
    normalized_rgb[~tissue_mask] = 255
    return normalized_rgb


def normalize_macenko(
    rgb: "np.ndarray",
    fit_rgb: "np.ndarray",
    target: MethodTarget,
    rng,
    od_threshold: float,
    sample_size: int,
    alpha: float,
) -> "np.ndarray":
    source_matrix = fit_macenko_stain_matrix(
        rgb=fit_rgb,
        rng=rng,
        od_threshold=od_threshold,
        sample_size=sample_size,
        alpha=alpha,
    )
    return normalize_with_target(rgb=rgb, source_matrix=source_matrix, target=target, od_threshold=od_threshold)


def normalize_vahadane(
    rgb: "np.ndarray",
    fit_rgb: "np.ndarray",
    target: MethodTarget,
    rng,
    od_threshold: float,
    sample_size: int,
    lambda_value: float,
    max_iter: int,
    seed: int,
) -> "np.ndarray":
    source_matrix = fit_vahadane_stain_matrix(
        rgb=fit_rgb,
        rng=rng,
        od_threshold=od_threshold,
        sample_size=sample_size,
        lambda_value=lambda_value,
        max_iter=max_iter,
        seed=seed,
    )
    return normalize_with_target(rgb=rgb, source_matrix=source_matrix, target=target, od_threshold=od_threshold)


def build_summary_row(
    image_path: Path,
    method: str,
    tma_name: str,
    rgb: "np.ndarray",
    tissue_mask: "np.ndarray",
) -> dict[str, object]:
    import numpy as np

    tissue_fraction = float(np.mean(tissue_mask))
    if np.any(tissue_mask):
        od_pixels = rgb_to_od(rgb)[tissue_mask]
        mean_od = od_pixels.mean(axis=0)
        total_mean_od = float(od_pixels.mean())
    else:
        mean_od = np.array([math.nan, math.nan, math.nan], dtype=np.float32)
        total_mean_od = math.nan

    return {
        "filename": image_path.name,
        "method": method,
        "tma": tma_name,
        "mean_od_r": float(mean_od[0]),
        "mean_od_g": float(mean_od[1]),
        "mean_od_b": float(mean_od[2]),
        "mean_od_total": total_mean_od,
        "tissue_fraction": tissue_fraction,
    }


def write_summary_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = [
        "filename",
        "method",
        "tma",
        "mean_od_r",
        "mean_od_g",
        "mean_od_b",
        "mean_od_total",
        "tissue_fraction",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_method_boxplots(
    summary_rows: Sequence[dict[str, object]],
    method: str,
    output_path: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import matplotlib.pyplot as plt
    import numpy as np

    method_rows = [row for row in summary_rows if row["method"] == method]
    if not method_rows:
        return

    tma_names = sorted({str(row["tma"]) for row in method_rows})
    metrics = [
        ("mean_od_total", "Mean OD (All Channels)"),
        ("mean_od_r", "Mean OD - R"),
        ("mean_od_g", "Mean OD - G"),
        ("mean_od_b", "Mean OD - B"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(12, len(tma_names) * 0.65), 14), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric_key, title) in zip(axes, metrics):
        data = []
        labels = []
        for tma_name in tma_names:
            values = [
                float(row[metric_key])
                for row in method_rows
                if row["tma"] == tma_name and not math.isnan(float(row[metric_key]))
            ]
            if values:
                data.append(values)
                labels.append(tma_name)

        if not data:
            ax.set_visible(False)
            continue

        ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch in ax.artists:
            patch.set_facecolor("#87b5ff")
        ax.set_ylabel("OD")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=60, ha="right")

    fig.suptitle(f"{method.capitalize()} optical density distributions per TMA", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
