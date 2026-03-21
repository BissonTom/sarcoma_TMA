#!/usr/bin/env python3
"""Create ABMIL attention galleries split into correct and incorrect predictions."""

from __future__ import annotations

import argparse
import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn


@dataclass(frozen=True)
class PredictionRecord:
    fold: int
    original_filename: str
    original_path: Path
    label: str
    predicted_label: str
    embedding_path: Path


class GatedAttention(nn.Module):
    def __init__(self, in_dim: int, attention_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention_v = nn.Sequential(
            nn.Linear(in_dim, attention_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attention_u = nn.Sequential(
            nn.Linear(in_dim, attention_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attention_weights = nn.Linear(attention_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a_v = self.attention_v(x)
        a_u = self.attention_u(x)
        logits = self.attention_weights(a_v * a_u).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        attn = torch.softmax(logits, dim=1)
        pooled = torch.sum(x * attn.unsqueeze(-1), dim=1)
        return pooled, attn


class ABMILClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        attention_dim: int,
        n_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.instance_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention = GatedAttention(hidden_dim, attention_dim, dropout)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.instance_proj(features)
        bag_repr, attn = self.attention(projected, mask)
        logits = self.classifier(bag_repr)
        return logits, attn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build ABMIL attention galleries from out-of-fold predictions, with separate "
            "correct and incorrect panels for each true subtype."
        )
    )
    parser.add_argument("run_dir", type=Path, help="ABMIL run directory containing fold checkpoints and oof_predictions.csv.")
    parser.add_argument("output_dir", type=Path, help="Directory where gallery PNGs and metadata CSVs will be written.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top-attention patches to show per case.")
    parser.add_argument(
        "--cases-per-panel",
        type=int,
        default=10,
        help="Maximum number of randomly selected cases per subtype and correctness group.",
    )
    parser.add_argument("--seed", type=int, default=17, help="Random seed for case sampling.")
    parser.add_argument("--patch-size", type=int, default=None, help="Patch size in pixels. Auto-detected if omitted.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--font-size", type=int, default=16, help="Font size for gallery labels.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    predictions_csv = run_dir / "oof_predictions.csv"

    if not predictions_csv.is_file():
        raise FileNotFoundError(f"Missing OOF predictions file: {predictions_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_prediction_records(predictions_csv)
    if not records:
        raise ValueError(f"No rows found in {predictions_csv}")

    patch_size = args.patch_size or infer_patch_size(records[0].embedding_path)
    rng = random.Random(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    fold_models = load_fold_models(run_dir=run_dir, device=device)

    subtypes = sorted({record.label for record in records})
    for subtype in subtypes:
        subtype_records = [record for record in records if record.label == subtype]
        for group_name, is_correct in (("correct", True), ("incorrect", False)):
            group_records = [
                record for record in subtype_records if (record.predicted_label == record.label) == is_correct
            ]
            if not group_records:
                continue

            selected_records = list(group_records)
            rng.shuffle(selected_records)
            selected_records = selected_records[: args.cases_per_panel]

            cases = []
            metadata_rows = []
            for record in selected_records:
                model = fold_models[record.fold]
                patch_entries = extract_top_attention_patches(
                    record=record,
                    model=model,
                    device=device,
                    patch_size=patch_size,
                    top_k=args.top_k,
                )
                if not patch_entries:
                    continue
                cases.append((record, patch_entries))
                for rank, patch_entry in enumerate(patch_entries, start=1):
                    metadata_rows.append(
                        {
                            "true_label": record.label,
                            "predicted_label": record.predicted_label,
                            "group": group_name,
                            "fold": record.fold,
                            "original_filename": record.original_filename,
                            "original_path": str(record.original_path),
                            "embedding_path": str(record.embedding_path),
                            "patch_rank": rank,
                            "attention_score": patch_entry["attention_score"],
                            "x": patch_entry["x"],
                            "y": patch_entry["y"],
                            "width": patch_entry["width"],
                            "height": patch_entry["height"],
                        }
                    )

            if not cases:
                continue

            gallery_path = output_dir / f"{sanitize_name(subtype)}_{group_name}_gallery.png"
            metadata_path = output_dir / f"{sanitize_name(subtype)}_{group_name}_gallery.csv"
            render_gallery(
                cases=cases,
                output_path=gallery_path,
                top_k=args.top_k,
                font_size=args.font_size,
                title=f"{subtype} ({group_name})",
            )
            write_metadata_csv(metadata_path, metadata_rows)
            print(f"Wrote {gallery_path}")

    return 0


def load_prediction_records(predictions_csv: Path) -> list[PredictionRecord]:
    records = []
    with predictions_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"fold", "original_filename", "original_path", "label", "predicted_label", "embedding_path"}
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {predictions_csv}: {', '.join(sorted(missing))}")
        for row in reader:
            records.append(
                PredictionRecord(
                    fold=int(row["fold"]),
                    original_filename=row["original_filename"],
                    original_path=Path(row["original_path"]),
                    label=row["label"],
                    predicted_label=row["predicted_label"],
                    embedding_path=Path(row["embedding_path"]),
                )
            )
    return records


def infer_patch_size(embedding_path: Path) -> int:
    match = re.search(r"_(\d+)px_", str(embedding_path))
    if not match:
        raise ValueError(f"Could not infer patch size from embedding path: {embedding_path}")
    return int(match.group(1))


def load_fold_models(run_dir: Path, device: torch.device) -> dict[int, ABMILClassifier]:
    models = {}
    for fold_dir in sorted(run_dir.glob("fold_*")):
        checkpoint_path = fold_dir / "best_model.pt"
        if not checkpoint_path.is_file():
            continue
        checkpoint = torch.load(checkpoint_path, map_location=device)
        fold = int(fold_dir.name.split("_")[-1])
        model = ABMILClassifier(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            attention_dim=int(checkpoint["attention_dim"]),
            n_classes=len(checkpoint["label_names"]),
            dropout=float(checkpoint["dropout"]),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models[fold] = model
    if not models:
        raise FileNotFoundError(f"No fold checkpoints found in {run_dir}")
    return models


def extract_top_attention_patches(
    record: PredictionRecord,
    model: ABMILClassifier,
    device: torch.device,
    patch_size: int,
    top_k: int,
) -> list[dict[str, object]]:
    with h5py.File(record.embedding_path, "r") as handle:
        features = np.asarray(handle["features"], dtype=np.float32)
        coords = np.asarray(handle["coords"])

    if len(features) == 0:
        return []

    feature_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    mask = torch.ones(1, features.shape[0], dtype=torch.bool, device=device)
    with torch.inference_mode():
        _, attn = model(feature_tensor, mask)
    attn_scores = attn.squeeze(0).detach().cpu().numpy()

    top_indices = np.argsort(attn_scores)[::-1][:top_k]
    image = Image.open(record.original_path).convert("RGB")

    patches = []
    for index in top_indices:
        x, y = int(coords[index][0]), int(coords[index][1])
        crop = image.crop((x, y, x + patch_size, y + patch_size))
        patches.append(
            {
                "image": crop,
                "attention_score": float(attn_scores[index]),
                "x": x,
                "y": y,
                "width": patch_size,
                "height": patch_size,
            }
        )
    return patches


def render_gallery(
    cases: Sequence[tuple[PredictionRecord, Sequence[dict[str, object]]]],
    output_path: Path,
    top_k: int,
    font_size: int,
    title: str,
) -> None:
    patch_width = max(case[1][0]["image"].width for case in cases)
    patch_height = max(case[1][0]["image"].height for case in cases)
    header_height = int(font_size * 4.8)
    gutter_x = 24
    gutter_y = 20
    title_height = int(font_size * 2.5)

    canvas_width = gutter_x + len(cases) * (patch_width + gutter_x)
    canvas_height = title_height + header_height + gutter_y + top_k * (patch_height + gutter_y) + gutter_y
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((gutter_x, 10), title, fill=(0, 0, 0))

    for case_index, (record, patches) in enumerate(cases):
        x0 = gutter_x + case_index * (patch_width + gutter_x)
        header_y = title_height
        draw.text((x0, header_y), truncate_text(record.original_filename, 18), fill=(0, 0, 0))
        draw.text((x0, header_y + int(font_size * 1.3)), f"true: {record.label}", fill=(0, 0, 0))
        draw.text((x0, header_y + int(font_size * 2.6)), f"pred: {record.predicted_label}", fill=(0, 0, 0))

        for rank in range(top_k):
            patch_y = title_height + header_height + rank * (patch_height + gutter_y)
            if rank < len(patches):
                patch_image = patches[rank]["image"]
                canvas.paste(patch_image, (x0, patch_y))
                draw.rectangle(
                    [(x0, patch_y), (x0 + patch_image.width - 1, patch_y + patch_image.height - 1)],
                    outline=(40, 40, 40),
                    width=1,
                )
                draw.text(
                    (x0 + 4, patch_y + 4),
                    f"{rank + 1}: {patches[rank]['attention_score']:.3f}",
                    fill=(255, 255, 255),
                )

    canvas.save(output_path)


def write_metadata_csv(output_csv: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sanitize_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_").lower()


def truncate_text(value: str, max_len: int) -> str:
    return value if len(value) <= max_len else value[: max_len - 1] + "…"


if __name__ == "__main__":
    raise SystemExit(main())
