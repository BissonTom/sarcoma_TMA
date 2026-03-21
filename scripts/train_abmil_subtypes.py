#!/usr/bin/env python3
"""Train an ABMIL classifier on CONCH embeddings for sarcoma subtype prediction."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import h5py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, auc, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.utils.data import DataLoader, Dataset


LABEL_ALIASES = {
    "wd-dd": "wd-ddlps",
}

EXCLUDED_LABELS = {
    "ris",
}


@dataclass(frozen=True)
class BagRecord:
    original_filename: str
    original_path: str
    label_name: str
    label_index: int
    embedding_path: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train an attention-based MIL model to predict sarcoma subtypes from "
            "TRIDENT CONCH embedding bags with stratified cross-validation."
        )
    )
    parser.add_argument("input_csv", type=Path, help="Tracking CSV produced by compute_conch_embeddings.py.")
    parser.add_argument("output_dir", type=Path, help="Directory for models, predictions, and metrics.")
    parser.add_argument("--folds", type=int, default=3, help="Number of stratified cross-validation folds.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum training epochs per fold.")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience in epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of bags per optimization step.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension for ABMIL.")
    parser.add_argument("--attention-dim", type=int, default=128, help="Attention dimension for ABMIL.")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=17, help="Random seed.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for confidence intervals on overall OOF metrics.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Number of label permutations for overall OOF metric p-values.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()

    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")

    set_seed(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    records, label_names = load_records(input_csv)
    labels = np.asarray([record.label_index for record in records], dtype=np.int64)
    n_classes = len(label_names)
    if n_classes < 2:
        raise ValueError("Need at least 2 classes for classification")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    input_dim = infer_feature_dim(Path(records[0].embedding_path))

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_summaries = []
    oof_rows: list[dict[str, object]] = []
    oof_probs = np.zeros((len(records), n_classes), dtype=np.float32)
    oof_preds = np.full(len(records), -1, dtype=np.int64)

    for fold_idx, (train_indices, val_indices) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_records = [records[index] for index in train_indices]
        val_records = [records[index] for index in val_indices]
        train_dataset = EmbeddingBagDataset(train_records)
        val_dataset = EmbeddingBagDataset(val_records)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_bags,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_bags,
            pin_memory=device.type == "cuda",
        )

        model = ABMILClassifier(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            attention_dim=args.attention_dim,
            n_classes=n_classes,
            dropout=args.dropout,
        ).to(device)

        class_weights = make_class_weights(labels[train_indices], n_classes=n_classes, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_state = None
        best_metric = -math.inf
        best_epoch = 0
        patience_counter = 0
        history = []

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_probs, val_labels = predict_dataset(model, val_loader, criterion, device)
            val_preds = val_probs.argmax(axis=1)
            val_metrics = compute_metrics(y_true=val_labels, y_pred=val_preds, y_prob=val_probs)
            val_metrics["loss"] = val_loss
            val_metrics["epoch"] = epoch
            val_metrics["train_loss"] = train_loss
            history.append(val_metrics)

            monitor = val_metrics["balanced_accuracy"]
            if monitor > best_metric:
                best_metric = monitor
                best_epoch = epoch
                patience_counter = 0
                best_state = {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "metrics": val_metrics,
                    "label_names": label_names,
                    "input_dim": input_dim,
                    "hidden_dim": args.hidden_dim,
                    "attention_dim": args.attention_dim,
                    "dropout": args.dropout,
                }
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    break

        if best_state is None:
            raise RuntimeError(f"Fold {fold_idx}: training did not produce a checkpoint")

        checkpoint_path = fold_dir / "best_model.pt"
        torch.save(best_state, checkpoint_path)
        model.load_state_dict(best_state["model_state_dict"])

        _, val_probs, val_labels = predict_dataset(model, val_loader, criterion, device)
        val_preds = val_probs.argmax(axis=1)
        fold_metrics = compute_metrics(y_true=val_labels, y_pred=val_preds, y_prob=val_probs)
        fold_metrics["fold"] = fold_idx
        fold_metrics["best_epoch"] = best_epoch
        fold_summaries.append(fold_metrics)
        write_confusion_artifacts(
            output_dir=fold_dir,
            y_true=val_labels,
            y_pred=val_preds,
            label_names=label_names,
            prefix="confusion_matrix",
            title=f"Fold {fold_idx} Confusion Matrix",
        )

        history_path = fold_dir / "history.json"
        history_path.write_text(json.dumps(history, indent=2))

        for local_index, global_index in enumerate(val_indices):
            record = records[global_index]
            oof_probs[global_index] = val_probs[local_index]
            oof_preds[global_index] = int(val_preds[local_index])
            oof_rows.append(
                {
                    "fold": fold_idx,
                    "original_filename": record.original_filename,
                    "original_path": record.original_path,
                    "label": record.label_name,
                    "predicted_label": label_names[int(val_preds[local_index])],
                    "embedding_path": record.embedding_path,
                    **{
                        f"prob_{label_name}": float(val_probs[local_index, class_index])
                        for class_index, label_name in enumerate(label_names)
                    },
                }
            )

        print(
            f"[fold {fold_idx}] best_epoch={best_epoch} "
            f"roc_auc={fold_metrics['roc_auc_macro_ovr']:.4f} "
            f"bal_acc={fold_metrics['balanced_accuracy']:.4f} "
            f"acc={fold_metrics['accuracy']:.4f} "
            f"f1={fold_metrics['f1_macro']:.4f}"
        )

    overall_metrics = compute_metrics(y_true=labels, y_pred=oof_preds, y_prob=oof_probs)
    significance_summary = compute_significance_summary(
        y_true=labels,
        y_pred=oof_preds,
        y_prob=oof_probs,
        n_classes=n_classes,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
    )
    summary = {
        "n_samples": len(records),
        "n_classes": n_classes,
        "label_names": label_names,
        "folds": fold_summaries,
        "overall_oof": overall_metrics,
        "overall_oof_significance": significance_summary,
    }

    write_predictions_csv(output_dir / "oof_predictions.csv", oof_rows, label_names)
    write_confusion_artifacts(
        output_dir=output_dir,
        y_true=labels,
        y_pred=oof_preds,
        label_names=label_names,
        prefix="confusion_matrix_oof",
        title="Out-of-Fold Confusion Matrix",
    )
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    plot_roc_curves(
        y_true=labels,
        y_prob=oof_probs,
        label_names=label_names,
        output_path=output_dir / "roc_curves_oof.png",
    )
    plot_fold_metric_summary(
        fold_summaries=fold_summaries,
        output_path=output_dir / "fold_metric_summary.png",
    )
    plot_class_support(
        labels=labels,
        label_names=label_names,
        output_path=output_dir / "class_support.png",
    )
    plot_summary_figure(
        y_true=labels,
        y_pred=oof_preds,
        y_prob=oof_probs,
        label_names=label_names,
        fold_summaries=fold_summaries,
        significance_summary=significance_summary,
        output_path=output_dir / "summary_figure.png",
    )
    plot_significance_panel(
        fold_summaries=fold_summaries,
        significance_summary=significance_summary,
        output_path=output_dir / "significance_summary.png",
    )

    print("[overall]")
    print(
        f"roc_auc={overall_metrics['roc_auc_macro_ovr']:.4f} "
        f"bal_acc={overall_metrics['balanced_accuracy']:.4f} "
        f"acc={overall_metrics['accuracy']:.4f} "
        f"f1={overall_metrics['f1_macro']:.4f}"
    )
    print(f"Wrote outputs to {output_dir}")
    return 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_records(input_csv: Path) -> tuple[list[BagRecord], list[str]]:
    rows = []
    with input_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"original_filename", "original_path", "label", "embedding_path"}
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {input_csv}: {', '.join(sorted(missing))}")
        for row in reader:
            embedding_path = Path(row["embedding_path"])
            normalized_label = normalize_label(row["label"])
            if embedding_path.is_file() and not should_exclude_label(normalized_label):
                rows.append(row)

    label_names = sorted({normalize_label(row["label"]) for row in rows})
    label_to_index = {label: index for index, label in enumerate(label_names)}
    records = [
        BagRecord(
            original_filename=row["original_filename"],
            original_path=row["original_path"],
            label_name=normalize_label(row["label"]),
            label_index=label_to_index[normalize_label(row["label"])],
            embedding_path=row["embedding_path"],
        )
        for row in rows
    ]
    return records, label_names


def infer_feature_dim(embedding_path: Path) -> int:
    with h5py.File(embedding_path, "r") as handle:
        features = handle["features"]
        return int(features.shape[1])


def normalize_label(label: str) -> str:
    return LABEL_ALIASES.get(label, label)


def should_exclude_label(label: str) -> bool:
    return label in EXCLUDED_LABELS


class EmbeddingBagDataset(Dataset):
    def __init__(self, records: Sequence[BagRecord]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with h5py.File(record.embedding_path, "r") as handle:
            features = np.asarray(handle["features"], dtype=np.float32)
        return {
            "features": torch.from_numpy(features),
            "label": record.label_index,
            "record": record,
        }


def collate_bags(batch):
    bag_sizes = [item["features"].shape[0] for item in batch]
    feat_dim = batch[0]["features"].shape[1]
    max_len = max(bag_sizes)
    batch_size = len(batch)

    features = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    records = [item["record"] for item in batch]

    for idx, item in enumerate(batch):
        length = item["features"].shape[0]
        features[idx, :length] = item["features"]
        mask[idx, :length] = True

    return {"features": features, "mask": mask, "labels": labels, "records": records}


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


def make_class_weights(train_labels: np.ndarray, n_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(train_labels, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(model, loader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    losses = []
    for batch in loader:
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(features, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else math.nan


@torch.no_grad()
def predict_dataset(model, loader, criterion, device: torch.device):
    model.eval()
    all_probs = []
    all_labels = []
    losses = []
    for batch in loader:
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)
        logits, _ = model(features, mask)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)

        losses.append(float(loss.item()))
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return float(np.mean(losses)), probs, labels


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    try:
        metrics["roc_auc_macro_ovr"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["roc_auc_macro_ovr"] = math.nan
    return metrics


def compute_significance_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int,
    seed: int,
    n_bootstrap: int,
    n_permutations: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    metric_fns: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], float]] = {
        "roc_auc_macro_ovr": lambda yt, yp, ypr: compute_metrics(yt, yp, ypr)["roc_auc_macro_ovr"],
        "balanced_accuracy": lambda yt, yp, ypr: compute_metrics(yt, yp, ypr)["balanced_accuracy"],
        "accuracy": lambda yt, yp, ypr: compute_metrics(yt, yp, ypr)["accuracy"],
        "f1_macro": lambda yt, yp, ypr: compute_metrics(yt, yp, ypr)["f1_macro"],
    }

    summary = {}
    for metric_name, metric_fn in metric_fns.items():
        observed = metric_fn(y_true, y_pred, y_prob)
        ci_low, ci_high = bootstrap_confidence_interval(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            metric_fn=metric_fn,
            rng=rng,
            n_bootstrap=n_bootstrap,
        )
        p_value = permutation_p_value(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            metric_fn=metric_fn,
            observed=observed,
            rng=rng,
            n_permutations=n_permutations,
        )
        summary[metric_name] = {
            "value": float(observed),
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high),
            "permutation_p_value": float(p_value),
        }
    summary["chance_accuracy"] = {"value": float(1.0 / n_classes)}
    return summary


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float]:
    values = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        value = metric_fn(y_true[indices], y_pred[indices], y_prob[indices])
        if not math.isnan(value):
            values.append(value)
    if not values:
        return math.nan, math.nan
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def permutation_p_value(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    observed: float,
    rng: np.random.Generator,
    n_permutations: int,
) -> float:
    if math.isnan(observed):
        return math.nan

    null_values = []
    for _ in range(n_permutations):
        permuted = rng.permutation(y_true)
        value = metric_fn(permuted, y_pred, y_prob)
        if not math.isnan(value):
            null_values.append(value)
    if not null_values:
        return math.nan

    null_values = np.asarray(null_values)
    return float((1 + np.sum(null_values >= observed)) / (len(null_values) + 1))


def write_confusion_artifacts(
    output_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Sequence[str],
    prefix: str,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_names)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums != 0)

    write_confusion_csv(output_dir / f"{prefix}.csv", cm, label_names)
    write_confusion_csv(output_dir / f"{prefix}_normalized.csv", cm_normalized, label_names)
    plot_confusion_heatmap(
        matrix=cm,
        label_names=label_names,
        output_path=output_dir / f"{prefix}.png",
        title=title,
        normalized=False,
    )
    plot_confusion_heatmap(
        matrix=cm_normalized,
        label_names=label_names,
        output_path=output_dir / f"{prefix}_normalized.png",
        title=f"{title} (Row-Normalized)",
        normalized=True,
    )


def write_confusion_csv(path: Path, matrix: np.ndarray, label_names: Sequence[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *label_names])
        for label_name, row in zip(label_names, matrix):
            writer.writerow([label_name, *row.tolist()])


def plot_confusion_heatmap(
    matrix: np.ndarray,
    label_names: Sequence[str],
    output_path: Path,
    title: str,
    normalized: bool,
) -> None:
    import matplotlib.pyplot as plt

    fig_size = max(8, 0.6 * len(label_names))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Proportion" if normalized else "Count", rotation=270, labelpad=15)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = f"{value:.2f}" if normalized else f"{int(value)}"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: Sequence[str],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    y_true_bin = label_binarize(y_true, classes=np.arange(len(label_names)))
    if y_true_bin.shape[1] != len(label_names):
        padded = np.zeros((len(y_true), len(label_names)), dtype=np.int64)
        padded[:, : y_true_bin.shape[1]] = y_true_bin
        y_true_bin = padded

    fig, ax = plt.subplots(figsize=(8, 7))
    all_fpr = np.linspace(0.0, 1.0, 200)
    mean_tpr = np.zeros_like(all_fpr)
    valid_classes = 0

    for class_index, label_name in enumerate(label_names):
        if np.unique(y_true_bin[:, class_index]).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, class_index], y_prob[:, class_index])
        class_auc = auc(fpr, tpr)
        mean_tpr += np.interp(all_fpr, fpr, tpr)
        valid_classes += 1
        ax.plot(fpr, tpr, linewidth=1.5, label=f"{label_name} (AUC={class_auc:.3f})")

    if valid_classes > 0:
        mean_tpr /= valid_classes
        macro_auc = auc(all_fpr, mean_tpr)
        ax.plot(
            all_fpr,
            mean_tpr,
            color="black",
            linewidth=2.5,
            linestyle="--",
            label=f"Macro-average (AUC={macro_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Out-of-Fold ROC Curves")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fold_metric_summary(fold_summaries: Sequence[dict[str, float]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("roc_auc_macro_ovr", "Macro ROC AUC"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("accuracy", "Overall Accuracy"),
        ("f1_macro", "Macro F1"),
    ]
    folds = [summary["fold"] for summary in fold_summaries]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for ax, (metric_key, metric_title) in zip(axes, metrics):
        values = [summary[metric_key] for summary in fold_summaries]
        ax.plot(folds, values, marker="o", linewidth=1.5, color="#1f77b4")
        mean_value = float(np.nanmean(values))
        ax.axhline(mean_value, linestyle="--", linewidth=1.2, color="black")
        ax.set_title(metric_title)
        ax.set_xlabel("Fold")
        ax.set_ylabel("Score")
        ax.set_xticks(folds)
        ax.set_ylim(0, 1.02)
        for fold, value in zip(folds, values):
            ax.text(fold, value, f"{value:.3f}", fontsize=8, ha="center", va="bottom")

    fig.suptitle("Cross-Validation Performance Summary", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_class_support(labels: np.ndarray, label_names: Sequence[str], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    counts = np.bincount(labels, minlength=len(label_names))
    order = np.argsort(counts)[::-1]
    ordered_labels = [label_names[index] for index in order]
    ordered_counts = counts[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(ordered_labels, ordered_counts, color="#4c78a8")
    ax.set_xlabel("Subtype")
    ax.set_ylabel("Number of Cores")
    ax.set_title("Class Support")
    ax.tick_params(axis="x", rotation=45)
    for bar, count in zip(bars, ordered_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(count)), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_summary_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    label_names: Sequence[str],
    fold_summaries: Sequence[dict[str, float]],
    significance_summary: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_names)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums != 0)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 0.9], hspace=0.28, wspace=0.22)

    ax_cm = fig.add_subplot(gs[:, 0])
    draw_confusion_heatmap(ax=ax_cm, matrix=cm_normalized, label_names=label_names, title="Normalized OOF Confusion Matrix")

    ax_roc = fig.add_subplot(gs[0, 1])
    draw_roc_panel(ax=ax_roc, y_true=y_true, y_prob=y_prob, label_names=label_names)

    ax_box = fig.add_subplot(gs[1, 1])
    draw_metric_boxplots(ax=ax_box, fold_summaries=fold_summaries, significance_summary=significance_summary)

    fig.suptitle("ABMIL Subtype Classification Summary", fontsize=16, y=0.98)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_significance_panel(
    fold_summaries: Sequence[dict[str, float]],
    significance_summary: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    draw_metric_boxplots(ax=ax, fold_summaries=fold_summaries, significance_summary=significance_summary)
    ax.set_title("Performance Metrics with Confidence Intervals and Permutation p-values")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_confusion_heatmap(ax, matrix: np.ndarray, label_names: Sequence[str], title: str) -> None:
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=max(1.0, float(np.nanmax(matrix))))
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Proportion")


def draw_roc_panel(ax, y_true: np.ndarray, y_prob: np.ndarray, label_names: Sequence[str]) -> None:
    y_true_bin = label_binarize(y_true, classes=np.arange(len(label_names)))
    if y_true_bin.shape[1] != len(label_names):
        padded = np.zeros((len(y_true), len(label_names)), dtype=np.int64)
        padded[:, : y_true_bin.shape[1]] = y_true_bin
        y_true_bin = padded

    all_fpr = np.linspace(0.0, 1.0, 200)
    mean_tpr = np.zeros_like(all_fpr)
    valid_classes = 0

    for class_index, label_name in enumerate(label_names):
        if np.unique(y_true_bin[:, class_index]).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, class_index], y_prob[:, class_index])
        class_auc = auc(fpr, tpr)
        mean_tpr += np.interp(all_fpr, fpr, tpr)
        valid_classes += 1
        ax.plot(fpr, tpr, linewidth=1.3, label=f"{label_name} ({class_auc:.3f})")

    if valid_classes > 0:
        mean_tpr /= valid_classes
        macro_auc = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, color="black", linestyle="--", linewidth=2.2, label=f"Macro ({macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Out-of-Fold ROC Curves")
    ax.legend(loc="lower right", fontsize=7, frameon=False, ncol=2)


def draw_metric_boxplots(ax, fold_summaries: Sequence[dict[str, float]], significance_summary: dict[str, dict[str, float]]) -> None:
    metric_specs = [
        ("roc_auc_macro_ovr", "ROC AUC"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("accuracy", "Accuracy"),
        ("f1_macro", "Macro F1"),
    ]
    data = [[summary[key] for summary in fold_summaries] for key, _ in metric_specs]
    labels = [label for _, label in metric_specs]

    box = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55, showmeans=True)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    for index, (metric_key, _) in enumerate(metric_specs, start=1):
        summary = significance_summary[metric_key]
        ax.scatter(index, summary["value"], color="black", s=40, zorder=3)
        ax.text(
            index,
            min(1.02, summary["value"] + 0.05),
            (
                f"OOF {summary['value']:.3f}\n"
                f"95% CI [{summary['ci95_low']:.3f}, {summary['ci95_high']:.3f}]\n"
                f"p={format_p_value(summary['permutation_p_value'])}"
            ),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Cross-Validation Metrics with OOF Significance")
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)


def format_p_value(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def write_predictions_csv(output_csv: Path, rows: Sequence[dict[str, object]], label_names: Sequence[str]) -> None:
    fieldnames = [
        "fold",
        "original_filename",
        "original_path",
        "label",
        "predicted_label",
        "embedding_path",
        *[f"prob_{label_name}" for label_name in label_names],
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
