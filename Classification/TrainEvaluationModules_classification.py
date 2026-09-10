"""Training and evaluation utilities for binary QC classification."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from ResParams_utils.TrainEvaluationModules import _autocast_context, get_grad_norm, log_combined_memory_status


def make_loader(dataset, args, shuffle):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )


def forward_model(model, batch):
    return model(batch["image"][:, :1]).squeeze(-1)


def classification_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "samples": int(len(y_true)),
    }


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    threshold=0.5,
    gpu_update_every=0,
    grad_head_prefix=("fnn.2",),
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels_all, probs_all = [], []
    grad_norms = []
    for batch_idx, batch in enumerate(tqdm(loader, desc="train" if training else "eval", leave=False)):
        if training and gpu_update_every and (batch_idx + 1) % gpu_update_every == 0:
            log_combined_memory_status(f"Classification train batch {batch_idx + 1}")
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["lin_rate"].to(device, dtype=torch.float32, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device):
            logits = forward_model(model, {"image": images})
            loss = criterion(logits, labels)
        if training:
            loss.backward()
            grad_norm = get_grad_norm(model, tuple(grad_head_prefix))
            grad_norms.append(grad_norm)
            optimizer.step()
        total_loss += loss.item() * labels.numel()
        labels_all.extend(labels.detach().cpu().numpy().astype(int).tolist())
        probs_all.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())

    y_true = np.asarray(labels_all, dtype=int)
    y_prob = np.asarray(probs_all, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    finite_grad_norms = [value for value in grad_norms if np.isfinite(value)]
    grad_summary = {
        "mean_grad_norm": float(np.mean(finite_grad_norms)) if finite_grad_norms else float("nan"),
        "max_grad_norm": float(np.max(finite_grad_norms)) if finite_grad_norms else float("nan"),
    }
    return total_loss / max(1, len(y_true)), classification_metrics(y_true, y_pred, y_prob), grad_summary


def train_model(model, train_loader, val_loader, args, device, output_dir):
    """Train a classifier, save history/checkpoints, and return the model."""
    output_dir = Path(output_dir)
    best_model_dir = output_dir / "best_model_dir"
    log_dir = output_dir / "log_dir"
    checkpoints_dir = output_dir / "checkpoints"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if args.pos_weight is None:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([args.pos_weight], device=device)
        )

    lr = args.base_lr * (args.batch_size ** 0.5) if args.base_lr is not None else args.lr
    print(f"AdamW learning rate: base_lr={args.base_lr:.8g} * sqrt(batch_size={args.batch_size}) = {lr:.8g}")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        threshold=args.scheduler_threshold,
        threshold_mode=args.scheduler_threshold_mode,
    )

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("val_loss", best_val_loss))

    history = []
    patience_count = 0
    for epoch in range(start_epoch, args.max_epochs + 1):
        started = time.time()
        train_loss, train_metrics, train_grad_summary = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            args.threshold,
            args.gpu_update_every,
            args.grad_head_prefix,
        )
        with torch.no_grad():
            val_loss, val_metrics, _ = run_epoch(
                model, val_loader, criterion, device, None, args.threshold
            )
        scheduler.step(val_loss)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "seconds": time.time() - started,
            "train_mean_grad_norm": train_grad_summary["mean_grad_norm"],
            "train_max_grad_norm": train_grad_summary["max_grad_norm"],
        }
        record.update({f"train_{key}": value for key, value in train_metrics.items()})
        record.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(record)
        pd.DataFrame(history).to_csv(log_dir / "training_history.csv", index=False)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.5f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_loss:.5f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"grad_norm={train_grad_summary['mean_grad_norm']:.4f} "
            f"seconds={record['seconds']:.1f}"
        )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_count = 0
        else:
            patience_count += 1
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "args": vars(args),
        }
        checkpoint_filename = (
            f"epoch_{epoch:03d}_"
            f"TrainLoss_{train_loss:.4f}_"
            f"TrainAcc_{train_metrics['accuracy']:.4f}_"
            f"ValLoss_{val_loss:.4f}_"
            f"ValAcc_{val_metrics['accuracy']:.4f}.pt.tar"
        )
        torch.save(checkpoint, checkpoints_dir / "last_checkpoint.pt.tar")
        torch.save(checkpoint, checkpoints_dir / checkpoint_filename)
        if is_best:
            torch.save(checkpoint, best_model_dir / "best_checkpoint.pt.tar")
        if patience_count >= args.patience:
            break
    return model


def evaluate_predictions(model, loader, device, threshold):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predictions", leave=False):
            with _autocast_context(device):
                logits = forward_model(model, {"image": batch["image"].to(device)})
            probabilities = torch.sigmoid(logits).cpu().numpy()
            labels = batch["lin_rate"].numpy().astype(int)
            for subject_id, visit, label, logit, probability in zip(
                list(batch["subject_id"]),
                list(batch["subject_visit"]),
                labels,
                logits.cpu().numpy(),
                probabilities,
            ):
                rows.append(
                    {
                        "sbj_ID": subject_id,
                        "sbj_visit": visit,
                        "lin_motion_rate": int(label),
                        "pred_logits": float(logit),
                        "predicted_probability": float(probability),
                        "predicted_label": int(probability >= threshold),
                    }
                )
    return pd.DataFrame(rows)


def save_evaluation_results(predictions, output_dir, split_name):
    """Save per-split probabilities/predictions and classification metrics."""
    evaluation_dir = Path(output_dir) / "evaluation_results"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(evaluation_dir / f"{split_name}_predictions.csv", index=False)
    metrics = classification_metrics(
        predictions["lin_motion_rate"].to_numpy(),
        predictions["predicted_label"].to_numpy(),
        predictions["predicted_probability"].to_numpy(),
    )
    with (evaluation_dir / f"{split_name}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)
    return metrics
