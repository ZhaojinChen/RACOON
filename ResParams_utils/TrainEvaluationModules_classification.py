import torch
import torch.nn.functional as F
from tqdm import tqdm
import psutil
import pynvml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
GB = 1024 * 1024 * 1024


def get_gpu_memory_summary():
    """
    Retrieves the VRAM usage for all available GPUs using pynvml.
    Returns:
        list of dict: [{'index': 0, 'used_gb': 1.5, 'total_gb': 12.0, 'percent': 12.5}, ...]
    """
    gpu_memories = []
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            total_gb = info.total / GB
            used_gb = info.used / GB
            percent = (info.used / info.total) * 100 if info.total > 0 else 0.0

            gpu_memories.append(
                {"index": i, "used_gb": used_gb, "total_gb": total_gb, "percent": percent}
            )
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        return []
    return gpu_memories


def log_combined_memory_status(message="Memory Check"):
    """
    Logs both System RAM and GPU VRAM usage in the format: used/total (percentage).
    """
    mem = psutil.virtual_memory()
    ram_used_gb = mem.used / GB
    ram_total_gb = mem.total / GB
    ram_percent = mem.percent

    gpu_memories = get_gpu_memory_summary()
    gpu_status = (
        " | ".join(
            [
                f"GPU {g['index']}: {g['used_gb']:.2f}/{g['total_gb']:.2f} GB ({g['percent']:.1f}%)"
                for g in gpu_memories
            ]
        )
        if gpu_memories
        else "GPU: N/A"
    )

    print(
        f"[{message}] RAM: {ram_used_gb:.2f}/{ram_total_gb:.2f} GB ({ram_percent:.1f}%) | {gpu_status}"
    )


def get_grad_norm(model, prefix_tuple):
    total_norm = 0.0
    found_layer = False

    for name, param in model.named_parameters():
        if param.grad is not None and name.startswith(prefix_tuple):
            norm = param.grad.data.norm(2)
            total_norm += norm.item() ** 2
            found_layer = True

    if found_layer:
        return total_norm ** 0.5
    return float("nan")


def prepare_batch_classification(batch, device=None, non_blocking=False, dtype=None):
    x = batch["image"].to(device=device, dtype=dtype, non_blocking=non_blocking)
    params = batch["params"].to(device=device, dtype=dtype, non_blocking=non_blocking)
    labels = batch["label"].to(device=device, dtype=dtype, non_blocking=non_blocking)
    subject_id = batch.get("subject_id")
    return x, params, labels, subject_id


def prepare_batch_classification_with_image(batch, device=None, non_blocking=False, dtype=None):
    x = batch["image"].to(device=device, dtype=dtype, non_blocking=non_blocking)
    labels = batch["label"].to(device=device, dtype=dtype, non_blocking=non_blocking)
    subject_id = batch.get("subject_id")
    return x, labels, subject_id


def _ensure_binary_logits(logits):
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    if logits.ndim != 1:
        raise ValueError(
            f"Expected binary logits of shape (N) or (N,1), got {tuple(logits.shape)}. "
            "Update the model classifier output to 1 for binary QC."
        )
    return logits


def train_epoch_classification(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    max_iterations=None,
    log_ram_every_n_batches=50,
    grad_head_prefix=("classifier",),
):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    current_lr = optimizer.param_groups[0]["lr"]
    train_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(train_bar):
        images, params, labels, _ = prepare_batch_classification(batch, device=device)
        if (batch_idx + 1) % log_ram_every_n_batches == 0:
            log_combined_memory_status(f"Batch {batch_idx + 1}")
        optimizer.zero_grad()
        logits = model(images, params)
        logits = _ensure_binary_logits(logits)
        loss = criterion(logits, labels)
        loss.backward()
        grad_norm = get_grad_norm(model, grad_head_prefix)
        optimizer.step()

        current_lr = optimizer.param_groups[0]["lr"]
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()

        train_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            grad_norm=f"{grad_norm:.6f}",
            lr=f"{current_lr:.6f}",
        )

        if max_iterations is not None and (batch_idx + 1) >= max_iterations:
            break

    avg_loss = total_loss / total if total else 0.0
    acc = correct / total if total else 0.0
    return avg_loss, acc, current_lr


def validation_epoch_classification(model, dataloader, criterion, device, max_iterations=None):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation", leave=False)):
            images, params, labels, _ = prepare_batch_classification(batch, device=device)
            logits = model(images, params)
            logits = _ensure_binary_logits(logits)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()

            if max_iterations is not None and (batch_idx + 1) >= max_iterations:
                break

    avg_loss = total_loss / total if total else 0.0
    acc = correct / total if total else 0.0
    return avg_loss, acc


def train_epoch_classification_with_image(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    max_iterations=None,
    log_ram_every_n_batches=50,
    grad_head_prefix=("fnn",),
):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    current_lr = optimizer.param_groups[0]["lr"]
    train_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(train_bar):
        images, labels, _ = prepare_batch_classification_with_image(batch, device=device)
        if (batch_idx + 1) % log_ram_every_n_batches == 0:
            log_combined_memory_status(f"Batch {batch_idx + 1}")
        optimizer.zero_grad()
        logits = model(images)
        logits = _ensure_binary_logits(logits)
        loss = criterion(logits, labels)
        loss.backward()
        grad_norm = get_grad_norm(model, grad_head_prefix)
        optimizer.step()

        current_lr = optimizer.param_groups[0]["lr"]
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()

        train_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            grad_norm=f"{grad_norm:.6f}",
            lr=f"{current_lr:.6f}",
        )

        if max_iterations is not None and (batch_idx + 1) >= max_iterations:
            break

    avg_loss = total_loss / total if total else 0.0
    acc = correct / total if total else 0.0
    return avg_loss, acc, current_lr


def validation_epoch_classification_with_image(
    model, dataloader, criterion, device, max_iterations=None
):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation", leave=False)):
            images, labels, _ = prepare_batch_classification_with_image(batch, device=device)
            logits = model(images)
            logits = _ensure_binary_logits(logits)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()

            if max_iterations is not None and (batch_idx + 1) >= max_iterations:
                break

    avg_loss = total_loss / total if total else 0.0
    acc = correct / total if total else 0.0
    return avg_loss, acc


def _safe_subject_ids(subject_ids, batch_size):
    if subject_ids is None:
        return [None] * batch_size
    if torch.is_tensor(subject_ids):
        values = subject_ids.detach().cpu().tolist()
    elif isinstance(subject_ids, np.ndarray):
        values = subject_ids.tolist()
    else:
        values = list(subject_ids)
    if len(values) != batch_size:
        return [None] * batch_size
    return values


def evaluate_split_classification(model, dataloader, criterion, device, threshold, split_name):
    model.eval()

    subject_ids_all = []
    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    sample_loss_all = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating-{split_name}", leave=False):
            images = batch["image"].to(device)
            params = batch["params"].to(device)
            labels = batch["label"].to(device)

            logits = model(images, params)
            logits = _ensure_binary_logits(logits)

            sample_loss = criterion(logits, labels)
            if sample_loss.ndim == 0:
                sample_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            batch_size = labels.size(0)
            subject_ids = _safe_subject_ids(batch.get("subject_id"), batch_size)

            subject_ids_all.extend(subject_ids)
            y_true_all.extend(labels.detach().cpu().numpy().astype(int).tolist())
            y_pred_all.extend(preds.detach().cpu().numpy().astype(int).tolist())
            y_prob_all.extend(probs.detach().cpu().numpy().tolist())
            sample_loss_all.extend(sample_loss.detach().cpu().numpy().tolist())

    n_samples = len(y_true_all)
    if n_samples == 0:
        metrics = {
            "split": split_name,
            "n_samples": 0,
            "mean_loss": np.nan,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "auc": np.nan,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }
        predictions_df = pd.DataFrame(
            columns=[
                "subject_id",
                "true_label",
                "predicted_probability",
                "predicted_label",
                "sample_loss",
            ]
        )
        return metrics, predictions_df, np.zeros((2, 2), dtype=int), np.array([]), np.array([])

    y_true = np.asarray(y_true_all, dtype=int)
    y_pred = np.asarray(y_pred_all, dtype=int)
    y_prob = np.asarray(y_prob_all, dtype=float)
    sample_loss = np.asarray(sample_loss_all, dtype=float)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    if len(np.unique(y_true)) < 2:
        auc_score = np.nan
    else:
        auc_score = roc_auc_score(y_true, y_prob)

    metrics = {
        "split": split_name,
        "n_samples": int(n_samples),
        "mean_loss": float(sample_loss.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(auc_score) if not np.isnan(auc_score) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    predictions_df = pd.DataFrame(
        {
            "subject_id": subject_ids_all,
            "true_label": y_true,
            "predicted_probability": y_prob,
            "predicted_label": y_pred,
            "sample_loss": sample_loss,
        }
    )
    return metrics, predictions_df, cm, y_true, y_prob


def evaluate_split_classification_with_image(
    model, dataloader, criterion, device, threshold, split_name
):
    model.eval()

    subject_ids_all = []
    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    sample_loss_all = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating-{split_name}", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            logits = _ensure_binary_logits(logits)

            sample_loss = criterion(logits, labels)
            if sample_loss.ndim == 0:
                sample_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            batch_size = labels.size(0)
            subject_ids = _safe_subject_ids(batch.get("subject_id"), batch_size)

            subject_ids_all.extend(subject_ids)
            y_true_all.extend(labels.detach().cpu().numpy().astype(int).tolist())
            y_pred_all.extend(preds.detach().cpu().numpy().astype(int).tolist())
            y_prob_all.extend(probs.detach().cpu().numpy().tolist())
            sample_loss_all.extend(sample_loss.detach().cpu().numpy().tolist())

    n_samples = len(y_true_all)
    if n_samples == 0:
        metrics = {
            "split": split_name,
            "n_samples": 0,
            "mean_loss": np.nan,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "auc": np.nan,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }
        predictions_df = pd.DataFrame(
            columns=[
                "subject_id",
                "true_label",
                "predicted_probability",
                "predicted_label",
                "sample_loss",
            ]
        )
        return metrics, predictions_df, np.zeros((2, 2), dtype=int), np.array([]), np.array([])

    y_true = np.asarray(y_true_all, dtype=int)
    y_pred = np.asarray(y_pred_all, dtype=int)
    y_prob = np.asarray(y_prob_all, dtype=float)
    sample_loss = np.asarray(sample_loss_all, dtype=float)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    if len(np.unique(y_true)) < 2:
        auc_score = np.nan
    else:
        auc_score = roc_auc_score(y_true, y_prob)

    metrics = {
        "split": split_name,
        "n_samples": int(n_samples),
        "mean_loss": float(sample_loss.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(auc_score) if not np.isnan(auc_score) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    predictions_df = pd.DataFrame(
        {
            "subject_id": subject_ids_all,
            "true_label": y_true,
            "predicted_probability": y_prob,
            "predicted_label": y_pred,
            "sample_loss": sample_loss,
        }
    )
    return metrics, predictions_df, cm, y_true, y_prob


def save_confusion_matrix_plot_classification(cm, output_path, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_roc_curve_plot_classification(y_true, y_prob, output_path, title):
    if len(np.unique(y_true)) < 2:
        return False, np.nan
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True, auc_score

def plot_training_history(log_path, output_dir):
    """
    Reads the training log and saves a plot of losses and accuracies.
    """
    try:
        # Note: Your code uses sep="\t" in df_log.to_csv
        df = pd.read_csv(log_path, sep="\t")
        
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot Loss
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='tab:red')
        ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color='tab:red', linestyle='--')
        ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', color='red', linewidth=2)
        ax1.tick_params(axis='y', labelcolor='tab:red')

        # Create a second y-axis for Accuracy
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy', color='tab:blue')
        ax2.plot(df['epoch'], df['train_acc'], label='Train Acc', color='tab:blue', linestyle='--')
        ax2.plot(df['epoch'], df['val_acc'], label='Val Acc', color='blue', linewidth=2)
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        plt.title('Training and Validation Metrics')
        fig.tight_layout()
        
        # Merge legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        plot_save_path = os.path.join(output_dir, "training_plot.png")
        plt.savefig(plot_save_path)
        plt.close()
        print(f"Training plot saved to {plot_save_path}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
