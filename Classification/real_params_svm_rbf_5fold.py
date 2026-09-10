#!/usr/bin/env python3
"""Train SVM-RBF models from pred_raw_params across five predefined folds."""

import argparse
import ast
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DEFAULT_DATA_ROOT = "/home/cic/chezha/PycharmProjects/QC_yaml/data/classification_stratify"
DEFAULT_OUTPUT_DIR = "/home/cic/chezha/PycharmProjects/QC_yaml/Classification/real_params_models"


def parse_params(value, expected_dim):
    if pd.isna(value):
        raise ValueError("pred_raw_params contains a missing value")
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    array = np.asarray(parsed, dtype=np.float32).reshape(-1)
    if array.shape != (expected_dim,):
        raise ValueError(f"Expected {expected_dim} parameters, got shape {array.shape}")
    return array


def load_split(path, params_col, label_col, expected_dim):
    frame = pd.read_csv(path, low_memory=False)
    missing = [column for column in (params_col, label_col) if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    labels = pd.to_numeric(frame[label_col], errors="coerce")
    valid = labels.isin([0, 1]) & frame[params_col].notna()
    skipped = int((~valid).sum())
    frame = frame.loc[valid].copy()
    labels = labels.loc[valid].astype(int).to_numpy()
    features = np.vstack([parse_params(value, expected_dim) for value in frame[params_col]])
    return frame, features, labels, skipped


def calculate_metrics(labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else np.nan
    recall = float(tp / (tp + fn)) if tp + fn else np.nan
    return {
        "samples": int(len(labels)),
        "failed": int((labels == 0).sum()),
        "passed": int((labels == 1).sum()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": recall,
        "specificity": specificity,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }, predictions


def make_model(svm_c, svm_gamma, random_state):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm_rbf",
                SVC(
                    kernel="rbf",
                    C=svm_c,
                    gamma=svm_gamma,
                    probability=True,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--params_col", default="pred_raw_params")
    parser.add_argument("--label_col", default="lin_motion_rate")
    parser.add_argument("--expected_dim", type=int, default=9)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--svm_c", type=float, default=1.0)
    parser.add_argument("--svm_gamma", default="scale")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics = []

    for fold in range(1, 6):
        fold_dir = Path(args.data_root) / f"fold_{fold}"
        loaded = {}
        for split in ("train", "val", "test"):
            loaded[split] = load_split(
                fold_dir / f"{split}.csv",
                args.params_col,
                args.label_col,
                args.expected_dim,
            )

        train_frame, x_train, y_train, skipped_train = loaded["train"]
        model = make_model(args.svm_c, args.svm_gamma, args.random_state)
        model.fit(x_train, y_train)
        joblib.dump(model, output_dir / f"svm_rbf_pred_raw_params_fold_{fold}.joblib")

        for split in ("val", "test"):
            frame, features, labels, skipped = loaded[split]
            probabilities = model.predict_proba(features)[:, 1]
            metrics, predictions = calculate_metrics(labels, probabilities, args.threshold)
            metrics.update(
                {
                    "fold": fold,
                    "split": split,
                    "train_samples": int(len(y_train)),
                    "skipped_train": skipped_train,
                    "skipped_split": skipped,
                }
            )
            fold_metrics.append(metrics)

            prediction_frame = frame.copy()
            prediction_frame["fold"] = fold
            prediction_frame["svm_rbf_pass_probability"] = probabilities
            prediction_frame["svm_rbf_pred_lin_motion_rate"] = predictions
            prediction_frame.to_csv(
                output_dir / f"svm_rbf_pred_raw_params_fold_{fold}_{split}_predictions.csv",
                index=False,
            )

    metrics_frame = pd.DataFrame(fold_metrics)
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "auc",
    ]
    summary_rows = []
    for split, split_frame in metrics_frame.groupby("split", sort=False):
        for metric in metric_names:
            values = pd.to_numeric(split_frame[metric], errors="coerce")
            summary_rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                }
            )

    metrics_frame.to_csv(output_dir / "svm_rbf_pred_raw_params_per_fold_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "svm_rbf_pred_raw_params_mean_std_metrics.csv", index=False
    )
    with open(output_dir / "svm_rbf_pred_raw_params_run_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    print(metrics_frame.to_string(index=False))
    print("\nMean +/- standard deviation:")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\nResults written to {output_dir}")


if __name__ == "__main__":
    main()