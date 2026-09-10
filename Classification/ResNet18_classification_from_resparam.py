#!/usr/bin/env python3
"""Train ResNet18 QC classification models from the ResParam pipeline."""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_utils_classification import build_dataset
from TrainEvaluationModules_classification import (
    classification_metrics,
    evaluate_predictions,
    make_loader,
    save_evaluation_results,
    train_model,
)
from model.Classification_mocel.ResNet18 import (
    ResNet18,
)


DEFAULT_TRAIN_CSV = str(PROJECT_ROOT / "data/classification_stratify/fold_1/train.csv")
DEFAULT_VAL_CSV = str(PROJECT_ROOT / "data/classification_stratify/fold_1/val.csv")
DEFAULT_TEST_CSV = str(PROJECT_ROOT / "data/classification_stratify/fold_1/test.csv")
DEFAULT_RES_PARAM_CHECKPOINT = (
    "/data/dadmah/chezha/Results/Grid_search_results/training_results/ResParams/"
    "ResNet18/With_scheduler_with_template_slr_6e-05_epoch_80_w_065/best_model_dir/"
    "Epoch_74_TrainLoss_0.8024_ValLoss_0.2517_.pt.tar"
)
DEFAULT_TEMPLATE = "/data/dadmah/chezha/mni_icbm152_t1_tal_nlin_sym_09c.mnc"
DEFAULT_HDF5_ROOT = "/scratch20/Hdf5_classification"
DEFAULT_FACE_MASK_HDF5_PATH = "/scratch20/Hdf5_classification_face_masks"


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args()
    config = {}
    if pre_args.config:
        with open(os.path.expanduser(pre_args.config), "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        if not isinstance(config, dict):
            raise ValueError("Classification YAML must contain a top-level mapping")
    augmentation_config = config.pop("augmentation", {})
    if not isinstance(augmentation_config, dict):
        raise ValueError("The YAML augmentation section must be a mapping")
    if "face_mask_root" in config and "face_mask_hdf5_path" not in config:
        config["face_mask_hdf5_path"] = config.pop("face_mask_root")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument("--backbone_mode", choices=("scratch", "pretrained"), default="pretrained")
    parser.add_argument("--addon", type=str2bool, default=False, help="Use the additional ResNet layers before global average pooling.")
    parser.add_argument("--resparam_checkpoint", default=DEFAULT_RES_PARAM_CHECKPOINT)
    parser.add_argument("--train_csv", default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--val_csv", default=DEFAULT_VAL_CSV)
    parser.add_argument("--test_csv", default=DEFAULT_TEST_CSV)
    parser.add_argument("--hdf5_root", default=DEFAULT_HDF5_ROOT)
    parser.add_argument("--face_mask_hdf5_path", default=DEFAULT_FACE_MASK_HDF5_PATH)
    parser.add_argument("--template_path", default=DEFAULT_TEMPLATE)
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "Classification/classification_resnet18_runs"))
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--label_col", default="lin_motion_rate")
    parser.add_argument("--id_col", default="sbj_ID")
    parser.add_argument("--visit_col", default="sbj_visit")
    parser.add_argument("--image_size", type=int, nargs=3, default=(192, 224, 192))
    parser.add_argument("--normalization", default="MinMax11")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--gpu_update_every", type=int, default=200, help="Print RAM/GPU usage every N training batches; 0 disables it.")
    parser.add_argument("--grad_head_prefix", nargs="+", default=["fnn.2"], help="Parameter-name prefix used for gradient norm reporting.")
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--base_lr", type=float, default=2.5e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--scheduler_patience", type=int, default=5)
    parser.add_argument("--scheduler_threshold", type=float, default=1.0e-3)
    parser.add_argument("--scheduler_threshold_mode", choices=("rel", "abs"), default="rel")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--freeze_backbone", type=str2bool, default=False)
    parser.add_argument("--pos_weight", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--run_mode",
        choices=("auto", "train", "evaluate"),
        default="auto",
        help="auto/train run training followed by evaluation; evaluate only evaluates the best checkpoint.",
    )
    parser.add_argument("--use_augmentation", type=str2bool, default=True)
    parser.add_argument("--augmentation_unchanged_prob", type=float, default=0.2)
    parser.add_argument("--augmentation_use_rician", type=str2bool, default=True)
    parser.add_argument("--augmentation_use_bias", type=str2bool, default=True)
    parser.add_argument("--augmentation_use_deface", type=str2bool, default=True)
    parser.add_argument("--augmentation_rician_prob", type=float, default=0.4)
    parser.add_argument("--augmentation_bias_prob", type=float, default=0.4)
    parser.add_argument("--augmentation_deface_prob", type=float, default=0.4)
    parser.add_argument("--augmentation_rician_sigma", type=float, nargs=2, default=(0.0, 0.05))
    parser.add_argument("--augmentation_bias_coefficients", type=float, nargs=2, default=(-0.3, 0.3))
    parser.add_argument("--augmentation_bias_order", type=int, default=3)
    parser.add_argument("--augmentation_mixture", default=None)
    parser.add_argument("--deface_mask_to_icbm_xfm", default=None)
    valid_keys = {action.dest for action in parser._actions}
    unknown = sorted(set(config) - valid_keys)
    if unknown:
        raise ValueError(f"Unknown classification YAML keys: {unknown}")
    defaults = dict(config)
    defaults.update({
        "use_augmentation": augmentation_config.get("enabled", True),
        "augmentation_unchanged_prob": augmentation_config.get("unchanged_prob", 0.2),
        "augmentation_use_rician": augmentation_config.get("use_rician_noise", True),
        "augmentation_use_bias": augmentation_config.get("use_bias_field", True),
        "augmentation_use_deface": augmentation_config.get("use_deface", True),
        "augmentation_rician_prob": augmentation_config.get("rician_noise_prob", 0.4),
        "augmentation_bias_prob": augmentation_config.get("bias_field_prob", 0.4),
        "augmentation_deface_prob": augmentation_config.get("deface_prob", 0.4),
        "augmentation_rician_sigma": augmentation_config.get("rician_sigma_range", (0.0, 0.05)),
        "augmentation_bias_coefficients": augmentation_config.get("bias_coefficient_range", (-0.3, 0.3)),
        "augmentation_bias_order": augmentation_config.get("bias_order", 3),
        "face_mask_hdf5_path": augmentation_config.get(
            "face_mask_hdf5_path",
            defaults.get("face_mask_hdf5_path", DEFAULT_FACE_MASK_HDF5_PATH),
        ),
        "augmentation_mixture": augmentation_config.get("augmentation_mixture"),
        "deface_mask_to_icbm_xfm": augmentation_config.get("deface_mask_to_icbm_xfm"),
    })
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    if isinstance(args.grad_head_prefix, str):
        args.grad_head_prefix = [part.strip() for part in args.grad_head_prefix.split(",") if part.strip()]
    else:
        args.grad_head_prefix = list(args.grad_head_prefix)
    if args.run_name is None:
        args.run_name = f"resnet18_{args.backbone_mode}_bce"
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args):
    image_size = tuple(args.image_size)
    return ResNet18(input_size=(1, *image_size), dropout=args.dropout, addon=args.addon)


def prepare_output_dirs(args):
    output_dir = Path(args.output_dir) / args.run_name
    paths = {
        "output_dir": output_dir,
        "best_model_dir": output_dir / "best_model_dir",
        "log_dir": output_dir / "log_dir",
        "yaml_saved": output_dir / "yaml_saved",
        "evaluation_results": output_dir / "evaluation_results",
        "checkpoints": output_dir / "checkpoints",
    }
    for directory in paths.values():
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def save_run_configuration(args, yaml_saved_dir):
    yaml_saved_dir = Path(yaml_saved_dir)
    resolved_args = vars(args).copy()
    with (yaml_saved_dir / "resolved_args.json").open("w", encoding="utf-8") as handle:
        json.dump(resolved_args, handle, indent=2)
    with (yaml_saved_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_args, handle, sort_keys=True)
    if args.config:
        source_config = Path(os.path.expanduser(args.config))
        if source_config.is_file():
            shutil.copy2(source_config, yaml_saved_dir / source_config.name)


def load_backbone_weights(model, checkpoint_path):
    if not checkpoint_path:
        return
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"ResParam checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model_state = model.state_dict()
    compatible = {}
    for key, value in state.items():
        clean_key = str(key)
        for prefix in ("module.", "model.", "backbone."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        if clean_key.startswith("fnn.") or clean_key.startswith("addon."):
            continue
        if clean_key in model_state and model_state[clean_key].shape == value.shape:
            compatible[clean_key] = value
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(f"Loaded {len(compatible)} compatible ResParam backbone tensors")
    print("Loaded parameter names:")
    for parameter_name in sorted(compatible):
        print(f"  {parameter_name}")
    print(f"Backbone missing tensors: {len(missing)}; unexpected tensors: {len(unexpected)}")


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_paths = prepare_output_dirs(args)
    output_dir = output_paths["output_dir"]
    save_run_configuration(args, output_paths["yaml_saved"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Best model dir: {output_paths['best_model_dir']}")
    print(f"Log dir: {output_paths['log_dir']}")
    print(f"YAML saved dir: {output_paths['yaml_saved']}")
    print(f"Evaluation results dir: {output_paths['evaluation_results']}")
    print(f"Checkpoints dir: {output_paths['checkpoints']}")

    model = build_model(args)
    if args.backbone_mode in {"pretrained", "addon"}:
        load_backbone_weights(model, args.resparam_checkpoint)
    if args.freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("fnn") or (args.addon and name.startswith("addon"))
    model.to(device)

    if args.run_mode in {"auto", "train"}:
        train_data = build_dataset(args.train_csv, args, train=True)
        val_data = build_dataset(args.val_csv, args, train=False)
        train_loader = make_loader(train_data, args, True)
        val_loader = make_loader(val_data, args, False)

        train_model(model, train_loader, val_loader, args, device, output_dir)

    test_data = build_dataset(args.test_csv, args, train=False)
    val_data = build_dataset(args.val_csv, args, train=False)
    val_loader = make_loader(val_data, args, False)
    test_loader = make_loader(test_data, args, False)

    best_path = output_paths["best_model_dir"] / "best_checkpoint.pt.tar"
    if not best_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found at {best_path}. "
            "Run with --run_mode train or --run_mode auto first."
        )
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model_state_dict"], strict=False)
    results = {}
    for split_name, loader in (("val", val_loader), ("test", test_loader)):
        predictions = evaluate_predictions(model, loader, device, args.threshold)
        results[split_name] = save_evaluation_results(predictions, output_dir, split_name)
    with open(output_paths["evaluation_results"] / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=True)
    print(json.dumps(results, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()