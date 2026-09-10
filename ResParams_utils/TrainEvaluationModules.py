from contextlib import nullcontext
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from ResParams_utils.data_utils_image_cropping_logic import (
    GenImageWithParams,
    GenImageWithParams_center_based,
    Normalization,
    plot_mri_orthoview_comparison,
    load_image_func,
)
import psutil
import pynvml
import time
import torch.nn.functional as F
import pandas as pd
import json
import SimpleITK as sitk


DEFAULT_RMSE_POINTS = (
    (0.0, -105.0, 15.0),
    (-70.0, -21.0, 18.0),
    (70.0, 21.0, 18.0),
    (0.0, 73.0, 15.0),
    (0.0, -20.0, 83.0),
    (0.0, -20.0, -47.0),
)


def multiloss_collate_fn(batch):
    collated = {}
    keep_as_list = {"source_image"}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if key in keep_as_list:
            collated[key] = values
        elif torch.is_tensor(values[0]):
            collated[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], np.ndarray):
            shapes = [value.shape for value in values]
            if all(shape == shapes[0] for shape in shapes):
                collated[key] = torch.as_tensor(np.stack(values, axis=0))
            else:
                collated[key] = values
        else:
            collated[key] = values
    return collated

def prepare_batch(batch, device=None, non_blocking=False, dtype=None, renorm_scale=True,scale_factor=1.0,translation_factor=1.0):
    '''
    Here's the info we save in the data_utils (this is for futher analysis)
        return {
           #'image_raw': image_raw, 
           'image': image,
           'subject_id': subject_id,
           'age': age,
           'sex':sex,
           'lin_rate':lin_rate,
           'Affine_param':Affine_param,
           'dataset_origin':dataset_origin
        }'''
    x = batch['image'].to(device=device, dtype=dtype, non_blocking=non_blocking)
    y = batch['Affine_param'].to(device=device, dtype=dtype, non_blocking=non_blocking)
    if renorm_scale:
        y[:,3:6] = (1-y[:,3:6])*scale_factor
        y[:,6:9] = y[:,6:9]*translation_factor
    #x_raw =batch['image_raw'].to(device=device, dtype=dtype, non_blocking=non_blocking)
    subject_id = batch['subject_id']
    subject_visit = batch['subject_visit']
    lin_rate = batch['lin_rate']
    dataset_origin= batch['dataset_origin']
    # we didn't assume any co-vars here
    return x, y, subject_id,subject_visit,lin_rate,dataset_origin,{}


def prepare_batch_multiloss(batch, device=None, non_blocking=False, dtype=None, scale_factor=1.0, translation_factor=1.0):
    inputs = batch["image"].to(device=device, dtype=dtype, non_blocking=non_blocking)
    raw_extra_params = batch["Affine_param"].to(device=device, dtype=dtype, non_blocking=non_blocking)
    targets = raw_extra_params.clone()
    targets[:, 3:6] = (1.0 - targets[:, 3:6]) * scale_factor
    targets[:, 6:9] = targets[:, 6:9] * translation_factor

    reference = None
    if "reference_image" in batch:
        reference = batch["reference_image"].to(device=device, dtype=dtype, non_blocking=non_blocking)

    extra_affine = None
    if "extra_affine" in batch:
        extra_affine = batch["extra_affine"].to(device=device, dtype=dtype, non_blocking=non_blocking)

    ori_affine = None
    if "ori_affine" in batch:
        ori_affine = batch["ori_affine"].to(device=device, dtype=dtype, non_blocking=non_blocking)

    return {
        "inputs": inputs,
        "targets": targets,
        "raw_extra_params": raw_extra_params,
        "reference_image": reference,
        "extra_affine": extra_affine,
        "ori_affine": ori_affine,
        "source_image": batch.get("source_image"),
        "source_spacing": batch.get("source_spacing"),
        "source_origin": batch.get("source_origin"),
        "source_direction": batch.get("source_direction"),
        "subject_id": batch.get("subject_id"),
        "subject_visit": batch.get("subject_visit"),
        "lin_rate": batch.get("lin_rate"),
        "dataset_origin": batch.get("dataset_origin"),
        "augmentation_applied": batch.get("augmentation_applied"),
    }


def prepare_batch_parameter_only(batch, device=None, non_blocking=False, dtype=None, scale_factor=1.0, translation_factor=1.0):
    inputs = batch["image"].to(device=device, dtype=dtype, non_blocking=non_blocking)
    raw_targets = batch["Affine_param"].to(device=device, dtype=dtype, non_blocking=non_blocking)
    targets = raw_targets.clone()
    targets[:, 3:6] = (1.0 - targets[:, 3:6]) * scale_factor
    targets[:, 6:9] = targets[:, 6:9] * translation_factor
    return {
        "inputs": inputs,
        "targets": targets,
        "subject_id": batch.get("subject_id"),
        "subject_visit": batch.get("subject_visit"),
        "lin_rate": batch.get("lin_rate"),
        "dataset_origin": batch.get("dataset_origin"),
        "augmentation_applied": batch.get("augmentation_applied"),
    }


def multiloss_worker_init_fn(worker_id):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    torch.set_num_threads(1)

def save_checkpoint(epoch,model,optimizer,train_loss, val_loss,is_best,output_dir,best_model_dir):
    state ={
        'epoch':epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'train_loss':train_loss,
        'val_loss':val_loss,
    }

    # define the file name pattern
    FILENAME_PATTERN = (
        f"Epoch_{epoch}_"
        f"TrainLoss_{train_loss:.4f}_"
        f"ValLoss_{val_loss:.4f}_.pt.tar"
    )

    checkpoint_path = os.path.join(output_dir,'checkpoints',FILENAME_PATTERN)

    torch.save(state,checkpoint_path)

    if is_best:
        best_path = os.path.join(best_model_dir, FILENAME_PATTERN)
        torch.save(state, best_path)
        print(f"New best model saved to {best_path}")

def save_png_from_inputs(inputs, inputs_raw, labels,base_save_path,outline_path='/data/dadmah/chezha/mni_icbm152_t1_tal_nlin_sym_09c_outline.mnc'):
    #...save image in the path
    #load outline_data
    outline_image= load_image_func(outline_path)
    for idx, (input_data, input_raw_data, label_data) in enumerate(zip(inputs, inputs_raw, labels)):
        input_data_cpu = input_data.cpu().detach().squeeze()
        input_raw_data_cpu = input_raw_data.cpu().detach()
        label_val = label_data.cpu().item() if isinstance(label_data, torch.Tensor) else label_data
        
        try:
            plot_mri_orthoview_comparison(
                input_raw_data_cpu,   
                input_data_cpu,   
                outline_image,       
                label_val,          
                base_save_path         
            )
        except Exception as e:
            print(f"Error plotting sample {idx}: {e}")
            continue

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
            
            gpu_memories.append({
                'index': i, 
                'used_gb': used_gb, 
                'total_gb': total_gb, 
                'percent': percent
            })
        pynvml.nvmlShutdown()
    except pynvml.NVMLError as e:
        return [] 
    return gpu_memories


def log_combined_memory_status(message="Memory Check"):
    """
    Logs both System RAM and GPU VRAM usage in the format: used/total (percentage).
    """

    mem = psutil.virtual_memory()
    system_total_gb = mem.total / GB
    system_used_gb = mem.used / GB
    system_percent = mem.percent
    print(f"  -> System RAM: {system_used_gb:.2f} GB used / {system_total_gb:.2f} GB total ({system_percent:.1f}%)")
    
    gpu_memories = get_gpu_memory_summary()
    
    if gpu_memories:
        for gpu in gpu_memories:
            print(f"  -> GPU {gpu['index']} VRAM: {gpu['used_gb']:.2f} GB used / {gpu['total_gb']:.2f} GB total ({gpu['percent']:.1f}%)")
    else:
        print("  -> GPU VRAM: Monitoring N/A (pynvml error or CUDA not available)")

def get_grad_norm(model, prefix_tuple):
    """Calculates the L2 norm of the gradient for layers matching the given prefixes."""
    total_norm = 0.0
    found_layer = False
    
    # Iterate through named parameters of the model
    for name, param in model.named_parameters():
        # Check if the parameter name starts with one of the prefixes and has a gradient
        if param.grad is not None and name.startswith(prefix_tuple):
            # Calculate the L2 norm for this parameter's gradient
            norm = param.grad.data.norm(2)
            total_norm += norm.item() ** 2
            found_layer = True
    
    if found_layer:
        # Return the final L2 norm (square root of the sum of squared norms)
        return total_norm ** 0.5
    else:
        # Return NaN or a sentinel value if no matching layer was found
        return float('nan')


def _selected_trainable_parameters(model, prefix_tuple):
    if isinstance(prefix_tuple, str):
        prefix_tuple = (prefix_tuple,)
    params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and name.startswith(tuple(prefix_tuple))
    ]
    if params:
        return params
    return [param for param in model.parameters() if param.requires_grad]


def gradient_norm_from_loss(loss, parameters, eps=1e-12):
    if loss is None or not torch.is_tensor(loss) or not loss.requires_grad:
        return 0.0
    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    sq_norm = torch.zeros((), device=loss.device, dtype=torch.float32)
    found_grad = False
    for grad in grads:
        if grad is None:
            continue
        found_grad = True
        sq_norm = sq_norm + grad.detach().float().pow(2).sum()
    if not found_grad:
        return 0.0
    return float(torch.sqrt(sq_norm.clamp_min(eps)).item())


class GradientRatioLossWeighter:
    def __init__(
        self,
        enabled=False,
        update_every=100,
        ema_beta=0.9,
        eps=1e-8,
        target_ratios=None,
        min_weights=None,
        max_weights=None,
        initial_weights=None,
    ):
        self.enabled = bool(enabled)
        self.update_every = max(1, int(update_every))
        self.ema_beta = float(ema_beta)
        self.eps = float(eps)
        self.target_ratios = target_ratios or {"corr": 0.02, "ncc": 0.10, "mi": 0.05}
        self.min_weights = min_weights or {"corr": 0.0, "ncc": 0.0, "mi": 0.0}
        self.max_weights = max_weights or {"corr": 10.0, "ncc": 10.0, "mi": 10.0}
        self.weights = initial_weights or {"corr": 1.0, "ncc": 1.0, "mi": 1.0}
        self.ema_grad_norms = {}

    def _update_ema(self, key, value):
        value = float(value)
        if key not in self.ema_grad_norms:
            self.ema_grad_norms[key] = value
        else:
            self.ema_grad_norms[key] = self.ema_beta * self.ema_grad_norms[key] + (1.0 - self.ema_beta) * value

    def maybe_update(self, batch_index, model, prefix_tuple, losses):
        if not self.enabled:
            return {}
        if batch_index != 1 and batch_index % self.update_every != 0:
            return {}

        parameters = _selected_trainable_parameters(model, prefix_tuple)
        grad_norms = {}
        for key, loss in losses.items():
            grad_norms[key] = gradient_norm_from_loss(loss, parameters, eps=self.eps)
            self._update_ema(key, grad_norms[key])

        g_param = self.ema_grad_norms.get("param", 0.0)
        for key in ("corr", "ncc", "mi"):
            if key not in losses:
                continue
            g_aux = self.ema_grad_norms.get(key, 0.0)
            target_ratio = float(self.target_ratios.get(key, 0.0))
            if target_ratio <= 0.0 or g_aux <= 0.0:
                continue
            weight = target_ratio * g_param / (g_aux + self.eps)
            min_weight = float(self.min_weights.get(key, 0.0))
            max_weight = float(self.max_weights.get(key, 10.0))
            self.weights[key] = float(np.clip(weight, min_weight, max_weight))

        return {
            "grad_param": grad_norms.get("param", 0.0),
            "grad_corr": grad_norms.get("corr", 0.0),
            "grad_ncc": grad_norms.get("ncc", 0.0),
            "grad_mi": grad_norms.get("mi", 0.0),
            "ema_grad_param": self.ema_grad_norms.get("param", 0.0),
            "ema_grad_corr": self.ema_grad_norms.get("corr", 0.0),
            "ema_grad_ncc": self.ema_grad_norms.get("ncc", 0.0),
            "ema_grad_mi": self.ema_grad_norms.get("mi", 0.0),
            "weight_corr": self.weights.get("corr", 0.0),
            "weight_ncc": self.weights.get("ncc", 0.0),
            "weight_mi": self.weights.get("mi", 0.0),
        }

DATASET_ORIGINS = ("NACC", "UKBB", "ADNI", "PPMI", "ALLFTD")


def _init_dataset_loss_stats(dataset_origins):
    return {name: {"loss_sum": 0.0, "count": 0} for name in dataset_origins}


def _normalize_dataset_origin(dataset_origin, batch_size):
    if torch.is_tensor(dataset_origin):
        values = dataset_origin.detach().cpu().tolist()
    elif isinstance(dataset_origin, np.ndarray):
        values = dataset_origin.tolist()
    elif isinstance(dataset_origin, (list, tuple)):
        values = list(dataset_origin)
    else:
        values = [dataset_origin] * batch_size

    if len(values) != batch_size:
        values = (values + ["Unknown"] * batch_size)[:batch_size]

    return [str(v) for v in values]


def _update_dataset_loss_stats(stats, dataset_names, per_sample_loss):
    for dataset_name, sample_loss in zip(dataset_names, per_sample_loss):
        if dataset_name not in stats:
            stats[dataset_name] = {"loss_sum": 0.0, "count": 0}
        stats[dataset_name]["loss_sum"] += float(sample_loss)
        stats[dataset_name]["count"] += 1


def _finalize_dataset_loss_stats(stats):
    dataset_avg_loss = {}
    for dataset_name, dataset_stat in stats.items():
        if dataset_stat["count"] > 0:
            dataset_avg_loss[dataset_name] = dataset_stat["loss_sum"] / dataset_stat["count"]
        else:
            dataset_avg_loss[dataset_name] = float("nan")
    return dataset_avg_loss


def _autocast_context(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda")
    return nullcontext()


def _params_to_affine_matrix(params):
    params = np.asarray(params, dtype=np.float64)
    rotation_angle_x, rotation_angle_y, rotation_angle_z, scale_x, scale_y, scale_z, translation_x, translation_y, translation_z = params
    angle_rad_x = np.deg2rad(rotation_angle_x)
    angle_rad_y = np.deg2rad(rotation_angle_y)
    angle_rad_z = np.deg2rad(rotation_angle_z)

    rot_matrix_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle_rad_x), -np.sin(angle_rad_x)],
            [0, np.sin(angle_rad_x), np.cos(angle_rad_x)],
        ],
        dtype=np.float64,
    )
    rot_matrix_y = np.array(
        [
            [np.cos(angle_rad_y), 0, np.sin(angle_rad_y)],
            [0, 1, 0],
            [-np.sin(angle_rad_y), 0, np.cos(angle_rad_y)],
        ],
        dtype=np.float64,
    )
    rot_matrix_z = np.array(
        [
            [np.cos(angle_rad_z), -np.sin(angle_rad_z), 0],
            [np.sin(angle_rad_z), np.cos(angle_rad_z), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    rotation_matrix = rot_matrix_x @ rot_matrix_y @ rot_matrix_z
    scale_matrix = np.diag([scale_x, scale_y, scale_z])

    affine_matrix = np.eye(4, dtype=np.float64)
    affine_matrix[:3, :3] = rotation_matrix @ scale_matrix
    affine_matrix[:3, 3] = [translation_x, translation_y, translation_z]
    return affine_matrix


def _affine_matrix_to_params(affine_matrix):
    affine_matrix = np.asarray(affine_matrix, dtype=np.float64)
    linear = affine_matrix[:3, :3].copy()
    scales = np.linalg.norm(linear, axis=0)
    scales = np.where(np.abs(scales) < 1e-8, 1.0, scales)
    rotation = linear / scales

    # Project the numeric result back to the nearest proper rotation matrix.
    u, _, vh = np.linalg.svd(rotation)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0: # deal with reflection case, which is not expected but can happen due to numerical issues when the scale is close to zero
        u[:, -1] *= -1
        rotation = u @ vh
        scales[-1] *= -1 #?
        print("Reflection detected in rotation matrix. Adjusting to nearest proper rotation and flipping the sign of the last scale component.")

    sin_y = np.clip(rotation[0, 2], -1.0, 1.0)
    angle_y = np.arcsin(sin_y)
    cos_y = np.cos(angle_y)
    if abs(cos_y) > 1e-6:
        angle_x = np.arctan2(-rotation[1, 2], rotation[2, 2])
        angle_z = np.arctan2(-rotation[0, 1], rotation[0, 0])
    else:
        angle_x = np.arctan2(rotation[2, 1], rotation[1, 1])
        angle_z = 0.0
        print("Gimbal lock detected: cos(angle_y) is close to zero. Setting angle_z to 0 and computing angle_x from rotation[2,1] and rotation[1,1].")

    params = np.zeros(9, dtype=np.float32)
    params[0:3] = np.rad2deg([angle_x, angle_y, angle_z]).astype(np.float32)
    params[3:6] = scales.astype(np.float32)
    params[6:9] = affine_matrix[:3, 3].astype(np.float32)
    return params


def _target_to_raw_params(params, scale_factor, translation_factor):
    raw = np.asarray(params, dtype=np.float32).copy()
    raw[..., 3:6] = 1.0 - raw[..., 3:6] / scale_factor
    raw[..., 6:9] = raw[..., 6:9] / translation_factor
    return raw


def _raw_to_target_params(params, scale_factor, translation_factor):
    target = np.asarray(params, dtype=np.float32).copy()
    target[..., 3:6] = (1.0 - target[..., 3:6]) * scale_factor
    target[..., 6:9] = target[..., 6:9] * translation_factor
    return target


def _compute_residual_params(raw_params, pred_params, residual_composition_order):
    residual_params = []
    for raw_param, pred_param in zip(raw_params, pred_params):
        raw_matrix = _params_to_affine_matrix(raw_param)
        pred_matrix = _params_to_affine_matrix(pred_param)
        if residual_composition_order == "raw_after_pred":
            residual_matrix = raw_matrix @ np.linalg.inv(pred_matrix)
        else:
            residual_matrix = np.linalg.inv(pred_matrix) @ raw_matrix
        residual_params.append(_affine_matrix_to_params(residual_matrix))
        # residual_params need to be transform to [batch_size,1,256,256,256]
    return np.asarray(residual_params, dtype=np.float32) 


def _as_residual_points(point_position):
    if point_position is None:
        point_position = DEFAULT_RMSE_POINTS
    points = np.asarray(point_position, dtype=np.float64)
    if points.shape == (3,):
        points = points.reshape(1, 3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"residual RMSE points must have shape (3,) or (N, 3), got {points.shape}")
    return points


def _compute_residual_point_metrics(pred_target_params, true_target_params, scale_factor, translation_factor, point_position=None):
    pred_raw_params = _target_to_raw_params(pred_target_params, scale_factor, translation_factor)
    true_raw_params = _target_to_raw_params(true_target_params, scale_factor, translation_factor)
    points = _as_residual_points(point_position)
    homogeneous_points = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )

    pred_affines = []
    residual_affines = []
    residual_displacements = []
    residual_point_rmses = []
    residual_rmses = []

    for pred_raw, true_raw in zip(pred_raw_params, true_raw_params):
        pred_affine = _params_to_affine_matrix(pred_raw)
        true_affine = _params_to_affine_matrix(true_raw)
        residual_affine = np.linalg.inv(pred_affine) @ true_affine
        transformed_points = (residual_affine @ homogeneous_points.T).T[:, :3]
        displacement = transformed_points - points
        point_rmses = np.sqrt(np.sum(displacement**2, axis=1))

        pred_affines.append(pred_affine)
        residual_affines.append(residual_affine)
        residual_displacements.append(displacement)
        residual_point_rmses.append(point_rmses)
        residual_rmses.append(float(np.mean(point_rmses)))

    return {
        "pred_raw_params": np.asarray(pred_raw_params, dtype=np.float64),
        "true_raw_params": np.asarray(true_raw_params, dtype=np.float64),
        "pred_affine_matrix": np.asarray(pred_affines, dtype=np.float64),
        "residual_affine": np.asarray(residual_affines, dtype=np.float64),
        "residual_point_displacement": np.asarray(residual_displacements, dtype=np.float64),
        "residual_point_rmse": np.asarray(residual_point_rmses, dtype=np.float64),
        "residual_rmse": np.asarray(residual_rmses, dtype=np.float64),
    }


def _summarize_scalar_values(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _apply_transform_to_raw_batch(raw_images, residual_params, transform, center_based_transform):
    transformed_images = []
    image_generator = GenImageWithParams_center_based if center_based_transform else GenImageWithParams
    for raw_image, residual_param in zip(raw_images, residual_params):
        _, transformed_image = image_generator(raw_image, residual_param)
        if transform is not None:
            transformed_image = transform(transformed_image)
        transformed_images.append(transformed_image.astype(np.float32, copy=False))
    return np.stack(transformed_images, axis=0)


def train_residual_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    args,
    grad_head_prefix,
    scale_factor,
    translation_factor,
):
    model.train()
    total_loss = 0.0
    total_samples = 0
    current_lr = optimizer.param_groups[0]["lr"]
    transform = Normalization(args.normalization)

    train_bar = tqdm(dataloader, desc="Residual self-training", leave=False)
    for batch_idx, batch in enumerate(train_bar):
        inputs, Affine_params, _,_,_,dataset_origin, _ = prepare_batch(batch,device=device,scale_factor=scale_factor,translation_factor=translation_factor)
        #inputs = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        raw_target = Affine_params.detach().cpu().numpy().astype(np.float32)

        model.eval()
        with torch.no_grad():
            with _autocast_context(device):
                pred_target = model(inputs)
        model.train()

        pred_raw = _target_to_raw_params(pred_target.detach().cpu().numpy(), scale_factor, translation_factor)
        raw_target = _target_to_raw_params(raw_target, scale_factor, translation_factor)
        residual_raw = _compute_residual_params(raw_target, pred_raw, args.residual_composition_order)
        residual_target = _raw_to_target_params(residual_raw, scale_factor, translation_factor)

        raw_images = batch["image_raw"].detach().cpu().numpy().astype(np.float32)
        residual_images = _apply_transform_to_raw_batch(
            raw_images,
            residual_raw,
            transform,
            args.center_based_transform,
        )
        residual_inputs = torch.from_numpy(residual_images).to(device=device, dtype=torch.float32, non_blocking=True).unsqueeze(1)
        if inputs.shape[1] > 1:
            residual_inputs = torch.cat((residual_inputs, inputs[:, 1:, ...]), dim=1) 

        residual_target_tensor = torch.from_numpy(residual_target).to(device=device, dtype=torch.float32, non_blocking=True)

        if (batch_idx + 1) % args.log_ram_every_n_batches == 0:
            log_combined_memory_status(f"Residual batch {batch_idx + 1}")

        optimizer.zero_grad()
        with _autocast_context(device):
            outputs = model(residual_inputs)
            loss = criterion(outputs, residual_target_tensor)
        loss.backward()
        grad_norm = get_grad_norm(model, grad_head_prefix)
        optimizer.step()

        bs = residual_target_tensor.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        current_lr = optimizer.param_groups[0]["lr"]
        train_bar.set_postfix(loss=f"{loss.item():.4f}", grad_norm=f"{grad_norm:.6f}", lr=f"{current_lr:.6f}")

    avg_loss = total_loss / total_samples if total_samples else 0.0
    return avg_loss, current_lr

# training and validation epochs
def train_epoch(model,dataloader,criterion,optimizer,device,max_iterations=None, log_ram_every_n_batches=50,grad_head_prefix=("fnn_rmse.4",),scale_factor=1.0,translation_factor=1.0, return_dataset_losses=False, dataset_origins=DATASET_ORIGINS):
    model.train()
    total_loss = 0
    total_samples = 0
    current_lr = optimizer.param_groups[0]['lr']
    dataset_loss_stats = _init_dataset_loss_stats(dataset_origins)
    
    train_bar = tqdm(dataloader, desc="Training", leave=False)
    #for batch in dataloader:
    for batch_idx, batch in enumerate(train_bar):
        inputs, Affine_params, _,_,_,dataset_origin, _ = prepare_batch(batch,device=device,scale_factor=scale_factor,translation_factor=translation_factor)
        #Check min=inputs.min() max=inputs.max() inputs.mean() inputs.std().
        #Check whether the dataloader modified the image too much
        #if batch_idx < 10:
        #    save_png_from_inputs(inputs, inputs_raw, RMSE,base_save_path=f'/data/dadmah/chezha/Results/training_results/CNN/pretrain2/train_example_{batch_idx}.png')
        if (batch_idx + 1) % log_ram_every_n_batches == 0:
            log_combined_memory_status(f"Batch {batch_idx + 1}")
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda"):
            outputs = model(inputs)
            loss=criterion(outputs,Affine_params)

        loss.backward()
        grad_norm = get_grad_norm(model,grad_head_prefix)
        optimizer.step()

        current_lr = optimizer.param_groups[0]['lr']

        bs = Affine_params.size(0)
        
        per_sample_loss = (
            F.mse_loss(outputs, Affine_params, reduction="none")
            .reshape(bs, -1)
            .mean(dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        '''
        per_sample_loss = (
            F.huber_loss(outputs, Affine_params, delta=1.0, reduction="none")
            .reshape(bs, -1)
            .mean(dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        '''
        dataset_names = _normalize_dataset_origin(dataset_origin, bs)
        _update_dataset_loss_stats(dataset_loss_stats, dataset_names, per_sample_loss)
        total_loss+=loss.item() * bs
        total_samples += bs


        train_bar.set_postfix(
            loss=f"{loss.item():.4f}", 
            grad_norm=f"{grad_norm:.6f}", # Display the calculated gradient norm
            lr=f"{current_lr:.6f}"
        )

        if max_iterations is not None and (batch_idx + 1) >= max_iterations:
            break

    avg_loss = total_loss / total_samples if total_samples else 0.0

    if return_dataset_losses:
        return avg_loss, current_lr, _finalize_dataset_loss_stats(dataset_loss_stats)
    return avg_loss, current_lr

def validation_epoch(
    model,
    dataloader,
    criterion,
    device,
    max_iterations=None,
    scale_factor=1.0,
    translation_factor=1.0,
    return_dataset_losses=False,
    dataset_origins=DATASET_ORIGINS,
    return_residual_metrics=False,
    residual_point_position=DEFAULT_RMSE_POINTS,
):
    model.eval()
    total_loss = 0
    total_samples = 0
    dataset_loss_stats = _init_dataset_loss_stats(dataset_origins)
    all_true_raw_params = []
    all_residual_rmses = []
    all_residual_point_rmses = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation", leave=False)):
            batch_data = prepare_batch_parameter_only(
                batch,
                device=device,
                non_blocking=True,
                dtype=torch.float32,
                scale_factor=scale_factor,
                translation_factor=translation_factor,
            )
            inputs = batch_data["inputs"]
            Affine_params = batch_data["targets"]
            dataset_origin = batch_data["dataset_origin"]
            #Check inputs.min() inputs.max() inputs.mean() inputs.std()
            #if batch_idx < 10:
            #    save_png_from_inputs(inputs, inputs_raw, RMSE,base_save_path=f'/data/dadmah/chezha/Results/training_results/CNN/pretrain2/validation_example_{batch_idx}.png')
            with torch.autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, Affine_params)

            if return_residual_metrics:
                residual_metrics = _compute_residual_point_metrics(
                    outputs.detach().cpu().numpy(),
                    Affine_params.detach().cpu().numpy(),
                    scale_factor,
                    translation_factor,
                    residual_point_position,
                )
                all_true_raw_params.append(residual_metrics["true_raw_params"])
                all_residual_rmses.append(residual_metrics["residual_rmse"])
                all_residual_point_rmses.append(residual_metrics["residual_point_rmse"])
            
            # Accumulate metrics
            bs = Affine_params.size(0)
            
            per_sample_loss = (
                F.mse_loss(outputs, Affine_params, reduction="none")
                .reshape(bs, -1)
                .mean(dim=1)
                .detach()
                .cpu()
                .numpy()
            )
            '''
            per_sample_loss = (
                F.huber_loss(outputs, Affine_params, delta=1.0, reduction="none")
                .reshape(bs, -1)
                .mean(dim=1)
                .detach()
                .cpu()
                .numpy()
            )
            '''
            dataset_names = _normalize_dataset_origin(dataset_origin, bs)
            _update_dataset_loss_stats(dataset_loss_stats, dataset_names, per_sample_loss)
            total_loss += loss.item() * bs
            total_samples += bs

            if max_iterations is not None and (batch_idx + 1) >= max_iterations:
                break
    avg_loss = total_loss / total_samples if total_samples else 0.0

    results = [avg_loss]
    if return_dataset_losses:
        results.append(_finalize_dataset_loss_stats(dataset_loss_stats))
    if return_residual_metrics:
        if all_true_raw_params:
            residual_epoch_metrics = {
                "true_raw_params": np.concatenate(all_true_raw_params, axis=0),
                "rmse": np.concatenate(all_residual_rmses),
                "residual_point_rmse": np.concatenate(all_residual_point_rmses, axis=0),
                "residual_point_position": _as_residual_points(residual_point_position),
            }
        else:
            residual_epoch_metrics = {
                "true_raw_params": np.empty((0, 9), dtype=np.float64),
                "rmse": np.empty((0,), dtype=np.float64),
                "residual_point_rmse": np.empty((0, _as_residual_points(residual_point_position).shape[0]), dtype=np.float64),
                "residual_point_position": _as_residual_points(residual_point_position),
            }
        results.append(residual_epoch_metrics)
    if len(results) > 1:
        return tuple(results)
    return avg_loss


class RollingCorrelationBuffer:
    def __init__(self, buffer_size=256):
        self.buffer_size = int(buffer_size)
        self.error = None
        self.target = None

    def get(self, device):
        if self.error is None:
            return None, None
        return self.error.to(device=device), self.target.to(device=device)

    def update(self, error, target):
        error = error.detach().cpu()
        target = target.detach().cpu()
        if self.error is None:
            self.error = error[-self.buffer_size:]
            self.target = target[-self.buffer_size:]
            return
        self.error = torch.cat([self.error, error], dim=0)[-self.buffer_size:]
        self.target = torch.cat([self.target, target], dim=0)[-self.buffer_size:]


def correlation_bias_loss(error, target, mode="slope", eps=1e-6):
    error = error.float()
    target = target.float()
    error_centered = error - error.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    cov = (error_centered * target_centered).mean(dim=0)
    var_target = target_centered.pow(2).mean(dim=0).clamp_min(eps)
    if mode == "corr":
        var_error = error_centered.pow(2).mean(dim=0).clamp_min(eps)
        values = cov / torch.sqrt(var_error * var_target)
    elif mode == "slope":
        values = cov / var_target
    else:
        raise ValueError("corr mode must be 'corr' or 'slope'.")
    return values.pow(2).mean()


def extract_qc_slices(volume, n_slices_per_view=20, output_size=None):
    if volume.dim() == 4:
        volume = volume.unsqueeze(1)
    if volume.dim() != 5:
        raise ValueError(f"Expected volume shape [B,1,D,H,W] or [B,D,H,W], got {tuple(volume.shape)}")
    b, c, d, h, w = volume.shape
    axis_sizes = (d, h, w)
    view_slices = []
    for axis, axis_size in enumerate(axis_sizes):
        indices = torch.linspace(0.1 * (axis_size - 1), 0.9 * (axis_size - 1), steps=n_slices_per_view, device=volume.device)
        indices = indices.round().long().clamp(0, axis_size - 1)
        if axis == 0:
            slices = volume.index_select(2, indices).permute(0, 2, 1, 3, 4)
        elif axis == 1:
            slices = volume.index_select(3, indices).permute(0, 3, 1, 2, 4)
        else:
            slices = volume.index_select(4, indices).permute(0, 4, 1, 2, 3)

        view_slices.append(slices)
    return view_slices


def ncc_loss_from_slices(moving, reference, eps=1e-6):
    if isinstance(moving, (list, tuple)):
        losses = [ncc_loss_from_slices(mov_view, ref_view, eps=eps) for mov_view, ref_view in zip(moving, reference)]
        return torch.stack(losses).mean()
    moving = moving.float().flatten(start_dim=1)
    reference = reference.float().flatten(start_dim=1)
    moving = moving - moving.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)
    numerator = (moving * reference).mean(dim=-1)
    denominator = torch.sqrt(moving.pow(2).mean(dim=-1) * reference.pow(2).mean(dim=-1)).clamp_min(eps)
    return 1.0 - (numerator / denominator).mean()


def nmi_loss_from_slices(moving, reference, bins=32, sigma=0.04, eps=1e-8):
    if isinstance(moving, (list, tuple)):
        losses = [nmi_loss_from_slices(mov_view, ref_view, bins=bins, sigma=sigma, eps=eps) for mov_view, ref_view in zip(moving, reference)]
        return torch.stack(losses).mean()
    moving = moving.float().flatten(start_dim=1)
    reference = reference.float().flatten(start_dim=1)
    moving = (moving - moving.amin(dim=1, keepdim=True)) / (moving.amax(dim=1, keepdim=True) - moving.amin(dim=1, keepdim=True) + eps)
    reference = (reference - reference.amin(dim=1, keepdim=True)) / (reference.amax(dim=1, keepdim=True) - reference.amin(dim=1, keepdim=True) + eps)
    centers = torch.linspace(0.0, 1.0, bins, device=moving.device, dtype=moving.dtype)
    moving_w = torch.softmax(-((moving.unsqueeze(-1) - centers) ** 2) / (2 * sigma**2), dim=-1)
    reference_w = torch.softmax(-((reference.unsqueeze(-1) - centers) ** 2) / (2 * sigma**2), dim=-1)
    pxy = torch.bmm(moving_w.transpose(1, 2), reference_w)
    pxy = pxy / pxy.sum(dim=(1, 2), keepdim=True).clamp_min(eps)
    px = pxy.sum(dim=2)
    py = pxy.sum(dim=1)
    hx = -(px * torch.log(px.clamp_min(eps))).sum(dim=1)
    hy = -(py * torch.log(py.clamp_min(eps))).sum(dim=1)
    hxy = -(pxy * torch.log(pxy.clamp_min(eps))).sum(dim=(1, 2))
    nmi = (hx + hy) / hxy.clamp_min(eps)
    return 2.0 - nmi.mean()


def _normalize_numpy_volume(image, normalization):
    if normalization is None:
        return image.astype(np.float32)
    normalization = str(normalization)
    image = image.astype(np.float32)
    if normalization in {"MinMax", "MinMax01"}:
        image_min = image.min()
        image_max = image.max()
        if image_max <= image_min:
            return np.zeros_like(image, dtype=np.float32)
        return ((image - image_min) / (image_max - image_min)).astype(np.float32)
    if normalization == "MinMax11":
        image_min = image.min()
        image_max = image.max()
        if image_max <= image_min:
            return np.zeros_like(image, dtype=np.float32)
        return (((image - image_min) / (image_max - image_min) * 2.0) - 1.0).astype(np.float32)
    if normalization == "Gaussian":
        image_std = image.std()
        if image_std <= 0:
            return np.zeros_like(image, dtype=np.float32)
        return ((image - image.mean()) / image_std).astype(np.float32)
    return image.astype(np.float32)


def get_template_reference_slices(loss_config, device, dtype, n_slices_per_view):
    sim_cfg = loss_config.get("similarity", {})
    template_path = sim_cfg.get("template_path")
    if not template_path:
        raise ValueError("loss.similarity.reference='template' requires template_path in loss_config.")

    cache = loss_config.setdefault("_template_reference_cache", {})
    cache_key = (template_path, sim_cfg.get("normalization"), int(n_slices_per_view), str(device), str(dtype))
    if cache_key not in cache:
        template_image = sitk.ReadImage(template_path)
        template_array = sitk.GetArrayFromImage(template_image).astype(np.float32)
        template_array = _normalize_numpy_volume(template_array, sim_cfg.get("normalization"))
        template_tensor = torch.from_numpy(template_array).to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        cache[cache_key] = extract_qc_slices(template_tensor, int(n_slices_per_view))
    return cache[cache_key]


def get_template_geometry(loss_config, device, dtype):
    sim_cfg = loss_config.get("similarity", {})
    template_path = sim_cfg.get("template_path")
    if not template_path:
        raise ValueError("Template geometry requires template_path in loss_config.")

    cache = loss_config.setdefault("_template_geometry_cache", {})
    cache_key = (template_path, str(device), str(dtype))
    if cache_key not in cache:
        template_image = sitk.ReadImage(template_path)
        geometry = {
            "shape": torch.as_tensor(sitk.GetArrayFromImage(template_image).shape, device=device, dtype=torch.long),
            "spacing": torch.as_tensor(template_image.GetSpacing(), device=device, dtype=dtype),
            "origin": torch.as_tensor(template_image.GetOrigin(), device=device, dtype=dtype),
            "direction": torch.as_tensor(template_image.GetDirection(), device=device, dtype=dtype),
        }
        cache[cache_key] = geometry
    return cache[cache_key]


def _target_params_to_raw_tensor(params, scale_factor=1.0, translation_factor=1.0):
    raw = params.clone()
    raw[:, 3:6] = 1.0 - raw[:, 3:6] / scale_factor
    raw[:, 6:9] = raw[:, 6:9] / translation_factor
    return raw


def _params_to_affine_matrix_torch(params):
    raw = params.float()
    rx, ry, rz = torch.deg2rad(raw[:, 0]), torch.deg2rad(raw[:, 1]), torch.deg2rad(raw[:, 2])
    sx, sy, sz = raw[:, 3], raw[:, 4], raw[:, 5]
    tx, ty, tz = raw[:, 6], raw[:, 7], raw[:, 8]
    b = raw.shape[0]
    zeros = torch.zeros_like(rx)
    ones = torch.ones_like(rx)
    rxm = torch.stack([ones, zeros, zeros, zeros, torch.cos(rx), -torch.sin(rx), zeros, torch.sin(rx), torch.cos(rx)], dim=1).view(b, 3, 3)
    rym = torch.stack([torch.cos(ry), zeros, torch.sin(ry), zeros, ones, zeros, -torch.sin(ry), zeros, torch.cos(ry)], dim=1).view(b, 3, 3)
    rzm = torch.stack([torch.cos(rz), -torch.sin(rz), zeros, torch.sin(rz), torch.cos(rz), zeros, zeros, zeros, ones], dim=1).view(b, 3, 3)
    scale = torch.diag_embed(torch.stack([sx, sy, sz], dim=1))
    affine = torch.eye(4, device=params.device, dtype=params.dtype).unsqueeze(0).repeat(b, 1, 1)
    affine[:, :3, :3] = rxm @ rym @ rzm @ scale
    affine[:, :3, 3] = torch.stack([tx, ty, tz], dim=1)
    return affine


def _physical_from_template_indices(index_xyz, spacing, origin, direction):
    return origin.view(1, 3) + (index_xyz * spacing.view(1, 3)) @ direction.view(3, 3).T


def _source_normalized_grid_from_physical(source_physical, source_shape, source_spacing, source_origin, source_direction):
    source_index_xyz = (source_physical - source_origin.view(1, 3)) @ torch.linalg.inv(source_direction.view(3, 3)).T
    source_index_xyz = source_index_xyz / source_spacing.view(1, 3).clamp_min(1e-8)
    d, h, w = source_shape
    x = 2.0 * source_index_xyz[:, 0] / max(w - 1, 1) - 1.0
    y = 2.0 * source_index_xyz[:, 1] / max(h - 1, 1) - 1.0
    z = 2.0 * source_index_xyz[:, 2] / max(d - 1, 1) - 1.0
    return torch.stack([x, y, z], dim=1)


def _template_qc_slice_index_xyz(template_shape, n_slices_per_view, device, dtype):
    d, h, w = [int(v) for v in template_shape]
    all_coords = []
    axis_sizes = (d, h, w)
    for axis, axis_size in enumerate(axis_sizes):
        slice_indices = torch.linspace(0.1 * (axis_size - 1), 0.9 * (axis_size - 1), steps=n_slices_per_view, device=device, dtype=dtype).round()
        view_coords = []
        for idx in slice_indices:
            if axis == 0:
                yy = torch.linspace(0, h - 1, h, device=device, dtype=dtype)
                xx = torch.linspace(0, w - 1, w, device=device, dtype=dtype)
                y_grid, x_grid = torch.meshgrid(yy, xx, indexing="ij")
                z_grid = torch.full_like(x_grid, idx)
            elif axis == 1:
                zz = torch.linspace(0, d - 1, d, device=device, dtype=dtype)
                xx = torch.linspace(0, w - 1, w, device=device, dtype=dtype)
                z_grid, x_grid = torch.meshgrid(zz, xx, indexing="ij")
                y_grid = torch.full_like(x_grid, idx)
            else:
                zz = torch.linspace(0, d - 1, d, device=device, dtype=dtype)
                yy = torch.linspace(0, h - 1, h, device=device, dtype=dtype)
                z_grid, y_grid = torch.meshgrid(zz, yy, indexing="ij")
                x_grid = torch.full_like(y_grid, idx)
            view_coords.append(torch.stack([x_grid, y_grid, z_grid], dim=-1))
        all_coords.append(torch.stack(view_coords, dim=0))
    return all_coords


def resample_source_qc_slices_with_affines(
    source_images,
    source_spacing,
    source_origin,
    source_direction,
    template_shape,
    template_spacing,
    template_origin,
    template_direction,
    recovered_affine,
    n_slices_per_view=20,
    output_size=None,
):
    recovered_slices_by_view = [[], [], []]
    device = recovered_affine.device
    dtype = torch.float32
    recovered_affine = recovered_affine.float()
    for i, source_array in enumerate(source_images):
        source_tensor = torch.as_tensor(source_array, device=device, dtype=dtype).view(1, 1, *source_array.shape)
        view_index_xyz = _template_qc_slice_index_xyz(
            template_shape[i].detach().cpu().tolist(),
            n_slices_per_view,
            device,
            dtype,
        )
        for axis, index_xyz in enumerate(view_index_xyz):
            flat_index_xyz = index_xyz.reshape(-1, 3)
            output_physical = _physical_from_template_indices(
                flat_index_xyz,
                template_spacing[i].to(device=device, dtype=dtype),
                template_origin[i].to(device=device, dtype=dtype),
                template_direction[i].to(device=device, dtype=dtype),
            )
            output_h = torch.ones(output_physical.shape[0], 1, device=device, dtype=dtype)
            output_h = torch.cat([output_physical, output_h], dim=1)
            source_physical = (torch.linalg.inv(recovered_affine[i]) @ output_h.T).T[:, :3]
            grid = _source_normalized_grid_from_physical(
                source_physical,
                source_array.shape,
                source_spacing[i].to(device=device, dtype=dtype),
                source_origin[i].to(device=device, dtype=dtype),
                source_direction[i].to(device=device, dtype=dtype),
            )
            grid = grid.view(1, index_xyz.shape[0], index_xyz.shape[1], index_xyz.shape[2], 3)
            sampled = F.grid_sample(source_tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            recovered_slices_by_view[axis].append(sampled.squeeze(0).permute(1, 0, 2, 3))
    return [torch.stack(view_slices, dim=0) for view_slices in recovered_slices_by_view]


def train_epoch_multiloss(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    loss_config,
    epoch,
    corr_buffer=None,
    grad_ratio_weighter=None,
    max_iterations=None,
    log_ram_every_n_batches=50,
    grad_head_prefix=("fnn.3",),
    scale_factor=1.0,
    translation_factor=1.0,
):
    model.train()
    auxiliary_losses_requested = bool(
        loss_config.get("use_corr", True)
        or loss_config.get("use_ncc", True)
        or loss_config.get("use_mi", True)
    )
    warmup_epochs = int(loss_config.get("loss_warmup_epochs", 40))
    use_extra = epoch > warmup_epochs and auxiliary_losses_requested
    if use_extra:
        corr_buffer = corr_buffer or RollingCorrelationBuffer(loss_config.get("corr", {}).get("buffer_size", 256))
    grad_ratio_cfg = loss_config.get("grad_ratio", {})
    if use_extra and grad_ratio_cfg.get("enabled", False) and grad_ratio_weighter is None:
        grad_ratio_weighter = GradientRatioLossWeighter(
            enabled=True,
            update_every=grad_ratio_cfg.get("update_every", 100),
            ema_beta=grad_ratio_cfg.get("ema_beta", 0.9),
            eps=grad_ratio_cfg.get("eps", 1e-8),
            target_ratios=grad_ratio_cfg.get("target_ratios"),
            min_weights=grad_ratio_cfg.get("min_weights"),
            max_weights=grad_ratio_cfg.get("max_weights"),
            initial_weights={
                "corr": loss_config.get("weight_corr", 1.0),
                "ncc": loss_config.get("weight_ncc", 1.0),
                "mi": loss_config.get("weight_mi", 1.0),
            },
        )
    totals = {
        "total_loss": 0.0,
        "Lparam": 0.0,
        "Lcorr": 0.0,
        "Lncc": 0.0,
        "Lmi": 0.0,
        "weight_corr": 0.0,
        "weight_ncc": 0.0,
        "weight_mi": 0.0,
        "grad_param": 0.0,
        "grad_corr": 0.0,
        "grad_ncc": 0.0,
        "grad_mi": 0.0,
        "ema_grad_param": 0.0,
        "ema_grad_corr": 0.0,
        "ema_grad_ncc": 0.0,
        "ema_grad_mi": 0.0,
    }
    total_samples = 0
    current_lr = optimizer.param_groups[0]["lr"]
    train_bar = tqdm(dataloader, desc="Training multi-loss", leave=False)
    for batch_idx, batch in enumerate(train_bar):
        batch_data = (
            prepare_batch_multiloss(
                batch,
                device=device,
                non_blocking=True,
                dtype=torch.float32,
                scale_factor=scale_factor,
                translation_factor=translation_factor,
            )
            if use_extra
            else prepare_batch_parameter_only(
                batch,
                device=device,
                non_blocking=True,
                dtype=torch.float32,
                scale_factor=scale_factor,
                translation_factor=translation_factor,
            )
        )
        multi_batch = batch_data
        inputs = batch_data["inputs"]
        targets = batch_data["targets"]
        if (batch_idx + 1) % log_ram_every_n_batches == 0:
            log_combined_memory_status(f"Batch {batch_idx + 1}")
        optimizer.zero_grad(set_to_none=True)
        if not use_extra:
            with _autocast_context(device):
                outputs = model(inputs)
                loss_param = criterion(outputs, targets)
                total_loss = loss_config.get("weight_param", 1.0) * loss_param if loss_config.get("use_param", True) else torch.zeros((), device=device)
            loss_corr = torch.zeros((), device=device)
            loss_ncc = torch.zeros((), device=device)
            loss_mi = torch.zeros((), device=device)
            weight_corr = 0.0
            weight_ncc = 0.0
            weight_mi = 0.0
            grad_ratio_logs = {}
            total_loss.backward()
            grad_norm = get_grad_norm(model, grad_head_prefix)
            optimizer.step()
            bs = targets.size(0)
            total_samples += bs
            for key, value in (("total_loss", total_loss), ("Lparam", loss_param), ("Lcorr", loss_corr), ("Lncc", loss_ncc), ("Lmi", loss_mi)):
                totals[key] += float(value.detach().item()) * bs
            current_lr = optimizer.param_groups[0]["lr"]
            train_bar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                param=f"{loss_param.item():.4f}",
                grad_norm=f"{grad_norm:.4f}",
                w_corr="0",
                w_ncc="0",
            )
            if max_iterations is not None and (batch_idx + 1) >= max_iterations:
                break
            continue

        with _autocast_context(device):
            outputs = model(inputs)
            loss_param = criterion(outputs, targets)
            total_loss = loss_config.get("weight_param", 1.0) * loss_param if loss_config.get("use_param", True) else torch.zeros((), device=device)
            loss_corr = torch.zeros((), device=device)
            loss_ncc = torch.zeros((), device=device)
            loss_mi = torch.zeros((), device=device)
            weight_corr = float(loss_config.get("weight_corr", 1.0))
            weight_ncc = float(loss_config.get("weight_ncc", 1.0))
            weight_mi = float(loss_config.get("weight_mi", 1.0))
            grad_ratio_logs = {}
            if use_extra:
                if loss_config.get("use_corr", True):
                    current_error = outputs - targets
                    buffer_error, buffer_target = corr_buffer.get(device)
                    if buffer_error is not None:
                        all_error = torch.cat([buffer_error.detach(), current_error], dim=0)
                        all_target = torch.cat([buffer_target.detach(), targets], dim=0)
                    else:
                        all_error, all_target = current_error, targets
                    loss_corr = correlation_bias_loss(all_error, all_target, mode=loss_config.get("corr", {}).get("mode", "slope"))
                sim_cfg = loss_config.get("similarity", {})
                similarity_reference = sim_cfg.get("reference", "ori")
                has_source_similarity_fields = all(
                    multi_batch.get(key) is not None
                    for key in (
                        "source_image",
                        "source_spacing",
                        "source_origin",
                        "source_direction",
                        "extra_affine",
                        "ori_affine",
                    )
                )
                ncc_or_mi_requested = loss_config.get("use_ncc", True) or loss_config.get("use_mi", True)
                if ncc_or_mi_requested and not has_source_similarity_fields:
                    raise ValueError(
                        "NCC/NMI losses require source_image, source_spacing, source_origin, "
                        "source_direction, extra_affine, and ori_affine. Rebuild the Dataset with "
                        "return_similarity_data=True or disable loss_use_ncc/loss_use_mi."
                    )
                if ncc_or_mi_requested:
                    with torch.amp.autocast(device_type=device.type, enabled=False):
                        n_slices = int(sim_cfg.get("n_slices_per_view", 20))
                        pred_raw = _target_params_to_raw_tensor(outputs.float(), scale_factor, translation_factor)
                        pred_affine = _params_to_affine_matrix_torch(pred_raw)
                        recovered_affine = torch.linalg.inv(pred_affine.float()) @ multi_batch["extra_affine"].float() @ multi_batch["ori_affine"].float()
                        template_geometry = get_template_geometry(loss_config, recovered_affine.device, torch.float32)
                        batch_size = recovered_affine.shape[0]
                        rec_slices = resample_source_qc_slices_with_affines(
                            multi_batch["source_image"],
                            multi_batch["source_spacing"],
                            multi_batch["source_origin"],
                            multi_batch["source_direction"],
                            template_geometry["shape"].view(1, 3).repeat(batch_size, 1),
                            template_geometry["spacing"].view(1, 3).repeat(batch_size, 1),
                            template_geometry["origin"].view(1, 3).repeat(batch_size, 1),
                            template_geometry["direction"].view(1, 9).repeat(batch_size, 1),
                            recovered_affine,
                            n_slices_per_view=n_slices,
                        )
                        if similarity_reference == "template":
                            ref_slices = get_template_reference_slices(loss_config, rec_slices[0].device, torch.float32, n_slices)
                            ref_slices = [
                                ref_view.expand(rec_slices[0].shape[0], -1, -1, -1, -1)
                                for ref_view in ref_slices
                            ]
                        else:
                            if multi_batch["reference_image"] is None:
                                raise ValueError("loss.similarity.reference='ori' requires reference_image in the batch.")
                            reference = multi_batch["reference_image"].float()
                            ref_slices = extract_qc_slices(reference, n_slices)
                    if loss_config.get("use_ncc", True):
                        loss_ncc = ncc_loss_from_slices(rec_slices, ref_slices)
                    if loss_config.get("use_mi", True):
                        loss_mi = nmi_loss_from_slices(rec_slices, ref_slices)
                if grad_ratio_weighter is not None and grad_ratio_weighter.enabled:
                    losses_for_weight = {"param": loss_param}
                    if loss_config.get("use_corr", True):
                        losses_for_weight["corr"] = loss_corr
                    if loss_config.get("use_ncc", True):
                        losses_for_weight["ncc"] = loss_ncc
                    if loss_config.get("use_mi", True):
                        losses_for_weight["mi"] = loss_mi
                    grad_ratio_logs = grad_ratio_weighter.maybe_update(
                        batch_idx + 1,
                        model,
                        grad_head_prefix,
                        losses_for_weight,
                    )
                    weight_corr = float(grad_ratio_weighter.weights.get("corr", weight_corr))
                    weight_ncc = float(grad_ratio_weighter.weights.get("ncc", weight_ncc))
                    weight_mi = float(grad_ratio_weighter.weights.get("mi", weight_mi))
                    grad_ratio_logs.setdefault("weight_corr", weight_corr)
                    grad_ratio_logs.setdefault("weight_ncc", weight_ncc)
                    grad_ratio_logs.setdefault("weight_mi", weight_mi)
                    grad_ratio_logs.setdefault("ema_grad_param", grad_ratio_weighter.ema_grad_norms.get("param", 0.0))
                    grad_ratio_logs.setdefault("ema_grad_corr", grad_ratio_weighter.ema_grad_norms.get("corr", 0.0))
                    grad_ratio_logs.setdefault("ema_grad_ncc", grad_ratio_weighter.ema_grad_norms.get("ncc", 0.0))
                    grad_ratio_logs.setdefault("ema_grad_mi", grad_ratio_weighter.ema_grad_norms.get("mi", 0.0))
                if loss_config.get("use_corr", True):
                    total_loss = total_loss + weight_corr * loss_corr
                if loss_config.get("use_ncc", True):
                    total_loss = total_loss + weight_ncc * loss_ncc
                if loss_config.get("use_mi", True):
                    total_loss = total_loss + weight_mi * loss_mi
        total_loss.backward()
        grad_norm = get_grad_norm(model, grad_head_prefix)
        optimizer.step()
        if use_extra and loss_config.get("use_corr", True):
            corr_buffer.update((outputs - targets).detach(), targets.detach())
        bs = targets.size(0)
        total_samples += bs
        for key, value in (("total_loss", total_loss), ("Lparam", loss_param), ("Lcorr", loss_corr), ("Lncc", loss_ncc), ("Lmi", loss_mi)):
            totals[key] += float(value.detach().item()) * bs
        for key, value in (
            ("weight_corr", weight_corr),
            ("weight_ncc", weight_ncc),
            ("weight_mi", weight_mi),
            ("grad_param", grad_ratio_logs.get("grad_param", 0.0)),
            ("grad_corr", grad_ratio_logs.get("grad_corr", 0.0)),
            ("grad_ncc", grad_ratio_logs.get("grad_ncc", 0.0)),
            ("grad_mi", grad_ratio_logs.get("grad_mi", 0.0)),
            ("ema_grad_param", grad_ratio_logs.get("ema_grad_param", 0.0)),
            ("ema_grad_corr", grad_ratio_logs.get("ema_grad_corr", 0.0)),
            ("ema_grad_ncc", grad_ratio_logs.get("ema_grad_ncc", 0.0)),
            ("ema_grad_mi", grad_ratio_logs.get("ema_grad_mi", 0.0)),
        ):
            totals[key] += float(value) * bs
        current_lr = optimizer.param_groups[0]["lr"]
        train_bar.set_postfix(
            loss=f"{total_loss.item():.4f}",
            param=f"{loss_param.item():.4f}",
            grad_norm=f"{grad_norm:.4f}",
            w_corr=f"{weight_corr:.3g}",
            w_ncc=f"{weight_ncc:.3g}",
        )
        if max_iterations is not None and (batch_idx + 1) >= max_iterations:
            break
    logs = {key: value / total_samples if total_samples else 0.0 for key, value in totals.items()}
    return logs["total_loss"], current_lr, logs, corr_buffer, grad_ratio_weighter

'''
def evaluate_dataset(model,dataloader,device,max_iterations=None):
    model.eval()
    all_preds = []
    all_true = []
    all_params = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluation", leave=False)):
            inputs, Affine_params, subject_id,age,sex,lin_rate, _ = prepare_batch(batch,device=device)
            outputs = model(inputs)
            predicted = outputs.data
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(RMSE.cpu().numpy())
            all_params.extend(params.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            if max_iterations is not None and (batch_idx + 1) >= max_iterations:
                #max_iteration is for test plot code
                break
    return np.array(all_preds),np.array(all_true),np.array(all_params),np.array(all_labels)  
'''

def evaluate_dataset(
    model,
    dataloader,
    device,
    max_iterations=None,
    scale_factor=1.0,
    translation_factor=1.0,
    residual_point_position=DEFAULT_RMSE_POINTS,
):
    model.eval()
    
    all_subject_ids = []
    all_subject_visits = []
    all_lin_rates = []
    all_true_affine = []
    all_pred_affine = []
    all_pred_raw_params = []
    all_true_raw_params = []
    all_pred_affine_matrices = []
    all_residual_affines = []
    all_residual_point_displacements = []
    all_residual_point_rmses = []
    all_residual_rmses = []

    with torch.no_grad(): # This disables gradient tracking safely
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            inputs, affine_true, subject_id, subject_visit, lin_rate, _, _ = prepare_batch(
                batch,
                device=device,
                scale_factor=scale_factor,
                translation_factor=translation_factor,
            )
            
            # Inference
            with _autocast_context(device):
                outputs = model(inputs) 
            pred_np = outputs.detach().cpu().numpy()
            true_np = affine_true.detach().cpu().numpy()
            residual_metrics = _compute_residual_point_metrics(
                pred_np,
                true_np,
                scale_factor,
                translation_factor,
                residual_point_position,
            )
            
            all_subject_ids.extend(subject_id.detach().cpu().tolist() if torch.is_tensor(subject_id) else subject_id)
            all_subject_visits.extend(subject_visit.cpu().tolist() if torch.is_tensor(subject_visit) else subject_visit)
            all_lin_rates.extend(lin_rate.cpu().numpy() if torch.is_tensor(lin_rate) else lin_rate)
            
            all_true_affine.append(true_np)
            all_pred_affine.append(pred_np)
            all_pred_raw_params.append(residual_metrics["pred_raw_params"])
            all_true_raw_params.append(residual_metrics["true_raw_params"])
            all_pred_affine_matrices.append(residual_metrics["pred_affine_matrix"])
            all_residual_affines.append(residual_metrics["residual_affine"])
            all_residual_point_displacements.append(residual_metrics["residual_point_displacement"])
            all_residual_point_rmses.append(residual_metrics["residual_point_rmse"])
            all_residual_rmses.append(residual_metrics["residual_rmse"])

            if max_iterations is not None and (batch_idx + 1) >= max_iterations:
                break

    return {
        "subject_id": all_subject_ids,
        "subject_visit": all_subject_visits,
        "lin_rate": np.array(all_lin_rates),
        "true_affine": np.vstack(all_true_affine),
        "pred_affine": np.vstack(all_pred_affine),
        "pred_raw_params": np.concatenate(all_pred_raw_params, axis=0),
        "true_raw_params": np.concatenate(all_true_raw_params, axis=0),
        "pred_affine_matrix": np.concatenate(all_pred_affine_matrices, axis=0),
        "residual_affine": np.concatenate(all_residual_affines, axis=0),
        "residual_point_position": _as_residual_points(residual_point_position),
        "residual_point_displacement": np.concatenate(all_residual_point_displacements, axis=0),
        "residual_point_rmse": np.concatenate(all_residual_point_rmses, axis=0),
        "rmse": np.concatenate(all_residual_rmses),
    }


def save_results_to_csv(eval_results, original_csv_path, output_path):
    """Formats the evaluation dictionary into a CSV file and merges with original info."""
    rows = []
    num_samples = len(eval_results["subject_id"])
    
    # Corrected read_csv (removed index=False)
    ori_df = pd.read_csv(original_csv_path)

    for i in range(num_samples):
        row = {
            'subject_id': eval_results["subject_id"][i],
            'subject_visit': eval_results["subject_visit"][i],
            'lin_rate': eval_results["lin_rate"][i]
        }
        
        # Save affine params as single JSON-encoded columns.
        row["pred_params"] = json.dumps(np.asarray(eval_results["pred_affine"][i]).tolist())
        row["true_params"] = json.dumps(np.asarray(eval_results["true_affine"][i]).tolist())
        if "rmse" in eval_results:
            row["pred_raw_params"] = json.dumps(np.asarray(eval_results["pred_raw_params"][i]).tolist())
            row["true_raw_params"] = json.dumps(np.asarray(eval_results["true_raw_params"][i]).tolist())
            row["pred_affine"] = json.dumps(np.asarray(eval_results["pred_affine_matrix"][i]).tolist())
            row["residual_affine"] = json.dumps(np.asarray(eval_results["residual_affine"][i]).tolist())
            row["residual_point_position"] = json.dumps(np.asarray(eval_results["residual_point_position"]).tolist())
            row["residual_point_displacement"] = json.dumps(
                np.asarray(eval_results["residual_point_displacement"][i]).tolist()
            )
            row["residual_point_rmse"] = json.dumps(
                np.asarray(eval_results["residual_point_rmse"][i]).tolist()
            )
            row["rmse"] = float(eval_results["rmse"][i])
            
        rows.append(row)
    
    df = pd.DataFrame(rows)

    # 1. Rename columns to match your standard and the ori_df keys
    df = df.rename(columns={
        'subject_id': 'sbj_ID',
        'subject_visit': 'sbj_visit',
        'lin_rate': 'lin_motion_rate'
    })

    # 2. Ensure both dataframes use string types for IDs to prevent merge errors
    df['sbj_ID'] = df['sbj_ID'].astype(str)
    ori_df['sbj_ID'] = ori_df['sbj_ID'].astype(str)
    df['sbj_visit'] = df['sbj_visit'].astype(str)
    ori_df['sbj_visit'] = ori_df['sbj_visit'].astype(str)

    # 3. Keep only one copy of overlapping columns when merging.
    key_cols = ['sbj_ID', 'sbj_visit']
    overlap_cols = [col for col in df.columns if col in ori_df.columns and col not in key_cols]
    ori_df = ori_df.drop(columns=overlap_cols)

    # 4. Merge df and ori_df according to sbj_ID and sbj_visit.
    # 'left' keeps all rows from your evaluation results.
    final_df = pd.merge(df, ori_df, on=key_cols, how='left')

    final_df.to_csv(output_path, index=False)
    print(f"Successfully saved results to: {output_path}")

    if "rmse" in eval_results:
        summary = _summarize_scalar_values(eval_results["rmse"])
        if summary is not None:
            summary_path = os.path.splitext(output_path)[0] + "_residual_rmse_summary.csv"
            pd.DataFrame([summary]).to_csv(summary_path, index=False)
            print(f"Successfully saved residual RMSE summary to: {summary_path}")
    return final_df
