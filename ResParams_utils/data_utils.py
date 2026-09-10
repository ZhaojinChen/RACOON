import ast
import math
import os
import random

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from scipy.ndimage import affine_transform
from torch.utils.data import Dataset


def load_image_func(filepath):
    image_sitk = sitk.ReadImage(filepath)
    return sitk.GetArrayFromImage(image_sitk)


class MinMaxNormalization(object):
    def __call__(self, image):
        image_min = image.min()
        image_max = image.max()
        if image_max <= image_min:
            return np.zeros_like(image, dtype=np.float32)
        return ((image - image_min) / (image_max - image_min)).astype(np.float32)


class MinMax01Normalization(object):
    def __call__(self, image):
        image_min = image.min()
        image_max = image.max()
        if image_max <= image_min:
            return np.zeros_like(image, dtype=np.float32)
        return ((image - image_min) / (image_max - image_min)).astype(np.float32)


class MinMax11Normalization(object):
    def __call__(self, image):
        image_min = image.min()
        image_max = image.max()
        if image_max <= image_min:
            return np.zeros_like(image, dtype=np.float32)
        return (((image - image_min) / (image_max - image_min) * 2.0) - 1.0).astype(np.float32)


class GaussianNormalization(object):
    def __call__(self, image):
        image_std = image.std()
        if image_std <= 0:
            return np.zeros_like(image, dtype=np.float32)
        return ((image - image.mean()) / image_std).astype(np.float32)


class Normalization(object):
    def __init__(self, normalization):
        if normalization is not None:
            self.transform = globals()[normalization + "Normalization"]()
        else:
            self.transform = None

    def __call__(self, image):
        if self.transform is not None:
            return self.transform(image)
        return image


def GenParams_large():
    params = np.zeros(9, dtype=float)
    params[0:3] = np.random.uniform(-10, 10, 3)
    params[3:6] = np.random.uniform(0.7, 1.3, 3)
    params[6:9] = np.random.uniform(-14, 14, 3)
    return params


def GenParams():
    params = np.zeros(9, dtype=float)
    params[0:3] = np.random.uniform(-5, 5, 3)
    params[3:6] = np.random.uniform(0.85, 1.15, 3)
    params[6:9] = np.random.uniform(-7, 7, 3)
    return params


def GenParams_small():
    params = np.zeros(9, dtype=float)
    params[0:3] = np.random.uniform(-2, 2, 3)
    params[3:6] = np.random.uniform(0.95, 1.05, 3)
    params[6:9] = np.random.uniform(-3, 3, 3)
    return params


def GenParamsCorrelated(
    ventricle_volume,
    volume_mean=42034.0,
    volume_std=22765.0,
    target_corrs=(-0.8, -0.6, -0.6),
):
    params = np.zeros(9, dtype=float)
    v_latent = (float(ventricle_volume) - volume_mean) / volume_std
    for i, target_rho in zip((3, 4, 5), target_corrs):
        noise = random.gauss(0, 1)
        latent_scale = target_rho * v_latent + math.sqrt(1 - target_rho**2) * noise
        unit_uniform = 0.5 * (1 + math.erf(latent_scale / math.sqrt(2)))
        params[i] = 0.85 + unit_uniform * (1.15 - 0.85)
    params[0:3] = np.random.uniform(-5, 5, 3)
    params[6:9] = np.random.uniform(-7, 7, 3)
    return params


def add_noise(x, vmap=None, noise_type="R", min_amp=0.0, max_amp=0.05):
    max_val = np.max(x) if np.max(x) > 0 else 1.0
    noise_level = np.random.uniform(min_amp, max_amp) * max_val
    if noise_type == "G":
        x_noise = x + np.random.normal(scale=noise_level, size=x.shape).astype(np.single)
    elif noise_type == "R":
        noise_r = np.random.normal(scale=noise_level, size=x.shape).astype(np.single)
        noise_i = np.random.normal(scale=noise_level, size=x.shape).astype(np.single)
        if vmap is not None:
            vmap = vmap.astype(np.single)
            noise_r *= vmap
            noise_i *= vmap
        x_noise = np.sqrt((x + noise_r) ** 2 + noise_i**2)
    else:
        return x
    return np.clip(x_noise, 0.0, max_val).astype(np.single)


def extract_normalized_patch(data, patch_size=(192, 224, 192)):
    x, y, z = data.shape
    px, py, pz = patch_size
    sx = max(0, (x - px) // 2)
    sy = max(0, (y - py) // 2)
    sz = max(0, (z - pz) // 2)
    ex = sx + min(x, px)
    ey = sy + min(y, py)
    ez = sz + min(z, pz)
    patch = data[sx:ex, sy:ey, sz:ez]
    if patch.shape != tuple(patch_size):
        dx = px - patch.shape[0]
        dy = py - patch.shape[1]
        dz = pz - patch.shape[2]
        patch = np.pad(
            patch,
            ((dx // 2, dx - dx // 2), (dy // 2, dy - dy // 2), (dz // 2, dz - dz // 2)),
            mode="constant",
            constant_values=0,
        )
    return patch.astype(np.float32)


def _read_hdf5_image(hf, image_key, h5_idx):
    image_container = hf[image_key]
    if isinstance(image_container, h5py.Group):
        dataset_name = str(int(h5_idx))
        if dataset_name not in image_container:
            raise KeyError(f"HDF5 image group '{image_key}' does not contain dataset '{dataset_name}'.")
        return image_container[dataset_name][:].astype(np.float32, copy=False)
    return image_container[int(h5_idx)].astype(np.float32, copy=False)


def _parse_affine_matrix(value, column_name="ori_affine"):
    if value is None or pd.isna(value):
        raise ValueError(f"{column_name} is empty.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{column_name} is empty.")
    try:
        parsed = ast.literal_eval(text)
        values = np.asarray(parsed, dtype=np.float64).reshape(-1)
    except Exception:
        cleaned = (
            text.replace("np.array", "")
            .replace("array", "")
            .replace("dtype=np.float64", "")
            .replace("dtype=float64", "")
            .replace("dtype=np.float32", "")
            .replace("dtype=float32", "")
        )
        values = np.fromstring(cleaned.replace("[", " ").replace("]", " ").replace(",", " "), sep=" ")
    if values.size == 12:
        affine = np.eye(4, dtype=np.float64)
        affine[:3, :] = values.reshape(3, 4)
        return affine
    if values.size == 16:
        return values.reshape(4, 4).astype(np.float64)
    raise ValueError(f"{column_name} must contain 12 or 16 numeric values, got {values.size}.")


def _affine_matrix_from_params(params):
    rotation_angle_x, rotation_angle_y, rotation_angle_z, scale_x, scale_y, scale_z, translation_x, translation_y, translation_z = params
    angle_rad_x = np.deg2rad(rotation_angle_x)
    angle_rad_y = np.deg2rad(rotation_angle_y)
    angle_rad_z = np.deg2rad(rotation_angle_z)
    rot_matrix_x = np.array([[1, 0, 0], [0, np.cos(angle_rad_x), -np.sin(angle_rad_x)], [0, np.sin(angle_rad_x), np.cos(angle_rad_x)]])
    rot_matrix_y = np.array([[np.cos(angle_rad_y), 0, np.sin(angle_rad_y)], [0, 1, 0], [-np.sin(angle_rad_y), 0, np.cos(angle_rad_y)]])
    rot_matrix_z = np.array([[np.cos(angle_rad_z), -np.sin(angle_rad_z), 0], [np.sin(angle_rad_z), np.cos(angle_rad_z), 0], [0, 0, 1]])
    new_affine = np.eye(4, dtype=np.float64)
    new_affine[:3, :3] = rot_matrix_x @ rot_matrix_y @ rot_matrix_z @ np.diag([scale_x, scale_y, scale_z])
    new_affine[:3, 3] = np.array([translation_x, translation_y, translation_z], dtype=np.float64)
    return new_affine


def _sitk_image_from_hdf5_array(image_array, spacing, origin, direction):
    image_sitk = sitk.GetImageFromArray(image_array.astype(np.float32, copy=False))
    image_sitk.SetSpacing(tuple(np.asarray(spacing, dtype=float).tolist()))
    image_sitk.SetOrigin(tuple(np.asarray(origin, dtype=float).tolist()))
    image_sitk.SetDirection(tuple(np.asarray(direction, dtype=float).tolist()))
    return image_sitk


def _make_sitk_resample_transform(forward_affine):
    inverse_affine = np.linalg.inv(forward_affine)
    tx = sitk.AffineTransform(3)
    tx.SetMatrix(inverse_affine[:3, :3].reshape(-1).tolist())
    tx.SetTranslation(inverse_affine[:3, 3].tolist())
    return tx


def GenImageWithAffineMatrix_sitk_hdf5(image_array, spacing, origin, direction, template_image, affine_matrix):
    source_image = _sitk_image_from_hdf5_array(image_array, spacing, origin, direction)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(template_image)
    resampler.SetTransform(_make_sitk_resample_transform(np.asarray(affine_matrix, dtype=np.float64)))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    return sitk.GetArrayFromImage(resampler.Execute(source_image)).astype(np.float32)


def GenImageWithParams_sitk_hdf5(image_array, spacing, origin, direction, template_image, params):
    new_affine = _affine_matrix_from_params(params)
    transformed_image = GenImageWithAffineMatrix_sitk_hdf5(image_array, spacing, origin, direction, template_image, new_affine)
    return new_affine, transformed_image


def GenImageWithParams(image, params):
    new_affine = _affine_matrix_from_params(params)
    transformed_image = affine_transform(
        image,
        np.linalg.inv(new_affine),
        output_shape=image.shape,
        order=1,
        mode="constant",
        cval=0,
    )
    return new_affine, transformed_image


def GenImageWithParams_center_based(image, params):
    new_affine = _affine_matrix_from_params(params)
    c = np.array([(image.shape[0] - 1) / 2, (image.shape[1] - 1) / 2, (image.shape[2] - 1) / 2], dtype=np.float64)
    linear = new_affine[:3, :3]
    translation = new_affine[:3, 3]
    new_affine[:3, 3] = c - linear @ c + translation
    transformed_image = affine_transform(
        image,
        np.linalg.inv(new_affine),
        output_shape=image.shape,
        order=1,
        mode="constant",
        cval=0,
    )
    return new_affine, transformed_image


def plot_mri_orthoview_comparison(raw_mri_data, transformed_mri_data, outline_data, label, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for row, (title, volume) in enumerate((("Raw", raw_mri_data), ("Transformed", transformed_mri_data))):
        slices = [
            volume[volume.shape[0] // 2, :, :],
            volume[:, volume.shape[1] // 2, :],
            volume[:, :, volume.shape[2] // 2],
        ]
        for col, image_slice in enumerate(slices):
            axes[row, col].imshow(image_slice, cmap="gray", origin="lower")
            axes[row, col].set_title(f"{title} {label}")
            axes[row, col].axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


class MRIDatasetsAug(Dataset):
    def __init__(
        self,
        csv_path,
        hdf5s_path,
        transform=None,
        TrainFlag=False,
        LazyLoading=True,
        center_based_transform=False,
        template=False,
        template_path="",
        image_size=(192, 224, 192),
        add_coord=False,
        add_noise=False,
        return_raw_image=False,
        return_source_image=False,
        train_param_mode="default",
        use_ventricle_volume_for_scale=False,
        use_sitk_hdf5_resample=True,
        sitk_resample_template_path=None,
        ori_affine_column="ori_affine",
        ori_affine_composition_order="extra_after_ori",
        force_identity_extra_params=False,
        **unused_kwargs,
    ):
        if not LazyLoading:
            raise ValueError("The ori_affine data_utils.py supports LazyLoading=True only.")
        if not use_sitk_hdf5_resample:
            raise ValueError("The ori_affine data_utils.py requires use_sitk_hdf5_resample=True.")
        if return_raw_image:
            raise ValueError("return_raw_image is not supported in the ori_affine data_utils.py.")
        if return_source_image and not use_sitk_hdf5_resample:
            raise ValueError("return_source_image=True requires use_sitk_hdf5_resample=True.")
        if ori_affine_composition_order not in {"extra_after_ori", "ori_after_extra"}:
            raise ValueError("ori_affine_composition_order must be 'extra_after_ori' or 'ori_after_extra'.")

        self.info = pd.read_csv(csv_path)
        self.total_samples = len(self.info)
        self.hdf5s_path = hdf5s_path
        self.transform = transform
        self.TrainFlag = TrainFlag
        self.image_size = tuple(image_size)
        self.add_coord = add_coord
        self.add_noise = add_noise
        self.train_param_mode = train_param_mode
        self.use_ventricle_volume_for_scale = use_ventricle_volume_for_scale
        self.ori_affine_column = ori_affine_column
        self.ori_affine_composition_order = ori_affine_composition_order
        self.return_source_image = return_source_image
        self.force_identity_extra_params = force_identity_extra_params
        self.has_total_ventricle_col = "total_ventricle" in self.info.columns

        for required_column in ("hdf5_file", "hdf5_index", ori_affine_column):
            if required_column not in self.info.columns:
                raise ValueError(f"{csv_path} is missing required column '{required_column}'.")

        for candidate in ("transform_params", "transformed_param", "transformed_params"):
            if candidate in self.info.columns:
                self.transform_param_column = candidate
                break
        else:
            self.transform_param_column = None
            print(f"Warning: No transform parameter column found in {csv_path}. Will use default parameters or generate param for training all samples.")

        self.sitk_template_image = sitk.ReadImage(sitk_resample_template_path or template_path)
        self.istemplate = template
        if self.istemplate:
            temp_img = load_image_func(template_path)
            temp_img = extract_normalized_patch(temp_img, self.image_size)
            if self.transform:
                temp_img = self.transform(temp_img)
            self.template = torch.from_numpy(temp_img).float().unsqueeze(0)

        if self.add_coord:
            self.coords = self._coord_map_3d(self.image_size)

    def __len__(self):
        return self.total_samples

    def _coord_map_3d(self, image_size, device=None, dtype=torch.float32, start=-1.0, end=1.0):
        d, h, w = image_size
        z = torch.linspace(start, end, steps=d, device=device, dtype=dtype)
        y = torch.linspace(start, end, steps=h, device=device, dtype=dtype)
        x = torch.linspace(start, end, steps=w, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack([zz, yy, xx], dim=0)

    def _load_extra_params_from_row(self, row):
        if self.force_identity_extra_params or self.transform_param_column is None or pd.isna(row[self.transform_param_column]):
            return np.array([0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=np.float32)
        raw_val = row[self.transform_param_column]
        parsed = ast.literal_eval(raw_val) if isinstance(raw_val, str) else raw_val
        affine_array = np.array(parsed, dtype=np.float32)
        if affine_array.shape[0] != 9:
            raise ValueError(f"{self.transform_param_column} must contain 9 values, got shape {affine_array.shape}.")
        return affine_array

    def _sample_train_params(self, row):
        total_ventricle = row["total_ventricle"] if self.has_total_ventricle_col else None
        if self.use_ventricle_volume_for_scale and self.has_total_ventricle_col and pd.notna(total_ventricle):
            return np.array(GenParamsCorrelated(total_ventricle), dtype=np.float32)
        if self.train_param_mode == "small":
            return np.array(GenParams_small(), dtype=np.float32)
        if self.train_param_mode == "large":
            return np.array(GenParams_large(), dtype=np.float32)
        return np.array(GenParams(), dtype=np.float32)

    def __getitem__(self, index):
        row = self.info.iloc[index]
        hdf5_index = row["hdf5_index"]
        if isinstance(hdf5_index, str):
            hdf5_index = int(float(hdf5_index.strip()))
        else:
            hdf5_index = int(hdf5_index)

        hdf5_path = os.path.join(self.hdf5s_path, row["hdf5_file"])
        with h5py.File(hdf5_path, "r") as hf:
            source_image_key = "source_imgs" if "source_imgs" in hf else "imgs"
            image = _read_hdf5_image(hf, source_image_key, hdf5_index)
            source_image = image.copy() if self.return_source_image else None
            source_spacing = hf["source_spacing"][hdf5_index]
            source_origin = hf["source_origin"][hdf5_index]
            source_direction = hf["source_direction"][hdf5_index]

        extra_params = self._sample_train_params(row) if self.TrainFlag else self._load_extra_params_from_row(row)
        ori_affine = _parse_affine_matrix(row[self.ori_affine_column], self.ori_affine_column)
        extra_affine = _affine_matrix_from_params(extra_params)
        if self.ori_affine_composition_order == "extra_after_ori":
            final_affine = extra_affine @ ori_affine
        else:
            final_affine = ori_affine @ extra_affine

        image = GenImageWithAffineMatrix_sitk_hdf5(
            image,
            source_spacing,
            source_origin,
            source_direction,
            self.sitk_template_image,
            final_affine,
        )

        if self.transform:
            image = self.transform(image)
        image = extract_normalized_patch(image, self.image_size)
        if self.add_noise and self.TrainFlag:
            image = add_noise(image, noise_type="R")

        image = torch.from_numpy(image).float().unsqueeze(0)
        if self.istemplate:
            image = torch.cat((image, self.template), dim=0)
        if self.add_coord:
            image = torch.cat((image, self.coords), dim=0)

        return {
            "image": image,
            "subject_id": row["sbj_ID"],
            "subject_visit": row["sbj_visit"],
            "lin_rate": row["lin_motion_rate"],
            "Affine_param": extra_params,
            "ori_affine": ori_affine.astype(np.float32),
            "dataset_origin": row["dataset_origin"],
            **(
                {
                    "source_image": source_image,
                    "source_spacing": np.asarray(source_spacing, dtype=np.float32),
                    "source_origin": np.asarray(source_origin, dtype=np.float32),
                    "source_direction": np.asarray(source_direction, dtype=np.float32),
                }
                if self.return_source_image
                else {}
            ),
        }
