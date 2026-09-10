import ast
import math
import os
import random
import re

import h5py
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torchio as tio
from torch.utils.data import Dataset


DEFAULT_FACE_MASK_TO_ICBM_XFM = "/scratch20/ADNI/stxlin_xfm_cc/002_S_0295_20060418_V1_t1_to_icbm.xfm"
IDENTITY_EXTRA_PARAMS = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=np.float32)


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


def _parse_path_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    if os.path.isdir(text):
        return sorted(
            os.path.join(text, name)
            for name in os.listdir(text)
            if name.endswith((".mnc", ".nii", ".nii.gz"))
        )
    if os.path.isfile(text) and not text.endswith((".mnc", ".nii", ".nii.gz")):
        with open(text, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_minc_xfm_matrix(xfm_path):
    with open(xfm_path, "r", encoding="utf-8") as f:
        text = f.read()
    match = re.search(r"Linear_Transform\s*=\s*(.*?);", text, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        raise ValueError(f"Could not find Linear_Transform block in {xfm_path}")
    values = [float(v) for v in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", match.group(1))]
    if len(values) != 12:
        raise ValueError(f"Expected 12 affine values in {xfm_path}, found {len(values)}")
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :] = np.asarray(values, dtype=np.float64).reshape(3, 4)
    return affine


def _apply_rician_noise(image, sigma_range=(0.0, 0.05)):
    sigma = random.uniform(float(sigma_range[0]), float(sigma_range[1]))
    scale = max(float(np.max(np.abs(image))), 1e-6)
    sigma = sigma * scale
    noise_real = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
    noise_imag = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
    return np.sqrt((image + noise_real) ** 2 + noise_imag**2).astype(np.float32)


def _apply_numpy_bias_field(image, coefficient_range=(0.1, 0.3)):
    shape = image.shape
    grids = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[0], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[1], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[2], dtype=np.float32),
        indexing="ij",
    )
    coeffs = np.random.uniform(float(coefficient_range[0]), float(coefficient_range[1]), size=4).astype(np.float32)
    field = coeffs[0] * grids[0] + coeffs[1] * grids[1] + coeffs[2] * grids[2] + coeffs[3] * grids[0] * grids[1]
    return (image * np.exp(field)).astype(np.float32)


def _apply_bias_field(image, coefficient_range=(-0.3, 0.3)):
    try:
        transform = tio.RandomBiasField(coefficients=coefficient_range, p=1.0)
        tensor = torch.from_numpy(image[None]).float()
        transformed = transform(tensor)
        return transformed.squeeze(0).numpy().astype(np.float32)
    except Exception:
        return _apply_numpy_bias_field(image, coefficient_range)


def _normalize_augmentation_ops(ops):
    supported = {"rician", "bias", "deface"}
    normalized = []
    for op in ops:
        op = str(op).strip().lower()
        if op not in supported:
            raise ValueError(f"Unsupported augmentation op '{op}'. Supported ops are {sorted(supported)}.")
        if op not in normalized:
            normalized.append(op)
    ordered = [op for op in ("rician", "bias", "deface") if op in normalized]
    return ordered


def _sample_deface_mask_extra_affine():
    params = np.zeros(9, dtype=np.float64)
    params[0:3] = np.random.uniform(-0.5, 0.5, 3)
    params[3:6] = np.random.uniform(0.98, 1.02, 3)
    params[6:9] = np.random.uniform(-1.0, 1.0, 3)
    return _affine_matrix_from_params(params), params


def _resample_mask(mask_path, template_image, affine_matrix):
    mask_image = sitk.ReadImage(mask_path)
    inverse_affine = np.linalg.inv(np.asarray(affine_matrix, dtype=np.float64))
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(inverse_affine[:3, :3].reshape(-1).tolist())
    transform.SetTranslation(inverse_affine[:3, 3].tolist())
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(template_image)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(1.0)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    return sitk.GetArrayFromImage(resampler.Execute(mask_image)).astype(np.float32)


class MRIDatasetsAugMultiLoss(Dataset):
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
        force_identity_extra_params=True,
        train_identity_extra_params_prob=0.0,
        augmentation_config=None,
        similarity_reference="ori",
        return_similarity_data=True,
        return_reference_image=True,
        deface_mask_paths=None,
        face_mask_hdf5_path=None,
        deface_mask_to_icbm_xfm=DEFAULT_FACE_MASK_TO_ICBM_XFM,
        use_epoch_transform_params=False,
        epoch_transform_param_prefix="epoch",
        epoch_transform_param_suffix="_transform_params",
        **kwargs,
    ):
        if not LazyLoading:
            raise ValueError("The ori_affine multiloss dataset supports LazyLoading=True only.")
        if not use_sitk_hdf5_resample:
            raise ValueError("The ori_affine multiloss dataset requires use_sitk_hdf5_resample=True.")
        if return_raw_image:
            raise ValueError("return_raw_image is not supported in the ori_affine multiloss dataset.")
        if ori_affine_composition_order not in {"extra_after_ori", "ori_after_extra"}:
            raise ValueError("ori_affine_composition_order must be 'extra_after_ori' or 'ori_after_extra'.")
        if similarity_reference not in {"ori", "template"}:
            raise ValueError("similarity_reference must be 'ori' or 'template'.")
        train_identity_extra_params_prob = float(train_identity_extra_params_prob)
        if not 0.0 <= train_identity_extra_params_prob <= 1.0:
            raise ValueError("train_identity_extra_params_prob must be between 0 and 1.")

        self.info = pd.read_csv(csv_path, low_memory=False)
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
        self.return_similarity_data = return_similarity_data
        self.return_reference_image = return_reference_image
        self.force_identity_extra_params = force_identity_extra_params
        self.train_identity_extra_params_prob = train_identity_extra_params_prob
        self.use_epoch_transform_params = bool(use_epoch_transform_params)
        self.epoch_transform_param_prefix = str(epoch_transform_param_prefix)
        self.epoch_transform_param_suffix = str(epoch_transform_param_suffix)
        self.active_epoch = None
        self.has_total_ventricle_col = "total_ventricle" in self.info.columns
        self._bias_transform = None

        for required_column in ("hdf5_file", "hdf5_index", ori_affine_column):
            if required_column not in self.info.columns:
                raise ValueError(f"{csv_path} is missing required column '{required_column}'.")

        epoch_columns = [
            column for column in self.info.columns
            if str(column).startswith(self.epoch_transform_param_prefix)
            and str(column).endswith(self.epoch_transform_param_suffix)
        ]
        if self.TrainFlag and self.use_epoch_transform_params:
            self.transform_param_column = None
            if not epoch_columns:
                raise ValueError(
                    f"{csv_path} has use_epoch_transform_params=True, but no columns matching "
                    f"'{self.epoch_transform_param_prefix}*{self.epoch_transform_param_suffix}' were found."
                )
            print(
                f"Training transform params will be loaded from {len(epoch_columns)} precomputed epoch columns "
                f"in {csv_path}; example column: {sorted(epoch_columns)[0]}"
            )
        else:
            for candidate in ("transform_params", "transformed_param", "transformed_params"):
                if candidate in self.info.columns:
                    self.transform_param_column = candidate
                    print(f"Transform params will be loaded from column '{candidate}' in {csv_path}.")
                    break
            else:
                self.transform_param_column = None
                if self.TrainFlag:
                    print(f"Warning: No transform parameter column found in {csv_path}. Will generate params during training.")
                else:
                    print(f"Warning: No transform parameter column found in {csv_path}. Evaluation will use identity extra params.")

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

        self.augmentation_config = augmentation_config or {}
        self.similarity_reference = similarity_reference
        self.deface_mask_paths = _parse_path_list(deface_mask_paths or self.augmentation_config.get("deface_mask_paths"))
        self.face_mask_hdf5_path = face_mask_hdf5_path or self.augmentation_config.get("face_mask_hdf5_path") or self.hdf5s_path
        self.face_mask_columns = []
        face_mask_indices = sorted(
            int(match.group(1))
            for column in self.info.columns
            for match in [re.fullmatch(r"FM(\d+)_H5File", str(column))]
            if match is not None
        )
        for mask_index in face_mask_indices:
            file_col = f"FM{mask_index}_H5File"
            index_col = f"FM{mask_index}_H5Index"
            if file_col in self.info.columns and index_col in self.info.columns:
                self.face_mask_columns.append((file_col, index_col))
        self.face_mask_columns = self._validate_face_mask_columns(self.face_mask_columns)
        self.deface_mask_to_icbm = (
            _parse_minc_xfm_matrix(deface_mask_to_icbm_xfm)
            if deface_mask_to_icbm_xfm and os.path.isfile(deface_mask_to_icbm_xfm)
            else np.eye(4, dtype=np.float64)
        )
        self.augmentation_mixture = self._validate_augmentation_mixture(self.augmentation_config.get("augmentation_mixture"))

    def _has_deface_source(self):
        return bool(self.face_mask_columns or self.deface_mask_paths)

    def set_epoch(self, epoch):
        self.active_epoch = int(epoch)

    def _epoch_transform_param_column(self):
        if self.active_epoch is None:
            return None
        return f"{self.epoch_transform_param_prefix}{int(self.active_epoch)}{self.epoch_transform_param_suffix}"

    def _validate_face_mask_columns(self, face_mask_columns):
        if not face_mask_columns:
            return []
        if not self.augmentation_config.get("validate_face_mask_hdf5", True):
            return face_mask_columns

        valid_columns = []
        for file_col, index_col in face_mask_columns:
            try:
                file_names = [str(name) for name in self.info[file_col].dropna().unique()]
                if not file_names:
                    raise ValueError(f"{file_col} has no HDF5 file names.")
                for file_name in file_names:
                    hdf5_path = os.path.join(self.face_mask_hdf5_path, file_name)
                    if not os.path.isfile(hdf5_path):
                        raise FileNotFoundError(hdf5_path)
                    with h5py.File(hdf5_path, "r") as hf:
                        if "masks" not in hf:
                            raise KeyError(f"{hdf5_path} does not contain a 'masks' group.")
                        mask_group = hf["masks"]
                        matching_rows = self.info[self.info[file_col].astype(str) == file_name]
                        if matching_rows.empty:
                            continue
                        index_names = [
                            str(int(float(str(value).strip())))
                            for value in matching_rows[index_col].dropna().unique()
                        ]
                        missing = [name for name in index_names if name not in mask_group]
                        if missing:
                            raise KeyError(
                                f"{hdf5_path} masks group is missing {len(missing)} referenced datasets; "
                                f"first missing dataset is '{missing[0]}'."
                            )
                        for dataset_name in index_names:
                            _ = mask_group[dataset_name].shape
                valid_columns.append((file_col, index_col))
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid face-mask HDF5 source {file_col}/{index_col}: {exc}. "
                    "Regenerate the referenced face-mask HDF5 files with FaceMask_Hdf5_creation.py."
                ) from exc
        return valid_columns

    def _validate_augmentation_mixture(self, mixture):
        if mixture is None:
            return None
        if not isinstance(mixture, (list, tuple)):
            raise ValueError("augmentation_mixture must be a list of entries with 'ops' and 'prob'.")
        validated = []
        total_prob = 0.0
        has_deface_source = self._has_deface_source()
        for entry in mixture:
            if not isinstance(entry, dict):
                raise ValueError("Each augmentation_mixture entry must be a mapping with 'ops' and 'prob'.")
            ops = _normalize_augmentation_ops(entry.get("ops", []))
            prob = float(entry.get("prob", 0.0))
            if prob < 0.0:
                raise ValueError("augmentation_mixture probabilities must be nonnegative.")
            if "deface" in ops and not has_deface_source:
                raise ValueError("augmentation_mixture includes deface, but no valid deface mask source is available.")
            total_prob += prob
            validated.append((ops, prob))
        if not np.isclose(total_prob, 1.0, rtol=1e-6, atol=1e-6):
            raise ValueError(f"augmentation_mixture probabilities must sum to 1.0, got {total_prob}.")
        return validated

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
            #print("Using identity extra parameters.")
            return IDENTITY_EXTRA_PARAMS.copy()
        raw_val = row[self.transform_param_column]
        parsed = ast.literal_eval(raw_val) if isinstance(raw_val, str) else raw_val
        affine_array = np.array(parsed, dtype=np.float32)
        if affine_array.shape[0] != 9:
            raise ValueError(f"{self.transform_param_column} must contain 9 values, got shape {affine_array.shape}.")
        return affine_array

    def _sample_train_params(self, row):
        if self.use_epoch_transform_params:
            epoch_column = self._epoch_transform_param_column()
            if epoch_column is None:
                raise ValueError("use_epoch_transform_params=True requires set_epoch(epoch) before training.")
            if epoch_column not in self.info.columns:
                raise KeyError(f"Training CSV is missing epoch transform column '{epoch_column}'.")
            if pd.isna(row[epoch_column]):
                raise ValueError(f"Epoch transform column '{epoch_column}' contains NaN.")
            raw_val = row[epoch_column]
            parsed = ast.literal_eval(raw_val) if isinstance(raw_val, str) else raw_val
            affine_array = np.array(parsed, dtype=np.float32)
            if affine_array.shape[0] != 9:
                raise ValueError(f"{epoch_column} must contain 9 values, got shape {affine_array.shape}.")
            return affine_array
        if self.train_identity_extra_params_prob > 0.0 and random.random() < self.train_identity_extra_params_prob:
            return IDENTITY_EXTRA_PARAMS.copy()
        total_ventricle = row["total_ventricle"] if self.has_total_ventricle_col else None
        if self.use_ventricle_volume_for_scale and self.has_total_ventricle_col and pd.notna(total_ventricle):
            return np.array(GenParamsCorrelated(total_ventricle), dtype=np.float32)
        if self.train_param_mode == "small":
            return np.array(GenParams_small(), dtype=np.float32)
        if self.train_param_mode == "large":
            return np.array(GenParams_large(), dtype=np.float32)
        return np.array(GenParams(), dtype=np.float32)

    def _select_augmentations(self):
        cfg = self.augmentation_config
        if not self.TrainFlag or not cfg.get("enabled", False):
            return []
        if self.augmentation_mixture is not None:
            ops, probs = zip(*self.augmentation_mixture)
            return list(random.choices(ops, weights=probs, k=1)[0])
        unchanged_prob = float(cfg.get("unchanged_prob", 0.2))
        if random.random() < unchanged_prob:
            return []

        candidates = []
        if cfg.get("use_rician_noise", False):
            candidates.append(("rician", float(cfg.get("rician_noise_prob", 0.4))))
        if cfg.get("use_bias_field", False):
            candidates.append(("bias", float(cfg.get("bias_field_prob", 0.4))))
        if cfg.get("use_deface", False) and self._has_deface_source():
            candidates.append(("deface", float(cfg.get("deface_prob", 0.4))))

        selected = [name for name, prob in candidates if random.random() < prob]
        if not selected and candidates:
            names, probs = zip(*candidates)
            total_prob = sum(probs)
            weights = [prob / total_prob if total_prob > 0 else 1.0 / len(candidates) for prob in probs]
            selected = [random.choices(names, weights=weights, k=1)[0]]
        return selected

    def _get_bias_transform(self):
        if self._bias_transform is None:
            cfg = self.augmentation_config
            self._bias_transform = tio.RandomBiasField(
                coefficients=cfg.get("bias_coefficient_range", (-1.0, 1.0)),
                order=int(cfg.get("bias_order", 3)),
                p=1.0,
            )
        return self._bias_transform

    def _apply_cached_bias_field(self, image):
        array = np.ascontiguousarray(np.asarray(image, dtype=np.float32))
        tensor = torch.from_numpy(array[None])
        try:
            transformed = self._get_bias_transform()(tensor)
        except Exception as exc:
            raise RuntimeError(f"TorchIO RandomBiasField failed for sample bias augmentation: {exc}") from exc
        return transformed.squeeze(0).numpy().astype(np.float32, copy=False)

    def _apply_source_intensity_augmentations(self, image, selected):
        if "rician" not in selected and "bias" not in selected:
            return np.asarray(image, dtype=np.float32)
        cfg = self.augmentation_config
        image = np.asarray(image, dtype=np.float32)

        if "rician" in selected:
            image = _apply_rician_noise(image, cfg.get("rician_sigma_range", (0.0, 0.05)))

        if "bias" in selected:
            image = self._apply_cached_bias_field(image)
        return image.astype(np.float32, copy=False)

    def _read_precomputed_native_face_mask(self, row):
        file_col, index_col = random.choice(self.face_mask_columns)
        hdf5_index = row[index_col]
        if isinstance(hdf5_index, str):
            hdf5_index = int(float(hdf5_index.strip()))
        else:
            hdf5_index = int(hdf5_index)
        hdf5_path = os.path.join(self.face_mask_hdf5_path, str(row[file_col]))
        try:
            with h5py.File(hdf5_path, "r") as hf:
                return _read_hdf5_image(hf, "masks", hdf5_index).astype(np.uint8, copy=False)
        except Exception as exc:
            raise OSError(
                f"Failed to read native face-mask HDF5 file '{hdf5_path}' "
                f"from columns {file_col}/{index_col} at mask index {hdf5_index}. "
                "The file may be corrupted, truncated, or inconsistent with the CSV. "
                "Regenerate this mask HDF5 file with FaceMask_Hdf5_creation.py."
            ) from exc

    def _apply_native_deface_augmentation(self, image, row, selected):
        if "deface" not in selected or not self.face_mask_columns:
            return np.asarray(image, dtype=np.float32)
        mask = self._read_precomputed_native_face_mask(row)
        if mask.shape != image.shape:
            raise ValueError(
                f"Native face mask shape {mask.shape} does not match source image shape {image.shape} "
                f"for subject {row.get('sbj_ID', 'NA')} {row.get('sbj_visit', 'NA')}."
            )
        return (np.asarray(image, dtype=np.float32) * (mask < 0.5).astype(np.float32, copy=False)).astype(np.float32, copy=False)


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
            source_image = _read_hdf5_image(hf, source_image_key, hdf5_index)
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

        selected_aug = self._select_augmentations()
        augmented_source_image = self._apply_source_intensity_augmentations(source_image, selected_aug)
        augmented_source_image = self._apply_native_deface_augmentation(augmented_source_image, row, selected_aug)
       

        reference_image = None
        if self.return_similarity_data and self.return_reference_image and self.similarity_reference == "ori":
            reference_image = GenImageWithAffineMatrix_sitk_hdf5(
                source_image,
                source_spacing,
                source_origin,
                source_direction,
                self.sitk_template_image,
                ori_affine,
            )

        image = GenImageWithAffineMatrix_sitk_hdf5(
            augmented_source_image,
            source_spacing,
            source_origin,
            source_direction,
            self.sitk_template_image,
            final_affine,
        )
        
        image = extract_normalized_patch(image, self.image_size)
        if self.add_noise and self.TrainFlag:
            image = add_noise(image, noise_type="R")
        if self.transform:
            image = self.transform(image)

        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        if self.istemplate:
            image_tensor = torch.cat((image_tensor, self.template), dim=0)
        if self.add_coord:
            image_tensor = torch.cat((image_tensor, self.coords), dim=0)

        sample = {
            "image": image_tensor,
            "subject_id": row["sbj_ID"],
            "subject_visit": row["sbj_visit"],
            "lin_rate": row["lin_motion_rate"],
            "Affine_param": extra_params,
            "dataset_origin": row["dataset_origin"],
            "augmentation_applied": "+".join(selected_aug) if selected_aug else "none",
        }
        if self.return_similarity_data:
            sample.update(
                {
                    "ori_affine": ori_affine.astype(np.float32),
                    "source_image": augmented_source_image.astype(np.float32, copy=False),
                    "source_spacing": np.asarray(source_spacing, dtype=np.float32),
                    "source_origin": np.asarray(source_origin, dtype=np.float32),
                    "source_direction": np.asarray(source_direction, dtype=np.float32),
                    "extra_affine": extra_affine.astype(np.float32),
                }
            )
            if reference_image is not None:
                sample["reference_image"] = torch.from_numpy(np.ascontiguousarray(reference_image)).float().unsqueeze(0)
        return sample
