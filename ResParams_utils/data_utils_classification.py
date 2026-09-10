import os
import re

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from ResParams_utils.data_utils_image_cropping_logic import extract_normalized_patch, load_image_func


def _parse_subject_id(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value)
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else None


def _coord_map_3d(shape, device=None, dtype=torch.float32, start=-1.0, end=1.0):
    d, h, w = shape
    z = torch.linspace(start, end, steps=d, device=device, dtype=dtype)
    y = torch.linspace(start, end, steps=h, device=device, dtype=dtype)
    x = torch.linspace(start, end, steps=w, device=device, dtype=dtype)

    zz = z[:, None, None].expand(d, h, w)
    yy = y[None, :, None].expand(d, h, w)
    xx = x[None, None, :].expand(d, h, w)
    return torch.stack([xx, yy, zz], dim=0)


class MRIDatasetWithParams(Dataset):
    def __init__(
        self,
        image_csv,
        hdf5_root,
        params_csv=None,
        transform=None,
        image_size=(192, 224, 192),
        template=False,
        template_path=None,
        add_coord=False,
        LazyLoading=True,
        id_col="sbj_ID",
        params_id_col="subject_id",
        label_col="lin_motion_rate",
        param_prefix="pred_param_",
    ):
        super().__init__()
        self.transform = transform
        self.image_size = image_size
        self.hdf5_root = hdf5_root
        self.add_coord = add_coord
        self.LazyLoading = LazyLoading

        info_df = pd.read_csv(image_csv)
        for col in ("hdf5_file", "hdf5_index"):
            if col not in info_df.columns:
                raise ValueError(f"image_csv missing required column '{col}'")
        if id_col not in info_df.columns:
            raise ValueError(f"image_csv missing id column '{id_col}'")
        if label_col not in info_df.columns:
            raise ValueError(f"image_csv missing label column '{label_col}'")
        for col in ("age_years", "sex"):
            if col not in info_df.columns:
                raise ValueError(f"image_csv missing required column '{col}'")

        info_df["subject_id_clean"] = info_df[id_col].apply(_parse_subject_id)
        info_df = info_df.dropna(subset=["subject_id_clean"]).reset_index(drop=True)

        if params_csv:
            params_df = pd.read_csv(params_csv)
            if params_id_col not in params_df.columns:
                raise ValueError(f"params_csv missing id column '{params_id_col}'")
            params_df["subject_id_clean"] = params_df[params_id_col].apply(_parse_subject_id)
            params_df = params_df.dropna(subset=["subject_id_clean"])
            params_cols = [c for c in params_df.columns if c.startswith(param_prefix)]
            if len(params_cols) != 9:
                raise ValueError(
                    f"Expected 9 '{param_prefix}*' columns in params_csv, got {len(params_cols)}"
                )
            params_df = params_df.drop_duplicates(subset=["subject_id_clean"])
            params_map = {
                int(row["subject_id_clean"]): row[params_cols].to_numpy(dtype=np.float32)
                for _, row in params_df.iterrows()
            }
            info_df = info_df[info_df["subject_id_clean"].isin(params_map.keys())].reset_index(drop=True)
            self.params_map = params_map
            self.params_cols = params_cols
        else:
            params_cols = [c for c in info_df.columns if c.startswith(param_prefix)]
            if len(params_cols) != 9:
                raise ValueError(
                    f"Expected 9 '{param_prefix}*' columns in image_csv, got {len(params_cols)}"
                )
            self.params_map = None
            self.params_cols = params_cols

        self.total_samples = len(info_df)
        self.label_col = label_col
        self.id_col = id_col

        if template:
            if not template_path:
                raise ValueError("template_path is required when template=True")
            temp_img = load_image_func(template_path)
            temp_img = extract_normalized_patch(temp_img, self.image_size)
            if self.transform:
                temp_img = self.transform(temp_img)
            self.template = torch.from_numpy(temp_img).float().unsqueeze(0)
        else:
            self.template = None

        self.coord_map = _coord_map_3d(self.image_size) if self.add_coord else None

        if self.LazyLoading:
            self.info = info_df
            self.in_memory_data = None
        else:
            self.info = None
            self.in_memory_data = []
            unique_h5_files = info_df["hdf5_file"].unique()
            print("Non-lazy loading mode: reading all data listed in the csv file into RAM")
            for h5_path in unique_h5_files:
                file_subset = info_df[info_df["hdf5_file"] == h5_path]
                h5_full_path = os.path.join(self.hdf5_root, h5_path)
                with h5py.File(h5_full_path, "r") as hf:
                    hdf5_images = hf["imgs"]
                    for _, row in tqdm(
                        file_subset.iterrows(),
                        total=len(file_subset),
                        desc=f"Reading {os.path.basename(h5_full_path)}",
                    ):
                        h5_idx = int(row["hdf5_index"])
                        image = hdf5_images[h5_idx]
                        image = extract_normalized_patch(image, self.image_size)

                        subject_id = int(row["subject_id_clean"])
                        if self.params_map is not None:
                            params = self.params_map[subject_id]
                        else:
                            params = row[self.params_cols].to_numpy(dtype=np.float32)

                        self.in_memory_data.append(
                            {
                                "image": image,
                                "subject_id": subject_id,
                                "age": row["age_years"],
                                "sex": row["sex"],
                                "label": row[self.label_col],
                                "params": params,
                            }
                        )

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        if index < 0 or index >= self.total_samples:
            raise IndexError(f"Index {index} out of range for dataset of size {self.total_samples}")

        if self.LazyLoading:
            row = self.info.iloc[index]
            hdf5_file = os.path.join(self.hdf5_root, row["hdf5_file"])
            hdf5_index = int(row["hdf5_index"])
            with h5py.File(hdf5_file, "r") as hf:
                image = hf["imgs"][hdf5_index]
                image = extract_normalized_patch(image, self.image_size)

            subject_id = int(row["subject_id_clean"])
            if self.params_map is not None:
                params = self.params_map[subject_id]
            else:
                params = row[self.params_cols].to_numpy(dtype=np.float32)
            age = row["age_years"]
            sex = row["sex"]
            label = row[self.label_col]
        else:
            sample = self.in_memory_data[index]
            image = sample["image"]
            subject_id = sample["subject_id"]
            age = sample["age"]
            sex = sample["sex"]
            params = sample["params"]
            label = sample["label"]

        if self.transform:
            image = self.transform(image)

        image = torch.from_numpy(image).float().unsqueeze(0)
        if self.template is not None:
            image = torch.cat((image, self.template), dim=0)
        if self.coord_map is not None:
            image = torch.cat((image, self.coord_map), dim=0)

        params = torch.from_numpy(np.asarray(params, dtype=np.float32)).float()
        label = torch.tensor(label).float()

        return {
            "image": image,
            "params": params,
            "label": label,
            "subject_id": subject_id,
            "age": age,
            "sex": sex,
        }
