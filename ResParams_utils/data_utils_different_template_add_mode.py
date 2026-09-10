from cProfile import label
import os
import torch
import pandas as pd
import numpy as np
import os.path as path
from warnings import warn
from torch.utils.data import Dataset, sampler
import SimpleITK as sitk
import random
import matplotlib.pyplot as plt
import nibabel as nib
import h5py
import math
from math import sin
from scipy.ndimage import affine_transform
from tqdm import tqdm
import ast
import time
import torch.nn as nn
import yaml
from model.ResParam_model.ResNet18 import ResNet18
from model.ResParam_model.CNN_5layers import CNN_5layers
from model.ResParam_model.CNN_5layers_average_pooling import CNN_5layers_AP
from model.ResParam_model.VGG16 import VGG16
from model.ResParam_model.CNNTrans_GN import CNNViT3DRegressor

DEFAULT_AFFINE_PARAMS = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=np.float32)
TEMPLATE_MODES = {"without", "withstatic", "withsameaffine", "withdiffaffine"}


# csv files should contain hdf5 index for each subject and hdf5 path. So, we can know where the MRI image for a specific subject is in these hdf5 files. And this is for easier loading.
# While for lazy loading, the writing logic will changes a lot and may takes more time because we couldn't load all the MRI imgs in these hdf5 files and they may belong to different types of datasets(train,val,test).
class MRIDatasetsAug(Dataset):
    """
    In this function, we load the training dataset with TrainFlag=1 and randomly sample
    failed cases for the training set.

    To locate data in HDF5 files, we follow Raven's lazy-loading approach. We first build an
    index from all HDF5 files in the target folder after scanning them. Then we use a binary
    search to retrieve the corresponding image and label entries efficiently.

    all_hdf5_Paths: List of paths to text files or direct .hdf5 files. Each text file should contain one HDF5 file path per line.
    """

    def __init__(
        self,
        csv_path,
        hdf5s_path,
        transform=None,
        TrainFlag=False,
        LazyLoading=True,
        center_based_transform=False,
        template=False,
        template_path=' ',
        image_size=(192, 224, 192),
        add_coord=False,
        add_noise=False,
        template_mode=None,
    ):
        super().__init__()
        self.transform = transform
        self.TrainFlag = TrainFlag
        self.LazyLoading = LazyLoading
        if center_based_transform:
            raise ValueError(
                "data_utils_different_template_add_mode.py only supports non-center-based affine transformations. "
                "Set center_based_transform=False."
            )
        self.add_noise = add_noise
        self.image_size = image_size
        self.hdf5s_path = hdf5s_path
        self.add_coord = add_coord
        self.template_mode = self._resolve_template_mode(template_mode, template)
        self.istemplate = self.template_mode != "without"

        info_df = pd.read_csv(csv_path)
        self.total_samples = len(info_df)
        self.has_transform_col = 'transform_params' in info_df.columns

        if self.istemplate:
            temp_img = load_image_func(template_path)
            self.template_np = extract_normalized_patch(temp_img, self.image_size)
            if self.template_mode == "withstatic":
                self.template = self._image_to_tensor(self.template_np)

        if self.add_coord:
            self.coords = self._coord_map_3d(self.image_size)

        if self.LazyLoading:
            self.info = info_df
        else:
            self.in_memory_data = []
            unique_h5_files = info_df['hdf5_file'].unique()
            print("Non-lazy loading mode: reading all data listed in the csv file into RAM")
            for h5_path in unique_h5_files:
                file_subset = info_df[info_df['hdf5_file'] == h5_path]
                fhf_path = os.path.join(self.hdf5s_path, h5_path)
                with h5py.File(fhf_path, 'r') as hf:
                    hdf5_images = hf['imgs']
                    for _, row in tqdm(file_subset.iterrows(), total=len(file_subset), desc=f"Reading {fhf_path.split('/')[-1]}"):
                        h5_idx = row['hdf5_index']
                        if isinstance(h5_idx, str):
                            h5_idx = int(float(h5_idx.strip()))
                        else:
                            h5_idx = int(h5_idx)
                        image = extract_normalized_patch(hdf5_images[h5_idx], self.image_size)
                        self.in_memory_data.append({
                            'image': image,
                            'subject_id': row['sbj_ID'],
                            'subject_visit': row['sbj_visit'],
                            'lin_rate': row['lin_motion_rate'],
                            'Affine_param': self._parse_affine_params(row.get('transform_params')),
                            'dataset_origin': row.get('dataset_origin', 'Unknown'),
                        })

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        if index < 0 or index >= self.total_samples:
            raise IndexError(f"Index {index} out of range for dataset of size {self.total_samples}")

        if self.LazyLoading:
            row = self.info.iloc[index]
            hdf5_file_path = os.path.join(self.hdf5s_path, row['hdf5_file'])
            hdf5_index = row['hdf5_index']
            if isinstance(hdf5_index, str):
                hdf5_index = int(float(hdf5_index.strip()))
            else:
                hdf5_index = int(hdf5_index)
            try:
                with h5py.File(hdf5_file_path, "r") as hf:
                    image = extract_normalized_patch(hf['imgs'][hdf5_index], self.image_size)
                subject_id = row['sbj_ID']
                subject_visit = row['sbj_visit']
                lin_rate = row['lin_motion_rate']
                dataset_origin = row.get('dataset_origin', 'Unknown')
                base_affine_param = self._parse_affine_params(row.get('transform_params'))
            except KeyError as e:
                raise KeyError(f"Key {e} not found in file {hdf5_file_path}")
            except Exception as e:
                raise RuntimeError(f"Error accessing data in {hdf5_file_path}: {e}")
        else:
            sample = self.in_memory_data[index]
            image = sample['image']
            subject_id = sample['subject_id']
            subject_visit = sample['subject_visit']
            lin_rate = sample['lin_rate']
            base_affine_param = sample['Affine_param']
            dataset_origin = sample['dataset_origin']

        image_affine_param, template_affine_param, target_affine_param = self._resolve_sample_affines(base_affine_param)

        image = self._apply_affine(image, image_affine_param)
        if self.transform:
            image = self.transform(image)

        if self.add_noise and self.TrainFlag:
            image = add_noise(image, noise_type='R')

        image = torch.from_numpy(image).float().unsqueeze(0)

        if self.istemplate:
            template_tensor = self._build_template_tensor(template_affine_param)
            if self.transform:
                template_tensor = self.transform(template_tensor)
            image = torch.cat((image, template_tensor), dim=0)

        if self.add_coord:
            image = torch.cat((image, self.coords), dim=0)

        return {
            'image': image,
            'subject_id': subject_id,
            'subject_visit': subject_visit,
            'lin_rate': lin_rate,
            'Affine_param': target_affine_param,
            'dataset_origin': dataset_origin,
        }

    def _resolve_template_mode(self, template_mode, template):
        if template_mode is None:
            return "withstatic" if template else "without"
        mode = str(template_mode).strip().lower()
        if mode not in TEMPLATE_MODES:
            raise ValueError(f"Unknown template_mode '{template_mode}'. Expected one of {sorted(TEMPLATE_MODES)}.")
        return mode

    def _parse_affine_params(self, raw_val):
        if not self.has_transform_col or raw_val is None:
            return DEFAULT_AFFINE_PARAMS.copy()
        if isinstance(raw_val, (float, np.floating)) and pd.isna(raw_val):
            return DEFAULT_AFFINE_PARAMS.copy()

        parsed = ast.literal_eval(raw_val) if isinstance(raw_val, str) else raw_val
        affine_array = np.asarray(parsed, dtype=np.float32)
        if affine_array.shape != (9,):
            raise ValueError(f"Expected transform_params with shape (9,), got {affine_array.shape}.")
        return affine_array.copy()

    def _resolve_sample_affines(self, base_affine_param):
        image_affine_param = np.asarray(base_affine_param, dtype=np.float32).copy()
        template_affine_param = DEFAULT_AFFINE_PARAMS.copy()

        if self.TrainFlag:
            image_affine_param = np.asarray(GenParams(), dtype=np.float32)

        if self.template_mode == "withsameaffine":
            template_affine_param = image_affine_param.copy()
            target_affine_param = image_affine_param.copy()
        elif self.template_mode == "withdiffaffine":
            if self.TrainFlag:
                template_affine_param = np.asarray(GensmallParams(), dtype=np.float32)
            target_affine_param = get_relative_affine_params(
                moving_params=template_affine_param,
                fixed_params=image_affine_param,
            )
        else:
            target_affine_param = image_affine_param.copy()

        return image_affine_param, template_affine_param, target_affine_param.astype(np.float32)

    def _apply_affine(self, image, affine_param):
        _, transformed_image = GenImageWithParams(image, affine_param)
        return transformed_image

    def _image_to_tensor(self, image):
        if self.transform:
            image = self.transform(image)
        return torch.from_numpy(np.asarray(image, dtype=np.float32)).float().unsqueeze(0)

    def _build_template_tensor(self, template_affine_param):
        if self.template_mode == "withstatic":
            return self.template

        transformed_template = self._apply_affine(self.template_np, template_affine_param)
        return self._image_to_tensor(transformed_template)

    def _coord_map_3d(self, image_size, device=None, dtype=torch.float32, start=-1.0, end=1.0):
        D, H, W = image_size
        z = torch.linspace(start, end, steps=D, device=device, dtype=dtype)
        y = torch.linspace(start, end, steps=H, device=device, dtype=dtype)
        x = torch.linspace(start, end, steps=W, device=device, dtype=dtype)

        zz = z[:, None, None].expand(D, H, W)
        yy = y[None, :, None].expand(D, H, W)
        xx = x[None, None, :].expand(D, H, W)

        return torch.stack([xx, yy, zz], dim=0)

'''
class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        w = torch.as_tensor(weights, dtype=torch.float32)
        self.register_buffer("w", w)   # <- key
        MSE=torch.nn.MSELoss()
    def forward(self, pred, target):
        err2 = (pred - target)
        return err2
''' 
class MinMaxNormalization(object):
   def __call__(self, image):
       return (image - image.min()) / (image.max() - image.min())
   
class MinMax01Normalization(object):
   def __call__(self, image):
       return (image - image.min()) / (image.max() - image.min())
   
class MinMax11Normalization(object):
   def __call__(self, image):
       return ((image - image.min()) / (image.max() - image.min())*2)-1

class GaussianNormalization(object):
   def __call__(self, image):
       return (image-image.mean()) / image.std()

class Normalization(object):
   def __init__(self,normalization):
       if normalization is not None:
           self.transform = eval(normalization+'Normalization')()
       else:
           self.transform = None
   def __call__(self, image):
       if self.transform is not None:
           return self.transform(image)
       return image

def GensmallParams():
   params=np.zeros(9,dtype=float)
   # keep all changes small
   for i in range(0,9):
        if i in(3,4,5):
            params[i]=random.uniform(0.97,1.03) #(0.85,1.15)
        if i in(0,1,2):
            params[i]=random.uniform(-1,1) #(-5,5)
        if i in(6,7,8):
            params[i]=random.uniform(-1.4,1.4) #(-7,7)
   return params
def GenParams_large():
   params=np.zeros(9,dtype=float)
   # keep all changes small
   for i in range(0,9):
        if i in(3,4,5):
            params[i]=random.uniform(0.7,1.3) #(0.85,1.15)
        if i in(0,1,2):
            params[i]=random.uniform(-10,10) #(-5,5)
        if i in(6,7,8):
            params[i]=random.uniform(-14,14) #(-7,7)
   return params


def GenParams():
   params=np.zeros(9,dtype=float)
   # keep all changes small
   for i in range(0,9):
        if i in(3,4,5):
            params[i]=random.uniform(0.85,1.15)
        if i in(0,1,2):
            params[i]=random.uniform(-5,5)
        if i in(6,7,8):
            params[i]=random.uniform(-7,7)
   return params


def build_linear_transform(rotation_angle_x, rotation_angle_y, rotation_angle_z, scale_x, scale_y, scale_z):
    angle_rad_x = np.deg2rad(rotation_angle_x)
    angle_rad_y = np.deg2rad(rotation_angle_y)
    angle_rad_z = np.deg2rad(rotation_angle_z)

    rot_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad_x), -np.sin(angle_rad_x)],
        [0, np.sin(angle_rad_x), np.cos(angle_rad_x)],
    ], dtype=np.float32)
    rot_matrix_y = np.array([
        [np.cos(angle_rad_y), 0, np.sin(angle_rad_y)],
        [0, 1, 0],
        [-np.sin(angle_rad_y), 0, np.cos(angle_rad_y)],
    ], dtype=np.float32)
    rot_matrix_z = np.array([
        [np.cos(angle_rad_z), -np.sin(angle_rad_z), 0],
        [np.sin(angle_rad_z), np.cos(angle_rad_z), 0],
        [0, 0, 1],
    ], dtype=np.float32)

    rotation_matrix = rot_matrix_x @ rot_matrix_y @ rot_matrix_z
    scale_matrix = np.diag([scale_x, scale_y, scale_z]).astype(np.float32)
    return rotation_matrix @ scale_matrix


def build_affine_matrix(params):
    params = np.asarray(params, dtype=np.float32)
    if params.shape != (9,):
        raise ValueError(f"Expected params with shape (9,), got {params.shape}.")

    rotation_angle_x, rotation_angle_y, rotation_angle_z, scale_x, scale_y, scale_z, translation_x, translation_y, translation_z = params
    transformation_matrix = build_linear_transform(
        rotation_angle_x,
        rotation_angle_y,
        rotation_angle_z,
        scale_x,
        scale_y,
        scale_z,
    )

    translation_vector = np.array([translation_x, translation_y, translation_z], dtype=np.float32)

    new_affine = np.eye(4, dtype=np.float32)
    new_affine[:3, :3] = transformation_matrix
    new_affine[:3, 3] = translation_vector
    return new_affine


def _project_linear_matrix_to_params(linear_matrix):
    rotation_scale = np.asarray(linear_matrix, dtype=np.float64)
    scale = np.linalg.norm(rotation_scale, axis=0)
    scale = np.where(scale < 1e-8, 1e-8, scale)

    rotation_guess = rotation_scale / scale
    u, _, vh = np.linalg.svd(rotation_guess)
    rotation_matrix = u @ vh
    if np.linalg.det(rotation_matrix) < 0:
        u[:, -1] *= -1
        rotation_matrix = u @ vh

    sy = float(np.clip(rotation_matrix[0, 2], -1.0, 1.0))
    rotation_angle_y = math.asin(sy)
    cos_y = math.cos(rotation_angle_y)

    if abs(cos_y) > 1e-6:
        rotation_angle_x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[2, 2])
        rotation_angle_z = math.atan2(-rotation_matrix[0, 1], rotation_matrix[0, 0])
    else:
        rotation_angle_z = 0.0
        if sy >= 0:
            rotation_angle_x = math.atan2(rotation_matrix[2, 1], rotation_matrix[1, 1])
        else:
            rotation_angle_x = math.atan2(-rotation_matrix[2, 1], rotation_matrix[1, 1])

    return np.array([
        np.rad2deg(rotation_angle_x),
        np.rad2deg(rotation_angle_y),
        np.rad2deg(rotation_angle_z),
        scale[0],
        scale[1],
        scale[2],
    ], dtype=np.float32)


def affine_matrix_to_params(affine_matrix):
    affine_matrix = np.asarray(affine_matrix, dtype=np.float64)
    if affine_matrix.shape != (4, 4):
        raise ValueError(f"Expected affine_matrix with shape (4, 4), got {affine_matrix.shape}.")

    params = _project_linear_matrix_to_params(affine_matrix[:3, :3])
    translation_vector = affine_matrix[:3, 3].astype(np.float32)

    return np.concatenate([params, translation_vector.astype(np.float32)]).astype(np.float32)


def get_relative_affine_params(moving_params, fixed_params):
    moving_affine = build_affine_matrix(moving_params)
    fixed_affine = build_affine_matrix(fixed_params)
    # The exact relative affine can contain shear, so project it back to the
    # rotation + anisotropic-scale + translation parameterization used by the model.
    relative_affine = fixed_affine @ np.linalg.inv(moving_affine)
    return affine_matrix_to_params(relative_affine)

def add_noise(x, vmap=None, noise_type='R', min_amp=0.0, max_amp=0.05):
    """
    x: Ground truth image (assumed normalized 0-1 for amplitude logic)
    vmap: Spatially varying noise map (mask)
    noise_type: 'G' for Gaussian, 'R' for Rician
    """
    # 1. Determine amplitude relative to max intensity if x isn't 0-1
    # If x is already 0-1, max_val is 1.0.
    max_val = np.max(x) if np.max(x) > 0 else 1.0
    noise_level = np.random.uniform(min_amp, max_amp) * max_val
    
    # Cast to single precision early to save memory/time
    x = x.astype(np.single)
    
    if noise_type == 'G':
        noises = np.random.normal(scale=noise_level, size=x.shape).astype(np.single)
        x_noise = x + noises
        
    elif noise_type == 'R':
        # Prepare noise components
        noiseR = np.random.normal(scale=noise_level, size=x.shape).astype(np.single)
        noiseI = np.random.normal(scale=noise_level, size=x.shape).astype(np.single)
        
        # Apply spatial variation map if provided
        if vmap is not None:
            # Ensure vmap is single precision and matches x shape logic
            # (Adjust moveaxis if your vmap shape is specifically transposed)
            vmap = vmap.astype(np.single)
            noiseR *= vmap
            noiseI *= vmap
            
        # Rician Calculation: Magnitude of (Signal + Real Noise) + j(Imaginary Noise)
        x_noise = np.sqrt((x + noiseR)**2 + (noiseI)**2)
    
    else:
        return x

    # Final clip to maintain valid range and return
    return np.clip(x_noise, 0.0, max_val).astype(np.single)
    
def extract_normalized_patch(data, patch_size=(192,224,192)):
    """
    Return a centred patch of shape `patch_size`.
    Uses memory-mapping where possible to avoid loading the whole volume.
    """

    X, Y, Z = data.shape
    px, py, pz = patch_size
    sx = max(0, (X - px) // 2)
    sy = max(0, (Y - py) // 2)
    sz = max(0, (Z - pz) // 2)

    patch = data[sx:sx + px, sy:sy + py, sz:sz + pz]

    # If the source is smaller than the patch, zero-pad to fit
    if patch.shape != patch_size:
        patch = np.pad(
            patch,
            ((0, px - patch.shape[0]),
             (0, py - patch.shape[1]),
             (0, pz - patch.shape[2])),
            mode='constant',
            constant_values=0
        )
    ''' we don't want to the image is normalized now. It's better to normalize the image after the extra linear transformation
    vmin, vmax = patch.min(), patch.max()
    if vmax > vmin:
        patch = (patch - vmin) / (vmax - vmin)
    else:
        patch.fill(0.)
    '''

    return patch.astype(np.float32)

def restore_image_size(trimmed_image, target_size=(193, 229, 193)):
    """
    Restores the image size from trimmed_size (e.g., 192x224x192) 
    to target_size (e.g., 193x229x193) using centre padding with zeros.
    
    Args:
        trimmed_image (np.ndarray): The input image with shape (P_x, P_y, P_z).
        target_size (tuple): The desired output shape (T_x, T_y, T_z).
        
    Returns:
        np.ndarray: The restored image with shape target_size.
    """
    tx, ty, tz = target_size
    px, py, pz = trimmed_image.shape
    
    # calculate Padding Width
    if px > tx or py > ty or pz > tz:
        raise ValueError("Target size must be greater than or equal to the trimmed image size.")

    # calculate padding number at the start and end entity
    pad_x_start = (tx - px) // 2
    pad_y_start = (ty - py) // 2
    pad_z_start = (tz - pz) // 2
    
    pad_x_end = (tx - px) - pad_x_start
    pad_y_end = (ty - py) - pad_y_start
    pad_z_end = (tz - pz) - pad_z_start
    
    padding = (
        (pad_x_start, pad_x_end),
        (pad_y_start, pad_y_end),
        (pad_z_start, pad_z_end)
    )
    
    #padding
    restored_image = np.pad(
        trimmed_image, 
        padding, 
        mode='edge', 
    )
    
    return restored_image

def displacement_vox(W=193,H=229,D=193,Affine_matrix=[]):
    x,y,z = np.meshgrid(np.arange(W),np.arange(H),np.arange(D),indexing='ij')
    pts = np.stack((x.flatten(),y.flatten(),z.flatten(),np.ones(W*H*D)))
    transformed_pts = Affine_matrix @ pts
    disp = transformed_pts[:3,:] - pts[:3,:]

    sq_mag = np.sum(disp**2, axis=0)          
    rmse = float(np.mean(np.sqrt(sq_mag)))

    mean_abs_dx = float(np.mean(np.abs(disp[0, :])))
    mean_abs_dy = float(np.mean(np.abs(disp[1, :])))
    mean_abs_dz = float(np.mean(np.abs(disp[2, :])))

    return rmse, mean_abs_dx, mean_abs_dy, mean_abs_dz

def GenImageWithParams(image,params):
   new_affine = build_affine_matrix(params)
   transform_to_apply = np.linalg.inv(new_affine)

   transformed_image = affine_transform(
       image,
       transform_to_apply,
       output_shape=image.shape,
       order=1,
       mode='constant',
       cval=0
   )
   return new_affine, transformed_image


def GenImagereverse_WithParams(image,params):
   new_affine = build_affine_matrix(params)
   transform_to_apply = new_affine

   transformed_image = affine_transform(
       image,
       transform_to_apply,
       output_shape=image.shape,
       order=1,
       mode='constant',
       cval=0
   )
   return new_affine, transformed_image

def load_image_func(filepath):
    image_sitk = sitk.ReadImage(filepath)
    ori_data = sitk.GetArrayFromImage(image_sitk)
    return ori_data

def plot_mri_orthoview_with_outline(mri_data, outline_data,output_path):
    """
    Loads an MRI volume and an outline volume (MINC files), extracts the three
    medial orthogonal slices, and plots the MRI with the outline as a contour overlay
    for each view (Sagittal, Coronal, Axial).

    :param mri_path: Path to the subject MRI file.
    :param outline_path: Path to the outline file.
    """
    x_center = mri_data.shape[0] // 2
    y_center = mri_data.shape[1] // 2 
    z_center = mri_data.shape[2] // 2 

    slices_to_plot = [
        (mri_data[x_center, :, :], outline_data[x_center, :, :], f"Sagittal (X={x_center})", 0), 
        (mri_data[:, y_center, :], outline_data[:, y_center, :], f"Coronal (Y={y_center})", 0),
        (mri_data[:, :, z_center], outline_data[:, :, z_center], f"Axial (Z={z_center})", 0), 
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    outline_color = 'red'
    
    for i, (mri_slice, outline_slice, title, rotation) in enumerate(slices_to_plot):
        ax = axes[i]
        if rotation == 1:
            mri_slice = np.rot90(mri_slice)
            outline_slice = np.rot90(outline_slice)
            
        ax.imshow(mri_slice, cmap='gray', origin='lower')

        outline_mask = outline_slice > 0
        
        ax.contour(
            outline_mask, 
            levels=[0.5],          
            colors=outline_color,     
            linewidths=1.5,
            alpha=0.8,
            origin='lower'
        )
        
        ax.set_title(title, fontsize=10)
        ax.axis('off') 
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure successfully saved to: {output_path}")

def _generate_stats_subtitle(data, is_transformed=False):
    """
    Calculates mean, std, min, max for the ENTIRE 3D data volume and 
    formats the title string. 
    
    The statistics calculation is now done BEFORE any slicing.
    """
    
    if isinstance(data, torch.Tensor):
        data = data.squeeze().cpu().numpy()
        
    data_mean = np.mean(data)
    data_std = np.std(data)
    data_min = np.min(data)
    data_max = np.max(data)
    
    prefix = "TRANSFORMED" if is_transformed else "RAW"
    
    title_string = (
        f"{prefix} Image | Mean: {data_mean:.3f}, Std: {data_std:.3f}\n"
        f"Min: {data_min:.3f}, Max: {data_max:.3f}"
    )
    return title_string

def plot_mri_orthoview_comparison(
    raw_mri_data, 
    transformed_mri_data, 
    outline_data, 
    label, 
    output_path
):
    """
    Plots the orthographic view of the RAW MRI and the TRANSFORMED MRI side-by-side.
    
    NOTE: The statistics displayed in the title are calculated on the 
    FULL 3D VOLUME (raw_mri_data or transformed_mri_data), not the slice.
    """
    
    raw_stats_title = _generate_stats_subtitle(raw_mri_data, is_transformed=False)
    trans_stats_title = _generate_stats_subtitle(transformed_mri_data, is_transformed=True)
    
    x_center = raw_mri_data.shape[0] // 2
    y_center = raw_mri_data.shape[1] // 2
    z_center = raw_mri_data.shape[2] // 2

    outline_color = 'red'
    
    view_data = [
        ("Sagittal", x_center, lambda data: data[x_center, :, :], 0), 
        ("Coronal", y_center, lambda data: data[:, y_center, :], 0), 
        ("Axial", z_center, lambda data: data[:, :, z_center], 0),   
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"MRI Comparison (Raw vs. Transformed) | Label: {label}", fontsize=16)

    for i, (view_name, center_idx, slice_extractor, rotation) in enumerate(view_data):
        
        ax_raw = axes[0, i]
        raw_slice = slice_extractor(raw_mri_data)
        outline_slice = slice_extractor(outline_data)
        
        if rotation == 1:
            raw_slice = np.rot90(raw_slice)
            outline_slice_raw = np.rot90(outline_slice)
        else:
            outline_slice_raw = outline_slice

        ax_raw.imshow(raw_slice, cmap='gray', origin='lower')
        ax_raw.contour(
            outline_slice_raw > 0, levels=[0.5], colors=outline_color, 
            linewidths=1.5, alpha=0.8, origin='lower'
        )
        
        ax_raw.set_title(f"{view_name}\n{raw_stats_title}", fontsize=9)
        ax_raw.axis('off')
        
        ax_trans = axes[1, i]
        trans_slice = slice_extractor(transformed_mri_data)
        
        if rotation == 1:
            trans_slice = np.rot90(trans_slice)
            outline_slice_trans = np.rot90(outline_slice)
        else:
            outline_slice_trans = outline_slice
            
        ax_trans.imshow(trans_slice, cmap='gray', origin='lower')
        ax_trans.contour(
            outline_slice_trans > 0, levels=[0.5], colors=outline_color, 
            linewidths=1.5, alpha=0.8, origin='lower'
        )
        
        ax_trans.set_title(f"{view_name}\n{trans_stats_title}", fontsize=9)
        ax_trans.axis('off')
        
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison figure successfully saved to: {output_path}")
def show_orth_views(img, x=None, y=None, z=None, cmap="gray"):
    """
    img: 3D numpy array, shape (X, Y, Z)
    x/y/z: slice indices (optional). If None, use center slices.
    """
    img = np.asarray(img)
    assert img.ndim == 3, f"Expected 3D array, got shape {img.shape}"

    X, Y, Z = img.shape
    if x is None: x = X // 2  # sagittal index
    if y is None: y = Y // 2  # coronal index
    if z is None: z = Z // 2  # axial index

    # Slices (transpose for nicer display: left-right, up-down)
    sag = img[x, :, :].T      # sagittal (YZ)
    cor = img[:, y, :].T      # coronal  (XZ)
    axi = img[:, :, z].T      # axial    (XY)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(sag, cmap=cmap, origin="lower")
    axes[0].set_title(f"Sagittal (x={x})")
    axes[0].axis("off")

    axes[1].imshow(cor, cmap=cmap, origin="lower")
    axes[1].set_title(f"Coronal (y={y})")
    axes[1].axis("off")

    axes[2].imshow(axi, cmap=cmap, origin="lower")
    axes[2].set_title(f"Axial (z={z})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

MODEL_REGISTRY = {
    "ResNet18": ResNet18,
    "CNN_5layers": CNN_5layers,
    "CNN_5layers_AP": CNN_5layers_AP,
    "VGG16": VGG16,
    "ResTrans_CNNViT3D": CNNViT3DRegressor,
}
def build_model(name, *, input_size, dropout=0.2):
    try:
        model_cls = MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown model_name '{name}'. Options: {sorted(MODEL_REGISTRY)}")
    return model_cls(input_size=input_size, dropout=dropout)

def _load_config(path):
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to load --config files. Please install pyyaml.")
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a top-level mapping of argument names to values.")
    return data
