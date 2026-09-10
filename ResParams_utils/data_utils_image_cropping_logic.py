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

def _axis_grid(shape, axis):
    grid_shape = [1, 1, 1]
    grid_shape[axis] = shape[axis]
    return np.arange(shape[axis], dtype=np.float32).reshape(grid_shape)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

def _sample_range_value(value_or_range, rng, name):
    values = np.asarray(value_or_range, dtype=np.float32).reshape(-1)
    if values.size == 1:
        return float(values[0])
    if values.size != 2:
        raise ValueError(f"{name} must be one value or two values for a random range.")
    low, high = float(values[0]), float(values[1])
    if high < low:
        raise ValueError(f"{name} range must be ordered as [low, high].")
    return float(rng.uniform(low, high))

def _soft_or_binary_keep(distance, transition_width, binary_keep):
    if transition_width <= 1e-6:
        return binary_keep.astype(np.float32)
    return _sigmoid(distance / transition_width).astype(np.float32)

def _load_and_dilate_mask(mask_path, radius_voxels):
    mask_image = sitk.ReadImage(mask_path)
    mask_uint8 = sitk.Cast(mask_image > 0, sitk.sitkUInt8)
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelRadius(tuple(int(v) for v in radius_voxels))
    dilate_filter.SetForegroundValue(1)
    return dilate_filter.Execute(mask_uint8)

def _image_geometry_matches(image_a, image_b):
    return (
        image_a.GetSize() == image_b.GetSize()
        and np.allclose(image_a.GetSpacing(), image_b.GetSpacing())
        and np.allclose(image_a.GetOrigin(), image_b.GetOrigin())
        and np.allclose(image_a.GetDirection(), image_b.GetDirection())
    )

def _resample_mask_to_reference(mask_image, reference_image):
    if _image_geometry_matches(mask_image, reference_image):
        return mask_image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(mask_image)

def _read_hdf5_image(hf, image_key, h5_idx):
    image_container = hf[image_key]
    if isinstance(image_container, h5py.Group):
        dataset_name = str(int(h5_idx))
        if dataset_name not in image_container:
            raise KeyError(f"HDF5 image group '{image_key}' does not contain dataset '{dataset_name}'.")
        return image_container[dataset_name][:].astype(np.float32, copy=False)
    return image_container[int(h5_idx)].astype(np.float32, copy=False)

def _sample_neck_face_crop_mode(probabilities):
    modes = ("none", "both", "neck", "face")
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape != (4,):
        raise ValueError("neck_face_crop_mode_probabilities must have four values: none, both, neck, face.")
    if np.any(probs < 0):
        raise ValueError("neck_face_crop_mode_probabilities cannot contain negative values.")
    total = probs.sum()
    if total <= 0:
        raise ValueError("neck_face_crop_mode_probabilities must sum to a positive value.")
    probs = probs / total
    return str(np.random.choice(modes, p=probs))

def create_slanted_neck_face_keep_mask(
    shape,
    safe_brain,
    mode,
    neck_z_range=(0, 49),
    face_y_range=(167, 214),
    soft=True,
    transition_width=4.0,
    tilt_voxels_range=(0.0, 10.0),
):
    safe_brain = np.asarray(safe_brain, dtype=bool)
    rng = np.random.default_rng()
    transition_width = _sample_range_value(transition_width, rng, "neck_face_crop_transition_width")

    neck_start, neck_stop = neck_z_range
    face_start, face_stop = face_y_range
    neck_start = max(0, int(neck_start))
    neck_stop = shape[0] if neck_stop is None else min(shape[0], int(neck_stop))
    face_start = max(0, int(face_start))
    face_stop = shape[1] if face_stop is None else min(shape[1], int(face_stop))

    apply_neck = mode in ("neck", "both") and neck_stop > neck_start
    apply_face = mode in ("face", "both") and face_stop > face_start

    z_grid = _axis_grid(shape, 0)
    y_grid = _axis_grid(shape, 1)
    x_grid = _axis_grid(shape, 2)
    z_center = (shape[0] - 1) / 2.0
    y_center = (shape[1] - 1) / 2.0
    x_center = (shape[2] - 1) / 2.0

    neck_keep_mask = np.ones(shape, dtype=np.float32)
    face_keep_mask = np.ones(shape, dtype=np.float32)

    if apply_neck:
        neck_cut = float(rng.integers(neck_start + 1, neck_stop + 1))
        neck_tilt_y = float(rng.choice([-1.0, 1.0]) * rng.uniform(*tilt_voxels_range))
        neck_tilt_x = float(rng.choice([-1.0, 1.0]) * rng.uniform(*tilt_voxels_range))
        neck_cut_grid = (
            neck_cut
            + neck_tilt_y * ((y_grid - y_center) / max(y_center, 1.0))
            + neck_tilt_x * ((x_grid - x_center) / max(x_center, 1.0))
        )
        neck_cut_grid = np.clip(neck_cut_grid, neck_start, neck_stop)
        if soft:
            neck_keep_mask = _soft_or_binary_keep(
                z_grid - neck_cut_grid,
                transition_width,
                z_grid >= neck_cut_grid,
            )
        else:
            neck_keep_mask = (z_grid >= neck_cut_grid).astype(np.float32)

    if apply_face:
        face_cut = float(rng.integers(face_start, face_stop))
        face_tilt_z = float(rng.choice([-1.0, 1.0]) * rng.uniform(*tilt_voxels_range))
        face_tilt_x = float(rng.choice([-1.0, 1.0]) * rng.uniform(*tilt_voxels_range))
        face_cut_grid = (
            face_cut
            + face_tilt_z * ((z_grid - z_center) / max(z_center, 1.0))
            + face_tilt_x * ((x_grid - x_center) / max(x_center, 1.0))
        )
        face_cut_grid = np.clip(face_cut_grid, face_start, face_stop)
        if soft:
            face_keep_mask = _soft_or_binary_keep(
                face_cut_grid - y_grid,
                transition_width,
                y_grid < face_cut_grid,
            )
        else:
            face_keep_mask = (y_grid < face_cut_grid).astype(np.float32)

    neck_keep_mask[safe_brain] = 1.0
    face_keep_mask[safe_brain] = 1.0
    keep_mask = np.minimum(neck_keep_mask, face_keep_mask).astype(np.float32)
    keep_mask[safe_brain] = 1.0
    crop_region = (keep_mask < 0.5).astype(np.uint8)
    return keep_mask, crop_region

# csv files should contain hdf5 index for each subject and hdf5 path. So, we can know where the MRI image for a specific subject is in these hdf5 files. And this is for easier loading. 
# While for lazy loading, the writing logic will changes a lot and may takes more time because we couldn't load all the MRI imgs in these hdf5 files and they may belong to different types of datasets(train,val,test).
class  MRIDatasetsAug(Dataset):
    '''
    In this function, we load the training dataset with TrainFlag=1 and randomly sample
    failed cases for the training set. 

    To locate data in HDF5 files, we follow Raven’s lazy-loading approach. We first build an
    index from all HDF5 files in the target folder after scanning them. Then we use a binary
    search to retrieve the corresponding image and label entries efficiently.

    all_hdf5_Paths: List of paths to text files or direct .hdf5 files. Each text file should contain one HDF5 file path per line.
    '''
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
        image_size=(192,224,192),
        add_coord=False,
        add_noise=False,
        return_raw_image=False,
        return_source_image=False,
        train_param_mode="default",
        use_ventricle_volume_for_scale=False,
        use_sitk_hdf5_resample=True,
        sitk_resample_template_path=None,
        use_neck_face_crop_augmentation=False,
        neck_face_brain_mask_path=None,
        neck_face_mask_dilation_radius=(6, 6, 6),
        neck_face_crop_mode_probabilities=(0.3, 0.2, 0.3, 0.2),
        neck_z_crop_range=(0, 49),
        face_y_crop_range=(167, 214),
        neck_face_crop_soft=True,
        neck_face_crop_transition_width=(0.0, 5.0),
        neck_face_crop_tilt_voxels_range=(0.0, 10.0),
    ):
        super().__init__()
        self.transform = transform
        self.TrainFlag = TrainFlag
        self.LazyLoading = LazyLoading
        self.center_based_transform = center_based_transform    
        self.add_noise = add_noise
        self.istemplate = template
        self.image_size = image_size
        self.hdf5s_path = hdf5s_path
        self.add_coord = add_coord
        self.return_raw_image = return_raw_image
        self.return_source_image = return_source_image
        self.train_param_mode = train_param_mode
        self.use_ventricle_volume_for_scale = use_ventricle_volume_for_scale
        self.use_sitk_hdf5_resample = use_sitk_hdf5_resample
        self.use_neck_face_crop_augmentation = use_neck_face_crop_augmentation
        self.neck_face_brain_mask_path = neck_face_brain_mask_path
        self.neck_face_mask_dilation_radius = tuple(neck_face_mask_dilation_radius)
        self.neck_face_crop_mode_probabilities = tuple(neck_face_crop_mode_probabilities)
        self.neck_z_crop_range = tuple(neck_z_crop_range)
        self.face_y_crop_range = tuple(face_y_crop_range)
        self.neck_face_crop_soft = neck_face_crop_soft
        self.neck_face_crop_transition_width = neck_face_crop_transition_width
        self.neck_face_crop_tilt_voxels_range = tuple(neck_face_crop_tilt_voxels_range)
        self.sitk_resample_template_path = sitk_resample_template_path or template_path
        if self.use_sitk_hdf5_resample:
            if not str(self.sitk_resample_template_path).strip():
                raise ValueError("sitk_resample_template_path or template_path is required when use_sitk_hdf5_resample=True.")
            self.sitk_template_image = sitk.ReadImage(self.sitk_resample_template_path)
        self.neck_face_dilated_mask_image = None
        if self.use_neck_face_crop_augmentation:
            if not self.TrainFlag:
                self.use_neck_face_crop_augmentation = False
            else:
                if not self.use_sitk_hdf5_resample:
                    raise ValueError("Neck/face crop augmentation requires use_sitk_hdf5_resample=True.")
                if not self.neck_face_brain_mask_path:
                    raise ValueError("neck_face_brain_mask_path is required when use_neck_face_crop_augmentation=True.")
                self.neck_face_dilated_mask_image = _load_and_dilate_mask(
                    self.neck_face_brain_mask_path,
                    self.neck_face_mask_dilation_radius,
                )
        #load csv files and collect all the info: (hdf5_path, hdf_index, age, sex, lin_motion_rate for linear registration qc, sbj_ID, params) 
        #for validation, the params should be added to csv, for the train part, params will be set during training process
        
        info_df = pd.read_csv(csv_path)
        self.total_samples = len(info_df)

        #check whether this csv files already has transform_params as a column
        self.has_transform_col = 'transform_params' in info_df.columns
        self.has_total_ventricle_col = 'total_ventricle' in info_df.columns
        # transform template in the initial part
        if template:
            temp_img = load_image_func(template_path)
            temp_img = extract_normalized_patch(temp_img,self.image_size)
            if self.transform:
                temp_img=self.transform(temp_img)
            self.template = torch.from_numpy(temp_img).float().unsqueeze(0)

        if self.add_coord:
            self.coords= self._coord_map_3d(self.image_size)
            

        # load datasets for lazy loading or traditional loading
        if self.LazyLoading:
            # just load the csv info
            self.info = info_df
        else:
            # load all imgs and info to the memory
            self.in_memory_data=[]
            unique_h5_files = info_df['hdf5_file'].unique()
            print("Non-lazy loading mode: reading all data listed in the csv file into RAM")
            for h5_path in unique_h5_files:
                file_subset = info_df[info_df['hdf5_file'] == h5_path]
                Fhf_path = os.path.join(self.hdf5s_path, h5_path)
                with h5py.File(Fhf_path, 'r') as hf:
                    if self.use_sitk_hdf5_resample:
                        required_keys = (
                            'imgs',
                            'source_spacing',
                            'source_origin',
                            'source_direction',
                        )
                        missing_keys = [key for key in required_keys if key not in hf]
                        if missing_keys:
                            raise KeyError(
                                f"use_sitk_hdf5_resample=True requires HDF5 datasets {required_keys}. "
                                f"Missing keys in {Fhf_path}: {missing_keys}"
                            )
                        source_image_key = 'source_imgs' if 'source_imgs' in hf else 'imgs'
                    for _, row in tqdm(file_subset.iterrows(), total=len(file_subset), desc=f"Reading {Fhf_path.split('/')[-1]}"):
                        h5_idx = row['hdf5_index']
                        if isinstance(h5_idx, str):
                            h5_idx = int(float(h5_idx.strip()))
                        else:
                            h5_idx = int(h5_idx)
                        if self.use_sitk_hdf5_resample:
                            image = _read_hdf5_image(hf, source_image_key, h5_idx)
                            source_spacing = hf['source_spacing'][h5_idx]
                            source_origin = hf['source_origin'][h5_idx]
                            source_direction = hf['source_direction'][h5_idx]
                        else:
                            image = _read_hdf5_image(hf, 'imgs', h5_idx)
                            image = extract_normalized_patch(image,self.image_size)
                        if self.has_transform_col:
                            raw_val = row['transform_params']
                            if pd.notna(raw_val):
                                if isinstance(raw_val, str):
                                    parsed = ast.literal_eval(raw_val)
                                else:
                                    parsed = raw_val
                                    
                                affine_array = np.array(parsed, dtype=np.float32)
                                if affine_array.shape[0] != 9:
                                    print('The shape of params array is not correct!!')
                                    #affine_array = np.zeros(9, dtype=np.float32)
                        else:
                            affine_array = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=np.float32)

                        self.in_memory_data.append({
                            'image': image, 
                            'subject_id': row['sbj_ID'],
                            'subject_visit': row['sbj_visit'],
                            'lin_rate': row['lin_motion_rate'],
                            'Affine_param': affine_array,
                            'dataset_origin':row['dataset_origin'],
                            'total_ventricle': row['total_ventricle'] if self.has_total_ventricle_col else None,
                            'source_spacing': source_spacing if self.use_sitk_hdf5_resample else None,
                            'source_origin': source_origin if self.use_sitk_hdf5_resample else None,
                            'source_direction': source_direction if self.use_sitk_hdf5_resample else None,
                        })
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self,index):
        if index<0 or index>=self.total_samples:
            raise IndexError(f"Index {index} out of range for dataset of size {self.total_samples}")
        
        #start_time = time.time()
        if self.LazyLoading:
            row = self.info.iloc[index]
            hdf5FilePath = os.path.join(self.hdf5s_path, row['hdf5_file'])
            hdf5index = row['hdf5_index']
            if isinstance(hdf5index, str):
                hdf5index = int(float(hdf5index.strip()))
            else:
                hdf5index = int(hdf5index)
            try:
                
                with h5py.File(hdf5FilePath,"r") as hf:
                    if self.use_sitk_hdf5_resample:
                        required_keys = (
                            'imgs',
                            'source_spacing',
                            'source_origin',
                            'source_direction',
                        )
                        missing_keys = [key for key in required_keys if key not in hf]
                        if missing_keys:
                            raise KeyError(
                                f"use_sitk_hdf5_resample=True requires HDF5 datasets {required_keys}. "
                                f"Missing keys in {hdf5FilePath}: {missing_keys}"
                            )
                        source_image_key = 'source_imgs' if 'source_imgs' in hf else 'imgs'
                        source_image_array = _read_hdf5_image(hf, source_image_key, hdf5index)
                        source_spacing = hf['source_spacing'][hdf5index]
                        source_origin = hf['source_origin'][hdf5index]
                        source_direction = hf['source_direction'][hdf5index]
                        image = source_image_array
                    else:
                        image = _read_hdf5_image(hf, 'imgs', hdf5index)
                        image = extract_normalized_patch(image,self.image_size)
                    if self.return_raw_image and self.use_sitk_hdf5_resample:
                        image_raw = extract_normalized_patch(image, self.image_size).copy()
                    else:
                        image_raw = image.copy() if self.return_raw_image else None
                    #image_raw = image
                    subject_id = row['sbj_ID']
                    subject_visit = row['sbj_visit']
                    lin_rate = row['lin_motion_rate']
                    dataset_origin = row['dataset_origin']
                    total_ventricle = row['total_ventricle'] if self.has_total_ventricle_col else None
                    if self.has_transform_col:
                        raw_val = row['transform_params']
                        if pd.notna(raw_val):
                            if isinstance(raw_val, str):
                                parsed = ast.literal_eval(raw_val)
                            else:
                                parsed = raw_val
                                    
                            affine_array = np.array(parsed, dtype=np.float32)
                            if affine_array.shape[0] != 9:
                                print('The shape of params array is not correct!!')
                                #affine_array = np.zeros(9, dtype=np.float32)
                    else:
                        affine_array = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=np.float32)
                    Affine_param = affine_array
            except KeyError as e:
                raise KeyError(f"Key {e} not found in file {hdf5FilePath}")
            except Exception as e:
                raise RuntimeError(f'Error accessing data in {hdf5FilePath}: {e}')

        else:
            sample = self.in_memory_data[index]
            image = sample['image']
            if self.return_raw_image and self.use_sitk_hdf5_resample:
                image_raw = extract_normalized_patch(image, self.image_size).copy()
            else:
                image_raw = image.copy() if self.return_raw_image else None
            #image_raw = image
            subject_id = sample['subject_id']
            subject_visit = sample['subject_visit']
            lin_rate = sample['lin_rate']
            Affine_param = sample['Affine_param']
            dataset_origin = sample['dataset_origin']
            total_ventricle = sample.get('total_ventricle')
            source_spacing = sample.get('source_spacing', None)
            source_origin = sample.get('source_origin', None)
            source_direction = sample.get('source_direction', None)

        source_image_for_return = None
        if self.return_source_image and self.use_sitk_hdf5_resample:
            source_image_for_return = image.copy()

        
        # if training model, we change the label and image randomly
        if self.TrainFlag:
            if (
                self.use_ventricle_volume_for_scale
                and self.has_total_ventricle_col
                and pd.notna(total_ventricle)
            ):
                Affine_param=np.array(GenParamsCorrelated(total_ventricle), dtype=np.float32)
            elif self.train_param_mode == "small":
                Affine_param=np.array(GenParams_small(), dtype=np.float32)
            else:
                Affine_param=np.array(GenParams(), dtype=np.float32)

        if self.TrainFlag and self.use_neck_face_crop_augmentation:
            image = self._apply_neck_face_crop_augmentation(
                image,
                source_spacing,
                source_origin,
                source_direction,
            )
        
        if self.use_sitk_hdf5_resample:
            _,image=GenImageWithParams_sitk_hdf5(
                image,
                source_spacing,
                source_origin,
                source_direction,
                self.sitk_template_image,
                Affine_param,
            )
        else:
            print("Warning: use_sitk_hdf5_resample is set to False. The affine transformation will be applied using scipy's affine_transform, which may not be as accurate as SimpleITK's resampling. Consider setting use_sitk_hdf5_resample=True for better results.")
        
        # transform to Gaussian distribution or MinMax normalization
        if self.transform:
           image = self.transform(image)

        image = extract_normalized_patch(image, self.image_size)

        if self.add_noise and self.TrainFlag:
            #image,_ = add_scaled_gaussian_noise(image)
            image = add_noise(image,noise_type='R')

        # transform image to torch.tensor
        image = torch.from_numpy(image).float().unsqueeze(0)
        
        #cancatenated template for channel section
        if self.istemplate:
            image = torch.cat((image, self.template), dim=0)
        
        if self.add_coord:
            image = torch.cat((image,self.coords),dim=0)
        #endtime=time.time()
        #print(endtime-start_time)
        sample = {
           #'image_raw': image_raw, 
           'image': image,
           'subject_id': subject_id,
           'subject_visit': subject_visit,
           'lin_rate':lin_rate,
           'Affine_param':Affine_param,
           'dataset_origin':dataset_origin,
        }
        if self.return_raw_image:
            sample['image_raw'] = image_raw
        if self.return_source_image:
            if not self.use_sitk_hdf5_resample:
                raise ValueError("return_source_image=True requires use_sitk_hdf5_resample=True.")
            sample['source_image'] = source_image_for_return
            sample['source_spacing'] = np.asarray(source_spacing, dtype=np.float32)
            sample['source_origin'] = np.asarray(source_origin, dtype=np.float32)
            sample['source_direction'] = np.asarray(source_direction, dtype=np.float32)
        return sample

    def _apply_neck_face_crop_augmentation(self, image, source_spacing, source_origin, source_direction):
        source_image = _sitk_image_from_hdf5_array(image, source_spacing, source_origin, source_direction)
        dilated_mask = _resample_mask_to_reference(self.neck_face_dilated_mask_image, source_image)
        safe_brain = sitk.GetArrayFromImage(dilated_mask).astype(bool)
        mode = _sample_neck_face_crop_mode(self.neck_face_crop_mode_probabilities)
        if mode == "none":
            return image

        keep_mask, _ = create_slanted_neck_face_keep_mask(
            image.shape,
            safe_brain,
            mode=mode,
            neck_z_range=self.neck_z_crop_range,
            face_y_range=self.face_y_crop_range,
            soft=self.neck_face_crop_soft,
            transition_width=self.neck_face_crop_transition_width,
            tilt_voxels_range=self.neck_face_crop_tilt_voxels_range,
        )
        return (image.astype(np.float32, copy=False) * keep_mask).astype(np.float32, copy=False)

    def _coord_map_3d(self, image_size, device=None, dtype=torch.float32, start=-1.0, end=1.0):
        D,H,W = image_size
        z = torch.linspace(start, end, steps=D, device=device, dtype=dtype)  # (D,)
        y = torch.linspace(start, end, steps=H, device=device, dtype=dtype)  # (H,)
        x = torch.linspace(start, end, steps=W, device=device, dtype=dtype)  # (W,)

        zz = z[:, None, None].expand(D, H, W)  # (D,H,W)
        yy = y[None, :, None].expand(D, H, W)  # (D,H,W)
        xx = x[None, None, :].expand(D, H, W)  # (D,H,W)

        coords = torch.stack([xx, yy, zz], dim=0)  # (3,D,H,W) in (x,y,z) order
        return coords

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

def GenParams_small():
   params=np.zeros(9,dtype=float)
   # keep all changes small
   for i in range(0,9):
        if i in(3,4,5):
            params[i]=random.uniform(0.95,1.05)
        if i in(0,1,2):
            params[i]=random.uniform(-2,2)
        if i in(6,7,8):
            params[i]=random.uniform(-3,3)
   return params

def GenParamsCorrelated(
    ventricle_volume,
    volume_mean=42034.0,
    volume_std=22765.0,
    target_corrs=(-0.8, -0.6, -0.6),
):
   params=np.zeros(9,dtype=float)
   v_latent = (float(ventricle_volume) - volume_mean) / volume_std

   for i, target_rho in zip((3, 4, 5), target_corrs):
        noise = random.gauss(0, 1)
        latent_scale = target_rho * v_latent + math.sqrt(1 - target_rho**2) * noise
        unit_uniform = 0.5 * (1 + math.erf(latent_scale / math.sqrt(2)))
        params[i] = 0.85 + unit_uniform * (1.15 - 0.85)

   params[0:3]=np.random.uniform(-5,5,3)
   params[6:9]=np.random.uniform(-7,7,3)
   return params

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
    #x = x.astype(np.single)
    
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
        return x_noise

    # Final clip to maintain valid range and return
    return np.clip(x_noise, 0.0, max_val).astype(np.single)
    
def extract_normalized_patch(data, patch_size=(192,224,192)):
    """
    Return a centered patch of shape `patch_size`.
    If the source image is smaller than `patch_size`, zero-pad it symmetrically.
    """

    X, Y, Z = data.shape
    px, py, pz = patch_size
    sx = max(0, (X - px) // 2)
    sy = max(0, (Y - py) // 2)
    sz = max(0, (Z - pz) // 2)
    ex = sx + min(X, px)
    ey = sy + min(Y, py)
    ez = sz + min(Z, pz)

    patch = data[sx:ex, sy:ey, sz:ez]

    # Keep the brain centered when padding smaller images to the model input size.
    if patch.shape != patch_size:
        dx = px - patch.shape[0]
        dy = py - patch.shape[1]
        dz = pz - patch.shape[2]
        patch = np.pad(
            patch,
            ((dx // 2, dx - dx // 2),
             (dy // 2, dy - dy // 2),
             (dz // 2, dz - dz // 2)),
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


def _affine_matrix_from_params(params):
   rotation_angle_x,rotation_angle_y,rotation_angle_z,scale_x,scale_y,scale_z,translation_x,translation_y,translation_z=params
   angle_rad_x = np.deg2rad(rotation_angle_x)
   angle_rad_y = np.deg2rad(rotation_angle_y)
   angle_rad_z = np.deg2rad(rotation_angle_z)

   rot_matrix_x = np.array([
       [1, 0, 0],
       [0, np.cos(angle_rad_x), -np.sin(angle_rad_x)],
       [0, np.sin(angle_rad_x), np.cos(angle_rad_x)]
   ])
   rot_matrix_y = np.array([
       [np.cos(angle_rad_y), 0, np.sin(angle_rad_y)],
       [0, 1, 0],
       [-np.sin(angle_rad_y), 0, np.cos(angle_rad_y)]
   ])
   rot_matrix_z = np.array([
       [np.cos(angle_rad_z), -np.sin(angle_rad_z), 0],
       [np.sin(angle_rad_z), np.cos(angle_rad_z), 0],
       [0, 0, 1]
   ])

   rotation_matrix = rot_matrix_x @ rot_matrix_y @ rot_matrix_z
   scale_matrix = np.array([
       [scale_x, 0, 0],
       [0, scale_y, 0],
       [0, 0, scale_z]
   ])

   new_affine = np.eye(4)
   new_affine[:3, :3] = rotation_matrix @ scale_matrix
   new_affine[:3, 3] = np.array([translation_x, translation_y, translation_z])
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


def GenImageWithParams_sitk_hdf5(image_array, spacing, origin, direction, template_image, params):
   new_affine = _affine_matrix_from_params(params)
   source_image = _sitk_image_from_hdf5_array(image_array, spacing, origin, direction)

   resampler = sitk.ResampleImageFilter()
   resampler.SetReferenceImage(template_image)
   resampler.SetTransform(_make_sitk_resample_transform(new_affine))
   resampler.SetInterpolator(sitk.sitkLinear)
   resampler.SetDefaultPixelValue(0.0)
   resampler.SetOutputPixelType(sitk.sitkFloat32)

   transformed_image = sitk.GetArrayFromImage(resampler.Execute(source_image)).astype(np.float32)
   return new_affine, transformed_image


def GenImageWithParams(image,params):
   #params: param vector with 9 params
   rotation_angle_x,rotation_angle_y,rotation_angle_z,scale_x,scale_y,scale_z,translation_x,translation_y,translation_z=params
   angle_rad_x = np.deg2rad(rotation_angle_x)
   angle_rad_y = np.deg2rad(rotation_angle_y)
   angle_rad_z = np.deg2rad(rotation_angle_z)

   rot_matrix_x = np.array([
       [1, 0, 0],
       [0, np.cos(angle_rad_x), -np.sin(angle_rad_x)],
       [0, np.sin(angle_rad_x), np.cos(angle_rad_x)]
   ])
   rot_matrix_y = np.array([
       [np.cos(angle_rad_y), 0, np.sin(angle_rad_y)],
       [0, 1, 0],
       [-np.sin(angle_rad_y), 0, np.cos(angle_rad_y)]
   ])
   rot_matrix_z = np.array([
       [np.cos(angle_rad_z), -np.sin(angle_rad_z), 0],
       [np.sin(angle_rad_z), np.cos(angle_rad_z), 0],
       [0, 0, 1]
   ])

   rotation_matrix = rot_matrix_x @ rot_matrix_y @ rot_matrix_z

   scale_matrix = np.array([
       [scale_x, 0, 0],
       [0, scale_y, 0],
       [0, 0, scale_z]
   ])

   transformation_matrix =  rotation_matrix @ scale_matrix

   translation_vector = np.array([
       translation_x,
       translation_y,
       translation_z
   ])
   # Create affine matrix
   new_affine = np.eye(4)
   new_affine[:3, :3] = transformation_matrix
   new_affine[:3, 3] = translation_vector

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
   #params: param vector with 9 params
   rotation_angle_x,rotation_angle_y,rotation_angle_z,scale_x,scale_y,scale_z,translation_x,translation_y,translation_z=params
   angle_rad_x = np.deg2rad(rotation_angle_x)
   angle_rad_y = np.deg2rad(rotation_angle_y)
   angle_rad_z = np.deg2rad(rotation_angle_z)

   rot_matrix_x = np.array([
       [1, 0, 0],
       [0, np.cos(angle_rad_x), -np.sin(angle_rad_x)],
       [0, np.sin(angle_rad_x), np.cos(angle_rad_x)]
   ])
   rot_matrix_y = np.array([
       [np.cos(angle_rad_y), 0, np.sin(angle_rad_y)],
       [0, 1, 0],
       [-np.sin(angle_rad_y), 0, np.cos(angle_rad_y)]
   ])
   rot_matrix_z = np.array([
       [np.cos(angle_rad_z), -np.sin(angle_rad_z), 0],
       [np.sin(angle_rad_z), np.cos(angle_rad_z), 0],
       [0, 0, 1]
   ])

   rotation_matrix = rot_matrix_x @ rot_matrix_y @ rot_matrix_z

   scale_matrix = np.array([
       [scale_x, 0, 0],
       [0, scale_y, 0],
       [0, 0, scale_z]
   ])

   transformation_matrix =  rotation_matrix @ scale_matrix

   translation_vector = np.array([
       translation_x,
       translation_y,
       translation_z
   ])
   # Create affine matrix
   new_affine = np.eye(4)
   new_affine[:3, :3] = transformation_matrix
   new_affine[:3, 3] = translation_vector

   # the only line we're changing
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


def GenImageWithParams_center_based(image,params):
   W,H,D=image.shape
   #params: param vector with 9 params
   rotation_angle_x,rotation_angle_y,rotation_angle_z,scale_x,scale_y,scale_z,translation_x,translation_y,translation_z=params
   angle_rad_x = np.deg2rad(rotation_angle_x)
   angle_rad_y = np.deg2rad(rotation_angle_y)
   angle_rad_z = np.deg2rad(rotation_angle_z)

   rot_matrix_x = np.array([
       [1, 0, 0],
       [0, np.cos(angle_rad_x), -np.sin(angle_rad_x)],
       [0, np.sin(angle_rad_x), np.cos(angle_rad_x)]
   ])
   rot_matrix_y = np.array([
       [np.cos(angle_rad_y), 0, np.sin(angle_rad_y)],
       [0, 1, 0],
       [-np.sin(angle_rad_y), 0, np.cos(angle_rad_y)]
   ])
   rot_matrix_z = np.array([
       [np.cos(angle_rad_z), -np.sin(angle_rad_z), 0],
       [np.sin(angle_rad_z), np.cos(angle_rad_z), 0],
       [0, 0, 1]
   ])

   rotation_matrix = rot_matrix_x @ rot_matrix_y @ rot_matrix_z

   scale_matrix = np.array([
       [scale_x, 0, 0],
       [0, scale_y, 0],
       [0, 0, scale_z]
   ])

   transformation_matrix =  rotation_matrix @ scale_matrix

   # adjust the affine to be center-based
   C=np.array([(W-1)/2,(H-1)/2,(D-1)/2],dtype=float)
   T=np.array([translation_x,translation_y,translation_z],dtype=float)
   adjusted_translation = C - transformation_matrix @ C + T
   # Create affine matrix
   new_affine = np.eye(4)
   new_affine[:3, :3] = transformation_matrix
   new_affine[:3, 3] = adjusted_translation
   
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
