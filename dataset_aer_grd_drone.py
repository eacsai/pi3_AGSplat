import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal
import math

import torch
import torchvision.transforms as tf
import torchvision.transforms.functional as TF
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset, Dataset
from torchvision import transforms
import torch.nn.functional as F

import numpy as np
import os

# ----------- 全局配置 -----------
ROOT_DIR   = '/data/zhongyao/aer-grd-map'   
GrdOriImg_H = 1080
GrdOriImg_W = 1920
GrdImg_H, GrdImg_W = 512, 1024             # 地面/航拍图 resize 后尺寸，原始数据：1080*1920
SatMap_SIDE = 1024                         # 卫星图输出边长，原始数据：2700*2700
PIXEL_LIMIT = 255000                       # 255000 pixels limit for resizing images
# --------------------------------
satmap_dir = 'satmap'
Default_lat = 49.015
Satmap_zoom = 18
SatMap_original_sidelength = 512 # 0.2 m per pixel
SatMap_process_sidelength = 512 # 0.2 m per pixel

colmap_to_opencv = np.array([
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
], dtype=np.float32)

Rot = False

def get_meter_per_pixel(lat=Default_lat, zoom=Satmap_zoom, scale=SatMap_process_sidelength/SatMap_original_sidelength):
    meter_per_pixel = 156543.03392 * np.cos(lat * np.pi/180.) / (2**zoom)	
    meter_per_pixel /= 2 # because use scale 2 to get satmap 
    meter_per_pixel /= scale
    return meter_per_pixel


def load_test_triplet(index=0):
    """
    从测试集中加载一组对应的地面图像、无人机图像和卫星图像

    Args:
        index (int): 要加载的测试样本索引，默认为0

    Returns:
        dict: 包含三张图像tensor的字典，键为'ground', 'drone', 'satellite'
    """
    # 读取测试文件列表
    with open('/data/zhongyao/aer-grd-map/test_files_1017.txt', 'r') as f:
        lines = f.readlines()

    if index >= len(lines):
        raise IndexError(f"Index {index} out of range. Test set has {len(lines)} samples.")

    # 解析文件路径
    line = lines[index].strip()
    test_line = line.split(' ')
    grd_path, drone_path, sat_path = test_line[0], test_line[1], test_line[2]

    # 定义图像变换（与DatasetAerGrdDrone类中的变换保持一致）
    padding_top = (512 - 256) // 2  # (final_h - GrdImg_H) // 2
    padding_left = 0  # (final_w - GrdImg_W) // 2

    W_orig, H_orig = 512, 512
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target: k -= 1
        else: m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    grd_transform = transforms.Compose([
        transforms.Resize(size=[256, 512]),  # GrdImg_H, GrdImg_W
        transforms.Pad(padding=(padding_left, padding_top, padding_left, padding_top), fill=0),
        transforms.Resize(size=[TARGET_W, TARGET_H]),
        transforms.ToTensor(),
    ])

    sat_transform = transforms.Compose([
        transforms.Resize(size=[TARGET_W, TARGET_H]),  # SatMap_process_sidelength
        transforms.ToTensor(),
    ])

    drone_transform = transforms.Compose([
        transforms.Resize(size=[256, 512]),  # 与地面图像相同
        transforms.Pad(padding=(padding_left, padding_top, padding_left, padding_top), fill=0),
        transforms.Resize(size=[TARGET_W, TARGET_H]),
        transforms.ToTensor(),
    ])

    # 加载图像
    try:
        grd_img = Image.open(grd_path).convert('RGB')
        drone_img = Image.open(drone_path).convert('RGB')
        sat_img = Image.open(sat_path).convert('RGB')

        # 对卫星图进行4倍下采样（与原数据集一致）
        new_size = (750, 750)
        sat_img = sat_img.resize(new_size, Image.BILINEAR)

        # 应用变换
        grd_tensor = grd_transform(grd_img)
        drone_tensor = drone_transform(drone_img)
        sat_tensor = sat_transform(sat_img)

        return {
            'ground': grd_tensor,
            'drone': drone_tensor,
            'satellite': sat_tensor,
            'paths': {
                'ground': grd_path,
                'drone': drone_path,
                'satellite': sat_path
            }
        }
    except Exception as e:
        raise RuntimeError(f"Error loading images: {e}")


class TestTripletDataset(Dataset):
    """
    PyTorch Dataset for loading test image triplets (ground, drone, satellite)
    """

    def __init__(self, test_file_path='/data/zhongyao/aer-grd-map/test_files_1024.txt',
                 target_size=None, auto_resize=True, amount=1.0):
        """
        Args:
            test_file_path (str): Path to the test file list
            target_size (tuple): Target size (width, height) for images. If None, auto-calculate
            auto_resize (bool): Whether to automatically calculate optimal size
        """
        self.test_file_path = test_file_path
        self.auto_resize = auto_resize
        self.stage = self.test_file_path.split('/')[-1].split('_')[0]
        # Load test file list
        with open(test_file_path, 'r') as f:
            self.lines = f.readlines()

        # Remove empty lines and strip whitespace
        self.lines = [line.strip() for line in self.lines if line.strip()]

        # Select only a fraction of the data
        total_samples = len(self.lines)
        num_selected = max(1, int(total_samples * amount))  # Ensure at least 1 sample
        self.lines = self.lines[:num_selected]  # Take first fraction

        print(f"Loaded {len(self.lines)} test samples from {test_file_path}")

        # Parse file paths for all samples
        self.file_paths = []
        for line in self.lines:
            test_line = line.split(' ')
            if len(test_line) >= 5:
                grd_path, drone_path, sat_path, grd_mask, drone_mask = test_line[0], test_line[1], test_line[2], test_line[3], test_line[4]
                if self.stage == 'test':
                    self.file_paths.append({
                        'ground': grd_path,
                        'drone': drone_path,
                        'satellite': sat_path,
                        'grd_mask': grd_mask,
                        'drone_mask': drone_mask,
                        'gt_shift_x': float(test_line[5]),
                        'gt_shift_y': float(test_line[6]),
                    })
                else:
                    self.file_paths.append({
                        'ground': grd_path,
                        'drone': drone_path,
                        'satellite': sat_path,
                        'grd_mask': grd_mask,
                        'drone_mask': drone_mask,
                    })

        # Calculate target size if not provided
        if target_size is None and auto_resize:
            self.TARGET_W, self.TARGET_H = self._calculate_target_size()
        elif target_size is not None:
            self.TARGET_W, self.TARGET_H = target_size
        else:
            self.TARGET_W, self.TARGET_H = 512, 512  # default size

        print(f"Images will be resized to uniform size: ({self.TARGET_W}, {self.TARGET_H})")

        # Setup transforms
        self._setup_transforms()

    def _calculate_target_size(self):
        """Calculate optimal target size based on pixel limit"""
        W_orig, H_orig = 512, 512
        scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
        W_target, H_target = W_orig * scale, H_orig * scale
        k, m = round(W_target / 14), round(H_target / 14)
        while (k * 14) * (m * 14) > PIXEL_LIMIT:
            if k / m > W_target / H_target: k -= 1
            else: m -= 1
        return max(1, k) * 14, max(1, m) * 14

    def _setup_transforms(self):
        """Setup image transforms"""
        padding_top = (512 - 256) // 2  # (final_h - GrdImg_H) // 2
        padding_left = 0  # (final_w - GrdImg_W) // 2

        self.grd_transform = transforms.Compose([
            transforms.Resize(size=[256, 512]),  # GrdImg_H, GrdImg_W
            transforms.Pad(padding=(padding_left, padding_top, padding_left, padding_top), fill=0),
            transforms.Resize(size=[self.TARGET_H, self.TARGET_W]),
            transforms.ToTensor(),
        ])

        self.sat_transform_ref = transforms.Compose([
            transforms.Resize(size=[512, 512]),  # SatMap_process_sidelength
            transforms.ToTensor(),
        ])

        self.sat_transform_pi3 = transforms.Compose([
            transforms.Resize(size=[self.TARGET_H, self.TARGET_W]),  # SatMap_process_sidelength
            transforms.ToTensor(),
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize(size=[256, 512]),
            transforms.Pad(padding=(padding_left, padding_top, padding_left, padding_top), fill=1),
            transforms.Resize(size=[self.TARGET_H, self.TARGET_W]),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Load a single triplet of images

        Returns:
            dict: Contains 'ground', 'drone', 'satellite' tensors and file paths
        """
        if idx >= len(self.file_paths):
            raise IndexError(f"Index {idx} out of range. Dataset has {len(self.file_paths)} samples.")

        paths = self.file_paths[idx]


        # Load images
        grd_img = Image.open(paths['ground']).convert('RGB')
        drone_img = Image.open(paths['drone']).convert('RGB')
        sat_img = Image.open(paths['satellite']).convert('RGB')
        grd_mask = Image.open(paths['grd_mask']).convert('L')
        drone_mask = Image.open(paths['drone_mask']).convert('L')

        # Load camera parameters
        npz_path = paths['ground'].replace('.jpeg.jpg', '.jpeg.npz')
        npz = np.load(npz_path)
        cam2world = npz['cam2world']
        cam2world = colmap_to_opencv @ cam2world
        cam2world = torch.from_numpy(cam2world.astype(np.float32))
        ground_cam_trans = cam2world[:3, 3]
        ground_cam_rot = cam2world[:3, :3]
        # 计算地面相机的朝向角度
        grd_angle_radians = torch.atan2(ground_cam_rot[0, 2], ground_cam_rot[2, 2])
        grd_angle_degrees = torch.rad2deg(grd_angle_radians)

        # 对卫星图进行下采样
        meter_per_pixel = 500 / min(sat_img.size)
        # sat_img = sat_img.resize(new_size, Image.BILINEAR)
        if Rot:
            sat_aligh_cam = sat_img.rotate(grd_angle_degrees)
            gt_shift_x = 0.0
            gt_shift_y = 0.0
            grd_angle_degrees = 0.0 # after rotation, the ground angle is zero
        else:
            sat_aligh_cam = sat_img
        
        if self.stage == 'test' and Rot is False:
            gt_shift_x, gt_shift_y = paths['gt_shift_x'], paths['gt_shift_y']
            dx_p = gt_shift_x * 20.0 / meter_per_pixel
            dy_p = gt_shift_y * 20.0 / meter_per_pixel
            sat_aligh_cam = sat_aligh_cam.transform(
                sat_aligh_cam.size, Image.AFFINE,
                (1, 0, dx_p, 0, 1, -dy_p), resample=Image.BILINEAR
            )

        sat_aligh_cam_ref = TF.center_crop(sat_aligh_cam, (210 / meter_per_pixel, 210 / meter_per_pixel))
        sat_aligh_cam_pi3 = TF.center_crop(sat_aligh_cam, (140 / meter_per_pixel, 140 / meter_per_pixel))
        # Apply transforms
        grd_tensor = self.grd_transform(grd_img)
        drone_tensor = self.grd_transform(drone_img)
        sat_tensor_ref = self.sat_transform_ref(sat_aligh_cam_ref)
        sat_tensor_pi3 = self.sat_transform_pi3(sat_aligh_cam_pi3)
        grd_mask = ~self.mask_transform(grd_mask).bool()
        grd_tensor = grd_tensor * grd_mask.float()
        grd_mask = rearrange(grd_mask, '1 h w -> h w 1')
        drone_mask = ~self.mask_transform(drone_mask).bool()
        drone_tensor = drone_tensor * drone_mask.float()
        drone_mask = rearrange(drone_mask, '1 h w -> h w 1')

        return {
            'ground': grd_tensor,
            'drone': drone_tensor,
            'sat_pi3':sat_tensor_pi3,
            'sat_ref':sat_tensor_ref,
            'grd_mask': grd_mask,
            'drone_mask': drone_mask,
            'grd_gt_angle_degrees': grd_angle_degrees,
            'grd_rot': ground_cam_rot,
            'grd_shift_z': -gt_shift_x * 20.0, # m
            'grd_shift_x': gt_shift_y * 20.0, # m
            'paths': paths,
            'index': idx
        }


    def get_sample_by_index(self, index):
        """
        Legacy method to maintain compatibility with load_test_triplet function
        """
        return self[index]

    def get_file_list(self):
        """Get list of all file paths"""
        return self.file_paths


def create_test_dataloader(test_file_path='/data/zhongyao/aer-grd-map/test_files_1027.txt',
                          batch_size=1, shuffle=False, num_workers=0,
                          target_size=None, auto_resize=True, amount=1.0, **kwargs):
    """
    Create a DataLoader for test triplet dataset

    Args:
        test_file_path (str): Path to test file list
        batch_size (int): Batch size for loading
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker processes for data loading
        target_size (tuple): Target size (width, height) for images
        auto_resize (bool): Whether to automatically calculate optimal size
        **kwargs: Additional arguments for DataLoader

    Returns:
        torch.utils.data.DataLoader: Configured DataLoader
    """
    dataset = TestTripletDataset(
        test_file_path=test_file_path,
        target_size=target_size,
        auto_resize=auto_resize,
        amount=amount
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )

    return dataloader


def load_test_triplet_batch(indices, test_file_path='/data/zhongyao/aer-grd-map/test_files_1024.txt'):
    """
    Load multiple test triplets by indices (batch version of load_test_triplet)

    Args:
        indices (list or int): List of indices or single index
        test_file_path (str): Path to test file list

    Returns:
        list or dict: List of dictionaries if multiple indices, single dict if one index
    """
    dataset = TestTripletDataset(test_file_path=test_file_path)

    if isinstance(indices, int):
        return dataset[indices]
    else:
        return [dataset[idx] for idx in indices]
