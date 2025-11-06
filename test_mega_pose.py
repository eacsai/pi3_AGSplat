"""
Ground Camera to Satellite Center Distance Analysis

This script processes satellite, ground, and drone image triplets to calculate
the distance from ground camera positions to satellite image centers.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 指定使用的GPU设备ID

import torch
import torch.nn.functional as F
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from typing import Tuple, Optional, Dict, List, Any
from functools import partial
from scipy.optimize import least_squares
from einops import einsum, rearrange, repeat
from image_pairs import make_pairs

# Local imports
from pi3.models.pi3 import Pi3
from dataset_aer_grd_drone import load_test_triplet, create_test_dataloader
from pi3.utils.geometry import intrinsics_from_focal_center, se3_inverse, homogenize_points, depth_edge

from cloud_opt import global_aligner, GlobalAlignerMode
# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

class Config:
    """Configuration constants for the analysis"""
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 2
    SATELLITE_WIDTH = 512
    PIXEL_TO_METER = 0.3  # Conversion factor from pixels to meters
    METER_PER_PIXEL = 0.2  # Default meter per pixel for projection
    DOWNSAMPLE_SIZE = (64, 64)
    CONF_THRESHOLD = 0.1
    MODEL_NAME = "yyfz233/Pi3"
    OUTPUT_FILE = "ground_camera_distances.txt"


# Transformation matrices and utilities
SAT_TO_OPENCV = torch.tensor([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
], dtype=torch.float32)

TO_PIL_IMAGE = transforms.ToPILImage()
POINT_COLORS = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'orange', 'magenta', 'lime', 'pink']


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_points_to_ply(points: torch.Tensor, colors: torch.Tensor, filename: str):
    """
    Save points and colors to a PLY file.

    Args:
        points: Tensor of shape (N, 3) containing 3D coordinates
        colors: Tensor of shape (N, 3) containing RGB colors in [0, 1] range
        filename: Output PLY filename
    """
    # Convert to numpy
    points_np = points.detach().cpu().numpy()
    colors_np = (colors.detach().cpu().numpy() * 255).astype(np.uint8)

    # Filter valid points (where z > 0 to avoid ground plane points)
    valid_mask = points_np[:, 2] > 0.1  # Filter out ground plane points
    valid_points = points_np[valid_mask]
    valid_colors = colors_np[valid_mask]

    if len(valid_points) == 0:
        print(f"Warning: No valid points to save for {filename}")
        return

    # Create PLY content
    ply_content = f"""ply
        format ascii 1.0
        element vertex {len(valid_points)}
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
    """

    # Add vertex data
    for point, color in zip(valid_points, valid_colors):
        ply_content += f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {color[0]} {color[1]} {color[2]}\n"

    # Write to file
    with open(filename, 'w') as f:
        f.write(ply_content)

    print(f"✅ Saved {len(valid_points)} points to {filename}")


def convert_pi3_to_vggt_format(pi3_output):
    """Convert Pi3 model output to VGGT-SLAM format."""
    # Extract outputs (remove batch dimension)
    points = pi3_output['points'][0]              # (S, H, W, 3)
    local_points = pi3_output['local_points'][0]  # (S, H, W, 3)
    conf = torch.sigmoid(pi3_output['conf'][0])   # (S, H, W, 1)
    camera_poses = pi3_output['camera_poses'][0]  # (S, 4, 4)

    # CRITICAL: Pi3 outputs cam2world, but VGGT expects world2cam (extrinsics)
    # Also, VGGT expects the first camera to be at identity (origin)

    # Step 1: Get the first camera's cam2world transformation
    T_c0_to_w = camera_poses[0]  # (4, 4) - first camera to world

    # Step 2: Compute world-to-first-camera transformation (to align world to cam0)
    # T_w_to_c0 = inv(T_c0_to_w)
    R_c0_to_w = T_c0_to_w[:3, :3]  # (3, 3)
    t_c0_to_w = T_c0_to_w[:3, 3:4]  # (3, 1)

    R_w_to_c0 = R_c0_to_w.T  # (3, 3)
    t_w_to_c0 = -R_w_to_c0 @ t_c0_to_w  # (3, 1)

    T_w_to_c0 = torch.eye(4, dtype=camera_poses.dtype, device=camera_poses.device)
    T_w_to_c0[:3, :3] = R_w_to_c0
    T_w_to_c0[:3, 3:4] = t_w_to_c0

    # Step 3: Transform all camera poses to the new world frame (where cam0 is at origin)
    # T_ci_to_new_world = T_w_to_c0 @ T_ci_to_old_world
    camera_poses_aligned = T_w_to_c0 @ camera_poses  # (S, 4, 4)

    # Step 4: Convert aligned cam2world to world2cam (extrinsics)
    R_c2w_aligned = camera_poses_aligned[:, :3, :3]  # (S, 3, 3)
    t_c2w_aligned = camera_poses_aligned[:, :3, 3:4]  # (S, 3, 1)

    R_w2c = R_c2w_aligned.transpose(-2, -1)  # (S, 3, 3)
    t_w2c = -R_w2c @ t_c2w_aligned  # (S, 3, 1)

    extrinsics = torch.cat([R_w2c, t_w2c], dim=-1)  # (S, 3, 4)

    # Step 5: Transform world points to the new coordinate frame
    # points_new = T_w_to_c0 @ points_old (in homogeneous coordinates)
    S, H, W, _ = points.shape
    points_flat = points.reshape(-1, 3)  # (S*H*W, 3)
    points_homo = torch.cat([points_flat, torch.ones(points_flat.shape[0], 1, dtype=points.dtype, device=points.device)], dim=-1)  # (S*H*W, 4)
    points_aligned_homo = (T_w_to_c0 @ points_homo.T).T  # (S*H*W, 4)
    points_aligned = points_aligned_homo[:, :3].reshape(S, H, W, 3)  # (S, H, W, 3)

    # Extract depth from local points (z-component)
    depth = local_points[..., 2:3]  # (S, H, W, 1)

    # Build predictions dict in VGGT-SLAM format
    # Add batch dimension to match VGGT output format (1, S, ...)
    predictions = {
        "world_points": points_aligned.unsqueeze(0),  # Use aligned points
        "world_points_conf": conf[..., 0].unsqueeze(0),
        "depth": depth.unsqueeze(0),
        "depth_conf": conf[..., 0].unsqueeze(0),
        "extrinsic": extrinsics.unsqueeze(0),
    }
    return predictions

def world_to_camera_broadcast(pts_all, extrinsics_c2w):
    """
    使用广播将世界坐标系点云批量转换到相机坐标系下。
    
    参数:
    pts_all: 世界坐标系点云, 形状 [b, v, N, gpv, 3]
    extrinsics_c2w: 相机到世界的外参, 形状 [b, 4, 4]
    
    返回:
    pts_cam: 相机坐标系点云, 形状 [b, v, N, gpv, 3]
    """
    
    # 1. 获取 w2c = (c2w)^-1
    b = pts_all.shape[0]
    extrinsics_w2c = torch.inverse(extrinsics_c2w)

    # 2. 提取 R (旋转) 和 t (平移)
    R = extrinsics_w2c[:, :3, :3] # [b, 3, 3]
    t = extrinsics_w2c[:, :3, 3:4] # [b, 3, 1]

    # 确保数据类型一致
    R = R.to(pts_all.dtype)
    t = t.to(pts_all.dtype)

    # 3. 准备 R, t 和 pts_all 以进行广播
    
    # 在 R 的 v, N, gpv 维度上添加 '1' 
    # [b, 3, 3] -> [b, 1, 1, 1, 3, 3]
    R_expanded = R.view(b, 1, 1, 1, 3, 3)
    
    # 在 t 的 v, N, gpv 维度上添加 '1' 
    # [b, 3, 1] -> [b, 1, 1, 1, 3, 1]
    t_expanded = t.view(b, 1, 1, 1, 3, 1)
    
    # 在 pts_all 的末尾添加一个 '1' 维度，使其成为 "列向量"
    # [b, v, N, gpv, 3] -> [b, v, N, gpv, 3, 1]
    pts_vec = pts_all.unsqueeze(-1)
    
    # 4. 应用变换: P_cam = R @ P_world + t
    # matmul 会自动广播 R_expanded 的 [1,1,1] 维度
    # [b, v, N, gpv, 3, 3] @ [b, v, N, gpv, 3, 1] -> [b, v, N, gpv, 3, 1]
    pts_rotated_vec = R_expanded @ pts_vec
    
    # 广播 t_expanded 并相加
    # [b, v, N, gpv, 3, 1] + [b, 1, 1, 1, 3, 1] -> [b, v, N, gpv, 3, 1]
    pts_cam_vec = pts_rotated_vec + t_expanded
    
    # 5. 移除最后的 '1' 维度
    # [b, v, N, gpv, 3, 1] -> [b, v, N, gpv, 3]
    pts_cam = pts_cam_vec.squeeze(-1)
    
    return pts_cam

def rotation_matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    将一个 [3, 3] 旋转矩阵转换为 [w, x, y, z] 四元数。
    (注意：这个简单实现不支持批处理)
    """
    if not matrix.shape == (3, 3):
        raise ValueError("输入必须是 [3, 3] 矩阵")

    m00, m01, m02 = matrix[0, 0], matrix[0, 1], matrix[0, 2]
    m10, m11, m12 = matrix[1, 0], matrix[1, 1], matrix[1, 2]
    m20, m21, m22 = matrix[2, 0], matrix[2, 1], matrix[2, 2]

    trace = m00 + m11 + m22
    
    # 使用数值稳定的方法
    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    else:
        if m00 > m11 and m00 > m22:
            s = 2.0 * torch.sqrt(1.0 + m00 - m11 - m22)
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * torch.sqrt(1.0 + m11 - m00 - m22)
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = 2.0 * torch.sqrt(1.0 + m22 - m00 - m11)
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s
            
    q = torch.stack([w, x, y, z])
    return q / torch.norm(q) # 归一化

def compare_rotations_manual(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """
    使用手动实现的函数比较两个 [3, 3] 旋转矩阵。
    """
    # 1. 转换为四元数
    q1 = rotation_matrix_to_quaternion(R1)
    q2 = rotation_matrix_to_quaternion(R2)
    
    # 2. 计算点积
    dot_product = torch.dot(q1, q2)
    
    # 3. 取绝对值
    dot_product_abs = torch.abs(dot_product)
    
    # 4. 裁剪
    dot_product_clamped = torch.clamp(dot_product_abs, -1.0, 1.0)
    
    # 5. 计算角度
    angle_rad = 2 * torch.acos(dot_product_clamped)
    
    return torch.rad2deg(angle_rad)



def forward_project(image_tensor: torch.Tensor, xyz_grd: torch.Tensor,
                   meter_per_pixel: float = Config.METER_PER_PIXEL,
                   sat_width: int = Config.SATELLITE_WIDTH) -> torch.Tensor:
    """
    Forward project 3D points to create a satellite view image.

    Args:
        image_tensor: Image tensor of shape (B, N_points, C)
        xyz_grd: 3D point coordinates
        meter_per_pixel: Conversion factor from meters to pixels
        sat_width: Satellite image width

    Returns:
        Projected image tensor
    """
    B, N_points, C = image_tensor.shape

    # Prepare data
    mask = image_tensor.any(dim=-1).float().unsqueeze(-1)
    xyz_grd = (xyz_grd * mask).reshape(B*N_points, -1)
    image_tensor = rearrange(image_tensor, 'b n c -> (b n) c')

    # Convert to pixel coordinates
    xyz_grd[:, 0] = xyz_grd[:, 0] / meter_per_pixel
    xyz_grd[:, 2] = xyz_grd[:, 2] / meter_per_pixel
    xyz_grd[:, 0] = xyz_grd[:, 0].long()
    xyz_grd[:, 2] = xyz_grd[:, 2].long()

    # Add batch indices
    batch_ix = torch.cat([torch.full([N_points, 1], ix, device=image_tensor.device) for ix in range(B)], dim=0)
    xyz_grd = torch.cat([xyz_grd, batch_ix], dim=-1)

    # Filter valid points within bounds
    kept = (xyz_grd[:,0] >= -(sat_width // 2)) & (xyz_grd[:,0] <= (sat_width // 2) - 1) & \
           (xyz_grd[:,2] >= -(sat_width // 2)) & (xyz_grd[:,2] <= (sat_width // 2) - 1)

    xyz_grd_kept = xyz_grd[kept]
    image_tensor_kept = image_tensor[kept]

    # Calculate height and transform coordinates
    max_height = xyz_grd_kept[:,1].max()
    xyz_grd_kept[:,0] = xyz_grd_kept[:,0] + sat_width // 2
    xyz_grd_kept[:,1] = max_height - xyz_grd_kept[:,1]
    xyz_grd_kept[:,2] = xyz_grd_kept[:,2] + sat_width // 2
    xyz_grd_kept = xyz_grd_kept[:,[2,0,1,3]]

    # Sort for proper rendering
    rank = torch.stack((xyz_grd_kept[:, 0] * sat_width * B + (xyz_grd_kept[:, 1] + 1) * B + xyz_grd_kept[:, 3],
                       xyz_grd_kept[:, 2]), dim=1)
    sorts_second = torch.argsort(rank[:, 1])
    xyz_grd_kept = xyz_grd_kept[sorts_second]
    image_tensor_kept = image_tensor_kept[sorts_second]
    sorted_rank = rank[sorts_second]
    sorts_first = torch.argsort(sorted_rank[:, 0], stable=True)
    xyz_grd_kept = xyz_grd_kept[sorts_first]
    image_tensor_kept = image_tensor_kept[sorts_first]
    sorted_rank = sorted_rank[sorts_first]
    kept = torch.ones_like(sorted_rank[:, 0])
    kept[:-1] = sorted_rank[:, 0][:-1] != sorted_rank[:, 0][1:]

    # Create final image
    res_xyz = xyz_grd_kept[kept.bool()]
    res_image = image_tensor_kept[kept.bool()]

    final = torch.zeros(B, sat_width, sat_width, C).to(torch.float32).to(Config.DEVICE)
    sat_height = torch.zeros(B, sat_width, sat_width, 1).to(torch.float32).to(Config.DEVICE)
    final[res_xyz[:,3].long(), res_xyz[:,1].long(), res_xyz[:,0].long(), :] = res_image

    res_xyz[:,2][res_xyz[:,2] < 1e-1] = 1e-1
    sat_height[res_xyz[:,3].long(), res_xyz[:,1].long(), res_xyz[:,0].long(), :] = res_xyz[:,2].unsqueeze(-1)
    sat_height = sat_height.permute(0,3,1,2)

    return final.permute(0,3,1,2)

def project_point_clouds(
    point_clouds: torch.Tensor,
    point_color: torch.Tensor,
    normalized_intrinsics: torch.Tensor, # <--- 参数名变化，表示接收归一化内参
    image_height: int = 256,
    image_width: int = 256,
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0) # 黑色背景
) -> torch.Tensor:
    """
    将一批点云投影到图像平面上（使用归一化的内参）。

    Args:
        point_clouds (torch.Tensor): 形状为 (B, N, 3) 的点云，坐标在相机坐标系下。
        point_color (torch.Tensor): 形状为 (B, N, 3) 的点云颜色，范围 [0, 1]。
        normalized_intrinsics (torch.Tensor): 形状为 (B, 1, 3, 3) 或 (B, 3, 3) 的归一化相机内参。
        image_height (int): 输出图像的高度。
        image_width (int): 输出图像的宽度。
        background_color (tuple[float, float, float]): 图像的背景色。

    Returns:
        torch.Tensor: 形状为 (B, 3, H, W) 的渲染图像张量。
    """
    # --- 1. 准备工作 ---
    B, N, _ = point_clouds.shape
    device = point_clouds.device

    if normalized_intrinsics.dim() == 4:
        K_norm = normalized_intrinsics[:,0,:,:]
    else:
        K_norm = normalized_intrinsics

    # --- 2. 几何变换：3D到2D投影 ---
    points_transposed = torch.transpose(point_clouds, 1, 2)
    points_2d_homogeneous = K_norm @ points_transposed
    
    # --- 3. 透视除法 ---
    depths = points_2d_homogeneous[:, 2:3, :] 
    eps = 1e-8
    
    # 这里得到的是归一化坐标 (u_norm, v_norm)
    normalized_coords = points_2d_homogeneous[:, :2, :] / (depths + eps)

    # --- 4. 反归一化：将归一化坐标转换为像素坐标 --- # <<< 关键修改步骤
    u_norm = normalized_coords[:, 0, :]
    v_norm = normalized_coords[:, 1, :]
    
    u_pix = u_norm * image_width
    v_pix = v_norm * image_height
    
    # 将 u_pix 和 v_pix 重新组合成一个张量，方便后续处理
    pixel_coords = torch.stack([u_pix, v_pix], dim=1)
    
    # --- 5. 过滤无效点 (现在基于像素坐标进行过滤) ---
    u_coords = pixel_coords[:, 0, :]
    v_coords = pixel_coords[:, 1, :]
    
    valid_mask = (depths.squeeze(1) > 0) & \
                 (u_coords >= 0) & (u_coords < image_width) & \
                 (v_coords >= 0) & (v_coords < image_height)
    
    # --- 6. 渲染/着色 (Splatting) ---
    bg_color_tensor = torch.tensor(background_color, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    rendered_images = bg_color_tensor.expand(B, 3, image_height, image_width).clone()
    
    for i in range(B):
        mask_i = valid_mask[i]
        if mask_i.sum() == 0:
            continue
            
        valid_coords = pixel_coords[i, :, mask_i]
        valid_colors = point_color[i, mask_i, :]

        u_indices = valid_coords[0, :].long()
        v_indices = valid_coords[1, :].long()
        
        rendered_images[i, :, v_indices, u_indices] = valid_colors.transpose(0, 1)

    return rendered_images


# =============================================================================
# CORE PROCESSING FUNCTIONS
# =============================================================================

def setup_model() -> Pi3:
    """
    Initialize and return the Pi3 model.

    Returns:
        Loaded Pi3 model
    """
    model = Pi3.from_pretrained(Config.MODEL_NAME).to(Config.DEVICE).eval()
    return model


def extract_single_sample_results(scene: Dict, imgs: torch.Tensor, batch: Dict,
                                idx: int) -> Tuple[Dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract results and data for a single sample from batch results.

    Args:
        results: Model inference results
        imgs: Input image batch
        batch: Input data batch
        idx: Sample index

    Returns:
        Tuple of (single_results, single_img, single_grd_mask, single_drone_mask)
    """
    camera_poses = scene.get_im_poses()  # [V, 4, 4]
    ref_pose = camera_poses[0][None]  # Reference pose (satellite)
    camera_poses = torch.einsum('bij, bnjk -> bnik', se3_inverse(ref_pose), camera_poses[None])  # Relative poses to reference
    points = torch.einsum('bij, bnhwj -> bnhwi', se3_inverse(ref_pose), homogenize_points(torch.stack(scene.get_pts3d(), dim=0)[None]))[..., :3]

    single_results = {
        'points': points,
        'camera_poses': camera_poses,
        'focals': scene.get_focals()[0,0] / 512,
        'conf': torch.stack(scene.get_masks(), dim=0)[None, :, :, :, None],
    }

    single_img = imgs[idx:idx+1]
    single_grd_mask = batch['grd_mask'][idx:idx+1].to(Config.DEVICE).float()
    single_drone_mask = batch['drone_mask'][idx:idx+1].to(Config.DEVICE).float()
    grd_rot = batch['grd_rot'][idx].to(Config.DEVICE).float()
    grd_shift_z = batch['grd_shift_z'][idx:idx+1].to(Config.DEVICE).float().item()
    grd_shift_x = batch['grd_shift_x'][idx:idx+1].to(Config.DEVICE).float().item()

    return single_results, single_img, single_grd_mask, single_drone_mask, grd_rot, grd_shift_z, grd_shift_x


def reconstruct_point_cloud(single_results: Dict, single_img: torch.Tensor,
                           single_grd_mask: torch.Tensor, single_drone_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Reconstruct point cloud in reference camera coordinate system.

    Args:
        single_results: Model results for single sample
        single_img: Single sample images
        single_grd_mask: Ground mask
        single_drone_mask: Drone mask

    Returns:
        Tuple of (pts_sat, colors_sat, meter_per_pixel)
    """
    B = single_results['points'].shape[0]
    mask_pts = torch.stack([
        single_grd_mask,
        single_drone_mask
    ], dim=1)

    # Transform to reference camera coordinate system
    camera_poses = single_results['camera_poses']
    pts_all = single_results['points']

    mask_pts = F.interpolate(rearrange(mask_pts, 'b v h w c -> (b v) c h w'), size=(128, 128), mode='bilinear', align_corners=False)
    mask_pts = rearrange(mask_pts, '(b v) c h w -> b v h w c', b=B)
    single_img = F.interpolate(rearrange(single_img, 'b v c h w -> (b v) c h w'), size=(128, 128), mode='bilinear', align_corners=False)
    single_img = rearrange(single_img, '(b v) c h w -> b v c h w', b=B)
    pts_all = F.interpolate(rearrange(pts_all, 'b v h w c -> (b v) c h w'), size=(128, 128), mode='bilinear', align_corners=False)
    pts_all = rearrange(pts_all, '(b v) c h w -> b v h w c', b=B)

    # Extract satellite points and colors
    pts_sat = pts_all[:, 0:1]
    pts_sat = rearrange(pts_sat, 'b v h w c -> b (v h w) c')
    colors_sat = single_img[:, 0:1]
    colors_sat = rearrange(colors_sat, 'b v c h w -> b (v h w) c')
    # Extract ground points and colors
    pts_gd = pts_all[:, 1:]
    ground_cam_pos = camera_poses[:, 1, :3, 3]  # Ground camera is index 1
    # 构建外参矩阵：单位旋转矩阵 + 指定平移
    extrinsics = torch.zeros(1, 4, 4, device=pts_gd.device)
    extrinsics[:, :3, :3] = torch.eye(3, device=pts_gd.device)  # 单位旋转矩阵
    extrinsics[:, :3, 3] = torch.stack([
        ground_cam_pos[:, 0],  # x方向平移
        ground_cam_pos[:, 1],  # y方向平移
        torch.zeros_like(ground_cam_pos[:, 2])  # z方向平移为0
    ], dim=1)
    extrinsics[:, 3, 3] = 1.0  # 齐次坐标
    pts_gd = world_to_camera_broadcast(pts_gd, extrinsics)
    pts_gd = pts_gd * mask_pts
    pts_gd = rearrange(pts_gd, 'b v h w c -> b (v h w) c')
    colors_gd = single_img[:, 1:]
    colors_gd = rearrange(colors_gd, 'b v c h w -> b (v h w) c')

    # Calculate meter per pixel scale
    max_min_range = pts_sat[..., 0].abs().amax() + pts_sat[..., 0].abs().amin()
    meter_per_pixel = max_min_range / Config.SATELLITE_WIDTH

    return pts_sat, colors_sat, pts_gd, colors_gd, meter_per_pixel


def project_ground_camera_to_satellite(camera_poses: torch.Tensor, 
                                       intrinsics: torch.Tensor,
                                       img_width: int, 
                                       img_height: int,
                                       grd_rot: torch.Tensor,
                                       grd_angle_degrees: float,
                                       grd_shift_z: float,
                                       grd_shift_x: float
                                    ) -> Dict[str, Any]:
    """
    Project ground camera position to satellite view coordinates.

    Args:
        camera_poses: Camera pose transformations
        intrinsics: Camera intrinsics matrix
        img_width: Image width
        img_height: Image height
        grd_rot: Ground camera rotation matrix
        grd_angle_degrees: Ground camera angle in degrees
        grd_shift_z: Ground camera shift in z direction
        grd_shift_x: Ground camera shift in x direction

    Returns:
        Dictionary containing projection information
    """
    center_x, center_y = img_width // 2, img_height // 2
    K = intrinsics if intrinsics.dim() == 2 else intrinsics[0]
    grd_camera = SAT_TO_OPENCV.to(camera_poses.device) @ camera_poses[0, 1]
    ground_cam_pos = camera_poses[0, 1, :3, 3]  # Ground camera is index 1
    ground_cam_rot = grd_camera[:3, :3]
    ground_angle_radians = torch.atan2(ground_cam_rot[0, 2], ground_cam_rot[2, 2])
    ground_angle_degrees = torch.rad2deg(ground_angle_radians)

    delta_grd_rot = compare_rotations_manual(ground_cam_rot, grd_rot)

    meter_per_pixel = 140.0 / img_width  # 140m for 504px
    projection_info = {
        'valid': False,
        'delta_x': None,
        'delta_y': None,
        'pixel_x': None,
        'pixel_y': None,
        'distance': None,
        'delta_grd_rot': delta_grd_rot.item(),
        'grd_angle_degrees': ground_angle_degrees.item(),
        'delta_grd_angle': np.abs(ground_angle_degrees.item() - grd_angle_degrees)
    }

    # Project ground camera to satellite view
    if ground_cam_pos[2] > 0:
        proj = K @ ground_cam_pos
        x_norm = proj[0] / (proj[2] + 1e-6)
        y_norm = proj[1] / (proj[2] + 1e-6)
        pixel_x = int(x_norm * img_width)
        pixel_y = int(y_norm * img_height)

        meter_x = (pixel_x - center_x) * meter_per_pixel
        meter_y = (pixel_y - center_y) * meter_per_pixel

        if 0 <= pixel_x < img_width and 0 <= pixel_y < img_height:
            distance = np.sqrt((meter_x - grd_shift_z)**2 + (meter_y - grd_shift_x)**2)
            projection_info.update({
                'valid': True,
                'delta_x': np.abs(meter_x - grd_shift_z),
                'delta_y': np.abs(meter_y - grd_shift_x),
                'pixel_x': pixel_x,
                'pixel_y': pixel_y,
                'distance': distance  # Scale to real-world distance(140m for 504px)
            })

    return projection_info, meter_per_pixel


def save_debug_images(single_img: torch.Tensor, pts_sat: torch.Tensor, colors_sat: torch.Tensor,
                     pts_gd: torch.Tensor, colors_gd: torch.Tensor, meter_per_pixel: float):
    """
    Save debug images for visualization (optional).

    Args:
        single_img: Single sample images
        pts_sat: Satellite points
        colors_sat: Satellite colors
        pts_gd: Ground points
        colors_gd: Ground colors
        meter_per_pixel: Scale factor
    """
    # Project and save ground to satellite view
    pts_gd_s = torch.einsum('ij, bnj -> bni', SAT_TO_OPENCV.to(Config.DEVICE).to(pts_gd.dtype), homogenize_points(pts_gd))[..., :3]
    g2s = forward_project(colors_gd, pts_gd_s, meter_per_pixel=meter_per_pixel, sat_width=Config.SATELLITE_WIDTH)
    test_img = TO_PIL_IMAGE(g2s[0].cpu())
    test_img.save('g2s.png')

    # Save individual view images
    for view_idx, view_name in enumerate(['sat', 'grd', 'drone']):
        test_img = TO_PIL_IMAGE(single_img[0, view_idx].cpu())
        test_img.save(f'{view_name}.png')

    # Project and save satellite to satellite view
    pts_sat_s = torch.einsum('ij, bnj -> bni', SAT_TO_OPENCV.to(Config.DEVICE).to(pts_sat.dtype), homogenize_points(pts_sat))[..., :3]
    s2s = forward_project(colors_sat, pts_sat_s, meter_per_pixel=meter_per_pixel, sat_width=Config.SATELLITE_WIDTH)
    test_img = TO_PIL_IMAGE(s2s[0].cpu())
    test_img.save('s2s.png')


def visualize_coordinates_on_satellite(satellite_image, pixel_x: int, pixel_y: int,
                                      center_x: int, center_y: int, 
                                      ground_angle_pred: float, ground_angle_gt: float,
                                      sample_idx: int,
                                      save_dir: str = "coordinate_visualizations"):
    """
    Visualize pixel_x, pixel_y and center_x, center_y coordinates on satellite image,
    including ground camera orientation angle.

    Args:
        satellite_image: Tensor [B, 3, H, W] of satellite projection image
        pixel_x: Ground camera projected x coordinate
        pixel_y: Ground camera projected y coordinate
        center_x: Satellite image center x coordinate
        center_y: Satellite image center y coordinate
        ground_angle_pred: Predicted ground camera angle in degrees (clockwise from right)
        ground_angle_gt: Ground truth ground camera angle in degrees (clockwise from right)
        sample_idx: Sample index for filename
        save_dir: Directory to save visualization results
    """

    # Create save directory if it doesn't exist
    os.makedirs(save_dir.split('/')[0], exist_ok=True)
    satellite_img = TO_PIL_IMAGE(satellite_image.cpu())

    draw = ImageDraw.Draw(satellite_img)

    # Try to use a larger font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None

    # Draw center point (satellite image center) - blue
    center_size = 8
    draw.ellipse([(center_x - center_size, center_y - center_size),
                    (center_x + center_size, center_y + center_size)],
                fill='blue', outline='white', width=2)

    # Draw ground camera projection point - red
    point_size = 6
    draw.ellipse([(pixel_x - point_size, pixel_y - point_size),
                    (pixel_x + point_size, pixel_y + point_size)],
                fill='red', outline='white', width=2)

    # Draw line connecting the two points
    draw.line([(center_x, center_y), (pixel_x, pixel_y)],
                fill='yellow', width=2)

    # Draw ground camera orientation arrow
    # Convert angle from degrees to radians (clockwise from right direction)
    angle_rad = math.radians(ground_angle_pred)
    arrow_length = 40  # Length of the orientation arrow in pixels

    # Calculate arrow end point
    arrow_end_x = pixel_x + arrow_length * math.cos(angle_rad)
    arrow_end_y = pixel_y + arrow_length * math.sin(angle_rad)

    # Draw the main arrow line
    draw.line([(pixel_x, pixel_y), (arrow_end_x, arrow_end_y)],
                fill='green', width=3)

    # Draw arrowhead
    arrowhead_length = 10
    arrowhead_angle = math.radians(150)  # Angle for arrowhead lines

    # Calculate arrowhead points
    left_x = arrow_end_x + arrowhead_length * math.cos(angle_rad + arrowhead_angle)
    left_y = arrow_end_y + arrowhead_length * math.sin(angle_rad + arrowhead_angle)
    right_x = arrow_end_x + arrowhead_length * math.cos(angle_rad - arrowhead_angle)
    right_y = arrow_end_y + arrowhead_length * math.sin(angle_rad - arrowhead_angle)

    # Draw arrowhead lines
    draw.line([(arrow_end_x, arrow_end_y), (left_x, left_y)],
                fill='green', width=3)
    draw.line([(arrow_end_x, arrow_end_y), (right_x, right_y)],
                fill='green', width=3)

    # Draw ground truth angle arrow at center position
    # Convert ground truth angle from degrees to radians (clockwise from right direction)
    gt_angle_rad = math.radians(ground_angle_gt)
    gt_arrow_length = 40  # Length of the ground truth orientation arrow in pixels

    # Calculate ground truth arrow end point from center
    gt_arrow_end_x = center_x + gt_arrow_length * math.cos(gt_angle_rad)
    gt_arrow_end_y = center_y + gt_arrow_length * math.sin(gt_angle_rad)

    # Draw the ground truth arrow line (orange color to distinguish from predicted)
    draw.line([(center_x, center_y), (gt_arrow_end_x, gt_arrow_end_y)],
                fill='orange', width=3)

    # Draw ground truth arrowhead
    gt_arrowhead_length = 10
    gt_arrowhead_angle = math.radians(150)  # Angle for arrowhead lines

    # Calculate ground truth arrowhead points
    gt_left_x = gt_arrow_end_x + gt_arrowhead_length * math.cos(gt_angle_rad + gt_arrowhead_angle)
    gt_left_y = gt_arrow_end_y + gt_arrowhead_length * math.sin(gt_angle_rad + gt_arrowhead_angle)
    gt_right_x = gt_arrow_end_x + gt_arrowhead_length * math.cos(gt_angle_rad - gt_arrowhead_angle)
    gt_right_y = gt_arrow_end_y + gt_arrowhead_length * math.sin(gt_angle_rad - gt_arrowhead_angle)

    # Draw ground truth arrowhead lines
    draw.line([(gt_arrow_end_x, gt_arrow_end_y), (gt_left_x, gt_left_y)],
                fill='orange', width=3)
    draw.line([(gt_arrow_end_x, gt_arrow_end_y), (gt_right_x, gt_right_y)],
                fill='orange', width=3)

    # Add labels
    if font:
        draw.text((center_x + 10, center_y - 25), "Center", fill='blue', font=font)
        draw.text((pixel_x + 10, pixel_y - 25), "Ground Camera", fill='red', font=font)

        # Add orientation labels for both predicted and ground truth angles
        pred_orientation_text_x = arrow_end_x + 10
        pred_orientation_text_y = arrow_end_y - 10
        draw.text((pred_orientation_text_x, pred_orientation_text_y),
                 f"Pred: {ground_angle_pred:.1f}°", fill='green', font=font)

        gt_orientation_text_x = gt_arrow_end_x + 10
        gt_orientation_text_y = gt_arrow_end_y - 10
        draw.text((gt_orientation_text_x, gt_orientation_text_y),
                 f"GT: {ground_angle_gt:.1f}°", fill='orange', font=font)

        # Add distance and angle information if available
        distance_pixels = np.sqrt((pixel_x - center_x)**2 + (pixel_y - center_y)**2)
        distance_meters = distance_pixels * Config.PIXEL_TO_METER
        draw.text((10, 10), f"Sample {sample_idx}", fill='white', font=font)
        draw.text((10, 30), f"Distance: {distance_pixels:.1f} px ({distance_meters:.2f} m)",
                    fill='white', font=font)
        draw.text((10, 50), f"Pred Angle: {ground_angle_pred:.1f}°", fill='green', font=font)
        draw.text((10, 70), f"GT Angle: {ground_angle_gt:.1f}°", fill='orange', font=font)
        draw.text((10, 90), "Angles: clockwise from right", fill='white', font=font)
    else:
        # Fallback for drawing text without font
        draw.text((center_x + 10, center_y - 25), "Center", fill='blue')
        draw.text((pixel_x + 10, pixel_y - 25), "Ground Cam", fill='red')

    # Save the visualization
    output_filename = os.path.join(save_dir)
    satellite_img.save(output_filename)

    print(f"✅ Coordinate visualization saved to: {output_filename}")
    return output_filename

def calculate_ground_camera_distance(batch: Dict, model: Pi3) -> Dict[str, List]:
    """
    Process a batch and calculate ground camera to satellite center distance.

    Args:
        batch: Input data batch
        model: Pi3 model

    Returns:
        Dictionary containing results for the batch
    """
    batch_size = batch['ground'].shape[0]
    results_dict = {
        'distances': [],
        'sample_indices': [],
        'projections': [],
        'image_sizes': [],
        'delta_grd_angle': [],
        'delta_grd_rot': [],
        'delta_shift_z': [],
        'delta_shift_x': []
    }


    # Prepare input images
    imgs = torch.stack([batch['sat_pi3'], batch['ground'], batch['drone']], dim=1)
    imgs = imgs.to(Config.DEVICE)

    # Run model inference
    # results = model(imgs)

    # Process each sample in the batch
    for i in range(batch_size):

        imgs_all = [
            {
                'img': batch['sat_pi3'][i:i+1].to(Config.DEVICE),
                'mask': batch['sat_mask'][i:i+1].to(Config.DEVICE),
                'true_shape': np.array([[518, 518]], dtype=np.int32),
                'idx': 0,
                'instance': '0'
            },
            {
                'img': batch['ground'][i:i+1].to(Config.DEVICE),
                'mask': batch['grd_mask'][i:i+1].to(Config.DEVICE),
                'true_shape': np.array([[518, 518]], dtype=np.int32),
                'idx': 1,
                'instance': '1'
            },
            {
                'img': batch['drone'][i:i+1].to(Config.DEVICE),
                'mask': batch['drone_mask'][i:i+1].to(Config.DEVICE),
                'true_shape': np.array([[518, 518]], dtype=np.int32),
                'idx': 2,
                'instance': '2'
            }
        ]

        output = {
            'view1':{
                'img': [],
                'true_shape': [],
                'idx': [],
                'instance': []
            },
            'view2':{
                'img': [],
                'true_shape': [],
                'idx': [],
                'instance': []
            },
            'pred1':{
                'pts3d': None,
                'conf': None,
            },
            'pred2':{
                'pts3d_in_other_view': None,
                'conf': None,
            },
        }


        pairs = make_pairs(imgs_all, scene_graph='complete', prefilter=None, symmetrize=False)
        for pair in pairs:
            pair_img = torch.stack([pair[0]['img'], pair[1]['img']], dim=1)
            # Run model inference
            with torch.no_grad():
                results = model(pair_img)
                predictions = convert_pi3_to_vggt_format(results)
            
                output['view1']['img'] = torch.cat((output['view1']['img'], pair[0]['img']), dim=0) if len(output['view1']['img'])>0 else pair[0]['img']
                output['view1']['true_shape'] = torch.cat((output['view1']['true_shape'], torch.tensor(pair[0]['true_shape']).to(Config.DEVICE)), dim=0) if len(output['view1']['true_shape'])>0 else torch.tensor(pair[0]['true_shape']).to(Config.DEVICE)
                output['view1']['idx'].append(pair[0]['idx'])
                output['view1']['instance'].append(pair[0]['instance'])
                output['view2']['img'] = torch.cat((output['view2']['img'], pair[1]['img']), dim=0) if len(output['view2']['img'])>0 else pair[1]['img']
                output['view2']['true_shape'] = torch.cat((output['view2']['true_shape'], torch.tensor(pair[1]['true_shape']).to(Config.DEVICE)), dim=0) if len(output['view2']['true_shape'])>0 else torch.tensor(pair[1]['true_shape']).to(Config.DEVICE)
                output['view2']['idx'].append(pair[1]['idx'])
                output['view2']['instance'].append(pair[1]['instance'])

                output['pred1']['pts3d'] = predictions['world_points'][0, :1] if output['pred1']['pts3d'] is None else torch.cat((output['pred1']['pts3d'], predictions['world_points'][0, :1]), dim=0)
                output['pred1']['conf'] = predictions['world_points_conf'][0, :1] if output['pred1']['conf'] is None else torch.cat((output['pred1']['conf'], predictions['world_points_conf'][0, :1]), dim=0)
                output['pred2']['pts3d_in_other_view'] = predictions['world_points'][0, 1:2] if output['pred2']['pts3d_in_other_view'] is None else torch.cat((output['pred2']['pts3d_in_other_view'], predictions['world_points'][0, 1:2]), dim=0)
                output['pred2']['conf'] = predictions['world_points_conf'][0, 1:2] if output['pred2']['conf'] is None else torch.cat((output['pred2']['conf'], predictions['world_points_conf'][0, 1:2]), dim=0)

            # colors = rearrange(pair_img, 'b v c h w -> (b v h w) c')
            # points = rearrange(pts_all, 'b v h w c -> (b v h w) c')
            # save_points_to_ply(points, colors, 'ground_points.ply')

        scene = global_aligner(output, device=Config.DEVICE, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = scene.compute_global_alignment(init="mst", niter=300, schedule='cosine', lr=0.001)

        with torch.no_grad():
            sample_idx = batch['index'][i].item()
            grd_gt_angle_degrees = batch['grd_gt_angle_degrees'][i].item()
            paths = batch['paths']
            # Extract single sample data
            single_results, single_img, single_grd_mask, single_drone_mask, grd_rot, grd_shift_z, grd_shift_x = extract_single_sample_results(
                scene, imgs, batch, i
            )

            # Reconstruct point cloud
            pts_sat, colors_sat, pts_gd, colors_gd, meter_per_pixel= reconstruct_point_cloud(
                single_results, single_img, single_grd_mask, single_drone_mask
            )

            # Save debug images (optional)
            save_debug_images(single_img, pts_sat, colors_sat, pts_gd, colors_gd, meter_per_pixel)

            # Recover intrinsics from local points
            points = single_results["points"][0, 0]
            masks = single_results["conf"][0, 0].float()
            points = points * masks
            width = torch.amax(points[..., 0], dim=(0,1)) - torch.amin(points[..., 0], dim=(0,1))
            height = torch.amax(points[..., 1], dim=(0,1)) - torch.amin(points[..., 1], dim=(0,1))

            # Recover focal length
            fx, fy = single_results['focals'], single_results['focals']
            intrinsics = intrinsics_from_focal_center(fx, fy, 0.5, 0.5)

            pts_sat = pts_sat.reshape(1, -1, 3)
            colors_sat = colors_sat.reshape(1, -1, 3)
            # Get project grd2sat img
            g2s_direct_proj = project_point_clouds(torch.cat((pts_gd, pts_sat), dim=1), torch.cat((colors_gd, colors_sat), dim=1), intrinsics, Config.SATELLITE_WIDTH, Config.SATELLITE_WIDTH)
            g2s_img = TO_PIL_IMAGE(g2s_direct_proj[0].cpu())
            g2s_img.save('all2s_proj.png')

            # Get project grd2sat img
            g2s_direct_proj = project_point_clouds(pts_gd, colors_gd, intrinsics, int(Config.SATELLITE_WIDTH * 0.67), int(Config.SATELLITE_WIDTH * 0.67))
            g2s_img = TO_PIL_IMAGE(g2s_direct_proj[0].cpu().clamp(0,1))
            g2s_img.save('g2s_proj.png')

            s2s_direct_proj = project_point_clouds(pts_sat, colors_sat, intrinsics, int(Config.SATELLITE_WIDTH * 0.67), int(Config.SATELLITE_WIDTH * 0.67))
            s2s_img = TO_PIL_IMAGE(s2s_direct_proj[0].cpu().clamp(0,1))
            s2s_img.save('s2s_proj.png')

            sat_ref_img = batch['sat_ref'][i:i+1].to(Config.DEVICE)
            sat_ref_img = TO_PIL_IMAGE(sat_ref_img[0].cpu().clamp(0,1))
            sat_ref_img.save('sat_ref.png')

            # Get image dimensions and project ground camera
            img_width, img_height = imgs[i, 0].shape[-1], imgs[i, 0].shape[-2]
            reference_cam = single_results['camera_poses'][:, 0]
            camera_poses = torch.einsum('bij, bnjk -> bnik', se3_inverse(reference_cam), single_results['camera_poses'])

            # Vis Points Cloud
            # colors = rearrange(single_img, 'b v c h w -> (b v h w) c')
            # points = rearrange(single_results["points"], 'b v h w c -> (b v h w) c')
            # masks = colors.any(dim=-1, keepdim=True).float()
            # points = points * masks
            # save_points_to_ply(points, colors, 'ground_points.ply')

            projection_info, meter_per_pixel = project_ground_camera_to_satellite(
                camera_poses, intrinsics, img_width, img_height, grd_rot, grd_gt_angle_degrees, grd_shift_z, grd_shift_x
            )

            # Create coordinate visualization if projection is valid
            if projection_info['valid']:
                center_x, center_y = img_width // 2, img_height // 2
                pixel_x = projection_info['pixel_x']
                pixel_y = projection_info['pixel_y']

                # Call visualization function
                visualize_coordinates_on_satellite(
                    imgs[i, 0],
                    pixel_x=pixel_x,
                    pixel_y=pixel_y,
                    center_x=center_x + grd_shift_z / meter_per_pixel,
                    center_y=center_y + grd_shift_x / meter_per_pixel,
                    ground_angle_pred=projection_info['grd_angle_degrees'],
                    ground_angle_gt=grd_gt_angle_degrees,
                    sample_idx=sample_idx,
                    save_dir=f"coordinate_visualizations/sat_pred_{len(results_dict['distances'])}.png"
                )

            # Store results
            results_dict['distances'].append(projection_info['distance'])
            results_dict['delta_grd_angle'].append(projection_info['delta_grd_angle'])
            results_dict['delta_grd_rot'].append(projection_info['delta_grd_rot'])
            results_dict['delta_shift_z'].append(projection_info['delta_x'])
            results_dict['delta_shift_x'].append(projection_info['delta_y'])
            results_dict['sample_indices'].append(sample_idx)
            results_dict['projections'].append(projection_info)
            results_dict['image_sizes'].append((img_width, img_height))
            
            grd_camera = {
                'pts_gd': pts_gd,  # [1, N, 3]
                'intrinsics': intrinsics, # [1, 3, 3]
                'width': width, # [1]
                'height': height, # [1]
            }
            grd_path = paths['ground'][i].split('/')
            grd_name = grd_path[-1].replace('.jpeg.jpg', '')
            drone_path = paths['drone'][i].split('/')
            drone_name = drone_path[-1].replace('.jpeg.jpg', '')
            # save_path = os.path.join('/', *grd_path[0:5], 'grd_camera', f'{grd_name}_{drone_name}_grd_camera_500_rot.pt')
            # 如果目录不存在就新建目录
            # os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # torch.save(grd_camera, save_path)

    return results_dict


# =============================================================================
# STATISTICS AND OUTPUT FUNCTIONS
# =============================================================================

def calculate_statistics(distances: List[float],
                         delta_grd_angle: List[float],
                         delta_grd_rot: List[float],
                         delta_shift_z: List[float],
                         delta_shift_x: List[float]
                         ) -> Dict[str, float]:
    """
    Calculate statistical measures for distances.

    Args:
        distances: List of distance values

    Returns:
        Dictionary containing statistics
    """
    return {
        'dis_mean': np.mean(distances),
        'dis_std': np.std(distances),
        'dis_min': np.min(distances),
        'dis_max': np.max(distances),
        'dis_median': np.median(distances),
        'delta_grd_rot': np.mean(delta_grd_rot),
        'delta_grd_angle': np.mean(delta_grd_angle),
        'delta_shift_z': delta_shift_z,
        'delta_shift_z_mean': np.mean(delta_shift_z),
        'delta_shift_x': delta_shift_x,
        'delta_shift_x_mean': np.mean(delta_shift_x),
    }


def save_results_to_file(all_distances: List[float], all_sample_indices: List[int],
                        total_samples: int, valid_projections_count: int,
                        stats: Dict[str, float], filename: str = Config.OUTPUT_FILE):
    """
    Save analysis results to a text file.

    Args:
        all_distances: List of valid distances
        all_sample_indices: List of sample indices
        total_samples: Total number of samples processed
        valid_projections_count: Number of valid projections
        stats: Statistical measures
        filename: Output filename
    """
    with open(filename, 'w') as f:
        f.write("Ground Camera to Satellite Center Distance Analysis\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Valid projections: {valid_projections_count}\n")
        f.write(f"Success rate: {100*valid_projections_count/total_samples:.2f}%\n\n")
        f.write(f"Distance Statistics (pixels):\n")
        f.write(f"  Mean: {stats['dis_mean']:.2f} ± {stats['dis_std']:.2f}\n")
        f.write(f"  Median: {stats['dis_median']:.2f}\n")
        f.write(f"  Min: {stats['dis_min']:.2f}\n")
        f.write(f"  Max: {stats['dis_max']:.2f}\n\n")
        f.write(f"  Heading Differences (degrees): {stats['delta_grd_angle']:.2f}\n")
        f.write(f"  Ground Camera Rot Differences (degrees): {stats['delta_grd_rot']:.2f}\n")
        f.write(f"Delta Shift Z (meters): {stats['delta_shift_z_mean']:.2f}\n")
        f.write(f"Delta Shift X (meters): {stats['delta_shift_x_mean']:.2f}\n\n")
        metrics = [1, 3, 5]
        f.write("Lateral Accuracy:\n")
        for threshold in metrics:
            pred_acc = np.sum(np.array(stats['delta_shift_z']) < threshold) / len(stats['delta_shift_z']) * 100
            line = f'  Distance within {threshold}m: {pred_acc:.2f}% (pred)\n'
            f.write(line)
        f.write("\nLongitudinal Accuracy:\n")
        for threshold in metrics:
            pred_acc = np.sum(np.array(stats['delta_shift_x']) < threshold) / len(stats['delta_shift_x']) * 100
            line = f'  Distance within {threshold}m: {pred_acc:.2f}% (pred)\n'
            f.write(line)
        f.write("\n")
        f.write("Individual Results:\n")
        for idx, dist in zip(all_sample_indices, all_distances):
            f.write(f"  Sample {idx}: {dist:.2f} pixels\n")


def print_statistics(stats: Dict[str, float], total_samples: int, valid_projections_count: int):
    """
    Print statistical analysis results.

    Args:
        stats: Statistical measures
        total_samples: Total number of samples processed
        valid_projections_count: Number of valid projections
    """
    print(f"\n📊 STATISTICS:")
    print(f"Total samples processed: {total_samples}")
    print(f"Valid projections: {valid_projections_count}")
    print(f"Invalid projections: {total_samples - valid_projections_count}")
    print(f"Success rate: {100*valid_projections_count/total_samples:.2f}%")
    print(f"\nGround camera to satellite center distances (pixels):")
    print(f"  Mean: {stats['dis_mean']:.2f} ± {stats['dis_std']:.2f}")
    print(f"  Median: {stats['dis_median']:.2f}")
    print(f"  Min: {stats['dis_min']:.2f}")
    print(f"  Max: {stats['dis_max']:.2f}")
    print(f"  Heading Differences (degrees): {stats['delta_grd_angle']:.2f}")
    print(f"  Ground Camera Rot Differences (degrees): {stats['delta_grd_rot']:.2f}")
    print(f"\nDelta Shift Z (meters): {stats['delta_shift_z_mean']:.2f}")
    print(f"Delta Shift X (meters): {stats['delta_shift_x_mean']:.2f}")
    metrics = [1, 3, 5]
    # Lateral accuracy
    delta_shift_z = np.array(stats['delta_shift_z']) if isinstance(stats['delta_shift_z'], list) else stats['delta_shift_z']
    for threshold in metrics:
        pred_acc = np.sum(delta_shift_z < threshold) / len(delta_shift_z) * 100
        line = f'Lateral distance within {threshold}m: {pred_acc:.2f}% (pred)\n'
        print(line, end='')

    # Longitudinal accuracy
    delta_shift_x = np.array(stats['delta_shift_x']) if isinstance(stats['delta_shift_x'], list) else stats['delta_shift_x']
    for threshold in metrics:
        pred_acc = np.sum(delta_shift_x < threshold) / len(delta_shift_x) * 100
        line = f'Longitudinal distance within {threshold}m: {pred_acc:.2f}% (pred)\n'
        print(line, end='')


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("Initializing ground camera to satellite distance analysis...")

    # Setup model and data
    model = setup_model()
    test_dataloader = create_test_dataloader(test_file_path='/data/zhongyao/aer-grd-map/test_files_1029.txt', batch_size=Config.BATCH_SIZE, shuffle=False, amount=0.02)

    print(f"Dataset contains {len(test_dataloader.dataset)} samples")
    print(f"Processing {len(test_dataloader)} batches")

    # Collect all results
    all_distances = []
    all_grd_angle = []
    all_grd_rot = []
    all_delta_shift_z = []
    all_delta_shift_x = []
    all_sample_indices = []
    valid_projections_count = 0
    total_samples = 0

    print("\nStarting batch processing...")
    for batch_idx, batch in enumerate(test_dataloader):
        print(f"\n{'='*60}")
        print(f"Processing batch {batch_idx + 1}/{len(test_dataloader)}")

        # Process the batch
        batch_results = calculate_ground_camera_distance(batch, model)

        # Collect results
        for i, (dist, delta_grd_angle, delta_grd_rot, delta_shift_z, delta_shift_x, idx, proj) in enumerate(
            zip(batch_results['distances'],
                batch_results['delta_grd_angle'],
                batch_results['delta_grd_rot'],
                batch_results['delta_shift_z'],
                batch_results['delta_shift_x'],
                batch_results['sample_indices'],
                batch_results['projections']
            )):
            total_samples += 1
            all_sample_indices.append(idx)

            if proj['valid']:
                all_distances.append(dist)
                all_grd_angle.append(delta_grd_angle)
                all_grd_rot.append(delta_grd_rot)
                all_delta_shift_z.append(delta_shift_z)
                all_delta_shift_x.append(delta_shift_x)
                valid_projections_count += 1

        # Memory cleanup
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("Processing complete!")

    # Calculate and display statistics
    if all_distances:
        stats = calculate_statistics(all_distances, all_grd_angle, all_grd_rot, all_delta_shift_z, all_delta_shift_x)
        print_statistics(stats, total_samples, valid_projections_count)
        save_results_to_file(all_distances, all_sample_indices, total_samples, valid_projections_count, stats)
        print(f"\n💾 Results saved to {Config.OUTPUT_FILE}")
    else:
        print("\n❌ No valid projections found!")
        print("This might indicate an issue with the camera projection process.")


if __name__ == "__main__":
    main()