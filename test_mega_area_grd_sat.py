"""
Ground Camera to Satellite Center Distance Analysis

This script processes satellite, ground, and drone image triplets to calculate
the distance from ground camera positions to satellite image centers.
"""

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
import os

# Local imports
from pi3.models.pi3 import Pi3
from dataset_aer_grd_drone import load_test_triplet, create_test_dataloader, TestTripletDataset
from pi3.utils.geometry import intrinsics_from_focal_center, se3_inverse, homogenize_points, depth_edge


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

def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None,
                           dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    """
    Generate normalized UV coordinates with left-top corner as (-width/diagonal, -height/diagonal)
    and right-bottom corner as (width/diagonal, height/diagonal).

    Args:
        width: Image width
        height: Image height
        aspect_ratio: Optional aspect ratio (defaults to width/height)
        dtype: Tensor data type
        device: Tensor device

    Returns:
        UV coordinates tensor of shape (H, W, 2)
    """
    if aspect_ratio is None:
        aspect_ratio = width / height

    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv


def solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray) -> Tuple[float, float]:
    """
    Solve `min |focal * xy / (z + shift) - uv|` with respect to shift and focal.

    Args:
        uv: UV coordinates
        xyz: 3D points

    Returns:
        Tuple of (optimal_shift, optimal_focal)
    """
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[:, None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        err = (f * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    optim_shift = solution['x'].squeeze().astype(np.float32)

    xy_proj = xy / (z + optim_shift)[:, None]
    optim_focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()

    return optim_shift, optim_focal


def solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray, focal: float) -> float:
    """
    Solve `min |focal * xy / (z + shift) - uv|` with respect to shift.

    Args:
        uv: UV coordinates
        xyz: 3D points
        focal: Known focal length

    Returns:
        Optimal shift value
    """
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[:, None]
        err = (focal * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    optim_shift = solution['x'].squeeze().astype(np.float32)

    return optim_shift


def recover_focal_shift(points: torch.Tensor, mask: torch.Tensor = None,
                       focal: torch.Tensor = None,
                       downsample_size: Tuple[int, int] = Config.DOWNSAMPLE_SIZE) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Recover the depth map and FoV from a point map with unknown z shift and focal.

    Args:
        points: Point map tensor of shape (..., H, W, 3)
        mask: Optional mask tensor
        focal: Optional focal length tensor
        downsample_size: Size for downsampling for efficient processing

    Returns:
        Tuple of (focal, shift) tensors
    """
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]

    points = points.reshape(-1, *shape[-3:])
    mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    focal = focal.reshape(-1) if focal is not None else None
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)

    # Downsample for efficient processing
    points_lr = F.interpolate(points.permute(0, 3, 1, 2), downsample_size, mode='nearest').permute(0, 2, 3, 1)
    uv_lr = F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode='nearest').squeeze(0).permute(1, 2, 0)
    mask_lr = None if mask is None else F.interpolate(mask.to(torch.float32).unsqueeze(1), downsample_size, mode='nearest').squeeze(1) > 0

    # Convert to numpy for optimization
    uv_lr_np = uv_lr.cpu().numpy()
    points_lr_np = points_lr.detach().cpu().numpy()
    focal_np = focal.cpu().numpy() if focal is not None else None
    mask_lr_np = None if mask is None else mask_lr.cpu().numpy()

    optim_shift, optim_focal = [], []
    for i in range(points.shape[0]):
        points_lr_i_np = points_lr_np[i] if mask is None else points_lr_np[i][mask_lr_np[i]]
        uv_lr_i_np = uv_lr_np if mask is None else uv_lr_np[mask_lr_np[i]]

        if uv_lr_i_np.shape[0] < 2:
            optim_focal.append(1)
            optim_shift.append(0)
            continue

        if focal is None:
            optim_shift_i, optim_focal_i = solve_optimal_focal_shift(uv_lr_i_np, points_lr_i_np)
            optim_focal.append(float(optim_focal_i))
        else:
            optim_shift_i = solve_optimal_shift(uv_lr_i_np, points_lr_i_np, focal_np[i])
        optim_shift.append(float(optim_shift_i))

    optim_shift = torch.tensor(optim_shift, device=points.device, dtype=points.dtype).reshape(shape[:-3])

    if focal is None:
        optim_focal = torch.tensor(optim_focal, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    else:
        optim_focal = focal.reshape(shape[:-3])

    return optim_focal, optim_shift


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


def extract_single_sample_results(results: Dict, imgs: torch.Tensor, batch: Dict,
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
    single_results = {}
    for key in results.keys():
        if isinstance(results[key], torch.Tensor):
            single_results[key] = results[key][idx:idx+1]

    single_img = imgs[idx:idx+1]
    single_grd_mask = batch['grd_mask'][idx:idx+1].to(Config.DEVICE).float()
    single_drone_mask = batch['drone_mask'][idx:idx+1].to(Config.DEVICE).float()

    return single_results, single_img, single_grd_mask, single_drone_mask


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
    mask_pts = torch.stack([
        torch.ones_like(single_grd_mask).to(single_grd_mask.device),
        single_grd_mask,
        single_drone_mask
    ], dim=1)

    # Transform to reference camera coordinate system
    reference_cam = single_results['camera_poses'][:, 0]
    pts_all = single_results['points']
    pts_all = torch.einsum('bij, bnhwj -> bnhwi', se3_inverse(reference_cam), homogenize_points(pts_all))[..., :3]
    # pts_all = torch.einsum('ij, bnhwj -> bnhwi', SAT_TO_OPENCV.to(Config.DEVICE).to(pts_all.dtype), homogenize_points(pts_all))[..., :3]
    pts_all = pts_all * mask_pts
    # Extract satellite points and colors
    pts_sat = pts_all[:, 0:1]
    pts_sat = rearrange(pts_sat, 'b v h w c -> b (v h w) c')
    colors_sat = single_img[:, 0:1]
    colors_sat = rearrange(colors_sat, 'b v c h w -> b (v h w) c')
    # Extract ground points and colors
    pts_gd = pts_all[:, 1:]
    pts_gd = rearrange(pts_gd, 'b v h w c -> b (v h w) c')
    colors_gd = single_img[:, 1:]
    colors_gd = rearrange(colors_gd, 'b v c h w -> b (v h w) c')

    # Calculate meter per pixel scale
    max_min_range = pts_sat[..., 0].abs().amax() + pts_sat[..., 0].abs().amin()
    meter_per_pixel = max_min_range / Config.SATELLITE_WIDTH

    return pts_sat, colors_sat, pts_gd, colors_gd, meter_per_pixel


def project_ground_camera_to_satellite(camera_poses: torch.Tensor, intrinsics: torch.Tensor,
                                     img_width: int, img_height: int) -> Dict[str, Any]:
    """
    Project ground camera position to satellite view coordinates.

    Args:
        camera_poses: Camera pose transformations
        intrinsics: Camera intrinsics matrix
        img_width: Image width
        img_height: Image height

    Returns:
        Dictionary containing projection information
    """
    center_x, center_y = img_width // 2, img_height // 2
    K = intrinsics if intrinsics.dim() == 2 else intrinsics[0]
    ground_cam_pos = camera_poses[0, 1, :3, 3]  # Ground camera is index 1
    ground_cam_rot = camera_poses[0, 1, :3, :3]
    ground_angle_radians = torch.atan2(ground_cam_rot[1, 2], ground_cam_rot[0, 2])
    ground_angle_degrees = torch.rad2deg(ground_angle_radians)

    projection_info = {
        'valid': False,
        'pixel_x': None,
        'pixel_y': None,
        'distance': None,
        'ground_angle_degrees': ground_angle_degrees.item()
    }

    # Project ground camera to satellite view
    if ground_cam_pos[2] > 0:
        proj = K @ ground_cam_pos
        x_norm = proj[0] / (proj[2] + 1e-6)
        y_norm = proj[1] / (proj[2] + 1e-6)
        pixel_x = int(x_norm * img_width)
        pixel_y = int(y_norm * img_height)

        if 0 <= pixel_x < img_width and 0 <= pixel_y < img_height:
            distance = np.sqrt((pixel_x - center_x)**2 + (pixel_y - center_y)**2)
            projection_info.update({
                'valid': True,
                'pixel_x': pixel_x,
                'pixel_y': pixel_y,
                'distance': distance * Config.PIXEL_TO_METER
            })

    return projection_info


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
        'image_sizes': []
    }

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16):
            # Prepare input images
            imgs = torch.stack([batch['satellite'], batch['ground'], batch['drone']], dim=1)
            imgs = imgs.to(Config.DEVICE)

            # Run model inference
            results = model(imgs)

            # Process each sample in the batch
            for i in range(batch_size):
                sample_idx = batch['index'][i].item()
                grd_gt_angle_degrees = batch['grd_gt_angle_degrees'][i].item()
                # Extract single sample data
                single_results, single_img, single_grd_mask, single_drone_mask = extract_single_sample_results(
                    results, imgs, batch, i
                )

                # Reconstruct point cloud
                pts_sat, colors_sat, pts_gd, colors_gd, meter_per_pixel = reconstruct_point_cloud(
                    single_results, single_img, single_grd_mask, single_drone_mask
                )

                # Save debug images (optional)
                save_debug_images(single_img, pts_sat, colors_sat, pts_gd, colors_gd, meter_per_pixel)

                # Recover intrinsics from local points
                points = single_results["local_points"][:, 0]
                masks = torch.sigmoid(single_results["conf"][:, 0, :, :, 0]) > Config.CONF_THRESHOLD
                original_height, original_width = points.shape[-3:-1]
                aspect_ratio = original_width / original_height

                # Recover focal length
                focal, shift = recover_focal_shift(points, masks)
                fx, fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
                intrinsics = intrinsics_from_focal_center(fx, fy, 0.5, 0.5)

                # Get project grd2sat img
                g2s_direct_proj = project_point_clouds(pts_gd, colors_gd, intrinsics, Config.SATELLITE_WIDTH, Config.SATELLITE_WIDTH)
                g2s_img = TO_PIL_IMAGE(g2s_direct_proj[0].cpu())
                g2s_img.save('g2s_proj.png')

                s2s_direct_proj = project_point_clouds(pts_sat.reshape(1, -1, 3), colors_sat.reshape(1, -1, 3), intrinsics, Config.SATELLITE_WIDTH, Config.SATELLITE_WIDTH)
                s2s_img = TO_PIL_IMAGE(s2s_direct_proj[0].cpu())
                s2s_img.save('s2s_proj.png')

                # Get image dimensions and project ground camera
                img_width, img_height = imgs[i, 0].shape[-1], imgs[i, 0].shape[-2]
                reference_cam = single_results['camera_poses'][:, 0]
                camera_poses = torch.einsum('bij, bnjk -> bnik', se3_inverse(reference_cam), single_results['camera_poses'])

                projection_info = project_ground_camera_to_satellite(
                    camera_poses, intrinsics, img_width, img_height
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
                        center_x=center_x,
                        center_y=center_y,
                        ground_angle_pred=projection_info['ground_angle_degrees'],
                        ground_angle_gt=grd_gt_angle_degrees,
                        sample_idx=sample_idx,
                        save_dir=f"coordinate_visualizations/sat_pred_{len(results_dict['distances'])}.png"
                    )

                # Store results
                results_dict['distances'].append(projection_info['distance'])
                results_dict['sample_indices'].append(sample_idx)
                results_dict['projections'].append(projection_info)
                results_dict['image_sizes'].append((img_width, img_height))

    return results_dict


# =============================================================================
# STATISTICS AND OUTPUT FUNCTIONS
# =============================================================================

def calculate_statistics(distances: List[float]) -> Dict[str, float]:
    """
    Calculate statistical measures for distances.

    Args:
        distances: List of distance values

    Returns:
        Dictionary containing statistics
    """
    return {
        'mean': np.mean(distances),
        'std': np.std(distances),
        'min': np.min(distances),
        'max': np.max(distances),
        'median': np.median(distances)
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
        f.write(f"  Mean: {stats['mean']:.2f} ± {stats['std']:.2f}\n")
        f.write(f"  Median: {stats['median']:.2f}\n")
        f.write(f"  Min: {stats['min']:.2f}\n")
        f.write(f"  Max: {stats['max']:.2f}\n\n")
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
    print(f"  Mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
    print(f"  Median: {stats['median']:.2f}")
    print(f"  Min: {stats['min']:.2f}")
    print(f"  Max: {stats['max']:.2f}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("Initializing ground camera to satellite distance analysis...")

    # Setup model and data
    model = setup_model()
    test_dataloader = create_test_dataloader(batch_size=Config.BATCH_SIZE, shuffle=False)

    print(f"Dataset contains {len(test_dataloader.dataset)} samples")
    print(f"Processing {len(test_dataloader)} batches")

    # Collect all results
    all_distances = []
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
        for i, (dist, idx, proj) in enumerate(zip(batch_results['distances'],
                                                  batch_results['sample_indices'],
                                                  batch_results['projections'])):
            total_samples += 1
            all_sample_indices.append(idx)

            if proj['valid']:
                all_distances.append(dist)
                valid_projections_count += 1

        # Memory cleanup
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("Processing complete!")

    # Calculate and display statistics
    if all_distances:
        stats = calculate_statistics(all_distances)
        print_statistics(stats, total_samples, valid_projections_count)
        save_results_to_file(all_distances, all_sample_indices, total_samples, valid_projections_count, stats)
        print(f"\n💾 Results saved to {Config.OUTPUT_FILE}")
    else:
        print("\n❌ No valid projections found!")
        print("This might indicate an issue with the camera projection process.")


if __name__ == "__main__":
    main()