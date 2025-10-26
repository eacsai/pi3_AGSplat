import torch
from pi3.models.pi3 import Pi3
from dataset_aer_grd_drone import load_test_triplet, create_test_dataloader, TestTripletDataset
from torchvision import transforms
from PIL import Image, ImageDraw
import torch.nn.functional as F
import numpy as np
from functools import partial
from pi3.utils.geometry import intrinsics_from_focal_center, se3_inverse, homogenize_points, depth_edge

from typing import *

to_pil_image = transforms.ToPILImage()
point_color = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'orange', 'magenta', 'lime', 'pink']

def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height
    
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv

def solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift and focal"
    from scipy.optimize import least_squares
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[: , None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        err = (f * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    optim_shift = solution['x'].squeeze().astype(np.float32)

    xy_proj = xy / (z + optim_shift)[: , None]
    optim_focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()

    return optim_shift, optim_focal

def solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray, focal: float):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift"
    from scipy.optimize import least_squares
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[: , None]
        err = (focal * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    optim_shift = solution['x'].squeeze().astype(np.float32)

    return optim_shift

def recover_focal_shift(points: torch.Tensor, mask: torch.Tensor = None, focal: torch.Tensor = None, downsample_size: Tuple[int, int] = (64, 64)):
    """
    Recover the depth map and FoV from a point map with unknown z shift and focal.

    Note that it assumes:
    - the optical center is at the center of the map
    - the map is undistorted
    - the map is isometric in the x and y directions

    ### Parameters:
    - `points: torch.Tensor` of shape (..., H, W, 3)
    - `downsample_size: Tuple[int, int]` in (height, width), the size of the downsampled map. Downsampling produces approximate solution and is efficient for large maps.

    ### Returns:
    - `focal`: torch.Tensor of shape (...) the estimated focal length, relative to the half diagonal of the map
    - `shift`: torch.Tensor of shape (...) Z-axis shift to translate the point map to camera space
    """
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]
    diagonal = (height ** 2 + width ** 2) ** 0.5

    points = points.reshape(-1, *shape[-3:])
    mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    focal = focal.reshape(-1) if focal is not None else None
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)  # (H, W, 2)

    points_lr = F.interpolate(points.permute(0, 3, 1, 2), downsample_size, mode='nearest').permute(0, 2, 3, 1)
    uv_lr = F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode='nearest').squeeze(0).permute(1, 2, 0)
    mask_lr = None if mask is None else F.interpolate(mask.to(torch.float32).unsqueeze(1), downsample_size, mode='nearest').squeeze(1) > 0
    
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

# --- Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
# or download checkpoints from `https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors`

# --- Load Data ---
# Method 1: Using original function (legacy)
# print("Loading test image triplet...")
# image_data = load_test_triplet(index=10)  # Load first test sample

# Method 2: Using new DataLoader (recommended for batch processing)
print("Creating test dataset and dataloader...")
dataset = TestTripletDataset()  # Load dataset info
image_data = dataset[10]  # Load specific sample

# Method 3: Using DataLoader for batch processing
# dataloader = create_test_dataloader(batch_size=1, shuffle=False)
# batch = next(iter(dataloader))
# image_data = {
#     'satellite': batch['satellite'].squeeze(0),
#     'ground': batch['ground'].squeeze(0),
#     'drone': batch['drone'].squeeze(0),
#     'paths': {
#         'satellite': batch['paths']['satellite'][0],
#         'ground': batch['paths']['ground'][0],
#         'drone': batch['paths']['drone'][0]
#     }
# }

print(f"Dataset contains {len(dataset)} samples")
print(f"Loaded sample {image_data['index']}")

# Stack the three images into a sequence tensor
# imgs shape: (3, 3, H, W) where 3 is the number of views (ground, drone, satellite)
imgs = torch.stack([image_data['satellite'], image_data['ground'], image_data['drone']], dim=0)
imgs = imgs.to(device)

print(f"Loaded images with shape: {imgs.shape}")

test_img = to_pil_image(image_data['ground'])
test_img.save("test_ground_image.png")
test_img = to_pil_image(image_data['drone'])
test_img.save("test_drone_image.png")
test_img = to_pil_image(image_data['satellite'])
# 在卫星图中心添加红点
draw = ImageDraw.Draw(test_img)
width, height = test_img.size
center_x, center_y = width // 2, height // 2
radius = 5  # 红点半径
draw.ellipse([center_x - radius, center_y - radius,
              center_x + radius, center_y + radius],
             fill='red', outline='red')
test_img.save("test_satellite_image.png")

# --- Inference ---
print("Running model inference...")
# Use mixed precision for better performance on compatible GPUs
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

with torch.no_grad():
    with torch.amp.autocast('cuda', dtype=dtype):
        # Add a batch dimension -> (1, N, 3, H, W)
        results = model(imgs[None])
        # Reconstruct point cloud in reference camera coordinate system
        global_points = torch.einsum('bij, bnhwj -> bnhwi', se3_inverse(results['camera_poses'][:, 0]), homogenize_points(results['points']))[..., :3]
        camera_poses = torch.einsum('bij, bnjk -> bnik', se3_inverse(results['camera_poses'][:, 0]), results['camera_poses']) # (B, N, 4, 4)

        # Recover intrinsics from local points
        points = results["local_points"][:, 0] # (B, H, W, 3)
        # Count points with negative z coordinates
        neg_z_count = torch.sum(points[0, :, :, 2] < 0).item()
        total_points = points[0, :, :, 2].numel()
        print(f"Points with z < 0: {neg_z_count} out of {total_points} total points ({100*neg_z_count/total_points:.2f}%)")

        masks = torch.sigmoid(results["conf"][:, 0, :, :, 0]) > 0.1
        original_height, original_width = points.shape[-3:-1]
        aspect_ratio = original_width / original_height
        # use recover_focal_shift function from MoGe
        focal, shift = recover_focal_shift(points, masks)
        fx, fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
        intrinsics = intrinsics_from_focal_center(fx, fy, 0.5, 0.5)

        # Project camera positions to first camera (satellite) view
        print("Projecting camera positions to satellite view...")
        batch_size = camera_poses.shape[0]
        num_cameras = camera_poses.shape[1]

        # Convert first camera image to PIL for drawing
        first_cam_img = to_pil_image(imgs[0].cpu())  # imgs[0] is the satellite view
        draw = ImageDraw.Draw(first_cam_img)
        img_width, img_height = first_cam_img.size

        # Camera positions in world coordinates (translation part of poses)
        # camera_poses shape: (B, N, 4, 4) where N is number of cameras
        cam_positions = camera_poses[0, :, :3, 3]  # (N, 3) - take first batch

        # Get intrinsics matrix for the first camera
        K = intrinsics if intrinsics.dim() == 2 else intrinsics[0]  # Ensure K is (3, 3)

        # Project each camera position to the first camera view
        for i in range(num_cameras):
            if i == 0:
                # Skip the first camera (reference camera) as it's at origin
                continue

            # Get camera position in first camera coordinate system
            cam_pos = cam_positions[i]  # (3,)

            # Only project if point is in front of camera (z > 0)
            if cam_pos[2] > 0:

                # Project: K @ [x, y, z, 1]^T -> [u*z, v*z, z]^T
                proj = K @ cam_pos  # (3,)

                # Convert to normalized image coordinates
                x_norm = proj[0] / (proj[2] + 1e-6)  # u coordinate
                y_norm = proj[1] / (proj[2] + 1e-6)  # v coordinate

                # Convert from normalized coordinates to pixel coordinates
                pixel_x = int(x_norm * img_width)
                pixel_y = int(y_norm * img_height)

                print(f"Camera {i}: 3D pos=({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}), "
                      f"pixel=({pixel_x}, {pixel_y})")

                # Draw red circle if within image bounds
                if 0 <= pixel_x < img_width and 0 <= pixel_y < img_height:
                    radius = 8
                    draw.ellipse([pixel_x - radius, pixel_y - radius,
                                pixel_x + radius, pixel_y + radius],
                               fill=point_color[i % len(point_color)], outline='darkred', width=2)

                    # Add camera number label
                    draw.text((pixel_x + radius + 5, pixel_y - 10), f"Cam {i}", fill=point_color[i % len(point_color)])
                else:
                    print(f"Camera {i} projection outside image bounds: ({pixel_x}, {pixel_y})")
        
        draw.ellipse([img_width//2 - radius, img_width//2 - radius,
                    img_width//2 + radius, img_width//2 + radius],
                    fill=point_color[0], outline='darkred', width=2)

        # Add camera number label
        draw.text((img_width//2 + radius + 5, img_width//2 - 10), f"GT", fill=point_color[0])

        # Save the annotated image
        first_cam_img.save("satellite_with_camera_projections.png")
        print(f"Saved camera projections to satellite_with_camera_projections.png")

print("Reconstruction complete!")
# Access outputs: results['points'], results['camera_poses'] and results['local_points'].


# 'input_images_20251018_201814_909289/images'