import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def load_test_data(
    data_path: Optional[str] = None,
    device="cuda",
    scene_crop: Tuple[float, float, float, float, float, float] = (-2, -2, -2, 2, 2, 2),
    scene_grid: int = 1,
):
    """Load the test data."""
    assert scene_grid % 2 == 1, "scene_grid must be odd"

    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz")
    data = np.load(data_path)
    height, width = data["height"].item(), data["width"].item()
    viewmats = torch.from_numpy(data["viewmats"]).float().to(device)
    Ks = torch.from_numpy(data["Ks"]).float().to(device)
    means = torch.from_numpy(data["means3d"]).float().to(device)
    colors = torch.from_numpy(data["colors"] / 255.0).float().to(device)
    C = len(viewmats)

    # crop
    aabb = torch.tensor(scene_crop, device=device)
    edges = aabb[3:] - aabb[:3]
    sel = ((means >= aabb[:3]) & (means <= aabb[3:])).all(dim=-1)
    sel = torch.where(sel)[0]
    means, colors = means[sel], colors[sel]

    # repeat the scene into a grid (to mimic a large-scale setting)
    repeats = scene_grid
    gridx, gridy = torch.meshgrid(
        [
            torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
            torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
        ],
        indexing="ij",
    )
    grid = torch.stack([gridx, gridy, torch.zeros_like(gridx)], dim=-1).reshape(-1, 3)
    means = means[None, :, :] + grid[:, None, :] * edges[None, None, :]
    means = means.reshape(-1, 3)
    colors = colors.repeat(repeats**2, 1)

    # create gaussian attributes
    N = len(means)
    scales = torch.rand((N, 3), device=device) * 0.02
    quats = F.normalize(torch.randn((N, 4), device=device), dim=-1)
    opacities = torch.rand((N,), device=device)

    return means, quats, scales, opacities, colors, viewmats, Ks, width, height

import os
from PIL import Image
def downsample_image(image_path, factor, output_folder):
    with Image.open(image_path) as img:
        new_width = img.width // factor
        new_height = img.height // factor
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        base_name = os.path.basename(image_path)
        new_image_path = os.path.join(output_folder, base_name)
        img_resized.save(new_image_path)
        print(f"Processed {base_name}")


def process_images(input_folder, factor):
    output_folder = f"{input_folder}_{factor}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            file_path = os.path.join(input_folder, file_name)
            downsample_image(file_path, factor, output_folder)


if __name__ == "__main__":
    input_folder = rf"{os.path.dirname(__file__)}/../examples/datasets/lego/images_backup"
    factor = 2
    process_images(input_folder, factor)