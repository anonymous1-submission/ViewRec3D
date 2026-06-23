import torch

from src.vggt.utils.camera import build_intrinsic, build_extrinsic
from src.vggt.utils.geometry import get_world_rays


def build_collator(
    patch_size: int,
    use_rays: bool = True,
):
    def collator(features):
        images = [f["image"] for f in features]
        pixel_values = torch.stack(images)
        masks = [f["mask"] for f in features]
        masks = torch.stack(masks)

        change_orientation = torch.tensor(
            [[f["change_orientation"]] for f in features],
            dtype=torch.float32
        )

        pose = torch.stack([
            torch.tensor(f["pose"], dtype=torch.float32)
            for f in features
        ])

        intrinsic = torch.stack([
            torch.tensor(f["intrinsic"], dtype=torch.float32)
            for f in features
        ])


        batch = {
            "pixel_values": pixel_values,
            "masks": masks,
            "change_orientation": change_orientation,
            "pose": pose,
            "rays": None,
        }

        if use_rays:
            trans = pose[:, :3]
            quat = pose[:, 3:7]

            B, _, H, W = pixel_values.shape
            extrinsic = build_extrinsic(trans, quat)

            coords = make_patch_grid(H=H, W=W, patch_size=patch_size, batch_size=B, device=intrinsic.device)
            intrinsic = intrinsic[:, None, None, :, :]
            extrinsic = extrinsic[:, None, None, :, :]
            rays_o, rays_d = get_world_rays(
                coords,
                extrinsic,
                intrinsic,
            )
            rays = torch.cat([rays_o, rays_d], dim=-1)  # [H, W, 6]

            batch["rays"] = rays

        return batch
    return collator


def make_patch_grid(
    H: int,
    W: int,
    patch_size: int,
    batch_size: int,
    device=None,
) -> torch.Tensor:
    H_patch = H // patch_size
    W_patch = W // patch_size

    ys, xs = torch.meshgrid(
        torch.arange(H_patch, dtype=torch.float32, device=device),
        torch.arange(W_patch, dtype=torch.float32, device=device),
        indexing="ij",
    )

    coords = torch.stack([
        xs * patch_size + (patch_size - 1) / 2,
        ys * patch_size + (patch_size - 1) / 2
    ], dim=-1)  # [H_patch, W_patch, 2]

    coords = coords.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [B, H_patch, W_patch, 2]
    return coords
