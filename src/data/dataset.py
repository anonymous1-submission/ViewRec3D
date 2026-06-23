import random
from pathlib import Path

import torch
from PIL import Image

from src.data.parser import load_annotation
from torch.utils.data import Dataset


def load_image(
    image_path: Path,
    target_size: int,
    patch_size: int,
    image_processor,
):
    with Image.open(image_path) as image:
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        image = image.convert("RGB")
        width, height = image.size

        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / patch_size) * patch_size
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / patch_size) * patch_size

        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        image = image_processor(image)
        h_padding = target_size - image.shape[1]
        w_padding = target_size - image.shape[2]
        num_patches = target_size // patch_size
        patch_mask = torch.zeros((num_patches, num_patches), dtype=torch.bool)

        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            image = torch.nn.functional.pad(
                image, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0
            )
            valid_top = (pad_top + patch_size - 1) // patch_size
            valid_left = (pad_left + patch_size - 1) // patch_size
            valid_bottom = (target_size - pad_bottom) // patch_size
            valid_right = (target_size - pad_right) // patch_size
            patch_mask[valid_top:valid_bottom, valid_left:valid_right] = True
        else:
            patch_mask[:, :] = True
        return image, patch_mask

def build_dataset_item(
    sample: dict,
    target_size: int,
    patch_size: int,
    image_processor,
):
    """
    {
        "original": ,
        "generated": ,
    }
    """
    generated_image, generated_image_mask = load_image(Path(sample["generated_path"]), target_size, patch_size, image_processor)
    change_orientation, pose, intrinsic = load_annotation(Path(sample["json_path"]))
    item = {
        "id": Path(sample["generated_path"]).parent.name,
        "image": generated_image,
        "mask": generated_image_mask,
        "change_orientation": change_orientation,
        "pose": pose,
        "intrinsic": intrinsic
    }

    return item

class AIPhotographerDataset(Dataset):
    """
     {
        "image": ,
        "change_orientation": ,
        "pose": ,
     }
    """
    def __init__(
        self,
        samples,
        target_size: int,
        patch_size: int,
        image_processor,
        seed: int,
    ):
        self.samples = samples
        self.target_size = target_size
        self.patch_size = patch_size
        self.image_processor = image_processor
        self.seed = seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        item = build_dataset_item(sample, self.target_size, self.patch_size, self.image_processor)

        return item
