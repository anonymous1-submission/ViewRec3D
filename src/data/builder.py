from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from torchvision import transforms

from src.data.collator import build_collator
from src.data.dataset import AIPhotographerDataset
from src.data.indexing import build_samples


@dataclass
class DataModule:
    train_dataset: Optional[AIPhotographerDataset]
    val_dataset: Optional[AIPhotographerDataset]
    data_collator: callable

def build_transform(mean, std):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std,
        ),
    ])

def build_train_dataset(
    train_dataset_dir: Path,
    seed: int,
    target_size: int,
    patch_size: int,
    image_processor,
    data_ratio: float,
    data_angle: int
):
    print("=============train dataset===============")
    train_samples = build_samples(train_dataset_dir, data_ratio, data_angle)

    train_dataset = AIPhotographerDataset(
        samples=train_samples,
        seed=seed,
        target_size=target_size,
        patch_size=patch_size,
        image_processor=image_processor,
    )
    print("Create train dataset successfully!")
    print(f"samples: {len(train_dataset)}")

    return train_dataset

def build_val_dataset(
    val_dataset_dir: Path,
    seed: int,
    target_size: int,
    patch_size: int,
    image_processor,
    data_ratio: float,
    data_angle: int
):
    print("=============validation dataset===============")
    val_samples = build_samples(val_dataset_dir, data_ratio, data_angle)

    val_dataset = AIPhotographerDataset(
        samples=val_samples,
        seed=seed,
        target_size=target_size,
        patch_size=patch_size,
        image_processor=image_processor,
    )
    print("Create validation dataset successfully!")
    print(f"samples: {len(val_dataset)}")

    return val_dataset

def build_data_module(
    train_dataset_dir: Path | None,
    val_dataset_dir: Path | None,
    seed: int,
    target_size: int,
    patch_size: int,
    use_rays: bool,
    mean,
    std,
    data_ratio: float,
    data_angle: int,
):
    image_processor = build_transform(mean, std)

    # 根据是否提供train_dataset_dir来决定是否加载训练数据集
    if train_dataset_dir is not None:
        train_dataset = build_train_dataset(
            train_dataset_dir=train_dataset_dir,
            seed=seed,
            target_size=target_size,
            patch_size=patch_size,
            image_processor=image_processor,
            data_ratio=data_ratio,
            data_angle=data_angle,
        )
    else:
        train_dataset = None

    if val_dataset_dir is not None:
        val_dataset = build_val_dataset(
            val_dataset_dir=val_dataset_dir,
            seed=seed,
            target_size=target_size,
            patch_size=patch_size,
            image_processor=image_processor,
            data_ratio=data_ratio,
            data_angle=data_angle,
        )
    else:
        val_dataset = None

    return DataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        data_collator=build_collator(patch_size, use_rays),
    )