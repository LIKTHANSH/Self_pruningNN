"""
data.py
=======
CIFAR-10 Data Loading and Preprocessing

This module handles downloading, augmenting, and loading the CIFAR-10 dataset.
Standard preprocessing includes normalization with per-channel mean and
standard deviation, and training augmentations (random horizontal flip,
random crop with padding).

CIFAR-10 Statistics:
    - 50,000 training images, 10,000 test images
    - 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    - Image size: 32x32 pixels, 3 color channels (RGB)

Author: Likthansh Anisetti
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Tuple

# ---------------------------------------------------------------
# CIFAR-10 per-channel normalization statistics
# Computed from the training set to center and scale the data.
# ---------------------------------------------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Class names for CIFAR-10
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def get_train_transforms() -> transforms.Compose:
    """
    Returns the training data augmentation pipeline.

    Augmentations:
        1. RandomCrop(32, padding=4): Introduces spatial translation invariance.
        2. RandomHorizontalFlip: 50% chance to mirror horizontally.
        3. ToTensor: Converts PIL image to tensor and scales [0, 255] → [0, 1].
        4. Normalize: Centers and scales using CIFAR-10 statistics.

    Returns:
        transforms.Compose: Training transform pipeline.
    """
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_test_transforms() -> transforms.Compose:
    """
    Returns the test data preprocessing pipeline (no augmentation).

    Returns:
        transforms.Compose: Test transform pipeline.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def load_cifar10(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Download and prepare CIFAR-10 data loaders.

    Args:
        data_dir (str): Directory to store/load dataset. Default: "./data".
        batch_size (int): Batch size for both train and test loaders. Default: 128.
        num_workers (int): Number of worker processes for data loading. Default: 2.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    # Training dataset with augmentation
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=get_train_transforms(),
    )

    # Test dataset without augmentation
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=get_test_transforms(),
    )

    # Data loaders
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )

    return train_loader, test_loader
