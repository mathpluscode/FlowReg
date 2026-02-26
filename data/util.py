"""Shared utilities for cardiac image registration datasets."""

import random
import numpy as np

VOLUME_SHAPE = (32, 128, 128)  # (z, y, x)
SPACING = (3.15, 1.5, 1.5)  # (z, y, x) in mm


def normalize(img: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1]."""
    img = img - img.min()
    img = img / (img.std() + 1e-8)
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img


def center_crop_or_pad(img: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Center crop or pad image to target shape."""
    result = np.zeros(target_shape, dtype=img.dtype)
    slices_src, slices_dst = [], []
    for src_size, tgt_size in zip(img.shape, target_shape):
        if src_size >= tgt_size:
            start = (src_size - tgt_size) // 2
            slices_src.append(slice(start, start + tgt_size))
            slices_dst.append(slice(None))
        else:
            start = (tgt_size - src_size) // 2
            slices_src.append(slice(None))
            slices_dst.append(slice(start, start + src_size))
    result[tuple(slices_dst)] = img[tuple(slices_src)]
    return result


def augment(imgs, training):
    """Apply consistent augmentation to list of images."""
    if not training:
        return imgs

    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot90 = random.random() < 0.5

    def _aug(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = np.rot90(img, axes=(1, 2))
        return img

    return [_aug(img) for img in imgs]
