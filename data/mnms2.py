"""MnMs2 dataset for cardiac image registration."""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
import SimpleITK as sitk

from data.util import VOLUME_SHAPE, normalize, center_crop_or_pad, augment

TRAIN_RANGE = range(1, 161)
VAL_RANGE = range(161, 201)


class MnMs2Dataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = Path(data_dir)
        self.training = split == 'train'

        if split == 'test':
            self.images_dir = self.data_dir / "imagesTs"
            self.labels_dir = self.data_dir / "labelsTs"
        else:
            self.images_dir = self.data_dir / "imagesTr"
            self.labels_dir = self.data_dir / "labelsTr"

        # Discover patient IDs from ED image filenames
        all_pids = sorted(
            p.name.removesuffix("_ED_0000.nii.gz")
            for p in self.images_dir.glob("*_ED_0000.nii.gz")
        )

        if split == 'train':
            self.pids = [p for p in all_pids if int(p) in TRAIN_RANGE]
        elif split == 'val':
            self.pids = [p for p in all_pids if int(p) in VAL_RANGE]
        else:
            self.pids = all_pids

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, index):
        pid = self.pids[index]

        ed_img = sitk.GetArrayFromImage(sitk.ReadImage(str(self.images_dir / f"{pid}_ED_0000.nii.gz"))).astype(np.float32)
        es_img = sitk.GetArrayFromImage(sitk.ReadImage(str(self.images_dir / f"{pid}_ES_0000.nii.gz"))).astype(np.float32)
        ed_lbl = sitk.GetArrayFromImage(sitk.ReadImage(str(self.labels_dir / f"{pid}_ED.nii.gz"))).astype(np.float32)
        es_lbl = sitk.GetArrayFromImage(sitk.ReadImage(str(self.labels_dir / f"{pid}_ES.nii.gz"))).astype(np.float32)

        ed_img = center_crop_or_pad(normalize(ed_img), VOLUME_SHAPE)
        es_img = center_crop_or_pad(normalize(es_img), VOLUME_SHAPE)
        ed_lbl = center_crop_or_pad(ed_lbl, VOLUME_SHAPE)
        es_lbl = center_crop_or_pad(es_lbl, VOLUME_SHAPE)

        ed_img, es_img, ed_lbl, es_lbl = augment([ed_img, es_img, ed_lbl, es_lbl], self.training)

        return {
            'M': torch.from_numpy(np.ascontiguousarray(ed_img)).unsqueeze(0) * 2 - 1,
            'F': torch.from_numpy(np.ascontiguousarray(es_img)).unsqueeze(0) * 2 - 1,
            'MS': torch.from_numpy(np.ascontiguousarray(ed_lbl)),
            'FS': torch.from_numpy(np.ascontiguousarray(es_lbl)),
            'Index': index
        }
