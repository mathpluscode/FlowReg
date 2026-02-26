"""Preprocess MnMs2 dataset into nnUNet v2 format.

Preprocessing steps (consistent with ACDC / DiffuseMorph):
1. Resample to (1.5, 1.5, 3.15) voxel spacing using BSpline
2. Center crop to (128, 128) in x-y plane

Output: nnUNet v2 directory structure with ED/ES as separate cases.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from data.util import SPACING

TARGET_SPACING = SPACING[::-1]  # (x, y, z) for SimpleITK
TARGET_XY_SIZE = (128, 128)

# MnMs2 splits: 1-160 train, 161-200 val, 201-360 test
TRAIN_RANGE = range(1, 161)
VAL_RANGE = range(161, 201)
TEST_RANGE = range(201, 361)


def resample_image(image: sitk.Image, is_label: bool) -> sitk.Image:
    """Resample 3D image to target spacing."""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = tuple(
        int(np.round(orig_sz * orig_sp / tgt_sp))
        for orig_sz, orig_sp, tgt_sp in zip(original_size, original_spacing, TARGET_SPACING)
    )

    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline

    return sitk.Resample(
        image,
        size=new_size,
        transform=sitk.Transform(),
        interpolator=interpolator,
        outputOrigin=image.GetOrigin(),
        outputSpacing=TARGET_SPACING,
        outputDirection=image.GetDirection(),
        defaultPixelValue=0,
        outputPixelType=image.GetPixelID(),
    )


def center_crop_xy(image: sitk.Image) -> sitk.Image:
    """Center crop in x-y plane to TARGET_XY_SIZE."""
    current_size = image.GetSize()  # (x, y, z)

    crop_lower = []
    crop_upper = []

    for i in range(3):
        if i < 2:  # x, y axes
            target_len = TARGET_XY_SIZE[i]
            current_len = current_size[i]
            total_crop = current_len - target_len
            crop_lower.append(max(0, total_crop // 2))
            crop_upper.append(max(0, total_crop - total_crop // 2))
        else:  # z axis - no crop
            crop_lower.append(0)
            crop_upper.append(0)

    return sitk.Crop(image, tuple(crop_lower), tuple(crop_upper))


def preprocess_patient(patient_dir: Path, pid_str: str, images_dir: Path, labels_dir: Path):
    """Preprocess a single patient and write nnUNet cases."""
    ed_img = sitk.ReadImage(str(patient_dir / f"{pid_str}_SA_ED.nii.gz"))
    ed_lbl = sitk.ReadImage(str(patient_dir / f"{pid_str}_SA_ED_gt.nii.gz"), sitk.sitkUInt8)
    es_img = sitk.ReadImage(str(patient_dir / f"{pid_str}_SA_ES.nii.gz"))
    es_lbl = sitk.ReadImage(str(patient_dir / f"{pid_str}_SA_ES_gt.nii.gz"), sitk.sitkUInt8)

    # Resample + center crop
    ed_img = center_crop_xy(resample_image(ed_img, is_label=False))
    ed_lbl = center_crop_xy(resample_image(ed_lbl, is_label=True))
    es_img = center_crop_xy(resample_image(es_img, is_label=False))
    es_lbl = center_crop_xy(resample_image(es_lbl, is_label=True))

    # Write nnUNet cases
    sitk.WriteImage(ed_img, str(images_dir / f"{pid_str}_ED_0000.nii.gz"), useCompression=True)
    sitk.WriteImage(ed_lbl, str(labels_dir / f"{pid_str}_ED.nii.gz"), useCompression=True)
    sitk.WriteImage(es_img, str(images_dir / f"{pid_str}_ES_0000.nii.gz"), useCompression=True)
    sitk.WriteImage(es_lbl, str(labels_dir / f"{pid_str}_ES.nii.gz"), useCompression=True)


def preprocess_split(data_dir: Path, subject_ids: range, images_dir: Path, labels_dir: Path) -> int:
    """Preprocess all patients in a split. Returns number of processed patients."""
    print(f"Processing {len(subject_ids)} patients...")

    count = 0
    for sid in tqdm(subject_ids):
        pid_str = f"{sid:03d}"
        patient_dir = data_dir / pid_str
        if not patient_dir.exists():
            print(f"Skipping {pid_str}: directory not found")
            continue
        try:
            preprocess_patient(patient_dir, pid_str, images_dir, labels_dir)
            count += 1
        except Exception as e:
            print(f"Error processing {pid_str}: {e}")

    return count


def write_dataset_json(out_dir: Path, num_training: int):
    """Write nnUNet v2 dataset.json."""
    dataset_json = {
        "channel_names": {"0": "cardiac_mri"},
        "labels": {"background": 0, "RV": 1, "Myo": 2, "LV": 3},
        "numTraining": num_training,
        "file_ending": ".nii.gz",
    }
    with open(out_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--out_dir", type=Path,
                        help="nnUNet v2 dataset dir (e.g. nnUNet_raw/Dataset002_MnMs2)")
    args = parser.parse_args()

    sitk.ProcessObject.SetGlobalWarningDisplay(False)

    print(f"Target spacing: {TARGET_SPACING} mm")
    print(f"Target x-y size: {TARGET_XY_SIZE}")

    # Create nnUNet directories
    images_tr = args.out_dir / "imagesTr"
    labels_tr = args.out_dir / "labelsTr"
    images_ts = args.out_dir / "imagesTs"
    labels_ts = args.out_dir / "labelsTs"
    for d in [images_tr, labels_tr, images_ts, labels_ts]:
        d.mkdir(parents=True, exist_ok=True)

    # Train + val → imagesTr (nnUNet does its own cross-validation)
    num_train = preprocess_split(args.data_dir, TRAIN_RANGE, images_tr, labels_tr)
    num_val = preprocess_split(args.data_dir, VAL_RANGE, images_tr, labels_tr)
    # Test → imagesTs + labelsTs
    preprocess_split(args.data_dir, TEST_RANGE, images_ts, labels_ts)

    num_training = (num_train + num_val) * 2  # ED + ES per patient
    write_dataset_json(args.out_dir, num_training=num_training)
    print(f"Done! {num_training} training cases written to {args.out_dir}")


if __name__ == "__main__":
    main()
