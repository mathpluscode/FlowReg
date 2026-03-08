"""Example: CorrMLP inference on ACDC using pretrained HuggingFace checkpoint."""

import argparse
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
from huggingface_hub import snapshot_download

from corrmlp.networks import CorrMLP
from data.acdc import ACDCDataset
from utils.device import get_device
from utils.metrics import compute_dice, warp_segmentation


def main():
    parser = argparse.ArgumentParser(description="CorrMLP inference example")
    parser.add_argument(
        "--model_id",
        type=str,
        default="acdc_ed2es",
        help="Model ID (default: acdc_ed2es)",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=1,
        help="Test sample index to run inference on",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Download model from HuggingFace
    subfolder = f"corrmlp/{args.model_id}"
    path = snapshot_download(
        "mathpluscode/FlowReg", allow_patterns=f"{subfolder}/*"
    )
    model = CorrMLP.from_pretrained(f"{path}/{subfolder}")
    model.to(device).eval()
    print(f"Loaded CorrMLP from {subfolder}")

    # Download dataset from HuggingFace
    data_path = snapshot_download(
        "mathpluscode/ACDC-FlowReg", repo_type="dataset"
    )
    test_set = ACDCDataset(data_path, split="test")
    print(f"Test samples: {len(test_set)}")

    # Run inference on one sample
    data = test_set[args.sample_index]
    direction = "es2ed" if "es2ed" in args.model_id else "ed2es"
    with torch.no_grad():
        if direction == "ed2es":
            fixed = (data["F"].unsqueeze(0).to(device) + 1) / 2   # ES
            moving = (data["M"].unsqueeze(0).to(device) + 1) / 2  # ED
            fixed_seg = data["FS"].unsqueeze(0).to(device)
            moving_seg = data["MS"].unsqueeze(0).to(device)
        else:
            fixed = (data["M"].unsqueeze(0).to(device) + 1) / 2   # ED
            moving = (data["F"].unsqueeze(0).to(device) + 1) / 2  # ES
            fixed_seg = data["MS"].unsqueeze(0).to(device)
            moving_seg = data["FS"].unsqueeze(0).to(device)

        warped, flow = model(fixed, moving)

        warped_seg = warp_segmentation(moving_seg, flow, device)
        dice = compute_dice(warped_seg, fixed_seg)
        pid = test_set.pids[args.sample_index]
        print(f"\n  {pid} ({direction}):")
        print(f"    RV={dice['dice_RV'].item():.4f}  Myo={dice['dice_Myo'].item():.4f}  LV={dice['dice_LV'].item():.4f}  Mean={dice['dice_mean'].item():.4f}")


if __name__ == "__main__":
    main()
