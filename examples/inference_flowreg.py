"""Example: FlowReg inference on ACDC using pretrained HuggingFace checkpoint."""

import argparse
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
from huggingface_hub import snapshot_download

from data.acdc import ACDCDataset
from flowreg.networks import FlowReg
from flowreg.test import Sampler
from utils.device import get_device
from utils.metrics import compute_dice, warp_segmentation


def main():
    parser = argparse.ArgumentParser(description="FlowReg inference example")
    parser.add_argument(
        "--model_id", type=str, default="acdc_ed2es",
        help="Model ID (default: acdc_ed2es)",
    )
    parser.add_argument(
        "--sample_index", type=int, default=1,
        help="Test sample index to run inference on",
    )
    parser.add_argument("--num_steps", type=int, default=10, help="ODE integration steps (default: 10)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--eta", type=float, default=5.0, help="SDE noise strength (0=ODE, >0=SDE, default: 5.0)")
    parser.add_argument("--no_recycle", action="store_true", help="Disable initial guess (ddf=pred at first step)")
    parser.add_argument("--no_heun", action="store_true", help="Skip Heun corrector (pure Euler)")
    parser.add_argument("--guidance_weight", type=float, default=0.05, help="NCC guidance weight (0=no guidance, default: 0.05)")

    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Download model from HuggingFace
    subfolder = f"flowreg/{args.model_id}"
    path = snapshot_download(
        "mathpluscode/FlowReg", allow_patterns=f"{subfolder}/*"
    )
    model = FlowReg.from_pretrained(f"{path}/{subfolder}")
    model.to(device).eval()
    print(f"Loaded FlowReg from {subfolder}")

    # Download dataset from HuggingFace
    data_path = snapshot_download(
        "mathpluscode/ACDC-FlowReg", repo_type="dataset"
    )
    test_set = ACDCDataset(data_path, split="test")
    print(f"Test samples: {len(test_set)}")

    # Build sampler
    sampler = Sampler(
        model, args.num_steps,
        eta=args.eta,
        recycle=not args.no_recycle,
        no_heun=args.no_heun,
        guidance_weight=args.guidance_weight,
        guidance_ncc_win=7,
        guidance_reg=True,
    )

    # Run inference on one sample
    data = test_set[args.sample_index]
    direction = "es2ed" if "es2ed" in args.model_id else "ed2es"
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

    torch.manual_seed(args.seed)
    intermediates = sampler(fixed, moving)

    pid = test_set.pids[args.sample_index]
    print(f"\n  {pid} ({direction}) Dice per step:")
    print(f"    {'Step':>4s}  {'RV':>6s}  {'Myo':>6s}  {'LV':>6s}  {'Mean':>6s}")
    for step, flow in enumerate(intermediates, 1):
        warped_seg = warp_segmentation(moving_seg, flow, device)
        dice = compute_dice(warped_seg, fixed_seg)
        print(f"    {step:4d}  {dice['dice_RV'].item():6.4f}  {dice['dice_Myo'].item():6.4f}  {dice['dice_LV'].item():6.4f}  {dice['dice_mean'].item():6.4f}")


if __name__ == "__main__":
    main()
