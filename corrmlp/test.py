"""Test script for CorrMLP."""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse
import csv
import time
from pathlib import Path
import numpy as np
import torch

from torch.utils.data import DataLoader
from data import DATASETS
from utils.device import get_device
from utils.metrics import compute_all_metrics, warp_segmentation

from . import networks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['acdc', 'mnms2'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--direction', type=str, default='ed2es', choices=['ed2es', 'es2ed'],
                        help='Registration direction: ed2es (Fixed=ES, Moving=ED) or es2ed (Fixed=ED, Moving=ES)')
    parser.add_argument('--num_steps', type=int, default=1,
                        help='Iterative refinement steps: register, warp, re-register (1=single pass)')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Dataset
    Dataset = DATASETS[args.dataset]
    test_set = Dataset(args.data_dir, 'test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    print(f'Test samples: {len(test_set)}')

    # Model
    model = networks.CorrMLP()
    print(f'Loading checkpoint: {args.checkpoint}')
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    num_steps = args.num_steps

    # Test loop
    step_rows = {s: [] for s in range(num_steps)}
    all_runtime = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # Convert from [-1, 1] to [0, 1] range for CorrMLP
            if args.direction == 'ed2es':
                fixed = (data['F'].to(device) + 1) / 2   # ES
                moving = (data['M'].to(device) + 1) / 2  # ED
                fixed_seg = data['FS'].to(device)
                moving_seg = data['MS'].to(device)
            else:
                fixed = (data['M'].to(device) + 1) / 2   # ED
                moving = (data['F'].to(device) + 1) / 2  # ES
                fixed_seg = data['MS'].to(device)
                moving_seg = data['FS'].to(device)

            cur_moving = moving
            cur_moving_seg = moving_seg

            t_start = time.time()
            for si in range(num_steps):
                warped, flow = model(fixed, cur_moving)
                warped_seg = warp_segmentation(cur_moving_seg, flow, device)

                metrics = compute_all_metrics(flow, warped, fixed, warped_seg, fixed_seg, moving_seg, args.direction)
                row = {'pid': test_set.pids[i], 'step': si + 1}
                for k, v in metrics.items():
                    row[k] = v.item()
                step_rows[si].append(row)

                # Warp for next iteration
                cur_moving = warped
                cur_moving_seg = warped_seg

            runtime = time.time() - t_start
            all_runtime.append(runtime)
            # Store runtime on final step row
            for si in range(num_steps):
                step_rows[si][-1]['runtime'] = runtime

    # Summary — dice per step
    dice_keys = ['dice_RV', 'dice_Myo', 'dice_LV', 'dice_mean']
    print('\n' + '=' * 60)
    print(f'Dice per step ({len(test_set)} samples, {args.direction}):')
    header = f'  {"step":>4s}' + ''.join(f'  {k:>10s}' for k in dice_keys)
    print(header)
    for s in range(num_steps):
        vals = {k: np.mean([r[k] for r in step_rows[s]]) for k in dice_keys}
        line = f'  {s+1:4d}' + ''.join(f'  {vals[k]:10.4f}' for k in dice_keys)
        print(line)

    # Final step summary (all metrics)
    final_rows = step_rows[num_steps - 1]
    metric_keys = [k for k in final_rows[0] if k not in ('pid', 'step', 'runtime')]
    print(f'\nFinal step summary:')
    for k in metric_keys:
        vals = [r[k] for r in final_rows]
        print(f'  {k:20s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}')
    print(f'  {"runtime":20s}: {np.mean(all_runtime[1:]):.3f} +/- {np.std(all_runtime[1:]):.3f}s (excl. first)')

    # Save CSVs next to checkpoint
    ckpt_dir = Path(args.checkpoint).parent
    prefix = f'{args.dataset}_{args.direction}'
    if num_steps > 1:
        prefix += f'_steps{num_steps}'

    # Per-sample CSV (all steps)
    all_rows = [r for s in range(num_steps) for r in step_rows[s]]
    out = ckpt_dir / f'{prefix}.csv'
    fieldnames = list(all_rows[0].keys())
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'Saved per-sample metrics to {out}')

    # Summary CSV
    summary_path = ckpt_dir / f'{prefix}_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'metric', 'mean', 'std'])
        for s in range(num_steps):
            for k in metric_keys:
                vals = [r[k] for r in step_rows[s]]
                writer.writerow([s + 1, k, np.mean(vals), np.std(vals)])
    print(f'Saved summary to {summary_path}')


if __name__ == "__main__":
    main()
