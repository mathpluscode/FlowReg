"""Test script for FSDiffReg."""

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
from fsdiffreg import model as Model
from utils import logger as Logger
from utils.device import get_device
from utils.metrics import compute_all_metrics, warp_segmentation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['acdc', 'mnms2'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='fsdiffreg/train.yaml')
    parser.add_argument('--direction', type=str, default='ed2es', choices=['ed2es', 'es2ed'],
                        help='Registration direction: ed2es (Fixed=ES, Moving=ED) or es2ed (Fixed=ED, Moving=ES)')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Dataset
    Dataset = DATASETS[args.dataset]
    test_set = Dataset(args.data_dir, 'test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    print(f'Test samples: {len(test_set)}')

    # Model
    class ConfigArgs:
        config = args.config
    opt = Logger.dict_to_nonedict(Logger.parse(ConfigArgs()))
    opt['path']['resume_state'] = args.checkpoint
    opt['phase'] = 'test'
    diffusion = Model.create_model(opt)
    diffusion.netG.eval()
    print(f'Loaded checkpoint: {args.checkpoint}')

    # Test loop
    rows = []
    all_runtime = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if args.direction == 'es2ed':
                data = {'M': data['F'], 'F': data['M'], 'MS': data['FS'], 'FS': data['MS'], 'Index': data['Index']}

            fixed_img = (data['F'].to(device) + 1) / 2
            moving_seg = data['MS'].to(device)
            fixed_seg = data['FS'].to(device)

            t_start = time.time()
            diffusion.feed_data(data)
            diffusion.test_registration()
            runtime = time.time() - t_start

            reg = diffusion.get_current_registration()
            flow = reg['flow'].to(device)
            warped_img = (reg['out_M'].to(device) + 1) / 2

            warped_seg = warp_segmentation(moving_seg, flow, device)
            metrics = compute_all_metrics(flow, warped_img, fixed_img, warped_seg, fixed_seg, moving_seg, args.direction)

            row = {'pid': test_set.pids[i], 'runtime': runtime}
            for k, v in metrics.items():
                row[k] = v.item()
            rows.append(row)
            all_runtime.append(runtime)

    # Summary
    metric_keys = [k for k in rows[0] if k not in ('pid', 'runtime')]
    print('\n' + '=' * 60)
    print(f'Summary ({len(rows)} samples, {args.direction}):')
    for k in metric_keys:
        vals = [r[k] for r in rows]
        print(f'  {k:20s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}')
    print(f'  {"runtime":20s}: {np.mean(all_runtime[1:]):.3f} +/- {np.std(all_runtime[1:]):.3f}s (excl. first)')

    # Save CSVs next to checkpoint
    ckpt_dir = Path(args.checkpoint).parent
    prefix = f'{args.dataset}_{args.direction}'

    # Per-sample CSV
    out = ckpt_dir / f'{prefix}.csv'
    fieldnames = list(rows[0].keys())
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {out}')

    # Summary CSV
    summary_path = ckpt_dir / f'{prefix}_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'mean', 'std'])
        for k in metric_keys:
            vals = [r[k] for r in rows]
            writer.writerow([k, np.mean(vals), np.std(vals)])
    print(f'Saved {summary_path}')


if __name__ == "__main__":
    main()
