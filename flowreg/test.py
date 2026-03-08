"""Test script for FlowReg with ODE/SDE sampling."""

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
from data.util import SPACING
from utils.device import get_device
from utils.losses import NCC, Grad
from utils.metrics import compute_all_metrics, compute_dice, warp_segmentation

from . import networks


class Sampler:

    def __init__(self, model, num_steps, eta=0.0, alpha=1.0, beta=1.0,
                 recycle=False, no_heun=False, guidance_weight=0.0,
                 guidance_ncc_win=7, guidance_reg=False):
        self.model = model
        self.num_steps = num_steps
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.recycle = recycle
        self.no_heun = no_heun
        self.guidance_weight = guidance_weight
        self.guidance_reg = guidance_reg
        if guidance_weight > 0:
            self.guidance_ncc = NCC(win=guidance_ncc_win)
            self.guidance_stn = networks.SpatialTransformer(mode='bilinear')
            if guidance_reg:
                self.guidance_grad = Grad('l2')

    def __call__(self, fixed, moving):
        device = fixed.device
        ddf, shape, noise_scale = self._init_ddf(fixed, device)
        gamma = min(self.eta / self.num_steps, 0.4142) if self.eta > 0 else 0.0
        ts = torch.linspace(0, 1, self.num_steps + 1, device=device)
        intermediates = []

        with torch.no_grad():
            for i in range(self.num_steps):
                t_cur = ts[i].item()
                t_next = ts[i + 1].item()
                last_step = (i == self.num_steps - 1)

                # SDE noise injection (skip first step — already at max noise)
                if self.eta > 0 and i > 0:
                    sigma = 1 - t_cur
                    sigma_hat = sigma * (1 + gamma)
                    noise = torch.randn(shape, device=device) * noise_scale * 5
                    noise = self.beta * (sigma_hat ** 2 - sigma ** 2) ** 0.5 * noise
                    ddf = ddf + noise
                    t_cur = max(1 - sigma_hat, 0.0)

                delta, pred = self._heun_step(fixed, moving, ddf, t_cur, t_next, last_step)

                if self.recycle and i == 0:
                    ddf = pred
                else:
                    ddf = ddf + self.alpha * delta
                intermediates.append(pred)

        return intermediates

    def _init_ddf(self, fixed, device):
        """Sample initial DDF from isotropic noise scaled by voxel spacing."""
        B = fixed.shape[0]
        shape = (B, 3, *fixed.shape[2:])
        min_sp = min(SPACING)
        noise_scale = torch.tensor([min_sp / s for s in SPACING], device=device).view(1, 3, 1, 1, 1)
        return torch.randn(shape, device=device) * noise_scale * 5, shape, noise_scale

    def _heun_step(self, fixed, moving, ddf, t_cur, t_next, last_step):
        """One Heun's 2nd-order step from t_cur to t_next."""
        B = fixed.shape[0]
        device = fixed.device
        h = t_next - t_cur

        t_batch = torch.full((B,), t_cur, device=device)
        pred = self.model(fixed, moving, ddf, t_batch)
        if self.guidance_weight > 0:
            pred = self._guide(pred, fixed, moving)
        t_exp = t_batch[:, None, None, None, None].clamp(max=1 - 1e-5)
        v1 = (pred - ddf) / (1 - t_exp)

        delta = v1 * h

        # Heun corrector
        if not last_step and not self.no_heun:
            ddf_next = ddf + delta
            t_next_batch = torch.full((B,), t_next, device=device)
            pred_next = self.model(fixed, moving, ddf_next, t_next_batch)
            if self.guidance_weight > 0:
                pred_next = self._guide(pred_next, fixed, moving)
            t_next_exp = t_next_batch[:, None, None, None, None].clamp(max=1 - 1e-5)
            v2 = (pred_next - ddf_next) / (1 - t_next_exp)
            delta = (v1 + v2) / 2 * h
            pred = pred_next

        return delta, pred

    def _guide(self, pred, fixed, moving):
        stn = self.guidance_stn.to(pred.device)
        ncc = self.guidance_ncc
        with torch.enable_grad():
            p = pred.detach().requires_grad_(True)
            warped = stn(moving, p)
            loss = ncc.loss(fixed, warped)
            if self.guidance_reg:
                loss = loss + self.guidance_grad.loss(None, p)
            grad = torch.autograd.grad(loss, p)[0]
        grad = grad * (pred.norm() / grad.norm())
        return pred.detach() - self.guidance_weight * grad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['acdc', 'mnms2'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--direction', type=str, default='ed2es', choices=['ed2es', 'es2ed'],
                        help='Registration direction: ed2es (Fixed=ES, Moving=ED) or es2ed (Fixed=ED, Moving=ES)')
    parser.add_argument('--num_seeds', type=int, default=1, help='Number of seeds (0, 1, ..., N-1)')
    parser.add_argument('--num_steps', type=int, default=10, help='ODE integration steps')
    parser.add_argument('--eta', type=float, default=0.0, help='SDE noise strength (0=ODE, >0=SDE)')
    parser.add_argument('--alpha', type=float, default=1.0, help='SDE step scale (1=full step)')
    parser.add_argument('--beta', type=float, default=1.0, help='SDE noise injection scale (1=exact, <1=less stochastic)')
    parser.add_argument('--recycle', action='store_true',
                        help='Use ddf=pred at the first step, normal ODE for remaining steps')
    parser.add_argument('--no_heun', action='store_true',
                        help='Skip Heun corrector (pure Euler, N NFEs instead of 2N-1)')
    parser.add_argument('--guidance_weight', type=float, default=0.0,
                        help='NCC guidance weight during sampling (0=no guidance)')
    parser.add_argument('--guidance_ncc_win', type=int, default=9,
                        help='NCC window size for guidance')
    parser.add_argument('--guidance_reg', action='store_true',
                        help='Add DDF L2 spatial gradient to guidance loss')
    args = parser.parse_args()

    seeds = list(range(args.num_seeds))

    device = get_device()
    print(f'Device: {device}')

    # Dataset
    Dataset = DATASETS[args.dataset]
    test_set = Dataset(args.data_dir, 'test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    print(f'Test samples: {len(test_set)}')

    # Model
    model = networks.FlowReg()
    print(f'Loading checkpoint: {args.checkpoint}')
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    stn = networks.SpatialTransformer(mode='bilinear').to(device)
    ncc_fn = NCC(win=9)
    grad_l2_fn = Grad('l2')

    # Sampler
    sampler = Sampler(model, args.num_steps,
                      eta=args.eta, alpha=args.alpha, beta=args.beta,
                      recycle=args.recycle, no_heun=args.no_heun,
                      guidance_weight=args.guidance_weight,
                      guidance_ncc_win=args.guidance_ncc_win,
                      guidance_reg=args.guidance_reg)

    # Build output prefix
    ckpt_dir = Path(args.checkpoint).parent
    prefix = f'{args.dataset}_{args.direction}_steps{args.num_steps}'
    if args.eta > 0:
        prefix += f'_eta{int(args.eta) if args.eta == int(args.eta) else args.eta}'
        if args.alpha != 1.0:
            prefix += f'_alpha{args.alpha}'
        if args.beta != 1.0:
            prefix += f'_beta{args.beta}'
    if args.num_seeds > 1:
        prefix += f'_nseeds{args.num_seeds}'
    if args.recycle:
        prefix += '_recycle'
    if args.no_heun:
        prefix += '_euler'
    if args.guidance_weight > 0:
        gw = args.guidance_weight
        prefix += f'_gw{int(gw) if gw == int(gw) else gw}_w{args.guidance_ncc_win}'
        if args.guidance_reg:
            prefix += '_reg'

    # Test loop
    num_seeds = len(seeds)
    step_rows = {s: [] for s in range(args.num_steps)}
    diversity_per_step = {s: [] for s in range(args.num_steps)}
    all_runtime = []

    for i, data in enumerate(test_loader):
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

        # Run all seeds
        all_intermediates = []
        t_start = time.time()
        for seed in seeds:
            torch.manual_seed(seed)
            all_intermediates.append(sampler(fixed, moving))
        runtime = time.time() - t_start
        all_runtime.append(runtime)

        with torch.no_grad():
            for si in range(args.num_steps):
                warped_imgs = []
                warped_segs = []
                for seed_idx, seed in enumerate(seeds):
                    ddf_i = all_intermediates[seed_idx][si]
                    warped = stn(moving, ddf_i)
                    wseg = warp_segmentation(moving_seg, ddf_i, device)
                    metrics = compute_all_metrics(ddf_i, warped, fixed, wseg, fixed_seg, moving_seg, args.direction)
                    row = {'pid': test_set.pids[i], 'step': si + 1, 'seed': seed, 'runtime': runtime}
                    row['ncc'] = -ncc_fn.loss(fixed, warped).item()
                    row['grad_l2'] = grad_l2_fn.loss(None, ddf_i).item()
                    for k, v in metrics.items():
                        row[k] = v.item()
                    step_rows[si].append(row)
                    warped_imgs.append(warped)
                    warped_segs.append(wseg)

                # Pairwise diversity between seeds (dice on labels + NCC on images)
                if num_seeds > 1:
                    pair_dices = []
                    pair_nccs = []
                    for a in range(num_seeds):
                        for b in range(a + 1, num_seeds):
                            d = compute_dice(warped_segs[a], warped_segs[b])
                            pair_dices.append(d['dice_mean'].item())
                            pair_nccs.append(-ncc_fn.loss(warped_imgs[a], warped_imgs[b]).item())
                    diversity_per_step[si].append((np.mean(pair_dices), np.mean(pair_nccs)))

    # Summary — dice per step
    dice_keys = ['dice_RV', 'dice_Myo', 'dice_LV', 'dice_mean']
    show_diversity = num_seeds > 1
    print('\n' + '=' * 60)
    print(f'Dice per step ({len(test_set)} samples, {num_seeds} seed{"s" if num_seeds > 1 else ""}):')
    header = f'  {"step":>4s}' + ''.join(f'  {k:>10s}' for k in dice_keys)
    if show_diversity:
        header += f'  {"pw_dice":>10s}  {"pw_ncc":>10s}'
    print(header)
    for s in range(args.num_steps):
        vals = {k: np.mean([r[k] for r in step_rows[s]]) for k in dice_keys}
        line = f'  {s+1:4d}' + ''.join(f'  {vals[k]:10.4f}' for k in dice_keys)
        if show_diversity:
            pw_dices = [d for d, _ in diversity_per_step[s]]
            pw_nccs = [n for _, n in diversity_per_step[s]]
            line += f'  {np.mean(pw_dices):10.4f}  {np.mean(pw_nccs):10.4f}'
        print(line)

    # Final step summary (all metrics)
    final_rows = step_rows[args.num_steps - 1]
    metric_keys = [k for k in final_rows[0] if k not in ('pid', 'step', 'seed', 'runtime')]
    print(f'\nFinal step summary:')
    for k in metric_keys:
        vals = [r[k] for r in final_rows]
        print(f'  {k:20s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}')
    if show_diversity:
        final_pw_dices = [d for d, _ in diversity_per_step[args.num_steps-1]]
        final_pw_nccs = [n for _, n in diversity_per_step[args.num_steps-1]]
        print(f'  {"pairwise_dice":20s}: {np.mean(final_pw_dices):.4f} +/- {np.std(final_pw_dices):.4f}')
        print(f'  {"pairwise_ncc":20s}: {np.mean(final_pw_nccs):.4f} +/- {np.std(final_pw_nccs):.4f}')
    print(f'  {"runtime":20s}: {np.mean(all_runtime[1:]):.3f} +/- {np.std(all_runtime[1:]):.3f}s (excl. first)')

    # Per-sample CSV (all steps × all seeds)
    all_rows = [r for s in range(args.num_steps) for r in step_rows[s]]
    out = ckpt_dir / f'{prefix}.csv'
    fieldnames = list(all_rows[0].keys())
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'Saved {out}')

    # Summary CSV
    summary_path = ckpt_dir / f'{prefix}_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'metric', 'mean', 'std'])
        for s in range(args.num_steps):
            for k in metric_keys:
                vals = [r[k] for r in step_rows[s]]
                writer.writerow([s + 1, k, np.mean(vals), np.std(vals)])
            if show_diversity:
                pw_dices = [d for d, _ in diversity_per_step[s]]
                pw_nccs = [n for _, n in diversity_per_step[s]]
                writer.writerow([s + 1, 'pairwise_dice', np.mean(pw_dices), np.std(pw_dices)])
                writer.writerow([s + 1, 'pairwise_ncc', np.mean(pw_nccs), np.std(pw_nccs)])
    print(f'Saved {summary_path}')


if __name__ == "__main__":
    main()
