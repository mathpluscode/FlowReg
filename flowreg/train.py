"""Training script for FlowReg with flow matching."""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse
import copy
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from math import ceil
from pathlib import Path
import yaml
from torch.utils.data import DataLoader
from data import DATASETS
from utils.device import get_device
from data.util import SPACING
from utils.metrics import compute_all_metrics, warp_segmentation

from . import networks
from utils.losses import NCC, Grad


@torch.no_grad()
def ema_update(teacher, student, momentum=0.99):
    """Update teacher weights with exponential moving average of student."""
    for t_p, s_p in zip(teacher.parameters(), student.parameters()):
        t_p.data.mul_(momentum).add_(s_p.data, alpha=1 - momentum)


@torch.no_grad()
def sample(model, fixed, moving, num_steps=5):
    """ODE sampling with Heun's method. Returns list of DDFs (one per step)."""
    B = fixed.shape[0]
    min_sp = min(SPACING)
    noise_scale = torch.tensor([min_sp / s for s in SPACING], device=fixed.device).view(1, 3, 1, 1, 1)
    ddf = torch.randn(B, 3, *fixed.shape[2:], device=fixed.device) * noise_scale

    ts = torch.linspace(0, 1, num_steps + 1, device=fixed.device)
    intermediates = []

    for i in range(num_steps):
        t_cur, t_next = ts[i], ts[i + 1]
        h = t_next - t_cur

        # Velocity at current point
        t_batch = torch.full((B,), t_cur.item(), device=fixed.device)
        pred = model(fixed, moving, ddf, t_batch)
        t_exp = t_batch[:, None, None, None, None].clamp(max=1 - 1e-5)
        v1 = (pred - ddf) / (1 - t_exp)

        # Euler predictor
        ddf_next = ddf + v1 * h

        # Heun corrector (skip last step to avoid t=1 singularity)
        if i < num_steps - 1:
            t_next_batch = torch.full((B,), t_next.item(), device=fixed.device)
            pred_next = model(fixed, moving, ddf_next, t_next_batch)
            t_next_exp = t_next_batch[:, None, None, None, None].clamp(max=1 - 1e-5)
            v2 = (pred_next - ddf_next) / (1 - t_next_exp)
            ddf_next = ddf + (v1 + v2) / 2 * h
            pred = pred_next  # prediction from better state

        ddf = ddf_next
        intermediates.append(pred)

    return intermediates


def evaluate(model, test_loader, device, num_steps=5, direction='ed2es'):
    model.eval()
    stn = networks.SpatialTransformer(mode='bilinear').to(device)
    # per_step_metrics[step_idx][metric_name] = list of values
    per_step_metrics = [{} for _ in range(num_steps)]

    with torch.no_grad():
        for data in test_loader:
            # F=ES, M=ED in dataloader
            if direction == 'ed2es':
                fixed = (data['F'].to(device) + 1) / 2
                moving = (data['M'].to(device) + 1) / 2
                fixed_seg = data['FS'].to(device)
                moving_seg = data['MS'].to(device)
            else:
                fixed = (data['M'].to(device) + 1) / 2
                moving = (data['F'].to(device) + 1) / 2
                fixed_seg = data['MS'].to(device)
                moving_seg = data['FS'].to(device)

            intermediates = sample(model, fixed, moving, num_steps)

            for step_idx, flow in enumerate(intermediates):
                warped = stn(moving, flow)
                warped_seg = warp_segmentation(moving_seg, flow, device)
                metrics = compute_all_metrics(flow, warped, fixed, warped_seg, fixed_seg, moving_seg, direction)
                for k, v in metrics.items():
                    per_step_metrics[step_idx].setdefault(k, []).extend(v.tolist())

    # Aggregate: {metric_name: (mean, std)} for each step
    results_per_step = []
    for step_metrics in per_step_metrics:
        results_per_step.append({k: (np.mean(v), np.std(v)) for k, v in step_metrics.items()})

    return results_per_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['acdc', 'mnms2'])
    parser.add_argument('--config', type=str, default='flowreg/train.yaml')
    parser.add_argument('--direction', type=str, default='ed2es', choices=['ed2es', 'es2ed'],
                        help='Registration direction: ed2es (Fixed=ES, Moving=ED) or es2ed (Fixed=ED, Moving=ES)')
    parser.add_argument('--debug', action='store_true', help='Run 1 batch only')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    with open(args.config, 'r') as f:
        opt = yaml.safe_load(f)

    device = get_device()
    print(f"Device: {device}")

    # Initialize wandb
    wandb.init(
        entity=opt['wandb']['entity'],
        project=opt['wandb']['project'],
        name=f"{opt['name']}_{args.dataset}_{args.direction}_{time.strftime('%m%d_%H%M')}",
        tags=[opt['name'], args.dataset, args.direction],
        config={
            'n_epoch': opt['train']['n_epoch'],
            'batch_size': opt['datasets']['train']['batch_size'],
            'lr': opt['train']['optimizer']['lr'],
            'debug': args.debug,
        },
        mode='disabled' if args.debug else 'online',
    )

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # Dataset
    dataset_opt = opt['datasets']['train']
    Dataset = DATASETS[args.dataset]
    train_set = Dataset(args.data_dir, 'train')
    train_loader = DataLoader(train_set, batch_size=dataset_opt['batch_size'],
                              shuffle=dataset_opt['use_shuffle'], num_workers=dataset_opt['num_workers'], pin_memory=True)
    val_set = Dataset(args.data_dir, 'val')
    val_loader = DataLoader(val_set, batch_size=dataset_opt['batch_size'], shuffle=False, num_workers=dataset_opt['num_workers'], pin_memory=True)
    test_set = Dataset(args.data_dir, 'test')
    test_loader = DataLoader(test_set, batch_size=dataset_opt['batch_size'], shuffle=False, num_workers=dataset_opt['num_workers'], pin_memory=True)
    n_iters = ceil(len(train_set) / dataset_opt['batch_size'])
    print(f'Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}')

    # Model (student)
    model = networks.FlowReg(
        in_chs=opt['model']['in_channels'],
        enc_chs=opt['model']['enc_channels'],
        dec_chs=opt['model']['dec_channels'],
        t_chs=opt['model']['t_channels'],
    )
    model.to(device)
    print(f"Model on {device}")

    # Teacher (EMA copy, frozen)
    teacher = copy.deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    stn = networks.SpatialTransformer(mode='bilinear').to(device)

    ema_momentum = opt['model'].get('ema_momentum', 0.99)
    num_steps = opt['model'].get('num_inference_steps', 5)

    # Optimizer
    if opt['train']['optimizer']['type'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt['train']['optimizer']['lr'])
    elif opt['train']['optimizer']['type'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt['train']['optimizer']['lr'],
            weight_decay=opt['train']['optimizer'].get('weight_decay', 1e-4)
        )
    else:
        raise NotImplementedError(opt['train']['optimizer']['type'])

    # Learning rate scheduler
    if opt['train']['optimizer'].get('schedule') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            eta_min=opt['train']['optimizer'].get('eta_min', 1e-6),
            T_max=opt['train']['n_epoch']
        )
    else:
        scheduler = None

    # Losses
    ncc_loss = NCC(win=opt['train']['loss']['ncc_win'])
    grad_loss = Grad('l2')
    reg_weight = opt['train']['loss']['reg_weight']
    v_mse_weight = opt['train']['loss'].get('v_mse_weight', 1.0)

    # Checkpoint dir (inside wandb run dir)
    ckpt_dir = Path(wandb.run.dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Early stopping
    best_val_dice = -1.0
    patience_counter = 0
    patience = 10

    # Train
    n_epoch = 1 if args.debug else opt['train']['n_epoch']
    max_iters = 1 if args.debug else n_iters
    warmup_epochs = opt['train'].get('warmup_epochs', 50)

    print(f"\nTraining{' (debug)' if args.debug else ''}...")
    for current_epoch in range(1, n_epoch + 1):
        epoch_start = time.time()
        model.train()
        is_warmup = current_epoch <= warmup_epochs

        epoch_losses = {'ncc': [], 'reg': [], 'total': [], 'l_fm_mse': [], 'l_fm_ncc': []}

        for i, data in enumerate(train_loader):
            if i >= max_iters:
                break

            if args.direction == 'ed2es':
                fixed = (data['F'].to(device) + 1) / 2   # ES
                moving = (data['M'].to(device) + 1) / 2  # ED
            else:
                fixed = (data['M'].to(device) + 1) / 2   # ED
                moving = (data['F'].to(device) + 1) / 2  # ES
            B = fixed.shape[0]

            # SPACING is (z, y, x) — scale noise per axis by min_spacing / spacing_i
            min_sp = min(SPACING)
            noise_scale = torch.tensor([min_sp / s for s in SPACING], device=device).view(1, 3, 1, 1, 1)
            noise = torch.randn(B, 3, *fixed.shape[2:], device=device) * noise_scale * 5
            t_zero = torch.zeros(B, device=device)
            if is_warmup:
                # Warmup: student predicts DDF from noise at t=0 directly
                pred_ddf = model(fixed, moving, noise, t_zero)
            else:
                # Reflow: teacher ODE target -> straight-line interpolation -> student
                # Same noise used for teacher and interpolation to preserve (x_0, x_1) coupling
                with torch.no_grad():
                    target_ddf = teacher(fixed, moving, noise, t_zero)
                # Logit-normal time sampling (Esser et al., "Scaling Rectified Flow
                # Transformers for High-Resolution Image Synthesis", ICML 2024, Sec 3.3)
                t = torch.sigmoid(torch.randn(B, device=device))  # m=0, s=1
                t_expand = t[:, None, None, None, None]
                noisy_ddf = (t_expand * target_ddf + (1 - t_expand) * noise).detach()
                pred_ddf = model(fixed, moving, noisy_ddf, t)

            # Registration loss on predicted DDF
            warped = stn(moving, pred_ddf)
            loss_ncc = ncc_loss.loss(fixed, warped)
            loss_reg = grad_loss.loss(None, pred_ddf) * reg_weight
            loss = loss_ncc + loss_reg

            # Velocity MSE loss (flow matching phase only)
            loss_fm_mse = torch.tensor(0.0, device=device)
            loss_fm_ncc = torch.tensor(0.0, device=device)
            if not is_warmup:
                # MSE on velocity
                v_true = (target_ddf - noise).detach()
                v_pred = pred_ddf - noise
                loss_fm_mse = F.mse_loss(v_pred, v_true) * v_mse_weight

                # NCC on warped image
                warped_target = stn(fixed, pred_ddf)
                loss_fm_ncc = ncc_loss.loss(warped_target, warped)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # EMA update teacher
            ema_update(teacher, model, ema_momentum)

            epoch_losses['ncc'].append(loss_ncc.item())
            epoch_losses['reg'].append(loss_reg.item())
            epoch_losses['total'].append(loss.item())
            epoch_losses['l_fm_mse'].append(loss_fm_mse.item())
            epoch_losses['l_fm_ncc'].append(loss_fm_ncc.item())

            log_step = {
                'train/grad_norm': grad_norm.item(),
                'train/l_ncc': loss_ncc.item(),
                'train/l_reg': loss_reg.item(),
                'train/l_tot': loss.item(),
                'train/l_fm_mse': loss_fm_mse.item(),
                'train/l_fm_ncc': loss_fm_ncc.item(),
                'train/pred_ddf_abs_mean': pred_ddf.abs().mean().item(),
                'train/pred_ddf_abs_min': pred_ddf.abs().min().item(),
                'train/pred_ddf_abs_max': pred_ddf.abs().max().item(),
            }
            if not is_warmup:
                log_step.update({
                    'train/t_mean': t.mean().item(),
                    'train/t_min': t.min().item(),
                    'train/t_max': t.max().item(),
                    'train/target_ddf_abs_mean': target_ddf.abs().mean().item(),
                    'train/target_ddf_abs_min': target_ddf.abs().min().item(),
                    'train/target_ddf_abs_max': target_ddf.abs().max().item(),
                })
            wandb.log(log_step)

        # After warmup: copy student -> teacher for flow matching
        if current_epoch == warmup_epochs:
            teacher.load_state_dict(model.state_dict())
            print(f'Warmup done. Copied student -> teacher at epoch {current_epoch}.')

        if scheduler:
            scheduler.step()
            wandb.log({'lr': scheduler.get_last_lr()[0], 'epoch': current_epoch})

        # Evaluate both directions
        eval_freq = opt['train'].get('eval_freq', 10)
        if current_epoch % eval_freq == 0 or current_epoch == n_epoch:
            torch.cuda.empty_cache()
            log_dict = {'epoch': current_epoch}
            for direction in ['ed2es', 'es2ed']:
                val_per_step = evaluate(model, val_loader, device, num_steps, direction)
                test_per_step = evaluate(model, test_loader, device, num_steps, direction)

                # Log per-step metrics
                for step_idx in range(num_steps):
                    step_label = f'step{step_idx + 1}'
                    val_r = val_per_step[step_idx]
                    test_r = test_per_step[step_idx]
                    log_dict.update({
                        **{f'val/{direction}/{step_label}/{k}': v[0] for k, v in val_r.items()},
                        **{f'test/{direction}/{step_label}/{k}': v[0] for k, v in test_r.items()},
                    })

                # Print per-step dice_mean summary
                val_steps_str = ' -> '.join([f'{val_per_step[s]["dice_mean"][0]:.3f}' for s in range(num_steps)])
                test_steps_str = ' -> '.join([f'{test_per_step[s]["dice_mean"][0]:.3f}' for s in range(num_steps)])
                print(f'--- Epoch {current_epoch} [{direction}] | Val dice/step:  {val_steps_str}')
                print(f'--- Epoch {current_epoch} [{direction}] | Test dice/step: {test_steps_str}')

                # Also log final step without step label for backward compat
                val_final = val_per_step[-1]
                test_final = test_per_step[-1]
                log_dict.update({
                    **{f'val/{direction}/{k}': v[0] for k, v in val_final.items()},
                    **{f'test/{direction}/{k}': v[0] for k, v in test_final.items()},
                })

            print(f'--- {time.time()-epoch_start:.1f}s ---\n')
            wandb.log(log_dict)

            # Early stopping on val dice_mean for the training direction (final step)
            val_dice_mean = log_dict[f'val/{args.direction}/dice_mean']
            if val_dice_mean > best_val_dice:
                best_val_dice = val_dice_mean
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_dir / 'best.pt')
                print(f'New best val/{args.direction}/dice_mean: {best_val_dice:.4f}, saved best.pt')
            else:
                patience_counter += 1
                print(f'No improvement ({patience_counter}/{patience})')
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {current_epoch}')
                    break
        else:
            avg_loss = np.mean(epoch_losses['total'])
            print(f'--- Epoch {current_epoch} | Loss: {avg_loss:.4f} | {time.time()-epoch_start:.1f}s ---\n')

        # Save checkpoint
        if not args.debug and current_epoch % opt['train']['save_checkpoint_epoch'] == 0:
            torch.save(model.state_dict(), ckpt_dir / f'epoch_{current_epoch:03d}.pt')
            print(f'Saved checkpoint at epoch {current_epoch}')

    if best_val_dice > 0:
        print(f'\nBest checkpoint: {ckpt_dir / "best.pt"} (dice_mean={best_val_dice:.4f})')

    wandb.finish()


if __name__ == "__main__":
    main()
