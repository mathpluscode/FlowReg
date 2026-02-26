"""Training script for CorrMLP."""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse
import random
import time
import numpy as np
import torch
import wandb
from math import ceil
from pathlib import Path

from torch.utils.data import DataLoader
from data import DATASETS
from utils import logger as Logger
from utils.device import get_device
from utils.metrics import warp_segmentation, compute_all_metrics

from . import networks
from utils.losses import NCC, Grad


def evaluate(model, test_loader, device, direction='ed2es'):
    """Evaluate model on test set.

    Args:
        direction: 'ed2es' (Fixed=ES, Moving=ED, default) or 'es2ed' (Fixed=ED, Moving=ES).
    """
    model.eval()
    all_metrics = {}

    with torch.no_grad():
        for data in test_loader:
            # F=ES, M=ED in dataloader
            if direction == 'ed2es':
                # Fixed=ES, Moving=ED -> warp ED seg toward ES (default)
                fixed = (data['F'].to(device) + 1) / 2
                moving = (data['M'].to(device) + 1) / 2
                fixed_seg = data['FS'].to(device)
                moving_seg = data['MS'].to(device)
            else:
                # Fixed=ED, Moving=ES -> warp ES seg toward ED
                fixed = (data['M'].to(device) + 1) / 2
                moving = (data['F'].to(device) + 1) / 2
                fixed_seg = data['MS'].to(device)
                moving_seg = data['FS'].to(device)

            warped, flow = model(fixed, moving)
            warped_seg = warp_segmentation(moving_seg, flow, device)

            metrics = compute_all_metrics(flow, warped, fixed, warped_seg, fixed_seg, moving_seg, direction)
            for k, v in metrics.items():
                all_metrics.setdefault(k, []).extend(v.tolist())

    return {k: (np.mean(v), np.std(v)) for k, v in all_metrics.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['acdc', 'mnms2'])
    parser.add_argument('--config', type=str, default='corrmlp/train.yaml')
    parser.add_argument('--direction', type=str, default='ed2es', choices=['ed2es', 'es2ed'],
                        help='Registration direction: ed2es (Fixed=ES, Moving=ED) or es2ed (Fixed=ED, Moving=ES)')
    parser.add_argument('--debug', action='store_true', help='Run 1 batch only')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    class ConfigArgs:
        config = args.config
    opt = Logger.dict_to_nonedict(Logger.parse(ConfigArgs()))

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
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    test_set = Dataset(args.data_dir, 'test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    n_iters = ceil(len(train_set) / dataset_opt['batch_size'])
    print(f'Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}')

    # Model
    model = networks.CorrMLP(
        in_channels=opt['model']['in_channels'],
        enc_channels=opt['model']['enc_channels'],
        dec_channels=opt['model']['dec_channels'],
        use_checkpoint=opt['model']['use_checkpoint']
    )
    model.to(device)
    print(f"Model on {device}")

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

    print(f"\nTraining{' (debug)' if args.debug else ''}...")
    for current_epoch in range(1, n_epoch + 1):
        epoch_start = time.time()
        model.train()

        epoch_losses = {'ncc': [], 'reg': [], 'total': []}

        for i, data in enumerate(train_loader):
            if i >= max_iters:
                break

            # Convert from [-1, 1] to [0, 1] range for CorrMLP
            if args.direction == 'ed2es':
                fixed = (data['F'].to(device) + 1) / 2   # ES
                moving = (data['M'].to(device) + 1) / 2  # ED
            else:
                fixed = (data['M'].to(device) + 1) / 2   # ED
                moving = (data['F'].to(device) + 1) / 2  # ES

            # Forward
            warped, flow = model(fixed, moving)

            # Loss
            loss_ncc = ncc_loss.loss(fixed, warped)
            loss_reg = grad_loss.loss(None, flow) * reg_weight
            loss = loss_ncc + loss_reg

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses['ncc'].append(loss_ncc.item())
            epoch_losses['reg'].append(loss_reg.item())
            epoch_losses['total'].append(loss.item())

            wandb.log({
                'train/l_ncc': loss_ncc.item(),
                'train/l_reg': loss_reg.item(),
                'train/l_tot': loss.item(),
            })

        if scheduler:
            scheduler.step()
            wandb.log({'lr': scheduler.get_last_lr()[0], 'epoch': current_epoch})

        # Evaluate both directions
        eval_freq = opt['train'].get('eval_freq', 10)
        if current_epoch % eval_freq == 0 or current_epoch == n_epoch:
            log_dict = {'epoch': current_epoch}
            for direction in ['ed2es', 'es2ed']:
                val_results = evaluate(model, val_loader, device, direction)
                test_results = evaluate(model, test_loader, device, direction)

                val_str = ' | '.join([f'{k}: {v[0]:.3f}' for k, v in val_results.items()])
                test_str = ' | '.join([f'{k}: {v[0]:.3f}' for k, v in test_results.items()])
                print(f'--- Epoch {current_epoch} [{direction}] | Val: {val_str}')
                print(f'--- Epoch {current_epoch} [{direction}] | Test: {test_str}')

                log_dict.update({
                    **{f'val/{direction}/{k}': v[0] for k, v in val_results.items()},
                    **{f'test/{direction}/{k}': v[0] for k, v in test_results.items()},
                })

            print(f'--- {time.time()-epoch_start:.1f}s ---\n')
            wandb.log(log_dict)

            # Early stopping on val dice_mean for the training direction
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

    print(f'\nBest checkpoint: {ckpt_dir / "best.pt"} (dice_mean={best_val_dice:.4f})')
    wandb.finish()


if __name__ == "__main__":
    main()
