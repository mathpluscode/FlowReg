"""Training script for FSDiffReg."""

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
from fsdiffreg import model as Model
from utils import logger as Logger
from utils.device import get_device
from utils.metrics import warp_segmentation, compute_all_metrics


def evaluate(diffusion, test_loader, direction='ed2es'):
    """Evaluate model on test set.

    Args:
        direction: 'ed2es' (Fixed=ES, Moving=ED, default) or 'es2ed' (Fixed=ED, Moving=ES).
    """
    diffusion.netG.eval()
    all_metrics = {}

    with torch.no_grad():
        for data in test_loader:
            if direction == 'ed2es':
                feed = data
                fixed_img = (data['F'].to(diffusion.device) + 1) / 2
                moving_seg = data['MS'].to(diffusion.device)
                fixed_seg = data['FS'].to(diffusion.device)
            else:
                feed = {'M': data['F'], 'F': data['M'], 'MS': data['FS'], 'FS': data['MS'], 'Index': data['Index']}
                fixed_img = (data['M'].to(diffusion.device) + 1) / 2
                moving_seg = data['FS'].to(diffusion.device)
                fixed_seg = data['MS'].to(diffusion.device)

            diffusion.feed_data(feed)
            diffusion.test_registration()
            reg = diffusion.get_current_registration()
            flow = reg['flow'].to(diffusion.device)
            warped_img = reg['out_M'].to(diffusion.device)

            # Convert warped image to [0, 1] if needed
            warped_img = (warped_img + 1) / 2

            warped_seg = warp_segmentation(moving_seg, flow, diffusion.device)

            metrics = compute_all_metrics(flow, warped_img, fixed_img, warped_seg, fixed_seg, moving_seg, direction)
            for k, v in metrics.items():
                all_metrics.setdefault(k, []).extend(v.tolist())

    return {k: (np.mean(v), np.std(v)) for k, v in all_metrics.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['acdc', 'mnms2'])
    parser.add_argument('--config', type=str, default='fsdiffreg/train.yaml')
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
            'loss_lambda': opt['model']['loss_lambda'],
            'debug': args.debug,
        },
        mode='disabled' if args.debug else 'online',
    )

    # Save checkpoints inside wandb run dir
    opt['path']['checkpoint'] = wandb.run.dir

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
    diffusion = Model.create_model(opt)
    print(f"Model on {diffusion.device}")

    # Early stopping
    best_val_dice = -1.0
    patience_counter = 0
    patience = 10

    # Train
    n_epoch = 1 if args.debug else opt['train']['n_epoch']
    max_iters = 1 if args.debug else n_iters
    current_epoch, current_step = diffusion.begin_epoch, diffusion.begin_step

    print(f"\nTraining{' (debug)' if args.debug else ''}...")
    while current_epoch < n_epoch:
        current_epoch += 1
        epoch_start = time.time()
        diffusion.netG.train()

        for i, data in enumerate(train_loader):
            if i >= max_iters:
                break
            current_step += 1
            if args.direction == 'es2ed':
                data = {'M': data['F'], 'F': data['M'], 'MS': data['FS'], 'FS': data['MS'], 'Index': data['Index']}
            diffusion.feed_data(data)
            diffusion.optimize_parameters()

            logs = diffusion.get_current_log()
            wandb.log({
                'train/l_pix': logs['l_pix'],
                'train/l_sim': logs['l_sim'],
                'train/l_smt': logs['l_smt'],
                'train/l_tot': logs['l_tot'],
                'step': current_step,
            })

        # Evaluate both directions
        eval_freq = opt['train'].get('eval_freq', 10)
        if current_epoch % eval_freq == 0 or current_epoch == n_epoch:
            log_dict = {'epoch': current_epoch}
            for direction in ['ed2es', 'es2ed']:
                val_results = evaluate(diffusion, val_loader, direction)
                test_results = evaluate(diffusion, test_loader, direction)

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
                best_path = Path(opt['path']['checkpoint']) / 'best_gen_G.pth'
                network = diffusion.netG.module if isinstance(diffusion.netG, torch.nn.DataParallel) else diffusion.netG
                torch.save(network.state_dict(), best_path)
                print(f'New best val/{args.direction}/dice_mean: {best_val_dice:.4f}, saved best checkpoint')
            else:
                patience_counter += 1
                print(f'No improvement ({patience_counter}/{patience})')
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {current_epoch}')
                    break
        else:
            print(f'--- Epoch {current_epoch} | {time.time()-epoch_start:.1f}s ---\n')

        if not args.debug and current_epoch % opt['train']['save_checkpoint_epoch'] == 0:
            diffusion.save_network(current_epoch, current_step)
            print(f'Saved checkpoint at epoch {current_epoch}')

    best_path = Path(opt['path']['checkpoint']) / 'best_gen_G.pth'
    print(f'\nBest checkpoint: {best_path} (dice_mean={best_val_dice:.4f})')
    wandb.finish()


if __name__ == "__main__":
    main()
