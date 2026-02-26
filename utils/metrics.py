"""Evaluation metrics for registration."""

import numpy as np
import torch
import torch.nn.functional as F
from monai.metrics import compute_dice as monai_compute_dice
from monai.metrics import compute_hausdorff_distance, SSIMMetric, PSNRMetric
from monai.networks.utils import one_hot

from data.util import SPACING
from utils.clinical import compute_volumes, compute_ef, compute_myo_thickness



def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = 2 * np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    return intersection / (union + 1e-8)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute Dice scores for ACDC classes (numpy, single sample)."""
    rv = (pred == 1).astype(float), (gt == 1).astype(float)
    myo = (pred == 2).astype(float), (gt == 2).astype(float)
    lv = (pred == 3).astype(float), (gt == 3).astype(float)
    scores = {
        'RV': _dice(*rv), 'Myo': _dice(*myo), 'LV': _dice(*lv),
        'LV+Myo': _dice(lv[0] + myo[0], lv[1] + myo[1]),
        'All': _dice(rv[0] + myo[0] + lv[0], rv[1] + myo[1] + lv[1]),
    }
    scores['mean'] = np.mean([scores['RV'], scores['Myo'], scores['LV']])
    return scores



def warp_segmentation(seg: torch.Tensor, flow: torch.Tensor, device) -> torch.Tensor:
    """Warp segmentation using flow field.

    Args:
        seg: Segmentation [B, D, H, W]
        flow: Flow field [B, 3, D, H, W]
        device: Target device

    Returns:
        Warped segmentation [B, D, H, W]
    """
    seg = seg.to(device)
    flow = flow.to(device)

    B, D, H, W = seg.shape

    # Create identity grid
    d = torch.linspace(-1, 1, D, device=device)
    h = torch.linspace(-1, 1, H, device=device)
    w = torch.linspace(-1, 1, W, device=device)
    grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')
    grid = torch.stack([grid_w, grid_h, grid_d], dim=-1).unsqueeze(0).expand(B, -1, -1, -1, -1)

    # Convert flow to normalized coordinates
    flow = flow.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]
    flow_norm = torch.zeros_like(flow)
    flow_norm[..., 0] = flow[..., 2] / (W - 1) * 2
    flow_norm[..., 1] = flow[..., 1] / (H - 1) * 2
    flow_norm[..., 2] = flow[..., 0] / (D - 1) * 2

    new_grid = grid + flow_norm

    # Warp with nearest neighbor for segmentation
    seg_input = seg.unsqueeze(1).float()  # [B, 1, D, H, W]
    warped = F.grid_sample(seg_input, new_grid, mode='nearest', padding_mode='border', align_corners=True)
    return warped.squeeze(1)



def compute_spatial_derivatives(flow_phys):
    """Compute spatial derivatives of physical-unit displacement field.

    Args:
        flow_phys: [B, 3, D, H, W] displacement in mm.

    Returns:
        dz, dy, dx: each [B, 3, D-1, H-1, W-1] spatial derivatives.
    """
    dz = (flow_phys[:, :, 1:, :-1, :-1] - flow_phys[:, :, :-1, :-1, :-1]) / SPACING[0]
    dy = (flow_phys[:, :, :-1, 1:, :-1] - flow_phys[:, :, :-1, :-1, :-1]) / SPACING[1]
    dx = (flow_phys[:, :, :-1, :-1, 1:] - flow_phys[:, :, :-1, :-1, :-1]) / SPACING[2]
    return dz, dy, dx


def compute_jacobian_metrics(dz, dy, dx):
    """Compute Jacobian-based metrics from spatial derivatives.

    Args:
        dz, dy, dx: each [B, 3, D-1, H-1, W-1] from compute_spatial_derivatives.

    Returns:
        njd: NJD percentage per sample [B]
        jacobian_mae: MAE from identity per sample [B]
        jacobian_std: STD of |Jφ| per sample [B]
    """
    jac = torch.stack([
        torch.stack([dz[:, 0] + 1, dz[:, 1],     dz[:, 2]], dim=-1),
        torch.stack([dy[:, 0],     dy[:, 1] + 1, dy[:, 2]], dim=-1),
        torch.stack([dx[:, 0],     dx[:, 1],     dx[:, 2] + 1], dim=-1),
    ], dim=-1)
    det = torch.linalg.det(jac)
    njd = (det < 0).float().mean(dim=[1, 2, 3]) * 100.0
    eye = torch.eye(3, dtype=jac.dtype, device=jac.device)
    mae = (jac - eye).abs().mean(dim=[1, 2, 3, 4, 5])
    jacobian_std = det.std(dim=[1, 2, 3])
    return njd, mae, jacobian_std




def compute_ssim(warped, fixed):
    """Structural Similarity Index.

    Args:
        warped: [B, 1, D, H, W] in [0, 1]
        fixed: [B, 1, D, H, W] in [0, 1]

    Returns:
        SSIM per sample [B]
    """
    return SSIMMetric(spatial_dims=3, data_range=1.0)(warped, fixed).squeeze(1)


def compute_psnr(warped, fixed):
    """Peak Signal-to-Noise Ratio.

    Args:
        warped: [B, 1, D, H, W] in [0, 1]
        fixed: [B, 1, D, H, W] in [0, 1]

    Returns:
        PSNR per sample [B]
    """
    return PSNRMetric(max_val=1.0)(warped, fixed).squeeze(1)



def compute_dice(pred_seg, gt_seg):
    """Compute Dice scores for cardiac classes using MONAI.

    Args:
        pred_seg: [B, D, H, W] integer labels (0=BG, 1=RV, 2=Myo, 3=LV)
        gt_seg: [B, D, H, W] integer labels

    Returns:
        Dict with keys: dice_RV, dice_Myo, dice_LV, dice_mean, dice_foreground
        Each value is a numpy array of shape [B].
    """
    pred_oh = one_hot(pred_seg[:, None].long(), num_classes=4).float()  # [B, 4, D, H, W]
    gt_oh = one_hot(gt_seg[:, None].long(), num_classes=4).float()

    # Per-class dice: [B, 3] (RV, Myo, LV)
    dice = monai_compute_dice(y_pred=pred_oh, y=gt_oh, num_classes=4, include_background=False)

    results = {
        'dice_RV': dice[:, 0].cpu().numpy(),
        'dice_Myo': dice[:, 1].cpu().numpy(),
        'dice_LV': dice[:, 2].cpu().numpy(),
    }
    results['dice_mean'] = np.mean([results['dice_RV'], results['dice_Myo'], results['dice_LV']], axis=0)

    # Foreground dice
    pred_fg = torch.stack([(pred_seg == 0).float(), (pred_seg > 0).float()], dim=1)
    gt_fg = torch.stack([(gt_seg == 0).float(), (gt_seg > 0).float()], dim=1)
    dice_fg = monai_compute_dice(y_pred=pred_fg, y=gt_fg, num_classes=2, include_background=False)
    results['dice_foreground'] = dice_fg[:, 0].cpu().numpy()

    return results


def compute_hausdorff(pred_seg, gt_seg):
    """95th percentile Hausdorff distance for cardiac classes.

    Args:
        pred_seg: [B, D, H, W] integer labels
        gt_seg: [B, D, H, W] integer labels

    Returns:
        Dict with keys: hd_RV, hd_Myo, hd_LV, hd_foreground
        Each value is a numpy array of shape [B].
    """
    pred_oh = one_hot(pred_seg[:, None].long(), num_classes=4).float()
    gt_oh = one_hot(gt_seg[:, None].long(), num_classes=4).float()

    # Per-class HD: [B, 3]
    hd = compute_hausdorff_distance(
        y_pred=pred_oh, y=gt_oh,
        include_background=False,
        percentile=95,
        spacing=SPACING,
    )

    results = {
        'hd_RV': hd[:, 0].cpu().numpy(),
        'hd_Myo': hd[:, 1].cpu().numpy(),
        'hd_LV': hd[:, 2].cpu().numpy(),
    }
    results['hd_mean'] = np.mean([results['hd_RV'], results['hd_Myo'], results['hd_LV']], axis=0)

    # Foreground HD
    pred_fg = torch.stack([(pred_seg == 0).float(), (pred_seg > 0).float()], dim=1)
    gt_fg = torch.stack([(gt_seg == 0).float(), (gt_seg > 0).float()], dim=1)
    hd_fg = compute_hausdorff_distance(
        y_pred=pred_fg, y=gt_fg,
        include_background=False,
        percentile=95,
        spacing=SPACING,
    )
    results['hd_foreground'] = hd_fg[:, 0].cpu().numpy()

    return results



def bending_energy(ddf, reduction='mean'):
    """Bending energy: mean of squared second-order spatial derivatives.

    Uses direct second-order finite difference stencils (not nested central
    differences) to avoid even/odd voxel artifacts.

    Args:
        ddf: [B, 3, D, H, W] displacement field.
        reduction: 'mean' returns scalar, 'none' returns per-sample [B].

    Returns:
        Scalar tensor (reduction='mean') or [B] tensor (reduction='none').
    """
    spacing_t = torch.tensor(SPACING, dtype=ddf.dtype, device=ddf.device).view(1, 3, 1, 1, 1)
    u = ddf * spacing_t  # convert to physical units

    d2z = (u[:, :, 2:, :, :] - 2 * u[:, :, 1:-1, :, :] + u[:, :, :-2, :, :]) / (SPACING[0] ** 2)
    d2y = (u[:, :, :, 2:, :] - 2 * u[:, :, :, 1:-1, :] + u[:, :, :, :-2, :]) / (SPACING[1] ** 2)
    d2x = (u[:, :, :, :, 2:] - 2 * u[:, :, :, :, 1:-1] + u[:, :, :, :, :-2]) / (SPACING[2] ** 2)

    d2zy = (u[:, :, 1:, 1:, :] - u[:, :, 1:, :-1, :] - u[:, :, :-1, 1:, :] + u[:, :, :-1, :-1, :]) / (SPACING[0] * SPACING[1])
    d2zx = (u[:, :, 1:, :, 1:] - u[:, :, 1:, :, :-1] - u[:, :, :-1, :, 1:] + u[:, :, :-1, :, :-1]) / (SPACING[0] * SPACING[2])
    d2yx = (u[:, :, :, 1:, 1:] - u[:, :, :, 1:, :-1] - u[:, :, :, :-1, 1:] + u[:, :, :, :-1, :-1]) / (SPACING[1] * SPACING[2])

    if reduction == 'none':
        def _sq_mean(t):
            return t.pow(2).mean(dim=[1, 2, 3, 4])
        be = (_sq_mean(d2z) + _sq_mean(d2y) + _sq_mean(d2x)
              + 2 * _sq_mean(d2zy) + 2 * _sq_mean(d2zx) + 2 * _sq_mean(d2yx))
    else:
        be = (torch.mean(d2z ** 2) + torch.mean(d2y ** 2) + torch.mean(d2x ** 2)
              + 2 * torch.mean(d2zy ** 2) + 2 * torch.mean(d2zx ** 2) + 2 * torch.mean(d2yx ** 2))
    return be / 9.0



def compute_all_metrics(flow, warped_img, fixed_img, warped_seg, fixed_seg, moving_seg,
                        direction):
    """Compute all registration metrics for a batch.

    Args:
        flow: [B, 3, D, H, W] displacement field
        warped_img: [B, 1, D, H, W] warped image in [0, 1]
        fixed_img: [B, 1, D, H, W] fixed image in [0, 1]
        warped_seg: [B, D, H, W] warped segmentation (integer labels)
        fixed_seg: [B, D, H, W] fixed segmentation (integer labels)
        moving_seg: [B, D, H, W] original moving segmentation (for volume diff)
        direction: 'ed2es' or 'es2ed' (for EF computation)

    Returns:
        Dict[str, np.ndarray] where each value has shape [B].
    """
    results = {}

    # Dice
    results.update(compute_dice(warped_seg, fixed_seg))

    # Hausdorff
    results.update(compute_hausdorff(warped_seg, fixed_seg))

    # Image quality
    results['ssim'] = compute_ssim(warped_img, fixed_img).cpu().numpy()
    results['psnr'] = compute_psnr(warped_img, fixed_img).cpu().numpy()

    # Flow quality — convert to physical (mm) once, reuse derivatives
    spacing = torch.tensor(SPACING, dtype=flow.dtype, device=flow.device).view(1, 3, 1, 1, 1)
    flow_phys = flow * spacing
    dz, dy, dx = compute_spatial_derivatives(flow_phys)

    njd, jacobian_mae, jacobian_std = compute_jacobian_metrics(dz, dy, dx)
    results['njd'] = njd.cpu().numpy()
    results['jacobian_mae'] = jacobian_mae.cpu().numpy()
    results['jacobian_std'] = jacobian_std.cpu().numpy()

    # Myocardium divergence: mean div(u)² in myo region
    div = dz[:, 0] + dy[:, 1] + dx[:, 2]  # [B, D-1, H-1, W-1]
    myo_mask = (moving_seg == 2).float()  # [B, D, H, W]
    myo_crop = myo_mask[:, :-1, :-1, :-1]
    results['myo_div'] = ((div.pow(2) * myo_crop).sum(dim=[1, 2, 3]) / myo_crop.sum(dim=[1, 2, 3]).clamp(min=1)).cpu().numpy()

    # Bending energy (per sample)
    results['bending_energy'] = bending_energy(flow, reduction='none').cpu().numpy()

    # Clinical metrics (EF, myo thickness)
    results.update(_compute_clinical_metrics(warped_seg, fixed_seg, moving_seg, direction))

    return results


def _compute_clinical_metrics(warped_seg, fixed_seg, moving_seg, direction):
    """Compute clinical metrics per sample in the batch.

    Convention:
        ed2es: moving=ED, fixed=ES, warped≈ES → pred ESV from warped
        es2ed: moving=ES, fixed=ED, warped≈ED → pred EDV from warped
    """
    B = warped_seg.shape[0]
    keys = ['LV_EF_true', 'LV_EF_pred', 'LV_EF_error',
            'RV_EF_true', 'RV_EF_pred', 'RV_EF_error',
            'myo_thickness_true', 'myo_thickness_pred', 'myo_thickness_error']
    results = {k: np.zeros(B) for k in keys}

    for b in range(B):
        moving_np = moving_seg[b].cpu().numpy().astype(int)
        fixed_np = fixed_seg[b].cpu().numpy().astype(int)
        warped_np = warped_seg[b].cpu().numpy().astype(int)

        moving_vols = compute_volumes(moving_np)
        fixed_vols = compute_volumes(fixed_np)
        warped_vols = compute_volumes(warped_np)

        for chamber in ['LV', 'RV']:
            if direction == 'ed2es':
                # moving=ED, fixed=ES, warped≈ES
                edv_true = moving_vols[chamber]
                esv_true = fixed_vols[chamber]
                esv_pred = warped_vols[chamber]
                ef_true = compute_ef(edv_true, esv_true)
                ef_pred = compute_ef(edv_true, esv_pred)
            else:
                # moving=ES, fixed=ED, warped≈ED
                esv_true = moving_vols[chamber]
                edv_true = fixed_vols[chamber]
                edv_pred = warped_vols[chamber]
                ef_true = compute_ef(edv_true, esv_true)
                ef_pred = compute_ef(edv_pred, esv_true)

            results[f'{chamber}_EF_true'][b] = ef_true
            results[f'{chamber}_EF_pred'][b] = ef_pred
            results[f'{chamber}_EF_error'][b] = abs(ef_true - ef_pred)

        # Myo thickness: ground truth target (fixed) vs predicted (warped)
        results['myo_thickness_true'][b] = compute_myo_thickness(fixed_np)
        results['myo_thickness_pred'][b] = compute_myo_thickness(warped_np)
        mt_true = results['myo_thickness_true'][b]
        mt_pred = results['myo_thickness_pred'][b]
        results['myo_thickness_error'][b] = abs(mt_true - mt_pred) if not (np.isnan(mt_true) or np.isnan(mt_pred)) else float('nan')

    return results
