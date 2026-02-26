import torch
import torch.nn.functional as F
import numpy as np
import math

from data.util import SPACING


class NCC:
    def __init__(self, win=9):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [self.win] * ndims

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(y_pred.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class Grad:
    def __init__(self, penalty='l2', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        # SPACING is (z, y, x) — convert DDF to physical units, then compute spatial derivatives
        spacing = torch.tensor(SPACING, dtype=y_pred.dtype, device=y_pred.device).view(1, 3, 1, 1, 1)
        _pred = y_pred * spacing

        dz = torch.abs(_pred[:, :, 1:, :, :] - _pred[:, :, :-1, :, :]) / SPACING[0]
        dy = torch.abs(_pred[:, :, :, 1:, :] - _pred[:, :, :, :-1, :]) / SPACING[1]
        dx = torch.abs(_pred[:, :, :, :, 1:] - _pred[:, :, :, :, :-1]) / SPACING[2]

        if self.penalty == 'l2':
            dz = dz * dz
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dz) + torch.mean(dy) + torch.mean(dx)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad


class SoftDice:

    def __init__(self, labels=(1, 2, 3)):
        self.labels = labels

    def loss(self, fixed_seg, warped_seg_soft):
        """Args:
            fixed_seg: [B, D, H, W] integer labels.
            warped_seg_soft: [B, C, D, H, W] soft warped one-hot (differentiable).
        Returns negative mean Dice (to minimize).
        """
        dice_sum = 0.0
        for i, label in enumerate(self.labels):
            p = warped_seg_soft[:, i]  # [B, D, H, W]
            g = (fixed_seg == label).float()
            inter = (p * g).sum()
            union = p.sum() + g.sum()
            dice_sum = dice_sum + 2 * inter / (union + 1e-8)
        return -dice_sum / len(self.labels)
