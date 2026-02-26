import torch
import torch.nn as nn
import torch.nn.functional as F


class gradientLoss(nn.Module):
    def __init__(self, penalty='l1'):
        super().__init__()
        self.penalty = penalty

    def forward(self, input):
        dD = torch.abs(input[:, :, 1:, :, :] - input[:, :, :-1, :, :])
        dH = torch.abs(input[:, :, :, 1:, :] - input[:, :, :, :-1, :])
        dW = torch.abs(input[:, :, :, :, 1:] - input[:, :, :, :, :-1])
        if self.penalty == "l2":
            dD, dH, dW = dD * dD, dH * dH, dW * dW
        return (torch.mean(dD) + torch.mean(dH) + torch.mean(dW)) / 3.0


class crossCorrelation3D(nn.Module):
    def __init__(self, in_ch, kernel=(9, 9, 9), gamma=1):
        super().__init__()
        self.kernel = kernel
        self.gamma = gamma
        self.register_buffer('filt', torch.ones(1, in_ch, *kernel))

    def forward(self, input, target, flow):
        target = (target + 1) / 2  # from [-1,1] to [0,1]

        II, TT, IT = input * input, target * target, input * target
        flow = F.sigmoid(flow) ** self.gamma
        pad = tuple(k // 2 for k in self.kernel)

        T_sum = F.conv3d(target, self.filt, stride=1, padding=pad)
        I_sum = F.conv3d(input, self.filt, stride=1, padding=pad)
        TT_sum = F.conv3d(TT, self.filt, stride=1, padding=pad)
        II_sum = F.conv3d(II, self.filt, stride=1, padding=pad)
        IT_sum = F.conv3d(IT, self.filt, stride=1, padding=pad)

        ks = self.kernel[0] * self.kernel[1] * self.kernel[2]
        Ihat, That = I_sum / ks, T_sum / ks

        cross = IT_sum - Ihat * T_sum - That * I_sum + That * Ihat * ks
        T_var = TT_sum - 2 * That * T_sum + That * That * ks
        I_var = II_sum - 2 * Ihat * I_sum + Ihat * Ihat * ks
        cc = cross * cross * flow / (T_var * I_var + 1e-5)

        return -torch.mean(cc)
