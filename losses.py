import torch
import torch.nn as nn
import torch.nn.functional as F


def tv_loss(x, beta = 0.5, reg_coeff = 5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    a,b,c,d=x.shape
    return reg_coeff*(torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))/(a*b*c*d))


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]



class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, channels=3):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channels = channels
        # 固定结构：高斯核作为 Buffer 存入，随模型移动到 GPU
        self.register_buffer("window", self._create_window(window_size, sigma, channels))

    def _create_window(self, window_size, sigma, channels):
        coords = torch.arange(window_size).float() - window_size // 2
        gauss = torch.exp(-(coords**2) / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        return _2D_window.expand(channels, 1, window_size, window_size).contiguous()

    def forward(self, img1, img2, as_loss=True):
        # 核心计算逻辑
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channels)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channels)

        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=self.channels) - mu1_mu2

        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        ssim_score = ssim_map.mean()

        # 根据调用时的需求返回不同结果
        if as_loss:
            return 1 - ssim_score
        else:
            return ssim_score
