import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ==============================================================================
# 1. 损失函数 (Loss Functions) - 注入模型的"灵魂"
# ==============================================================================

class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (L1 的平滑变体)
    相比 MSE (L2)，它对异常值更不敏感，能产生更锐利的边缘。
    公式: sqrt( (x-y)^2 + eps^2 )
    """

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


class FrequencyLoss(nn.Module):
    """
    频域损失 (Frequency Domain Loss)
    由于我们的模型强调 Wavelet 和频域特性，显式地约束频域一致性非常重要。
    使用 FFT 变换比较预测图和真值的频谱差异。
    """

    def __init__(self, loss_type='l1'):
        super(FrequencyLoss, self).__init__()
        self.loss_type = loss_type
        self.criterion = nn.L1Loss() if loss_type == 'l1' else nn.MSELoss()

    def forward(self, x, y):
        # 转换到频域
        # rfft2 适用于实数输入，输出一半的频谱 (共轭对称)
        x_fft = torch.fft.rfft2(x, norm='ortho')
        y_fft = torch.fft.rfft2(y, norm='ortho')

        # 分别比较实部和虚部，或者直接比较模长
        # 这里我们比较复数的模长 (Amplitude) 和实虚部
        loss_amp = self.criterion(torch.abs(x_fft), torch.abs(y_fft))
        loss_real = self.criterion(x_fft.real, y_fft.real)
        loss_imag = self.criterion(x_fft.imag, y_fft.imag)

        return loss_amp + 0.5 * (loss_real + loss_imag)


class EdgeLoss(nn.Module):
    """
    边缘损失 (Edge Loss)
    利用拉普拉斯算子提取边缘，强制模型恢复锐利的结构。
    """

    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[0.05, 0.25, 0.05], [0.25, -1.2, 0.25], [0.05, 0.25, 0.05]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel', k)
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        # 动态适配通道数
        b, c, h, w = x.shape
        kernel = self.kernel.expand(c, 1, 3, 3)

        x_edge = F.conv2d(x, kernel, padding=1, groups=c)
        y_edge = F.conv2d(y, kernel, padding=1, groups=c)

        return self.criterion(x_edge, y_edge)


class HybridLoss(nn.Module):
    """
    [终极武器] 混合损失函数
    Loss = alpha * Pixel_Loss + beta * Freq_Loss + gamma * Edge_Loss
    """

    def __init__(self, alpha=1.0, beta=0.1, gamma=0.05):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.pixel_loss = CharbonnierLoss()
        self.freq_loss = FrequencyLoss()
        self.edge_loss = EdgeLoss()

    def forward(self, pred, target):
        l_pix = self.pixel_loss(pred, target)
        l_freq = self.freq_loss(pred, target)
        l_edge = self.edge_loss(pred, target)

        total_loss = self.alpha * l_pix + self.beta * l_freq + self.gamma * l_edge
        return total_loss, l_pix, l_freq, l_edge


# ==============================================================================
# 2. 评估指标 (Evaluation Metrics) - GPU 加速版
# ==============================================================================

def calculate_psnr_torch(img1, img2, data_range=1.0):
    """
    PyTorch 版 PSNR，支持 GPU 计算，支持反向传播(虽然通常只用于Eval)。
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(100.0).to(img1.device)
    return 20 * torch.log10(data_range / torch.sqrt(mse))


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def calculate_ssim_torch(img1, img2, window_size=11, channel=1, data_range=1.0):
    """
    PyTorch 版 SSIM (结构相似性)。
    """
    # 动态获取 window，如果是第一次调用或者 device 变了，需要重新创建
    # 为了简化，这里每次创建（开销很小），实际工程可以缓存
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# ==============================================================================
# 3. 原始辅助函数 (保留并优化)
# ==============================================================================

def my_zero_pad(img, block_size=32):
    """
    对 numpy 图像进行 padding，使其能被 block_size 整除
    """
    old_h, old_w = img.shape
    delta_h = (block_size - np.mod(old_h, block_size)) % block_size
    delta_w = (block_size - np.mod(old_w, block_size)) % block_size

    img_pad = np.concatenate((img, np.zeros([old_h, delta_w])), axis=1)
    img_pad = np.concatenate((img_pad, np.zeros([delta_h, old_w + delta_w])), axis=0)

    new_h, new_w = img_pad.shape
    return img, old_h, old_w, img_pad, new_h, new_w


# 兼容旧代码的 numpy PSNR，但建议在 Training Loop 中使用 calculate_psnr_torch
def psnr_numpy(img1, img2, data_range=255.0):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(data_range / math.sqrt(mse))


# 数据增强辅助 (保持不变)
def data_augment(img, mode=0):
    # mode: 0-7
    if mode == 0: return img
    rot = mode // 2
    flip = mode % 2
    if flip:
        img = np.flip(img, axis=1)  # Horizontal flip
    img = np.rot90(img, k=rot, axes=(0, 1))  # Rotate
    return img
