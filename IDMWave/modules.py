import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarWavelet(nn.Module):
    """
    Haar 小波变换模块：用于显式提取图像的频率信息。
    输入: (B, 1, H, W)
    输出: (B, 4, H, W) - 包含 LL(低频), LH(垂直), HL(水平), HH(对角)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (B, 1, H, W)
        # 1. 计算平均值 (LL)
        ll = F.avg_pool2d(x, 2)

        # 2. 为了计算差值，使用切片操作 (比卷积更快且无参数)
        # 假设 H, W 是偶数，IDM 中通常满足此条件
        x_tl = x[:, :, 0::2, 0::2]  # Top-Left
        x_tr = x[:, :, 0::2, 1::2]  # Top-Right
        x_bl = x[:, :, 1::2, 0::2]  # Bottom-Left
        x_br = x[:, :, 1::2, 1::2]  # Bottom-Right

        # 3. 计算 Haar 小波分量
        # LH: 垂直边缘 (左+左下 - 右-右下 ?) -> 实际上是 (Top - Bottom)
        # 标准 Haar:
        # L = (x + y)/2, H = (x - y)/2
        # LL = L(L) = (TL+TR+BL+BR)/4 (已由 avg_pool 完成)
        # LH = L(H) = (TL-TR + BL-BR)/4 (垂直纹理)
        # HL = H(L) = (TL+TR - BL-BR)/4 (水平纹理)
        # HH = H(H) = (TL-TR - BL+BR)/4 (对角纹理)

        lh = (x_tl - x_tr + x_bl - x_br) / 4.0
        hl = (x_tl + x_tr - x_bl - x_br) / 4.0
        hh = (x_tl - x_tr - x_bl + x_br) / 4.0

        # 4. 上采样回原尺寸以便拼接 (使用最近邻插值保持高频特征的锐度)
        upsample = lambda z: F.interpolate(z, scale_factor=2, mode='nearest')

        return torch.cat([upsample(ll), upsample(lh), upsample(hl), upsample(hh)], dim=1)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (通道注意力 + 空间注意力)
    将被嵌入到可逆块 (RevModule) 中。
    """

    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        # 1. 通道注意力 (Channel Attention)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels)
        )
        self.sigmoid = nn.Sigmoid()

        # 2. 空间注意力 (Spatial Attention)
        # 输入2通道 (Max+Avg)，输出1通道权重
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.size()

        # --- Channel Attention ---
        avg_pool = F.avg_pool2d(x, (h, w), stride=(h, w))
        max_pool = F.max_pool2d(x, (h, w), stride=(h, w))
        channel_weight = self.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        channel_weight = channel_weight.view(b, c, 1, 1)
        x = x * channel_weight

        # --- Spatial Attention ---
        # 在通道维度上做 Max 和 Mean
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_out = torch.mean(x, dim=1, keepdim=True)
        spatial_in = torch.cat([max_out, mean_out], dim=1)
        spatial_weight = self.sigmoid(self.spatial_conv(spatial_in))
        x = x * spatial_weight

        return x
