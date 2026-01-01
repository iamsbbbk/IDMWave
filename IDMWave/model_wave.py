import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================================
# 1. 基础小波变换核 (DWT/IWT Kernel Factory)
# ==============================================================================

def dwt_init(weights):
    """
    初始化 Haar 小波核。
    这将创建用于提取 LL, LH, HL, HH 四个频带的卷积核。
    """
    # Haar Wavelet Basis
    # Low-pass filter (L): [1, 1] / 2
    # High-pass filter (H): [-1, 1] / 2

    # LL: L x L
    w0 = torch.tensor([1, 1], dtype=torch.float32)
    w1 = torch.tensor([1, 1], dtype=torch.float32)
    # LH: L x H
    w2 = torch.tensor([-1, 1], dtype=torch.float32)
    w3 = torch.tensor([1, 1], dtype=torch.float32)
    # HL: H x L
    w4 = torch.tensor([1, 1], dtype=torch.float32)
    w5 = torch.tensor([-1, 1], dtype=torch.float32)
    # HH: H x H
    w6 = torch.tensor([-1, 1], dtype=torch.float32)
    w7 = torch.tensor([-1, 1], dtype=torch.float32)

    # 组合成 2D 核
    # LL, LH, HL, HH
    weight_LL = torch.ger(w0, w1).unsqueeze(0).unsqueeze(0)
    weight_LH = torch.ger(w2, w3).unsqueeze(0).unsqueeze(0)
    weight_HL = torch.ger(w4, w5).unsqueeze(0).unsqueeze(0)
    weight_HH = torch.ger(w6, w7).unsqueeze(0).unsqueeze(0)

    # Stack: (4, 1, 2, 2)
    cat_weights = torch.cat([weight_LL, weight_LH, weight_HL, weight_HH], dim=0)

    # 赋值给传入的权重变量
    # 需要对每个输入通道进行处理，使用 group convolution 逻辑
    # weights shape 预期: (C_in * 4, 1, 2, 2)
    c_in = weights.shape[0] // 4

    # 将标准核复制到所有通道
    # 结果 shape: (C_in * 4, 1, 2, 2)
    new_weights = cat_weights.repeat(c_in, 1, 1, 1)

    weights.data.copy_(new_weights * 0.5)  # 0.5 是 Haar 的归一化因子

    return weights


# ==============================================================================
# 2. 离散小波变换与逆变换层 (DWT & IWT)
#    创新点: 替代传统的 Pool 和 Upsample，实现无损特征缩放
# ==============================================================================

class DWT(nn.Module):
    """
    离散小波变换层 (Downsampling)
    Input: (B, C, H, W) -> Output: (B, 4*C, H/2, W/2)
    通道顺序: [LL_c1, LL_c2..., LH_c1..., HL_c1..., HH_c1...]
    """

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.requires_grad = False  # Haar 小波核通常不需要训练

        # 定义卷积权重
        self.register_buffer('weight', torch.zeros(in_channels * 4, 1, 2, 2))
        dwt_init(self.weight)

    def forward(self, x):
        return F.conv2d(x, self.weight, stride=2, groups=self.in_channels)


class IWT(nn.Module):
    """
    逆离散小波变换层 (Upsampling)
    Input: (B, 4*C, H/2, W/2) -> Output: (B, C, H, W)
    """

    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.register_buffer('weight', torch.zeros(out_channels * 4, 1, 2, 2))
        dwt_init(self.weight)

    def forward(self, x):
        # IWT 等价于转置卷积
        return F.conv_transpose2d(x, self.weight, stride=2, groups=self.out_channels)


# ==============================================================================
# 3. 可微分软阈值门控 (DSTG)
#    创新点: 模仿传统压缩感知的去噪步骤，但阈值是自适应学习的
# ==============================================================================

class LearnableSpectralGating(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 为每个通道学习一个阈值 lambda
        self.threshold = nn.Parameter(torch.zeros(1, channels, 1, 1) + 0.01)
        self.act = nn.SiLU()
        # 通道注意力，决定哪些频带更重要
        self.channel_scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.SiLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. 软阈值去噪公式: sign(x) * max(|x| - lambda, 0)
        # 这是一个显式的稀疏化操作，非常适合去除高频噪声
        thresh = torch.abs(self.threshold)
        x_abs = torch.abs(x)

        # 软阈值操作
        x_clean = torch.sign(x) * torch.relu(x_abs - thresh)

        # 2. 结合通道注意力进行重加权
        scale = self.channel_scale(x_clean)
        return x_clean * scale


# ==============================================================================
# 4. 鬼影小波注意力 (Ghost Wavelet Attention - GWA)
#    创新点: 在频域分离特征，低频做全局感知，高频做局部细节增强
# ==============================================================================

class GhostWaveletAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dwt = DWT(in_channels)
        self.iwt = IWT(in_channels)

        # DWT 后通道数变为 4 倍: LL, LH, HL, HH
        # 我们将 LL (低频) 和 High-Freq (LH+HL+HH) 分开处理

        # 1. 低频处理分支 (LL): 捕捉大致结构，使用大感受野
        self.ll_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),  # Depthwise
            nn.Conv2d(in_channels, in_channels, 1),  # Pointwise
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

        # 2. 高频处理分支 (Concatenated High Freqs): 捕捉纹理，使用 DSTG
        # 高频通道数为 3 * in_channels
        self.high_process = LearnableSpectralGating(in_channels * 3)

        # 3. 融合卷积
        self.fusion = nn.Conv2d(in_channels * 4, in_channels * 4, 1)

    def forward(self, x):
        # Time-Frequency Decomposition
        # x: (B, C, H, W) -> (B, 4C, H/2, W/2)
        freq_components = self.dwt(x)

        # Split into [LL, LH, HL, HH]
        # 注意: DWT 的输出排列是交织的，我们需要 reshape 或 split
        # DWT output group logic: [LL_c1, LH_c1, HL_c1, HH_c1, LL_c2...]
        # 这比较难直接 split，所以我们采用 view 变形

        B, C4, H, W = freq_components.shape
        C = C4 // 4

        # Reshape to (B, C, 4, H, W)
        freq_reshaped = freq_components.view(B, C, 4, H, W)

        # Extract components
        x_ll = freq_reshaped[:, :, 0, :, :]  # (B, C, H, W)
        x_high = freq_reshaped[:, :, 1:, :, :]  # (B, C, 3, H, W)

        # Flatten high freq for processing: (B, 3C, H, W)
        x_high_flat = x_high.reshape(B, C * 3, H, W)

        # === Process ===
        # Branch 1: Low Frequency (Structure)
        out_ll = self.ll_conv(x_ll) + x_ll  # Residual

        # Branch 2: High Frequency (Texture/Noise)
        out_high = self.high_process(x_high_flat) + x_high_flat  # Residual

        # === Recombine ===
        # Pack back to DWT format
        # Need to interleave them back: LL, LH, HL, HH
        out_ll_expanded = out_ll.unsqueeze(2)  # (B, C, 1, H, W)
        out_high_reshaped = out_high.view(B, C, 3, H, W)

        out_merged = torch.cat([out_ll_expanded, out_high_reshaped], dim=2)  # (B, C, 4, H, W)
        out_ready_for_iwt = out_merged.view(B, C4, H, W)

        # Channel Fusion before IWT
        out_fused = self.fusion(out_ready_for_iwt)

        # Inverse Transform
        return self.iwt(out_fused)


# ==============================================================================
# 5. 高级波浪模块 (WaveBlock) - 可直接用于 IDM Layer
# ==============================================================================

class WaveBlock(nn.Module):
    """
    IDMWave 专属的构建块，替代 ResnetBlock。
    集成 GhostWaveletAttention，具有超强的多尺度特征提取能力。
    """

    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, dim)
        self.attn = GhostWaveletAttention(dim)

        self.norm2 = nn.GroupNorm(32, dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, int(dim * mlp_ratio), 1),
            nn.SiLU(),
            nn.Conv2d(int(dim * mlp_ratio), dim, 1)
        )

        # Layer Scale (仿照 ConvNeXt)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim, 1, 1), requires_grad=True)

    def forward(self, x):
        # 1. Wavelet Attention Path
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.gamma * x

        # 2. MLP Path
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        return shortcut + x  # 此处不需要 gamma，保持标准 ResNet 风格


# ==============================================================================
# 单元测试
# ==============================================================================
if __name__ == '__main__':
    # Test DWT/IWT reversibility
    x = torch.randn(2, 64, 128, 128)
    dwt = DWT(64)
    iwt = IWT(64)

    y = dwt(x)
    x_rec = iwt(y)

    print(f"Input: {x.shape}")
    print(f"DWT Output: {y.shape}")
    print(f"Reconstruction Error: {(x - x_rec).abs().max().item():.6f}")

    # Test GhostWaveletAttention
    gwa = GhostWaveletAttention(64)
    out = gwa(x)
    print(f"GWA Output: {out.shape}")

    # Test WaveBlock
    block = WaveBlock(64)
    out_block = block(x)
    print(f"WaveBlock Output: {out_block.shape}")
