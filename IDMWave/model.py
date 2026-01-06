import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# 引入底层驱动 (请确保 backprop.py 在同一目录下)
from backprop import RevModule, RevBackProp


# ==============================================================================
# 1. 增强型基础组件 (RCAB & Learnable Wavelet)
# ==============================================================================

class ChannelAttention(nn.Module):
    """
    通道注意力机制：让网络学会“看哪里”
    """

    def __init__(self, num_feat, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 确保中间通道数至少为1
        mid_channels = max(1, num_feat // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(num_feat, mid_channels, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_feat, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class RCAB(nn.Module):
    """
    [Upgrade] 替代 ResnetBlock。
    包含 Residual Channel Attention，是提升 PSNR 的关键。
    """

    def __init__(self, in_c, out_c=None, reduction=16):
        super().__init__()
        out_c = in_c if out_c is None else out_c

        self.body = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            ChannelAttention(out_c, reduction=reduction)
        )
        # 如果通道数改变，需要 shortcut 投影
        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        res = self.body(x)
        return res + self.shortcut(x)


def init_wavelet_filters():
    """初始化 Haar 小波核"""
    w_ll = np.array([[0.5, 0.5], [0.5, 0.5]])
    w_lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
    w_hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
    w_hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
    weights = np.stack([w_ll, w_lh, w_hl, w_hh], axis=0)  # (4, 2, 2)
    weights = np.expand_dims(weights, axis=1)  # (4, 1, 2, 2)
    return torch.from_numpy(weights).float()


class LearnableDWT(nn.Module):
    """[Upgrade] 可学习的小波分解"""

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        # 初始化为 Haar，但允许梯度更新 (requires_grad=True)
        base_filter = init_wavelet_filters()
        self.filters = nn.Parameter(base_filter.repeat(in_channels, 1, 1, 1), requires_grad=True)

    def forward(self, x):
        # Depthwise Conv 实现每个通道独立的小波变换
        return F.conv2d(x, self.filters, stride=2, groups=self.in_channels)


class LearnableIWT(nn.Module):
    """[Upgrade] 可学习的小波重构"""

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        base_filter = init_wavelet_filters()
        self.filters = nn.Parameter(base_filter.repeat(in_channels, 1, 1, 1), requires_grad=True)

    def forward(self, x):
        return F.conv_transpose2d(x, self.filters, stride=2, groups=self.in_channels)


class WaveBlock(nn.Module):
    """在小波域进行处理的模块"""

    def __init__(self, in_c):
        super().__init__()
        # 小波分解后通道数变4倍，处理后再变回来
        # 这里我们只处理高频部分或者整体处理
        self.dwt = LearnableDWT(in_c)
        self.conv = RCAB(in_c * 4)  # 在频域使用 RCAB
        self.iwt = LearnableIWT(in_c)

    def forward(self, x):
        skip = x
        x = self.dwt(x)
        x = self.conv(x)
        x = self.iwt(x)
        return x + skip


# ==============================================================================
# 2. 物理测量算子
# ==============================================================================
class MeasurementOperator(nn.Module):
    def __init__(self, cs_ratio, block_size=32):
        super().__init__()
        self.block_size = block_size
        n = block_size * block_size
        m = int(n * cs_ratio)
        self.A = nn.Conv2d(1, m, kernel_size=block_size, stride=block_size, bias=False)
        self.AT_conv = nn.Conv2d(m, n, kernel_size=1, bias=False)
        self.ps = nn.PixelShuffle(block_size)
        nn.init.orthogonal_(self.A.weight)
        # 冻结测量矩阵通常能稳定训练，或者设为 True 联合优化
        self.A.weight.requires_grad = True

    def forward(self, x):
        return self.A(x)

    def transpose(self, y):
        feat = self.AT_conv(y)
        return self.ps(feat)


# ==============================================================================
# 3. 注入器与 Attention (优化版)
# ==============================================================================

class ChannelWiseInjector(nn.Module):
    """
    [Upgrade] 通道级注入器。
    相比原来的标量 scale，这里每个通道有独立的步长，优化更精细。
    """

    def __init__(self, channels, operator, downscale_factor):
        super().__init__()
        self.operator = operator
        self.r = downscale_factor
        target_c = self.r * self.r

        self.f2i_conv = nn.Conv2d(channels, target_c, 1)
        self.ps = nn.PixelShuffle(self.r)
        self.pus = nn.PixelUnshuffle(self.r)
        self.i2f_conv = nn.Conv2d(target_c, channels, 1)

        # [Upgrade] 这里的 scale 是 (1, C, 1, 1)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # 1. 映射到图像空间
        img_proxy = self.ps(self.f2i_conv(x))
        # 2. 物理一致性计算: A^T(y - Ax)
        y_hat = self.operator(img_proxy)
        proj_back = self.operator.transpose(y_hat)
        # 3. 映射回特征空间
        feat_update = self.i2f_conv(self.pus(proj_back))
        # 4. 梯度更新
        return x + self.scale * feat_update


class EfficientAttention(nn.Module):
    """保持原有的显存高效 Attention，但加入 GroupNorm 稳定训练"""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_in = x
        x = self.norm(x)

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(B, C, H * W)
        k = k.reshape(B, C, H * W)
        v = v.reshape(B, C, H * W)

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-1)
        q = q * (C ** -0.5)

        context = torch.bmm(k, v.transpose(1, 2))
        out = torch.bmm(context, q)

        out = out.reshape(B, C, H, W)
        return self.proj(out) + x_in


# ==============================================================================
# 4. 核心 IDM 结构
# ==============================================================================
class IDMBlock(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 简单的输入增强
        x_in = torch.cat([x, self.alpha * x], dim=1)
        # 可逆反向传播
        x_out = RevBackProp.apply(x_in, self.layers)
        x1, x2 = x_out.chunk(2, dim=1)
        return x1 + self.beta * x2


class IDMDownBlock(nn.Module):
    def __init__(self, in_c, out_c, operator, level, use_wave=False):
        super().__init__()

        # 下采样策略
        if level > 0:
            self.downsample = nn.Sequential(
                LearnableDWT(in_c),  # 使用可学习小波下采样
                nn.Conv2d(in_c * 4, out_c, 1)  # 降维
            )
            conv_in_needed = False
        else:
            self.downsample = nn.Identity()
            conv_in_needed = True

        self.conv_in = nn.Conv2d(in_c, out_c, 1) if (conv_in_needed and in_c != out_c) else nn.Identity()

        rev_list = []
        scale = 2 ** level

        # Block 1: Feature Extraction
        if use_wave:
            rev_list.append(RevModule(WaveBlock(out_c), v=0.5))
        else:
            rev_list.append(RevModule(RCAB(out_c), v=0.5))  # 使用 RCAB

        # Block 2: Physics Injection
        rev_list.append(RevModule(ChannelWiseInjector(out_c, operator, scale), v=0.5))

        # Block 3: Global Context
        rev_list.append(RevModule(EfficientAttention(out_c), v=0.5))

        # Block 4: Physics Injection Again
        rev_list.append(RevModule(ChannelWiseInjector(out_c, operator, scale), v=0.5))

        self.idm_processor = IDMBlock(rev_list)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv_in(x)
        x = self.idm_processor(x)
        return x, x


class IDMUpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c, operator, level, use_wave=False):
        super().__init__()

        self.upsample_prep = nn.Conv2d(in_c, out_c * 4, 1)
        self.iwt = LearnableIWT(out_c)  # 使用可学习小波上采样

        self.conv_merge = nn.Conv2d(out_c + skip_c, out_c, 1)

        rev_list = []
        scale = 2 ** level

        if use_wave:
            rev_list.append(RevModule(WaveBlock(out_c), v=0.5))
        else:
            rev_list.append(RevModule(RCAB(out_c), v=0.5))

        rev_list.append(RevModule(ChannelWiseInjector(out_c, operator, scale), v=0.5))

        if not use_wave:
            rev_list.append(RevModule(EfficientAttention(out_c), v=0.5))

        rev_list.append(RevModule(ChannelWiseInjector(out_c, operator, scale), v=0.5))

        self.idm_processor = IDMBlock(rev_list)

    def forward(self, x, skip):
        x = self.upsample_prep(x)
        x = self.iwt(x)

        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)

        x = self.conv_merge(x)
        x = self.idm_processor(x)
        return x


# ==============================================================================
# 5. FFT 频域模块 (Bottleneck 增强)
# ==============================================================================
class FFTBlock(nn.Module):
    """
    在 Bottleneck 处使用，捕捉全局频率信息。
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels * 2, 1)
        )

    def forward(self, x):
        # Real FFT
        # x: (B, C, H, W)
        x_fft = torch.fft.rfft2(x, norm='backward')
        # Stack real and imag parts: (B, C, H, W/2+1) -> (B, 2C, H, W/2+1)
        x_real = x_fft.real
        x_imag = x_fft.imag
        x_cat = torch.cat([x_real, x_imag], dim=1)

        # Process in freq domain
        x_cat = self.conv(x_cat)

        # Restore
        x_real, x_imag = x_cat.chunk(2, dim=1)
        x_fft_new = torch.complex(x_real, x_imag)
        x_out = torch.fft.irfft2(x_fft_new, s=x.shape[2:], norm='backward')

        return x + x_out


# ==============================================================================
# 6. 主模型: IDMWaveUNet (Refined)
# ==============================================================================
class IDMWaveUNet(nn.Module):
    def __init__(self, cs_ratio=0.1, block_size=32, base_dim=64):
        super().__init__()

        self.operator = MeasurementOperator(cs_ratio, block_size)
        self.head_conv = nn.Conv2d(1, base_dim, 3, padding=1)

        # Encoder
        self.down0 = IDMDownBlock(base_dim, base_dim, self.operator, level=0, use_wave=False)
        self.down1 = IDMDownBlock(base_dim, base_dim * 2, self.operator, level=1, use_wave=True)
        self.down2 = IDMDownBlock(base_dim * 2, base_dim * 4, self.operator, level=2, use_wave=True)

        # Bottleneck [Enhanced with FFT]
        # 混合使用 WaveBlock, RCAB 和 FFTBlock
        mid_layers = [
            RevModule(WaveBlock(base_dim * 4), v=0.5),
            RevModule(FFTBlock(base_dim * 4), v=0.5),  # 新增：FFT 全局感知
            RevModule(RCAB(base_dim * 4), v=0.5)  # 替换了原来的 ResnetBlock
        ]
        self.mid_block = IDMBlock(mid_layers)

        # Decoder
        self.up2 = IDMUpBlock(in_c=base_dim * 4, skip_c=base_dim * 4, out_c=base_dim * 2,
                              operator=self.operator, level=2, use_wave=True)
        self.up1 = IDMUpBlock(in_c=base_dim * 2, skip_c=base_dim * 2, out_c=base_dim,
                              operator=self.operator, level=1, use_wave=True)
        self.up0 = IDMUpBlock(in_c=base_dim, skip_c=base_dim, out_c=base_dim,
                              operator=self.operator, level=0, use_wave=False)

        self.tail_norm = nn.GroupNorm(min(32, base_dim), base_dim)
        self.tail_act = nn.SiLU()
        self.tail_conv = nn.Conv2d(base_dim, 1, 3, padding=1)

        # 最后的精修模块 (Refine Module)
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            RCAB(16, reduction=4),
            nn.Conv2d(16, 1, 3, 1, 1)
        )

    def forward(self, x_gt):
        # 1. 采样与初始重建
        y = self.operator(x_gt)
        x_init = self.operator.transpose(y)

        # 2. Deep Feature Extraction
        h = self.head_conv(x_init)

        # Encoder
        h0, s0 = self.down0(h)
        h1, s1 = self.down1(h0)
        h2, s2 = self.down2(h1)

        # Bottleneck
        h_mid = self.mid_block(h2)

        # Decoder
        u2 = self.up2(h_mid, s2)
        u1 = self.up1(u2, s1)
        u0 = self.up0(u1, s0)

        # 3. 输出
        res = self.tail_conv(self.tail_act(self.tail_norm(u0)))
        x_pre = x_init + res

        # 4. 额外的精修步骤 (Optional, but helps PSNR)
        x_final = self.refine(x_pre) + x_pre

        return x_final, y


# ==============================================================================
# 测试代码
# ==============================================================================
if __name__ == "__main__":
    import time

    if not torch.cuda.is_available():
        print("CUDA required for test.")
        exit()

    device = torch.device('cuda')
    # 使用 64 dim 以匹配训练设置
    model = IDMWaveUNet(cs_ratio=0.1, block_size=32, base_dim=64).to(device)

    print("Model initialized.")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {param_count / 1e6:.2f} M")

    x = torch.randn(2, 1, 256, 256).to(device)

    # 测试前向
    try:
        out, y = model(x)
        print(f"Forward success. Output: {out.shape}")

        # 测试反向 (验证 RevModule 是否正常)
        loss = out.mean()
        loss.backward()
        print("Backward success.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
