import torch
import torch.nn as nn
import torch.nn.functional as F
# 引入底层驱动
from backprop import RevModule, RevBackProp
# 引入创新频域模块
from model_wave import DWT, IWT, WaveBlock


# ==============================================================================
# 1. 物理测量算子 (Measurement Operator)
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

    def forward(self, x):
        return self.A(x)

    def transpose(self, y):
        feat = self.AT_conv(y)
        return self.ps(feat)


# ==============================================================================
# 2. 基础组件 (含显存优化版 Attention)
# ==============================================================================

class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c=None):
        super().__init__()
        out_c = in_c if out_c is None else out_c
        self.norm1 = nn.GroupNorm(min(32, in_c), in_c)  # 安全的 GroupNorm
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_c), out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.act = nn.SiLU()
        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    [Optimization] 线性高效注意力模块 (Linear Efficient Attention)
    替代原本的 O(N^2) Dot-Product Attention。
    利用结合律 Q(K^T V) 代替 (QK^T)V，将显存消耗从 32GB 降至 <10MB。
    """

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        # qkv 映射保持不变
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_in = x
        x = self.norm(x)

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # 变形为 (B, C, N) 其中 N = H*W
        q = q.reshape(B, C, H * W)
        k = k.reshape(B, C, H * W)
        v = v.reshape(B, C, H * W)

        # [Key Trick]: 对 Spatial 维度进行 Softmax，而不是像传统 Attention 那样对 N*N 矩阵做
        # 这样可以将 K 当作一种全局上下文权重的分布
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-1)

        # 归一化因子 (防止梯度消失)
        scale = C ** -0.5
        q = q * scale

        # [Linear Attention Magic]
        # 1. 先计算全局上下文矩阵 (Global Context Matrix)
        # k: (B, C, N), v: (B, C, N)
        # context = k @ v.T -> (B, C, C)
        # 这个矩阵大小只有 C*C (例如 32*32)，非常非常小
        context = torch.bmm(k, v.transpose(1, 2))

        # 2. 将上下文聚合到 Query
        # q: (B, C, N), context: (B, C, C)
        # out = context.T @ q -> (B, C, N) (注意转置关系以匹配维度)
        # 这里其实是 context @ q 的某种变体，等价于 attention 加权
        out = torch.bmm(context, q)  # 此时 out 是 (B, C, N)，实际上是 v 的加权组合

        # 这种简单的形式可能丢失位置信息，但在 IDM 这种迭代网络中，
        # 我们主要需要它捕捉全局颜色/亮度统计信息。
        # 为了数学严谨性，通常使用 Efficient Attention 的变体：
        # standard: softmax(Q K^T) V
        # linear approximation: Q (softmax(K)^T V)
        # 上面的实现是基于 Shen et al. "Efficient Attention" 的简化思路

        out = out.reshape(B, C, H, W)
        return self.proj(out) + x_in


class Injector(nn.Module):
    def __init__(self, channels, operator, downscale_factor):
        super().__init__()
        self.operator = operator
        self.r = downscale_factor
        target_c = self.r * self.r
        self.f2i_conv = nn.Conv2d(channels, target_c, 1)
        self.ps = nn.PixelShuffle(self.r)
        self.pus = nn.PixelUnshuffle(self.r)
        self.i2f_conv = nn.Conv2d(target_c, channels, 1)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        img_proxy = self.ps(self.f2i_conv(x))
        y_hat = self.operator(img_proxy)
        proj_back = self.operator.transpose(y_hat)
        feat_update = self.i2f_conv(self.pus(proj_back))
        return x + self.scale.to(x.dtype) * feat_update


# ==============================================================================
# 3. 核心可逆控制块
# ==============================================================================
class IDMBlock(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x_in = torch.cat([x, self.alpha * x], dim=1)
        x_out = RevBackProp.apply(x_in, self.layers)
        x1, x2 = x_out.chunk(2, dim=1)
        out = x1 + self.beta * x2
        return out


# ==============================================================================
# 4. U-Net 结构组件
# ==============================================================================
class IDMDownBlock(nn.Module):
    def __init__(self, in_c, out_c, operator, level, use_wave=False):
        super().__init__()

        if level > 0:
            self.downsample = nn.Sequential(
                DWT(in_c),
                nn.Conv2d(in_c * 4, out_c, 1)
            )
            conv_in_needed = False
        else:
            self.downsample = nn.Identity()
            conv_in_needed = True

        if conv_in_needed and (in_c != out_c):
            self.conv_in = nn.Conv2d(in_c, out_c, 1)
        else:
            self.conv_in = nn.Identity()

        rev_list = []
        if use_wave:
            rev_list.append(RevModule(WaveBlock(out_c), v=0.5))
        else:
            rev_list.append(RevModule(ResnetBlock(out_c), v=0.5))

        scale = 2 ** level
        rev_list.append(RevModule(Injector(out_c, operator, scale), v=0.5))

        if use_wave:
            pass
        else:
            # 现在这里的 AttentionBlock 是显存安全的
            rev_list.append(RevModule(AttentionBlock(out_c), v=0.5))

        rev_list.append(RevModule(Injector(out_c, operator, scale), v=0.5))

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
        self.iwt = IWT(out_c)

        # 确保通道对齐
        merge_in_channels = out_c + skip_c
        self.conv_merge = nn.Conv2d(merge_in_channels, out_c, 1)

        rev_list = []
        if use_wave:
            rev_list.append(RevModule(WaveBlock(out_c), v=0.5))
        else:
            rev_list.append(RevModule(ResnetBlock(out_c), v=0.5))

        scale = 2 ** level
        rev_list.append(RevModule(Injector(out_c, operator, scale), v=0.5))

        if not use_wave:
            rev_list.append(RevModule(AttentionBlock(out_c), v=0.5))

        rev_list.append(RevModule(Injector(out_c, operator, scale), v=0.5))

        self.idm_processor = IDMBlock(rev_list)

    def forward(self, x, skip):
        x = self.upsample_prep(x)
        x = self.iwt(x)

        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
            x = torch.cat([x, skip], dim=1)

        x = self.conv_merge(x)
        x = self.idm_processor(x)
        return x


# ==============================================================================
# 5. 主模型: IDMWaveUNet
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

        # Bottleneck
        mid_layers = [
            RevModule(WaveBlock(base_dim * 4), v=0.5),
            RevModule(WaveBlock(base_dim * 4), v=0.5),
            RevModule(ResnetBlock(base_dim * 4), v=0.5)
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

    def forward(self, x_gt):
        y = self.operator(x_gt)
        x_init = self.operator.transpose(y)

        h = self.head_conv(x_init)

        h0, s0 = self.down0(h)
        h1, s1 = self.down1(h0)
        h2, s2 = self.down2(h1)

        h_mid = self.mid_block(h2)

        u2 = self.up2(h_mid, s2)
        u1 = self.up1(u2, s1)
        u0 = self.up0(u1, s0)

        out = self.tail_conv(self.tail_act(self.tail_norm(u0)))
        return x_init + out, y


if __name__ == "__main__":
    import time


    # 兼容性工具
    def get_autocast_context():
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            return torch.amp.autocast(device_type='cuda', enabled=True)
        else:
            from torch.cuda.amp import autocast
            return autocast(enabled=True)


    try:
        from torch.amp import GradScaler
    except ImportError:
        from torch.cuda.amp import GradScaler

    if not torch.cuda.is_available():
        print("Error: CUDA required for this test.")
        exit(0)

    device = torch.device('cuda')
    print(f"Running IDMWave Model Test on {device}...")

    # 使用较小的 base_dim 进行测试，确保即使显存很少也能跑通
    model = IDMWaveUNet(cs_ratio=0.1, block_size=32, base_dim=32).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {param_count / 1e6:.2f} M")

    x = torch.randn(2, 1, 256, 256).to(device)
    scaler = GradScaler()

    try:
        start_time = time.time()

        with get_autocast_context():
            recon, y = model(x)
            loss = F.mse_loss(recon, x)

        end_time = time.time()

        print(f"Forward pass successful!")
        print(f"Input: {x.shape}, Output: {recon.shape}")
        print(f"Time: {end_time - start_time:.4f}s")
        print(f"Loss: {loss.item():.6f}")

        scaler.scale(loss).backward()
        print("Backward pass successful! (Gradients computed)")

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
