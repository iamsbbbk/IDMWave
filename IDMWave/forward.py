import torch
import torch.nn as nn
import torch.utils.checkpoint
from backprop import RevModule, RevBackProp


# ==============================================================================
# 1. 基础工具与 Checkpoint 封装
# ==============================================================================

def run_checkpoint(module, x):
    """
    对普通模块使用 Checkpoint 技术，以时间换空间。
    如果 requires_grad 为 False (推理模式)，则直接运行。
    """
    if x.requires_grad:
        # custom_forward 包装器用于适应 checkpoint 的接口
        def custom_forward(x_in):
            return module(x_in)

        return torch.utils.checkpoint.checkpoint(custom_forward, x, use_reentrant=False)
    else:
        return module(x)


class DWT_2D(nn.Module):
    """ Haar 离散小波变换: (B, C, H, W) -> (B, 4C, H/2, W/2) """

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1, x2 = x01[:, :, :, 0::2], x02[:, :, :, 0::2]
        x3, x4 = x01[:, :, :, 1::2], x02[:, :, :, 1::2]
        return torch.cat([x1 + x2 + x3 + x4, -x1 - x2 + x3 + x4, -x1 + x2 - x3 + x4, x1 - x2 - x3 + x4], dim=1)


class IWT_2D(nn.Module):
    """ Haar 逆离散小波变换: (B, 4C, H/2, W/2) -> (B, C, H, W) """

    def forward(self, x):
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
        out_channel = in_channel // 4
        x1, x2, x3, x4 = x.chunk(4, dim=1)
        x1, x2, x3, x4 = x1 / 2, x2 / 2, x3 / 2, x4 / 2

        h = torch.zeros([in_batch, out_channel, in_height * r, in_width * r], device=x.device, dtype=x.dtype)
        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
        return h


# ==============================================================================
# 2. 可逆网络构建块 (The Body)
# ==============================================================================

class ResnetBody(nn.Module):
    """
    RevModule 内部的非线性变换 F(x)。
    使用 GroupNorm + SiLU + Conv 结构，更加稳定。
    """

    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()  # SiLU (Swish) 通常比 ReLU 表现更好

    def forward(self, x):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h


# ==============================================================================
# 3. 核心逻辑：IDM 风格的可逆 Block
# ==============================================================================

class IDMWaveBlock(nn.Module):
    """
    实现论文逻辑的核心块：
    Input -> Expand([x, alpha*x]) -> RevBackProp -> Merge(x1 + beta*x2) -> Output
    """

    def __init__(self, channels, num_layers=2, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha  # input_help_scale_factor
        self.beta = beta  # merge_scale_factor

        # 构建 RevModule 列表
        # 注意：因为我们做了 Expand，通道数翻倍，所以 RevModule 内部处理的单边通道数就是原始 channels
        self.rev_layers = nn.ModuleList([
            RevModule(body=ResnetBody(channels), v=0.5)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # 1. Expand: 构造可逆所需的双通道输入
        # x: (B, C, H, W) -> (B, 2C, H, W)
        x_in = torch.cat([x, self.alpha * x], dim=1)

        # 2. Reversible Propagation
        # 使用 apply 调用自定义 Function
        x_rev = RevBackProp.apply(x_in, self.rev_layers)

        # 3. Merge: 融合双通道输出
        x1, x2 = x_rev.chunk(2, dim=1)
        out = x1 + self.beta * x2

        return out


# ==============================================================================
# 4. 下采样与上采样模块 (Down/Up Blocks)
# ==============================================================================

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2):
        super().__init__()
        # DWT 会将通道数 x4
        self.dwt = DWT_2D()

        # 通道调整: DWT后通道是 4*in_channels，我们需要映射到 out_channels
        self.adapter = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1)

        # IDM 风格的可逆处理块
        self.idm_block = IDMWaveBlock(out_channels, num_layers=num_layers)

    def forward(self, x):
        # 1. DWT 下采样 (Checkpointing 节省内存)
        x = run_checkpoint(self.dwt, x)

        # 2. 调整通道
        x = run_checkpoint(self.adapter, x)

        # 3. 可逆特征提取 (内部自含 RevBackProp)
        x = self.idm_block(x)

        # 返回 x 用于传递给下一层，同时返回 x 作为 skip connection
        return x, x


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, num_layers=2):
        super().__init__()
        # 1. 融合 Skip Connection 的层
        self.merge_conv = nn.Conv2d(in_channels + skip_channels, in_channels, kernel_size=1)

        # 2. IDM 风格的可逆处理块
        self.idm_block = IDMWaveBlock(in_channels, num_layers=num_layers)

        # 3. IWT 上采样准备
        # IWT 需要输入通道是 4 的倍数，且输出通道 = 输入 / 4
        # 所以我们需要先将通道映射到 out_channels * 4
        self.adapter = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1)
        self.iwt = IWT_2D()

    def forward(self, x, skip):
        # 1. 拼接 Skip Connection
        x = torch.cat([x, skip], dim=1)

        # 2. 融合通道
        x = run_checkpoint(self.merge_conv, x)

        # 3. 可逆特征提取
        x = self.idm_block(x)

        # 4. 调整通道以适应 IWT
        x = run_checkpoint(self.adapter, x)

        # 5. IWT 上采样
        x = run_checkpoint(self.iwt, x)

        return x


# ==============================================================================
# 5. IDMWave 主模型
# ==============================================================================

class IDMWave(nn.Module):
    def __init__(self, cs_ratio=0.10, base_dim=32, layers_per_block=2):
        super().__init__()

        # --- 采样部分 (CS Sampling) ---
        # 假设输入是单通道灰度图
        self.cs_ratio = cs_ratio
        block_size = 32
        self.M = int(block_size * block_size * cs_ratio)

        # 模拟采样: (B, 1, H, W) -> (B, M, H/32, W/32)
        self.sampling = nn.Conv2d(1, self.M, kernel_size=block_size, stride=block_size, bias=False)

        # --- 初始化部分 (Initialization) ---
        # (B, M, H/32, W/32) -> (B, base_dim, H, W)
        self.init_conv = nn.Conv2d(self.M, block_size * block_size, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(block_size)  # 恢复分辨率
        self.feature_in = nn.Conv2d(1, base_dim, kernel_size=3, padding=1)

        # --- Encoder (Downsampling Path) ---
        # Level 1: base -> 2*base (Size / 2)
        self.down1 = DownBlock(base_dim, base_dim * 2, num_layers=layers_per_block)
        # Level 2: 2*base -> 4*base (Size / 4)
        self.down2 = DownBlock(base_dim * 2, base_dim * 4, num_layers=layers_per_block)

        # --- Bottleneck ---
        self.bottleneck = IDMWaveBlock(base_dim * 4, num_layers=layers_per_block)

        # --- Decoder (Upsampling Path) ---
        # Level 2: Input 4*base, Skip 4*base -> Output 2*base
        self.up2 = UpBlock(base_dim * 4, base_dim * 4, base_dim * 2, num_layers=layers_per_block)
        # Level 1: Input 2*base, Skip 2*base -> Output base
        self.up1 = UpBlock(base_dim * 2, base_dim * 2, base_dim, num_layers=layers_per_block)

        # --- Reconstruction ---
        self.tail = nn.Conv2d(base_dim, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (B, 1, H, W)

        # 1. CS Sampling
        y_measure = self.sampling(x)

        # 2. Initialization
        x_init = self.init_conv(y_measure)
        x_init = self.pixel_shuffle(x_init)  # (B, 1, H, W)
        f0 = self.feature_in(x_init)  # (B, base_dim, H, W)

        # 3. Encoder
        f1, skip1 = self.down1(f0)  # f1: (B, 2*base, H/2, W/2)
        f2, skip2 = self.down2(f1)  # f2: (B, 4*base, H/4, W/4)

        # 4. Bottleneck
        # 可以在 bottleneck 中加入注意力机制，这里保持纯 IDM Block
        f_mid = self.bottleneck(f2)

        # 5. Decoder
        u1 = self.up2(f_mid, skip2)  # u1: (B, 2*base, H/2, W/2)
        u0 = self.up1(u1, skip1)  # u0: (B, base, H, W)

        # 6. Reconstruction
        out = self.tail(u0)

        # Residual Learning: 输出的是与 x_init 的残差
        return out + x_init


if __name__ == "__main__":
    # 测试代码强健性
    model = IDMWave(cs_ratio=0.1, base_dim=32).cuda()
    x = torch.randn(2, 1, 256, 256).cuda()

    # 打印显存占用测试
    try:
        out = model(x)
        print(f"Model Forward Successful.")
        print(f"Input: {x.shape}")
        print(f"Output: {out.shape}")

        loss = out.mean()
        loss.backward()
        print("Model Backward Successful.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

