import torch
import torch.nn as nn
# 导入原代码中的类和全局变量
from model import Net, Injector, Step, RevModule
import model as original_model_module  # 用于访问全局变量
from modules import HaarWavelet, CBAM


class WaveletInjector(Injector):
    """
    改进点 3: 频率感知输入
    继承自原 Injector，额外注入 Haar 小波特征。
    """

    def __init__(self, nf, r, T):
        super().__init__(nf, r, T)
        self.wavelet = HaarWavelet()

        # 重新定义 i2f (Image to Feature) 卷积层
        # 原代码输入通道为 3: [x, AT(Ax), ATy]
        # 新代码输入通道为 7: [x, AT(Ax), ATy] + [LL, LH, HL, HH]
        self.i2f = nn.ModuleList([nn.Sequential(
            nn.Conv2d(7, nf // (r * r), 1),  # 修改输入通道 3 -> 7
            nn.PixelUnshuffle(r),
        ) for _ in range(T)])

    def forward(self, x_in):
        # 获取全局时间步 t (兼容原代码风格)
        t = original_model_module.t

        # 1. 特征 -> 图像空间 (单通道)
        x = self.f2i[t - 1](x_in)

        # 2. 获取全局物理量
        ATy = original_model_module.ATy
        A = original_model_module.A
        AT = original_model_module.AT

        # 3. 计算测量代理的小波特征
        # ATy 是固定的测量反投影，包含这种先验对恢复高频细节很有帮助
        w_feat = self.wavelet(ATy)

        # 4. 拼接所有信息 (通道数 1+1+1+4 = 7)
        physics_info = torch.cat([x, AT(A(x)), ATy, w_feat], dim=1)

        # 5. 图像空间 -> 特征空间并残差连接
        return x_in + self.i2f[t - 1](physics_info)


class NetWAVE(Net):
    """
    IDM-WAVE 主模型
    包含：WaveletInjector, CBAM 注意力, 动态推理
    """

    def __init__(self, T, unet):
        # 1. 初始化父类 (构建基础 U-Net 结构)
        super().__init__(T, unet)

        # 2. 执行改进点 1 & 3: 替换 Injector 并插入 CBAM
        # 我们需要重新运行构建可逆模块的逻辑
        self.rebuild_components_with_wavelet_and_cbam(T)

    def rebuild_components_with_wavelet_and_cbam(self, T):
        # --- 下采样路径 (Down Blocks) ---
        # 替换 Injector 为 WaveletInjector
        self.unet.down_blocks[0].injectors = nn.ModuleList([WaveletInjector(320, 2, T) for _ in range(4)])
        self.unet.down_blocks[1].injectors = nn.ModuleList([WaveletInjector(640, 4, T) for _ in range(4)])

        for i in range(2):
            self.unet.down_blocks[i].rev_module_lists = nn.ModuleList([])
            for j in range(2):
                rev_list = nn.ModuleList([])
                # 原 ResNet
                if self.unet.down_blocks[i].resnets[j].in_channels == self.unet.down_blocks[i].resnets[j].out_channels:
                    rev_list.append(RevModule(self.unet.down_blocks[i].resnets[j]))

                # Wavelet Injector 1
                rev_list.append(RevModule(self.unet.down_blocks[i].injectors[2 * j]))

                # 原 Attention
                rev_list.append(RevModule(self.unet.down_blocks[i].attentions[j]))

                # === 改进点 1: 插入 CBAM ===
                # 注意：RevModule 会将输入切分为两半 (x1, x2)，body 只处理其中一半
                # 因此 CBAM 的输入通道数应该是总通道数的一半
                current_channels = self.unet.down_blocks[i].resnets[j].out_channels
                rev_list.append(RevModule(CBAM(current_channels // 2)))
                # =========================

                # Wavelet Injector 2
                rev_list.append(RevModule(self.unet.down_blocks[i].injectors[2 * j + 1]))

                self.unet.down_blocks[i].rev_module_lists.append(rev_list)

        # --- 上采样路径 (Up Blocks) ---
        # 替换 Injector 为 WaveletInjector
        self.unet.up_blocks[0].injectors = nn.ModuleList([WaveletInjector(640, 4, T) for _ in range(6)])
        self.unet.up_blocks[1].injectors = nn.ModuleList([WaveletInjector(320, 2, T) for _ in range(6)])

        for i in range(2):
            rev_full_list = nn.ModuleList([])
            for j in range(3):
                if j > 0:
                    rev_full_list.append(RevModule(self.unet.up_blocks[i].resnets[j]))

                rev_full_list.append(RevModule(self.unet.up_blocks[i].injectors[2 * j]))
                rev_full_list.append(RevModule(self.unet.up_blocks[i].attentions[j]))

                # === 改进点 1: 插入 CBAM ===
                current_channels = self.unet.up_blocks[i].resnets[j].out_channels
                rev_full_list.append(RevModule(CBAM(current_channels // 2)))
                # =========================

                rev_full_list.append(RevModule(self.unet.up_blocks[i].injectors[2 * j + 1]))

            self.unet.up_blocks[i].rev_module_list = rev_full_list

    def infer_dynamic(self, y_, A_, AT_, tol=1e-4, min_steps=5):
        """
        改进点 4: 动态推理调度器
        不使用 RevBackProp (无需梯度)，手动循环 Step 并在收敛时提前退出。
        """
        # 设置全局变量 (兼容 Step 模块)
        original_model_module.y = y_
        original_model_module.A = A_
        original_model_module.AT = AT_
        original_model_module.unet = self.unet
        original_model_module.use_amp = False  # 推理关闭 AMP

        # 初始化 Alpha Bar
        alpha_bar = torch.cat([torch.ones(1, device=y_.device), self.alpha.cumprod(dim=0)])
        original_model_module.alpha_bar = alpha_bar

        # 初始化输入
        x = AT_(y_)
        original_model_module.ATy = x  # 供 WaveletInjector 使用

        # 初始 Scaling
        x = alpha_bar[-1].pow(0.5) * torch.cat([x, self.input_help_scale_factor * x], dim=1)

        # 分割为可逆输入
        x1, x2 = x.chunk(2, dim=1)

        # 手动执行循环
        x_prev = None

        # self.body 是 Step 模块的列表
        for i, step_module in enumerate(self.body):
            # 执行单步 (包含去噪 + 物理一致性 + DDIM)
            # Step 是 RevModule，可以直接调用
            x1, x2 = step_module(x1, x2)

            # --- 动态退出检查逻辑 ---
            if i >= min_steps:
                # 重建当前特征
                x_curr = torch.cat([x1, x2], dim=1)

                if x_prev is not None:
                    # 计算特征变化的相对范数
                    diff = torch.norm(x_curr - x_prev) / (torch.norm(x_prev) + 1e-8)

                    if diff < tol:
                        print(f"[Dynamic Inference] Converged at step {i + 1}/{len(self.body)}. Early exiting.")
                        break

                x_prev = x_curr
            # ----------------------

        # 合并结果
        x = torch.cat([x1, x2], dim=1)
        return x[:, :1] + self.merge_scale_factor * x[:, 1:]
