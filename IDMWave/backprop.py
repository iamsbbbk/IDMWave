import torch
import torch.nn as nn
import torch.autograd

# ==============================================================================
# 1. 基础设置与兼容性
# ==============================================================================
try:
    from torch.amp import custom_fwd, custom_bwd
except ImportError:
    from torch.cuda.amp import custom_fwd, custom_bwd


def get_cuda_rng_state():
    try:
        return torch.cuda.get_rng_state()
    except Exception:
        return None


def set_cuda_rng_state(state):
    if state is not None:
        try:
            torch.cuda.set_rng_state(state)
        except Exception:
            pass


# ==============================================================================
# 2. 核心可逆引擎 (RevBackProp) - Fixed Generator Bug
# ==============================================================================
class RevBackProp(torch.autograd.Function):
    """
    可逆反向传播引擎 (Robust Version)

    Fix Log:
    1. [CRITICAL] Added `modules = list(modules)` in forward.
       原因: @custom_fwd 装饰器可能会把参数变成 generator。
       我们必须将其显式转换为 list，否则 len() 会报错，且无法在 backward 中复用。
    """

    @staticmethod
    @custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, x, modules):
        # [核心修复] 强行转为列表
        # 无论传入的是 nn.ModuleList 还是被装饰器转换成的 generator，
        # 这里都将其固定为 list。确保拥有 len() 属性且内容不会被消耗掉。
        modules = list(modules)

        # 1. 拆分输入 (FP32)
        x1, x2 = x.chunk(2, dim=1)
        x1, x2 = x1.contiguous(), x2.contiguous()

        # 用于存储随机状态
        rng_states = []

        # 2. 前向传播
        for module in modules:
            # 记录随机状态
            rng_states.append(get_cuda_rng_state())

            # 计算: y1 = x1 + f(x2)
            out = module.body(x2)
            x1 = x1 + out

            # 交换通道
            x1, x2 = x2, x1

        # 3. 保存上下文
        y = torch.cat([x1, x2], dim=1)

        # 保存 modules 列表 (现在它是安全的 list) 和 随机状态
        ctx.modules = modules
        ctx.rng_states = rng_states
        ctx.save_for_backward(y)

        return y

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        # 1. 恢复上下文
        y = ctx.saved_tensors[0]
        modules = ctx.modules
        rng_states = ctx.rng_states

        # 2. 准备数据
        y1, y2 = y.chunk(2, dim=1)
        y1, y2 = y1.contiguous(), y2.contiguous()

        dy1, dy2 = grad_output.chunk(2, dim=1)
        dy1, dy2 = dy1.contiguous(), dy2.contiguous()

        # 3. 逆向遍历
        # 此时 modules 是 list，len(modules) 绝对安全
        for i in range(len(modules) - 1, -1, -1):
            module = modules[i]
            rng_state = rng_states[i]

            # --- 恢复输入 (Reconstruction) ---
            # 逆向 Swap: (y1, y2) -> (y2, y1)
            y1, y2 = y2, y1
            dy1, dy2 = dy2, dy1

            # 恢复环境并计算梯度
            current_rng = get_cuda_rng_state()
            set_cuda_rng_state(rng_state)

            with torch.enable_grad():
                y2.requires_grad_(True)
                f_y2 = module.body(y2)

                # 梯度反传
                torch.autograd.backward(f_y2, dy1)

            set_cuda_rng_state(current_rng)

            # 数值恢复
            with torch.no_grad():
                x1 = y1 - f_y2

                # 累加梯度
                if y2.grad is not None:
                    dy2 = dy2 + y2.grad
                    y2.grad = None
                    y2.requires_grad_(False)

            # 更新指针
            y1 = x1

        grad_input = torch.cat([dy1, dy2], dim=1)

        return grad_input, None


class RevModule(nn.Module):
    def __init__(self, body, v=0.5):
        super().__init__()
        self.body = body
        self.v = v

    def forward(self, x):
        return self.body(x)
