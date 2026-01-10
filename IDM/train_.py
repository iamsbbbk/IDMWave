import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import os
import cv2
import random
import numpy as np
import time
import csv
import matplotlib

# 设置后端为 Agg，防止在无显示器的服务器上报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm

# ==============================================================================
# 0. 导入新模型
# ==============================================================================
# 假设你将改进后的模型保存为了 model_refined.py
# 如果你直接覆盖了 model.py，则保持原样
from model import IDMWaveUNet
from utils import calculate_psnr_torch, calculate_ssim_torch, my_zero_pad, data_augment


# ==============================================================================
# 1. 针对 IDMWave 优化的混合损失函数 (New Hybrid Loss)
# ==============================================================================
class CharbonnierLoss(nn.Module):
    """L1 Loss 的鲁棒变体，比 MSE 更容易让 PSNR 收敛到高分"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)


class FrequencyLoss(nn.Module):
    """频域损失，强化高频细节 (SSIM)"""

    def __init__(self):
        super(FrequencyLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        # 转换到频域
        x_fft = torch.fft.rfft2(x, norm='backward')
        y_fft = torch.fft.rfft2(y, norm='backward')
        # 比较幅值
        return self.criterion(torch.abs(x_fft), torch.abs(y_fft))


class IDMLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.05, gamma=0.1):  # 调整了权重
        super(IDMLoss, self).__init__()
        self.alpha = alpha  # Pixel Loss (Charbonnier) - 主攻 PSNR
        self.beta = beta  # Frequency Loss - 辅助 SSIM
        self.gamma = gamma  # Gradient/Edge Loss - 辅助锐度

        self.pix_loss = CharbonnierLoss()
        self.freq_loss = FrequencyLoss()

    def forward(self, x_recon, x_gt):
        # 1. 像素损失 (基础)
        l_pix = self.pix_loss(x_recon, x_gt)

        # 2. 频域损失 (可选，辅助)
        l_freq = self.freq_loss(x_recon, x_gt)

        # 3. 测量一致性损失 (可选，如果模型输出包含 y_hat)
        # 这里我们主要关注重建质量

        loss = self.alpha * l_pix + self.beta * l_freq
        return loss, l_pix, l_freq


# ==============================================================================
# 2. 智能数据搜索工具
# ==============================================================================

def get_image_paths(root_dir):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp')
    image_paths = []
    if not os.path.exists(root_dir): return []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths


# ==============================================================================
# 3. Dataset 定义
# ==============================================================================

class CS_Dataset(Dataset):
    def __init__(self, img_paths, patch_size, batch_size, iter_num=1000):
        self.img_paths = img_paths
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.iter_num = iter_num
        if len(self.img_paths) == 0: raise ValueError("No images found!")

    def __getitem__(self, index):
        for _ in range(10):
            path = random.choice(self.img_paths)
            try:
                img = cv2.imread(path, 1)
                if img is None: continue
                img_y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
                img_norm = img_y.astype(np.float32) / 255.0
                h, w = img_norm.shape
                if h < self.patch_size or w < self.patch_size: continue

                start_h = random.randint(0, h - self.patch_size)
                start_w = random.randint(0, w - self.patch_size)
                patch = img_norm[start_h:start_h + self.patch_size, start_w:start_w + self.patch_size]

                # 数据增强对达到 32dB 至关重要
                mode = random.randint(0, 7)
                patch_aug = data_augment(patch, mode)
                return torch.from_numpy(patch_aug.copy()).unsqueeze(0)
            except:
                continue
        return torch.zeros((1, self.patch_size, self.patch_size), dtype=torch.float32)

    def __len__(self):
        return self.iter_num * self.batch_size


# ==============================================================================
# 4. 可视化辅助
# ==============================================================================

def save_comparison_result(model, img_path, save_path, device, block_size):
    model.eval()
    try:
        img_bgr = cv2.imread(img_path, 1)
        if img_bgr is None: return
        img_y = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        img_y, old_h, old_w, img_pad, new_h, new_w = my_zero_pad(img_y, block_size=block_size)
        x_gt = torch.from_numpy(img_pad).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            x_recon, _ = model(x_gt)  # 注意新模型返回 (x, y)

        x_recon = x_recon[..., :old_h, :old_w].clamp(0, 1).cpu().squeeze().numpy()
        x_gt_crop = x_gt[..., :old_h, :old_w].cpu().squeeze().numpy()

        psnr_val = 20 * np.log10(1.0 / np.sqrt(np.mean((x_gt_crop - x_recon) ** 2)))

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(x_gt_crop, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(x_recon, cmap='gray')
        plt.title(f'Ours (PSNR: {psnr_val:.2f} dB)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Visual] Saved to {save_path}")
    except Exception as e:
        print(f"[Visual] Error: {e}")


# ==============================================================================
# 5. 主程序 (Main)
# ==============================================================================

def main():
    parser = ArgumentParser()
    parser.add_argument("--epoch", type=int, default=10)  # 建议跑 100 epoch 以充分收敛
    parser.add_argument("--learning_rate", type=float, default=2e-4)  # 初始 LR

    # [优化] 由于使用了 RevModule 节省显存，你可以尝试调大 batch_size (如 16 或 32)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--cs_ratio", type=float, default=0.1)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--base_dim", type=int, default=64)
    parser.add_argument("--model_dir", type=str, default="checkpoints")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--train_set", type=str, default="train")
    parser.add_argument("--test_set", type=str, default="Set11")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--save_interval", type=int, default=10)
    args = parser.parse_args()

    # --- Init ---
    def init_dist():
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            dist.init_process_group(backend="nccl", init_method="env://")
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
            return dist.get_rank(), int(os.environ['LOCAL_RANK'])
        else:
            return 0, 0

    rank, local_rank = init_dist()
    device = torch.device(f"cuda:{local_rank}")

    seed = 2025 + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    # --- Data ---
    if rank == 0: print(f"Scanning data in {args.data_dir}...")
    train_paths = get_image_paths(os.path.join(args.data_dir, args.train_set))
    if not train_paths: train_paths = get_image_paths(args.data_dir)
    test_paths = get_image_paths(os.path.join(args.data_dir, args.test_set))

    if rank == 0: print(f"Train images: {len(train_paths)} | Test images: {len(test_paths)}")
    if not train_paths: raise RuntimeError("No training data!")

    train_dataset = CS_Dataset(train_paths, args.patch_size, args.batch_size, iter_num=1000)
    train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True)

    # --- Model ---
    model = IDMWaveUNet(cs_ratio=args.cs_ratio, block_size=args.block_size, base_dim=args.base_dim).to(device)
    if rank == 0: print(f"Model Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")

    if dist.is_initialized():
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # [优化] 优化器设置
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    # [优化] 学习率衰减，设置 eta_min 防止后期学习率为0导致可学习参数无法更新
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)

    # [优化] 使用针对性优化的 Loss
    criterion = IDMLoss(alpha=1.0, beta=0.1).to(device)
    scaler = torch.amp.GradScaler('cuda')

    # --- Logging ---
    timestamp = time.strftime("%Y%m%d_%H%M")
    exp_name = f"IDM_Refined_R{args.cs_ratio}_{timestamp}"
    save_dir = os.path.join(args.model_dir, exp_name)
    log_file = os.path.join(args.log_dir, f"{exp_name}.txt")
    csv_file = os.path.join(args.log_dir, f"{exp_name}_metrics.csv")

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        with open(csv_file, 'w', newline='') as f:
            csv.writer(f).writerow(['Epoch', 'Loss', 'Pix_Loss', 'Freq_Loss', 'Val_PSNR', 'Val_SSIM', 'LR'])

    # --- Validation ---
    def validate():
        model.eval()
        psnr_list, ssim_list = [], []
        if not test_paths or rank != 0: return 0.0, 0.0

        with torch.no_grad():
            for path in test_paths:
                img_bgr = cv2.imread(path, 1)
                if img_bgr is None: continue
                img_y = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
                img_y, old_h, old_w, img_pad, _, _ = my_zero_pad(img_y, block_size=args.block_size)
                x_gt = torch.from_numpy(img_pad).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)

                # 新模型返回两个值
                x_recon, _ = model(x_gt)

                x_recon = x_recon[..., :old_h, :old_w].clamp(0, 1)
                x_gt_crop = x_gt[..., :old_h, :old_w]

                psnr_list.append(calculate_psnr_torch(x_recon, x_gt_crop).item())
                ssim_list.append(calculate_ssim_torch(x_recon, x_gt_crop).item())
        return np.mean(psnr_list), np.mean(ssim_list)

    # --- Training Loop ---
    if rank == 0: print("Start training...")
    best_psnr = 0.0
    best_model_path = ""

    for epoch_i in range(1, args.epoch + 1):
        start_time = time.time()
        model.train()
        if dist.is_initialized(): train_sampler.set_epoch(epoch_i)

        avg_loss, avg_pix, avg_freq = 0.0, 0.0, 0.0
        iterator = tqdm(train_loader, desc=f"Ep {epoch_i}") if rank == 0 else train_loader

        for x_gt in iterator:
            x_gt = x_gt.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                # 新模型 forward 返回 (recon, measurement)
                x_recon, y_meas = model(x_gt)
                loss, l_pix, l_freq = criterion(x_recon, x_gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            avg_loss += loss.item()
            avg_pix += l_pix.item()
            avg_freq += l_freq.item()

            # ================= [新增] 实时刷新进度条后缀 =================
            if rank == 0:
                # 显示当前 step 的 total loss 和 pixel loss
                iterator.set_postfix(loss=f"{loss.item():.4f}", pix=f"{l_pix.item():.4f}")
            # ==========================================================

        scheduler.step()

    if rank == 0:
            avg_loss /= len(train_loader)
            avg_pix /= len(train_loader)
            avg_freq /= len(train_loader)
            val_psnr, val_ssim = validate()
            time_cost = time.time() - start_time
            curr_lr = scheduler.get_last_lr()[0]

            log_str = (f"[Ep {epoch_i}] Loss: {avg_loss:.4f} (Pix:{avg_pix:.4f}) | "
                       f"PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f} | LR: {curr_lr:.1e} | T: {time_cost:.0f}s")
            print(log_str)

            with open(log_file, "a") as f:
                f.write(log_str + "\n")
            with open(csv_file, 'a', newline='') as f:
                csv.writer(f).writerow([epoch_i, avg_loss, avg_pix, avg_freq, val_psnr, val_ssim, curr_lr])

            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            if epoch_i % args.save_interval == 0:
                torch.save(state_dict, os.path.join(save_dir, f"epoch_{epoch_i}.pth"))

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_model_path = os.path.join(save_dir, "best_model.pth")
                torch.save(state_dict, best_model_path)
                print(f" -> Best PSNR!")

    if rank == 0:
        print("\nTraining Done. Visualizing...")
        if test_paths and best_model_path:
            infer_model = IDMWaveUNet(cs_ratio=args.cs_ratio, block_size=args.block_size, base_dim=args.base_dim).to(
                device)
            infer_model.load_state_dict(torch.load(best_model_path))
            save_comparison_result(infer_model, test_paths[0], os.path.join(args.log_dir, f"vis_result.png"), device,
                                   args.block_size)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        traceback.print_exc()
        time.sleep(1)
