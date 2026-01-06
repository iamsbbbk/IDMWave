import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math

# ==============================================================================
# 1. 配置区域 (请根据你的实际情况修改)
# ==============================================================================
# 待测试的图像路径
TEST_IMG_PATH = "../data/CBSD68/original/3096.jpg"

# 训练好的模型权重路径 (请修改为你保存的 .pth 文件路径)
# 例如: "checkpoints/IDM_Refined_R0.1_xxxx/best_model.pth"
MODEL_WEIGHTS_PATH = "./checkpoints/IDM_R0.1_20260103_2121/best_model.pth"

# 模型参数 (必须与训练时一致)
CS_RATIO = 0.1
BLOCK_SIZE = 32
BASE_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 2. 导入模型
# ==============================================================================
try:
    from model import IDMWaveUNet
except ImportError:
    # 如果文件名是 model_refined.py，尝试从那里导入
    try:
        from model_refined import IDMWaveUNet
    except ImportError:
        raise ImportError("找不到 model.py 或 IDMWaveUNet 类，请检查文件位置。")


# ==============================================================================
# 3. 工具函数
# ==============================================================================
def calc_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(1.0 / np.sqrt(mse))


def calc_ssim(img1, img2):
    # 简易 SSIM 计算 (为了不依赖 skimage，这里使用简化的计算，或者你可以安装 scikit-image)
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def pad_image(img, block_size):
    """将图像填充为 block_size 的倍数"""
    h, w = img.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    img_pad = np.pad(img, ((0, pad_h), (0, pad_w)), 'edge')
    return img_pad, h, w


# ==============================================================================
# 4. 核心测试逻辑
# ==============================================================================
def run_test():
    print(f"Running on device: {DEVICE}")

    # 1. 检查文件
    if not os.path.exists(TEST_IMG_PATH):
        print(f"Error: 找不到图片 {TEST_IMG_PATH}")
        # 创建一个假图片用于演示
        print("Generating a dummy image for demonstration...")
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.putText(dummy, "Test Image", (50, 128), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        os.makedirs("../data", exist_ok=True)
        cv2.imwrite(TEST_IMG_PATH, dummy)

    # 2. 加载模型
    print("Loading model...")
    model = IDMWaveUNet(cs_ratio=CS_RATIO, block_size=BLOCK_SIZE, base_dim=BASE_DIM).to(DEVICE)

    if os.path.exists(MODEL_WEIGHTS_PATH):
        try:
            state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load weights ({e}). Using random weights.")
    else:
        print(f"Warning: Weights file '{MODEL_WEIGHTS_PATH}' not found. Using random weights (Output will be noise).")

    model.eval()

    # 3. 处理图像
    img_bgr = cv2.imread(TEST_IMG_PATH)
    # 压缩感知通常只处理 Y 通道 (亮度)
    img_y = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    img_norm = img_y.astype(np.float32) / 255.0

    # Padding
    img_pad, old_h, old_w = pad_image(img_norm, BLOCK_SIZE)

    # 转 Tensor
    x_gt = torch.from_numpy(img_pad).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    # 4. 推理
    print("Inference start...")
    with torch.no_grad():
        start_t = time.time()
        # 兼容新旧模型返回格式
        res = model(x_gt)
        if isinstance(res, tuple):
            x_recon = res[0]  # (recon, y)
        else:
            x_recon = res

        end_t = time.time()
        print(f"Inference done in {(end_t - start_t) * 1000:.2f} ms")

    # 5. 后处理
    x_recon = x_recon.squeeze().cpu().numpy()
    x_recon = np.clip(x_recon, 0, 1)
    # Crop back
    x_recon_crop = x_recon[:old_h, :old_w]
    x_gt_crop = img_norm  # Original size

    # 6. 计算指标
    psnr_val = calc_psnr(x_gt_crop, x_recon_crop)
    ssim_val = calc_ssim(x_gt_crop, x_recon_crop)
    print(f"Metrics -> PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")

    # 7. 可视化 (Creative Visualization)
    plot_visualization(x_gt_crop, x_recon_crop, psnr_val, ssim_val)


def plot_visualization(gt, recon, psnr, ssim):
    import matplotlib.gridspec as gridspec

    # 计算误差图 (|GT - Recon|)
    diff = np.abs(gt - recon)
    # 稍微增强误差图的对比度以便观察
    diff_vis = diff * 5
    diff_vis = np.clip(diff_vis, 0, 1)

    # 设置画布风格
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.1])  # 最后一列宽一点给colorbar

    # --- Plot 1: Ground Truth ---
    ax0 = plt.subplot(gs[0])
    ax0.imshow(gt, cmap='gray')
    ax0.set_title('Original Image (GT)', fontsize=14, color='white')
    ax0.axis('off')

    # --- Plot 2: Reconstruction ---
    ax1 = plt.subplot(gs[1])
    ax1.imshow(recon, cmap='gray')
    ax1.set_title(f'Reconstruction\nPSNR: {psnr:.2f}dB | SSIM: {ssim:.4f}', fontsize=14, color='#00ff00')
    ax1.axis('off')

    # --- Plot 3: Error Heatmap ---
    ax2 = plt.subplot(gs[2])
    # 使用 'inferno' 或 'magma' 这种对人眼感知均匀的色图
    im = ax2.imshow(diff, cmap='inferno', vmin=0, vmax=0.1)
    ax2.set_title('Error Heatmap (Darker is Better)', fontsize=14, color='orange')
    ax2.axis('off')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Absolute Error Magnitude', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    plt.tight_layout()

    # 保存结果
    save_path = "result_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {os.path.abspath(save_path)}")

    # 尝试弹出窗口显示 (如果环境支持)
    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    import time

    run_test()
