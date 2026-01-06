import os, glob, cv2, random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from argparse import ArgumentParser
from model import Net
from utils import *
from skimage.metrics import structural_similarity as ssim
from time import time
from tqdm import tqdm


def ddp_is_enabled() -> bool:
    return ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ) and int(os.environ["WORLD_SIZE"]) > 1


def find_images(root_dir: str):
    exts = ("png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(root_dir, f"*.{ext}")))
        paths.extend(glob.glob(os.path.join(root_dir, f"*.{ext.upper()}")))
    return sorted(set(paths))


use_ddp = ddp_is_enabled()
rank = 0
local_rank = 0
world_size = 1

if use_ddp:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
else:
    print("Running in single-GPU (non-DDP) mode.")


parser = ArgumentParser()
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--step_number", type=int, default=3)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--patch_size", type=int, default=256)
parser.add_argument("--cs_ratio", type=float, default=0.1)
parser.add_argument("--block_size", type=int, default=32)
parser.add_argument("--model_dir", type=str, default="weight")
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--log_dir", type=str, default="log")
parser.add_argument("--save_interval", type=int, default=10)

# 关键修正：训练集名称单独配置
parser.add_argument("--trainset_name", type=str, default="Set11")
parser.add_argument("--testset_name", type=str, default="Set11")

args = parser.parse_args()


seed = 2025 + rank
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

epoch = args.epoch
learning_rate = args.learning_rate
T = args.step_number
B = args.block_size
bsz = args.batch_size
psz = args.patch_size
ratio = args.cs_ratio

device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

if rank == 0:
    print("cs ratio =", ratio)
    print("batch size per gpu =", bsz)
    print("patch size =", psz)
    print("use_ddp =", use_ddp, "world_size =", world_size)
    print("data_dir =", os.path.abspath(args.data_dir))
    print("trainset_name =", args.trainset_name)
    print("testset_name =", args.testset_name)

iter_num = 1000
N = B * B
q = int(np.ceil(ratio * N))

U, S, V = torch.linalg.svd(torch.randn(N, N, device=device))
Phi = (U @ V)[:, :q]

print("reading training files...")
start_time = time()

# 关键修正：训练路径改为 data/Set11
train_root = os.path.join(args.data_dir, args.trainset_name)
training_image_paths = find_images(train_root)

if rank == 0:
    print("train_root =", os.path.abspath(train_root))
    print("training_image_num", len(training_image_paths), "read time", time() - start_time)

if len(training_image_paths) == 0:
    raise RuntimeError(
        "No training images found.\n"
        f"- train_root = {os.path.abspath(train_root)}\n"
        "- Fix: put images under that folder, or set --data_dir/--trainset_name correctly.\n"
        "- Supported extensions: png/jpg/jpeg/bmp/tif/tiff/webp"
    )

from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("sd-legacy/stable-diffusion-v1-5").to(device)

net = Net(T, pipe.unet).to(device)

if use_ddp:
    model = DDP(net, device_ids=[local_rank] if torch.cuda.is_available() else None)
    model._set_static_graph()
else:
    model = net

if rank == 0:
    param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("#Param.", param_cnt / 1e6, "M")


class MyDataset(Dataset):
    def __getitem__(self, index):
        max_tries = 50
        for _ in range(max_tries):
            path = random.choice(training_image_paths)
            img_bgr = cv2.imread(path, 1)
            if img_bgr is None:
                continue
            x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
            x = x[:, :, 0]  # Y
            h, w = x.shape
            max_h, max_w = h - psz, w - psz
            if max_h < 0 or max_w < 0:
                continue
            start_h = random.randint(0, max_h)
            start_w = random.randint(0, max_w)
            patch = x[start_h:start_h + psz, start_w:start_w + psz].astype(np.float32) / 255.0
            return torch.from_numpy(patch)

        raise RuntimeError(
            f"Failed to sample a valid patch after {max_tries} tries. "
            f"Ensure all training images are >= patch_size ({psz})."
        )

    def __len__(self):
        return iter_num * bsz


dataset = MyDataset()

if use_ddp:
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
else:
    sampler = None

num_workers = 0 if os.name == "nt" else 8

dataloader = DataLoader(
    dataset,
    batch_size=bsz,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    sampler=sampler,
    shuffle=(sampler is None),
    drop_last=True,
)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.5)
scaler = torch.cuda.amp.GradScaler()

model_dir = "./%s/R_%.2f_T_%d_B_%d" % (args.model_dir, ratio, T, B)
log_path = "./%s/R_%.2f_T_%d_B_%d.txt" % (args.log_dir, ratio, T, B)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)

# 测试集路径也用 data/Set11（保持原逻辑）
test_root = os.path.join(args.data_dir, args.testset_name)
test_image_paths = find_images(test_root)

if rank == 0:
    print("test_root =", os.path.abspath(test_root))
    print("test_image_num =", len(test_image_paths))


def test():
    if use_ddp and rank != 0:
        return None, None
    if len(test_image_paths) == 0:
        return None, None

    with torch.no_grad():
        PSNR_list, SSIM_list = [], []
        for path in test_image_paths:
            test_image = cv2.cvtColor(cv2.imread(path, 1), cv2.COLOR_BGR2YCrCb)
            img, old_h, old_w, img_pad, new_h, new_w = my_zero_pad(test_image[:, :, 0], block_size=B)
            img_pad = img_pad.reshape(1, 1, new_h, new_w) / 255.0
            x = torch.from_numpy(img_pad).to(device).float()

            perm = torch.randperm(new_h * new_w, device=device)
            perm_inv = torch.empty_like(perm)
            perm_inv[perm] = torch.arange(perm.shape[0], device=device)

            A = lambda z: (z.reshape(-1,)[perm].reshape(-1, N) @ Phi)
            AT = lambda z: (z @ Phi.t()).reshape(-1,)[perm_inv].reshape(1, 1, new_h, new_w)

            y = A(x)
            x_out = model(y, A, AT, use_amp_=False)[..., :old_h, :old_w]
            x_out = (x_out.clamp(min=0.0, max=1.0) * 255.0).cpu().numpy().squeeze()

            PSNR_list.append(psnr(x_out, img))
            SSIM_list.append(ssim(x_out, img, data_range=255))

    return float(np.mean(PSNR_list)), float(np.mean(SSIM_list))


print("start training...")
for epoch_i in range(1, epoch + 1):
    start_time = time()
    loss_avg = 0.0

    if use_ddp:
        sampler.set_epoch(epoch_i)
        dist.barrier()

    iterator = tqdm(dataloader) if rank == 0 else dataloader

    for x in iterator:
        x = x.unsqueeze(1).to(device, non_blocking=True)
        x = H(x, random.randint(0, 7))

        perm = torch.randperm(psz * psz, device=device)
        perm_inv = torch.empty_like(perm)
        perm_inv[perm] = torch.arange(perm.shape[0], device=device)

        A = lambda z: (z.reshape(bsz, -1)[:, perm].reshape(bsz, -1, N) @ Phi)
        AT = lambda z: (z @ Phi.t()).reshape(bsz, -1)[:, perm_inv].reshape(bsz, 1, psz, psz)

        y = A(x)
        x_out = model(y, A, AT)
        loss = (x_out - x).abs().mean()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_avg += float(loss.item())

    scheduler.step()
    loss_avg /= iter_num

    if rank == 0:
        log_data = "[%d/%d] Average loss: %f, time cost: %.2fs, cur lr is %f." % (
            epoch_i, epoch, loss_avg, time() - start_time, scheduler.get_last_lr()[0]
        )
        print(log_data)
        with open(log_path, "a") as log_file:
            log_file.write(log_data + "\n")

        if epoch_i % args.save_interval == 0:
            state_dict = model.module.state_dict() if use_ddp else model.state_dict()
            torch.save(state_dict, "./%s/net_params_%d.pkl" % (model_dir, epoch_i))

        cur_psnr, cur_ssim = test()
        if cur_psnr is not None:
            log_data = "CS Ratio is %.2f, PSNR is %.2f, SSIM is %.4f." % (ratio, cur_psnr, cur_ssim)
            print(log_data)
            with open(log_path, "a") as log_file:
                log_file.write(log_data + "\n")

if use_ddp:
    dist.destroy_process_group()
