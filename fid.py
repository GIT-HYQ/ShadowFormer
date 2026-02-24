import torch
import os
from pytorch_fid.fid_score import calculate_fid_given_paths
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision.io import read_image
from tqdm import tqdm

def calculate_fid_kid_metrics(path_gen, path_gt, device='cuda'):
    """
    一次性计算 FID 和 KID
    path_gen: 生成图文件夹
    path_gt:  GT图文件夹
    """
    # --- 1. 计算 FID ---
    print("===> Calculating FID...")
    fid_value = calculate_fid_given_paths(
        paths=[path_gt, path_gen],
        batch_size=50,
        device=torch.device(device),
        dims=2048
    )

    # --- 2. 计算 KID ---
    print("===> Calculating KID...")
    # subset_size 设为 100 适合你 1000+ 的量级
    kid_metric = KernelInceptionDistance(subset_size=100).to(device)
    
    gen_files = sorted([os.path.join(path_gen, f) for f in os.listdir(path_gen) if f.endswith(('.png', '.jpg'))])
    gt_files = sorted([os.path.join(path_gt, f) for f in os.listdir(path_gt) if f.endswith(('.png', '.jpg'))])

    for g_p, t_p in tqdm(zip(gen_files, gt_files), total=len(gen_files)):
        img_gen = read_image(g_p).to(device).unsqueeze(0) # [1, 3, H, W]
        img_gt = read_image(t_p).to(device).unsqueeze(0)
        
        # KID 内部会自动处理特征提取
        kid_metric.update(img_gt, real=True)
        kid_metric.update(img_gen, real=False)

    kid_mean, kid_std = kid_metric.compute()

    return fid_value, kid_mean.item(), kid_std.item()

# 使用示例
fid, kid_m, kid_s = calculate_fid_kid_metrics('./results/fid_gen', './results/fid_gt')
print(f"FID: {fid:.4f}")
print(f"KID: {kid_m:.5f} ± {kid_s:.5f}")