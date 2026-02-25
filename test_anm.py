
import numpy as np
import os,sys
import argparse
from tqdm import tqdm
from einops import rearrange, repeat

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from ptflops import get_model_complexity_info

import scipy.io as sio
from utils.loader import get_test_data
import utils
import cv2
from model import UNet
from scipy.stats import entropy
from scipy.fft import fft2, fftshift

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from sklearn.metrics import mean_squared_error as mse_loss

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default='../ISTD_Dataset/test/',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./log/ShadowFormer/models/model_best.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='ShadowFormer', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--cal_metrics', action='store_true', help='Measure denoised images with GT')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')    
parser.add_argument('--win_size', type=int, default=10, help='number of data loading workers')
parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
# args for vit
parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')
parser.add_argument('--train_ps', type=int, default=320, help='patch size of training sample')
parser.add_argument('--tile', type=int, default=None, help='Tile size (e.g 720). None means testing on the original resolution image')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
parser.add_argument('--anm', action='store_true', default=False, help='adaptive noise module')
args = parser.parse_args()


def calc_kl_divergence(residual_pred, residual_gt, bm=None, bins=256, eps=1e-10):
    """
    residual_pred/residual_gt: [H,W,C]
    bm: [H,W,1], 1表示掩码内；None表示全图
    返回 KL(Q||P), Q=gt residual, P=pred residual
    """
    if bm is not None:
        m = (bm[..., 0] > 0.5)
        pred = residual_pred[m]
        gt = residual_gt[m]
    else:
        pred = residual_pred.reshape(-1, residual_pred.shape[2])
        gt = residual_gt.reshape(-1, residual_gt.shape[2])

    if pred.size == 0 or gt.size == 0:
        return 0.0

    kls = []
    for c in range(pred.shape[1]):
        p = pred[:, c]
        q = gt[:, c]
        vmin = min(p.min(), q.min())
        vmax = max(p.max(), q.max())
        hp, _ = np.histogram(p, bins=bins, range=(vmin, vmax), density=True)
        hq, _ = np.histogram(q, bins=bins, range=(vmin, vmax), density=True)

        hp = np.clip(hp / (hp.sum() + eps), eps, 1.0)
        hq = np.clip(hq / (hq.sum() + eps), eps, 1.0)
        kls.append(entropy(hq, hp))  # KL(Q||P)

    return float(np.mean(kls))


def calc_psd_distance(residual_pred, residual_gt, bm=None, eps=1e-10):
    """
    返回 PSD 曲线L2距离（越小越好）
    """
    rp = residual_pred.mean(axis=2)
    rg = residual_gt.mean(axis=2)

    if bm is not None:
        m = (bm[..., 0] > 0.5).astype(np.float32)
        rp = rp * m
        rg = rg * m

    P = np.abs(fftshift(fft2(rp))) ** 2
    G = np.abs(fftshift(fft2(rg))) ** 2

    P = P / (P.sum() + eps)
    G = G / (G.sum() + eps)
    return float(np.sqrt(np.mean((P - G) ** 2)))



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)

test_dataset = get_test_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

if args.anm:
    args.arch = args.arch+'_anm'  # 强制使用带 ANM 的模型版本
model_restoration = utils.get_arch(args)
model_restoration = torch.nn.DataParallel(model_restoration)

utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()
model_restoration.eval()

img_multiple_of = 8 * args.win_size

def eval_metric_pack(pred_rgb, gt_rgb, noisy_rgb, bm):
    """
    pred_rgb/gt_rgb/noisy_rgb: [H,W,C], range [0,1]
    bm: [H,W,1], 1=mask内
    """
    nbm = 1 - bm

    # SSIM (gray)
    gray_pred = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2GRAY)
    gray_gt = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2GRAY)
    ssim_rgb = ssim_loss(gray_pred, gray_gt, channel_axis=None, data_range=1.0)
    ssim_s = ssim_loss(gray_pred * bm.squeeze(), gray_gt * bm.squeeze(), channel_axis=None, data_range=1.0)
    ssim_ns = ssim_loss(gray_pred * nbm.squeeze(), gray_gt * nbm.squeeze(), channel_axis=None, data_range=1.0)

    # PSNR
    psnr_rgb = psnr_loss(pred_rgb, gt_rgb)
    psnr_s = psnr_loss(pred_rgb * bm, gt_rgb * bm)
    psnr_ns = psnr_loss(pred_rgb * nbm, gt_rgb * nbm)

    # RMSE (LAB)
    rmse_rgb = np.abs(cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2LAB) - cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2LAB)).mean() * 3
    rmse_s = np.abs(cv2.cvtColor(pred_rgb * bm, cv2.COLOR_RGB2LAB) - cv2.cvtColor(gt_rgb * bm, cv2.COLOR_RGB2LAB)).sum() / (bm.sum() + 1e-10)
    rmse_ns = np.abs(cv2.cvtColor(pred_rgb * nbm, cv2.COLOR_RGB2LAB) - cv2.cvtColor(gt_rgb * nbm, cv2.COLOR_RGB2LAB)).sum() / (nbm.sum() + 1e-10)

    # KL / PSD
    residual_pred = pred_rgb - gt_rgb
    residual_gt = noisy_rgb - gt_rgb
    kl = calc_kl_divergence(residual_pred, residual_gt)
    skl = calc_kl_divergence(residual_pred, residual_gt, bm)
    nskl = calc_kl_divergence(residual_pred, residual_gt, nbm)

    psd = calc_psd_distance(residual_pred, residual_gt)
    spsd = calc_psd_distance(residual_pred, residual_gt, bm)
    nspsd = calc_psd_distance(residual_pred, residual_gt, nbm)

    return {
        "psnr": psnr_rgb, "ssim": ssim_rgb, "rmse": rmse_rgb,
        "spsnr": psnr_s, "sssim": ssim_s, "srmse": rmse_s,
        "nspsnr": psnr_ns, "nsssim": ssim_ns, "nsrmse": rmse_ns,
        "kl": kl, "skl": skl, "nskl": nskl,
        "psd": psd, "spsd": spsd, "nspsd": nspsd
    }

def init_metric_store():
    return {k: [] for k in [
        "psnr","ssim","rmse","spsnr","sssim","srmse","nspsnr","nsssim","nsrmse",
        "kl","skl","nskl","psd","spsd","nspsd"
    ]}

def append_metric_store(store, pack):
    for k, v in pack.items():
        store[k].append(v)

def print_metric_store(title, store):
    print(title)
    print("PSNR: %f, SSIM: %f, RMSE: %f " % (
        np.mean(store["psnr"]), np.mean(store["ssim"]), np.mean(store["rmse"])))
    print("SPSNR: %f, SSSIM: %f, SRMSE: %f " % (
        np.mean(store["spsnr"]), np.mean(store["sssim"]), np.mean(store["srmse"])))
    print("NSPSNR: %f, NSSSIM: %f, NSRMSE: %f " % (
        np.mean(store["nspsnr"]), np.mean(store["nsssim"]), np.mean(store["nsrmse"])))
    print("KL: %f, SKL: %f, NSKL: %f " % (
        np.mean(store["kl"]), np.mean(store["skl"]), np.mean(store["nskl"])))
    print("PSD: %f, SPSD: %f, NSPSD: %f " % (
        np.mean(store["psd"]), np.mean(store["spsd"]), np.mean(store["nspsd"]))
    )


with torch.no_grad():
    metric_main = init_metric_store()
    metric_anm = init_metric_store()
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_gt = data_test[0].numpy().squeeze().transpose((1, 2, 0))
        rgb_noisy = data_test[1].cuda()
        rgb_noisy_np = data_test[1].numpy().squeeze().transpose((1, 2, 0))  # 用于KL/PSD
        mask = data_test[2].cuda()
        filenames = data_test[3]

        # Pad the input if not_multiple_of win_size * 8
        height, width = rgb_noisy.shape[2], rgb_noisy.shape[3]
        H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                    (width + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - height if height % img_multiple_of != 0 else 0
        padw = W - width if width % img_multiple_of != 0 else 0
        rgb_noisy = F.pad(rgb_noisy, (0, padw, 0, padh), 'reflect')
        mask = F.pad(mask, (0, padw, 0, padh), 'reflect')

        if args.tile is None:
            if args.anm:
                 rgb_res_noisy, rgb_restored = model_restoration(rgb_noisy, mask)
            else:
                rgb_restored = model_restoration(rgb_noisy, mask)
                rgb_res_noisy = None
        else:
            # test the image tile by tile
            b, c, h, w = rgb_noisy.shape
            tile = min(args.tile, h, w)
            assert tile % 8 == 0, "tile size should be multiple of 8"
            tile_overlap = args.tile_overlap

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
            w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
            E = torch.zeros(b, c, h, w).type_as(rgb_noisy)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = rgb_noisy[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    mask_patch = mask[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    out_patch = model_restoration(in_patch, mask_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
                    W[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch_mask)
            rgb_restored = E.div_(W)
            rgb_res_noisy = None
    

        rgb_restored = torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))

        # Unpad the output
        rgb_restored = rgb_restored[:height, :width, :]

        if rgb_res_noisy is not None:
            rgb_res_noisy = torch.clamp(rgb_res_noisy, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))
            rgb_res_noisy = rgb_res_noisy[:height, :width, :]

        if args.cal_metrics:
            # 关键修复：使用裁剪后的 mask 计算 bm，避免和 gt/pred 尺寸不一致
            mask_eval = mask[..., :height, :width]
            bm = torch.where(mask_eval == 0, torch.zeros_like(mask_eval), torch.ones_like(mask_eval))  #binarize mask
            bm = np.expand_dims(bm.cpu().numpy().squeeze(), axis=2)

            append_metric_store(metric_main, eval_metric_pack(rgb_restored, rgb_gt, rgb_noisy_np, bm))

            if args.anm and (rgb_res_noisy is not None):
                append_metric_store(metric_anm, eval_metric_pack(rgb_res_noisy, rgb_gt, rgb_noisy_np, bm))

        if args.save_images:
            utils.save_img(rgb_restored*255.0, os.path.join(args.result_dir, filenames[0]))

if args.cal_metrics:
    print_metric_store("=== Restored Metrics ===", metric_main)
    if args.anm and len(metric_anm["psnr"]) > 0:
        print_metric_store("=== rgb_res_noisy Metrics ===", metric_anm)
