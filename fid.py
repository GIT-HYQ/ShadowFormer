import os
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.io import read_image, write_png
from tqdm import tqdm

from pytorch_fid.fid_score import calculate_fid_given_paths
from torchmetrics.image.kid import KernelInceptionDistance


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _build_stem_map(folder):
    m = {}
    for f in os.listdir(folder):
        p = Path(folder) / f
        if p.suffix.lower() in IMG_EXTS:
            m[p.stem] = str(p)
    return m


def _load_mask(mask_path, h, w):
    # [1,H,W], uint8
    m = read_image(mask_path)[:1]  # 取单通道
    if m.shape[-2:] != (h, w):
        m = F.interpolate(m.unsqueeze(0).float(), size=(h, w), mode="nearest").squeeze(0).to(torch.uint8)
    return m


def _apply_region_mask(img, mask, region="in"):
    """
    img: [3,H,W] uint8
    mask: [1,H,W] uint8, >127 视为mask内
    """
    mb = (mask > 127)
    if region == "out":
        mb = ~mb
    return img * mb.expand_as(img).to(img.dtype)


def _paired_files(path_gen, path_gt, path_mask=None):
    gen_map = _build_stem_map(path_gen)
    gt_map = _build_stem_map(path_gt)
    common = sorted(set(gen_map.keys()) & set(gt_map.keys()))

    if path_mask is None:
        return [(gen_map[s], gt_map[s], None, s) for s in common]

    mask_map = _build_stem_map(path_mask)
    common = [s for s in common if s in mask_map]
    return [(gen_map[s], gt_map[s], mask_map[s], s) for s in common]


def calculate_fid_kid_metrics(path_gen, path_gt, path_mask=None, device="cuda"):
    """
    path_mask=None: 返回全图 FID/KID
    path_mask!=None: 额外返回 mask内/外 FID/KID
    """
    pairs = _paired_files(path_gen, path_gt, path_mask)
    if len(pairs) == 0:
        raise RuntimeError("No matched image pairs found.")

    # -------- 全图 FID --------
    print("===> Calculating global FID...")
    fid_global = calculate_fid_given_paths(
        paths=[path_gt, path_gen],
        batch_size=50,
        device=torch.device(device),
        dims=2048
    )

    # -------- 全图 KID --------
    print("===> Calculating global KID...")
    subset_size = min(100, max(2, len(pairs)))
    kid_global = KernelInceptionDistance(subset_size=subset_size).to(device)

    for g_p, t_p, _, _ in tqdm(pairs, total=len(pairs)):
        img_gen = read_image(g_p).to(device).unsqueeze(0)  # uint8 [1,3,H,W]
        img_gt = read_image(t_p).to(device).unsqueeze(0)
        kid_global.update(img_gt, real=True)
        kid_global.update(img_gen, real=False)

    kid_g_mean, kid_g_std = kid_global.compute()

    result = {
        "global": {
            "fid": float(fid_global),
            "kid_mean": float(kid_g_mean.item()),
            "kid_std": float(kid_g_std.item()),
        }
    }

    if path_mask is None:
        return result

    # -------- mask内/外 FID: 需落盘给 pytorch_fid --------
    with tempfile.TemporaryDirectory() as td:
        gt_in = Path(td) / "gt_in"
        gen_in = Path(td) / "gen_in"
        gt_out = Path(td) / "gt_out"
        gen_out = Path(td) / "gen_out"
        gt_in.mkdir(parents=True, exist_ok=True)
        gen_in.mkdir(parents=True, exist_ok=True)
        gt_out.mkdir(parents=True, exist_ok=True)
        gen_out.mkdir(parents=True, exist_ok=True)

        print("===> Building masked image sets (in/out)...")
        for g_p, t_p, m_p, stem in tqdm(pairs, total=len(pairs)):
            img_gen = read_image(g_p)  # uint8 [3,H,W]
            img_gt = read_image(t_p)

            mask_g = _load_mask(m_p, img_gen.shape[-2], img_gen.shape[-1])
            mask_t = _load_mask(m_p, img_gt.shape[-2], img_gt.shape[-1])

            img_gen_in = _apply_region_mask(img_gen, mask_g, "in")
            img_gt_in = _apply_region_mask(img_gt, mask_t, "in")
            img_gen_out = _apply_region_mask(img_gen, mask_g, "out")
            img_gt_out = _apply_region_mask(img_gt, mask_t, "out")

            write_png(img_gen_in.cpu(), str(gen_in / f"{stem}.png"))
            write_png(img_gt_in.cpu(), str(gt_in / f"{stem}.png"))
            write_png(img_gen_out.cpu(), str(gen_out / f"{stem}.png"))
            write_png(img_gt_out.cpu(), str(gt_out / f"{stem}.png"))

        print("===> Calculating FID (mask-in)...")
        fid_in = calculate_fid_given_paths(
            paths=[str(gt_in), str(gen_in)],
            batch_size=50,
            device=torch.device(device),
            dims=2048
        )

        print("===> Calculating FID (mask-out)...")
        fid_out = calculate_fid_given_paths(
            paths=[str(gt_out), str(gen_out)],
            batch_size=50,
            device=torch.device(device),
            dims=2048
        )

    # -------- mask内/外 KID: 直接张量更新 --------
    print("===> Calculating KID (mask-in/out)...")
    kid_in = KernelInceptionDistance(subset_size=subset_size).to(device)
    kid_out = KernelInceptionDistance(subset_size=subset_size).to(device)

    for g_p, t_p, m_p, _ in tqdm(pairs, total=len(pairs)):
        img_gen = read_image(g_p)  # uint8 [3,H,W]
        img_gt = read_image(t_p)

        mask_g = _load_mask(m_p, img_gen.shape[-2], img_gen.shape[-1])
        mask_t = _load_mask(m_p, img_gt.shape[-2], img_gt.shape[-1])

        img_gen_in = _apply_region_mask(img_gen, mask_g, "in").to(device).unsqueeze(0)
        img_gt_in = _apply_region_mask(img_gt, mask_t, "in").to(device).unsqueeze(0)
        img_gen_out = _apply_region_mask(img_gen, mask_g, "out").to(device).unsqueeze(0)
        img_gt_out = _apply_region_mask(img_gt, mask_t, "out").to(device).unsqueeze(0)

        kid_in.update(img_gt_in, real=True)
        kid_in.update(img_gen_in, real=False)
        kid_out.update(img_gt_out, real=True)
        kid_out.update(img_gen_out, real=False)

    kid_in_mean, kid_in_std = kid_in.compute()
    kid_out_mean, kid_out_std = kid_out.compute()

    result["mask_in"] = {
        "fid": float(fid_in),
        "kid_mean": float(kid_in_mean.item()),
        "kid_std": float(kid_in_std.item()),
    }
    result["mask_out"] = {
        "fid": float(fid_out),
        "kid_mean": float(kid_out_mean.item()),
        "kid_std": float(kid_out_std.item()),
    }
    return result


if __name__ == "__main__":
    metrics = calculate_fid_kid_metrics(
        # path_gen="/home/share/clr/share/work/CAGSubtraction/ShadowFormer/exp/anm",
        # path_gen="/home/share/clr/share/work/CAGSubtraction/ShadowFormer/exp/anm_gray",
        path_gen="/home/share/clr/share/work/CAGSubtraction/ShadowFormer/exp/anm_gray_ns4",        
        # path_gen="/home/share/clr/share/work/CAGSubtraction/ShadowFormer/exp/baseline",
        # path_gen="/home/share/clr/share/work/CAGSubtraction/ShadowDiffusion/output/res_v2/results512",
        path_gt="/home/share/clr/share/data/cag_res_v2/test/test_C",
        path_mask="/home/share/clr/share/data/cag_res_v2/test/test_B",  # 可设为 None
        device="cuda"
    )

    print(metrics)