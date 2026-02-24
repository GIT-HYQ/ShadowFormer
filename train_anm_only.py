import os
import sys
from log import Logger

import utils
import torch
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True
# from piqa import SSIM
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# logger.info(device)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime
from pdb import set_trace as stx
from utils import save_img
from losses import CharbonnierLoss, SSIMLoss

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler

from utils.loader import get_training_data, get_validation_data
from collections import OrderedDict

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'./auxiliary/'))

import argparse
import options
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()
######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

######### Logs dir ###########
log_dir = os.path.join(dir_name, 'log', opt.arch+opt.env)
logger = Logger(log_dir)
logger.info("Now time is : ", datetime.datetime.now().isoformat())

logger.info(dir_name)
logger.info(opt)

result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(model_dir)

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)



######### Model ###########
model_restoration = utils.get_arch(opt)

######### Resume ###########
if opt.resume:
    path_chk_rest = opt.pretrain_weights
    logger.info(f"===> Resuming from {path_chk_rest}")
    
    # 建议手动处理加载，确保不被旧的 optimizer 状态干扰
    checkpoint = torch.load(path_chk_rest)
    state_dict = checkpoint["state_dict"]
    # 移除 state_dict 中的 'module.' 前缀（如果之前的 checkpoint 是用 DataParallel 存的）
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k 
        new_state_dict[name] = v
        
    # 此时加载权重 (strict=False 允许加载 ANM 模块，即使旧 checkpoint 里没有它)
    model_restoration.load_state_dict(new_state_dict, strict=False)
    
    # start_epoch 可以重置为 1，因为这是一个新的阶段
    start_epoch = 1
    logger.info("===> Weights loaded successfully. Starting new training phase (ANM).")


# --- 修改 1: 基于 "adaptive_noise" 命名精准冻结 ---
logger.info("===> Strategy: Freezing Backbone, Tuning Noise Module Only...")
trainable_count = 0
for name, param in model_restoration.named_parameters():
    # 匹配 self.adaptive_noise 及其内部的所有参数
    if 'adaptive_noise' in name:
        param.requires_grad = True
        trainable_count += 1
        # logger.info(f" [Trainable] {name}") # 调试时可开启
    else:
        param.requires_grad = False
if trainable_count == 0:
    logger.error("ERROR: No parameters found with 'adaptive_noise'. Check model naming!")
    sys.exit()
else:
    logger.info(f"Successfully locked backbone. Found {trainable_count} trainable noise parameters.")
# -----------------------------------------------

logger.info(str(opt))
logger.info(str(model_restoration))


######### DataParallel ###########
model_restoration.cuda()
model_restoration = torch.nn.DataParallel (model_restoration)


######### Optimizer ###########
start_epoch = 1
# --- 修改 2: 只将需要梯度的参数传给优化器 ---
trainable_params = [p for p in model_restoration.parameters() if p.requires_grad]

if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(trainable_params, lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
    optimizer = optim.AdamW(trainable_params, lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.weight_decay)
# ------------------------------------------

# ######### Scheduler ###########
if opt.warmup:
    logger.info("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 50
    logger.info("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()

def statistical_loss(pred, target, mask, window_size=11):
    # --- 1. 计算局部统计量 ---
    mu_p = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
    mu_t = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
    
    # Var(X) = E[X^2] - (E[X])^2
    var_p = F.avg_pool2d(pred**2, window_size, stride=1, padding=window_size//2) - mu_p**2
    var_t = F.avg_pool2d(target**2, window_size, stride=1, padding=window_size//2) - mu_t**2
    
    # 限制方差下限，防止 Log(0) 报错
    var_p = torch.clamp(var_p, min=1e-7)
    var_t = torch.clamp(var_t, min=1e-7)

    # --- 2. 构造空间权重图 (针对血管区域进行梯度放大) ---
    with torch.no_grad():
        # 通过 max_pool2d 模拟膨胀，扩大血管周边的关注范围
        # 血管及其周围 11x11 区域权重为 10，背景权重为 1
        weight_map = F.max_pool2d(mask, kernel_size=window_size, stride=1, padding=window_size//2)
        weight_map = weight_map * 9.0 + 1.0  

    # --- 3. 计算加权损失 ---
    # Log 空间方差对齐：捕捉 Alpha 和 Beta 的细微物理变化
    diff_log_var = torch.abs(torch.log(var_p) - torch.log(var_t))
    l_var = torch.mean(diff_log_var * weight_map)
    
    # 局部均值对齐：低权重，防止 Beta 干扰亮度
    l_mean = torch.mean(torch.abs(mu_p - mu_t) * weight_map) * 0.1
    
    return l_var + l_mean

######### Loss ###########
criterion = CharbonnierLoss().cuda()
# 建议加入 SSIM Loss，因为医学影像对结构非常敏感，能辅助物理噪声层更好地定位背景
criterion_ssim = SSIMLoss().cuda()
# 1. 调整 Loss 权重 (此时 lambda_adv 不再使用)
lambda_structure = 10.0   # 保证 clean_bg 准确去血管
lambda_noise_fit = 1.0    # 引导物理噪声层模拟背景分布

######### DataLoader ###########
logger.info('===> Loading datasets')
img_options_train = {'patch_size':opt.train_ps}
train_dataset = get_training_data(opt.train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, 
        num_workers=opt.train_workers, pin_memory=True, drop_last=False)

val_dataset = get_validation_data(opt.val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
logger.info("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)

######### train ###########
logger.info('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
best_psnr = 0
best_score = -1e10  # 初始化为一个极小值
best_epoch = 0
best_iter = 0
eval_now = 1000
logger.info("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()
ii=0
index = 0
for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
    epoch_ssim_loss = 0
    for i, data in enumerate(train_loader, 0): 
        # zero_grad
        index += 1
        optimizer.zero_grad()
        target = data[0].cuda()
        input_ = data[1].cuda()
        mask = data[2].cuda()
        if epoch > 5:
            target, input_, mask = utils.MixUp_AUG().aug(target, input_, mask)
        with torch.cuda.amp.autocast():
            # 前向传播：final_y 是带噪声的合成图，clean_bg 是去血管后的干净图
            final_y, clean_bg, (alpha, beta) = model_restoration(input_, mask)
            
            # --- 核心损失计算 ---
            # 约束 1：保证生成的“干净图”与“目标图”结构一致
            loss_structure = criterion(clean_bg, target)

            # 2. 结构级损失 (SSIM) - 重点消除血管移除后的残留边缘
            # 注意：SSIM 必须输入已经 Sigmoid 处理过的 [0, 1] 图像
            loss_ssim = criterion_ssim(clean_bg, target)
            
            # --- 核心：使用 Mask 引导的统计损失 ---
            # 现在我们需要把 mask 传入函数中
            loss_noise = statistical_loss(final_y, target, mask)
            
            # --- 最终 Loss 配置 ---
            # 在此阶段建议只使用 loss_noise 驱动参数拟合
            total_loss = 1.0 * loss_noise
        loss_scaler(
                total_loss, optimizer,parameters=model_restoration.parameters())
        epoch_loss += total_loss.item()

        # 监控打印 (保持之前的详细打印，以便观察 alpha/beta 摆动)
        if index % 10 == 0:
            logger.info(f"Epoch: {epoch} | Iter: {index} | Total Loss: {total_loss.item():.4f}")
            logger.info(f"Char_Loss: {loss_structure.item():.4f} | SSIM_Loss: {loss_ssim.item():.4f} | Noise_Loss: {loss_noise.item():.4f}")
            logger.info(f"Alpha-{alpha.min().item():.6f}:{alpha.max().item():.6f}:{alpha.mean().item():.6f} | Beta-{beta.min().item():.6f}:{beta.max().item():.6f}:{beta.mean().item():.6f}")
        
        #### Evaluation ####
        if (index+1)%eval_now==0 and i>0:
            eval_shadow_rmse = 0
            eval_nonshadow_rmse = 0
            eval_rmse = 0
            with torch.no_grad():
                model_restoration.eval()
                psnr_noisy_list = []
                psnr_clean_list = []
                val_ssim_scores = []
                val_mglsa_losses = []
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    mask = data_val[2].cuda()
                    filenames = data_val[3]
                    with torch.cuda.amp.autocast():
                        # 接收修改后的返回值
                        res_noisy, res_clean = model_restoration(input_, mask)

                    # 必须 clamp 到 [0, 1] 才能计算正确的 PSNR
                    res_noisy = torch.clamp(res_noisy, 0, 1)
                    res_clean = torch.clamp(res_clean, 0, 1)

                    # 计算两种 PSNR
                    psnr_noisy_list.append(utils.batch_PSNR(res_noisy, target, False).item())
                    psnr_clean_list.append(utils.batch_PSNR(res_clean, target, False).item())

                    # 1. 使用改造后的 SSIMLoss 获取分数 (as_loss=False)
                    ssim_score = criterion_ssim(res_noisy, target, as_loss=False)
                    val_ssim_scores.append(ssim_score.item())

                    # 2. 计算物理统计损失
                    mglsa_loss = statistical_loss(res_noisy, target, mask)
                    val_mglsa_losses.append(mglsa_loss.item())

                avg_psnr_noisy = sum(psnr_noisy_list) / len(val_loader)
                avg_psnr_clean = sum(psnr_clean_list) / len(val_loader)

                # --- 指标汇总 ---
                avg_ssim = sum(val_ssim_scores) / len(val_loader)
                avg_mglsa = sum(val_mglsa_losses) / len(val_loader)

                # --- 综合得分 (Composite Score) ---
                # 逻辑：SSIM 越高越好，MGLSA 越低越好。
                # 这里的 10.0 是平衡系数，旨在让 0.05 左右的 Loss 影响到 0.8 左右的 SSIM
                current_score = avg_ssim - (10.0 * avg_mglsa)

                logger.info(f"[Eval] Epoch {epoch} | SSIM: {avg_ssim:.4f} | MGLSA: {avg_mglsa:.6e} | Score: {current_score:.4f}")
                        
                if current_score > best_score:
                    best_score = current_score
                    best_epoch = epoch
                    best_iter = i
                    torch.save({'epoch': epoch,
                                'state_dict': model_restoration.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(model_dir,"model_best.pth"))
                logger.info("[Ep %d it %d\t PSNR : %.4f\t PSNR_noisy : %.4f] " % (epoch, i, avg_psnr_clean, avg_psnr_noisy))
                logger.warn(f"New Best Model Saved! Score: {best_score:.4f}")
                
                model_restoration.train()
                torch.cuda.empty_cache()
    scheduler.step()
    
    logger.info("------------------------------------------------------------------")
    logger.info("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss,scheduler.get_last_lr()[0]))
    logger.info("------------------------------------------------------------------")

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))   

    if epoch%opt.checkpoint == 0:
        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 
logger.info("Now time is : ",datetime.datetime.now().isoformat())



