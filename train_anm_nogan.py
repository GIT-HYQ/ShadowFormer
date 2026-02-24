import os
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'./auxiliary/'))
print(dir_name)

import argparse
import options
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()
print(opt)

import utils
######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True
# from piqa import SSIM
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
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


######### Logs dir ###########
log_dir = os.path.join(dir_name, 'log', opt.arch+opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt') 
print("Now time is : ", datetime.datetime.now().isoformat())
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

with open(logname,'a') as f:
    f.write(str(opt)+'\n')
    f.write(str(model_restoration)+'\n')

######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")


######### DataParallel ###########
model_restoration = torch.nn.DataParallel (model_restoration)
model_restoration.cuda()

######### Resume ###########
if opt.resume:
    path_chk_rest = opt.pretrain_weights
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1

# ######### Scheduler ###########
if opt.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 50
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()


######### Loss ###########
criterion = CharbonnierLoss().cuda()
# 建议加入 SSIM Loss，因为医学影像对结构非常敏感，能辅助物理噪声层更好地定位背景
criterion_ssim = SSIMLoss().cuda()
# 1. 调整 Loss 权重 (此时 lambda_adv 不再使用)
lambda_structure = 10.0   # 保证 clean_bg 准确去血管
lambda_noise_fit = 1.0    # 引导物理噪声层模拟背景分布

######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size':opt.train_ps}
train_dataset = get_training_data(opt.train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, 
        num_workers=opt.train_workers, pin_memory=True, drop_last=False)

val_dataset = get_validation_data(opt.val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)

######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
best_psnr = 0
best_epoch = 0
best_iter = 0
eval_now = 1000
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

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
            
            # 约束 3：保证注入物理噪声后的“合成图”也与“目标图”分布一致
            # 这一步是让 AdaptiveNoiseModule 学习物理参数 alpha 和 beta 的关键
            def statistical_loss0(pred, target):
                # 均值对齐
                l_mean = torch.abs(pred.mean() - target.mean())
                # 能量（方差）对齐：这才是物理参数 alpha 应该对齐的目标
                l_var = torch.abs(pred.var() - target.var()) 
                return l_mean + l_var
            
            def statistical_loss(pred, target, window_size=11):
                # 使用平均池化计算局部均值
                mu_p = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
                mu_t = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
                
                # 使用局部方差公式: Var(X) = E[X^2] - (E[X])^2
                mu2_p = F.avg_pool2d(pred**2, window_size, stride=1, padding=window_size//2)
                mu2_t = F.avg_pool2d(target**2, window_size, stride=1, padding=window_size//2)
                
                var_p = mu2_p - mu_p**2
                var_t = mu2_t - mu_t**2
                
                # 局部均值对齐 + 局部方差对齐
                l_mean = torch.mean(torch.abs(mu_p - mu_t))
                l_var = torch.mean(torch.abs(var_p - var_t)) 
                return l_mean + l_var
            
            # C. 噪声拟合损失 (专门优化 alpha/beta)
            # 使用 .detach() 截断背景图的梯度传导
            detached_final_y = model_restoration.module.get_final_y(clean_bg.detach(), alpha, beta, mask)
            
            # 使用之前建议的“局部方差统计 Loss”效果更佳
            loss_noise = statistical_loss(detached_final_y, target)

            # loss_noise = statistical_loss(final_y, target)
            # loss_noise = criterion(final_y, target)
            
            # 组合损失 (推荐比例)
            total_loss = 1.0 * loss_structure + 1.0 * loss_ssim + 1.0 * loss_noise
        loss_scaler(
                total_loss, optimizer,parameters=model_restoration.parameters(), clip_grad=1.0)
        epoch_loss += total_loss.item()

        # 监控打印 (保持之前的详细打印，以便观察 alpha/beta 摆动)
        if index % 100 == 0:
            print(f"Epoch: {epoch} | Iter: {index} | Total Loss: {total_loss.item():.4f}")
            print(f"Char_Loss: {loss_structure.item():.4f} | SSIM_Loss: {loss_ssim.item():.4f} | Noise_Loss: {loss_noise.item():.4f}")
            print(f"Alpha: {alpha.min().item():.6f}:{alpha.max().item():.6f} | Beta: {beta.min().item():.6f}:{beta.max().item():.6f}")
        
        #### Evaluation ####
        if (index+1)%eval_now==0 and i>0:
            eval_shadow_rmse = 0
            eval_nonshadow_rmse = 0
            eval_rmse = 0
            with torch.no_grad():
                model_restoration.eval()
                psnr_noisy_list = []
                psnr_clean_list = []
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

                avg_psnr_noisy = sum(psnr_noisy_list) / len(val_loader)
                avg_psnr_clean = sum(psnr_clean_list) / len(val_loader)
                if avg_psnr_clean > best_psnr:
                    best_psnr = avg_psnr_clean
                    best_epoch = epoch
                    best_iter = i
                    torch.save({'epoch': epoch,
                                'state_dict': model_restoration.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(model_dir,"model_best.pth"))
                print("[Ep %d it %d\t PSNR : %.4f\t PSNR_noisy : %.4f] " % (epoch, i, avg_psnr_clean, avg_psnr_noisy))
                with open(logname,'a') as f:
                    f.write("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " \
                        % (epoch, i, avg_psnr_clean, best_epoch, best_iter, best_psnr)+'\n')
                model_restoration.train()
                torch.cuda.empty_cache()
    scheduler.step()
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss,scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    with open(logname,'a') as f:
        f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0])+'\n')

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))   

    if epoch%opt.checkpoint == 0:
        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 
print("Now time is : ",datetime.datetime.now().isoformat())



