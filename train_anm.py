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
from losses import CharbonnierLoss

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler

from utils.loader import get_training_data, get_validation_data
from adative_noise_layer import PatchGANDiscriminator


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
# 判别器：输入为单通道（造影图通常为灰度）
model_D = PatchGANDiscriminator(in_channels=3).cuda()

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

# 2. 判别器优化器 (新增)
# 注意：betas 使用 (0.5, 0.999) 是为了 GAN 训练的稳定性
optimizer_D = optim.Adam(model_D.parameters(), lr=opt.lr_initial, betas=(0.5, 0.999), eps=1e-8)

######### DataParallel ###########
model_restoration = torch.nn.DataParallel (model_restoration)
model_restoration.cuda()
model_D = torch.nn.DataParallel(model_D)

######### Resume ###########
if opt.resume:
    path_chk_rest = opt.pretrain_weights
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1

    # 建议保存模型时也保存 model_D
    utils.load_checkpoint(model_D, path_chk_rest.replace('model_latest', 'model_D_latest'))


# ######### Scheduler ###########
# 判别器调度器 (与主调度器逻辑保持一致)
if opt.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()

    scheduler_cosine_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, opt.nepoch-warmup_epochs, eta_min=1e-6)
    scheduler_D = GradualWarmupScheduler(optimizer_D, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine_D)
    scheduler_D.step()
else:
    step = 50
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()

    scheduler_D = StepLR(optimizer_D, step_size=50, gamma=0.5)
    scheduler_D.step()


######### Loss ###########
criterion = CharbonnierLoss().cuda()
# 损失函数
criterion_GAN = nn.BCEWithLogitsLoss().cuda()

# 权重分配
lambda_L1 = 10.0
lambda_adv = 1.0

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
        
        # --- (1) 更新判别器 D ---
        optimizer_D.zero_grad()
        with torch.cuda.amp.autocast():
            # 获取 G 的输出但切断梯度 (detach)
            with torch.no_grad():
                # 注意：这里接收三个参数：注入噪声图, 干净图, 物理参数
                final_y, _, _ = model_restoration(input_, mask)
            
            # 判别真实图片
            pred_real = model_D(target)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            
            # 判别生成的噪声图片
            pred_fake = model_D(final_y.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        
        # 判别器反向传播 (使用原本的 NativeScaler)
        loss_scaler(loss_D, optimizer_D, parameters=model_D.parameters(), clip_grad=10.0)

        # --- (2) 更新生成器 G (ShadowFormer) ---
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            # 完整前向传播
            final_y, clean_bg, (alpha, beta) = model_restoration(input_, mask)
            
            # A. 结构 Loss: 保证去血管准确 (作用于 clean_bg)
            loss_char = criterion(clean_bg, target)
            
            # B. 对抗 Loss: 保证注入噪声后的纹理真实 (作用于 final_y)
            pred_g_fake = model_D(final_y)
            loss_adv = criterion_GAN(pred_g_fake, torch.ones_like(pred_g_fake))
            
            # C. 总 Loss
            total_loss_G = lambda_L1 * loss_char + lambda_adv * loss_adv

        # 生成器反向传播
        loss_scaler(total_loss_G, optimizer, parameters=model_restoration.parameters())
        
        epoch_loss += total_loss_G.item()

        # --- 增强版监控逻辑 ---
        if index % 100 == 0:
            # 计算不带系数的原始数值
            raw_char = loss_char.item()
            raw_adv = loss_adv.item()
            
            print(f"\n" + "-"*30)
            print(f"[Iter {index}] Total Loss: {total_loss_G.item():.4f}")
            # 这里的打印能让你一眼看出 2580 是由谁构成的
            print(f"Loss_Char (Raw): {raw_char:.4f} | Scaled: {lambda_L1 * raw_char:.4f}")
            print(f"Loss_Adv (Raw): {raw_adv:.4f} | Scaled: {lambda_adv * raw_adv:.4f}")
            
            # 数值范围监控
            print(f"DEBUG -> Alpha Mean: {alpha.mean().item():.6f}, Max: {alpha.max().item():.6f}")
            print(f"DEBUG -> Clean_BG Max: {clean_bg.max().item():.4f}, Min: {clean_bg.min().item():.4f}")
            print(f"DEBUG -> Target Max: {target.max().item():.4f}, Min: {target.min().item():.4f}")
            print("-"*30)
        
        
        #### Evaluation ####
        if (index+1)%eval_now==0 and i>0:
            eval_shadow_rmse = 0
            eval_nonshadow_rmse = 0
            eval_rmse = 0
            with torch.no_grad():
                model_restoration.eval()
                psnr_val_rgb = []
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    mask = data_val[2].cuda()
                    filenames = data_val[3]
                    with torch.cuda.amp.autocast():
                        restored = model_restoration(input_, mask)
                    restored = torch.clamp(restored,0,1)
                    psnr_val_rgb.append(utils.batch_PSNR(restored, target, False).item())

                psnr_val_rgb = sum(psnr_val_rgb)/len(val_loader)
                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    torch.save({'epoch': epoch,
                                'state_dict': model_restoration.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(model_dir,"model_best.pth"))
                print("[Ep %d it %d\t PSNR : %.4f] " % (epoch, i, psnr_val_rgb))
                with open(logname,'a') as f:
                    f.write("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " \
                        % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr)+'\n')
                model_restoration.train()
                torch.cuda.empty_cache()
    scheduler.step()
    scheduler_D.step()  # 判别器调度
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss,scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    with open(logname,'a') as f:
        f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0])+'\n')
    
    torch.save({
        'epoch': epoch, 
        'state_dict': model_restoration.state_dict(),
        'state_dict_D': model_D.state_dict(), # 增加判别器权重保存
        'optimizer' : optimizer.state_dict(),
        'optimizer_D': optimizer_D.state_dict()
    }, os.path.join(model_dir, "model_latest.pth"))

    if epoch%opt.checkpoint == 0:
        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'state_dict_D': model_D.state_dict(), # 增加判别器权重保存
                    'optimizer' : optimizer.state_dict(),
                    'optimizer_D': optimizer_D.state_dict()
                    }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 
print("Now time is : ",datetime.datetime.now().isoformat())



