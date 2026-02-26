import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian_blur(x, kernel_size=5, sigma=1.0):
    """
    手动实现高斯模糊，兼容所有 PyTorch 版本
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    
    def get_gaussian_kernel(ksize, sigma):
        # 创建 1D 高斯核
        x = torch.linspace(-(ksize // 2), ksize // 2, ksize)
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        return gauss / gauss.sum()

    # 生成 2D 高斯核
    kx = get_gaussian_kernel(kernel_size[0], sigma)
    ky = get_gaussian_kernel(kernel_size[1], sigma)
    kernel = kx[:, None] * ky[None, :]
    kernel = kernel.expand(x.shape[1], 1, kernel_size[0], kernel_size[1]).to(x.device)
    
    # 使用深度卷积 (groups=in_channels) 实现平滑
    pad = (kernel_size[0] // 2, kernel_size[1] // 2)
    return F.conv2d(x, kernel, padding=pad, groups=x.shape[1])


class AdaptiveNoiseModule0(nn.Module):
    def __init__(self, feat_dim, max_alpha=0.00015, max_beta=0.00015):
        super().__init__()
        # 物理参数上限：通过 Sigmoid 锁定，防止梯度爆炸
        self.max_alpha = max_alpha
        self.max_beta = max_beta
        
        self.param_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.LayerNorm(feat_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 4, 2)
        )
        
        # 初始化：bias 设为 0，让模型从 max_val 的一半位置开始自适应调整
        nn.init.constant_(self.param_head[-1].weight, 0)
        nn.init.constant_(self.param_head[-1].bias, 0)

    def synthesize(self, x_bg, alpha, beta, mask, noise_grain=0.05):
        """
        核心合成函数：严格保证 x_bg 在血管外不被修改
        """
        # 1. 计算基于物理模型的局部标准差 sigma = sqrt(alpha * I + beta)
        # alpha 和 beta 此时已经是映射后的物理量级
        variance = alpha * x_bg + beta
        sigma = torch.sqrt(torch.clamp(variance, min=1e-7))
        
        # 2. 生成随机噪声并模拟 X 射线质感
        noise = torch.randn_like(x_bg) 
        if noise_grain > 0:
            noise = gaussian_blur(noise, kernel_size=(7, 7), sigma=noise_grain)
            noise = noise * 1.5 

        # 3. 软化 Mask：确保噪声注入边缘平滑过度
        mask_soft = gaussian_blur(mask, kernel_size=(21, 21), sigma=5.0)
        
        # 5. 合成
        return x_bg + (noise * sigma) * mask_soft

    def forward(self, x_bg, bottleneck_feat, mask, noise_grain=0.05):
        """
        x_bg: 外部传入的已修复背景
        """
        # 1. 提取全局特征并预测原始参数
        if bottleneck_feat.dim() == 3:
            feat_global = bottleneck_feat.mean(dim=1)
        else:
            feat_global = bottleneck_feat

        params = self.param_head(feat_global)
        
        # 2. 映射到物理量级：Sigmoid 提供了平滑的渐近线，防止参数无限增长
        alpha = torch.sigmoid(params[:, 0]).view(-1, 1, 1, 1) * self.max_alpha
        beta = torch.sigmoid(params[:, 1]).view(-1, 1, 1, 1) * self.max_beta

        # 3. 调用合成函数
        noisy_output = self.synthesize(x_bg, alpha, beta, mask, noise_grain)
        
        return noisy_output, (alpha, beta)



class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_h * a_w

class AdaptiveNoiseModule1(nn.Module):
    def __init__(self, feat_dim, max_alpha=0.00015, max_beta=0.00015):
        super().__init__()
        self.max_alpha = max_alpha
        self.max_beta = max_beta

        # 1. 引入 注意力模块
        self.attention = CoordAtt(feat_dim, feat_dim)

        # 改进：解耦头 + 双倍输入维度 (Mean + Max)
        input_dim = feat_dim * 2 
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, feat_dim // 2),
            nn.LayerNorm(feat_dim // 2),
            nn.GELU()
        )
        self.alpha_head = nn.Linear(feat_dim // 2, 1)
        self.beta_head = nn.Linear(feat_dim // 2, 1)

        nn.init.constant_(self.alpha_head.bias, 1.0) # 让 sigmoid(1.0) ≈ 0.73，初始噪声较强
        nn.init.constant_(self.beta_head.bias, 1.0)

    def forward(self, x_bg, bottleneck_feat, mask, noise_grain=0.05):
        """
        x_bg: 预测的干净背景图像 [B, 1, H, W]
        bottleneck_feat: Transformer 的特征 [B, L, C] 或 CNN 的特征 [B, C, H, W]
        mask: 血管掩膜 [B, 1, H, W]
        """
        
        # --- 步骤 A: 维度适配 (3D Sequence 转 4D Map) ---
        if bottleneck_feat.dim() == 3:
            B, L, C = bottleneck_feat.shape
            # 假设特征图是正方形，还原为 2D 结构
            H_f = W_f = int(L ** 0.5) 
            feat_4d = bottleneck_feat.transpose(1, 2).view(B, C, H_f, W_f)
        else:
            B, C, H_f, W_f = bottleneck_feat.shape
            feat_4d = bottleneck_feat

        # --- 步骤 B: SOTA 注意力增强 ---
        # 使用坐标注意力捕捉血管的空间位置信息
        enhanced_feat = self.attention(feat_4d)

        # --- 步骤 C: 掩膜引导的特征池化 (考虑血管周围) ---
        # 1. 将 Mask 下采样到特征图尺寸
        mask_down = F.interpolate(mask, size=(H_f, W_f), mode='bilinear', align_corners=False)
        
        # 2. 空间扩张：通过 MaxPool 模拟膨胀，包含血管边缘及背景
        mask_dilated = F.max_pool2d(mask_down, kernel_size=3, stride=1, padding=1)
        
        # 3. 特征聚合权重的软化 (小尺度)
        mask_pool_soft = gaussian_blur(mask_dilated, kernel_size=(3, 3), sigma=1.0)
        
        # 4. 加权特征聚合：仅提取血管及其邻域的噪声相关特征
        # 计算公式: sum(feat * w) / sum(w)
        feat_masked = enhanced_feat * mask_pool_soft

        # --- 步骤 D: 物理参数预测 ---
        # 改进的特征聚合 (Mean + Max)
        f_mean = torch.sum(feat_masked, dim=(2, 3)) / (torch.sum(mask_pool_soft, dim=(2, 3)) + 1e-6)
        f_max = torch.amax(feat_masked, dim=(2, 3))
        f_combined = torch.cat([f_mean, f_max], dim=-1)
        
        # 解耦预测
        shared = self.shared_net(f_combined)
        # alpha = torch.sigmoid(self.alpha_head(shared)).view(-1, 1, 1, 1) * self.max_alpha
        # beta = torch.sigmoid(self.beta_head(shared)).view(-1, 1, 1, 1) * self.max_beta

        # 假设原来的逻辑是 params * max_alpha
        # 修改为： min + sigmoid * (max - min)
        min_alpha = self.max_alpha * 0.2 # 至少保留 10% 的强度作为底噪
        min_beta = self.max_beta * 0.2

        alpha = (torch.sigmoid(self.alpha_head(shared)) * (self.max_alpha - min_alpha) + min_alpha).view(-1, 1, 1, 1)
        beta = (torch.sigmoid(self.beta_head(shared)) * (self.max_beta - min_beta) + min_beta).view(-1, 1, 1, 1)
        
        # 执行物理合成逻辑
        noisy_output = self.synthesize(x_bg, alpha, beta, mask, noise_grain)
        
        return noisy_output, (alpha, beta)


    def synthesize(self, x_bg, alpha, beta, mask, noise_grain=0.05):
        """
        核心合成函数：严格保证 x_bg 在血管外不被修改
        """
        # 1. 计算基于物理模型的局部标准差 sigma = sqrt(alpha * I + beta)
        # alpha 和 beta 此时已经是映射后的物理量级
        variance = alpha * x_bg + beta
        sigma = torch.sqrt(torch.clamp(variance, min=1e-7))
        
        # 2. 生成随机噪声并模拟 X 射线质感
        noise = torch.randn_like(x_bg) 
        if noise_grain > 0:
            noise = gaussian_blur(noise, kernel_size=(7, 7), sigma=noise_grain)
            noise = noise * 1.5 

        # 3. 软化 Mask：确保噪声注入边缘平滑过度
        mask_soft = gaussian_blur(mask, kernel_size=(21, 21), sigma=5.0)
        
        # 5. 合成
        return x_bg + (noise * sigma) * mask_soft


class AdaptiveNoiseModule(nn.Module):
    def __init__(self, feat_dim, max_alpha=0.00015, max_beta=0.00015):
        super().__init__()
        self.max_alpha = max_alpha
        self.max_beta = max_beta

        # 1. 坐标注意力
        self.attention = CoordAtt(feat_dim, feat_dim)

        # 2. 解耦头
        input_dim = feat_dim * 2 
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, feat_dim // 2),
            nn.LayerNorm(feat_dim // 2),
            nn.GELU()
        )
        self.alpha_head = nn.Linear(feat_dim // 2, 1)
        self.beta_head = nn.Linear(feat_dim // 2, 1)

        nn.init.constant_(self.alpha_head.bias, 1.0) 
        nn.init.constant_(self.beta_head.bias, 1.0)

    def forward(self, x_bg, bottleneck_feat, mask, noise_grain=0.05):
        """
        x_bg: [B, 3, H, W] 输入是 3 通道
        """
        # --- 步骤 A: 维度适配 (3D Sequence 转 4D Map) ---
        if bottleneck_feat.dim() == 3:
            B, L, C = bottleneck_feat.shape
            # 假设特征图是正方形，还原为 2D 结构
            H_f = W_f = int(L ** 0.5) 
            feat_4d = bottleneck_feat.transpose(1, 2).view(B, C, H_f, W_f)
        else:
            B, C, H_f, W_f = bottleneck_feat.shape
            feat_4d = bottleneck_feat

        # --- 步骤 B: SOTA 注意力增强 ---
        # 使用坐标注意力捕捉血管的空间位置信息
        enhanced_feat = self.attention(feat_4d)

        # --- 步骤 C: 掩膜引导的特征池化 (考虑血管周围) ---
        # 1. 将 Mask 下采样到特征图尺寸
        mask_down = F.interpolate(mask, size=(H_f, W_f), mode='bilinear', align_corners=False)
        
        # 2. 空间扩张：通过 MaxPool 模拟膨胀，包含血管边缘及背景
        mask_dilated = F.max_pool2d(mask_down, kernel_size=3, stride=1, padding=1)
        
        # 3. 特征聚合权重的软化 (小尺度)
        mask_pool_soft = gaussian_blur(mask_dilated, kernel_size=(3, 3), sigma=1.0)
        
        # 4. 加权特征聚合：仅提取血管及其邻域的噪声相关特征
        # 计算公式: sum(feat * w) / sum(w)
        feat_masked = enhanced_feat * mask_pool_soft

        # --- 步骤 D: 物理参数预测 ---
        # 改进的特征聚合 (Mean + Max)
        f_mean = torch.sum(feat_masked, dim=(2, 3)) / (torch.sum(mask_pool_soft, dim=(2, 3)) + 1e-6)
        f_max = torch.amax(feat_masked, dim=(2, 3))
        f_combined = torch.cat([f_mean, f_max], dim=-1)

        # --- 步骤 D: 物理参数预测 ---
        # 即使输入是 3 通道，alpha 和 beta 依然是标量参数 [B, 1, 1, 1]
        shared = self.shared_net(f_combined)
        
        min_alpha = self.max_alpha * 0.2 
        min_beta = self.max_beta * 0.2

        alpha = (torch.sigmoid(self.alpha_head(shared)) * (self.max_alpha - min_alpha) + min_alpha).view(-1, 1, 1, 1)
        beta = (torch.sigmoid(self.beta_head(shared)) * (self.max_beta - min_beta) + min_beta).view(-1, 1, 1, 1)
        
        # --- 步骤 E: 执行物理合成 ---
        # 核心逻辑：在 synthesize 内部处理 3 通道广播
        noisy_output = self.synthesize(x_bg, alpha, beta, mask, noise_grain)
        
        return noisy_output, (alpha, beta)

    def synthesize0(self, x_bg, alpha, beta, mask, noise_grain=0.05):
        """
        核心逻辑：生成单通道随机噪声，强制广播到 3 通道输入上
        """
        B, C, H, W = x_bg.shape
        
        # 1. 物理方差计算 (此时 variance 为 [B, 3, H, W])
        variance = alpha * x_bg + beta
        sigma = torch.sqrt(torch.clamp(variance, min=1e-7))
        
        # 2. 【关键】生成严格的单通道随机噪声 [B, 1, H, W]
        # 这样可以确保随机波动的“形态”在 R/G/B 上完全一致
        noise_gray = torch.randn(B, 1, H, W).to(x_bg.device)
        
        # 3. 模拟颗粒感 (在单通道上处理，效率更高)
        if noise_grain > 0:
            noise_gray = gaussian_blur(noise_gray, kernel_size=(7, 7), sigma=noise_grain)
            noise_gray = noise_gray * 1.5 

        # 4. 掩膜软化 [B, 1, H, W]
        mask_soft = gaussian_blur(mask, kernel_size=(21, 21), sigma=5.0)
        
        # 5. 【广播合成】
        # noise_gray [B, 1, H, W] 会自动广播适配 x_bg [B, 3, H, W]
        # 这保证了每个通道增加的噪声在每个像素点上数值完全相同
        res_noisy = x_bg + (noise_gray * sigma) * mask_soft
        
        return res_noisy
    

    # 将 beta 也纳入掩膜控制（物理最合理）修改物理公式，让底噪也只在血管及其邻域产生
    # 避免NSPSNR过低，同时保持背景区域的干净
    def synthesize(self, x_bg, alpha, beta, mask, noise_grain=0.05):
        B, C, H, W = x_bg.shape
        
        # 1. 软化掩膜
        mask_soft = gaussian_blur(mask, kernel_size=(15, 15), sigma=3.0)
        
        # 2. 物理方差计算：将 mask 整合进方差，确保背景 variance 极小
        # 这样 alpha * x_bg 和基础底噪 beta 都不会溢出到远端背景
        variance = (alpha * x_bg + beta) * mask_soft
        sigma = torch.sqrt(torch.clamp(variance, min=1e-7))
        
        # 3. 生成单通道噪声
        noise_gray = torch.randn(B, 1, H, W).to(x_bg.device)
        if noise_grain > 0:
            noise_gray = gaussian_blur(noise_gray, kernel_size=(7, 7), sigma=noise_grain)
            noise_gray = noise_gray * 1.5 
        
        # 4. 广播合成
        # 此时 sigma 已经包含了 mask 信息，背景处的 sigma 接近 0
        res_noisy = x_bg + (noise_gray * sigma)
        
        return res_noisy