"""
核心子模块：INN 可逆解耦 + DCN 跨模态对齐 + OGF 前景融合 + SPR 金字塔重建
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


# ============================================================
# A. AffineCouplingLayer — 单层仿射耦合
# ============================================================

class _SubNet(nn.Module):
    """I₁ / I₂ / I₃ 共用的小型 ConvNet: Conv3x3-BN-ReLU-Conv3x3"""

    def __init__(self, channels, zero_init=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )
        if zero_init:
            # 最后一层 Conv 权重设为 0，使初始变换接近恒等
            nn.init.zeros_(self.net[-1].weight)

    def forward(self, x):
        return self.net(x)


class AffineCouplingLayer(nn.Module):
    """
    单层仿射耦合。
    输入 Φ_k 按通道分为 x1 = Φ[1:c] (前半), x2 = Φ[c+1:C] (后半)。

    Forward (先更新后半，再更新前半):
        y2 = x2 + I₁(x1)
        y1 = x1 ⊙ exp(clamp(I₂(y2))) + I₃(y2)

    Inverse:
        x1 = (y1 - I₃(y2)) ⊙ exp(-clamp(I₂(y2)))
        x2 = y2 - I₁(x1)
    """
    CLAMP = 2.0  # I₂ 输出 clamp 范围，防 exp 溢出

    def __init__(self, channels):
        """channels: 总通道数 C，内部按 C//2 拆分"""
        super().__init__()
        half = channels // 2
        self.I1 = _SubNet(half, zero_init=True)
        self.I2 = _SubNet(half, zero_init=True)
        self.I3 = _SubNet(half, zero_init=True)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)  # x1 = Φ[1:c], x2 = Φ[c+1:C]
        # 先更新后半通道（加性残差，变化小）
        y2 = x2 + self.I1(x1)
        # 再更新前半通道（乘性 + 加性，变化大）
        s = self.I2(y2).clamp(-self.CLAMP, self.CLAMP)
        y1 = x1 * torch.exp(s) + self.I3(y2)
        return torch.cat([y1, y2], dim=1)

    def inverse(self, y):
        y1, y2 = y.chunk(2, dim=1)
        s = self.I2(y2).clamp(-self.CLAMP, self.CLAMP)
        x1 = (y1 - self.I3(y2)) * torch.exp(-s)
        x2 = y2 - self.I1(x1)
        return torch.cat([x1, x2], dim=1)


# ============================================================
# B. InvertibleBlock — K 层 AffineCouplingLayer 堆叠
# ============================================================

class InvertibleBlock(nn.Module):
    """
    K 层仿射耦合堆叠的可逆块。

    channels 须为偶数；输出按前/后半通道拆分为 PΔ（变化大）与 PC（变化小）。
    """

    def __init__(self, channels, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList(
            [AffineCouplingLayer(channels) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def inverse(self, y):
        for layer in reversed(self.layers):
            y = layer.inverse(y)
        return y


# ============================================================
# C. DeltaEncoder — 全分辨率 p_delta → c5 @ H/32
# ============================================================

class DeltaEncoder(nn.Module):
    """
    将全分辨率 p_delta [B, c_in, H, W] 编码为 [B, c_out, H/32, W/32]，
    使其与 f5_fuse 同尺度同通道，可直接输入 OffsetGenerator。
    5 个 stride-2 卷积块完成 32× 空间下采样。
    """

    def __init__(self, c_in, c_out):
        super().__init__()
        mids = [16, 32, 64, 128]
        dims = [c_in] + mids + [c_out]
        layers = []
        for i in range(5):
            layers.extend([
                nn.Conv2d(dims[i], dims[i + 1], 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dims[i + 1]),
                nn.SiLU(inplace=True),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, x, target_size=None):
        out = self.net(x)
        if target_size is not None and out.shape[-2:] != target_size:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        return out


# ============================================================
# D. OffsetGenerator — 从 PΔ 和 F5_fuse 预测 S3/S4 的偏移场
# ============================================================

class OffsetGenerator(nn.Module):
    """
    利用 PΔ（差异特征）和 F5_fuse（高层语义）预测 DCN 偏移量。
    输出 Δ3 [B, 2k², H/8, W/8] 和 Δ4 [B, 2k², H/16, W/16]，k=3。
    """

    def __init__(self, c_delta, c_fuse, kernel_size=3):
        super().__init__()
        offset_channels = 2 * kernel_size * kernel_size  # 18

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(c_delta + c_fuse, c_fuse, 1, bias=False),
            nn.BatchNorm2d(c_fuse),
            nn.ReLU(inplace=True),
        )

        # branch for S4 offset: 2× upsample (H/32 → H/16)
        self.branch_s4 = nn.Sequential(
            nn.Conv2d(c_fuse, c_fuse // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_fuse // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_fuse // 2, offset_channels, 3, padding=1),
        )

        # branch for S3 offset: 再 2× upsample (H/16 → H/8)
        self.branch_s3 = nn.Sequential(
            nn.Conv2d(c_fuse, c_fuse // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_fuse // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_fuse // 2, offset_channels, 3, padding=1),
        )

        # 初始化偏移为 0
        nn.init.zeros_(self.branch_s4[-1].weight)
        nn.init.zeros_(self.branch_s4[-1].bias)
        nn.init.zeros_(self.branch_s3[-1].weight)
        nn.init.zeros_(self.branch_s3[-1].bias)

    def forward(self, p_delta, f5_fuse, size_s4, size_s3):
        """
        Args:
            p_delta: [B, C5, H/32, W/32]  PΔ 差异特征
            f5_fuse: [B, C5, H/32, W/32]  高层语义
            size_s4: (H_s4, W_s4)  S4 的空间尺寸
            size_s3: (H_s3, W_s3)  S3 的空间尺寸
        Returns:
            delta4: [B, 18, H/16, W/16]
            delta3: [B, 18, H/8, W/8]
        """
        x = self.fuse_conv(torch.cat([p_delta, f5_fuse], dim=1))

        # S4 offset
        x_s4 = F.interpolate(x, size=size_s4, mode='bilinear', align_corners=False)
        delta4 = self.branch_s4(x_s4)

        # S3 offset
        x_s3 = F.interpolate(x, size=size_s3, mode='bilinear', align_corners=False)
        delta3 = self.branch_s3(x_s3)

        return delta3, delta4


# ============================================================
# E. DCNv2Align — 可变形卷积对齐模块
# ============================================================

class DCNv2Align(nn.Module):
    """
    使用 torchvision DeformConv2d 对 RGB 低层特征进行空间对齐。
    偏移由外部 OffsetGenerator 提供。
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.dcn = DeformConv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x, offset):
        """
        Args:
            x:      [B, C, H, W]  待对齐的 RGB 特征
            offset: [B, 2*k², H, W]  偏移场
        Returns:
            对齐后的特征 [B, C, H, W]
        """
        return self.dcn(x, offset)


# ============================================================
# F. OGFModule — 前景感知融合 (Object-Guided Fusion)
# ============================================================

class OGFModule(nn.Module):
    """
    利用跨模态一致的轮廓/前景特征，输出可学习的重要性权重。
    在关键区域赋予更高响应，在背景区域自动抑制。
    """

    def __init__(self, c_feat, c_semantic):
        """
        Args:
            c_feat:     RGB/IR 特征通道数（相同）
            c_semantic: 上采样后的高层语义通道数
        """
        super().__init__()
        # 前景注意力分支
        self.fg_attn = nn.Sequential(
            nn.Conv2d(c_feat * 2, c_feat, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_feat),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_feat, c_feat, 1),
            nn.Sigmoid(),
        )

        # 语义引导投影
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(c_semantic, c_feat, 1, bias=False),
            nn.BatchNorm2d(c_feat),
            nn.SiLU(inplace=True),
        )

        # 最终融合
        self.out_conv = nn.Sequential(
            nn.Conv2d(c_feat, c_feat, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_feat),
            nn.SiLU(inplace=True),
        )

    def forward(self, aligned_rgb, ir, semantic):
        """
        Args:
            aligned_rgb: [B, C, H, W]  对齐后的 RGB 特征
            ir:          [B, C, H, W]  IR 特征
            semantic:    [B, C_sem, H, W]  上采样后的高层语义
        """
        # 前景重要性权重
        fg_weight = self.fg_attn(torch.cat([aligned_rgb, ir], dim=1))
        # 加权融合
        fused = fg_weight * aligned_rgb + (1 - fg_weight) * ir
        # 语义增强
        fused = fused + self.semantic_proj(semantic)
        return self.out_conv(fused)


# ============================================================
# G. SPRModule — 尺度感知金字塔重建
# ============================================================

class SPRModule(nn.Module):
    """对每个尺度的融合特征做精炼，输出给 Neck/Head。"""

    def __init__(self, c3, c4, c5):
        super().__init__()
        self.refine3 = nn.Sequential(
            nn.Conv2d(c3, c3, 3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.SiLU(inplace=True),
        )
        self.refine4 = nn.Sequential(
            nn.Conv2d(c4, c4, 3, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.SiLU(inplace=True),
        )
        self.refine5 = nn.Sequential(
            nn.Conv2d(c5, c5, 3, padding=1, bias=False),
            nn.BatchNorm2d(c5),
            nn.SiLU(inplace=True),
        )

    def forward(self, s3, s4, s5):
        return self.refine3(s3), self.refine4(s4), self.refine5(s5)
