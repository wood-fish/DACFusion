"""
DACNet_INN_mul: 全分辨率 INN 可逆解耦 + 高层语义引导对齐 + 多尺度融合 主干网络

流程:
  全分辨率 INN(cat(RGB,IR)) → PΔ/PC
  独立双 DACNet → 高层语义融合(F5_fuse)
  DeltaEncoder(PΔ) + F5_fuse → OffsetPrediction → DCN 对齐 → OGF 融合 → SPR → [P3, P4, P5]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.backbone.DACnet_mul import (
    DACNet, SimpleFusion, ConvBNAct, update_weight,
)
from ultralytics.nn.backbone.inn_modules import (
    InvertibleBlock, DeltaEncoder, OffsetGenerator, DCNv2Align, OGFModule, SPRModule,
)


class DACNet_INN_mul(nn.Module):
    """
    输入:  x_rgb [B, 3, H, W],  x_ir [B, 3, H, W]
    输出:
        训练时: (features_list, aux_dict)
        推理时: features_list  = [P3, P4, P5]
    """

    def __init__(
        self,
        rgb_weights='',
        ir_weights='',
        backbone_type='s',
        img_size=640,
        inn_num_layers=3,
    ):
        super().__init__()

        # -------- 1. 两个独立 DACNet 单模态主干 --------
        if backbone_type == 's':
            cfg = dict(
                img_size=img_size,
                embed_dims=[64, 128, 320, 512],
                depths=[2, 2, 1, 1],
                drop_rate=0.1,
                drop_path_rate=0.1,
            )
            c3, c4, c5 = 128, 320, 512
        elif backbone_type == 't':
            cfg = dict(
                img_size=img_size,
                embed_dims=[32, 64, 160, 256],
                depths=[3, 3, 5, 2],
                drop_rate=0.1,
                drop_path_rate=0.1,
            )
            c3, c4, c5 = 64, 160, 256
        else:
            raise ValueError(f'Unsupported backbone_type: {backbone_type}')

        self.c3, self.c4, self.c5 = c3, c4, c5

        self.rgb_backbone = DACNet(**cfg)
        self.ir_backbone = DACNet(**cfg)

        if rgb_weights:
            ckpt = torch.load(rgb_weights, map_location='cpu')
            sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            self.rgb_backbone.load_state_dict(
                update_weight(self.rgb_backbone.state_dict(), sd), strict=False
            )
        if ir_weights:
            ckpt = torch.load(ir_weights, map_location='cpu')
            sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            self.ir_backbone.load_state_dict(
                update_weight(self.ir_backbone.state_dict(), sd), strict=False
            )

        # -------- 2. 高层语义融合 --------
        self.fuse5 = SimpleFusion(c5, c5, c5)

        # -------- 3. 全分辨率 INN 可逆解耦 (Stem 6→12 + InvertibleBlock C=12) --------
        self.inn_channels = 12
        self.inn_half = self.inn_channels // 2   # p_delta=6, p_shared=6
        self.inn_stem = ConvBNAct(6, self.inn_channels, k=3, s=1)
        self.inn = InvertibleBlock(channels=self.inn_channels, num_layers=inn_num_layers)

        # -------- 4. DeltaEncoder: p_delta(6ch,H,W) → (c5,H/32,W/32) --------
        self.delta_encoder = DeltaEncoder(c_in=self.inn_half, c_out=c5)

        # -------- 5. 偏移预测 --------
        self.offset_gen = OffsetGenerator(c_delta=c5, c_fuse=c5, kernel_size=3)

        # -------- 6. DCN 对齐 --------
        self.dcn_s3 = DCNv2Align(channels=c3, kernel_size=3)
        self.dcn_s4 = DCNv2Align(channels=c4, kernel_size=3)

        # -------- 7. 语义下采样投影 --------
        self.reduce5_to_4 = ConvBNAct(c5, c4, k=1, s=1)
        self.reduce5_to_3 = ConvBNAct(c5, c3, k=1, s=1)

        # -------- 8. OGF 前景融合 --------
        self.ogf_s3 = OGFModule(c_feat=c3, c_semantic=c3)
        self.ogf_s4 = OGFModule(c_feat=c4, c_semantic=c4)
        self.ogf_s5 = OGFModule(c_feat=c5, c_semantic=c5)

        # -------- 9. SPR 金字塔重建 --------
        self.spr = SPRModule(c3, c4, c5)

        # -------- 10. 输出通道（供 YAML parse_model 读取）--------
        self.channel = [c3, c4, c5]

    def forward(self, x_rgb, x_ir):
        # --- Step 1: 全分辨率 INN 可逆解耦 (Stem 6→12 + InvertibleBlock) ---
        inn_in6 = torch.cat([x_rgb, x_ir], dim=1)         # [B, 6, H, W]
        inn_input = self.inn_stem(inn_in6)                 # [B, 12, H, W]
        phi = self.inn(inn_input)                          # [B, 12, H, W]
        p_delta = phi[:, :self.inn_half]                   # PΔ [B, 6, H, W]
        p_shared = phi[:, self.inn_half:]                  # PC [B, 6, H, W]
        pc_rgb_half = p_shared[:, :3]                      # Φ_shared^rgb [B, 3, H, W]
        pc_ir_half = p_shared[:, 3:]                       # Φ_shared^ir  [B, 3, H, W]

        # --- Step 2: 独立双主干提取 ---
        rgb_feats = self.rgb_backbone(x_rgb)               # [s2, s3, s4, s5]
        ir_feats = self.ir_backbone(x_ir)

        s3_rgb, s4_rgb, s5_rgb = rgb_feats[1], rgb_feats[2], rgb_feats[3]
        s3_ir, s4_ir, s5_ir = ir_feats[1], ir_feats[2], ir_feats[3]

        # --- Step 3: 高层语义融合 ---
        f5_fuse = self.fuse5(s5_rgb, s5_ir)                # [B, c5, H/32, W/32]

        # --- Step 4: DeltaEncoder 桥接 + 偏移预测 + DCN 对齐 ---
        p_delta_s5 = self.delta_encoder(
            p_delta, target_size=f5_fuse.shape[-2:]
        )                                                   # [B, c5, H/32, W/32]

        delta3, delta4 = self.offset_gen(
            p_delta_s5, f5_fuse,
            size_s4=s4_rgb.shape[-2:],
            size_s3=s3_rgb.shape[-2:],
        )
        s3_aligned = self.dcn_s3(s3_rgb, delta3)
        s4_aligned = self.dcn_s4(s4_rgb, delta4)

        # --- Step 5: OGF 前景加权多尺度融合 ---
        sem_s4 = F.interpolate(
            self.reduce5_to_4(f5_fuse), size=s4_rgb.shape[-2:], mode='bilinear', align_corners=False
        )
        sem_s3 = F.interpolate(
            self.reduce5_to_3(f5_fuse), size=s3_rgb.shape[-2:], mode='bilinear', align_corners=False
        )

        out_s3 = self.ogf_s3(s3_aligned, s3_ir, sem_s3)
        out_s4 = self.ogf_s4(s4_aligned, s4_ir, sem_s4)
        out_s5 = self.ogf_s5(s5_rgb, s5_ir, f5_fuse)

        # --- Step 6: SPR 金字塔重建 ---
        p3, p4, p5 = self.spr(out_s3, out_s4, out_s5)

        features = [p3, p4, p5]

        if self.training:
            inn_rec = self.inn.inverse(phi)
            aux = {
                'inn_input': inn_input,           # [B, 12, H, W]
                'inn_reconstruction': inn_rec,    # [B, 12, H, W]
                'pc_rgb_half': pc_rgb_half,       # [B, 3, H, W]
                'pc_ir_half': pc_ir_half,         # [B, 3, H, W]
                's3_aligned': s3_aligned,
                's3_ir': s3_ir,
                's4_aligned': s4_aligned,
                's4_ir': s4_ir,
            }
            return features, aux

        return features


# ============================================================
# 工厂函数（供 YAML 配置引用）
# ============================================================

def DACnet_inn_s_mul(weights=''):
    """Small 版本: embed_dims=[64, 128, 320, 512], depths=[2, 2, 1, 1]"""
    return DACNet_INN_mul(
        rgb_weights=weights, ir_weights=weights,
        backbone_type='s', img_size=640, inn_num_layers=3,
    )


def DACnet_inn_t_mul(weights=''):
    """Tiny 版本: embed_dims=[32, 64, 160, 256], depths=[3, 3, 5, 2]"""
    return DACNet_INN_mul(
        rgb_weights=weights, ir_weights=weights,
        backbone_type='t', img_size=640, inn_num_layers=3,
    )
