import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair as to_2tuple
from timm.layers import DropPath
from functools import partial
import numpy as np

__all__ = 'DACnet_t_mul', 'DACnet_s_mul','DACnet_t','DACnet_s'

patch_size = 7
stride=4

conv0_k=5
conv0_p=2
conv1_k=7
conv1_p=9
conv1_d=3

conv_squeeze_k = 7
conv_squeeze_p = 3


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AlignModule(nn.Module):
    def __init__(self, in_channels, device):
        super(AlignModule, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels,device=device)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(x))

class DACblock_mul(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=conv0_k, padding=conv0_p, groups=dim)
        self.conv0_ir = nn.Conv2d(dim, dim, kernel_size=conv0_k, padding=conv0_p, groups=dim)
        self.conv_spatial_ir = nn.Conv2d(dim, dim, kernel_size=conv1_k, stride=1, padding=conv1_p, groups=dim,dilation=conv1_d)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv1_ir = nn.Conv2d(dim, dim // 2, 1)
        self.conv2_ir = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, conv_squeeze_k, padding=conv_squeeze_p)
        self.conv = nn.Conv2d(dim // 2, dim, 1)
        self.attention = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim, 1),
            nn.Sigmoid()
        )
        self.conv_concatation = nn.Conv2d(dim,dim//2,1)

    def forward(self, x, x_ir):
        attn1 = self.conv0(x)
        attn1_ir = self.conv0(x_ir)
        attn1_ir = self.conv_spatial_ir(attn1_ir)

        attn1 = self.conv1(attn1)
        attn1_ir = self.conv1_ir(attn1_ir)
        align_module = AlignModule(in_channels=attn1.shape[1],device = attn1.device)
        attn1 = align_module(attn1)
        attn1_ir = align_module(attn1_ir)

        attn = torch.cat([attn1, attn1_ir], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = (attn1) * sig[:, 0, :, :].unsqueeze(1) + (attn1_ir) * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


class DACblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=conv0_k, padding=conv0_p, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=conv1_k, stride=1, padding=conv1_p, groups=dim, dilation=conv1_d)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, conv_squeeze_k, padding=conv_squeeze_p)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn

class Attention_mul(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = DACblock_mul(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x, x_ir):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x, x_ir)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = DACblock(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

class Block_mul(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_cfg=None):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.norm1_ir = nn.BatchNorm2d(dim)
        self.norm2_ir = nn.BatchNorm2d(dim)
        self.attn_mul = Attention_mul(dim)
        self.attn_mul2 = Attention_mul(dim)
        self.attn= Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_ir = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_ir = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_1_ir = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2_ir = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, x_ir):
        norm1x = self.norm1(x)
        norm1_irx = self.norm1_ir(x_ir)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn_mul(norm1x, norm1_irx))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        x_ir = x_ir + self.drop_path_ir(self.layer_scale_1_ir.unsqueeze(-1).unsqueeze(-1) * self.attn_mul2(norm1_irx,norm1x))
        # x_ir = x_ir + self.drop_path_ir(self.layer_scale_1_ir.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1_ir(x_ir)))
        x_ir = x_ir + self.drop_path_ir(self.layer_scale_1_ir.unsqueeze(-1).unsqueeze(-1) * self.mlp_ir(self.norm2_ir(x_ir)))
        return x, x_ir

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_cfg=None):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W



class AttentionFusionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rgb_conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.ir_conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.attention = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim * 2, 1),
            nn.Sigmoid()
        )
        self.fusion_conv = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x_rgb, x_ir):
        x_rgb = self.rgb_conv(x_rgb)
        x_ir = self.ir_conv(x_ir)
        x_fused = torch.cat([x_rgb, x_ir], dim=1)
        attention = self.attention(x_fused)
        x_fused = x_fused * attention
        x_fused = self.fusion_conv(x_fused)
        return x_fused


class DACNet_mul(nn.Module):
    def __init__(self, img_size=1024, in_chans=3, embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[8, 8, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], num_stages=4,
                 norm_cfg=None):
        super().__init__()

        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        dims = embed_dims

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=patch_size if i == 0 else 3,#patch_size=7 if i == 0 else 3
                                            stride=stride if i == 0 else 2 ,#stride=4 if i == 0 else 2
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i], norm_cfg=norm_cfg)
            patch_embed_ir = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=patch_size if i == 0 else 3,#patch_size=7 if i == 0 else 3
                                            stride=stride if i == 0 else 2 ,#stride=4 if i == 0 else 2
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i], norm_cfg=norm_cfg)
            self.atten = AttentionFusionBlock(dims[i])
            if i>=2:
                block_mul = nn.ModuleList([Block_mul(
                    dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j], norm_cfg=norm_cfg)
                    for j in range(depths[i])])
            else:
                block = nn.ModuleList([Block(
                    dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j], norm_cfg=norm_cfg)
                    for j in range(depths[i])])
                block_ir = nn.ModuleList([Block(
                    dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j], norm_cfg=norm_cfg)
                    for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            norm_ir = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"patch_embed_ir{i + 1}", patch_embed_ir)
            if i>=2:
                setattr(self, f"block_mul{i + 1}", block_mul)
            else:
                setattr(self, f"block{i + 1}", block)
                setattr(self, f"block_ir{i + 1}", block_ir)
            setattr(self, f"norm{i + 1}", norm)
            setattr(self, f"norm_ir{i + 1}", norm_ir)



        self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, img_size, img_size), torch.randn(1, 3, img_size, img_size))]

    def forward(self, x, x_ir):
        # if x is None or x_ir is None:
        #     raise ValueError("Input tensors should not be None")
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            patch_embed_ir = getattr(self, f"patch_embed_ir{i + 1}")
            if i>=2:
                block_mul = getattr(self, f"block_mul{i + 1}")
            else:
                block = getattr(self, f"block{i + 1}")
                block_ir = getattr(self, f"block_ir{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            norm_ir = getattr(self, f"norm_ir{i + 1}")


            x, H, W = patch_embed(x)
            x_ir, _, _ = patch_embed_ir(x_ir)
            if i==0 or i==1: # or i==1
                for blk in block:
                    x = blk(x)
                for blk in block_ir:
                    x_ir = blk(x_ir)
            else:
                for blk in block_mul:
                    x, x_ir = blk(x, x_ir)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x_ir = x_ir.flatten(2).transpose(1, 2)
            x_ir = norm_ir(x_ir)
            x_ir = x_ir.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            if i == self.num_stages-1:
                x = self.atten(x, x_ir)
            outs.append(x)
        return outs

    def visualize_feature_map(self, feature_map, title):
        import matplotlib.pyplot as plt
        feature_map = feature_map[0].cpu().detach()
        feature_map = feature_map[0].numpy()

        plt.imshow(feature_map, cmap='viridis')
        plt.title(title)
        plt.colorbar()
        plt.show()


class DACNet(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[8, 8, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], num_stages=4,
                 norm_cfg=None):
        super().__init__()

        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i], norm_cfg=norm_cfg)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j], norm_cfg=norm_cfg)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]

    def forward(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


def update_weight(model_dict, weight_dict):
    idx, temp_dict = 0, {}
    for k, v in weight_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            idx += 1
    model_dict.update(temp_dict)
    print(f'loading weights... {idx}/{len(model_dict)} items')
    return model_dict



def DACnet_t_mul(weights='/home/qjc/Project/yolov10-main/DAC_t_backbone.pth.tar'):
    model = DACNet_mul(embed_dims=[32, 64, 160, 256], depths=[3, 3, 5, 2], drop_rate=0.1, drop_path_rate=0.1)
    if weights:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(weights)['state_dict']))
    return model


def DACnet_s_mul(weights=''):   #/home/qjc/Project/yolov10-main/DAC_s_backbone.pth
    model = DACNet_mul(embed_dims=[64, 128, 256, 512], depths=[2, 2, 1, 1], drop_rate=0.1, drop_path_rate=0.1)
    if weights:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(weights)['state_dict']))
    return model

def DACnet_t(weights='/home/qjc/Project/yolov10-main/DAC_t_backbone.pth.tar'):
    model = DACNet(embed_dims=[32, 64, 160, 256], depths=[3, 3, 5, 2], drop_rate=0.1, drop_path_rate=0.1)
    if weights:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(weights)['state_dict']))
    return model


def DACnet_s(weights=''): #/home/qjc/Project/yolov10-main/DAC_s_backbone.pth
    model = DACNet(embed_dims=[64, 128, 256, 512], depths=[2, 2, 4, 2], drop_rate=0.1, drop_path_rate=0.1)
    if weights:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(weights)['state_dict']))
    return model

if __name__ == '__main__':
    model = DACnet_t_mul('/home/qjc/Project/yolov10-main/DAC_t_backbone.pth.tar')
    inputs = torch.randn((1, 3, 1024, 1024))
    inputs_ir = torch.randn((1, 3, 1024, 1024))
    # inputs = torch.randn((1, 3, 640, 640))
    # inputs_ir = torch.randn((1, 3, 640, 640))
    for i in model(inputs, inputs_ir):
        print(i.size())
