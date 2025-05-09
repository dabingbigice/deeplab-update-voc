import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch.einops import rearrange
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, convnext_base, ConvNeXt_Base_Weights

from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2
import torch
import torch.nn as nn
from functools import partial
from nets.LRSAmodule import LRSA
from nets.deeplab_startnet import StarNet
from nets.WTConv_module import WTConv2d

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import torch
import torch.nn as nn
from torchvision.models import resnext50_32x4d

import torch
import torch.nn as nn
import timm  # 需要安装 timm 库: pip install timm

import torch
import torch.nn as nn
import timm

import torch
import torch.nn as nn
import timm  # 确保 timm 版本 >= 0.6.11

import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer


# Swin-Tiny

class SwinTransformer_Encoder(nn.Module):
    def __init__(self, pretrained=False, model_name='swin_tiny_patch4_window7_224'):
        super().__init__()

        # 加载预训练Swin Transformer主干 [1,6](@ref)
        self.model = SwinTransformer(
            img_size=512,
            patch_size=4,
            in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],  # 各阶段块数配置
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.2,
            pretrained=pretrained,
            ape=True,
        )

        # 分解模型结构 -------------------------------------------------
        # Stem层替换为Patch Embedding [6,8](@ref)
        self.stem = nn.Sequential(
            self.model.patch_embed,  # 4x4卷积将图像分割为56x56 patches (输出56x56x96)
        )

        # 特征阶段划分 (对应原EfficientNet的四阶段)
        self.stage1 = self.model.layers[0]  # 56x56 -> 56x56 (保持分辨率)
        self.stage2 = self.model.layers[1]  # 56x56 -> 28x28 (下采样)
        self.stage3 = self.model.layers[2]  # 28x28 -> 14x14
        self.stage4 = self.model.layers[3]  # 14x14 -> 7x7

        # 通道调整层 (适配原编码器输出维度)
        self.adjust_x = nn.Conv2d(self.model.num_features, 96, 1)  # 768→96
        self.adjust_low = nn.Conv2d(96, 12, 1)  # stem输出通道调整

    def forward(self, x):
        # Stem处理 (保持[B, H, W, C]格式)
        x = self.stem(x)  # 输出[B, 128, 128, 96]（img_size=512时）

        # 多级特征提取（无需permute）
        s1 = self.stage1(x)  # 输入应为[B, H, W, C]
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        x = self.stage4(s3)

        # 通道调整前统一维度顺序
        x = x.permute(0, 3, 1, 2)  # [B, 768, 7, 7]（假设最终H/W=7）
        low_level = s1.permute(0, 3, 1, 2)  # [B, 96, 128, 128]

        x = self.adjust_x(x)
        low_level = self.adjust_low(low_level)
        return low_level, x


#
class Res2Net_Encoder(nn.Module):
    def __init__(self, pretrained=False, model_name='res2net50_26w_4s'):
        super().__init__()

        # 使用 timm 的正确接口加载 Res2Net_97.53 [9](@ref)
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            num_classes=0  # 禁用分类头
        )
        print(self.model)
        # 分解模型结构 [7](@ref)
        self.stem = nn.Sequential(
            self.model.conv1,  # 初始7x7卷积 (输出64通道)
            self.model.bn1,
            nn.ReLU(inplace=True),
            self.model.maxpool  # 最大池化层 (输出1/4分辨率)
        )

        # 提取各阶段特征层
        self.stage1 = self.model.layer1  # stride=1 (输出1/4分辨率)
        self.stage2 = self.model.layer2  # stride=2 (输出1/8分辨率)
        self.stage3 = self.model.layer3  # stride=2 (输出1/16分辨率)
        self.stage4 = self.model.layer4  # stride=2 (输出1/32分辨率)

        # 通道调整层
        self.adjust_x = nn.Conv2d(2048, 96, 1)  # 调整高级特征通道数
        self.adjust_low = nn.Conv2d(256, 12, 1)  # 调整低级特征通道数

    def forward(self, x):
        # Stem处理 (1/4分辨率)
        x = self.stem(x)  # [B, 64, H/4, W/4]

        # 多级特征提取
        s1 = self.stage1(x)  # [B, 256, H/4, W/4]
        s2 = self.stage2(s1)  # [B, 512, H/8, W/8]
        s3 = self.stage3(s2)  # [B, 1024, H/16, W/16]
        x = self.stage4(s3)  # [B, 2048, H/32, W/32]

        # 通道维度调整
        x = self.adjust_x(x)  # 2048→512
        low_level = self.adjust_low(s1)  # 256→64

        return low_level, x


class ConvNeXt(nn.Module):
    def __init__(self, downsample_factor=16, pretrained=False):
        super().__init__()
        # 初始化ConvNeXt基础模型（基于torchvision官方实现）
        model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None)

        # 修正特征阶段划分方式
        self.stage1 = nn.Sequential(
            model.features[0],  # 初始4x4卷积下采样
            model.features[1]  # 直接使用完整Stage1特征
        )
        self.stage2 = model.features[2]
        self.stage3 = model.features[3]
        self.stage4 = model.features[4]

        # 通道适配模块（关键设计点）
        self.adjust_low = nn.Sequential(
            nn.Conv2d(128, 12, 1),  # Stage1输出通道调整
        )
        self.adjust_high = nn.Sequential(
            nn.Conv2d(512, 96, 1),  # Stage4输出通道调整
        )

    def forward(self, x):
        # 四阶段特征提取
        s1 = self.stage1(x)  # [B,128,H/4,W/4]
        s2 = self.stage2(s1)  # [B,256,H/8,W/8]
        s3 = self.stage3(s2)  # [B,512,H/16,W/16]
        s4 = self.stage4(s3)  # [B,768,H/32,W/32]

        # 通道调整与特征选择
        return self.adjust_low(s1), self.adjust_high(s4)


# train
class ResNeXtEncoder(nn.Module):
    def __init__(self, downsample_factor=16, pretrained=True):
        super(ResNeXtEncoder, self).__init__()

        # 加载预训练ResNeXt主干
        model = resnext50_32x4d(pretrained=pretrained)

        # 特征层分解策略（根据ResNeXt50结构）
        self.low_level_features = nn.Sequential(
            model.conv1,  # 初始卷积层 (输出64通道)
            model.bn1,
            model.relu,
            model.maxpool  # 最大池化层
        )

        self.stage2 = model.layer1  # Stage2输出256通道
        self.stage3 = model.layer2  # Stage3输出512通道（下采样）
        self.stage4 = model.layer3  # Stage4输出1024通道（下采样）
        self.stage5 = model.layer4  # Stage5输出2048通道（下采样）

        # 通道适配层（匹配DeepLabv3+解码器输入要求）
        self.adjust_x = nn.Conv2d(2048, 96, 1)  # 高层特征通道调整
        self.adjust_low = nn.Conv2d(256, 12, 1)  # 低层特征通道调整

    def forward(self, x):
        # 低级特征提取（原图1/4分辨率）
        low_level = self.low_level_features(x)  # [B,64,H/4,W/4]
        s2 = self.stage2(low_level)  # [B,256,H/4,W/4]

        # 高级语义特征（原图1/16或1/32分辨率）
        s3 = self.stage3(s2)  # [B,512,H/8,W/8]
        s4 = self.stage4(s3)  # [B,1024,H/16,W/16]
        x = self.stage5(s4)  # [B,2048,H/32,W/32]

        # 通道维度适配
        x = self.adjust_x(x)  # 2048→256
        low_level = self.adjust_low(s2)  # 256→64

        return low_level, x


# train
class EfficientNetB0Encoder(nn.Module):
    def __init__(self, pretrained=False):
        super(EfficientNetB0Encoder, self).__init__()

        # 加载预训练模型并提取特征层 [6,8](@ref)
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        features = model.features

        # 网络结构分解策略
        self.low_level_layers = nn.Sequential(
            features[0],  # 初始卷积层 (Conv3x3)
            features[1],  # MBConv1 (stride2下采样)
            features[2]  # MBConv2 (通道扩展)
        )

        # 深层特征提取层 [1,6](@ref)
        self.high_level_layers = nn.Sequential(
            features[3],  # MBConv3 (stage3)
            features[4],  # MBConv4 (stage4)
            features[5],  # MBConv5 (stage5)
            features[6],  # MBConv6 (stage6)
            features[7],  # MBConv7 (stage7)
            features[8]  # MBConv8 (stage8)
        )

        # 通道调整层（保持与MobileNetV2相同接口）[2,5](@ref)
        self.adjust_x = nn.Conv2d(1280, 96, 1)  # 调整高级特征通道数
        self.adjust_low = nn.Conv2d(24, 12, 1)  # 调整低级特征通道数

    def forward(self, x):
        # 低级特征（分辨率原图的1/4）[6,8](@ref)
        low_level = self.low_level_layers(x)  # 输出形状: [B, 24, H/4, W/4]

        # 高级语义特征（分辨率根据下采样因子调整）
        x = self.high_level_layers(low_level)  # 默认输出形状: [B, 320, H/32, W/32]

        # 通道调整 [2,5](@ref)
        x = self.adjust_x(x)  # 1280 -> 96
        low_level = self.adjust_low(low_level)  # 24 -> 12
        return low_level, x


import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights


class ShuffleNetV2_Encoder(nn.Module):
    def __init__(self, pretrained=False):
        super(ShuffleNetV2_Encoder, self).__init__()

        # 加载0.5x版本的预训练模型 [2,5](@ref)
        model = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1 if pretrained else None)

        # 特征层分解策略 [1,6](@ref)
        self.stage1 = nn.Sequential(
            model.conv1,  # 初始卷积层 (输出24通道)
            model.maxpool  # 最大池化层
        )
        self.stage2 = model.stage2  # stride=2的下采样模块
        self.stage3 = model.stage3  # stride=2的下采样模块
        self.stage4 = model.stage4  # 最终特征提取层

        # 通道调整层 [5,8](@ref)
        self.adjust_x = nn.Conv2d(192, 96, 1)  # 调整高级特征通道数(原stage4输出192通道)
        self.adjust_low = nn.Conv2d(24, 12, 1)  # 调整低级特征通道数(原stage1输出24通道)

    def forward(self, x):
        # 低级特征提取 (原图1/4分辨率) [2,6](@ref)
        low_level = self.stage1(x)  # 输出形状: [B, 24, H/4, W/4]

        # 多级特征提取
        s2 = self.stage2(low_level)  # stride2下采样 (1/8)
        s3 = self.stage3(s2)  # stride2下采样 (1/16)
        x = self.stage4(s3)  # 最终输出 (1/32)

        # 通道维度调整 [5,8](@ref)
        x = self.adjust_x(x)  # 192 -> 96
        low_level = self.adjust_low(low_level)  # 24 -> 12

        return low_level, x


import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(ResNet18, self).__init__()
        # 加载预训练ResNet18主干网络
        resnet = models.resnet18(pretrained=pretrained)

        # 提取特征层（去除最后两层：avgpool和fc）
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # 输出通道64
            resnet.layer2,  # 输出通道128
            resnet.layer3,  # 输出通道256
            resnet.layer4  # 输出通道512
        )

        # 空洞卷积配置（适配不同下采样因子）
        self.down_idx = [3, 4, 5, 6]  # 对应layer1~layer4的索引位置
        # if downsample_factor == 8:
        #     # 对layer3和layer4设置空洞卷积
        #     self._set_dilation(self.features[5], dilate=0)  # layer3
        #     self._set_dilation(self.features[6], dilate=0)  # layer4
        # elif downsample_factor == 16:
        #     self._set_dilation(self.features[6], dilate=0)  # layer4

        # 特征维度调整层（适配ResNet18输出通道）
        self.adjust_x = nn.Conv2d(512, 96, 1)  # 调整高层特征通道
        self.adjust_low = nn.Conv2d(64, 12, 1)  # 调整低层特征通道

    def _set_dilation(self, layer, dilate):
        """修改残差块的卷积参数实现空洞卷积"""
        for block in layer.children():
            if isinstance(block, models.resnet.BasicBlock):
                # 调整两个卷积层的dilation参数
                for conv in [block.conv1, block.conv2]:
                    if conv.kernel_size == (3, 3):
                        conv.dilation = (dilate, dilate)
                        conv.padding = (dilate, dilate)
                # 调整跳跃连接的卷积（如果存在）
                if block.downsample is not None:
                    block.downsample[0].stride = (1, 1)

    def forward(self, x):
        # 提取低层特征（layer1输出）
        low_level_features = self.features[:4](x)  # [conv1,bn1,relu,maxpool]
        # 提取高层特征
        x = self.features[4:](low_level_features)  # 经过layer1~layer4

        # 调整特征维度
        x = self.adjust_x(x)
        low_level_features = self.adjust_low(low_level_features)

        return low_level_features, x


class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()

        model = mobilenetv2(False)
        self.features = model.features[:-1]

        self.adjust_x = nn.Conv2d(320, 96, 1)
        self.adjust_low = nn.Conv2d(24, 12, 1)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        x = self.adjust_x(x)
        low_level_features = self.adjust_low(low_level_features)
        return low_level_features, x


def patch_divide(x, step, ps):
    """Crop image into patches."""
    b, c, h, w = x.size()
    if h == ps and w == ps:
        step = ps
    crop_x = []
    nh = 0
    for i in range(0, h + step - ps, step):
        down = min(i + ps, h)
        top = down - ps
        nh += 1
        for j in range(0, w + step - ps, step):
            right = min(j + ps, w)
            left = right - ps
            crop_x.append(x[:, :, top:down, left:right])
    nw = len(crop_x) // nh
    crop_x = torch.stack(crop_x, dim=0)
    crop_x = crop_x.permute(1, 0, 2, 3, 4).contiguous()
    return crop_x, nh, nw


def patch_reverse(crop_x, x, step, ps):
    """Reverse patches into image."""
    b, c, h, w = x.size()
    output = torch.zeros_like(x)
    index = 0
    for i in range(0, h + step - ps, step):
        down = min(i + ps, h)
        top = down - ps
        for j in range(0, w + step - ps, step):
            right = min(j + ps, w)
            left = right - ps
            output[:, :, top:down, left:right] += crop_x[:, index]
            index += 1
    # 处理重叠区域的均值化
    for i in range(step, h + step - ps, step):
        top = i
        down = min(top + ps - step, h)
        if top < down:
            output[:, :, top:down, :] /= 2
    for j in range(step, w + step - ps, step):
        left = j
        right = min(left + ps - step, w)
        if left < right:
            output[:, :, :, left:right] /= 2
    return output


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, heads, qk_dim):
        super().__init__()
        self.heads = heads
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super().__init__()
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size,
                                padding=(kernel_size - 1) // 2, groups=hidden_features)
        self.gelu = nn.GELU()

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], -1, x_size[0], x_size[1])
        x = self.gelu(self.dwconv(x))
        return x.flatten(2).transpose(1, 2).contiguous()


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, kernel_size=5):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.dwconv = dwconv(hidden_features, kernel_size)
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        return self.fc2(x)


class LRSA(nn.Module):
    def __init__(self, dim, qk_dim, mlp_dim, ps, heads=1):
        super().__init__()
        self.ps = ps
        self.attn = PreNorm(dim, Attention(dim, heads, qk_dim))
        self.ffn = PreNorm(dim, ConvFFN(dim, mlp_dim))

    def forward(self, x):
        step = self.ps - 2
        crop_x, nh, nw = patch_divide(x, step, self.ps)
        b, n, c, ph, pw = crop_x.shape
        crop_x = rearrange(crop_x, 'b n c h w -> (b n) (h w) c')
        # Attention
        attned = self.attn(crop_x) + crop_x
        # FFN
        attned = self.ffn(attned, x_size=(ph, pw)) + attned
        # Rebuild
        attned = rearrange(attned, '(b n) (h w) c -> b n c h w', n=n, h=ph)
        return patch_reverse(attned, x, step, self.ps)


class ChannelShuffle(nn.Module):
    """通道混洗实现（需与原有实现保持一致）"""

    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, self.groups, c // self.groups, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.reshape(b, c, h, w)








class ASPP_WT_star_x_x1(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.9):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        # --------------------------------
        # 分支1: 1x1卷积 + 通道调整
        # --------------------------------
        self.branch1 = nn.Sequential(
            # 新增通道调整层 ↓↓↓
            nn.Conv2d(dim_in, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

        # --------------------------------
        # 分支2: 3x3卷积 + 通道调整
        # --------------------------------
        self.branch2 = nn.Sequential(
            WTConv2d(in_channels=dim_in, out_channels=dim_in, kernel_size=3),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            nn.ReLU(inplace=True),
            # 确保输出通道为dim_out ↓↓↓
            nn.Conv2d(dim_in, dim_out, 1, groups=2, bias=False),
            ChannelShuffle(groups=2),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # --------------------------------
        # 分支3: 5x5卷积 + 通道调整
        # --------------------------------
        self.branch3 = nn.Sequential(
            WTConv2d(in_channels=dim_in, out_channels=dim_in, kernel_size=5),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            nn.ReLU(inplace=True),
            # 确保输出通道为dim_out ↓↓↓
            nn.Conv2d(dim_in, dim_out, 1, groups=2, bias=False),
            ChannelShuffle(groups=2),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # --------------------------------
        # 分支4: 7x7卷积 + 通道调整
        # --------------------------------
        self.branch4 = nn.Sequential(
            WTConv2d(in_channels=dim_in, out_channels=dim_in, kernel_size=7),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            nn.ReLU(inplace=True),
            # 确保输出通道为dim_out ↓↓↓
            nn.Conv2d(dim_in, dim_out, 1, groups=2, bias=False),
            ChannelShuffle(groups=2),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # --------------------------------
        # 分支4: 9x9卷积 + 通道调整

        # --------------------------------
        self.branch5 = nn.Sequential(
            WTConv2d(in_channels=dim_in, out_channels=dim_in, kernel_size=9),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            nn.ReLU(inplace=True),
            # 确保输出通道为dim_out ↓↓↓
            nn.Conv2d(dim_in, dim_out, 1, groups=2, bias=False),
            ChannelShuffle(groups=2),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # --------------------------------
        # 融合层（输入通道应为dim_out*5=1280）
        # --------------------------------
        self.fusion = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.act = nn.ReLU6(inplace=False)
        self.adjust = nn.Conv2d(dim_in, dim_out, 1)

    def forward(self, x):
        b, c, h, w = x.size()

        # 分支1处理流程
        branch1_out = self.branch1(x)

        # 分支2处理流程
        branch2_out = self.branch2(x)

        # 分支3处理流程
        branch3_out = self.branch3(x)

        # 分支4处理流程
        branch4_out = self.branch4(x)

        # 分支5处理流程
        branch5_out = self.branch5(x)

        # 特征拼接与融合
        concat_feat = torch.cat([
            branch1_out,
            branch2_out,
            branch3_out,
            branch4_out,
            branch5_out,
        ], dim=1)

        x1 = self.adjust(x)

        return x1 + self.act(x1) * self.act(self.fusion(concat_feat))

class ASPP_WT_star_x1_x2(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.9):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        # --------------------------------
        # 分支1: 1x1卷积 + 通道调整
        # --------------------------------
        self.branch1 = nn.Sequential(
            # 新增通道调整层 ↓↓↓
            nn.Conv2d(dim_in, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

        # --------------------------------
        # 分支2: 3x3卷积 + 通道调整
        # --------------------------------
        self.branch2 = nn.Sequential(
            WTConv2d(in_channels=dim_in, out_channels=dim_in, kernel_size=3),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            nn.ReLU(inplace=True),
            # 确保输出通道为dim_out ↓↓↓
            nn.Conv2d(dim_in, dim_out, 1, groups=2, bias=False),
            ChannelShuffle(groups=2),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # --------------------------------
        # 分支3: 5x5卷积 + 通道调整
        # --------------------------------
        self.branch3 = nn.Sequential(
            WTConv2d(in_channels=dim_in, out_channels=dim_in, kernel_size=5),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            nn.ReLU(inplace=True),
            # 确保输出通道为dim_out ↓↓↓
            nn.Conv2d(dim_in, dim_out, 1, groups=2, bias=False),
            ChannelShuffle(groups=2),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # --------------------------------
        # 分支4: 7x7卷积 + 通道调整
        # --------------------------------
        self.branch4 = nn.Sequential(
            WTConv2d(in_channels=dim_in, out_channels=dim_in, kernel_size=7),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            nn.ReLU(inplace=True),
            # 确保输出通道为dim_out ↓↓↓
            nn.Conv2d(dim_in, dim_out, 1, groups=2, bias=False),
            ChannelShuffle(groups=2),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # --------------------------------
        # 分支4: 9x9卷积 + 通道调整

        # --------------------------------
        self.branch5 = nn.Sequential(
            WTConv2d(in_channels=dim_in, out_channels=dim_in, kernel_size=9),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            nn.ReLU(inplace=True),
            # 确保输出通道为dim_out ↓↓↓
            nn.Conv2d(dim_in, dim_out, 1, groups=2, bias=False),
            ChannelShuffle(groups=2),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # --------------------------------
        # 融合层（输入通道应为dim_out*5=1280）
        # --------------------------------
        self.fusion = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.act = nn.ReLU6(inplace=False)
        self.adjust1 = nn.Conv2d(dim_out, dim_out, 1)
        self.adjust2 = nn.Conv2d(dim_out, dim_out, 1)

        self.adjustx = nn.Conv2d(dim_in, dim_out, 1)

    def forward(self, x):
        b, c, h, w = x.size()

        # 分支1处理流程
        branch1_out = self.branch1(x)

        # 分支2处理流程
        branch2_out = self.branch2(x)

        # 分支3处理流程
        branch3_out = self.branch3(x)

        # 分支4处理流程
        branch4_out = self.branch4(x)

        # 分支5处理流程
        branch5_out = self.branch5(x)

        # 特征拼接与融合
        concat_feat = torch.cat([
            branch1_out,
            branch2_out,
            branch3_out,
            branch4_out,
            branch5_out,
        ], dim=1)
        fusion = self.fusion(concat_feat)

        x1 = self.adjust1(fusion)
        x2 = self.adjust2(fusion)

        x = self.adjustx(x)

        return x + self.act(x1) * self.act(x2)

class ASPP_WT(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.9):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        # --------------------------------
        # 分支1: 1x1卷积 + 通道调整
        # --------------------------------
        self.branch1 = nn.Sequential(
            # 新增通道调整层 ↓↓↓
            nn.Conv2d(dim_in, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

        # --------------------------------
        # 分支2: 3x3卷积 + 通道调整
        # --------------------------------
        self.branch2 = nn.Sequential(
            WTConv2d(in_channels=dim_in, out_channels=dim_in, kernel_size=3),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            nn.ReLU(inplace=True),
            # 确保输出通道为dim_out ↓↓↓
            nn.Conv2d(dim_in, dim_out, 1, groups=2, bias=False),
            ChannelShuffle(groups=2),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # --------------------------------
        # 分支3: 5x5卷积 + 通道调整
        # --------------------------------
        self.branch3 = nn.Sequential(
            WTConv2d(in_channels=dim_in, out_channels=dim_in, kernel_size=5),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            nn.ReLU(inplace=True),
            # 确保输出通道为dim_out ↓↓↓
            nn.Conv2d(dim_in, dim_out, 1, groups=2, bias=False),
            ChannelShuffle(groups=2),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # --------------------------------
        # 分支4: 7x7卷积 + 通道调整
        # --------------------------------
        self.branch4 = nn.Sequential(
            WTConv2d(in_channels=dim_in, out_channels=dim_in, kernel_size=7),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            nn.ReLU(inplace=True),
            # 确保输出通道为dim_out ↓↓↓
            nn.Conv2d(dim_in, dim_out, 1, groups=2, bias=False),
            ChannelShuffle(groups=2),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # --------------------------------
        # 分支4: 9x9卷积 + 通道调整

        # --------------------------------
        self.branch5 = nn.Sequential(
            WTConv2d(in_channels=dim_in, out_channels=dim_in, kernel_size=9),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            nn.ReLU(inplace=True),
            # 确保输出通道为dim_out ↓↓↓
            nn.Conv2d(dim_in, dim_out, 1, groups=2, bias=False),
            ChannelShuffle(groups=2),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # --------------------------------
        # 融合层（输入通道应为dim_out*5=1280）
        # --------------------------------
        self.fusion = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        b, c, h, w = x.size()

        # 分支1处理流程
        branch1_out = self.branch1(x)

        # 分支2处理流程
        branch2_out = self.branch2(x)

        # 分支3处理流程
        branch3_out = self.branch3(x)

        # 分支4处理流程
        branch4_out = self.branch4(x)

        # 分支5处理流程
        branch5_out = self.branch5(x)

        # 特征拼接与融合
        concat_feat = torch.cat([
            branch1_out,
            branch2_out,
            branch3_out,
            branch4_out,
            branch5_out,
        ], dim=1)
        fusion = self.fusion(concat_feat)

        return fusion

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1):
        super().__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)  # 确保通道数能被整除[2](@ref)
        new_channels = init_channels * (ratio - 1)

        # 主卷积路径
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride,
                      kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )

        # 幻影生成路径（关键修正点）
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1,
                      dw_size // 2, groups=init_channels, bias=False),  # groups必须等于输入通道数[7](@ref)
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]  # 通道维度保护


import math


class GhostNet(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=False):
        super().__init__()
        # 特征提取主干网络
        self.features = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 16, 3, 2, 1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            ),
            GhostModule(16, 24, stride=2),  # 输出尺寸减半
            GhostModule(24, 32),
            GhostModule(32, 64, stride=2),  # 低级特征截止点
            GhostModule(64, 96),
            GhostModule(96, 160, stride=2),
            GhostModule(160, 320)  # 高级特征输出
        )

        # 通道调整层（修正适配）
        self.adjust_x = nn.Sequential(
            nn.Conv2d(320, 96, 1, bias=False),  # 高级特征压缩
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.adjust_low = nn.Sequential(
            nn.Conv2d(32, 12, 1, bias=False),  # 低级特征压缩
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 低级特征提取（前3个模块）
        low_level_features = self.features[:3](x)  # 输出通道24
        # 高级特征提取（剩余模块）
        x = self.features[3:](low_level_features)  # 输出通道320
        # 通道调整
        x = self.adjust_x(x)
        low_level_features = self.adjust_low(low_level_features)
        return low_level_features, x


class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=8):
        super(DeepLab, self).__init__()
        if backbone == "xception":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            # ----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 96
            low_level_channels = 12
        elif backbone == "mobilenet":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            # ----------------------------------#

            in_channels = 96
            low_level_channels = 12
        elif backbone == "shufllenent":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            # ----------------------------------#
            self.backbone = ShuffleNetV2_Encoder()
            in_channels = 96
            low_level_channels = 12
        elif backbone == "startnet":
            self.backbone = StarNet(downsample_factor=8, pretrained=pretrained)
            in_channels = 96  # 对应stage4输出通道
            low_level_channels = 12  # 对应stage1输出通道
        elif backbone == "ghostnet":
            self.backbone = GhostNet(downsample_factor=8, pretrained=pretrained)
            in_channels = 96  # 对应stage4输出通道
            low_level_channels = 12  # 对应stage1输出通道

        elif backbone == "resnet":
            self.backbone = ResNet18(downsample_factor=8, pretrained=pretrained)
            in_channels = 96  # 对应stage4输出通道
            low_level_channels = 12  # 对应stage1输出通道
        elif backbone == "resnext":
            self.backbone = ResNeXtEncoder(downsample_factor=8, pretrained=pretrained)
            in_channels = 96  # 对应stage4输出通道
            low_level_channels = 12  # 对应stage1输出通道
        elif backbone == "efficientnetbo":
            self.backbone = EfficientNetB0Encoder()
            in_channels = 96  # 对应stage4输出通道
            low_level_channels = 12  # 对应stage1输出通道
        elif backbone == "res2net":
            self.backbone = Res2Net_Encoder()
            in_channels = 96  # 对应stage4输出通道
            low_level_channels = 12  # 对应stage1输出通道
        elif backbone == "swintransformer":
            self.backbone = SwinTransformer_Encoder()
            in_channels = 96  # 对应stage4输出通道
            low_level_channels = 12  # 对应stage1输出通道
        elif backbone == "convnext":
            self.backbone = ConvNeXt()
            in_channels = 96  # 对应stage4输出通道
            low_level_channels = 12  # 对应stage1输出通道
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        # -----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        # -----------------------------------------#
        self.aspp_lrsa = nn.Sequential(
            LRSA(in_channels, qk_dim=32, mlp_dim=64, ps=16),
        )

        # self.aspp = ASPP_group_point_conv_concat_before(dim_in=in_channels, dim_out=128, rate=16 // downsample_factor)
        # self.aspp = ASPP_WT_star_x_x1(dim_in=in_channels, dim_out=128, rate=16 // downsample_factor)
        # self.aspp = ASPP_WT(dim_in=in_channels, dim_out=128, rate=16 // downsample_factor)
        self.aspp = ASPP_WT_star_x1_x2(dim_in=in_channels, dim_out=128, rate=16 // downsample_factor)

        # ----------------------------------#
        #   浅层特征边
        # ----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 24, 1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),

        )
        self.cat_conv = nn.Sequential(
            # --------------------------------
            # 第一个融合块：深度可分离卷积 + 空洞卷积 + ECA
            # --------------------------------
            # 深度卷积（空洞率=2）
            WTConv2d(in_channels=152, out_channels=152, kernel_size=3),
            # Depthwise卷积[1,6](@ref)
            nn.BatchNorm2d(152),
            nn.ReLU(inplace=True),
            nn.Conv2d(152, 256, kernel_size=1, groups=2, bias=False),  # Pointwise分组卷积[6,8](@ref)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # --------------------------------
            # 第二个融合块：深度可分离卷积 + 空洞卷积
            # --------------------------------
            # 深度卷积（空洞率=4）
            WTConv2d(in_channels=256, out_channels=256, kernel_size=3),
            # 更高空洞率[9,11](@ref)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1, groups=2, bias=False),  # 更大分组数[5,7](@ref)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        self.cls_conv = nn.Conv2d(128, num_classes, 1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        # -----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        # -----------------------------------------#
        low_level_features, x = self.backbone(x)

        x = self.aspp_lrsa(x)
        x = self.aspp(x)

        low_level_features = self.shortcut_conv(low_level_features)

        # -----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        # -----------------------------------------#
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                          align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
