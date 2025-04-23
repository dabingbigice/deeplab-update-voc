import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch.einops import rearrange

from nets.shufllenetv2 import ShuffleUnit
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2
import torch
import torch.nn as nn
from functools import partial
from nets.LRSAmodule import LRSA


class ShuffleNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(ShuffleNetV2, self).__init__()
        from nets.shufllenetv2 import ShuffleNetV2 as SNV2  # 引用提供的ShuffleNetV2实现

        model = SNV2(n_class=1000, model_size='1.0x')
        if pretrained:
            # 加载预训练权重（需根据实际路径调整）
            state_dict = torch.load('shufflenetv2_x1.pth')
            model.load_state_dict(state_dict, strict=False)
        self.channel_adjust = nn.Sequential(
            nn.Conv2d(1024, 320, 1),
            nn.BatchNorm2d(320),
            nn.ReLU()
        )
        # 构建特征序列
        self.features = nn.Sequential(
            *list(model.conv1.children()),  # 转换conv1的children
            model.maxpool,
            *model.stage2.children(),  # 直接展开stage2的children生成器
            *model.stage3.children(),
            *model.stage4.children(),
            *list(model.conv5.children())[:-1]  # 转换为列表后切片 ✅
        )

        # 确定下采样层索引（示例值，需根据实际结构调整）
        self.total_idx = len(self.features)
        self.down_idx = [3, 4, 7, 14]  # 示例索引，需根据实际模型结构调整

        # 调整下采样策略
        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname == 'Conv2d':
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
        elif isinstance(m, ShuffleUnit):
            # 处理ShuffleUnit内部卷积层
            for layer in m.branch1:
                if isinstance(layer, nn.Conv2d) and layer.stride == (2, 2):
                    layer.stride = (1, 1)
                    if layer.kernel_size == (3, 3):
                        layer.dilation = (dilate // 2, dilate // 2)
                        layer.padding = (dilate // 2, dilate // 2)
            for layer in m.branch2:
                if isinstance(layer, nn.Conv2d) and layer.stride == (2, 2):
                    layer.stride = (1, 1)
                    if layer.kernel_size == (3, 3):
                        layer.dilation = (dilate // 2, dilate // 2)
                        layer.padding = (dilate // 2, dilate // 2)

    def forward(self, x):
        # 获取浅层特征和深层特征
        low_level_features = self.features[:4](x)  # 前4层为浅层特征
        x = self.features[4:](low_level_features)  # 后续层为深层特征
        x = self.channel_adjust(x)
        return low_level_features, x


class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x


# -----------------------------------------#
#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
# #-----------------------------------------#
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


# # 深度可分离+空洞卷积
# class ASPP(nn.Module):
#     def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
#         super(ASPP, self).__init__()
#
#         # ------------------------------------------
#         #   分支1：1x1普通卷积保持不变
#         # ------------------------------------------
#         self.branch1 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#
#         # ------------------------------------------
#         #   分支2-4：深度可分离卷积+空洞卷积
#         #   分解为：深度卷积(depthwise) + 逐点卷积(pointwise)
#         # ------------------------------------------
#         def make_aspp_branch(in_channels, out_channels, dilation):
#             return nn.Sequential(
#                 # 深度卷积（分组数=输入通道数）
#                 nn.Conv2d(in_channels, in_channels, 3,
#                           padding=dilation, dilation=dilation,
#                           groups=in_channels, bias=True),  # 关键修改点
#                 nn.BatchNorm2d(in_channels, momentum=bn_mom),
#                 nn.ReLU(inplace=True),
#                 # 逐点卷积调整通道数
#                 nn.Conv2d(in_channels, out_channels, 1, bias=True),
#                 nn.BatchNorm2d(out_channels, momentum=bn_mom),
#                 nn.ReLU(inplace=True),
#             )
#
#         # 创建不同膨胀率的三个分支
#         self.branch2 = make_aspp_branch(dim_in, dim_out, 6 * rate)  # 膨胀率6
#         self.branch3 = make_aspp_branch(dim_in, dim_out, 12 * rate)  # 膨胀率12
#         self.branch4 = make_aspp_branch(dim_in, dim_out, 18 * rate)  # 膨胀率18
#
#         # ------------------------------------------
#         #   分支5：全局特征分支保持不变
#         # ------------------------------------------
#         self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, bias=True)
#         self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
#         self.branch5_relu = nn.ReLU(inplace=True)
#
#         # ------------------------------------------
#         #   最终融合卷积（保持原结构）
#         # ------------------------------------------
#         self.conv_cat = nn.Sequential(
#             nn.Conv2d(dim_out * 5, dim_out, 1, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         [b, c, row, col] = x.size()
#
#         # 各分支前向计算
#         conv1x1 = self.branch1(x)
#         conv3x3_1 = self.branch2(x)  # 深度可分离版本
#         conv3x3_2 = self.branch3(x)  # 深度可分离版本
#         conv3x3_3 = self.branch4(x)  # 深度可分离版本
#
#         # 全局特征分支
#         global_feature = torch.mean(x, dim=[2, 3], keepdim=True)
#         global_feature = self.branch5_conv(global_feature)
#         global_feature = self.branch5_bn(global_feature)
#         global_feature = self.branch5_relu(global_feature)
#         global_feature = F.interpolate(global_feature, (row, col), mode='bilinear', align_corners=True)
#
#         # 特征拼接与融合
#         feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
#         result = self.conv_cat(feature_cat)
#         return result


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.ops import DeformConv2d
#
# # --------------------------------
# # 可变形卷积模块（含Offset生成）
# # --------------------------------
# class DeformableConv(nn.Module):
#     def __init__(self, in_channels, out_channels, dilation=1):
#         super().__init__()
#         # Offset生成网络
#         self.offset_conv = nn.Sequential(
#             nn.Conv2d(in_channels, 2 * 3 * 3,  # 2*K*K (K=3)
#                       kernel_size=3, padding=dilation, dilation=dilation),
#             nn.BatchNorm2d(2 * 3 * 3),
#             nn.ReLU(inplace=True)
#         )
#         # 可变形卷积
#         self.deform_conv = DeformConv2d(
#             in_channels, out_channels,
#             kernel_size=3,
#             padding=dilation,
#             dilation=dilation
#         )
#
#     def forward(self, x):
#         offset = self.offset_conv(x)  # [B, 18, H, W]
#         return self.deform_conv(x, offset)


#
# # --------------------------------
# # 可变形深度可分离卷积模块（修复分支4）
# # --------------------------------
# class DeformableDepthwiseConv(nn.Module):
#     def __init__(self, in_channels, out_channels, dilation=1):
#         super().__init__()
#         # Offset生成网络（适配深度卷积）
#         self.offset_conv = nn.Sequential(
#             nn.Conv2d(in_channels, 2 * 3 * 3*in_channels,  # 2*K*K*C_in
#                       kernel_size=3, padding=dilation, dilation=dilation),
#             nn.BatchNorm2d(2 * 3 * 3*in_channels),
#             nn.ReLU(inplace=True)
#         )
#         # 可变形深度卷积（分组=输入通道）
#         self.deform_conv = DeformConv2d(
#             in_channels, in_channels,
#             kernel_size=3,
#             padding=dilation,
#             dilation=dilation,
#             groups=in_channels  # 深度卷积
#         )
#         # 逐点卷积
#         self.pointwise = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         offset = self.offset_conv(x)
#         x = self.deform_conv(x, offset)
#         return self.pointwise(x)
#
# # 修复后的ASPP模块可变形卷积
# # --------------------------------
# class ASPP_Enhanced(nn.Module):
#     def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
#         super().__init__()
#         # 分支1：普通1x1卷积
#         self.branch1 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 1, bias=False),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True)
#         )
#         # 分支2：可变形卷积（带Offset）
#         self.branch2 = nn.Sequential(
#             DeformableConv(dim_in, dim_out, dilation=6*rate),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True)
#         )
#         # 分支3：深度可分离+空洞
#         self.branch3 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_in, 3, padding=12*rate,
#                      dilation=12*rate, groups=dim_in, bias=False),
#             nn.BatchNorm2d(dim_in, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(dim_in, dim_out, 1, bias=False),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True)
#         )
#         # 分支4：可变形深度可分离
#         self.branch4 = DeformableDepthwiseConv(dim_in, dim_out, dilation=18*rate)
#         # 分支5：全局上下文
#         self.branch5 = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(dim_in, dim_out, 1, bias=False),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True)
#         )
#         # 特征融合
#         self.fusion = nn.Sequential(
#             nn.Conv2d(5*dim_out, dim_out, 1, bias=False),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5)
#         )
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         # 各分支前向
#         conv1x1 = self.branch1(x)
#         deform_out = self.branch2(x)
#         depthwise_out = self.branch3(x)
#         deform_depth_out = self.branch4(x)
#         # 全局特征
#         global_feat = self.branch5(x)
#         global_feat = F.interpolate(global_feat, (h, w), mode='bilinear', align_corners=True)
#         # 特征融合
#         concat = torch.cat([conv1x1, deform_out, depthwise_out, deform_depth_out, global_feat], dim=1)
# #         return self.fusion(concat)
# class ChannelShuffle(nn.Module):
#     def __init__(self, groups):
#         super().__init__()
#         self.groups = groups
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#         x = x.view(b, self.groups, c//self.groups, h, w)
#         x = x.transpose(1, 2).contiguous()
#         return x.view(b, -1, h, w)

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


class ECALayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super().__init__()
        # 确保k为奇数（防止Conv1d输出维度变化）
        k = int(abs((math.log(channel, 2) + b) / gamma))
        k = k if k % 2 else k + 1  # 强制k为奇数
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化并压缩维度
        y = x.mean((2, 3), keepdim=False)  # [B, C]
        y = y.unsqueeze(-1)  # [B, C, 1]
        y = y.transpose(1, 2)  # [B, 1, C]
        y = self.conv(y)  # [B, 1, C]
        y = self.sigmoid(y.transpose(1, 2))  # [B, C, 1]
        y = y.unsqueeze(-1)  # [B, C, 1, 1]
        return x * y.expand_as(x)


# 改进的aspp模块
class ASPP_group_point_conv(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP_group_point_conv, self).__init__()

        # --------------------------------
        # 分组逐点卷积 + 通道混洗
        # --------------------------------
        # --------------------------------
        # 分支1：小窗口密集交互 (PS=16)
        # --------------------------------
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, groups=4, padding=0, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),

        )

        # --------------------------------
        # 分支2：中等窗口区域关联 (PS=24)
        # --------------------------------
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 3, padding=6 * rate, dilation=6 * rate, groups=dim_in, bias=False),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, dim_out, 1, groups=4, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),

        )

        # --------------------------------
        # 分支3：级联多尺度LRSA (PS=32 → 16)
        # --------------------------------
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 3, padding=12 * rate, dilation=12 * rate, groups=dim_in, bias=False),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, dim_out, 1, groups=4, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),

        )

        # --------------------------------
        # 分支4：自适应窗口LRSA
        # --------------------------------
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 3, padding=18 * rate, dilation=18 * rate, groups=dim_in, bias=False),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, dim_out, 1, groups=4, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),

        )
        # --------------------------------
        # 分支5：全局特征增强
        # --------------------------------
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim_in, dim_out, 1, groups=4, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),

        )

        self.fusion = nn.Sequential(
            # 分组逐点卷积（Group=4）
            nn.Conv2d(dim_out * 5, dim_out, 1, groups=4, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
            LRSA(dim_out, qk_dim=32, mlp_dim=64, ps=32),
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # 各分支前向传播
        conv1x1 = self.branch1(x)  # [B,C,H,W]
        depthwise_out = self.branch2(x)  # [B,C,H,W]
        spatial_sep_out = self.branch3(x)  # [B,C,H,W]
        depth_spatial_out = self.branch4(x)  # [B,C,H,W]

        # 全局特征上采样
        global_feat = self.branch5(x)  # [B,C,1,1]
        global_feat = F.interpolate(global_feat, (h, w), mode='bilinear', align_corners=True)  # [B,C,H,W]

        # 特征拼接（确保所有分支输出尺寸为 [B,C,H,W]）
        concat_feat = torch.cat([
            conv1x1,
            depthwise_out,
            spatial_sep_out,
            depth_spatial_out,
            global_feat
        ], dim=1)  # 在通道维度拼接 → [B,5C,H,W]

        return self.fusion(concat_feat)  # [B,C,H,W]


class ASPP_Avg_pool(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super().__init__()
        # 多级池化层

        reduction = 4  # 通道压缩比例
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),  # 步长自动等于kernel_size
            nn.Conv2d(dim_in, dim_in // reduction, 1),  # 1x1卷积降维
            nn.BatchNorm2d(dim_in // reduction),
            nn.ReLU(inplace=True),

            nn.Conv2d(dim_in // reduction, dim_out, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4),  # 步长自动等于kernel_size
            nn.Conv2d(dim_in, dim_in // reduction, 1),  # 1x1卷积降维
            nn.BatchNorm2d(dim_in // reduction),
            nn.ReLU(inplace=True),

            # 调整通道
            nn.Conv2d(dim_in // reduction, dim_out, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=8),  # 步长自动等于kernel_size
            nn.Conv2d(dim_in, dim_in // reduction, 1),  # 1x1卷积降维
            nn.BatchNorm2d(dim_in // reduction),
            nn.ReLU(inplace=True),

            # 调整通道
            nn.Conv2d(dim_in // reduction, dim_out, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 3, padding=6 * rate, dilation=6 * rate, groups=dim_in, bias=False),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            nn.ReLU(inplace=True),

            # 调整通道
            nn.Conv2d(dim_in, dim_out, 1, groups=4, bias=False),
            ChannelShuffle(groups=4),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),

        )

        self.branch5 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 3, padding=12 * rate, dilation=12 * rate, groups=dim_in, bias=False),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, dim_out, 1, groups=4, bias=False),
            ChannelShuffle(groups=4),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),

        )
        self.pool = [self.branch1, self.branch2, self.branch3]

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
            LRSA(dim_out, qk_dim=32, mlp_dim=64, ps=32),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        # conv1x1 = self.branch1(x)
        # conv3x3_1 = self.branch2(x)
        # conv3x3_2 = self.branch3(x)
        conv3x3_1 = self.branch4(x)
        conv3x3_2 = self.branch5(x)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        # global_feature = torch.mean(x, 2, True)
        # global_feature = torch.mean(global_feature, 3, True)
        # global_feature = self.branch5_conv(global_feature)
        # global_feature = self.branch5_bn(global_feature)
        # global_feature = self.branch5_relu(global_feature)
        # global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        features = []
        for i, branch in enumerate(self.pool):
            # 执行池化和降维
            pooled = branch(x)
            # 上采样至原始尺寸
            upsampled = F.interpolate(pooled, size=(H, W), mode='bilinear', align_corners=False)
            features.append(upsampled)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([*features, conv3x3_1, conv3x3_2], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class HCA(nn.Module):
    def __init__(self, in_ch, levels=4):
        super().__init__()
        self.levels = levels
        reduction = 4  # 通道压缩比例

        # 多级池化层
        self.pool = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool2d(kernel_size=2 ** i),  # 步长自动等于kernel_size
                nn.Conv2d(in_ch, in_ch // reduction, 1),  # 1x1卷积降维
                nn.BatchNorm2d(in_ch // reduction),
                nn.ReLU(inplace=True)
            ) for i in range(levels)
        ])

        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv2d((in_ch // reduction) * levels, in_ch, 3, padding=1, groups=in_ch),  # 深度卷积
            nn.Conv2d(in_ch, in_ch, 1),  # 点卷积
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        features = []
        for i, branch in enumerate(self.pool):
            # 执行池化和降维
            pooled = branch(x)
            # 上采样至原始尺寸
            upsampled = F.interpolate(pooled, size=(H, W), mode='bilinear', align_corners=False)
            features.append(upsampled)

        # 拼接多级特征
        concated = torch.cat(features, dim=1)
        # 融合输出
        return self.fusion(concated)


class ASPP_group_point_conv_concat_before(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.9):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        # --------------------------------
        # 分支定义（保持原有结构不变）
        # --------------------------------
        # 分支1
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, groups=2, padding=0, bias=False),
            ChannelShuffle(groups=2),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # 分支2
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 3, padding=6 * rate, dilation=6 * rate, groups=dim_in, bias=False),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, dim_out, 1, groups=2, bias=False),
            ChannelShuffle(groups=2),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # 分支3
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 3, padding=12 * rate, dilation=12 * rate, groups=dim_in, bias=False),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, dim_out, 1, groups=2, bias=False),
            ChannelShuffle(groups=2),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # 分支4
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 3, padding=18 * rate, dilation=18 * rate, groups=dim_in, bias=False),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, dim_out, 1, groups=2, bias=False),
            ChannelShuffle(groups=2),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # 分支5
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim_in, dim_out, 1, groups=2, bias=False),
            ChannelShuffle(groups=2),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # --------------------------------
        # 为每个分支添加后处理层
        # --------------------------------


        # --------------------------------
        # 融合层（保持原有结构）
        # --------------------------------
        self.fusion = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
            LRSA(dim_out, qk_dim=32, mlp_dim=64, ps=16),
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
        global_feat = self.branch5(x)
        global_feat = F.interpolate(global_feat, (h, w), mode='bilinear', align_corners=True)

        # 特征拼接与融合
        concat_feat = torch.cat([
            branch1_out,
            branch2_out,
            branch3_out,
            branch4_out,
            global_feat
        ], dim=1)

        return self.fusion(concat_feat)


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
            in_channels = 2048
            low_level_channels = 256
        elif backbone == "mobilenet":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            # ----------------------------------#
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        elif backbone == "shufllenent":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            # ----------------------------------#
            self.backbone = ShuffleNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        # -----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        # -----------------------------------------#
        self.aspp_lrsa = nn.Sequential(
            LRSA(in_channels, qk_dim=32, mlp_dim=64, ps=16),
        )

        self.aspp = ASPP_group_point_conv_concat_before(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)

        self.aspp_last_concat_fusion = nn.Sequential(
            nn.Conv2d(in_channels + 256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(inplace=True)
        )
        # ----------------------------------#
        #   浅层特征边
        # ----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),

        )
        # 普通3*3卷积
        # self.cat_conv = nn.Sequential(
        #     nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #
        #     nn.Conv2d(256, 256, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Dropout(0.1),
        # )
        # self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)
        # 深度空间可分离卷积
        self.cat_conv = nn.Sequential(
            # --------------------------------
            # 第一个融合块：深度可分离卷积 + 空洞卷积 + ECA
            # --------------------------------
            # 深度卷积（空洞率=2）
            nn.Conv2d(304, 304, kernel_size=3, padding=2,
                      dilation=2, groups=304, bias=False),  # Depthwise卷积[1,6](@ref)
            nn.BatchNorm2d(304),
            nn.ReLU(inplace=True),
            nn.Conv2d(304, 256, kernel_size=1,groups=2, bias=False),  # Pointwise分组卷积[6,8](@ref)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # --------------------------------
            # 第二个融合块：深度可分离卷积 + 空洞卷积
            # --------------------------------
            # 深度卷积（空洞率=4）
            nn.Conv2d(256, 256, kernel_size=3, padding=4,
                      dilation=4, groups=256, bias=False),  # 更高空洞率[9,11](@ref)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1,groups=2, bias=False),  # 更大分组数[5,7](@ref)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        # -----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        # -----------------------------------------#
        low_level_features, x_aspp_before = self.backbone(x)

        x = self.aspp_lrsa(x_aspp_before)
        x = self.aspp(x)
        x = torch.cat([x_aspp_before, x], dim=1)
        x = self.aspp_last_concat_fusion(x)
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
