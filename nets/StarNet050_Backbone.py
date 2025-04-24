import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from torch.nn import functional as F


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x

class StarNet050_Backbone(nn.Module):
    def __init__(self, output_stride=16):
        super().__init__()
        self.output_stride = output_stride
        base_dim = 16
        depths = [1, 1, 3, 1]
        self.in_channel = 32

        # Stem layer
        self.stem = nn.Sequential(
            ConvBN(3, base_dim, 3, stride=2, padding=1),
            nn.ReLU6(),
            ConvBN(base_dim, base_dim * 2, 3, stride=1, padding=1),
            nn.ReLU6()
        )

        # Build stages with dilation rates
        self.stages = nn.ModuleList()
        self.feature_channels = []
        current_stride = 4  # After stem (2x downsampling)

        for i_layer in range(4):
            embed_dim = base_dim * 2 ** (i_layer + 1)
            self.feature_channels.append(embed_dim)

            # Adjust dilation rates for output stride
            if self.output_stride == 8:
                dilations = [1, 1, 2, 4]
            else:  # output_stride=16 (default)
                dilations = [1, 1, 1, 1]

            blocks = []
            down_sampler = ConvBN(self.in_channel, embed_dim, 3,
                                  stride=2 if i_layer > 0 else 1,
                                  padding=1,
                                  dilation=dilations[i_layer])
            self.in_channel = embed_dim
            blocks.append(down_sampler)

            # Add blocks with adjusted dilation
            for _ in range(depths[i_layer]):
                blocks.append(
                    Block(embed_dim, mlp_ratio=3,
                          dilation=dilations[i_layer])
                )

            self.stages.append(nn.Sequential(*blocks))
            current_stride *= 2 if i_layer > 0 else 1

    def forward(self, x):
        # Feature hierarchy storage
        low_level_features = None
        features = []

        # Stem
        x = self.stem(x)

        # Stages
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i == 1:  # Stage2 for low-level features
                low_level_features = x
            if i == 3:  # Stage4 for high-level features
                high_level_features = x

        return {
            'low_level': low_level_features,
            'high_level': high_level_features,
            'features': features
        }


class DeepLabV3Plus(nn.Module):
    def __init__(self, backbone, num_classes=2):
        super().__init__()
        self.backbone = backbone
        in_channels = 256  # High-level feature channels

        # ASPP Module
        self.aspp = ASPP(in_channels, [6, 12, 18])

        # Decoder
        self.decoder = Decoder(
            low_level_channels=64,  # From stage2 features
            num_classes=num_classes
        )

        # Classifier
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        # Backbone features
        features = self.backbone(x)
        low_level = features['low_level']
        x = features['high_level']

        # ASPP
        x = self.aspp(x)

        # Decoder
        x = self.decoder(x, low_level)

        # Final classification
        x = self.classifier(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        return x


# ----------------- Helper Modules -----------------
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super().__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ))

        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, 256, rate))

        modules.append(ASPPPooling(in_channels, 256))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(1280, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super().forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_classes):
        super().__init__()
        self.low_level_reduce = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x, low_level):
        low_level = self.low_level_reduce(low_level)
        x = F.interpolate(x, size=low_level.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, low_level), dim=1)
        return self.conv(x)


# 使用示例
if __name__ == "__main__":
    # 初始化Backbone
    backbone = StarNet050_Backbone(output_stride=16)

    # 构建DeepLabV3+
    model = DeepLabV3Plus(backbone, num_classes=2)

    # 打印模型结构
    print(model)

    # 测试前向传播
    input_tensor = torch.randn(2, 3, 512, 512)
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
