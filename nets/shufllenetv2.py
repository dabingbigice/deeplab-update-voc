import math
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

BatchNorm2d = nn.BatchNorm2d


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class ShuffleUnit(nn.Module):
    def __init__(self, inp, oup, stride):
        super(ShuffleUnit, self).__init__()
        self.stride = stride
        branch_features = oup // 2

        assert self.stride != 1 or inp == branch_features * 2

        if self.stride > 1:
            # ------------------------------------------------#
            #   分支1：深度可分离卷积下采样
            # ------------------------------------------------#
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, 1, 1, 0, bias=False),
                BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        # ------------------------------------------------#
        #   分支2：常规卷积操作
        # ------------------------------------------------#
        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, 1, 1, 0, bias=False),
            BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_features, branch_features, 3, stride, 1,
                      groups=branch_features, bias=False),
            BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, 1, 1, 0, bias=False),
            BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)
        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, n_class=1000, model_size='1.0x'):
        super(ShuffleNetV2, self).__init__()
        self.stage_repeats = [4, 8, 4]

        # 不同规模模型配置[1](@ref)
        if model_size == '0.5x':
            self.stage_out_channels = [24, 48, 96, 192, 1024]
        elif model_size == '1.0x':
            self.stage_out_channels = [24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self.stage_out_channels = [24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            self.stage_out_channels = [24, 244, 488, 976, 2048]
        else:
            raise ValueError('Unsupported model size')

        # 输入层[6](@ref)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.stage_out_channels[0], 3, 2, 1, bias=False),
            BatchNorm2d(self.stage_out_channels[0]),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 构建三个阶段[2](@ref)
        self.stage2 = self._make_stage(self.stage_out_channels[0],
                                       self.stage_out_channels[1],
                                       self.stage_repeats[0])
        self.stage3 = self._make_stage(self.stage_out_channels[1],
                                       self.stage_out_channels[2],
                                       self.stage_repeats[1])
        self.stage4 = self._make_stage(self.stage_out_channels[2],
                                       self.stage_out_channels[3],
                                       self.stage_repeats[2])

        # 输出层[1](@ref)
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.stage_out_channels[3], self.stage_out_channels[4],
                      1, 1, 0, bias=False),
            BatchNorm2d(self.stage_out_channels[4]),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.stage_out_channels[4], n_class)
        )

        self._initialize_weights()

    def _make_stage(self, inp, oup, repeats):
        layers = []
        layers.append(ShuffleUnit(inp, oup, 2))  # 首层步长为2[6](@ref)
        for _ in range(repeats - 1):
            layers.append(ShuffleUnit(oup, oup, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def shufflenetv2(pretrained=False, model_size='1.0x'):
    model = ShuffleNetV2(model_size=model_size)
    if pretrained:
        # 预训练权重下载地址[1](@ref)
        model_urls = {
            '0.5x': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
            '1.0x': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
            '1.5x': None,
            '2.0x': 'https://download.pytorch.org/models/shufflenetv2_x2_0-8be3c8ee.pth'
        }
        if model_size in model_urls and model_urls[model_size]:
            state_dict = model_zoo.load_url(model_urls[model_size])
            model.load_state_dict(state_dict, strict=False)
    return model


if __name__ == "__main__":
    model = shufflenetv2(pretrained=True, model_size='1.0x')
    for i, layer in enumerate(model.children()):
        print(i, layer)