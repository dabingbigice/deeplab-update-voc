import torch
import torch.nn as nn
from timm.models.layers import DropPath


class StarNet(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True, version='s1'):
        super().__init__()
        # 根据下采样率调整stage结构 [1,3](@ref)
        self.stage_config = self._adjust_stages(downsample_factor)

        # 初始化基础网络结构
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        # 构建特征提取阶段
        self.stage1 = self._make_stage(32, 64, self.stage_config[0], stride=2)
        self.stage2 = self._make_stage(64, 128, self.stage_config[1], stride=2)
        self.stage3 = self._make_stage(128, 256, self.stage_config[2], stride=2)
        self.stage4 = self._make_stage(256, 512, self.stage_config[3], stride=1)  # 最后一层不下采样

        # 加载预训练权重 [1](@ref)
        if pretrained:
            self._load_pretrained(version)

        # 通道数配置（根据DeepLab需求）
        self.low_level_channels = 64  # 对应stage1输出通道
        self.aspp_channels = 512  # 对应stage4输出通道

    def _make_stage(self, in_ch, out_ch, num_blocks, stride):
        """ 构建网络阶段 """
        layers = []
        # 下采样层
        if stride == 2:
            layers.append(nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU6(inplace=True))
        else:
            layers.append(nn.Conv2d(in_ch, out_ch, 1, bias=False))

        # StarNet基础块 [3](@ref)
        for _ in range(num_blocks):
            layers.append(StarBlock(out_ch))
        return nn.Sequential(*layers)

    def _adjust_stages(self, downsample_factor):
        """ 根据下采样率调整阶段配置 [6](@ref) """
        if downsample_factor == 8:  # 原始配置：三次下采样
            return [2, 2, 8, 3]  # s1配置
        elif downsample_factor == 16:  # 原始论文配置
            return [3, 3, 12, 5]  # s4配置
        else:
            raise ValueError(f"不支持的downsample_factor: {downsample_factor}")

    def _load_pretrained(self, version):
        """ 加载预训练权重 [1](@ref) """
        model_urls = {
            's1': "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
            's2': "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar"
        }
        if version not in model_urls:
            raise ValueError(f"无效的版本: {version}")

        checkpoint = torch.hub.load_state_dict_from_url(model_urls[version])
        self.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"已加载 {version} 预训练权重")

    def forward(self, x):
        # Stem层
        x = self.stem(x)

        # 获取浅层特征 [3,6](@ref)
        low_level_feat = self.stage1(x)

        # 深层特征提取
        x = self.stage2(low_level_feat)
        x = self.stage3(x)
        x_aspp_before = self.stage4(x)

        return low_level_feat, x_aspp_before


class StarBlock(nn.Module):
    """ StarNet基础模块 [3,6](@ref) """

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU6(inplace=True)
        )
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1, bias=False),
            nn.BatchNorm2d(dim * 4),
            nn.ReLU6(inplace=True),
            nn.Conv2d(dim * 4, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = self.proj(x)
        return identity + self.act(x)

if __name__ == "__main__":
    # 测试特征输出
    model = StarNet(pretrained=False)
    x = torch.randn(2, 3, 512, 512)
    low, deep = model(x)
    print(model)
    print(f"浅层特征尺寸: {low.shape}")  # [2, 64, 128, 128]
    print(f"深层特征尺寸: {deep.shape}") # [2, 512, 64, 64]