import torch.nn as nn

# 假设 StarNet 定义在 models.py 中（根据用户提供的代码）
from startnet_module import StarNet

# 定义 StarNetEncoder 类
class StarNetEncoder(nn.Module):
    def __init__(self, out_channels=(16, 32, 64, 128), depth=4, **kwargs):
        super().__init__()
        self.starnet = StarNet(base_dim=16, depths=[1, 1, 3, 1], mlp_ratio=3, **kwargs)
        self.out_channels = out_channels
        self.depth = depth

    def get_features(self, x):
        features = []
        x = self.starnet.stem(x)  # 前向传播经过 stem 层，输出通道数为 32
        for stage in self.starnet.stages:
            x = stage(x)
            features.append(x)  # 收集每个阶段的特征图
        return tuple(features)

    def forward(self, x):
        return self.get_features(x)

# 重写 get_encoder 函数以支持 starnet_s050
def get_encoder(name: str = "starnet_s050", **kwargs):
    if name == "starnet_s050":
        return StarNetEncoder(**kwargs)




# 示例用法（可选）
# model = model.to(device)
# output = model(input_tensor)