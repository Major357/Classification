import torch
import torch.nn as nn
import torch.nn.functional as F


# 卷积+bn+relu模块
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channals, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channals, **kwargs)
        self.bn = nn.BatchNorm2d(out_channals)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

# Inception模块
class Inception(nn.Module):
    def __init__(self, in_planes,
                 n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = BasicConv2d(in_planes, n1x1, kernel_size=1)

        # 1x1 conv -> 3x3 conv branch
        self.b2_1x1_a = BasicConv2d(in_planes, n3x3red,
                                    kernel_size=1)
        self.b2_3x3_b = BasicConv2d(n3x3red, n3x3,
                                    kernel_size=3, padding=1)

        # 1x1 conv -> 3x3 conv -> 3x3 conv branch
        self.b3_1x1_a = BasicConv2d(in_planes, n5x5red,
                                    kernel_size=1)
        self.b3_3x3_b = BasicConv2d(n5x5red, n5x5,
                                    kernel_size=3, padding=1)
        self.b3_3x3_c = BasicConv2d(n5x5, n5x5,
                                    kernel_size=3, padding=1)

        # 3x3 pool -> 1x1 conv branch
        self.b4_pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.b4_1x1 = BasicConv2d(in_planes, pool_planes,
                                  kernel_size=1)

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2_3x3_b(self.b2_1x1_a(x))
        y3 = self.b3_3x3_c(self.b3_3x3_b(self.b3_1x1_a(x)))
        y4 = self.b4_1x1(self.b4_pool(x))
        # y的维度为[batch_size, out_channels, C_out,L_out]
        # 合并不同卷积下的特征图
        return torch.cat([y1, y2, y3, y4], 1)

class GoogLeNet(nn.Module):
    def __init__(self,num_classes,num_linear=2458624):
        super(GoogLeNet, self).__init__()
        self.pre_layers = BasicConv2d(3, 192,
                                      kernel_size=3, padding=1)

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(num_linear, num_classes)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


if __name__ == "__main__":
    # 随机生成输入数据
    rgb = torch.randn(1, 3, 224, 224)
    # 定义网络
    # num_linear的设置是为了，随着输入图片数据大小的改变，使线性层的神经元数量可以匹配成功
    # 默认输入图片数据大小为224*224
    net = GoogLeNet(num_classes=8, num_linear=2458624)
    # 前向传播
    out = net(rgb)
    print('-----' * 5)
    # 打印输出大小
    print(out.shape)
    print('-----' * 5)
