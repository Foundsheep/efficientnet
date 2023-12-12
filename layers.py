import torch
from torch import nn
from torch.nn import functional as F


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * F.sigmoid(x)


class SEBlock(nn.Module):
    def __init__(self, in_planes, se_ratio):
        super().__init__()
        assert 0 < se_ratio <= 1, "se_ratio should be within the range of 0 and 1"
        reduced_dim = max(1, int(in_planes * se_ratio))
        self.layer_1_avp = nn.AdaptiveAvgPool2d(1)
        self.layer_2_squeeze = nn.Conv2d(in_channels=in_planes, out_channels=reduced_dim, kernel_size=1)
        self.layer_3_swish = Swish()
        self.layer_4_expand = nn.Conv2d(in_channels=reduced_dim, out_channels=in_planes, kernel_size=1)
        self.layer_5_sigmoid = nn.Sigmoid()

    def forward(self, x):
        inputs = x
        x = self.layer_1_avp(x)
        x = self.layer_2_squeeze(x)
        x = self.layer_3_swish(x)
        x = self.layer_4_expand(x)
        x = self.layer_5_sigmoid(x)
        x = torch.multiply(x, inputs)  # scale
        return x


class BottleneckResidualBlock(nn.Module):

    def __init__(self, first_channel, last_channel, factor, stride):
        super().__init__()
        self.stride = stride
        self.conv_1 = nn.Conv2d(in_channels=first_channel, out_channels=int(first_channel*factor), kernel_size=1, stride=1)

        # channel to be used in the block
        c = self.conv_1.out_channels
        self.bn_1 = nn.BatchNorm2d(c)

        self.conv_2_dw = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=stride, groups=c, padding=1)
        self.bn_2 = nn.BatchNorm2d(c)

        self.conv_2_pw = nn.Conv2d(in_channels=c, out_channels=last_channel, kernel_size=1)
        self.bn_3 = nn.BatchNorm2d(last_channel)
        self.last_channel = last_channel

    def forward(self, inputs):
        identity = nn.Identity()(inputs)
        x = F.relu6(self.bn_1(self.conv_1(inputs)))
        x = F.relu6(self.bn_2(self.conv_2_dw(x)))
        x = self.bn_3(self.conv_2_pw(x))

        if self.stride == 1 and identity.size() == x.size():
            x += identity
        return x


class MBConvBlock_Naive(nn.Module):
    def __init__(self, first_channel, last_channel, factor, stride, ratio):
        super(MBConvBlock_Naive, self).__init__()
        self.bottleneck = BottleneckResidualBlock(first_channel, last_channel, factor, stride)
        self.se_block = SEBlock(last_channel, ratio)

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.se_block(x)
        return x


class MBConvBlock(nn.Module):
    def __init__(self, first_channel, last_channel, stride, se_ratio=0.25, is_six=True):
        super().__init__()

        # MBConv1 vs MBConv6 difference
        self.is_six =is_six
        c = int(first_channel * 6) if is_six else first_channel

        self.layer_1_conv = nn.Conv2d(in_channels=first_channel, out_channels=c, kernel_size=1, stride=1)
        self.layer_2_bn = nn.BatchNorm2d(self.layer_1_conv.out_channels)
        self.layer_3_swish = Swish()

        self.layer_4_dw_conv = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=stride, groups=c, padding=1)
        self.layer_5_bn = nn.BatchNorm2d(self.layer_4_dw_conv.out_channels)
        self.layer_6_swish = Swish()
        self.layer_7_se = SEBlock(in_planes=self.layer_4_dw_conv.out_channels, se_ratio=se_ratio)
        self.layer_8_pw = nn.Conv2d(in_channels=self.layer_4_dw_conv.out_channels, out_channels=last_channel, kernel_size=1)
        self.layer_9_bn = nn.BatchNorm2d(self.layer_8_pw.out_channels)

    def forward(self, x):
        if self.is_six:
            x = self.layer_1_conv(x)
            x = self.layer_2_bn(x)
            x = self.layer_3_swish(x)

        x = self.layer_4_dw_conv(x)
        x = self.layer_5_bn(x)
        x = self.layer_6_swish(x)
        x = self.layer_7_se(x)
        x = self.layer_8_pw(x)
        x = self.layer_9_bn(x)
        return x


class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        pass

    def forward(self, x):
        return x


def test():
    bottleneck_1 = BottleneckResidualBlock(first_channel=16, last_channel=32, factor=4, stride=1)
    x = torch.randn((100, 16, 224, 224))
    output = bottleneck_1(x)
    print(output.size())

    bottleneck_2 = BottleneckResidualBlock(first_channel=32, last_channel=32, factor=4, stride=1)
    output2 = bottleneck_2(output)
    print(output2.size())


def test_2():
    mb_conv_block = MBConvBlock_Naive(first_channel=16, last_channel=64, factor=4, stride=1, ratio=0.25)
    x = torch.randn((100, 16, 224, 224))
    output = mb_conv_block(x)
    print(output.size())


def test_3():
    mb_conv_block = MBConvBlock(first_channel=16, last_channel=64, stride=2, se_ratio=0.25, is_six=True)
    x = torch.randn((100, 16, 224, 224))
    output = mb_conv_block(x)
    print(output.size())



if __name__ == "__main__":
    test_3()
