
import torch.nn as nn
import math
import functools
from condconv import CondConv2d, route_func

__all__ = ['cond_mobilenetv2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, num_experts=None):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        self.expand_ratio = expand_ratio
        self.cond = num_experts is not None
        Conv2d = functools.partial(CondConv2d, num_experts=num_experts) if num_experts else nn.Conv2d

        if expand_ratio != 1:
            self.pw = Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
            self.bn_pw = nn.BatchNorm2d(hidden_dim)
        self.dw = Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
        self.bn_dw = nn.BatchNorm2d(hidden_dim)
        self.pw_linear = Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn_pw_linear = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU6(inplace=True)

        if num_experts:
            self.route = route_func(inp, num_experts)

    def forward(self, x):
        identity = x
        if self.cond:
            routing_weight = self.route(x)
            if self.expand_ratio != 1:
                x = self.relu(self.bn_pw(self.pw(x, routing_weight)))
            x = self.relu(self.bn_dw(self.dw(x, routing_weight)))
            x = self.bn_pw_linear(self.pw_linear(x, routing_weight))
        else:
            if self.expand_ratio != 1:
                x = self.relu(self.bn_pw(self.pw(x)))
            x = self.relu(self.bn_dw(self.dw(x)))
            x = self.bn_pw_linear(self.pw_linear(x))

        if self.identity:
            return x + identity
        else:
            return x


class CondMobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1., num_experts=8):
        super(CondMobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        self.num_experts = None
        for j, (t, c, n, s) in enumerate(self.cfgs):
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, self.num_experts))
                input_channel = output_channel
                if j == 4 and i == 0: # CondConv layers in the final 6 inverted residual blocks
                    self.num_experts = num_experts
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_route = route_func(output_channel, num_experts)
        self.classifier = CondConv2d(output_channel, num_classes, kernel_size=1, bias=False, num_experts=num_experts)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        routing_weight = self.classifier_route(x)
        x = self.classifier(x, routing_weight)
        x = x.squeeze_()
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def cond_mobilenetv2(**kwargs):
    """
    Constructs a CondConv-based MobileNet V2 model
    """
    return CondMobileNetV2(**kwargs)

