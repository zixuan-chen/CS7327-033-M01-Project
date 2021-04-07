import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.init as init
import math


class Quadratic(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Quadratic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = Parameter(torch.Tensor(out_features, in_features))
        self.weight2 = Parameter(torch.Tensor(out_features, in_features))
        self.bias = None
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.weight1, mean=0.0, std=1.0)
        init.normal_(self.weight2, mean=0.0, std=1.0)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight1)
            bound = 1 / math.sqrt(fan_in)
            init.normal_(self.bias)

    def forward(self, input):
        output = torch.matmul(input * input, self.weight1.t()) + torch.matmul(input, self.weight2.t())
        if self.bias is not None:
            output += self.bias
        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class MinMaxNet2x2(nn.Module):

    def __init__(self):
        super(MinMaxNet2x2, self).__init__()
        self.net = [nn.Sequential(Quadratic(2, 512), nn.Sigmoid(), Quadratic(512, 1)),
                    nn.Sequential(Quadratic(2, 512), nn.Sigmoid(), Quadratic(512, 1)),
                    nn.Sequential(Quadratic(2, 512), nn.Sigmoid(), Quadratic(512, 1)),
                    nn.Sequential(Quadratic(2, 512), nn.Sigmoid(), Quadratic(512, 1))]

    def forward(self, x):
        # ================== min-max net 1 ============================
        # min1 = torch.min(self.net[0](x), self.net[2](x))
        # min2 = torch.min(self.net[1](x), self.net[3](x))
        # ================== min-max net 2 ============================
        min1 = torch.min(self.net[0](x), self.net[1](x))
        min2 = torch.min(self.net[2](x), self.net[3](x))
        max = torch.max(min1, min2)
        return max
