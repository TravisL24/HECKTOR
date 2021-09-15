import torch
from torch import nn
from torch.nn import functional as F


# conv->norm->relu
class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)
        return x


# SimAM
class SimAM(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "SimAM"

    def forward(self, x):
        b, c, d, h, w = x.size()

        # 'n' for size of one channel obj
        n = d * h * w - 1

        # 'x_minus_mu_square' for square of (t - u)
        x_minus_mu_square = (x - x.mean(dim=[2, 3, 4], keepdim=True)).pow(2)

        # 'y' for E_inv, show the importance of X
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3, 4], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activation(y)


class SimAM_Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, e_lambda, **kwargs):
        super(SimAM_Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=True, **kwargs)
        self.norm = SimAM()

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.norm(x)
        return x


class ResSimAM_Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, e_lambda, **kwargs):
        super(ResSimAM_Conv3d, self).__init__()
        self.conv1 = SimAM_Conv3d(in_channels, out_channels, e_lambda, **kwargs)

        if in_channels != out_channels:
            self.res_conv = SimAM_Conv3d(in_channels, out_channels, e_lambda, kernel_size=1, stride=1, 
                                         padding=0)
        else:
            self.res_conv = None

    def forward(self, x):
        residual = self.res_conv(x) if self.res_conv else x
        x = self.conv1(x)
        x += residual
        return x


