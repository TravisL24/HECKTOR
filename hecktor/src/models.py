import torch
from torch import nn
from torch.nn import functional as F

from layers import BasicConv3d, FastSmoothSeNormConv3d, RESseNormConv3d, UpConv
from layers import SimAM_Conv3d, ResSimAM_Conv3d, SimAM_UpConv


class BaselineUNet(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters):
        super(BaselineUNet, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters

        self.block_1_1_left = BasicConv3d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_left = BasicConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)  # 64, 1/2
        self.block_2_1_left = BasicConv3d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = BasicConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)  # 128, 1/4
        self.block_3_1_left = BasicConv3d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left = BasicConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)  # 256, 1/8
        self.block_4_1_left = BasicConv3d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_left = BasicConv3d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_3 = nn.ConvTranspose3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_3_1_right = BasicConv3d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right = BasicConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_2 = nn.ConvTranspose3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_2_1_right = BasicConv3d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right = BasicConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_1 = nn.ConvTranspose3d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_1_1_right = BasicConv3d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right = BasicConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.conv1x1 = nn.Conv3d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        ds0 = self.block_1_2_left(self.block_1_1_left(x))
        ds1 = self.block_2_2_left(self.block_2_1_left(self.pool_1(ds0)))
        ds2 = self.block_3_2_left(self.block_3_1_left(self.pool_2(ds1)))
        x = self.block_4_2_left(self.block_4_1_left(self.pool_3(ds2)))

        x = self.block_3_2_right(self.block_3_1_right(torch.cat([self.upconv_3(x), ds2], 1)))
        x = self.block_2_2_right(self.block_2_1_right(torch.cat([self.upconv_2(x), ds1], 1)))
        x = self.block_1_2_right(self.block_1_1_right(torch.cat([self.upconv_1(x), ds0], 1)))

        x = self.conv1x1(x)

        if self.n_cls == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)


class ResSimAM_Unet(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters, e_lambda=1e-4, return_logits=False):
        super(ResSimAM_Unet, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters
        self.e_lambda = e_lambda
        self.return_logits = return_logits

        self.leftConv_1_1 = ResSimAM_Conv3d(in_channels, n_filters, e_lambda, kernel_size=7, stride=1, padding=3)
        self.leftConv_1_2 = ResSimAM_Conv3d(n_filters, n_filters, e_lambda, kernel_size=3, stride=1, padding=1)

        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.leftConv_2_1 = ResSimAM_Conv3d(n_filters, 2 * n_filters, e_lambda, kernel_size=3, stride=1, padding=1)
        self.leftConv_2_2 = ResSimAM_Conv3d(2 * n_filters, 2 * n_filters, e_lambda, kernel_size=3, stride=1, padding=1)
        self.leftConv_2_3 = ResSimAM_Conv3d(2 * n_filters, 2 * n_filters, e_lambda, kernel_size=3, stride=1, padding=1)

        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.leftConv_3_1 = ResSimAM_Conv3d(2 * n_filters, 4 * n_filters, e_lambda, kernel_size=3, stride=1, padding=1)
        self.leftConv_3_2 = ResSimAM_Conv3d(4 * n_filters, 4 * n_filters, e_lambda, kernel_size=3, stride=1, padding=1)
        self.leftConv_3_3 = ResSimAM_Conv3d(4 * n_filters, 4 * n_filters, e_lambda, kernel_size=3, stride=1, padding=1)

        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.leftConv_4_1 = ResSimAM_Conv3d(4 * n_filters, 8 * n_filters, e_lambda, kernel_size=3, stride=1, padding=1)
        self.leftConv_4_2 = ResSimAM_Conv3d(8 * n_filters, 8 * n_filters, e_lambda, kernel_size=3, stride=1, padding=1)
        self.leftConv_4_3 = ResSimAM_Conv3d(8 * n_filters, 8 * n_filters, e_lambda, kernel_size=3, stride=1, padding=1)

        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.leftConv_5_1 = ResSimAM_Conv3d(8 * n_filters, 16 * n_filters, e_lambda, kernel_size=3, stride=1, padding=1)
        self.leftConv_5_2 = ResSimAM_Conv3d(16 * n_filters, 16 * n_filters, e_lambda, kernel_size=3, stride=1, padding=1)
        self.leftConv_5_3 = ResSimAM_Conv3d(16 * n_filters, 16 * n_filters, e_lambda, kernel_size=3, stride=1, padding=1)

        self.upconv_4 = nn.ConvTranspose3d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.rightConv_4_1 = ResSimAM_Conv3d((8 + 8) * n_filters, 8 * n_filters, e_lambda, kernel_size=3, stride=1, padding=1)
        self.rightConv_4_2 = ResSimAM_Conv3d(8 * n_filters, 8 * n_filters, e_lambda, kernel_size=3, stride=1, padding=1)

        self.upconv_3 = nn.ConvTranspose3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.rightConv_3_1 = ResSimAM_Conv3d((4 + 4) * n_filters, 4 * n_filters, e_lambda, kernel_size=3, stride=1, padding=1)
        self.rightConv_3_2 = ResSimAM_Conv3d(4 * n_filters, 4 * n_filters, e_lambda, kernel_size=3, stride=1, padding=1)

        self.upconv_2 = nn.ConvTranspose3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.rightConv_2_1 = ResSimAM_Conv3d((2 + 2) * n_filters, 2 * n_filters, e_lambda, kernel_size=3, stride=1, padding=1)
        self.rightConv_2_2 = ResSimAM_Conv3d(2 * n_filters, 2 * n_filters, e_lambda, kernel_size=3, stride=1, padding=1)

        self.upconv_1 = nn.ConvTranspose3d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.rightConv_1_1 = ResSimAM_Conv3d(2 * n_filters, n_filters, e_lambda, kernel_size=3, stride=1, padding=1)
        self.rightConv_1_2 = ResSimAM_Conv3d(n_filters, n_filters, e_lambda, kernel_size=3, stride=1, padding=1)

        self.outConv1x1 = nn.Conv3d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        op0 = self.leftConv_1_2(self.leftConv_1_1(x))
        op1 = self.leftConv_2_3(self.leftConv_2_2(self.leftConv_2_1(self.pool_1(op0))))
        op2 = self.leftConv_3_3(self.leftConv_3_2(self.leftConv_3_1(self.pool_2(op1))))
        op3 = self.leftConv_4_3(self.leftConv_4_2(self.leftConv_4_1(self.pool_3(op2))))
        x = self.leftConv_5_3(self.leftConv_5_2(self.leftConv_5_1(self.pool_4(op3))))

        x = self.rightConv_4_2(self.rightConv_4_1(torch.cat([self.upconv_4(x), op3], 1)))
        x = self.rightConv_3_2(self.rightConv_3_1(torch.cat([self.upconv_3(x), op2], 1)))
        x = self.rightConv_2_2(self.rightConv_2_1(torch.cat([self.upconv_2(x), op1], 1)))
        x = self.rightConv_1_2(self.rightConv_1_1(torch.cat([self.upconv_1(x), op0], 1)))

        x = self.outConv1x1(x)

        if self.return_logits:
            return x
        else:
            if self.n_cls == 1:
                return torch.sigmoid(x)
            else:
                return F.softmax(x, dim=1)
