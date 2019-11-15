import torch.nn as nn

import logging


class Discriminator(nn.Module):
    _logger = logging.getLogger('discriminator')

    def __init__(self, feature_map_size: int, color_channels: int):
        super(Discriminator, self).__init__()

        self._fmapsize = feature_map_size
        self._nchannels = color_channels

        self._conv_kernel_size = (4, 4)
        self._conv_stride = (2, 2)
        self._conv_padding = (1, 1)
        self._conv_bias = False

        self._lrelu_negative_slope = 0.2
        self._lrelu_inplace = True

        self.main = nn.Sequential(
            # Input is (nchannels) x 64 x 64
            nn.Conv2d(
                in_channels  = self._nchannels,
                out_channels = self._fmapsize,
                kernel_size  = self._conv_kernel_size,
                stride       = self._conv_stride,
                padding      = self._conv_padding,
                bias         = self._conv_bias,
            ),
            nn.LeakyReLU(
                negative_slope = self._lrelu_negative_slope,
                inplace        = self._lrelu_inplace,
            ),

            # State Size: (self._fmapsize * 2) x 32 x 32
            nn.Conv2d(
                in_channels  = self._fmapsize,
                out_channels = self._fmapsize * 2,
                kernel_size  = self._conv_kernel_size,
                stride       = self._conv_stride,
                padding      = self._conv_padding,
                bias         = self._conv_bias,
            ),
            nn.BatchNorm2d(
                num_features = self._fmapsize * 2,
            ),
            nn.LeakyReLU(
                negative_slope = self._lrelu_negative_slope,
                inplace        = self._lrelu_inplace,
            ),

            # State Size: (self._fmapsize * 2) x 16 x 16
            nn.Conv2d(
                in_channels  = self._fmapsize * 2,
                out_channels = self._fmapsize * 4,
                kernel_size  = self._conv_kernel_size,
                stride       = self._conv_stride,
                padding      = self._conv_padding,
                bias         = self._conv_bias,
            ),
            nn.BatchNorm2d(
                num_features = self._fmapsize * 4,
            ),
            nn.LeakyReLU(
                negative_slope = self._lrelu_negative_slope,
                inplace        = self._lrelu_inplace,
            ),

            # State Size: (self._fmapsize * 4) x 8 x 8
            nn.Conv2d(
                in_channels  = self._fmapsize * 4,
                out_channels = self._fmapsize * 8,
                kernel_size  = self._conv_kernel_size,
                stride       = self._conv_stride,
                padding      = self._conv_padding,
                bias         = self._conv_bias,
            ),
            nn.BatchNorm2d(
                num_features = self._fmapsize * 8,
            ),
            nn.LeakyReLU(
                negative_slope = self._lrelu_negative_slope,
                inplace        = self._lrelu_inplace,
            ),

            # State Size: (self._fmapsize * 8) x 4 x 4
            nn.Conv2d(
                in_channels  = self._fmapsize * 8,
                out_channels = 1,
                kernel_size  = self._conv_kernel_size,
                stride       = (1, 1),
                padding      = 0,
                bias         = self._conv_bias,
            ),

            # Output is the probability that the input is a fake
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)
