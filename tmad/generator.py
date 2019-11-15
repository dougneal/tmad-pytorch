import torch.nn as nn

import logging


class Generator(nn.Module):
    _logger = logging.getLogger('generator')

    def __init__(self, feature_map_size: int, input_size: int, color_channels: int):
        super(Generator, self).__init__()

        self._fmapsize = feature_map_size
        self._inputsize = input_size
        self._nchannels = color_channels

        self._conv_kernel_size = (4, 4)
        self._conv_stride = (2, 2)
        self._conv_padding = (1, 1)
        self._conv_bias = False


        self.main = nn.Sequential(
            # Input is Z (latent space vector), going into a convolution
            nn.ConvTranspose2d(
                in_channels  = self._inputsize,
                out_channels = (self._fmapsize * 8),
                kernel_size  = self._conv_kernel_size,
                stride       = (1, 1),
                bias         = self._conv_bias,
            ),
            nn.BatchNorm2d(
                num_features = (self._fmapsize * 8)
            ),
            nn.ReLU(
                inplace = True
            ),

            # State Size: (self._fmapsize * 8) x 4 x 4
            nn.ConvTranspose2d(
                in_channels  = (self._fmapsize * 8),
                out_channels = (self._fmapsize * 4),
                kernel_size  = self._conv_kernel_size,
                stride       = self._conv_stride,
                padding      = self._conv_padding,
                bias         = self._conv_bias,
            ),
            nn.BatchNorm2d(
                num_features = (self._fmapsize * 4)
            ),
            nn.ReLU(
                inplace = True
            ),

            # State Size: (self._fmapsize * 4) x 8 x 8
            nn.ConvTranspose2d(
                in_channels  = (self._fmapsize * 4),
                out_channels = (self._fmapsize * 2),
                kernel_size  = self._conv_kernel_size,
                stride       = self._conv_stride,
                padding      = self._conv_padding,
                bias         = self._conv_bias,
            ),
            nn.BatchNorm2d(
                num_features = (self._fmapsize * 2)
            ),
            nn.ReLU(
                inplace = True
            ),

            # State Size: (self._fmapsize * 2) x 16 x 16
            nn.ConvTranspose2d(
                in_channels  = (self._fmapsize * 2),
                out_channels = self._fmapsize,
                kernel_size  = self._conv_kernel_size,
                stride       = self._conv_stride,
                padding      = self._conv_padding,
                bias         = self._conv_bias,
            ),
            nn.BatchNorm2d(
                num_features = self._fmapsize
            ),
            nn.ReLU(
                inplace = True
            ),

            # State Size: self._fmapsize x 32 x 32
            nn.ConvTranspose2d(
                in_channels  = self._fmapsize,
                out_channels = self._nchannels,
                kernel_size  = self._conv_kernel_size,
                stride       = self._conv_stride,
                padding      = self._conv_padding,
                bias         = self._conv_bias,
            ),
            nn.Tanh(),
            
            # State Size: (self._nchannels) x 64 x 64
            # Output is the fake image
        )

    def forward(self, input):
        return self.main(input)
