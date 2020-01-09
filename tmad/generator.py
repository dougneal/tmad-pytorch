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

        self.model = nn.Sequential()
        self.model.add_module('Input Block', self.__input_block())
        self.model.add_module('Intermediate Block 1', self.__intermediate_block(8, 4))
        self.model.add_module('Intermediate Block 2', self.__intermediate_block(4, 2))
        self.model.add_module('Intermediate Block 3', self.__intermediate_block(2, 1))
        self.model.add_module('Output Block', self.__output_block())

    def __input_block(self):
        return nn.Sequential(
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
        )

    def __intermediate_block(self, in_multiplier, out_multiplier):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels  = (self._fmapsize * in_multiplier),
                out_channels = (self._fmapsize * out_multiplier),
                kernel_size  = self._conv_kernel_size,
                stride       = self._conv_stride,
                padding      = self._conv_padding,
                bias         = self._conv_bias,
            ),
            nn.BatchNorm2d(
                num_features = (self._fmapsize * out_multiplier)
            ),
            nn.ReLU(
                inplace = True
            ),
        )

    def __output_block(self):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels  = self._fmapsize,
                out_channels = self._nchannels,
                kernel_size  = self._conv_kernel_size,
                stride       = self._conv_stride,
                padding      = self._conv_padding,
                bias         = self._conv_bias,
            ),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.model(input)
