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

        self.model = nn.Sequential()
        self.model.add_module('Input Block', self.__input_block())
        self.model.add_module('Intermediate Block 1', self.__intermediate_block(1, 2))
        self.model.add_module('Intermediate Block 2', self.__intermediate_block(2, 4))
        self.model.add_module('Intermediate Block 3', self.__intermediate_block(4, 8))
        self.model.add_module('Output Block', self.__output_block(8))

    def __input_block(self):
        return nn.Sequential(
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
        )

    def __intermediate_block(self, in_multiplier, out_multiplier):
        return nn.Sequential(
            nn.Conv2d(
                in_channels  = self._fmapsize * in_multiplier,
                out_channels = self._fmapsize * out_multiplier,
                kernel_size  = self._conv_kernel_size,
                stride       = self._conv_stride,
                padding      = self._conv_padding,
                bias         = self._conv_bias,
            ),
            nn.BatchNorm2d(
                num_features = self._fmapsize * out_multiplier,
            ),
            nn.LeakyReLU(
                negative_slope = self._lrelu_negative_slope,
                inplace        = self._lrelu_inplace,
            ),
        )

    def __output_block(self, in_multiplier):
        return nn.Sequential(
            # State Size: (self._fmapsize * 8) x 4 x 4
            nn.Conv2d(
                in_channels  = self._fmapsize * in_multiplier,
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
        return self.model(input)
