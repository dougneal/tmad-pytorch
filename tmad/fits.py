import os
import os.path
import logging

import numpy

from torch.utils.data import Dataset
from astropy.io import fits
from astropy.visualization import ZScaleInterval


class ZMaxInterval(ZScaleInterval):
    def __init__(self, *args, **kwargs):
        super(ZMaxInterval, self).__init__(*args, **kwargs)

    def get_limits(self, values):
        zscale_min, zscale_max = super(ZMaxInterval, self).get_limits(values)
        return (zscale_min, values.max())


class FitsTransformException(Exception):
    def __init__(self, message):
        super().__init__(message)


class TMAD_ACS_WFC_RAW_FITS_Transform:
    def __init__(
        self,
        interval_function: callable,
        stretch_function: callable,
        scale: tuple
    ):
        self.interval_function = interval_function
        self.stretch_function = stretch_function

        if len(scale) != 2 or scale[0] >= scale[1]:
            raise ValueError("Scale must be a tuple of (low, high)")

        self.scale = scale

        # Most of the ACS/WFC images are 4144x2068, with a few exceptions,
        # which we can just skip over. The shape tuple is (height, width)
        self.expected_shape = (2068, 4144)

        # We crop the above to 4096x2048 which tiles neatly into
        # 8 tiles of 1024x1024
        self.cropped_shape = (2048, 4096)
        self.tile_shape = (1024, 1024)
        self.tile_count = (
            numpy.prod(self.cropped_shape) / numpy.prod(self.tile_shape)
        )
        assert (self.tile_count == 8.0)
        self.tile_count = int(self.tile_count)

    """
    This signals to the Dataset class how many output images are created from
    a single input image
    """
    @property
    def multiplier(self):
        return self.tile_count

    def __call__(self, fits_data, tile_num):
        # Check dimensions
        if fits_data.shape != self.expected_shape:
            raise FitsTransformException(
                'Expected image shape {0}, got {1}'.format(
                    self.expected_shape.__str__(),
                    fits_data.shape.__str__(),
                )
            )

        # Crop to 4096x2048
        cropped = fits_data[12:2060, 24:4120]

        # Tile into 8 sub-images of 1024x1024
        sub_images = self._tile_slice(cropped)

        # Apply interval and stretch functions
        stretched = self.stretch_function(
            self.interval_function(
                sub_images[tile_num],
            )
        )

        # Rescale to target scale
        return self._rescale(stretched)

    """
    Slice the image up into tiles like so

    |---|---|---|---|
    | 0 | 1 | 2 | 3 |
    |---|---|---|---|
    | 4 | 5 | 6 | 7 |
    |---|---|---|---|

    """
    def _tile_slice(self, image):
        tiles = list()

        columns = self.cropped_shape[1] // self.tile_shape[1]
        rows = self.cropped_shape[0] // self.tile_shape[0]
        assert (columns * rows) == self.tile_count

        for tile_n in range(self.tile_count):
            column = tile_n % columns
            row = tile_n // columns

            left = column * self.tile_shape[1]
            right = ((column + 1) * self.tile_shape[1]) - 1
            top = row * self.tile_shape[0]
            bottom = ((row + 1) * self.tile_shape[0]) - 1

            tiles.append(image[top:bottom, left:right])

        return tiles

    def _rescale(self, image):
        return (image * self._scale_interval) - self.scale[0]

    @property
    def _scale_interval(self):
        return (self.scale[1] - self.scale[0])


class HSTImageDataset(Dataset):
    def __init__(self, root_dir: str, transform: callable):
        logger = logging.getLogger()
        self.root_dir = root_dir
        self.transform = transform

        # Scan for FITS files
        logger.info(f'Scanning {root_dir} for FITS files')
        self.fits_files = list(filter(
            lambda f: f[-5:].lower() == '.fits',
            os.listdir(self.root_dir),
        ))
        logger.info(f'Found {len(self.fits_files)} in {root_dir}')
        logger.info('Transform {0} has a multiplier of {1}'.format(
            self.transform.__class__.__name__, self.transform.multiplier
        ))
        logger.info(f'Total dataset size is {self.__len__()} images')

    def __len__(self):
        return len(self.fits_files) * self.transform.multiplier

    def __getitem__(self, index):
        file_index = index // self.transform.multiplier
        subimage_index = index % self.transform.multiplier

        filename = os.path.join(
            self.root_dir,
            self.fits_files[file_index],
        )
        fits_data = fits.getdata(filename)

        return self.transform(fits_data, subimage_index)
