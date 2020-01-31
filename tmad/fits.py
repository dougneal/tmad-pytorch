import collections
import hashlib
import os
import os.path
import logging
import tempfile

import numpy
import boto3

from torch.utils.data import Dataset
from astropy.io import fits
from astropy.visualization import ZScaleInterval


class ZMaxInterval(ZScaleInterval):
    """
    Astropy doesn't have a ZMaxInterval class, but we can derive it from the
    ZScaleInterval class pretty trivially. When visualising FITS images with
    the 'SAOImage ds9' tool, ZMax was found to highlight the detail in the
    images whilst keeping the background dark.
    """
    def __init__(self, *args, **kwargs):
        super(ZMaxInterval, self).__init__(*args, **kwargs)

    def get_limits(self, values):
        zscale_min, zscale_max = super(ZMaxInterval, self).get_limits(values)
        return (zscale_min, values.max())


class FitsTransformException(Exception):
    def __init__(self, message):
        super().__init__(message)


class TMAD_ACS_WFC_RAW_FITS_Transform:
    """
    This transform class is so named because it deals specifically with raw
    FITS images from the ACS/WFC instrument on the HST.

    This means that we know the image size we're expecting, and can reject any
    that don't match, as we have found a few anomalies.

    The images come in at 4144x2068. We crop that down to 4096x2048, partly
    because there are black bands around the edge of the images, and partly
    because 4096x2048 divides exactly into 8 tiles of 1024x1024, which is our
    working base image size for training the DCGAN.

    You'll find a few hard-coded values here relevant to the above.

    For potential future scenarios, say, working with images from a different
    instrument, or a processing level above RAW, where the input image size
    is different and a different tile-cutting strategy applies, I would suggest
    copying this class into a new one and tweaking the values.

    There may then be opportunities to refactor common bits out if so inclined.

    Parameters
    ----------
    interval_function: callable
        An instance of ZMaxInterval, or possibly some other interval function,
        such as one of many available in astropy.visualization.
    stretch_function: callable
        An instance of astropy.visualization.LogStretch(). Again, alternative
        stretch functions are available, but this is what we found to work
        "best" so far.
    scale: tuple
        A 2-tuple containing the target pixel values for (black, white).
        The output image will be rescaled accordingly. See comments further
        down.
    """
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
    a single input image.
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

        (tile_height, tile_width) = self.tile_shape

        for tile_n in range(self.tile_count):
            column = tile_n % columns
            row = tile_n // columns

            left = tile_width * column
            right = tile_width * (column + 1)
            top = tile_height * row
            bottom = tile_height * (row + 1)

            tiles.append(image[top:bottom, left:right])

        return tiles

    """
    "Scale" in this context refers to the interval (difference) between the
    pixel value representing the darkest shade (i.e. black) and the pixel value
    representing the lightest shade (i.e. white).

    For displaying on a monitor or exporting to familiar image formats like
    PNG, this range is from 0 to 255.

    FITS images as they come in have a much wider range than this.

    For feeding into the DCGAN for training, the range will be different again
    (TODO: specify here what that range is - currently unsure)
    """
    def _rescale(self, image):
        return (image * self._scale_interval) - self.scale[0]

    @property
    def _scale_interval(self):
        return (self.scale[1] - self.scale[0])


class CachingS3Downloader():
    def __init__(self, cache_size=4):
        self._logger = logging.getLogger('CachingS3Downloader')
        self._directory = tempfile.TemporaryDirectory()
        self._cache_index = collections.OrderedDict()
        self._cache_size = cache_size
        self.__s3 = None

    def get(self, url):
        local_filename = self._cache_index.get(url)
        if local_filename is None:
            local_filename = self._local_filename(url)
            self._cache_index[url] = local_filename
            self._download_to_cache(url)
            if len(self._cache_index) > self._cache_size:
                self._expire_oldest()
        else:
            self._logger.info(f'Serving {url} from local cache')

        return local_filename

    def _local_filename(self, url):
        return os.path.join(
            self._directory.name,
            hashlib.sha256(url.encode('utf-8')).hexdigest(),
        )

    def _download_to_cache(self, url):
        local_filename = self._cache_index[url]

        if url[0:5] == 's3://':
            url = url[5:]

        (bucket, key) = url.split('/', 1)

        self._logger.info(f'Starting download of {url} to {local_filename}')
        self._s3.download_file(
            Bucket=bucket,
            Key=key,
            Filename=local_filename,
            ExtraArgs={'RequestPayer': 'requester'},
        )
        self._logger.info(f'Finished download of {url} to {local_filename}')

    def _expire_oldest(self):
        (url, local_file) = self._cache_index.popitem(last=False)
        self._logger.info(f'Deleting {url} from the cache')
        os.remove(local_file)

    @property
    def _s3(self):
        if self.__s3 is None:
            self._logger.info('Initialising AWS S3 client')
            self.__s3 = boto3.client('s3')

        return self.__s3


class HSTImageDataset(Dataset):
    def __init__(self, directory: str, transform: callable):
        self.logger = logging.getLogger()
        self.transform = transform
        self.s3_downloader = CachingS3Downloader()

        if os.path.isdir(directory):
            self.fits_files = self._scan_local_directory(directory)

        elif os.path.isfile(directory):
            self.fits_files = self._scan_index_file(directory)

        else:
            raise ValueError(f'{directory} is not a valid directory a file')

        self.logger.info(f'Found {len(self.fits_files)} in {directory}')
        self.logger.info('Transform {0} has a multiplier of {1}'.format(
            self.transform.__class__.__name__, self.transform.multiplier
        ))
        self.logger.info(f'Total dataset size is {self.__len__()} images')

    def _scan_local_directory(self, directory):
        self.logger.info(f'Scanning local folder {directory} for FITS files')

        files = list(filter(
            lambda f: f[-5:].lower() == '.fits',
            os.listdir(directory),
        ))
        absolute_files = list(map(
            lambda f: os.path.join(directory, f),
            files,
        ))
        return absolute_files

    def _scan_index_file(self, index_file):
        self.logger.info(f'Scanning index file {index_file} for FITS files')
        return open(index_file).read().splitlines()

    def __len__(self):
        return len(self.fits_files) * self.transform.multiplier

    def __getitem__(self, index):
        file_index = index // self.transform.multiplier
        subimage_index = index % self.transform.multiplier

        filename = self.s3_downloader.get(self.fits_files[file_index])
        fits_data = fits.getdata(filename)

        return {
            'src_filename': filename,
            'file_index': file_index,
            'subimage_index': subimage_index,
            'image_data': self.transform(fits_data, subimage_index),
        }
