import os
import argparse
import re
import logging

import tmad.fits
from astropy.io import fits
import astropy.visualization
import imageio

from tmad.logging import configure_logging


def main():
    args = parse_arguments()
    configure_logging(args.log_level)
    logger = logging.getLogger()

    # Interval and stretch functions could easily be made configurable
    transform = tmad.fits.TMAD_ACS_WFC_RAW_FITS_Transform(
        interval_function=tmad.fits.ZMaxInterval(),
        stretch_function=astropy.visualization.LogStretch(),
        scale=(0, 255),
    )

    if args.fits_file:
        fits_data = fits.getdata(args.fits_file)
        for index in range(transform.multiplier):
            output_filename = make_output_filename(
                args.fits_file, index, args.output_dir,
            )
            imageio.imwrite(
                output_filename,
                transform(fits_data, index).astype('uint8'),
            )

    elif args.fits_fileset:
        dataset = tmad.fits.HSTImageDataset(
            directory=args.fits_fileset,
            transform=transform,
        )
        for image in dataset:
            output_filename = make_output_filename(
                image['src_filename'],
                image['subimage_index'],
                args.output_dir
            )
            logger.info('Exporting tile {0} from {1} to {2}'.format(
                image['subimage_index'],
                image['src_filename'],
                output_filename,
            ))
            imageio.imwrite(
                output_filename,
                image['image_data'].astype('uint8'),
            )


def make_output_filename(input_filename, index, output_dir):
    r = re.compile(r'\.fits', re.IGNORECASE)
    basename_in = os.path.basename(input_filename)
    basename_out = r.sub('', basename_in)
    basename_out += f'.{index}.png'
    absolute_out = os.path.join(output_dir, basename_out)
    return absolute_out


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="FITS Transform Tool"
    )

    input_selection = parser.add_mutually_exclusive_group()

    input_selection.add_argument(
        '--fits-file',
        type=str,
        help='Single FITS file to process',
    )

    input_selection.add_argument(
        '--fits-fileset',
        type=str,
        help=(
            'Either a directory of FITS files, '
            'or a text file containing a list of URLs'
        )
    )

    parser.add_argument(
        '--output-dir',
        required=False,
        type=str,
        default=os.curdir,
        help='Local folder to write images to (default: current directory)',
    )

    parser.add_argument(
        '--log-level',
        required=False,
        type=str,
        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
        default='INFO',
        help='Log level (default: INFO)',
    )

    return parser.parse_args()


if __name__ == '__main__':
    main()
