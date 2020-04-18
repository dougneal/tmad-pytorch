import os
import argparse
import re
import logging
import sys
import astropy.visualization

import tmad.fits

from tmad.logging import configure_logging
from tmad.fits import HSTImageDataset, HSTS3ImageDataset
from tmad.cloudio import LocalFilesystemSaver, GoogleDriveUploader


def main():
    args = parse_arguments()
    configure_logging(args.log_level)
    logger = logging.getLogger()

    if args.s3:
        dataset_class = HSTS3ImageDataset
    else:
        dataset_class = HSTImageDataset

    dataset = None

    if args.fits_file:
        dataset = dataset_class(files=[args.fits_file])

    elif args.index_file:
        dataset = dataset_class.from_index_file(args.index_file)

    elif args.fits_dir:
        dataset = dataset_class.from_directory(args.fits_dir)

    else:
        logger.fatal("Couldn't determine dataset?!")
        return 1

    # Interval and stretch functions could easily be made configurable
    dataset.transform = tmad.fits.TMAD_ACS_WFC_RAW_FITS_Transform(
        interval_function=tmad.fits.ZMaxInterval(),
        stretch_function=astropy.visualization.LogStretch(),
        scale=(0, 255),
    )

    saver = None
    if args.output_dir is not None:
        saver = LocalFilesystemSaver(args.output_dir)

    elif args.google_drive is not None:
        saver = GoogleDriveUploader(args.google_drive)

    else:
        logger.fatal("Couldn't determine output method")
        return 1

    for image in dataset:
        saver(image)

    return 0


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

    input_selection = parser.add_mutually_exclusive_group(required=True)

    input_selection.add_argument(
        '--fits-file',
        type=str,
        help='Single FITS file to process',
    )

    input_selection.add_argument(
        '--index-file',
        type=str,
        help=('Name of text file containing paths of FITS files'),
    )

    input_selection.add_argument(
        '--fits-dir',
        type=str,
        help=('A local directory containing FITS files'),
    )

    parser.add_argument(
        '--s3',
        action='store_true',
        help=('Load files from S3'),
    )

    output_selection = parser.add_mutually_exclusive_group(required=True)

    output_selection.add_argument(
        '--output-dir',
        type=str,
        help='Local folder to write images to (default: current directory)',
    )

    output_selection.add_argument(
        '--google-drive',
        type=str,
        help=(
            'Google Drive folder to write images to. '
            'Specify as the folder ID, not the name.'
        ),
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
    sys.exit(main())
