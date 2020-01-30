import argparse
import re
import tmad.fits
from astropy.io import fits
import astropy.visualization
import imageio


def main():
    args = parse_arguments()

    transform = tmad.fits.TMAD_ACS_WFC_RAW_FITS_Transform(
        interval_function=tmad.fits.ZMaxInterval(),
        stretch_function=astropy.visualization.LogStretch(),
        scale=(0, 255),
    )

    fits_data = fits.getdata(args.fits)
    for index in range(transform.multiplier):
        imageio.imwrite(
            output_filename(args.fits, index),
            transform(fits_data, index).astype('uint8'),
        )


def output_filename(input_filename, index):
    r = re.compile(r'\.fits', re.IGNORECASE)
    return r.sub(f'.{index}.png', input_filename)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="FITS Transform Tool"
    )

    parser.add_argument(
        '--fits',
        required=True,
        type=str,
        help='Input FITS file',
    )

    return parser.parse_args()


if __name__ == '__main__':
    main()
