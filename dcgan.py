import argparse
import logging

from tmad.logging import configure_logging
from tmad.utils import create_imagefolder_dataset, create_data_loader
from tmad.trainer import Trainer


def main():
    args = parse_arguments()
    configure_logging(args.log_level)

    logger = logging.getLogger()
    logger.info('Starting up')

    dataset = create_imagefolder_dataset(args.training_data, args.image_size)
    dataloader = create_data_loader(dataset, args.batch_size)

    trainer = Trainer(
        dataloader=dataloader,
        model_dir=args.model,
        output_dir=args.output,
        training_epochs=args.training_epochs,
    )

    if args.dump_model:
        trainer.dump_model()

    else:
        trainer.train()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='TMAD DCGAN'
    )

    parser.add_argument(
        '--log-level',
        required=False,
        type=str,
        choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
        default='INFO',
        help='Log level (default: INFO)',
    )

    parser.add_argument(
        '--training-data',
        required=True,
        type=str,
        help='Location of training images',
    )

    parser.add_argument(
        '--model',
        required=True,
        type=str,
        help='Filesystem path to save/load DCGAN model state',
    )

    parser.add_argument(
        '--output',
        required=True,
        type=str,
        help='Filesystem path to save images generated during training',
    )

    parser.add_argument(
        '--batch-size',
        required=False,
        type=int,
        default=128,
        help='Batch size during training (default: 128)',
    )

    parser.add_argument(
        '--image-size',
        required=False,
        type=int,
        default=64,
        choices=[64],
        help='Image size (square), both for training input and fake output',
    )

    parser.add_argument(
        '--training-epochs',
        required=False,
        type=int,
        default=50,
        help='Number of training epochs (default: 50)',
    )

    parser.add_argument(
        '--dump-model',
        required=False,
        action="store_true",
    )

    return parser.parse_args()


if __name__ == '__main__':
    main()
