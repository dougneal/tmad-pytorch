import argparse
import logging

import torchvision.datasets
import torchvision.transforms
import torch.utils.data

from tmad.container import Container

IMAGE_SIZE = 64


def main():
    args = parse_arguments()
    configure_logging(args.log_level)

    logger = logging.getLogger()
    logger.info('Starting up')

    dataset = create_imagefolder_dataset(args.training_data)
    dataloader = create_data_loader(dataset, args.batch_size)

    container = Container(
        dataloader=dataloader,
        model_dir=args.model,
        output_dir=args.output,
        training_epochs=args.training_epochs,
    )

    container.train()


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
        '--training-epochs',
        required=False,
        type=int,
        default=50,
        help='Number of training epochs (default: 50)',
    )

    return parser.parse_args()


def configure_logging(level_name):
    level = {
        'CRITICAL': logging.CRITICAL,
        'ERROR': logging.ERROR,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG
    }.get(level_name)

    log_format = (
        '%(asctime)s | %(levelname)s | '
        '%(name)s %(funcName)s | %(message)s'
    )
    logging.basicConfig(format=log_format, level=level)


# For now this is just the ImageFolder dataset lifted from the tutorial.
def create_imagefolder_dataset(datadir: str):
    logger = logging.getLogger()
    logger.info(f'Making torchvision.datasets.ImageFolder for {datadir}')

    return torchvision.datasets.ImageFolder(
        root=datadir,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(IMAGE_SIZE),
            torchvision.transforms.CenterCrop(IMAGE_SIZE),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            )
        ])
    )


def create_data_loader(
    dataset: torchvision.datasets.ImageFolder,
    batch_size: int
):
    logger = logging.getLogger()
    logger.info(f'Making torch.utils.data.DataLoader for dataset {dataset}')

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )


if __name__ == '__main__':
    main()
