import logging
import torchvision.datasets
import torch.utils.data


# For now this is just the ImageFolder dataset lifted from the tutorial.
def create_imagefolder_dataset(datadir: str, image_size: int):
    logger = logging.getLogger()
    logger.info(f'Making torchvision.datasets.ImageFolder for {datadir}')

    return torchvision.datasets.ImageFolder(
        root=datadir,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.CenterCrop(image_size),
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
        # Don't spawn multiple loader processes.
        # Do the loading in CPU-land, then migrate tensors to the GPUs.
        num_workers=0,
        pin_memory=False,
    )
