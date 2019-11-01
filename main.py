import torchvision.datasets
import torchvision.transforms
import torch.utils.data

from tmad.container import Container

IMAGE_SIZE=64

def main():
    dataset = torchvision.datasets.ImageFolder(
        root='data',
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
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=1,
    )

    container = Container(loader, '.')
    container.train(1)


if __name__ == '__main__':
    main()
