import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(batch_size=64, data_dir='../data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean & std
    ])

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_mnist_loaders()
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Pixel range: [{images.min():.2f}, {images.max():.2f}]")