import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_emnist_loaders(batch_size=64, data_dir='../data',
                       val_size=5000, augment=True):
    

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1736,), (0.3317,))
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3317,))
        ])
    else:
        train_transform = test_transform

    full_train = datasets.EMNIST(
        root=data_dir, split='balanced', train=True,
        download=True, transform=train_transform
    )
    test_dataset = datasets.EMNIST(
        root=data_dir, split='balanced', train=False,
        download=True, transform=test_transform
    )

    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_emnist_loaders()
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label range: [{labels.min()}, {labels.max()}]")
    print(f"Pixel range: [{images.min():.2f}, {images.max():.2f}]")