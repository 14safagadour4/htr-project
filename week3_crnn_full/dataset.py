import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
import torchvision.transforms as transforms


CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\"-:;() "
VOCAB = ['<blank>'] + list(CHARS)
CHAR2IDX = {c: i for i, c in enumerate(VOCAB)}
IDX2CHAR = {i: c for i, c in enumerate(VOCAB)}
NUM_CLASSES = len(VOCAB)


def encode_text(text):
    indices = [CHAR2IDX[c] for c in text if c in CHAR2IDX]
    return torch.tensor(indices, dtype=torch.long)


class IAMLineDataset(Dataset):
    def __init__(self, hf_split, img_height=32, img_width=512):
        self.data = hf_split
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample['image'].convert('L')
        text = sample['text']
        image = self.transform(image)
        target = encode_text(text)
        return image, target, text


def collate_fn(batch):
    imgs, targets, texts = zip(*batch)
    imgs = torch.stack(imgs)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets = torch.cat(targets)
    return imgs, targets, target_lengths, texts


def get_loaders(batch_size=16):
    print("Loading Teklia/IAM-line from Hugging Face...")
    hf = load_dataset("Teklia/IAM-line")
    train_ds = IAMLineDataset(hf['train'])
    val_ds = IAMLineDataset(hf['validation'])
    test_ds = IAMLineDataset(hf['test'])
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=2)
    return train_loader, val_loader, test_loader
