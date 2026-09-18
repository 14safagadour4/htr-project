import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

from dataset import get_mnist_loaders
from model import SimpleCNN


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1).cpu()

        all_preds.append(preds)
        all_labels.append(labels)
        all_probs.append(probs.cpu())

    return (torch.cat(all_preds).numpy(),
            torch.cat(all_labels).numpy(),
            torch.cat(all_probs).numpy())


def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_DIR = '../runs'
    os.makedirs(SAVE_DIR, exist_ok=True)

    _, test_loader = get_mnist_loaders(batch_size=256)

    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pth'),
                                     map_location=DEVICE))

    preds, labels, probs = collect_predictions(model, test_loader, DEVICE)

    acc = (preds == labels).mean()
    print(f"Test Accuracy: {acc:.4f}\n")
    print(classification_report(labels, preds, digits=4))

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title(f'MNIST Confusion Matrix (Acc={acc:.4f})')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'), dpi=150)
    print("Saved confusion_matrix.png")

    # Top-20 most confident errors
    errors = np.where(preds != labels)[0]
    error_conf = probs[errors, preds[errors]]
    top_error_idx = errors[np.argsort(-error_conf)[:20]]

    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    for ax, idx in zip(axes.flat, top_error_idx):
        img = test_loader.dataset[idx][0].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"T:{labels[idx]} P:{preds[idx]}\n"
                     f"({probs[idx, preds[idx]]:.2f})", fontsize=9)
        ax.axis('off')
    plt.suptitle('Top-20 Most Confident Errors', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'error_analysis.png'), dpi=150)
    print("Saved error_analysis.png")


if __name__ == "__main__":
    main()