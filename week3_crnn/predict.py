import torch
import matplotlib.pyplot as plt
from dataset import get_loaders, NUM_CLASSES, IDX2CHAR
from model import CRNN
from ctc_utils import greedy_decode

# Auto-detect device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

model = CRNN(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load('/content/drive/MyDrive/week3_crnn/best.pth',
                                  map_location=DEVICE))
model.eval()

_, _, test_loader = get_loaders(batch_size=8)

imgs_list, preds_list, targets_list = [], [], []
with torch.no_grad():
    for imgs, targets, target_lengths, texts in test_loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        log_probs = logits.log_softmax(2)
        preds = greedy_decode(log_probs.cpu(), IDX2CHAR)
        for i in range(len(imgs)):
            imgs_list.append(imgs[i].cpu())
            preds_list.append(preds[i])
            targets_list.append(texts[i])
        if len(imgs_list) >= 10:
            break

fig, axes = plt.subplots(10, 1, figsize=(14, 22))
for i, ax in enumerate(axes):
    ax.imshow(imgs_list[i].squeeze().numpy(), cmap='gray', aspect='auto')
    ax.set_title(f"TRUE: {targets_list[i]}\nPRED: {preds_list[i]}", fontsize=9)
    ax.axis('off')
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/week3_crnn/predictions.png', dpi=150)
print("Saved predictions.png")
