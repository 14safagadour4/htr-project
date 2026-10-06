import json
import matplotlib.pyplot as plt

with open('/content/drive/MyDrive/week3_crnn/results.json') as f:
    h = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(h['train_loss'], label='Train', color='blue')
axes[0].plot(h['val_loss'], label='Validation', color='red')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('CTC Loss')
axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(h['val_cer'], label='Val CER', color='green')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('CER')
axes[1].set_title('Character Error Rate'); axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(h['val_wer'], label='Val WER', color='purple')
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('WER')
axes[2].set_title('Word Error Rate'); axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/week3_crnn/curves.png', dpi=150)
print("Saved curves.png")
