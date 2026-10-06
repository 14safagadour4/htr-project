import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import os, json, argparse

from dataset import get_loaders, NUM_CLASSES, IDX2CHAR
from model import CRNN
from ctc_utils import greedy_decode, compute_cer, compute_wer


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    for imgs, targets, target_lengths, _ in tqdm(loader, desc="Train", leave=False):
        imgs, targets = imgs.to(device), targets.to(device)
        target_lengths = target_lengths.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        log_probs = logits.log_softmax(2)
        T, B, C = log_probs.shape
        input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
    return running_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []
    for imgs, targets, target_lengths, texts in loader:
        imgs, targets_dev = imgs.to(device), targets.to(device)
        target_lengths = target_lengths.to(device)
        logits = model(imgs)
        log_probs = logits.log_softmax(2)
        T, B, C = log_probs.shape
        input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
        loss = criterion(log_probs, targets_dev, input_lengths, target_lengths)
        total_loss += loss.item()
        preds = greedy_decode(log_probs.cpu(), IDX2CHAR)
        all_preds.extend(preds)
        all_targets.extend(texts)
    return total_loss / len(loader), compute_cer(all_preds, all_targets), compute_wer(all_preds, all_targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_DIR = '/content/drive/MyDrive/week3_crnn'
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    train_loader, val_loader, test_loader = get_loaders(args.batch_size)
    model = CRNN(NUM_CLASSES).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    warmup_steps = 2 * len(train_loader)

    scheduler = LambdaLR(optimizer, lambda s: s / max(1, warmup_steps) if s < warmup_steps else 1.0)

    history = {'train_loss': [], 'val_loss': [], 'val_cer': [], 'val_wer': []}
    best_cer = float('inf')

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, DEVICE)
        vl_loss, vl_cer, vl_wer = evaluate(model, val_loader, criterion, DEVICE)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['val_cer'].append(vl_cer)
        history['val_wer'].append(vl_wer)
        print(f"Epoch {epoch:02d} | Train {tr_loss:.4f} | Val {vl_loss:.4f} | CER {vl_cer:.4f} | WER {vl_wer:.4f}")
        if vl_cer < best_cer:
            best_cer = vl_cer
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best.pth'))

    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best.pth')))
    te_loss, te_cer, te_wer = evaluate(model, test_loader, criterion, DEVICE)
    history['test_cer'] = te_cer
    history['test_wer'] = te_wer
    with open(os.path.join(SAVE_DIR, 'results.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTest CER: {te_cer:.4f} | Test WER: {te_wer:.4f}")


if __name__ == "__main__":
    main()