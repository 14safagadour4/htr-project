import torch


def greedy_decode(log_probs, idx2char, blank=0):
    preds = log_probs.argmax(dim=2).transpose(0, 1)
    results = []
    for seq in preds:
        chars, prev = [], None
        for idx in seq:
            idx = idx.item()
            if idx != blank and idx != prev:
                chars.append(idx2char[idx])
            prev = idx
        results.append(''.join(chars))
    return results


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(curr[j-1]+1, prev[j]+1, prev[j-1]+(ca != cb)))
        prev = curr
    return prev[-1]


def compute_cer(preds, targets):
    total_chars = sum(len(t) for t in targets)
    total_errors = sum(edit_distance(p, t) for p, t in zip(preds, targets))
    return total_errors / max(total_chars, 1)


def compute_wer(preds, targets):
    total_words = sum(len(t.split()) for t in targets)
    total_errors = sum(edit_distance(p.split(), t.split()) for p, t in zip(preds, targets))
    return total_errors / max(total_words, 1)
