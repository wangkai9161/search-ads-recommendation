"""训练 DeepFM 风格召回基线。"""

import argparse
import random
import sys
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.movielens import load_sequences
from src.evaluation.retrieval import evaluate_retrieval
from src.models.deepfm import DeepFMRecall


class Samples(Dataset):
    def __init__(self, examples): self.examples = examples
    def __len__(self): return len(self.examples)
    def __getitem__(self, i):
        history, target = self.examples[i]
        return torch.tensor(history), target


def collate(batch):
    histories, targets = zip(*batch)
    lengths = torch.tensor([len(x) for x in histories])
    padded = pad_sequence(histories, batch_first=True)
    # mask 用于区分真实历史和 batch padding。
    mask = torch.arange(padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    return padded, mask, torch.tensor(targets)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--data-dir', default='data/raw/ml-1m')
    parser.add_argument('--epochs', type=int, default=1); parser.add_argument('--max-users', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=512); args = parser.parse_args()
    random.seed(42); torch.manual_seed(42); device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = load_sequences(args.data_dir, max_users=args.max_users)
    model = DeepFMRecall(data.num_items).to(device)
    loader = DataLoader(Samples(data.train), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print(f'device={device} users={data.num_users} items={data.num_items} train={len(data.train)} test={len(data.test)}')
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for histories, mask, targets in loader:
            histories, mask, targets = histories.to(device), mask.to(device), targets.to(device)
            optimizer.zero_grad(); loss = model.in_batch_loss(histories, targets, mask)
            loss.backward(); optimizer.step(); total += loss.item() * len(targets)
        model.eval(); metrics = evaluate_retrieval(model, data.test, data.num_items, device=device)
        text = ' '.join(f'{k}={v:.4f}' for k, v in metrics.items())
        print(f'epoch={epoch} loss={total / len(data.train):.4f} {text}')


if __name__ == '__main__': main()
