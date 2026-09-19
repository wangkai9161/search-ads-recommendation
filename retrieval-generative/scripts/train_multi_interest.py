"""训练多兴趣召回基线。"""

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
from src.models.multi_interest import MultiInterestRecall


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
    # 兴趣聚合时忽略 padding，避免产生虚假兴趣。
    mask = torch.arange(padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    return padded, mask, torch.tensor(targets)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--data-dir', default='data/raw/ml-1m')
    parser.add_argument('--epochs', type=int, default=2); parser.add_argument('--max-users', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=512); parser.add_argument('--num-interests', type=int, default=4)
    parser.add_argument('--diversity-weight', type=float, default=0.01)
    parser.add_argument('--temperature', type=float, default=0.2)
    args = parser.parse_args(); random.seed(42); torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'; data = load_sequences(args.data_dir, max_users=args.max_users)
    model = MultiInterestRecall(data.num_items, num_interests=args.num_interests).to(device)
    loader = DataLoader(Samples(data.train), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print(f'device={device} users={data.num_users} items={data.num_items} train={len(data.train)} test={len(data.test)} interests={args.num_interests}')
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for histories, mask, targets in loader:
            histories, mask, targets = histories.to(device), mask.to(device), targets.to(device)
            optimizer.zero_grad()
            # 主任务让所有兴趣共同参与匹配，辅助 Loss 只负责避免兴趣完全重合。
            loss = model.in_batch_loss(histories, targets, mask, args.temperature) + args.diversity_weight * model.diversity_loss(histories, mask)
            loss.backward(); optimizer.step(); total += loss.item() * len(targets)
        model.eval(); metrics = evaluate_retrieval(model, data.test, data.num_items, device=device)
        text = ' '.join(f'{k}={v:.4f}' for k, v in metrics.items())
        print(f'epoch={epoch} loss={total / len(data.train):.4f} {text}')


if __name__ == '__main__': main()
