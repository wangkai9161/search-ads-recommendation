"""训练小型 Decoder-only 下一物品预测模型。"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.movielens import load_sequences
from src.models.generative import DecoderOnlyRecall


class Samples(Dataset):
    def __init__(self, examples): self.examples = examples
    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]


def collate(batch):
    histories, targets = zip(*batch)
    # 限制最大序列长度，避免长历史导致 Transformer 计算量过大。
    histories = [torch.tensor(x[-50:], dtype=torch.long) for x in histories]
    lengths = torch.tensor([len(x) for x in histories])
    padded = pad_sequence(histories, batch_first=True)
    mask = torch.arange(padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    return padded, mask, torch.tensor(targets)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--data-dir', default='data/raw/ml-1m')
    parser.add_argument('--epochs', type=int, default=2); parser.add_argument('--max-users', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=256); parser.add_argument('--embedding-dim', type=int, default=64)
    args = parser.parse_args(); random.seed(42); torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'; data = load_sequences(args.data_dir, max_users=args.max_users)
    model = DecoderOnlyRecall(data.num_items, args.embedding_dim).to(device)
    loader = DataLoader(Samples(data.train), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print(f'device={device} users={data.num_users} items={data.num_items} train={len(data.train)} test={len(data.test)}')
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for histories, mask, targets in loader:
            histories, mask, targets = histories.to(device), mask.to(device), targets.to(device)
            optimizer.zero_grad(); logits = model(histories, mask); loss = torch.nn.functional.cross_entropy(logits, targets)
            loss.backward(); optimizer.step(); total += loss.item() * len(targets)
        model.eval(); hits10 = hits50 = 0; count = len(data.test); candidates = []
        popularity = Counter()
        for history, target in data.test:
            popularity.update(history); popularity[target] += 1
        cutoff = sorted(popularity.values())[max(0, len(popularity) // 2 - 1)]
        tail_items = {item for item, frequency in popularity.items() if frequency <= cutoff}
        tail_hits = 0
        # 将测试历史批量推理，避免逐用户重复调用 Transformer。
        for start in range(0, len(data.test), args.batch_size):
            batch = data.test[start:start + args.batch_size]
            histories = [history[-50:] for history, _ in batch]
            lengths = torch.tensor([len(history) for history in histories], device=device)
            padded = pad_sequence(
                [torch.tensor(history, dtype=torch.long) for history in histories],
                batch_first=True,
            ).to(device)
            mask = torch.arange(padded.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
            score_batch = model(padded, mask)
            for row, (history, target) in enumerate(batch):
                scores = score_batch[row]
                if history:
                    # 生成式召回评估同样排除历史已看物品。
                    scores[torch.tensor(history, device=device)] = -torch.inf
                top = scores.topk(50).indices.tolist(); candidates.extend(top)
                hits10 += int(target in top[:10]); hits50 += int(target in top)
                tail_hits += sum(item in tail_items for item in top)
        unique_ratio = len(set(candidates)) / max(len(candidates), 1)
        tail_ratio = tail_hits / max(len(candidates), 1)
        print(f'epoch={epoch} loss={total / len(data.train):.4f} recall@10={hits10/count:.4f} recall@50={hits50/count:.4f} item_coverage@50={len(set(candidates))/data.num_items:.4f} candidate_unique_ratio={unique_ratio:.4f} tail_ratio={tail_ratio:.4f}')


if __name__ == '__main__': main()
