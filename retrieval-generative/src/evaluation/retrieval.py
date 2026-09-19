"""全物品库召回评估：Recall、HitRate、NDCG 和 Item Coverage。"""

import math

import torch


@torch.no_grad()
def evaluate_retrieval(model, examples, num_items: int, ks=(10, 50), device: str = "cpu"):
    """评估每个用户的一个留出目标，并排除已经看过的历史物品。"""
    if not examples:
        return {f"recall@{k}": 0.0 for k in ks}

    item_ids = torch.arange(num_items, device=device)
    item_vectors = model.encode_item(item_ids)
    max_k = min(max(ks), num_items)
    hits = {k: 0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    covered_items = set()

    for history, target in examples:
        history_tensor = torch.tensor(history, dtype=torch.long, device=device).unsqueeze(0)
        if hasattr(model, "encode_user"):
            scores = (model.encode_user(history_tensor) @ item_vectors.T).squeeze(0)
        else:
            interests = model.encode_interests(history_tensor)
            scores = (interests @ item_vectors.T).amax(dim=1).squeeze(0)
        if history:
            # 召回评估不能把用户已经看过的物品当成新候选。
            scores[torch.tensor(history, dtype=torch.long, device=device)] = -torch.inf
        ranking = scores.topk(max_k).indices.tolist()
        # ranking 既用于命中率，也用于覆盖率和目标物品排名计算。
        covered_items.update(ranking)
        rank = ranking.index(target) + 1 if target in ranking else None
        for k in ks:
            if rank is not None and rank <= k:
                hits[k] += 1
                ndcg[k] += 1.0 / math.log2(rank + 1)

    count = len(examples)
    result = {}
    for k in ks:
        result[f"recall@{k}"] = hits[k] / count
        result[f"hitrate@{k}"] = hits[k] / count
        result[f"ndcg@{k}"] = ndcg[k] / count
    result[f"item_coverage@{max_k}"] = len(covered_items) / num_items
    return result


def recall_at_k(model, examples, num_items: int, k: int = 10, device: str = "cpu") -> float:
    return evaluate_retrieval(model, examples, num_items, ks=(k,), device=device)[f"recall@{k}"]
