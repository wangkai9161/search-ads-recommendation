"""Microsoft MIND behaviors.tsv 和 news.tsv 的中文说明版解析器。"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class MindSample:
    user_id: str
    history: list[str]
    candidate: str
    label: int
    impression_id: str


def load_behaviors(path: str | Path, split_impressions: bool = True) -> list[MindSample]:
    """把曝光列表拆成点击/未点击样本，保留 impression_id 便于分析曝光偏差。"""
    samples = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        impression_id, user_id, _, history_text, impressions_text = line.split("\t")
        history = [] if history_text == "" else history_text.split(" ")
        # 同一曝光列表中，点击为正样本，未点击为真实曝光负样本。
        for impression in impressions_text.split(" "):
            news_id, label_text = impression.rsplit("-", 1)
            samples.append(MindSample(user_id, history, news_id, int(label_text), impression_id))
        if not split_impressions:
            break
    return samples


def load_news(path: str | Path) -> dict[str, dict[str, str]]:
    """读取新闻类别和文本信息，并按 news_id 建立索引。"""
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) < 8:
            continue
        news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities = fields[:8]
        result[news_id] = {
            "category": category,
            "subcategory": subcategory,
            "title": title,
            "abstract": abstract,
        }
    return result
