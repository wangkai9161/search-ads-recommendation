"""Registry and factory for the nine core recommendation models."""

from __future__ import annotations

from .deepfm import DeepFM
from .dien import DIEN
from .din import DIN
from .dssm import DSSM
from .esmm import ESMM
from .fm import FM
from .lr import LR
from .transformer import SASRec
from .wide_deep import WideDeep


MODEL_REGISTRY = {
    "lr": LR,
    "fm": FM,
    "wide_deep": WideDeep,
    "deepfm": DeepFM,
    "dssm": DSSM,
    "din": DIN,
    "dien": DIEN,
    "esmm": ESMM,
    "transformer": SASRec,
}

_ALIASES = {
    "wideanddeep": "wide_deep",
    "wide-and-deep": "wide_deep",
    "deep_fm": "deepfm",
    "sasrec": "transformer",
    "transformer_sequence": "transformer",
}


def available_models() -> tuple[str, ...]:
    return tuple(MODEL_REGISTRY)


def _canonical_name(name: str) -> str:
    normalized = name.lower().strip()
    return _ALIASES.get(normalized, normalized)


def build_model(
    name: str,
    num_sparse_bins: int | None = None,
    num_dense: int | None = None,
    num_sparse_fields: int | None = None,
    embedding_dim: int = 16,
    **kwargs,
):
    """Build a core model using native kwargs or the legacy CVR arguments."""
    canonical = _canonical_name(name)
    if canonical not in MODEL_REGISTRY:
        available = ", ".join(available_models())
        raise ValueError(f"Unknown model {name!r}; choose one of: {available}")

    if canonical == "lr":
        if num_sparse_bins is None or num_dense is None:
            raise ValueError("lr requires num_sparse_bins and num_dense")
        return LR(num_sparse_bins, num_sparse_fields, num_dense)
    if canonical == "fm":
        if num_sparse_bins is None or num_sparse_fields is None or num_dense is None:
            raise ValueError("fm requires num_sparse_bins, num_sparse_fields, and num_dense")
        return FM(num_sparse_bins, num_sparse_fields, num_dense, embedding_dim=embedding_dim, **kwargs)
    if canonical == "wide_deep":
        if num_sparse_bins is None or num_sparse_fields is None or num_dense is None:
            raise ValueError("wide_deep requires num_sparse_bins, num_sparse_fields, and num_dense")
        return WideDeep(num_sparse_bins, num_sparse_fields, num_dense, embedding_dim=embedding_dim, **kwargs)
    if canonical == "deepfm":
        if num_sparse_bins is None or num_sparse_fields is None or num_dense is None:
            raise ValueError("deepfm requires num_sparse_bins, num_sparse_fields, and num_dense")
        return DeepFM(num_sparse_bins, num_sparse_fields, num_dense, embedding_dim=embedding_dim, **kwargs)
    if canonical == "dssm":
        return DSSM(kwargs.pop("num_items"), embedding_dim=embedding_dim, **kwargs)
    if canonical in {"din", "dien", "transformer"}:
        return MODEL_REGISTRY[canonical](kwargs.pop("num_items"), embedding_dim=embedding_dim, **kwargs)
    if canonical == "esmm":
        return ESMM(
            input_dim=kwargs.pop("input_dim", None),
            num_sparse_bins=num_sparse_bins,
            num_sparse_fields=num_sparse_fields,
            num_dense=num_dense,
            embedding_dim=embedding_dim,
            **kwargs,
        )
    raise AssertionError(f"unhandled model {canonical}")


__all__ = ["MODEL_REGISTRY", "available_models", "build_model"]
