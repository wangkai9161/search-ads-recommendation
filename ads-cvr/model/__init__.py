"""Nine classic recommendation models with one construction interface."""

from .deepfm import DeepFM, DeepFMModel, DeepFMCVR, DeepFMRecall
from .dien import DIEN, DIENModel
from .din import DIN, DINModel
from .dssm import DSSM
from .esmm import ESMM, ESMMModel
from .fm import FM, FMRecall
from .interface import ModelInterface
from .lr import LR, LogisticCVR
from .registry import MODEL_REGISTRY, available_models, build_model
from .transformer import DecoderOnlyRecall, SASRec, SASRecModel, Transformer, TransformerSequence
from .wide_deep import SparseDenseEncoder, WideAndDeep, WideDeep, WideDeepModel

__all__ = [
    "ModelInterface",
    "MODEL_REGISTRY",
    "available_models",
    "build_model",
    "LR",
    "LogisticCVR",
    "FM",
    "FMRecall",
    "WideDeep",
    "WideAndDeep",
    "WideDeepModel",
    "SparseDenseEncoder",
    "DeepFM",
    "DeepFMModel",
    "DeepFMCVR",
    "DeepFMRecall",
    "DSSM",
    "DIN",
    "DINModel",
    "DIEN",
    "DIENModel",
    "ESMM",
    "ESMMModel",
    "SASRec",
    "SASRecModel",
    "Transformer",
    "TransformerSequence",
    "DecoderOnlyRecall",
]
