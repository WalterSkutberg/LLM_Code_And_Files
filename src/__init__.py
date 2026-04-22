"""
Paket för språkmodellsimplementering
"""

from .data_loader import DataLoader
from .tokenizer import BPETokenizer
from .model import TransformerLM, MultiHeadAttention, TransformerBlock
from .trainer import ModelTrainer, TrainingConfig
from .evaluator import ModelEvaluator, LanguageVariationAnalyzer
from .utils import (
    count_parameters,
    get_model_size_mb,
    print_model_summary,
    setup_device,
    seed_everything
)

__version__ = "0.1.0"
__author__ = "LLM Development"

__all__ = [
    'DataLoader',
    'BPETokenizer',
    'TransformerLM',
    'ModelTrainer',
    'TrainingConfig',
    'ModelEvaluator',
    'LanguageVariationAnalyzer',
    'count_parameters',
    'get_model_size_mb',
    'print_model_summary',
    'setup_device',
    'seed_everything'
]
