"""
MRN (Modular Reasoning Network) Core Package

This package implements the complete MRN system based on exact matching
in modular space with the following components:

- modular_space: Modular arithmetic operations in Z_m^n
- matching: Exact matching functions
- embedding: Symbol embedding system
- derivation: Two-step derivation process
- sequence_generator: Sequence generation with dual-channel memory
- training: Loss functions and training algorithms
"""

from .modular_space import ModularSpace
from .matching import ExactMatcher, SmoothMatchingLoss
from .embedding import SymbolEmbedding, EmbeddingGradient
from .derivation import TwoStepDerivation, DerivationGraph
from .sequence_generator import MRNGenerator, HistoryWindow, ContextVector
from .training import LossFunction, MRNTrainer, CurriculumLearning

__all__ = [
    # Modular space
    'ModularSpace',
    
    # Matching
    'ExactMatcher',
    'SmoothMatchingLoss',
    
    # Embedding
    'SymbolEmbedding',
    'EmbeddingGradient',
    
    # Derivation
    'TwoStepDerivation',
    'DerivationGraph',
    
    # Sequence generation
    'MRNGenerator',
    'HistoryWindow',
    'ContextVector',
    
    # Training
    'LossFunction',
    'MRNTrainer',
    'CurriculumLearning',
]

__version__ = '0.1.0'
