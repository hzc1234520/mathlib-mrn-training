"""
MRN Core Package
Modular Reasoning Network - Complete Implementation
"""

from .symbol_embedding import SymbolEmbedding
from .modular_space import ModularSpace
from .matching import MatchingFunction, BestMatchSelector
from .memory import HistoryWindow, ContextVector, MemoryManager
from .derivation import DerivationPath, TwoStepDerivation, ParallelDerivation
from .sequence_generator import SequenceGenerator, BeamSearchGenerator
from .training import MRNTrainer, CurriculumLearning, LossFunction

__version__ = '1.0.0'

__all__ = [
    # Symbol embedding
    'SymbolEmbedding',
    
    # Modular space
    'ModularSpace',
    
    # Matching
    'MatchingFunction',
    'BestMatchSelector',
    
    # Memory
    'HistoryWindow',
    'ContextVector',
    'MemoryManager',
    
    # Derivation
    'DerivationPath',
    'TwoStepDerivation',
    'ParallelDerivation',
    
    # Generation
    'SequenceGenerator',
    'BeamSearchGenerator',
    
    # Training
    'MRNTrainer',
    'CurriculumLearning',
    'LossFunction',
]
