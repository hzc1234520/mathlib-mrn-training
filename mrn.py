"""
Main MRN (Modular Reasoning Network) Class

This module provides a unified interface for the complete MRN system,
integrating all components from initialization to training and inference.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple

from core import (
    ModularSpace,
    ExactMatcher,
    SymbolEmbedding,
    TwoStepDerivation,
    MRNGenerator,
    MRNTrainer,
    CurriculumLearning
)


class MRN:
    """
    Complete Modular Reasoning Network
    
    This class provides a high-level interface for the MRN system as described
    in the mathematical theory document. It integrates:
    - Symbol embedding in modular space
    - Exact matching functions
    - Two-step derivation
    - Dual-channel memory (history window + context vector)
    - Training with multiple loss functions
    
    Example usage:
        >>> vocab = ['a', 'b', 'c', 'd', 'e']
        >>> mrn = MRN(vocab, modulus=16, embedding_dim=32)
        >>> mrn.initialize(['a', 'b', 'c'])
        >>> next_symbol = mrn.generate_next()
        >>> sequence = mrn.generate_sequence(10)
    """
    
    def __init__(self,
                 vocab: List[str],
                 modulus: int = 16,
                 embedding_dim: int = 32,
                 history_window_size: int = 128,
                 parallel_window_size: int = 3,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 temperature: float = 1.0,
                 learning_rate: float = 0.01,
                 lambda_contrastive: float = 1.0,
                 lambda_reg: float = 1e-5):
        """
        Initialize MRN system
        
        Args:
            vocab: List of symbols V = {v_1, v_2, ..., v_N}
            modulus: Modulus m for Z_m space (m ≥ 2)
            embedding_dim: Embedding dimension n (n ≥ 1)
            history_window_size: History window size L
            parallel_window_size: Parallel derivation window size K (1 ≤ K ≤ L)
            alpha: Weight for first-step matching in derivation score
            beta: Weight for second-step matching in derivation score
            temperature: Temperature τ for softmax
            learning_rate: Learning rate for training
            lambda_contrastive: Weight for contrastive loss
            lambda_reg: Weight for L2 regularization
        """
        # Validate parameters
        if modulus < 2:
            raise ValueError("Modulus must be >= 2")
        if embedding_dim < 1:
            raise ValueError("Embedding dimension must be >= 1")
        if parallel_window_size > history_window_size:
            raise ValueError("Parallel window size K cannot exceed history window size L")
        
        self.vocab = vocab
        self.modulus = modulus
        self.embedding_dim = embedding_dim
        self.history_window_size = history_window_size
        self.parallel_window_size = parallel_window_size
        self.temperature = temperature
        
        # Initialize core components
        self.modular_space = ModularSpace(modulus, embedding_dim)
        self.matcher = ExactMatcher(modulus, embedding_dim)
        self.embedding = SymbolEmbedding(vocab, embedding_dim, self.modular_space)
        self.derivation = TwoStepDerivation(self.matcher, alpha, beta)
        self.generator = MRNGenerator(
            self.embedding,
            self.derivation,
            history_window_size,
            parallel_window_size,
            temperature
        )
        self.trainer = MRNTrainer(
            self.generator,
            learning_rate,
            lambda_contrastive,
            lambda_reg
        )
        
        # State
        self._initialized = False
    
    def initialize(self, initial_sequence: List[str]):
        """
        Initialize system with a sequence
        
        Args:
            initial_sequence: List of initial symbols [w_1, w_2, ..., w_T]
        """
        self.generator.initialize(initial_sequence)
        self._initialized = True
    
    def generate_next(self, deterministic: bool = False) -> Tuple[str, Dict]:
        """
        Generate next symbol
        
        Args:
            deterministic: If True, use argmax; if False, sample
            
        Returns:
            Tuple of (next_symbol, metadata)
        """
        if not self._initialized:
            raise ValueError("System not initialized. Call initialize() first.")
        return self.generator.generate_next(deterministic)
    
    def generate_sequence(self, length: int, deterministic: bool = False) -> List[str]:
        """
        Generate a sequence of symbols
        
        Args:
            length: Number of symbols to generate
            deterministic: If True, use argmax selection
            
        Returns:
            List of generated symbols
        """
        if not self._initialized:
            raise ValueError("System not initialized. Call initialize() first.")
        return self.generator.generate_sequence(length, deterministic)
    
    def train_step(self, 
                   initial_sequence: List[str],
                   target_sequence: List[str],
                   teacher_forcing: bool = True) -> Dict[str, float]:
        """
        Execute one training step
        
        Args:
            initial_sequence: Initial context
            target_sequence: Target sequence to predict
            teacher_forcing: Use teacher forcing
            
        Returns:
            Dictionary of loss components
        """
        return self.trainer.train_step(initial_sequence, target_sequence, teacher_forcing)
    
    def train(self,
              training_data: List[Tuple[List[str], List[str]]],
              num_epochs: int,
              teacher_forcing: bool = True,
              curriculum_stage: Optional[str] = None,
              verbose: bool = True) -> List[Dict[str, float]]:
        """
        Train the MRN system
        
        Args:
            training_data: List of (initial_sequence, target_sequence) pairs
            num_epochs: Number of training epochs
            teacher_forcing: Use teacher forcing strategy
            curriculum_stage: One of 'early', 'middle', 'late', or None
            verbose: Print training progress
            
        Returns:
            Training history
        """
        # Apply curriculum if specified
        if curriculum_stage is not None:
            CurriculumLearning.apply_schedule(self.generator, curriculum_stage)
        
        return self.trainer.train(training_data, num_epochs, teacher_forcing, verbose)
    
    def get_embedding_matrix(self) -> np.ndarray:
        """Get current embedding matrix E"""
        return self.embedding.embeddings.copy()
    
    def get_discrete_vectors(self) -> List[np.ndarray]:
        """Get all discrete vectors X = {x_v | v ∈ V}"""
        return self.embedding.get_all_discrete_vectors()
    
    def get_state(self) -> Dict:
        """Get complete system state for inspection"""
        return {
            'initialized': self._initialized,
            'generator_state': self.generator.get_state() if self._initialized else None,
            'vocab_size': len(self.vocab),
            'modulus': self.modulus,
            'embedding_dim': self.embedding_dim,
            'history_window_size': self.history_window_size,
            'parallel_window_size': self.parallel_window_size,
        }
    
    def save_embeddings(self, filepath: str):
        """Save embedding matrix to file"""
        np.save(filepath, self.embedding.embeddings)
    
    def load_embeddings(self, filepath: str):
        """Load embedding matrix from file"""
        embeddings = np.load(filepath)
        if embeddings.shape != self.embedding.embeddings.shape:
            raise ValueError(f"Shape mismatch: expected {self.embedding.embeddings.shape}, "
                           f"got {embeddings.shape}")
        self.embedding.embeddings = embeddings
    
    def __repr__(self) -> str:
        return (f"MRN(vocab_size={len(self.vocab)}, "
                f"modulus={self.modulus}, "
                f"embedding_dim={self.embedding_dim}, "
                f"L={self.history_window_size}, "
                f"K={self.parallel_window_size})")
