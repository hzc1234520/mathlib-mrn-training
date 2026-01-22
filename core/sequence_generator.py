"""
Sequence generation with history window and context vector (Section 4)

This module implements the complete sequence generation algorithm with
parallel derivation and dual-channel memory.
"""

import numpy as np
from typing import List, Optional, Tuple
from .modular_space import ModularSpace
from .embedding import SymbolEmbedding
from .derivation import TwoStepDerivation, DerivationGraph


class HistoryWindow:
    """Fixed-size history window H_t ∈ Z_m^{n×L}"""
    
    def __init__(self, window_size: int, dimension: int):
        """
        Initialize history window
        
        Args:
            window_size: Window size L
            dimension: Vector dimension n
        """
        self.window_size = window_size
        self.dimension = dimension
        # Initialize with zero vectors
        self.states = [np.zeros(dimension, dtype=int) 
                      for _ in range(window_size)]
    
    def update(self, new_state: np.ndarray):
        """
        Update window with new state (shift and insert)
        
        H_{t+1} = [x_{t+1}, H_t[0], H_t[1], ..., H_t[L-2]]
        
        Args:
            new_state: New state vector to add
        """
        # Shift: remove oldest, add newest at front
        self.states = [new_state] + self.states[:-1]
    
    def get_window(self, k: int) -> List[np.ndarray]:
        """
        Get parallel derivation window of size k
        
        W_t = [H_t[0], H_t[1], ..., H_t[k-1]]
        
        Args:
            k: Parallel window size K
            
        Returns:
            List of k most recent states
        """
        if k > self.window_size:
            k = self.window_size
        return self.states[:k]
    
    def get_state(self, index: int) -> np.ndarray:
        """Get state at specific index (0 = most recent)"""
        return self.states[index]


class ContextVector:
    """Recursive context vector with linear updates"""
    
    def __init__(self, modular_space: ModularSpace):
        """
        Initialize context vector
        
        Args:
            modular_space: ModularSpace for modular operations
        """
        self.modular_space = modular_space
        self.dimension = modular_space.dimension
        self.modulus = modular_space.modulus
        
        # Initialize context to zero
        self.vector = modular_space.zero_vector()
        
        # Fixed random matrices A, B ∈ Z_m^{n×n}
        self.A = modular_space.random_matrix(self.dimension, self.dimension)
        self.B = modular_space.random_matrix(self.dimension, self.dimension)
    
    def update(self, new_state: np.ndarray):
        """
        Update context vector
        
        c_{t+1} = (A·c_t + B·x_{t+1}) mod m
        
        Args:
            new_state: New state vector x_{t+1}
        """
        # c_{t+1} = A·c_t + B·x_{t+1}
        term1 = self.modular_space.matrix_multiply(self.A, self.vector)
        term2 = self.modular_space.matrix_multiply(self.B, new_state)
        self.vector = self.modular_space.add(term1, term2)
    
    def get(self) -> np.ndarray:
        """Get current context vector"""
        return self.vector
    
    def reset(self):
        """Reset context to zero"""
        self.vector = self.modular_space.zero_vector()


class MRNGenerator:
    """Complete MRN sequence generation system"""
    
    def __init__(self, 
                 embedding: SymbolEmbedding,
                 derivation: TwoStepDerivation,
                 history_window_size: int = 128,
                 parallel_window_size: int = 3,
                 temperature: float = 1.0):
        """
        Initialize MRN generator
        
        Args:
            embedding: SymbolEmbedding instance
            derivation: TwoStepDerivation instance
            history_window_size: History window size L
            parallel_window_size: Parallel derivation window size K
            temperature: Temperature for softmax τ
        """
        self.embedding = embedding
        self.derivation = derivation
        self.graph = DerivationGraph(derivation)
        
        self.history_window_size = history_window_size
        self.parallel_window_size = parallel_window_size
        self.temperature = temperature
        
        # Initialize memory components
        self.history = HistoryWindow(history_window_size, embedding.embedding_dim)
        self.context = ContextVector(embedding.modular_space)
        
        # Current state
        self.current_state = None
    
    def initialize(self, initial_sequence: List[str]):
        """
        Initialize with a sequence of symbols
        
        Args:
            initial_sequence: List of initial symbols [w_1, w_2, ..., w_T]
        """
        if len(initial_sequence) == 0:
            raise ValueError("Initial sequence cannot be empty")
        
        # Reset memory
        self.context.reset()
        
        # Process initial sequence
        for symbol in initial_sequence:
            x_t = self.embedding.get_discrete_vector(symbol)
            self.history.update(x_t)
            self.context.update(x_t)
            self.current_state = x_t
    
    def generate_next(self, deterministic: bool = False) -> Tuple[str, dict]:
        """
        Generate next symbol using complete MRN algorithm
        
        Algorithm steps:
        1. Construct parallel derivation window W_t
        2. Execute two-step derivation for each history state
        3. Aggregate candidate scores
        4. Compute probability distribution
        5. Sample next symbol
        6. Update state
        
        Args:
            deterministic: If True, use argmax; if False, sample
            
        Returns:
            Tuple of (next_symbol, metadata_dict)
        """
        if self.current_state is None:
            raise ValueError("Generator not initialized. Call initialize() first.")
        
        # Step 1: Get parallel window
        window = self.history.get_window(self.parallel_window_size)
        
        # Step 2-3: Derive and aggregate scores
        candidates = self.embedding.get_all_discrete_vectors()
        scores = self.graph.aggregate_candidates(
            self.current_state, window, candidates, self.embedding.vocab_size
        )
        
        # Step 4: Compute probability distribution
        probabilities = self.graph.compute_probability(scores, self.temperature)
        
        # Step 5: Sample next symbol
        next_idx = self.graph.sample(probabilities, deterministic)
        next_symbol = self.embedding.get_symbol(next_idx)
        next_state = self.embedding.get_discrete_vector_by_idx(next_idx)
        
        # Step 6: Update state
        self.history.update(next_state)
        self.context.update(next_state)
        self.current_state = next_state
        
        # Collect metadata
        metadata = {
            'scores': scores,
            'probabilities': probabilities,
            'selected_idx': next_idx,
            'context_vector': self.context.get().copy()
        }
        
        return next_symbol, metadata
    
    def generate_sequence(self, length: int, deterministic: bool = False) -> List[str]:
        """
        Generate a sequence of symbols
        
        Args:
            length: Number of symbols to generate
            deterministic: If True, use argmax selection
            
        Returns:
            List of generated symbols
        """
        sequence = []
        for _ in range(length):
            symbol, _ = self.generate_next(deterministic)
            sequence.append(symbol)
        return sequence
    
    def get_state(self) -> dict:
        """Get current generator state for debugging"""
        return {
            'current_state': self.current_state,
            'history': [s.copy() for s in self.history.states],
            'context': self.context.get().copy()
        }
