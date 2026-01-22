"""
Symbol embedding system for MRN (Section 1.1)

This module manages the trainable embedding matrix and discretization
to modular space.
"""

import numpy as np
from typing import List, Dict, Tuple
from .modular_space import ModularSpace


class SymbolEmbedding:
    """Trainable symbol embedding matrix E ∈ R^{N×n}"""
    
    def __init__(self, vocab: List[str], embedding_dim: int, 
                 modular_space: ModularSpace):
        """
        Initialize symbol embedding system
        
        Args:
            vocab: List of N symbols [v_1, v_2, ..., v_N]
            embedding_dim: Embedding dimension n
            modular_space: ModularSpace instance for discretization
        """
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding_dim = embedding_dim
        self.modular_space = modular_space
        
        # Build symbol to index mapping
        self.symbol_to_idx = {symbol: idx for idx, symbol in enumerate(vocab)}
        self.idx_to_symbol = {idx: symbol for idx, symbol in enumerate(vocab)}
        
        # Initialize embedding matrix E
        self.embeddings = self._initialize_embeddings()
    
    def _initialize_embeddings(self) -> np.ndarray:
        """
        Initialize embedding matrix with random values
        
        Returns:
            Embedding matrix E ∈ R^{N×n}
        """
        # Initialize with values in range suitable for modulus
        m = self.modular_space.modulus
        # Initialize near middle of modulus range for stability
        embeddings = np.random.randn(self.vocab_size, self.embedding_dim) * (m / 6.0)
        embeddings += m / 2.0  # Center around m/2
        return embeddings
    
    def get_embedding(self, symbol: str) -> np.ndarray:
        """Get continuous embedding vector for symbol"""
        idx = self.symbol_to_idx[symbol]
        return self.embeddings[idx]
    
    def get_embedding_by_idx(self, idx: int) -> np.ndarray:
        """Get continuous embedding vector by index"""
        return self.embeddings[idx]
    
    def get_discrete_vector(self, symbol: str) -> np.ndarray:
        """
        Get discrete modular space vector for symbol
        
        x_v = discretize(e_v) = round(e_v) mod m
        
        Args:
            symbol: Symbol string
            
        Returns:
            Discrete vector in Z_m^n
        """
        embedding = self.get_embedding(symbol)
        return self.modular_space.discretize(embedding)
    
    def get_discrete_vector_by_idx(self, idx: int) -> np.ndarray:
        """Get discrete vector by index"""
        embedding = self.get_embedding_by_idx(idx)
        return self.modular_space.discretize(embedding)
    
    def get_all_discrete_vectors(self) -> List[np.ndarray]:
        """
        Get all discrete vectors for the vocabulary
        
        Returns:
            List of discrete vectors X = {x_v | v ∈ V}
        """
        return [self.get_discrete_vector_by_idx(i) 
                for i in range(self.vocab_size)]
    
    def update_embeddings(self, gradients: np.ndarray, learning_rate: float):
        """
        Update embeddings using gradients (simple SGD)
        
        Args:
            gradients: Gradient matrix same shape as embeddings
            learning_rate: Learning rate
        """
        self.embeddings -= learning_rate * gradients
    
    def get_symbol(self, idx: int) -> str:
        """Get symbol string by index"""
        return self.idx_to_symbol[idx]
    
    def get_index(self, symbol: str) -> int:
        """Get index by symbol string"""
        return self.symbol_to_idx[symbol]
    
    def discretize_with_ste(self, embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discretize with Straight-Through Estimator support
        
        Returns both discrete vector and continuous embedding for gradient flow
        
        Args:
            embedding: Continuous embedding vector
            
        Returns:
            Tuple of (discrete_vector, embedding_for_gradient)
        """
        discrete = self.modular_space.discretize(embedding)
        # For gradient computation, keep continuous version
        return discrete, embedding


class EmbeddingGradient:
    """Compute gradients for embedding parameters"""
    
    @staticmethod
    def straight_through_gradient(output_grad: np.ndarray) -> np.ndarray:
        """
        Straight-through estimator: pass gradients through discretization
        
        ∂x/∂round(x) ≈ 1, ∂x/∂(x mod m) ≈ 1
        
        Args:
            output_grad: Gradient from downstream
            
        Returns:
            Gradient to pass to embedding
        """
        # Simply pass through the gradient
        return output_grad
    
    @staticmethod
    def compute_l2_regularization(embeddings: np.ndarray, 
                                  lambda_reg: float) -> Tuple[float, np.ndarray]:
        """
        Compute L2 regularization loss and gradient
        
        L_reg = ||E||_F^2 = sum_{v ∈ V} ||e_v||_2^2
        
        Args:
            embeddings: Embedding matrix E
            lambda_reg: Regularization strength λ_reg
            
        Returns:
            Tuple of (loss_value, gradient)
        """
        loss = np.sum(embeddings ** 2)
        gradient = 2.0 * embeddings
        return lambda_reg * loss, lambda_reg * gradient
