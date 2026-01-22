"""
Symbol Embedding System for MRN
Implements Section 1.1: 符号表与嵌入空间 (Symbol Table and Embedding Space)
"""

import numpy as np
from typing import List, Dict, Optional


class SymbolEmbedding:
    """
    Symbol embedding system with trainable real-valued embeddings.
    
    V = {v_1, v_2, ..., v_N} is the finite symbol vocabulary
    Each symbol v ∈ V has a trainable embedding e_v ∈ R^n
    Embedding matrix E ∈ R^{N×n}
    """
    
    def __init__(self, 
                 vocabulary: List[str],
                 embedding_dim: int,
                 modulus: int,
                 seed: Optional[int] = None):
        """
        Initialize symbol embedding system.
        
        Args:
            vocabulary: List of N symbols {v_1, v_2, ..., v_N}
            embedding_dim: Embedding dimension n (n ≥ 1)
            modulus: Modulus m for modular space (m ≥ 2)
            seed: Random seed for initialization
        """
        self.vocabulary = vocabulary
        self.N = len(vocabulary)  # Number of symbols
        self.n = embedding_dim    # Embedding dimension
        self.m = modulus          # Modulus
        
        # Symbol to index mapping
        self.symbol_to_idx = {symbol: idx for idx, symbol in enumerate(vocabulary)}
        self.idx_to_symbol = {idx: symbol for idx, symbol in enumerate(vocabulary)}
        
        # Initialize embedding matrix E ∈ R^{N×n}
        if seed is not None:
            np.random.seed(seed)
        self.E = self._initialize_embeddings()
        
    def _initialize_embeddings(self) -> np.ndarray:
        """
        Initialize embedding matrix with random values.
        Using Xavier/Glorot initialization scaled for modular space.
        
        Returns:
            E ∈ R^{N×n}: Embedding matrix
        """
        # Initialize with uniform distribution in range that makes sense for modular space
        # After rounding and mod, values should cover [0, m-1] reasonably
        scale = self.m / 2.0
        E = np.random.uniform(-scale, scale, size=(self.N, self.n))
        return E.astype(np.float32)
    
    def get_embedding(self, symbol: str) -> np.ndarray:
        """
        Get continuous embedding vector for a symbol.
        
        Args:
            symbol: Symbol v ∈ V
            
        Returns:
            e_v ∈ R^n: Embedding vector
        """
        idx = self.symbol_to_idx[symbol]
        return self.E[idx]
    
    def get_embeddings_batch(self, symbols: List[str]) -> np.ndarray:
        """
        Get embeddings for a batch of symbols.
        
        Args:
            symbols: List of symbols
            
        Returns:
            E_batch ∈ R^{batch_size×n}: Batch of embeddings
        """
        indices = [self.symbol_to_idx[s] for s in symbols]
        return self.E[indices]
    
    def discretize(self, symbol: str) -> np.ndarray:
        """
        Discretize symbol to modular space representation.
        
        x_{v,i} = round(e_{v,i}) mod m
        
        Args:
            symbol: Symbol v ∈ V
            
        Returns:
            x_v ∈ Z_m^n: Discrete modular representation
        """
        e_v = self.get_embedding(symbol)
        x_v = np.round(e_v).astype(int) % self.m
        return x_v
    
    def discretize_batch(self, symbols: List[str]) -> np.ndarray:
        """
        Discretize batch of symbols.
        
        Args:
            symbols: List of symbols
            
        Returns:
            X ∈ Z_m^{batch_size×n}: Discrete representations
        """
        E_batch = self.get_embeddings_batch(symbols)
        X = np.round(E_batch).astype(int) % self.m
        return X
    
    def get_all_discrete_vectors(self) -> Dict[str, np.ndarray]:
        """
        Get all symbol vectors in modular space.
        
        Returns:
            X = {x_v | v ∈ V} ⊂ Z_m^n: Dictionary mapping symbols to vectors
        """
        X = {}
        for symbol in self.vocabulary:
            X[symbol] = self.discretize(symbol)
        return X
    
    def update_embedding(self, symbol: str, gradient: np.ndarray, learning_rate: float):
        """
        Update embedding using gradient descent.
        
        e_v ← e_v - learning_rate * gradient
        
        Args:
            symbol: Symbol to update
            gradient: Gradient ∂L/∂e_v
            learning_rate: Learning rate η
        """
        idx = self.symbol_to_idx[symbol]
        self.E[idx] -= learning_rate * gradient
    
    def update_embeddings_batch(self, symbols: List[str], gradients: np.ndarray, learning_rate: float):
        """
        Update batch of embeddings.
        
        Args:
            symbols: List of symbols to update
            gradients: Gradients ∂L/∂E_batch
            learning_rate: Learning rate η
        """
        indices = [self.symbol_to_idx[s] for s in symbols]
        self.E[indices] -= learning_rate * gradients
    
    def get_parameter_count(self) -> int:
        """
        Get total number of trainable parameters.
        
        Returns:
            N × n: Total parameters in embedding matrix
        """
        return self.N * self.n
    
    def save_embeddings(self, filepath: str):
        """Save embedding matrix to file."""
        np.save(filepath, self.E)
    
    def load_embeddings(self, filepath: str):
        """Load embedding matrix from file."""
        self.E = np.load(filepath).astype(np.float32)
        assert self.E.shape == (self.N, self.n), "Loaded embeddings shape mismatch"
    
    def __repr__(self) -> str:
        return (f"SymbolEmbedding(vocab_size={self.N}, "
                f"embedding_dim={self.n}, modulus={self.m})")
