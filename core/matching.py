"""
Exact Matching Functions for MRN
Implements Section 2: 精确匹配函数 (Exact Matching Functions)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


class MatchingFunction:
    """
    Exact matching functions for modular space vectors.
    
    For q, x ∈ Z_m^n, match(q, x) = Σ_{i=1}^n 1(q_i = x_i)
    where 1(·) is the indicator function
    """
    
    def __init__(self, modulus: int, dimension: int):
        """
        Initialize matching function.
        
        Args:
            modulus: Modulus m for modular space
            dimension: Dimension n of vectors
        """
        self.m = modulus
        self.n = dimension
    
    @staticmethod
    def match(q: np.ndarray, x: np.ndarray) -> int:
        """
        Compute exact matching score between two vectors.
        
        match(q, x) = Σ_{i=1}^n 1(q_i = x_i)
        
        Properties:
        - Symmetry: match(q, x) = match(x, q)
        - Bounded: 0 ≤ match(q, x) ≤ n
        - Decomposable: Can be computed per dimension
        
        Args:
            q: Query vector ∈ Z_m^n
            x: Target vector ∈ Z_m^n
            
        Returns:
            Integer matching score [0, n]
        """
        return int(np.sum(q == x))
    
    @staticmethod
    def match_batch(q: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Compute matching scores between query and batch of vectors.
        
        Args:
            q: Query vector ∈ Z_m^n (shape: (n,))
            X: Batch of vectors ∈ Z_m^{batch_size×n} (shape: (batch_size, n))
            
        Returns:
            Array of matching scores (shape: (batch_size,))
        """
        # Broadcasting: q[None, :] creates (1, n), X is (batch_size, n)
        # Comparison gives (batch_size, n), sum over dimension gives (batch_size,)
        return np.sum(q[None, :] == X, axis=1).astype(int)
    
    def best_match(self, q: np.ndarray, X: np.ndarray, 
                   return_score: bool = False) -> Tuple[int, Optional[int]]:
        """
        Find best matching vector from a set.
        
        best(q) = argmax_{x∈X} match(q, x)
        
        When multiple vectors have same highest score, randomly selects one
        to introduce diversity.
        
        Args:
            q: Query vector ∈ Z_m^n
            X: Set of vectors ∈ Z_m^{N×n}
            return_score: Whether to return the matching score
            
        Returns:
            Index of best matching vector (and optionally its score)
        """
        scores = self.match_batch(q, X)
        max_score = np.max(scores)
        
        # Find all indices with maximum score
        best_indices = np.where(scores == max_score)[0]
        
        # Random selection among ties for diversity
        selected_idx = np.random.choice(best_indices)
        
        if return_score:
            return int(selected_idx), int(max_score)
        return int(selected_idx), None
    
    def top_k_matches(self, q: np.ndarray, X: np.ndarray, k: int) -> List[Tuple[int, int]]:
        """
        Get top-k best matching vectors.
        
        top_M(q) = {(x_1, m_1), (x_2, m_2), ..., (x_M, m_M)}
        where m_1 ≥ m_2 ≥ ... ≥ m_M
        
        Args:
            q: Query vector ∈ Z_m^n
            X: Set of vectors ∈ Z_m^{N×n}
            k: Number of top matches to return (M)
            
        Returns:
            List of (index, score) tuples sorted by score descending
        """
        scores = self.match_batch(q, X)
        
        # Get indices sorted by score (descending)
        sorted_indices = np.argsort(scores)[::-1]
        
        # Return top k
        k = min(k, len(sorted_indices))
        results = [(int(sorted_indices[i]), int(scores[sorted_indices[i]])) 
                   for i in range(k)]
        
        return results
    
    @staticmethod
    def smooth_match(q: np.ndarray, x: np.ndarray, gamma: float = 10.0) -> float:
        """
        Smooth approximation of matching function for training.
        
        match_σ(q, x) = Σ_{i=1}^n σ(γ · (q_i - x_i))
        
        where σ(·) is sigmoid function. As γ → ∞, converges to discrete match.
        This allows gradient flow in non-matching regions.
        
        Args:
            q: Query vector ∈ Z_m^n
            x: Target vector ∈ Z_m^n
            gamma: Temperature parameter (higher = sharper)
            
        Returns:
            Smooth matching score (float)
        """
        diff = q - x
        # Sigmoid centered at 0: when diff=0, sigmoid→0.5, penalize differences
        # We want high score when diff=0, so use sigmoid(-gamma * |diff|)
        sigmoid_values = 1.0 / (1.0 + np.exp(gamma * np.abs(diff)))
        return float(np.sum(sigmoid_values))
    
    def hamming_distance(self, q: np.ndarray, x: np.ndarray) -> int:
        """
        Compute Hamming distance (complement of matching score).
        
        distance(q, x) = n - match(q, x)
        
        Args:
            q: Query vector ∈ Z_m^n
            x: Target vector ∈ Z_m^n
            
        Returns:
            Hamming distance [0, n]
        """
        return self.n - self.match(q, x)
    
    def verify_properties(self, q: np.ndarray, x: np.ndarray, y: np.ndarray) -> Dict[str, bool]:
        """
        Verify mathematical properties of matching function.
        
        Returns:
            Dictionary of property verification results
        """
        m_qx = self.match(q, x)
        m_xq = self.match(x, q)
        m_qy = self.match(q, y)
        m_xy = self.match(x, y)
        
        properties = {
            'symmetry': m_qx == m_xq,
            'bounded_lower': 0 <= m_qx,
            'bounded_upper': m_qx <= self.n,
            'triangle_inequality_variant': abs(m_qx - m_qy) <= self.n - abs(m_qx - m_qy)
        }
        
        return properties


class BestMatchSelector:
    """
    Utility class for selecting best matches from symbol vocabulary.
    """
    
    def __init__(self, symbol_vectors: Dict[str, np.ndarray]):
        """
        Initialize selector with symbol vocabulary.
        
        Args:
            symbol_vectors: X = {x_v | v ∈ V} ⊂ Z_m^n
        """
        self.symbols = list(symbol_vectors.keys())
        self.vectors = np.array([symbol_vectors[s] for s in self.symbols])
        self.N = len(self.symbols)
        self.n = self.vectors.shape[1] if self.N > 0 else 0
        self.m = None  # Modulus not strictly needed here
        
        self.matcher = MatchingFunction(modulus=self.m or 0, dimension=self.n)
    
    def select_best(self, q: np.ndarray) -> Tuple[str, np.ndarray, int]:
        """
        Select best matching symbol for query vector.
        
        Args:
            q: Query vector ∈ Z_m^n
            
        Returns:
            (symbol, vector, match_score) tuple
        """
        idx, score = self.matcher.best_match(q, self.vectors, return_score=True)
        return self.symbols[idx], self.vectors[idx], score
    
    def select_top_k(self, q: np.ndarray, k: int) -> List[Tuple[str, np.ndarray, int]]:
        """
        Select top-k best matching symbols.
        
        Args:
            q: Query vector ∈ Z_m^n
            k: Number of top matches
            
        Returns:
            List of (symbol, vector, score) tuples
        """
        top_matches = self.matcher.top_k_matches(q, self.vectors, k)
        results = []
        for idx, score in top_matches:
            results.append((self.symbols[idx], self.vectors[idx], score))
        return results
