"""
Exact matching functions for MRN system (Section 2)

This module implements the exact matching functions based on discrete comparison
in modular space Z_m^n.
"""

import numpy as np
from typing import List, Tuple


class ExactMatcher:
    """Exact matching functions in modular space"""
    
    def __init__(self, modulus: int, dimension: int):
        """
        Initialize exact matcher
        
        Args:
            modulus: Modulus m for Z_m space
            dimension: Dimension n of vectors
        """
        self.modulus = modulus
        self.dimension = dimension
    
    def match(self, q: np.ndarray, x: np.ndarray) -> int:
        """
        Compute exact match score between two vectors
        
        match(q, x) = sum_{i=1}^{n} 1(q_i = x_i)
        
        Args:
            q: Query vector in Z_m^n
            x: Candidate vector in Z_m^n
            
        Returns:
            Number of matching dimensions (0 to n)
        """
        return int(np.sum(q == x))
    
    def match_smooth(self, q: np.ndarray, x: np.ndarray, gamma: float = 10.0) -> float:
        """
        Smooth approximation of match function for training
        
        match_σ(q, x) = sum_{i=1}^{n} σ(γ * (q_i - x_i))
        
        Args:
            q: Query vector
            x: Candidate vector
            gamma: Temperature parameter controlling sharpness
            
        Returns:
            Smooth matching score
        """
        diff = q - x
        # Use sigmoid: σ(x) = 1 / (1 + exp(-x))
        # When diff is 0, sigmoid gives 0.5, when diff is large, gives ~0
        # We want 1 when equal, 0 when different, so use σ(-γ*|diff|)
        sigmoid_vals = 1.0 / (1.0 + np.exp(gamma * np.abs(diff)))
        return float(np.sum(sigmoid_vals))
    
    def best(self, q: np.ndarray, candidates: List[np.ndarray], 
             random_on_tie: bool = True) -> Tuple[np.ndarray, int]:
        """
        Find best matching candidate
        
        best(q) = argmax_{x ∈ X} match(q, x)
        
        Args:
            q: Query vector in Z_m^n
            candidates: List of candidate vectors X
            random_on_tie: If True, randomly select among tied candidates
            
        Returns:
            Tuple of (best_vector, match_score)
        """
        if not candidates:
            raise ValueError("Candidates list cannot be empty")
        
        # Compute all match scores
        scores = [self.match(q, x) for x in candidates]
        max_score = max(scores)
        
        # Find all candidates with max score
        best_indices = [i for i, s in enumerate(scores) if s == max_score]
        
        # Select one (randomly if multiple)
        if random_on_tie and len(best_indices) > 1:
            selected_idx = np.random.choice(best_indices)
        else:
            selected_idx = best_indices[0]
        
        return candidates[selected_idx], max_score
    
    def top_m(self, q: np.ndarray, candidates: List[np.ndarray], 
              m: int) -> List[Tuple[np.ndarray, int]]:
        """
        Return top M best matching candidates with scores
        
        top_M(q) = {(x_1, m_1), (x_2, m_2), ..., (x_M, m_M) | m_1 ≥ m_2 ≥ ... ≥ m_M}
        
        Args:
            q: Query vector in Z_m^n
            candidates: List of candidate vectors X
            m: Number of top candidates to return
            
        Returns:
            List of (vector, score) tuples sorted by score descending
        """
        if m > len(candidates):
            m = len(candidates)
        
        # Compute all match scores
        scored_candidates = [(x, self.match(q, x)) for x in candidates]
        
        # Sort by score descending
        scored_candidates.sort(key=lambda item: item[1], reverse=True)
        
        return scored_candidates[:m]


class SmoothMatchingLoss:
    """Smooth matching approximation for gradient-based training"""
    
    @staticmethod
    def smooth_match_loss(embeddings: np.ndarray, targets: np.ndarray, 
                          gamma: float = 10.0) -> float:
        """
        Compute smooth matching loss for training
        
        Args:
            embeddings: Embedding vectors (before discretization)
            targets: Target discrete vectors
            gamma: Sharpness parameter
            
        Returns:
            Smooth matching loss (lower is better match)
        """
        # Smooth approximation of matching
        diff = embeddings - targets
        sigmoid_vals = 1.0 / (1.0 + np.exp(gamma * np.abs(diff)))
        # Return negative sum (so minimizing gives better match)
        return -np.sum(sigmoid_vals)
