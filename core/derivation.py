"""
Two-step derivation process for MRN system (Section 3)

This module implements the two-step derivation with intermediate matching
and path scoring.
"""

import numpy as np
from typing import Tuple, List
from .matching import ExactMatcher


class TwoStepDerivation:
    """Two-step derivation with intermediate vector generation"""
    
    def __init__(self, matcher: ExactMatcher, alpha: float = 0.5, beta: float = 0.5):
        """
        Initialize two-step derivation
        
        Args:
            matcher: ExactMatcher instance for matching operations
            alpha: Weight for first step matching in score
            beta: Weight for second step matching in score
        """
        self.matcher = matcher
        self.alpha = alpha  # Trainable parameter
        self.beta = beta    # Trainable parameter
    
    def derive(self, x_t: np.ndarray, h: np.ndarray, 
               candidates: List[np.ndarray]) -> Tuple[np.ndarray, float, dict]:
        """
        Execute complete two-step derivation
        
        Step 1: z_1 = (x_t + h) mod m, x_v = best(z_1)
        Step 2: z_2 = (x_v + h) mod m, x_w = best(z_2)
        Score: α·m_1 + β·m_2
        
        Args:
            x_t: Current state vector in Z_m^n
            h: History state vector in Z_m^n
            candidates: List of candidate symbol vectors
            
        Returns:
            Tuple of (output_vector, score, metadata_dict)
            metadata contains: intermediate_vector, z1, z2, m1, m2
        """
        modulus = self.matcher.modulus
        
        # Step 1: Intermediate vector generation and matching
        z_1 = (x_t + h) % modulus
        x_v, m_1 = self.matcher.best(z_1, candidates)
        
        # Step 2: Final vector generation and matching
        z_2 = (x_v + h) % modulus
        x_w, m_2 = self.matcher.best(z_2, candidates)
        
        # Compute path score
        score = self.alpha * m_1 + self.beta * m_2
        
        # Collect metadata for debugging/analysis
        metadata = {
            'intermediate_vector': x_v,
            'z1': z_1,
            'z2': z_2,
            'm1': m_1,
            'm2': m_2,
            'history_state': h
        }
        
        return x_w, score, metadata
    
    def derive_batch(self, x_t: np.ndarray, histories: List[np.ndarray],
                     candidates: List[np.ndarray]) -> List[Tuple[np.ndarray, float, dict]]:
        """
        Execute two-step derivation for multiple history states in parallel
        
        Args:
            x_t: Current state vector
            histories: List of history state vectors
            candidates: List of candidate symbol vectors
            
        Returns:
            List of (output_vector, score, metadata) for each history state
        """
        results = []
        for h in histories:
            result = self.derive(x_t, h, candidates)
            results.append(result)
        return results
    
    def set_weights(self, alpha: float, beta: float):
        """Update derivation weights (for training)"""
        self.alpha = alpha
        self.beta = beta


class DerivationGraph:
    """
    Manage derivation paths and candidate aggregation
    """
    
    def __init__(self, derivation: TwoStepDerivation):
        """
        Initialize derivation graph
        
        Args:
            derivation: TwoStepDerivation instance
        """
        self.derivation = derivation
    
    def aggregate_candidates(self, x_t: np.ndarray, 
                            window: List[np.ndarray],
                            candidates: List[np.ndarray],
                            num_symbols: int) -> np.ndarray:
        """
        Aggregate scores from multiple derivation paths
        
        For each symbol x_j, compute:
        S_j = (1/K) * sum_{(h,x_w,s)∈C} s·1(x_w = x_j)
        
        Args:
            x_t: Current state vector
            window: Parallel derivation window W_t (K history states)
            candidates: List of all candidate symbol vectors
            num_symbols: Total number of symbols N
            
        Returns:
            Score array S of length N for all symbols
        """
        K = len(window)
        if K == 0:
            raise ValueError("Window cannot be empty")
        
        # Initialize scores for all symbols
        scores = np.zeros(num_symbols, dtype=float)
        
        # Derive for each history state
        results = self.derivation.derive_batch(x_t, window, candidates)
        
        # Aggregate scores
        for x_w, s, metadata in results:
            # Find which symbol x_w corresponds to
            for j, candidate in enumerate(candidates):
                if np.array_equal(x_w, candidate):
                    scores[j] += s
                    break
        
        # Average by K
        scores = scores / K
        
        return scores
    
    def compute_probability(self, scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Convert scores to probability distribution using softmax
        
        P(x_j) = exp(S_j / τ) / sum_{j'} exp(S_{j'} / τ)
        
        Args:
            scores: Score array S
            temperature: Temperature parameter τ controlling sharpness
            
        Returns:
            Probability distribution over symbols
        """
        # Apply temperature scaling
        scaled_scores = scores / temperature
        
        # Numerical stability: subtract max
        scaled_scores = scaled_scores - np.max(scaled_scores)
        
        # Softmax
        exp_scores = np.exp(scaled_scores)
        probabilities = exp_scores / np.sum(exp_scores)
        
        return probabilities
    
    def sample(self, probabilities: np.ndarray, deterministic: bool = False) -> int:
        """
        Sample symbol index from probability distribution
        
        Args:
            probabilities: Probability distribution
            deterministic: If True, return argmax; if False, sample
            
        Returns:
            Selected symbol index
        """
        if deterministic:
            return int(np.argmax(probabilities))
        else:
            return int(np.random.choice(len(probabilities), p=probabilities))
