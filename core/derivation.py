"""
Two-Step Derivation and Inference Graph for MRN
Implements Section 3: 推导图结构 (Derivation Graph Structure)
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from .matching import BestMatchSelector, MatchingFunction
from .modular_space import ModularSpace


class DerivationPath:
    """
    Represents a single two-step derivation path.
    
    Path: (x_t, h) → x_v → x_w
    with scores m_1 (first step) and m_2 (second step)
    """
    
    def __init__(self, 
                 history_state: np.ndarray,
                 intermediate_symbol: str,
                 intermediate_vector: np.ndarray,
                 output_symbol: str,
                 output_vector: np.ndarray,
                 m1_score: int,
                 m2_score: int):
        """
        Initialize derivation path.
        
        Args:
            history_state: Historical state h ∈ Z_m^n
            intermediate_symbol: Intermediate symbol v
            intermediate_vector: x_v ∈ Z_m^n
            output_symbol: Output symbol w
            output_vector: x_w ∈ Z_m^n
            m1_score: First step matching score m_1
            m2_score: Second step matching score m_2
        """
        self.h = history_state
        self.v = intermediate_symbol
        self.x_v = intermediate_vector
        self.w = output_symbol
        self.x_w = output_vector
        self.m1 = m1_score
        self.m2 = m2_score
    
    def compute_score(self, alpha: float, beta: float) -> float:
        """
        Compute weighted path score.
        
        score(h → x_w) = α·m_1 + β·m_2
        
        Args:
            alpha: Weight for first step matching
            beta: Weight for second step matching
            
        Returns:
            Weighted score
        """
        return alpha * self.m1 + beta * self.m2
    
    def __repr__(self) -> str:
        return f"DerivationPath(h → {self.v} → {self.w}, m1={self.m1}, m2={self.m2})"


class TwoStepDerivation:
    """
    Two-step derivation mechanism with intermediate matching.
    
    Step 1: z_1 = (x_t + h) mod m  →  x_v = best(z_1)
    Step 2: z_2 = (x_v + h) mod m  →  x_w = best(z_2)
    """
    
    def __init__(self,
                 modular_space: ModularSpace,
                 symbol_selector: BestMatchSelector,
                 alpha: float = 0.5,
                 beta: float = 0.5):
        """
        Initialize two-step derivation.
        
        Args:
            modular_space: Modular space operations
            symbol_selector: Best match selector with vocabulary
            alpha: Weight for first matching step (α > 0)
            beta: Weight for second matching step (β > 0)
        """
        self.mod_space = modular_space
        self.selector = symbol_selector
        self.matcher = MatchingFunction(modular_space.modulus, modular_space.dimension)
        
        # Trainable weights for path scoring
        self.alpha = alpha
        self.beta = beta
        
        assert alpha > 0 and beta > 0, "Weights must be positive"
    
    def derive(self, 
               x_t: np.ndarray, 
               h: np.ndarray) -> DerivationPath:
        """
        Execute complete two-step derivation.
        
        Args:
            x_t: Current state ∈ Z_m^n
            h: History state ∈ Z_m^n
            
        Returns:
            DerivationPath with complete derivation information
        """
        # Step 1: Intermediate vector generation and matching
        # z_1 = (x_t + h) mod m
        z_1 = self.mod_space.add(x_t, h)
        
        # x_v = best(z_1)
        v_symbol, x_v, m1 = self.selector.select_best(z_1)
        
        # Step 2: Final vector generation and matching
        # z_2 = (x_v + h) mod m
        z_2 = self.mod_space.add(x_v, h)
        
        # x_w = best(z_2)
        w_symbol, x_w, m2 = self.selector.select_best(z_2)
        
        # Create derivation path
        path = DerivationPath(
            history_state=h,
            intermediate_symbol=v_symbol,
            intermediate_vector=x_v,
            output_symbol=w_symbol,
            output_vector=x_w,
            m1_score=m1,
            m2_score=m2
        )
        
        return path
    
    def derive_with_top_k_first_step(self,
                                      x_t: np.ndarray,
                                      h: np.ndarray,
                                      k: int = 3) -> List[DerivationPath]:
        """
        Derive with top-k candidates in first step for diversity.
        
        Args:
            x_t: Current state ∈ Z_m^n
            h: History state ∈ Z_m^n
            k: Number of top candidates to consider in first step
            
        Returns:
            List of k derivation paths (one per first-step candidate)
        """
        # Step 1: Get top-k intermediate matches
        z_1 = self.mod_space.add(x_t, h)
        top_intermediates = self.selector.select_top_k(z_1, k)
        
        paths = []
        for v_symbol, x_v, m1 in top_intermediates:
            # Step 2: Final matching for each intermediate
            z_2 = self.mod_space.add(x_v, h)
            w_symbol, x_w, m2 = self.selector.select_best(z_2)
            
            path = DerivationPath(
                history_state=h,
                intermediate_symbol=v_symbol,
                intermediate_vector=x_v,
                output_symbol=w_symbol,
                output_vector=x_w,
                m1_score=m1,
                m2_score=m2
            )
            paths.append(path)
        
        return paths
    
    def set_weights(self, alpha: float, beta: float):
        """
        Update derivation weights.
        
        Args:
            alpha: Weight for first step
            beta: Weight for second step
        """
        assert alpha > 0 and beta > 0, "Weights must be positive"
        self.alpha = alpha
        self.beta = beta
    
    def get_weights(self) -> Tuple[float, float]:
        """Get current weights."""
        return self.alpha, self.beta


class ParallelDerivation:
    """
    Parallel derivation over K history states.
    
    For each h ∈ W_t = [x_t, x_{t-1}, ..., x_{t-K+1}],
    performs two-step derivation and aggregates results.
    """
    
    def __init__(self, two_step_derivation: TwoStepDerivation):
        """
        Initialize parallel derivation.
        
        Args:
            two_step_derivation: Two-step derivation engine
        """
        self.derivation = two_step_derivation
    
    def derive_parallel(self,
                       x_t: np.ndarray,
                       W_t: np.ndarray) -> List[DerivationPath]:
        """
        Execute parallel derivation over window.
        
        Args:
            x_t: Current state ∈ Z_m^n
            W_t: Parallel window ∈ Z_m^{K×n}
            
        Returns:
            List of K derivation paths (one per history state)
        """
        K = W_t.shape[0]
        paths = []
        
        for i in range(K):
            h = W_t[i]
            path = self.derivation.derive(x_t, h)
            paths.append(path)
        
        return paths
    
    def aggregate_scores(self, 
                        paths: List[DerivationPath],
                        vocabulary: List[str]) -> Dict[str, float]:
        """
        Aggregate derivation scores across all paths.
        
        S_j = (1/K) Σ_{(h,x_w,s)∈C} s·1(x_w = x_j)
        
        For each symbol in vocabulary, average the scores of all paths
        that lead to that symbol.
        
        Args:
            paths: List of derivation paths from parallel derivation
            vocabulary: List of all symbols
            
        Returns:
            Dictionary mapping symbols to aggregated scores
        """
        K = len(paths)
        if K == 0:
            return {symbol: 0.0 for symbol in vocabulary}
        
        # Initialize scores
        scores = {symbol: 0.0 for symbol in vocabulary}
        counts = {symbol: 0 for symbol in vocabulary}
        
        # Aggregate scores
        alpha, beta = self.derivation.get_weights()
        for path in paths:
            score = path.compute_score(alpha, beta)
            scores[path.w] += score
            counts[path.w] += 1
        
        # Average scores (sum of scores where that symbol appeared)
        # Note: The formula uses 1/K * sum, so we keep the sum
        # and will normalize during softmax
        for symbol in vocabulary:
            scores[symbol] = scores[symbol] / K
        
        return scores
    
    def get_score_vector(self,
                        paths: List[DerivationPath],
                        symbol_to_idx: Dict[str, int],
                        vocab_size: int) -> np.ndarray:
        """
        Get aggregated scores as a vector for softmax.
        
        Args:
            paths: Derivation paths
            symbol_to_idx: Symbol to index mapping
            vocab_size: Size of vocabulary
            
        Returns:
            Score vector S ∈ R^N
        """
        vocabulary = list(symbol_to_idx.keys())
        scores_dict = self.aggregate_scores(paths, vocabulary)
        
        # Convert to vector
        S = np.zeros(vocab_size)
        for symbol, score in scores_dict.items():
            idx = symbol_to_idx[symbol]
            S[idx] = score
        
        return S
