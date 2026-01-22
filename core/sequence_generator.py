"""
Sequence Generation Algorithm for MRN
Implements Section 4: 序列生成算法 (Sequence Generation Algorithm)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from .symbol_embedding import SymbolEmbedding
from .modular_space import ModularSpace
from .matching import BestMatchSelector
from .memory import MemoryManager
from .derivation import TwoStepDerivation, ParallelDerivation


class SequenceGenerator:
    """
    Complete sequence generation system for MRN.
    
    Implements the full algorithm from Section 4 including:
    - Single-step generation with parallel derivation
    - State update mechanism
    - Probability distribution and sampling
    """
    
    def __init__(self,
                 symbol_embedding: SymbolEmbedding,
                 modular_space: ModularSpace,
                 memory_manager: MemoryManager,
                 parallel_window_size: int,
                 temperature: float = 1.0,
                 alpha: float = 0.5,
                 beta: float = 0.5):
        """
        Initialize sequence generator.
        
        Args:
            symbol_embedding: Symbol embedding system
            modular_space: Modular space operations
            memory_manager: Memory management (history + context)
            parallel_window_size: K (1 ≤ K ≤ L)
            temperature: Softmax temperature τ > 0
            alpha: First step weight
            beta: Second step weight
        """
        self.embedding = symbol_embedding
        self.mod_space = modular_space
        self.memory = memory_manager
        self.K = parallel_window_size
        self.tau = temperature
        
        # Validate K
        assert 1 <= self.K <= self.memory.L, \
            f"K={self.K} must be in range [1, {self.memory.L}]"
        
        # Build symbol selector
        symbol_vectors = self.embedding.get_all_discrete_vectors()
        self.selector = BestMatchSelector(symbol_vectors)
        
        # Build derivation engine
        two_step = TwoStepDerivation(
            modular_space=self.mod_space,
            symbol_selector=self.selector,
            alpha=alpha,
            beta=beta
        )
        self.parallel_derivation = ParallelDerivation(two_step)
    
    def initialize_from_sequence(self, initial_symbols: List[str]):
        """
        Initialize memory from initial symbol sequence.
        
        Args:
            initial_symbols: Initial sequence [w_1, w_2, ..., w_T]
        """
        self.memory.reset()
        
        for symbol in initial_symbols:
            x = self.embedding.discretize(symbol)
            self.memory.update(x)
    
    def generate_next(self, 
                     current_symbol: str,
                     deterministic: bool = False) -> Tuple[str, Dict]:
        """
        Generate next symbol in sequence.
        
        Implements complete single-step generation from Section 4.1.
        
        Args:
            current_symbol: Current symbol w_t
            deterministic: If True, use argmax; if False, sample from distribution
            
        Returns:
            (next_symbol, debug_info) tuple
        """
        # Get current state vector
        x_t = self.embedding.discretize(current_symbol)
        
        # Step 1: Construct parallel derivation window
        W_t = self.memory.get_parallel_window(self.K)
        
        # Step 2: Execute parallel derivation over all K history states
        paths = self.parallel_derivation.derive_parallel(x_t, W_t)
        
        # Step 3: Aggregate scores for all symbols
        S = self.parallel_derivation.get_score_vector(
            paths=paths,
            symbol_to_idx=self.embedding.symbol_to_idx,
            vocab_size=self.embedding.N
        )
        
        # Step 4: Convert to probability distribution
        P = self._softmax(S, self.tau)
        
        # Step 5: Select next symbol
        if deterministic:
            next_idx = np.argmax(P)
        else:
            next_idx = np.random.choice(len(P), p=P)
        
        next_symbol = self.embedding.idx_to_symbol[next_idx]
        
        # Collect debug information
        debug_info = {
            'scores': S,
            'probabilities': P,
            'selected_idx': next_idx,
            'paths': paths,
            'parallel_window': W_t,
            'current_state': x_t
        }
        
        return next_symbol, debug_info
    
    def update_state(self, symbol: str):
        """
        Update internal state after generating symbol.
        
        Implements Section 4.2: State Update Mechanism.
        
        Args:
            symbol: Generated symbol w_{t+1}
        """
        x = self.embedding.discretize(symbol)
        self.memory.update(x)
    
    def generate_sequence(self,
                         initial_sequence: List[str],
                         length: int,
                         deterministic: bool = False,
                         return_debug: bool = False) -> Tuple[List[str], Optional[List[Dict]]]:
        """
        Generate complete sequence.
        
        Implements Algorithm: MRN Sequence Generation from Section 4.3.
        
        Args:
            initial_sequence: Initial symbols [w_1, ..., w_T]
            length: Number of symbols to generate (T')
            deterministic: Whether to use deterministic selection
            return_debug: Whether to return debug information
            
        Returns:
            (generated_sequence, debug_info_list) tuple
        """
        # Initialize from initial sequence
        self.initialize_from_sequence(initial_sequence)
        
        # Current symbol is last in initial sequence
        current_symbol = initial_sequence[-1]
        
        # Generate sequence
        generated = []
        debug_list = [] if return_debug else None
        
        for t in range(length):
            # Generate next symbol
            next_symbol, debug_info = self.generate_next(
                current_symbol, 
                deterministic=deterministic
            )
            
            # Update state
            self.update_state(next_symbol)
            
            # Store results
            generated.append(next_symbol)
            if return_debug:
                debug_list.append(debug_info)
            
            # Update current symbol for next iteration
            current_symbol = next_symbol
        
        return generated, debug_list
    
    def _softmax(self, scores: np.ndarray, temperature: float) -> np.ndarray:
        """
        Compute softmax probability distribution.
        
        P(x_j) = exp(S_j / τ) / Σ_{j'} exp(S_{j'} / τ)
        
        Args:
            scores: Score vector S
            temperature: Temperature parameter τ > 0
            
        Returns:
            Probability distribution
        """
        # Numerical stability: subtract max
        scores_scaled = scores / temperature
        scores_stable = scores_scaled - np.max(scores_scaled)
        
        exp_scores = np.exp(scores_stable)
        probs = exp_scores / np.sum(exp_scores)
        
        return probs
    
    def set_temperature(self, tau: float):
        """
        Set temperature parameter.
        
        Args:
            tau: Temperature τ > 0 (higher = more uniform, lower = more peaked)
        """
        assert tau > 0, "Temperature must be positive"
        self.tau = tau
    
    def set_parallel_window_size(self, K: int):
        """
        Set parallel window size.
        
        Args:
            K: Parallel window size (1 ≤ K ≤ L)
        """
        assert 1 <= K <= self.memory.L, \
            f"K={K} must be in range [1, {self.memory.L}]"
        self.K = K
    
    def get_memory_state(self) -> Dict:
        """Get current memory state summary."""
        return self.memory.get_state_summary()
    
    def reset(self):
        """Reset memory to initial state."""
        self.memory.reset()
    
    def __repr__(self) -> str:
        return (f"SequenceGenerator(vocab_size={self.embedding.N}, "
                f"K={self.K}, tau={self.tau})")


class BeamSearchGenerator(SequenceGenerator):
    """
    Beam search variant of sequence generator.
    
    Maintains multiple candidate sequences and selects best overall.
    """
    
    def __init__(self, *args, beam_width: int = 5, **kwargs):
        """
        Initialize beam search generator.
        
        Args:
            beam_width: Number of beams to maintain
            *args, **kwargs: Arguments for SequenceGenerator
        """
        super().__init__(*args, **kwargs)
        self.beam_width = beam_width
    
    def generate_with_beam_search(self,
                                  initial_sequence: List[str],
                                  length: int) -> List[Tuple[List[str], float]]:
        """
        Generate sequences using beam search.
        
        Args:
            initial_sequence: Initial symbols
            length: Length to generate
            
        Returns:
            List of (sequence, log_probability) tuples, sorted by score
        """
        # Initialize beams: (sequence, log_prob, memory_state)
        beams = [(initial_sequence.copy(), 0.0)]
        
        for t in range(length):
            candidates = []
            
            for seq, log_prob in beams:
                # Restore memory state for this beam
                self.initialize_from_sequence(seq)
                current_symbol = seq[-1]
                
                # Get next symbol distribution
                next_symbol, debug_info = self.generate_next(
                    current_symbol, 
                    deterministic=False
                )
                
                # Get top beam_width candidates
                P = debug_info['probabilities']
                top_indices = np.argsort(P)[-self.beam_width:][::-1]
                
                for idx in top_indices:
                    next_sym = self.embedding.idx_to_symbol[idx]
                    new_seq = seq + [next_sym]
                    new_log_prob = log_prob + np.log(P[idx] + 1e-10)
                    candidates.append((new_seq, new_log_prob))
            
            # Select top beam_width candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:self.beam_width]
        
        return beams
