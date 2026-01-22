"""
Memory Management for MRN
Implements Section 1.2: 历史窗口与上下文向量 (History Window and Context Vector)
and Section 1.3: 并行推导窗口 (Parallel Derivation Window)
"""

import numpy as np
from typing import List, Optional


class HistoryWindow:
    """
    Fixed-size history window storing recent L states.
    
    H_t ∈ Z_m^{n×L} = [x_t, x_{t-1}, ..., x_{t-L+1}]
    
    Convention: H_t[d] returns state d steps in the past (x_{t-d})
    where d ∈ {0, 1, ..., L-1}
    """
    
    def __init__(self, size: int, dimension: int, modulus: int):
        """
        Initialize history window.
        
        Args:
            size: Window size L
            dimension: Vector dimension n
            modulus: Modulus m for modular space
        """
        self.L = size
        self.n = dimension
        self.m = modulus
        
        # Initialize with zero vectors
        # Shape: (L, n) where each row is a state vector
        self.window = np.zeros((self.L, self.n), dtype=int)
        self.current_size = 0  # Track how many actual states we have
    
    def push(self, x_t: np.ndarray):
        """
        Add new state to history, shifting older states.
        
        H_{t+1} = [x_{t+1}, H_t[0], H_t[1], ..., H_t[L-2]]
        
        Args:
            x_t: New state vector ∈ Z_m^n
        """
        assert x_t.shape == (self.n,), f"Expected shape ({self.n},), got {x_t.shape}"
        
        # Shift all states back by one position
        self.window[1:] = self.window[:-1]
        
        # Insert new state at front
        self.window[0] = x_t % self.m
        
        # Update size (capped at L)
        self.current_size = min(self.current_size + 1, self.L)
    
    def get(self, d: int) -> np.ndarray:
        """
        Get state d steps in the past.
        
        H_t[d] = x_{t-d}
        
        Args:
            d: Steps back in history (0 = current, 1 = previous, etc.)
            
        Returns:
            x_{t-d} ∈ Z_m^n (or zero vector if d exceeds history)
        """
        assert 0 <= d < self.L, f"Index {d} out of range [0, {self.L})"
        
        if d >= self.current_size:
            # Return zero vector for uninitialized history
            return np.zeros(self.n, dtype=int)
        
        return self.window[d].copy()
    
    def get_parallel_window(self, K: int) -> np.ndarray:
        """
        Get K consecutive states for parallel derivation.
        
        W_t = [H_t[0], H_t[1], ..., H_t[K-1]]
            = [x_t, x_{t-1}, ..., x_{t-K+1}]
        
        Args:
            K: Parallel window size (1 ≤ K ≤ L)
            
        Returns:
            W_t ∈ Z_m^{K×n}: K consecutive history states
        """
        assert 1 <= K <= self.L, f"K={K} must be in range [1, {self.L}]"
        
        # Return first K states
        parallel_window = np.zeros((K, self.n), dtype=int)
        for i in range(K):
            parallel_window[i] = self.get(i)
        
        return parallel_window
    
    def reset(self):
        """Reset window to all zeros."""
        self.window = np.zeros((self.L, self.n), dtype=int)
        self.current_size = 0
    
    def __len__(self) -> int:
        """Return current number of states in window."""
        return self.current_size
    
    def __repr__(self) -> str:
        return f"HistoryWindow(size={self.L}, current={self.current_size}, dim={self.n})"


class ContextVector:
    """
    Recursive context vector for long-term memory.
    
    c_t ∈ Z_m^n carries compressed history information
    Update: c_{t+1} = (A·c_t + B·x_{t+1}) mod m
    
    where A, B ∈ Z_m^{n×n} are fixed random matrices
    """
    
    def __init__(self, dimension: int, modulus: int, seed: Optional[int] = None):
        """
        Initialize context vector with fixed random matrices.
        
        Args:
            dimension: Vector dimension n
            modulus: Modulus m for modular space
            seed: Random seed for reproducibility
        """
        self.n = dimension
        self.m = modulus
        
        # Initialize context vector to zero
        self.c = np.zeros(self.n, dtype=int)
        
        # Initialize fixed random matrices A, B ∈ Z_m^{n×n}
        # These remain constant throughout training and inference
        if seed is not None:
            np.random.seed(seed)
        
        self.A = np.random.randint(0, self.m, size=(self.n, self.n))
        self.B = np.random.randint(0, self.m, size=(self.n, self.n))
    
    def update(self, x_t: np.ndarray) -> np.ndarray:
        """
        Update context vector with new state.
        
        c_{t+1} = (A·c_t + B·x_t) mod m
        
        Args:
            x_t: New state vector ∈ Z_m^n
            
        Returns:
            Updated context vector c_{t+1}
        """
        assert x_t.shape == (self.n,), f"Expected shape ({self.n},), got {x_t.shape}"
        
        # Matrix-vector multiplication in modular space
        Ac = np.dot(self.A, self.c).astype(int) % self.m
        Bx = np.dot(self.B, x_t).astype(int) % self.m
        
        # Update context
        self.c = (Ac + Bx) % self.m
        
        return self.c.copy()
    
    def get(self) -> np.ndarray:
        """Get current context vector."""
        return self.c.copy()
    
    def reset(self):
        """Reset context to zero vector."""
        self.c = np.zeros(self.n, dtype=int)
    
    def __repr__(self) -> str:
        return f"ContextVector(dim={self.n}, modulus={self.m})"


class MemoryManager:
    """
    Combined memory management for history window and context vector.
    Implements dual-channel memory mechanism.
    """
    
    def __init__(self, 
                 history_size: int,
                 dimension: int,
                 modulus: int,
                 context_seed: Optional[int] = None):
        """
        Initialize memory manager.
        
        Args:
            history_size: History window size L
            dimension: Vector dimension n
            modulus: Modulus m
            context_seed: Seed for context vector initialization
        """
        self.history = HistoryWindow(history_size, dimension, modulus)
        self.context = ContextVector(dimension, modulus, seed=context_seed)
        
        self.L = history_size
        self.n = dimension
        self.m = modulus
    
    def update(self, x_t: np.ndarray):
        """
        Update both history window and context vector.
        
        Args:
            x_t: New state vector ∈ Z_m^n
        """
        # Update history window (short-term memory)
        self.history.push(x_t)
        
        # Update context vector (long-term memory)
        self.context.update(x_t)
    
    def get_history_state(self, d: int) -> np.ndarray:
        """Get state d steps back in history."""
        return self.history.get(d)
    
    def get_parallel_window(self, K: int) -> np.ndarray:
        """Get K consecutive states for parallel derivation."""
        return self.history.get_parallel_window(K)
    
    def get_context(self) -> np.ndarray:
        """Get current context vector."""
        return self.context.get()
    
    def reset(self):
        """Reset both memory channels."""
        self.history.reset()
        self.context.reset()
    
    def get_state_summary(self) -> dict:
        """Get summary of current memory state."""
        return {
            'history_size': len(self.history),
            'context_norm': np.linalg.norm(self.context.get()),
            'history_capacity': self.L,
            'dimension': self.n,
            'modulus': self.m
        }
    
    def __repr__(self) -> str:
        return (f"MemoryManager(history_size={self.L}, "
                f"dimension={self.n}, modulus={self.m})")
