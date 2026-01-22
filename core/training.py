"""
Training Framework for MRN
Implements Section 5: 训练原理 (Training Principles)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from .symbol_embedding import SymbolEmbedding
from .sequence_generator import SequenceGenerator


class StraightThroughEstimator:
    """
    Straight-through estimator for gradient flow through discrete operations.
    
    Implements Section 5.2: 连续松弛 (Continuous Relaxation)
    
    Forward: discrete operation (round, mod)
    Backward: identity (gradient flows through)
    """
    
    @staticmethod
    def round_ste(x: np.ndarray) -> Tuple[np.ndarray, Callable]:
        """
        Round with straight-through gradient.
        
        Forward: round(x)
        Backward: ∂round(x)/∂x ≈ 1
        
        Returns:
            (rounded_value, backward_fn)
        """
        rounded = np.round(x)
        
        def backward(grad_output):
            # Gradient flows straight through
            return grad_output
        
        return rounded, backward
    
    @staticmethod
    def mod_ste(x: np.ndarray, m: int) -> Tuple[np.ndarray, Callable]:
        """
        Modulo with straight-through gradient.
        
        Forward: x mod m
        Backward: ∂(x mod m)/∂x ≈ 1
        
        Returns:
            (modded_value, backward_fn)
        """
        modded = x % m
        
        def backward(grad_output):
            # Gradient flows straight through
            return grad_output
        
        return modded, backward


class LossFunction:
    """
    Loss functions for MRN training.
    
    Implements Section 5.3: 损失函数 (Loss Functions)
    """
    
    @staticmethod
    def negative_log_likelihood(predictions: np.ndarray, 
                               target_idx: int) -> Tuple[float, np.ndarray]:
        """
        Compute negative log-likelihood loss.
        
        L_t^NLL = -log P(w_{t+1} | w_1, ..., w_t)
        
        Args:
            predictions: Probability distribution P ∈ R^N
            target_idx: Index of target symbol
            
        Returns:
            (loss, gradient) tuple
        """
        # Clip for numerical stability
        p_target = np.clip(predictions[target_idx], 1e-10, 1.0)
        loss = -np.log(p_target)
        
        # Gradient: ∂L/∂P_j = -1/P_target if j == target, else 0
        grad = np.zeros_like(predictions)
        grad[target_idx] = -1.0 / p_target
        
        return float(loss), grad
    
    @staticmethod
    def contrastive_loss(positive_score: float,
                        negative_scores: List[float],
                        margin: float = 1.0) -> Tuple[float, Dict[str, float]]:
        """
        Compute contrastive loss.
        
        L_contrastive = Σ_t max(0, margin - s_positive + max_{w≠w_{t+1}} s)
        
        Ensures correct symbol's score is higher than others by at least margin.
        
        Args:
            positive_score: Score for correct symbol s_positive
            negative_scores: Scores for incorrect symbols
            margin: Margin value (typically 1.0)
            
        Returns:
            (loss, gradients) tuple
        """
        if len(negative_scores) == 0:
            return 0.0, {'positive': 0.0, 'negative_max': 0.0}
        
        max_negative = max(negative_scores)
        loss = max(0.0, margin - positive_score + max_negative)
        
        # Gradients
        grad_positive = -1.0 if loss > 0 else 0.0
        grad_negative_max = 1.0 if loss > 0 else 0.0
        
        return loss, {
            'positive': grad_positive,
            'negative_max': grad_negative_max
        }
    
    @staticmethod
    def l2_regularization(embeddings: np.ndarray, 
                         lambda_reg: float) -> Tuple[float, np.ndarray]:
        """
        Compute L2 regularization loss.
        
        L_reg = ||E||_F^2 = Σ_v ||e_v||_2^2
        
        Args:
            embeddings: Embedding matrix E ∈ R^{N×n}
            lambda_reg: Regularization coefficient λ_reg
            
        Returns:
            (loss, gradient) tuple
        """
        loss = lambda_reg * np.sum(embeddings ** 2)
        grad = 2 * lambda_reg * embeddings
        
        return float(loss), grad
    
    @staticmethod
    def total_loss(nll_loss: float,
                  contrastive_loss: float,
                  reg_loss: float,
                  lambda_contrastive: float = 1.0,
                  lambda_reg: float = 1e-5) -> float:
        """
        Compute total loss.
        
        L = L_NLL + λ_contrastive·L_contrastive + λ_reg·L_reg
        
        Args:
            nll_loss: Negative log-likelihood loss
            contrastive_loss: Contrastive loss
            reg_loss: Regularization loss
            lambda_contrastive: Weight for contrastive loss
            lambda_reg: Weight for regularization
            
        Returns:
            Total loss
        """
        return nll_loss + lambda_contrastive * contrastive_loss + lambda_reg * reg_loss


class MRNTrainer:
    """
    Training system for MRN.
    
    Implements complete training algorithm with teacher forcing and
    curriculum learning.
    """
    
    def __init__(self,
                 generator: SequenceGenerator,
                 learning_rate: float = 0.01,
                 lambda_contrastive: float = 1.0,
                 lambda_reg: float = 1e-5,
                 use_teacher_forcing: bool = True):
        """
        Initialize trainer.
        
        Args:
            generator: Sequence generator
            learning_rate: Learning rate η
            lambda_contrastive: Weight for contrastive loss
            lambda_reg: Weight for regularization
            use_teacher_forcing: Whether to use teacher forcing
        """
        self.generator = generator
        self.embedding = generator.embedding
        self.lr = learning_rate
        self.lambda_contrastive = lambda_contrastive
        self.lambda_reg = lambda_reg
        self.use_teacher_forcing = use_teacher_forcing
        
        self.loss_fn = LossFunction()
        self.ste = StraightThroughEstimator()
        
        # Training statistics
        self.training_stats = {
            'total_loss': [],
            'nll_loss': [],
            'contrastive_loss': [],
            'reg_loss': []
        }
    
    def train_step(self, 
                   sequence: List[str],
                   return_details: bool = False) -> Dict:
        """
        Execute single training step on a sequence.
        
        Args:
            sequence: Training sequence [w_1, w_2, ..., w_T]
            return_details: Whether to return detailed loss breakdown
            
        Returns:
            Dictionary with loss information and gradients
        """
        T = len(sequence)
        if T < 2:
            raise ValueError("Sequence must have at least 2 symbols")
        
        total_nll = 0.0
        total_contrastive = 0.0
        
        # Accumulate gradients
        embedding_gradients = np.zeros_like(self.embedding.E)
        
        # Teacher forcing: use true history
        if self.use_teacher_forcing:
            # Initialize with first few symbols
            init_len = min(self.generator.memory.L, T - 1)
            self.generator.initialize_from_sequence(sequence[:init_len])
            
            # Train on remaining positions
            for t in range(init_len, T - 1):
                current_symbol = sequence[t]
                target_symbol = sequence[t + 1]
                target_idx = self.embedding.symbol_to_idx[target_symbol]
                
                # Generate next symbol distribution
                _, debug_info = self.generator.generate_next(
                    current_symbol, 
                    deterministic=False
                )
                
                # NLL loss
                predictions = debug_info['probabilities']
                nll, nll_grad = self.loss_fn.negative_log_likelihood(
                    predictions, target_idx
                )
                total_nll += nll
                
                # Contrastive loss from derivation paths
                paths = debug_info['paths']
                positive_score = None
                negative_scores = []
                
                alpha, beta = self.generator.parallel_derivation.derivation.get_weights()
                for path in paths:
                    score = path.compute_score(alpha, beta)
                    if path.w == target_symbol:
                        positive_score = score
                    else:
                        negative_scores.append(score)
                
                if positive_score is not None:
                    contrastive, _ = self.loss_fn.contrastive_loss(
                        positive_score, negative_scores
                    )
                    total_contrastive += contrastive
                
                # Update state with true symbol (teacher forcing)
                self.generator.update_state(target_symbol)
                
                # Accumulate gradients (simplified - full implementation would
                # require autograd or manual chain rule)
                # Here we just accumulate NLL gradient
                for i, symbol in enumerate(self.embedding.vocabulary):
                    if i == target_idx:
                        embedding_gradients[i] += nll_grad[i] * self.lr
        
        # Regularization loss
        reg_loss, reg_grad = self.loss_fn.l2_regularization(
            self.embedding.E, self.lambda_reg
        )
        
        # Total loss
        total_loss = self.loss_fn.total_loss(
            total_nll, total_contrastive, reg_loss,
            self.lambda_contrastive, self.lambda_reg
        )
        
        # Update embeddings
        self._update_embeddings(embedding_gradients, reg_grad)
        
        # Record statistics
        self.training_stats['total_loss'].append(total_loss)
        self.training_stats['nll_loss'].append(total_nll)
        self.training_stats['contrastive_loss'].append(total_contrastive)
        self.training_stats['reg_loss'].append(reg_loss)
        
        result = {
            'total_loss': total_loss,
            'nll_loss': total_nll,
            'contrastive_loss': total_contrastive,
            'reg_loss': reg_loss
        }
        
        if return_details:
            result['embedding_gradients'] = embedding_gradients
        
        return result
    
    def _update_embeddings(self, 
                          embedding_gradients: np.ndarray,
                          reg_gradients: np.ndarray):
        """
        Update embedding matrix using accumulated gradients.
        
        Args:
            embedding_gradients: Gradients from task loss
            reg_gradients: Gradients from regularization
        """
        # Combine gradients
        total_grad = embedding_gradients + reg_gradients
        
        # Gradient descent update
        self.embedding.E -= self.lr * total_grad
    
    def train_epoch(self, 
                   sequences: List[List[str]],
                   shuffle: bool = True) -> Dict:
        """
        Train for one epoch over dataset.
        
        Args:
            sequences: List of training sequences
            shuffle: Whether to shuffle sequences
            
        Returns:
            Epoch statistics
        """
        if shuffle:
            np.random.shuffle(sequences)
        
        epoch_stats = {
            'total_loss': 0.0,
            'nll_loss': 0.0,
            'contrastive_loss': 0.0,
            'reg_loss': 0.0,
            'num_sequences': len(sequences)
        }
        
        for seq in sequences:
            step_result = self.train_step(seq)
            for key in ['total_loss', 'nll_loss', 'contrastive_loss', 'reg_loss']:
                epoch_stats[key] += step_result[key]
        
        # Average losses
        for key in ['total_loss', 'nll_loss', 'contrastive_loss', 'reg_loss']:
            epoch_stats[key] /= len(sequences)
        
        return epoch_stats
    
    def get_training_stats(self) -> Dict:
        """Get accumulated training statistics."""
        return self.training_stats
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint."""
        checkpoint = {
            'embeddings': self.embedding.E,
            'training_stats': self.training_stats,
            'learning_rate': self.lr
        }
        np.savez(filepath, **checkpoint)
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = np.load(filepath, allow_pickle=True)
        self.embedding.E = checkpoint['embeddings']
        self.training_stats = checkpoint['training_stats'].item()
        self.lr = float(checkpoint['learning_rate'])


class CurriculumLearning:
    """
    Curriculum learning strategy for MRN.
    
    Implements Section 5.4: 训练算法 - 课程学习策略
    """
    
    def __init__(self,
                 initial_K: int = 1,
                 initial_L: int = 16,
                 final_K: int = 5,
                 final_L: int = 1024,
                 num_stages: int = 3):
        """
        Initialize curriculum.
        
        Args:
            initial_K: Initial parallel window size
            initial_L: Initial history window size
            final_K: Final parallel window size
            final_L: Final history window size
            num_stages: Number of curriculum stages
        """
        self.stages = []
        
        # Create stages
        K_values = np.linspace(initial_K, final_K, num_stages).astype(int)
        L_values = np.linspace(initial_L, final_L, num_stages).astype(int)
        
        for i in range(num_stages):
            self.stages.append({
                'K': int(K_values[i]),
                'L': int(L_values[i]),
                'stage': i + 1
            })
    
    def get_stage(self, stage_idx: int) -> Dict:
        """Get configuration for curriculum stage."""
        if stage_idx < 0 or stage_idx >= len(self.stages):
            return self.stages[-1]  # Return final stage
        return self.stages[stage_idx]
    
    def __repr__(self) -> str:
        return f"CurriculumLearning(stages={len(self.stages)})"
