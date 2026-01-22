"""
Training components for MRN (Section 5)

This module implements loss functions, training strategies, and optimization
for the MRN system.
"""

import numpy as np
from typing import List, Tuple, Dict
from .embedding import SymbolEmbedding, EmbeddingGradient
from .sequence_generator import MRNGenerator


class LossFunction:
    """Collection of loss functions for MRN training"""
    
    @staticmethod
    def nll_loss(probabilities: np.ndarray, target_idx: int) -> float:
        """
        Negative log-likelihood loss
        
        L_t^NLL = -log P(w_{t+1} | w_1, ..., w_t)
        
        Args:
            probabilities: Probability distribution over symbols
            target_idx: Index of target symbol
            
        Returns:
            NLL loss value
        """
        # Clip probability to avoid log(0)
        prob = np.clip(probabilities[target_idx], 1e-10, 1.0)
        return -np.log(prob)
    
    @staticmethod
    def contrastive_loss(scores: np.ndarray, target_idx: int, margin: float = 1.0) -> float:
        """
        Contrastive loss to push correct symbol score higher
        
        L_contrastive = max(0, margin - s_positive + max_{i≠target} s_i)
        
        Args:
            scores: Score array for all symbols
            target_idx: Index of correct symbol
            margin: Margin value (typically 1.0)
            
        Returns:
            Contrastive loss value
        """
        s_positive = scores[target_idx]
        
        # Find max score among non-target symbols
        s_negative_max = -np.inf
        for i, s in enumerate(scores):
            if i != target_idx and s > s_negative_max:
                s_negative_max = s
        
        loss = max(0.0, margin - s_positive + s_negative_max)
        return loss
    
    @staticmethod
    def total_loss(probabilities: np.ndarray, 
                   scores: np.ndarray,
                   target_idx: int,
                   embeddings: np.ndarray,
                   lambda_contrastive: float = 1.0,
                   lambda_reg: float = 1e-5,
                   margin: float = 1.0) -> Tuple[float, Dict[str, float]]:
        """
        Compute total loss with all components
        
        L = L_NLL + λ_contrastive * L_contrastive + λ_reg * L_reg
        
        Args:
            probabilities: Probability distribution
            scores: Score array
            target_idx: Target symbol index
            embeddings: Embedding matrix for regularization
            lambda_contrastive: Weight for contrastive loss
            lambda_reg: Weight for L2 regularization
            margin: Margin for contrastive loss
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # NLL loss
        l_nll = LossFunction.nll_loss(probabilities, target_idx)
        
        # Contrastive loss
        l_contrastive = LossFunction.contrastive_loss(scores, target_idx, margin)
        
        # L2 regularization
        l_reg, _ = EmbeddingGradient.compute_l2_regularization(
            embeddings, lambda_reg
        )
        
        # Total loss
        total = l_nll + lambda_contrastive * l_contrastive + l_reg
        
        components = {
            'nll': l_nll,
            'contrastive': l_contrastive,
            'regularization': l_reg,
            'total': total
        }
        
        return total, components


class MRNTrainer:
    """Training manager for MRN system"""
    
    def __init__(self,
                 generator: MRNGenerator,
                 learning_rate: float = 0.01,
                 lambda_contrastive: float = 1.0,
                 lambda_reg: float = 1e-5,
                 margin: float = 1.0):
        """
        Initialize trainer
        
        Args:
            generator: MRNGenerator instance
            learning_rate: Learning rate for SGD
            lambda_contrastive: Weight for contrastive loss
            lambda_reg: Weight for L2 regularization
            margin: Margin for contrastive loss
        """
        self.generator = generator
        self.embedding = generator.embedding
        self.learning_rate = learning_rate
        self.lambda_contrastive = lambda_contrastive
        self.lambda_reg = lambda_reg
        self.margin = margin
        
        # Training statistics
        self.training_history = []
    
    def train_step(self, 
                   initial_sequence: List[str], 
                   target_sequence: List[str],
                   teacher_forcing: bool = True) -> Dict[str, float]:
        """
        Execute one training step with teacher forcing
        
        Args:
            initial_sequence: Initial context sequence
            target_sequence: Target sequence to predict
            teacher_forcing: If True, use ground truth history
            
        Returns:
            Dictionary of loss components
        """
        # Initialize generator
        self.generator.initialize(initial_sequence)
        
        total_loss = 0.0
        loss_components = {'nll': 0.0, 'contrastive': 0.0, 'regularization': 0.0}
        
        # Process each target symbol
        for target_symbol in target_sequence:
            # Generate next symbol (get scores and probabilities)
            _, metadata = self.generator.generate_next(deterministic=False)
            
            scores = metadata['scores']
            probabilities = metadata['probabilities']
            target_idx = self.embedding.get_index(target_symbol)
            
            # Compute loss
            loss, components = LossFunction.total_loss(
                probabilities, scores, target_idx,
                self.embedding.embeddings,
                self.lambda_contrastive,
                self.lambda_reg,
                self.margin
            )
            
            total_loss += components['total']
            for key in loss_components:
                loss_components[key] += components[key]
            
            # Compute gradients and update embeddings
            # Simplified gradient: update towards target
            target_vector = self.embedding.get_discrete_vector(target_symbol)
            predicted_idx = metadata['selected_idx']
            
            if predicted_idx != target_idx:
                # Simple gradient: move predicted embedding closer to target
                pred_embedding = self.embedding.get_embedding_by_idx(predicted_idx)
                target_embedding = self.embedding.get_embedding_by_idx(target_idx)
                
                # Gradient for predicted (move away)
                grad_pred = (pred_embedding - target_embedding) * 0.1
                self.embedding.embeddings[predicted_idx] -= self.learning_rate * grad_pred
                
                # Gradient for target (move closer)
                grad_target = (target_embedding - pred_embedding) * 0.1
                self.embedding.embeddings[target_idx] -= self.learning_rate * grad_target
            
            # Apply L2 regularization gradient
            _, reg_grad = EmbeddingGradient.compute_l2_regularization(
                self.embedding.embeddings, self.lambda_reg
            )
            self.embedding.embeddings -= self.learning_rate * reg_grad
            
            # Teacher forcing: use ground truth for next step
            if teacher_forcing:
                target_state = self.embedding.get_discrete_vector(target_symbol)
                self.generator.history.update(target_state)
                self.generator.context.update(target_state)
                self.generator.current_state = target_state
        
        # Average losses
        n_steps = len(target_sequence)
        loss_components = {k: v / n_steps for k, v in loss_components.items()}
        loss_components['total'] = total_loss / n_steps
        
        return loss_components
    
    def train_epoch(self, 
                    training_data: List[Tuple[List[str], List[str]]],
                    teacher_forcing: bool = True) -> Dict[str, float]:
        """
        Train for one epoch over all training data
        
        Args:
            training_data: List of (initial_sequence, target_sequence) pairs
            teacher_forcing: Use teacher forcing
            
        Returns:
            Average loss components over epoch
        """
        epoch_losses = {'nll': 0.0, 'contrastive': 0.0, 'regularization': 0.0, 'total': 0.0}
        
        for initial_seq, target_seq in training_data:
            losses = self.train_step(initial_seq, target_seq, teacher_forcing)
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
        
        # Average over all samples
        n_samples = len(training_data)
        epoch_losses = {k: v / n_samples for k, v in epoch_losses.items()}
        
        self.training_history.append(epoch_losses)
        
        return epoch_losses
    
    def train(self, 
              training_data: List[Tuple[List[str], List[str]]],
              num_epochs: int,
              teacher_forcing: bool = True,
              verbose: bool = True) -> List[Dict[str, float]]:
        """
        Train for multiple epochs
        
        Args:
            training_data: Training data
            num_epochs: Number of epochs
            teacher_forcing: Use teacher forcing
            verbose: Print progress
            
        Returns:
            Training history
        """
        for epoch in range(num_epochs):
            losses = self.train_epoch(training_data, teacher_forcing)
            
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Loss={losses['total']:.4f}, "
                      f"NLL={losses['nll']:.4f}, "
                      f"Contrastive={losses['contrastive']:.4f}")
        
        return self.training_history


class CurriculumLearning:
    """Curriculum learning strategy for progressive training"""
    
    @staticmethod
    def get_schedule(stage: str) -> Dict[str, int]:
        """
        Get hyperparameters for training stage
        
        Args:
            stage: One of 'early', 'middle', 'late'
            
        Returns:
            Dictionary with K and L values
        """
        schedules = {
            'early': {'K': 1, 'L': 16},
            'middle': {'K': 3, 'L': 128},
            'late': {'K': 5, 'L': 1024}
        }
        return schedules.get(stage, schedules['middle'])
    
    @staticmethod
    def apply_schedule(generator: MRNGenerator, stage: str):
        """
        Apply curriculum schedule to generator
        
        Args:
            generator: MRNGenerator instance
            stage: Training stage
        """
        schedule = CurriculumLearning.get_schedule(stage)
        generator.parallel_window_size = schedule['K']
        generator.history_window_size = schedule['L']
        # Reinitialize history window with new size
        generator.history = HistoryWindow(schedule['L'], generator.embedding.embedding_dim)


# Import HistoryWindow for curriculum learning
from .sequence_generator import HistoryWindow
