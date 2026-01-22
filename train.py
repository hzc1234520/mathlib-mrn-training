"""
Main Training Script for MRN
"""

import argparse
import json
import numpy as np
from pathlib import Path

from core.symbol_embedding import SymbolEmbedding
from core.modular_space import ModularSpace
from core.memory import MemoryManager
from core.sequence_generator import SequenceGenerator
from core.training import MRNTrainer, CurriculumLearning


def create_sample_vocabulary():
    """Create sample mathematical symbol vocabulary."""
    # Basic mathematical symbols and operations
    symbols = [
        # Numbers
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        # Operations
        '+', '-', '*', '/', '=', '(', ')', ':',
        # Variables
        'x', 'y', 'z', 'a', 'b', 'c',
        # Special tokens
        '<START>', '<END>', '<PAD>',
        # Proof-related
        'theorem', 'proof', 'qed', 'lemma', 'axiom',
        'assume', 'let', 'show', 'have', 'by',
        # Logic
        'and', 'or', 'not', 'implies', 'iff',
        'forall', 'exists', 'in', 'subset',
        # Common words
        'is', 'are', 'if', 'then', 'for', 'the'
    ]
    return symbols


def create_sample_training_data():
    """Create sample training sequences."""
    sequences = [
        # Simple arithmetic
        ['<START>', '1', '+', '1', '=', '2', '<END>'],
        ['<START>', '2', '+', '2', '=', '4', '<END>'],
        ['<START>', '3', '+', '5', '=', '8', '<END>'],
        
        # Basic algebra
        ['<START>', 'let', 'x', '=', '1', '<END>'],
        ['<START>', 'let', 'y', '=', '2', '<END>'],
        ['<START>', 'x', '+', 'y', '=', '3', '<END>'],
        
        # Simple proofs
        ['<START>', 'theorem', ':', '1', '+', '1', '=', '2', '<END>'],
        ['<START>', 'proof', ':', 'by', 'axiom', '<END>'],
        ['<START>', 'lemma', ':', 'x', '=', 'x', '<END>'],
    ]
    return sequences


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MRN model')
    
    # Model hyperparameters
    parser.add_argument('--embedding-dim', type=int, default=32,
                       help='Embedding dimension n')
    parser.add_argument('--modulus', type=int, default=97,
                       help='Modulus m for modular space')
    parser.add_argument('--history-size', type=int, default=64,
                       help='History window size L')
    parser.add_argument('--parallel-window', type=int, default=3,
                       help='Parallel window size K')
    
    # Training hyperparameters
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Softmax temperature')
    parser.add_argument('--lambda-contrastive', type=float, default=1.0,
                       help='Weight for contrastive loss')
    parser.add_argument('--lambda-reg', type=float, default=1e-5,
                       help='Weight for regularization')
    
    # Training options
    parser.add_argument('--use-curriculum', action='store_true',
                       help='Use curriculum learning')
    parser.add_argument('--teacher-forcing', action='store_true', default=True,
                       help='Use teacher forcing')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # I/O
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for checkpoints')
    parser.add_argument('--vocab-file', type=str, default=None,
                       help='Custom vocabulary file (one symbol per line)')
    parser.add_argument('--data-file', type=str, default=None,
                       help='Training data file (JSON format)')
    
    return parser.parse_args()


def load_vocabulary(vocab_file):
    """Load vocabulary from file."""
    with open(vocab_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_training_data(data_file):
    """Load training data from JSON file."""
    with open(data_file, 'r') as f:
        data = json.load(f)
    return data['sequences']


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 80)
    print("MRN Training System")
    print("=" * 80)
    
    # Load or create vocabulary
    if args.vocab_file:
        vocabulary = load_vocabulary(args.vocab_file)
        print(f"Loaded vocabulary from {args.vocab_file}: {len(vocabulary)} symbols")
    else:
        vocabulary = create_sample_vocabulary()
        print(f"Using sample vocabulary: {len(vocabulary)} symbols")
    
    # Load or create training data
    if args.data_file:
        sequences = load_training_data(args.data_file)
        print(f"Loaded training data from {args.data_file}: {len(sequences)} sequences")
    else:
        sequences = create_sample_training_data()
        print(f"Using sample training data: {len(sequences)} sequences")
    
    # Initialize components
    print("\nInitializing model components...")
    
    # Symbol embedding
    embedding = SymbolEmbedding(
        vocabulary=vocabulary,
        embedding_dim=args.embedding_dim,
        modulus=args.modulus,
        seed=args.seed
    )
    print(f"  Symbol Embedding: {embedding}")
    print(f"  Parameters: {embedding.get_parameter_count():,}")
    
    # Modular space
    mod_space = ModularSpace(
        modulus=args.modulus,
        dimension=args.embedding_dim
    )
    print(f"  Modular Space: Z_{args.modulus}^{args.embedding_dim}")
    
    # Memory manager
    memory = MemoryManager(
        history_size=args.history_size,
        dimension=args.embedding_dim,
        modulus=args.modulus,
        context_seed=args.seed
    )
    print(f"  Memory: {memory}")
    
    # Sequence generator
    generator = SequenceGenerator(
        symbol_embedding=embedding,
        modular_space=mod_space,
        memory_manager=memory,
        parallel_window_size=args.parallel_window,
        temperature=args.temperature
    )
    print(f"  Generator: {generator}")
    
    # Trainer
    trainer = MRNTrainer(
        generator=generator,
        learning_rate=args.learning_rate,
        lambda_contrastive=args.lambda_contrastive,
        lambda_reg=args.lambda_reg,
        use_teacher_forcing=args.teacher_forcing
    )
    print(f"  Trainer initialized with lr={args.learning_rate}")
    
    # Curriculum learning
    if args.use_curriculum:
        curriculum = CurriculumLearning(
            initial_K=1,
            initial_L=16,
            final_K=args.parallel_window,
            final_L=args.history_size,
            num_stages=3
        )
        print(f"  Curriculum: {curriculum}")
    
    # Training loop
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)
    
    best_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # Update curriculum if enabled
        if args.use_curriculum:
            stage_idx = epoch // (args.num_epochs // 3)
            stage = curriculum.get_stage(stage_idx)
            generator.set_parallel_window_size(stage['K'])
            # Note: changing L would require recreating memory manager
        
        # Train epoch
        epoch_stats = trainer.train_epoch(sequences, shuffle=True)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
            print(f"  Total Loss:        {epoch_stats['total_loss']:.6f}")
            print(f"  NLL Loss:          {epoch_stats['nll_loss']:.6f}")
            print(f"  Contrastive Loss:  {epoch_stats['contrastive_loss']:.6f}")
            print(f"  Reg Loss:          {epoch_stats['reg_loss']:.6f}")
        
        # Save checkpoint
        if epoch_stats['total_loss'] < best_loss:
            best_loss = epoch_stats['total_loss']
            checkpoint_path = output_dir / 'best_model.npz'
            trainer.save_checkpoint(str(checkpoint_path))
            print(f"  Saved best model to {checkpoint_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 50 == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.npz'
            trainer.save_checkpoint(str(checkpoint_path))
    
    # Final save
    final_path = output_dir / 'final_model.npz'
    trainer.save_checkpoint(str(final_path))
    
    # Save embeddings separately
    embedding_path = output_dir / 'embeddings.npy'
    embedding.save_embeddings(str(embedding_path))
    
    # Save configuration
    config = {
        'vocabulary': vocabulary,
        'embedding_dim': args.embedding_dim,
        'modulus': args.modulus,
        'history_size': args.history_size,
        'parallel_window': args.parallel_window,
        'temperature': args.temperature
    }
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Model saved to: {output_dir}")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Parameters: {embedding.get_parameter_count():,}")


if __name__ == '__main__':
    main()
