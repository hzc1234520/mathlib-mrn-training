"""
Simple test to verify MRN implementation
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core import (
    SymbolEmbedding, ModularSpace, MemoryManager,
    SequenceGenerator, MRNTrainer
)


def test_basic_components():
    """Test basic component initialization."""
    print("Testing basic components...")
    
    # Test parameters
    vocabulary = ['a', 'b', 'c', 'd', 'e']
    embedding_dim = 8
    modulus = 7
    history_size = 16
    parallel_window = 2
    
    # Symbol embedding
    embedding = SymbolEmbedding(
        vocabulary=vocabulary,
        embedding_dim=embedding_dim,
        modulus=modulus,
        seed=42
    )
    assert embedding.N == len(vocabulary)
    assert embedding.n == embedding_dim
    print(f"  ✓ Symbol embedding: {embedding}")
    
    # Modular space
    mod_space = ModularSpace(modulus=modulus, dimension=embedding_dim)
    print(f"  ✓ Modular space: Z_{modulus}^{embedding_dim}")
    
    # Memory
    memory = MemoryManager(
        history_size=history_size,
        dimension=embedding_dim,
        modulus=modulus,
        context_seed=42
    )
    print(f"  ✓ Memory: {memory}")
    
    # Sequence generator
    generator = SequenceGenerator(
        symbol_embedding=embedding,
        modular_space=mod_space,
        memory_manager=memory,
        parallel_window_size=parallel_window,
        temperature=1.0
    )
    print(f"  ✓ Generator: {generator}")
    
    return embedding, mod_space, memory, generator


def test_sequence_generation():
    """Test sequence generation."""
    print("\nTesting sequence generation...")
    
    embedding, mod_space, memory, generator = test_basic_components()
    
    # Test generation
    initial_sequence = ['a', 'b']
    length = 5
    
    generated, debug_info = generator.generate_sequence(
        initial_sequence=initial_sequence,
        length=length,
        deterministic=True,
        return_debug=True
    )
    
    assert len(generated) == length
    print(f"  ✓ Generated sequence: {' '.join(initial_sequence + generated)}")
    
    return generator


def test_training():
    """Test training functionality."""
    print("\nTesting training...")
    
    generator = test_sequence_generation()
    
    # Create trainer
    trainer = MRNTrainer(
        generator=generator,
        learning_rate=0.01,
        use_teacher_forcing=True
    )
    
    # Test training step
    sequences = [
        ['a', 'b', 'c', 'a'],
        ['b', 'c', 'd', 'b'],
        ['c', 'd', 'e', 'c']
    ]
    
    result = trainer.train_step(sequences[0])
    print(f"  ✓ Training step complete")
    print(f"    Total loss: {result['total_loss']:.6f}")
    print(f"    NLL loss: {result['nll_loss']:.6f}")
    
    # Test epoch
    epoch_stats = trainer.train_epoch(sequences)
    print(f"  ✓ Epoch complete")
    print(f"    Avg total loss: {epoch_stats['total_loss']:.6f}")


def test_matching():
    """Test matching functions."""
    print("\nTesting matching functions...")
    
    from core.matching import MatchingFunction
    
    matcher = MatchingFunction(modulus=7, dimension=8)
    
    # Test exact matching
    q = np.array([1, 2, 3, 4, 5, 6, 7, 0])
    x1 = np.array([1, 2, 3, 4, 5, 6, 7, 0])  # Perfect match
    x2 = np.array([1, 2, 3, 0, 0, 0, 0, 1])  # Partial match (first 3 only)
    
    score1 = matcher.match(q, x1)
    score2 = matcher.match(q, x2)
    
    assert score1 == 8, "Perfect match should have score 8"
    assert score2 == 3, "Partial match should have score 3"
    
    print(f"  ✓ Match(q, x1) = {score1} (perfect)")
    print(f"  ✓ Match(q, x2) = {score2} (partial)")
    
    # Test batch matching
    X = np.array([x1, x2])
    scores = matcher.match_batch(q, X)
    assert len(scores) == 2
    print(f"  ✓ Batch matching: {scores}")
    
    # Test best match
    idx, score = matcher.best_match(q, X, return_score=True)
    assert idx == 0, "Best match should be index 0"
    assert score == 8
    print(f"  ✓ Best match: index {idx}, score {score}")


def test_memory():
    """Test memory management."""
    print("\nTesting memory management...")
    
    from core.memory import HistoryWindow, ContextVector
    
    # Test history window
    history = HistoryWindow(size=4, dimension=3, modulus=7)
    
    # Add states
    x1 = np.array([1, 2, 3])
    x2 = np.array([4, 5, 6])
    x3 = np.array([0, 1, 2])
    
    history.push(x1)
    history.push(x2)
    history.push(x3)
    
    # Check retrieval
    assert np.array_equal(history.get(0), x3), "Most recent should be x3"
    assert np.array_equal(history.get(1), x2), "Previous should be x2"
    assert np.array_equal(history.get(2), x1), "Two steps back should be x1"
    
    print(f"  ✓ History window working correctly")
    
    # Test parallel window
    W = history.get_parallel_window(2)
    assert W.shape == (2, 3)
    print(f"  ✓ Parallel window shape: {W.shape}")
    
    # Test context vector
    context = ContextVector(dimension=3, modulus=7, seed=42)
    c1 = context.update(x1)
    c2 = context.update(x2)
    
    assert c1.shape == (3,)
    assert c2.shape == (3,)
    print(f"  ✓ Context vector updates working")


def main():
    """Run all tests."""
    print("=" * 80)
    print("MRN Implementation Tests")
    print("=" * 80)
    
    np.random.seed(42)
    
    try:
        test_matching()
        test_memory()
        test_basic_components()
        test_sequence_generation()
        test_training()
        
        print("\n" + "=" * 80)
        print("All tests passed! ✓")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
