"""
Unit tests for MRN system components

This module tests all core components of the MRN system:
- Modular space operations
- Exact matching functions
- Two-step derivation
- Sequence generation
- Training components
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    ModularSpace,
    ExactMatcher,
    SymbolEmbedding,
    TwoStepDerivation,
    MRNGenerator,
    LossFunction
)


def test_modular_space():
    """Test modular space operations"""
    print("Testing ModularSpace...")
    
    m = 5
    n = 4
    space = ModularSpace(m, n)
    
    # Test discretization
    vec = np.array([1.4, 2.6, -0.3, 4.8])
    discrete = space.discretize(vec)
    assert discrete.shape == (n,), "Discrete vector shape mismatch"
    assert np.all(discrete >= 0) and np.all(discrete < m), "Values out of bounds"
    expected = np.array([1, 3, 0, 0])  # round(1.4, 2.6, -0.3, 4.8) = (1, 3, 0, 5), then mod 5 = (1, 3, 0, 0)
    assert np.array_equal(discrete, expected), f"Discretization failed: {discrete} != {expected}"
    
    # Test addition
    a = np.array([1, 2, 3, 4])
    b = np.array([2, 3, 4, 1])
    result = space.add(a, b)
    expected = np.array([3, 0, 2, 0])  # (1+2, 2+3, 3+4, 4+1) mod 5
    assert np.array_equal(result, expected), f"Addition failed: {result} != {expected}"
    
    # Test matrix multiply
    A = np.array([[1, 2], [3, 4]])
    v = np.array([1, 2])
    space2 = ModularSpace(m, 2)
    result = space2.matrix_multiply(A, v)
    expected = np.array([0, 1])  # (1*1+2*2, 3*1+4*2) = (5, 11) mod 5 = (0, 1)
    assert np.array_equal(result, expected), f"Matrix multiply failed: {result} != {expected}"
    
    print("  ✓ All ModularSpace tests passed")


def test_exact_matcher():
    """Test exact matching functions"""
    print("Testing ExactMatcher...")
    
    matcher = ExactMatcher(modulus=5, dimension=4)
    
    # Test match function
    q = np.array([1, 2, 3, 4])
    x1 = np.array([1, 2, 3, 4])
    x2 = np.array([1, 2, 0, 0])
    x3 = np.array([0, 0, 0, 0])
    
    assert matcher.match(q, x1) == 4, "Perfect match should be 4"
    assert matcher.match(q, x2) == 2, "Partial match should be 2"
    assert matcher.match(q, x3) == 0, "No match should be 0"
    
    # Test best matching
    candidates = [x1, x2, x3]
    best, score = matcher.best(q, candidates, random_on_tie=False)
    assert np.array_equal(best, x1), "Best should be x1"
    assert score == 4, "Best score should be 4"
    
    # Test top_m
    top_results = matcher.top_m(q, candidates, 2)
    assert len(top_results) == 2, "Should return 2 results"
    assert top_results[0][1] >= top_results[1][1], "Should be sorted descending"
    
    print("  ✓ All ExactMatcher tests passed")


def test_symbol_embedding():
    """Test symbol embedding system"""
    print("Testing SymbolEmbedding...")
    
    vocab = ['a', 'b', 'c']
    n = 8
    m = 5
    space = ModularSpace(m, n)
    embedding = SymbolEmbedding(vocab, n, space)
    
    # Test initialization
    assert embedding.vocab_size == 3, "Vocab size mismatch"
    assert embedding.embeddings.shape == (3, 8), "Embedding matrix shape mismatch"
    
    # Test symbol to vector mapping
    vec_a = embedding.get_discrete_vector('a')
    assert vec_a.shape == (n,), "Discrete vector shape mismatch"
    assert np.all(vec_a >= 0) and np.all(vec_a < m), "Values out of bounds"
    
    # Test index mapping
    idx_a = embedding.get_index('a')
    assert embedding.get_symbol(idx_a) == 'a', "Index mapping inconsistent"
    
    # Test all discrete vectors
    all_vecs = embedding.get_all_discrete_vectors()
    assert len(all_vecs) == 3, "Should return all vectors"
    
    print("  ✓ All SymbolEmbedding tests passed")


def test_two_step_derivation():
    """Test two-step derivation process"""
    print("Testing TwoStepDerivation...")
    
    m = 5
    n = 4
    matcher = ExactMatcher(m, n)
    derivation = TwoStepDerivation(matcher, alpha=0.5, beta=0.5)
    
    # Create test vectors
    x_t = np.array([1, 2, 3, 4])
    h = np.array([1, 1, 1, 1])
    candidates = [
        np.array([2, 3, 4, 0]),  # (x_t + h) mod 5
        np.array([0, 0, 0, 0]),
        np.array([1, 1, 1, 1])
    ]
    
    # Test derivation
    x_w, score, metadata = derivation.derive(x_t, h, candidates)
    
    assert any(np.array_equal(x_w, c) for c in candidates), "Output should be one of candidates"
    assert score >= 0, "Score should be non-negative"
    assert 'm1' in metadata and 'm2' in metadata, "Metadata should contain match scores"
    assert metadata['m1'] >= 0 and metadata['m1'] <= n, "m1 out of range"
    assert metadata['m2'] >= 0 and metadata['m2'] <= n, "m2 out of range"
    
    # Test batch derivation
    histories = [h, np.array([2, 2, 2, 2])]
    results = derivation.derive_batch(x_t, histories, candidates)
    assert len(results) == 2, "Should return results for all histories"
    
    print("  ✓ All TwoStepDerivation tests passed")


def test_mrn_generator():
    """Test MRN sequence generator"""
    print("Testing MRNGenerator...")
    
    vocab = ['a', 'b', 'c']
    n = 8
    m = 4
    space = ModularSpace(m, n)
    embedding = SymbolEmbedding(vocab, n, space)
    matcher = ExactMatcher(m, n)
    derivation = TwoStepDerivation(matcher, alpha=0.5, beta=0.5)
    
    generator = MRNGenerator(
        embedding, derivation,
        history_window_size=16,
        parallel_window_size=2,
        temperature=1.0
    )
    
    # Test initialization
    generator.initialize(['a', 'b'])
    assert generator.current_state is not None, "Generator should be initialized"
    
    # Test generation
    next_symbol, metadata = generator.generate_next(deterministic=True)
    assert next_symbol in vocab, "Generated symbol should be in vocab"
    assert 'scores' in metadata, "Metadata should contain scores"
    assert 'probabilities' in metadata, "Metadata should contain probabilities"
    
    # Test sequence generation
    generator.initialize(['a', 'b'])
    sequence = generator.generate_sequence(5, deterministic=False)
    assert len(sequence) == 5, "Should generate 5 symbols"
    assert all(s in vocab for s in sequence), "All symbols should be in vocab"
    
    print("  ✓ All MRNGenerator tests passed")


def test_loss_functions():
    """Test loss functions"""
    print("Testing LossFunction...")
    
    # Test NLL loss
    probs = np.array([0.1, 0.3, 0.5, 0.1])
    target_idx = 2
    nll = LossFunction.nll_loss(probs, target_idx)
    assert nll > 0, "NLL should be positive"
    assert np.isfinite(nll), "NLL should be finite"
    expected_nll = -np.log(0.5)
    assert abs(nll - expected_nll) < 1e-6, f"NLL mismatch: {nll} != {expected_nll}"
    
    # Test contrastive loss
    scores = np.array([1.0, 2.0, 3.0, 1.5])
    target_idx = 2
    contrastive = LossFunction.contrastive_loss(scores, target_idx, margin=1.0)
    assert contrastive >= 0, "Contrastive loss should be non-negative"
    # s_positive = 3.0, max_negative = 2.0, loss = max(0, 1.0 - 3.0 + 2.0) = 0
    assert contrastive == 0.0, f"Contrastive loss should be 0, got {contrastive}"
    
    # Test case where loss is positive
    scores2 = np.array([3.0, 2.0, 1.0, 1.5])
    contrastive2 = LossFunction.contrastive_loss(scores2, target_idx, margin=1.0)
    # s_positive = 1.0, max_negative = 3.0, loss = max(0, 1.0 - 1.0 + 3.0) = 3.0
    assert contrastive2 == 3.0, f"Contrastive loss should be 3.0, got {contrastive2}"
    
    print("  ✓ All LossFunction tests passed")


def test_integration():
    """Integration test with full MRN"""
    print("Testing full MRN integration...")
    
    from mrn import MRN
    
    vocab = ['x', 'y', 'z']
    mrn = MRN(vocab, modulus=4, embedding_dim=8, 
              history_window_size=16, parallel_window_size=2)
    
    # Test initialization and generation
    mrn.initialize(['x', 'y'])
    next_sym, _ = mrn.generate_next()
    assert next_sym in vocab, "Generated symbol should be in vocab"
    
    # Test sequence generation
    mrn.initialize(['x'])
    seq = mrn.generate_sequence(3)
    assert len(seq) == 3, "Should generate 3 symbols"
    
    # Test training step
    losses = mrn.train_step(['x'], ['y'], teacher_forcing=True)
    assert 'total' in losses, "Should return total loss"
    assert losses['total'] >= 0, "Loss should be non-negative"
    
    # Test state inspection
    state = mrn.get_state()
    assert state['initialized'] == True, "Should be initialized"
    
    print("  ✓ All integration tests passed")


def run_all_tests():
    """Run all unit tests"""
    print("\n" + "=" * 60)
    print("Running MRN Unit Tests")
    print("=" * 60 + "\n")
    
    try:
        test_modular_space()
        test_exact_matcher()
        test_symbol_embedding()
        test_two_step_derivation()
        test_mrn_generator()
        test_loss_functions()
        test_integration()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60 + "\n")
        return True
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
