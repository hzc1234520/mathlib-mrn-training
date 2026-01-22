"""
Example usage of the MRN system

This script demonstrates how to use the MRN (Modular Reasoning Network)
for sequence generation and training.
"""

from mrn import MRN


def example_basic_usage():
    """Basic example: initialize and generate"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Define vocabulary
    vocab = ['a', 'b', 'c', 'd', 'e']
    
    # Create MRN with small parameters for demo
    mrn = MRN(
        vocab=vocab,
        modulus=8,
        embedding_dim=16,
        history_window_size=32,
        parallel_window_size=2,
        temperature=1.0
    )
    
    print(f"Created {mrn}")
    print(f"Vocabulary: {vocab}")
    
    # Initialize with a sequence
    initial_seq = ['a', 'b', 'c']
    print(f"\nInitializing with: {initial_seq}")
    mrn.initialize(initial_seq)
    
    # Generate next symbol
    next_symbol, metadata = mrn.generate_next(deterministic=True)
    print(f"Next symbol (deterministic): {next_symbol}")
    print(f"Top 3 probabilities: {sorted(enumerate(metadata['probabilities']), key=lambda x: x[1], reverse=True)[:3]}")
    
    # Generate a sequence
    print("\nGenerating sequence of length 10:")
    mrn.initialize(['a', 'b'])  # Reset
    sequence = mrn.generate_sequence(10, deterministic=False)
    print(f"Generated: {sequence}")


def example_training():
    """Example: train MRN on simple patterns"""
    print("\n" + "=" * 60)
    print("Example 2: Training")
    print("=" * 60)
    
    # Define vocabulary
    vocab = ['a', 'b', 'c', 'd', 'e']
    
    # Create MRN
    mrn = MRN(
        vocab=vocab,
        modulus=8,
        embedding_dim=16,
        history_window_size=16,
        parallel_window_size=2,
        learning_rate=0.01,
        temperature=0.5
    )
    
    # Create simple training data: learn pattern "ab" -> "c", "bc" -> "d"
    training_data = [
        (['a', 'b'], ['c']),
        (['b', 'c'], ['d']),
        (['a', 'b'], ['c']),
        (['b', 'c'], ['d']),
    ]
    
    print("Training data patterns:")
    for initial, target in training_data:
        print(f"  {initial} -> {target}")
    
    # Train with curriculum (early stage)
    print("\nTraining for 5 epochs (early curriculum stage)...")
    history = mrn.train(
        training_data,
        num_epochs=5,
        curriculum_stage='early',
        verbose=True
    )
    
    # Test generation after training
    print("\nTesting after training:")
    mrn.initialize(['a', 'b'])
    next_sym, _ = mrn.generate_next(deterministic=True)
    print(f"  After ['a', 'b']: predicted '{next_sym}' (expected 'c')")
    
    mrn.initialize(['b', 'c'])
    next_sym, _ = mrn.generate_next(deterministic=True)
    print(f"  After ['b', 'c']: predicted '{next_sym}' (expected 'd')")


def example_state_inspection():
    """Example: inspect internal state"""
    print("\n" + "=" * 60)
    print("Example 3: State Inspection")
    print("=" * 60)
    
    vocab = ['x', 'y', 'z']
    mrn = MRN(vocab, modulus=4, embedding_dim=8)
    
    print("System configuration:")
    state = mrn.get_state()
    for key, value in state.items():
        if key != 'generator_state':
            print(f"  {key}: {value}")
    
    print("\nDiscrete vectors for vocabulary:")
    discrete_vecs = mrn.get_discrete_vectors()
    for symbol, vec in zip(vocab, discrete_vecs):
        print(f"  x_{symbol} = {vec}")
    
    print("\nEmbedding matrix shape:", mrn.get_embedding_matrix().shape)


def main():
    """Run all examples"""
    print("\nMRN (Modular Reasoning Network) Examples")
    print("=" * 60)
    
    example_basic_usage()
    example_training()
    example_state_inspection()
    
    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
