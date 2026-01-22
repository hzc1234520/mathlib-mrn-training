#!/usr/bin/env python3
"""
Simple demonstration of MRN capabilities
Shows basic sequence generation and symbolic reasoning
"""

import numpy as np
from core import (
    SymbolEmbedding,
    ModularSpace,
    MemoryManager,
    SequenceGenerator,
    MRNTrainer
)


def demo_basic_generation():
    """Demonstrate basic sequence generation."""
    print("=" * 80)
    print("Demo 1: Basic Sequence Generation")
    print("=" * 80)
    
    # Simple vocabulary
    vocabulary = ['a', 'b', 'c', 'd', 'e', '<START>', '<END>']
    
    # Initialize components
    embedding = SymbolEmbedding(vocabulary, embedding_dim=16, modulus=31, seed=42)
    mod_space = ModularSpace(modulus=31, dimension=16)
    memory = MemoryManager(history_size=32, dimension=16, modulus=31)
    
    generator = SequenceGenerator(
        symbol_embedding=embedding,
        modular_space=mod_space,
        memory_manager=memory,
        parallel_window_size=2,
        temperature=1.0
    )
    
    # Generate
    initial = ['<START>', 'a', 'b']
    generated, _ = generator.generate_sequence(initial, length=5, deterministic=True)
    
    print(f"\nInitial sequence: {' '.join(initial)}")
    print(f"Generated sequence: {' '.join(initial + generated)}")
    print(f"\nParameters: {embedding.get_parameter_count():,}")


def demo_training():
    """Demonstrate training on simple patterns."""
    print("\n" + "=" * 80)
    print("Demo 2: Learning Simple Patterns")
    print("=" * 80)
    
    vocabulary = ['a', 'b', 'c', 'd', 'e']
    
    # Initialize
    embedding = SymbolEmbedding(vocabulary, embedding_dim=8, modulus=13, seed=42)
    mod_space = ModularSpace(modulus=13, dimension=8)
    memory = MemoryManager(history_size=16, dimension=8, modulus=13)
    
    generator = SequenceGenerator(
        symbol_embedding=embedding,
        modular_space=mod_space,
        memory_manager=memory,
        parallel_window_size=2
    )
    
    # Training data: simple repeating patterns
    sequences = [
        ['a', 'b', 'c', 'a', 'b', 'c'],
        ['b', 'c', 'd', 'b', 'c', 'd'],
        ['c', 'd', 'e', 'c', 'd', 'e'],
    ]
    
    trainer = MRNTrainer(generator, learning_rate=0.05)
    
    print("\nTraining on repeating patterns...")
    for epoch in range(10):
        stats = trainer.train_epoch(sequences)
        if epoch % 3 == 0:
            print(f"  Epoch {epoch+1}: Loss = {stats['total_loss']:.6f}")
    
    # Test generation
    print("\nGenerating after training:")
    for test_init in [['a', 'b'], ['b', 'c'], ['c', 'd']]:
        generated, _ = generator.generate_sequence(test_init, length=4, deterministic=True)
        print(f"  {' '.join(test_init)} → {' '.join(test_init + generated)}")


def demo_mathematical_reasoning():
    """Demonstrate mathematical symbolic reasoning."""
    print("\n" + "=" * 80)
    print("Demo 3: Mathematical Symbolic Reasoning")
    print("=" * 80)
    
    # Mathematical vocabulary
    vocabulary = [
        '0', '1', '2', '3', '+', '=', ':',
        'x', 'y', 'let', 'theorem', 'proof',
        '<START>', '<END>'
    ]
    
    embedding = SymbolEmbedding(vocabulary, embedding_dim=24, modulus=47, seed=42)
    mod_space = ModularSpace(modulus=47, dimension=24)
    memory = MemoryManager(history_size=48, dimension=24, modulus=47)
    
    generator = SequenceGenerator(
        symbol_embedding=embedding,
        modular_space=mod_space,
        memory_manager=memory,
        parallel_window_size=3
    )
    
    # Mathematical sequences
    math_sequences = [
        ['<START>', '1', '+', '1', '=', '2', '<END>'],
        ['<START>', '2', '+', '1', '=', '3', '<END>'],
        ['<START>', '0', '+', '1', '=', '1', '<END>'],
        ['<START>', 'let', 'x', '=', '1', '<END>'],
        ['<START>', 'theorem', ':', '1', '=', '1', '<END>'],
    ]
    
    trainer = MRNTrainer(generator, learning_rate=0.02)
    
    print("\nTraining on mathematical statements...")
    for epoch in range(15):
        stats = trainer.train_epoch(math_sequences)
        if epoch % 5 == 0:
            print(f"  Epoch {epoch+1}: Loss = {stats['total_loss']:.6f}")
    
    # Generate mathematical statements
    print("\nGenerating mathematical statements:")
    test_prompts = [
        ['<START>', '1', '+'],
        ['<START>', 'let', 'x'],
        ['<START>', 'theorem', ':']
    ]
    
    for prompt in test_prompts:
        generated, _ = generator.generate_sequence(prompt, length=5, deterministic=True)
        print(f"  {' '.join(prompt)} → {' '.join(generated)}")


def demo_memory_visualization():
    """Demonstrate memory system."""
    print("\n" + "=" * 80)
    print("Demo 4: Memory System Visualization")
    print("=" * 80)
    
    vocabulary = ['a', 'b', 'c']
    
    embedding = SymbolEmbedding(vocabulary, embedding_dim=4, modulus=7, seed=42)
    memory = MemoryManager(history_size=5, dimension=4, modulus=7, context_seed=42)
    
    print("\nAdding sequence to memory: a → b → c → a → b")
    sequence = ['a', 'b', 'c', 'a', 'b']
    
    for i, symbol in enumerate(sequence):
        x = embedding.discretize(symbol)
        memory.update(x)
        
        print(f"\nStep {i+1}: Added '{symbol}'")
        print(f"  State vector: {x}")
        print(f"  History[0]: {memory.get_history_state(0)}")
        print(f"  Context: {memory.get_context()}")
        
        # Show parallel window
        if i >= 1:
            W = memory.get_parallel_window(min(2, i+1))
            print(f"  Parallel window (K=2): shape {W.shape}")


def main():
    """Run all demos."""
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "MRN System Demonstration" + " " * 39 + "║")
    print("║" + " " * 10 + "Modular Reasoning Network Examples" + " " * 34 + "║")
    print("╚" + "=" * 78 + "╝")
    
    np.random.seed(42)
    
    demo_basic_generation()
    demo_training()
    demo_mathematical_reasoning()
    demo_memory_visualization()
    
    print("\n" + "=" * 80)
    print("All demos completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Run 'python train.py' for full training")
    print("  2. Run 'python inference.py --model-dir outputs' for generation")
    print("  3. See MRN_DOCS.md for detailed documentation")


if __name__ == '__main__':
    main()
