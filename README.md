# MRN Training on Mathlib

## Overview
This project implements a complete **Modular Reasoning Network (MRN)** system based on exact matching in modular space Z_m^n. The MRN uses discrete mathematics and modular arithmetic to create an efficient, interpretable sequence generation system with infinite context capability through dual-channel memory.

## Key Features
* **Exact matching** in modular space for precise symbol relationships
* **Two-step derivation** process with intermediate vector generation
* **Dual-channel memory**: history window (short-term) + recursive context vector (long-term)
* **Parallel derivation** from multiple history states
* **Train vocabulary, not model**: fixed computation graph with trainable embeddings only
* **High interpretability**: all decisions are traceable through matching scores
* **Parameter efficient**: O(N×n) trainable parameters vs. O(d²·L) in transformers

## Mathematical Foundation
The system is based on the complete mathematical theory described in the included documentation, implementing:
- Symbol embeddings in modular space Z_m^n
- Exact matching function: match(q, x) = Σ 1(q_i = x_i)
- Two-step derivation with path scoring
- Sequence generation with softmax probability distribution
- Training with NLL, contrastive, and regularization losses

## Architecture Components
1. **ModularSpace**: Operations in Z_m^n (discretization, addition, matrix multiplication)
2. **ExactMatcher**: Exact matching functions (match, best, top_m)
3. **SymbolEmbedding**: Trainable embedding matrix E ∈ R^{N×n}
4. **TwoStepDerivation**: Two-step derivation with intermediate matching
5. **MRNGenerator**: Complete sequence generation with dual-channel memory
6. **MRNTrainer**: Training with multiple loss functions and teacher forcing

## Architecture Components
1. **ModularSpace**: Operations in Z_m^n (discretization, addition, matrix multiplication)
2. **ExactMatcher**: Exact matching functions (match, best, top_m)
3. **SymbolEmbedding**: Trainable embedding matrix E ∈ R^{N×n}
4. **TwoStepDerivation**: Two-step derivation with intermediate matching
5. **MRNGenerator**: Complete sequence generation with dual-channel memory
6. **MRNTrainer**: Training with multiple loss functions and teacher forcing

## Installation
To get started with this project, clone the repository and install the necessary dependencies:

```bash
# Clone the repository
git clone https://github.com/hzc1234520/mathlib-mrn-training.git
cd mathlib-mrn-training

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage
```python
from mrn import MRN

# Define vocabulary
vocab = ['a', 'b', 'c', 'd', 'e']

# Create MRN instance
mrn = MRN(
    vocab=vocab,
    modulus=16,           # Modulus m for Z_m space
    embedding_dim=32,     # Embedding dimension n
    history_window_size=128,  # L: history window size
    parallel_window_size=3    # K: parallel derivation window
)

# Initialize with a sequence
mrn.initialize(['a', 'b', 'c'])

# Generate next symbol
next_symbol, metadata = mrn.generate_next(deterministic=True)
print(f"Next: {next_symbol}")

# Generate a sequence
sequence = mrn.generate_sequence(10)
print(f"Generated: {sequence}")
```

### Training
```python
# Prepare training data: (initial_context, target_sequence) pairs
training_data = [
    (['a', 'b'], ['c', 'd']),
    (['b', 'c'], ['d', 'e']),
    # ... more examples
]

# Train the model
history = mrn.train(
    training_data,
    num_epochs=10,
    curriculum_stage='early',  # or 'middle', 'late'
    verbose=True
)
```

## Examples
Run the example script to see the MRN in action:

```bash
python example.py
```

This demonstrates:
- Basic initialization and generation
- Training on simple patterns
- State inspection and debugging

## Testing
Run the comprehensive test suite:

```bash
python test_mrn.py
```

Tests cover:
- Modular space operations
- Exact matching functions
- Symbol embedding system
- Two-step derivation
- Sequence generation
- Loss functions
- Full integration tests

## System Parameters

### Hyperparameters
- **modulus (m)**: Modulus for Z_m space (default: 16, recommended: 8-32)
- **embedding_dim (n)**: Embedding dimension (default: 32, recommended: 16-64)
- **history_window_size (L)**: Size of history window (default: 128, can scale to 65536)
- **parallel_window_size (K)**: Number of parallel derivations (default: 3, range: 1-5)
- **alpha, beta**: Derivation weights (default: 0.5 each)
- **temperature (τ)**: Softmax temperature (default: 1.0)

### Training Parameters
- **learning_rate**: Learning rate for SGD (default: 0.01)
- **lambda_contrastive**: Weight for contrastive loss (default: 1.0)
- **lambda_reg**: L2 regularization weight (default: 1e-5)
- **margin**: Margin for contrastive loss (default: 1.0)

### Curriculum Learning Stages
- **Early**: K=1, L=16 (simple patterns)
- **Middle**: K=3, L=128 (moderate complexity)
- **Late**: K=5, L=1024 (full capacity)

## Implementation Details

### Core Modules
- `core/modular_space.py`: Modular arithmetic operations
- `core/matching.py`: Exact matching functions
- `core/embedding.py`: Symbol embedding system with STE
- `core/derivation.py`: Two-step derivation and scoring
- `core/sequence_generator.py`: Sequence generation with memory
- `core/training.py`: Loss functions and training loop
- `mrn.py`: Main MRN class with unified interface

### Key Algorithms
1. **Discretization**: x_v = round(e_v) mod m with straight-through estimator
2. **Matching**: match(q, x) = Σ_i 1(q_i = x_i) - exact dimension counting
3. **Derivation**: z₁ = (x_t + h) mod m → x_v → z₂ = (x_v + h) mod m → x_w
4. **Aggregation**: S_j = (1/K) Σ s·1(x_w = x_j) - average scores across paths
5. **Sampling**: P(x_j) = softmax(S/τ) - temperature-controlled distribution

## Theoretical Properties
- **Expressiveness**: Can represent m^n distinct states
- **Parameter efficiency**: O(N×n) trainable parameters
- **Context capacity**: Theoretically infinite through dual-channel memory
- **Interpretability**: All decisions traceable through match scores
- **Anti-solidification**: Multiple diversity mechanisms prevent repetition

## Contributing
Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on contributing to this project.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
Special thanks to the contributors of the Mathlib project for providing the foundational tools and resources.