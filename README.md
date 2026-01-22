# MRN Training on Mathlib

## 概述 (Overview)

Complete implementation of **Modular Reasoning Network (MRN)** - a novel sequence generation system based on exact matching in modular space. This project implements the complete mathematical framework described in the theoretical foundation, providing a proof-based AI system for mathematical reasoning.

## 核心特性 (Key Features)

* **精确匹配机制** - Exact matching in discrete modular space instead of continuous distances
* **两步推导** - Two-step derivation with intermediate matching for creativity
* **双通道记忆** - Dual-channel memory (history window + context vector) for infinite context
* **仅训练词汇** - Train vocabulary embeddings only, not model parameters
* **高度可解释** - Fully traceable decision paths and derivation graphs
* **参数高效** - O(N×n) parameters vs O(d²) for neural networks

## 数学原理 (Mathematical Foundation)

### Symbol Embeddings in Modular Space
- V = {v₁, v₂, ..., vₙ} symbol vocabulary
- Each symbol v has trainable embedding eᵥ ∈ ℝⁿ
- Discretized to modular space: xᵥ = round(eᵥ) mod m ∈ Zₘⁿ

### Exact Matching Function
```
match(q, x) = Σᵢ₌₁ⁿ 1(qᵢ = xᵢ)
```
Counts number of exactly matching dimensions (0 to n).

### Two-Step Derivation
1. **Intermediate**: z₁ = (xₜ + h) mod m → xᵥ = best_match(z₁)
2. **Final**: z₂ = (xᵥ + h) mod m → xw = best_match(z₂)
3. **Score**: α·m₁ + β·m₂

### Memory System
- **History Window**: Hₜ ∈ Zₘⁿˣᴸ stores last L states
- **Context Vector**: cₜ₊₁ = (A·cₜ + B·xₜ₊₁) mod m for long-term memory

See [MRN_DOCS.md](MRN_DOCS.md) for complete mathematical details.

## 安装 (Installation)

```bash
# Clone repository
git clone https://github.com/hzc1234520/mathlib-mrn-training.git
cd mathlib-mrn-training

# Install dependencies
pip install -r requirements.txt
```

## 快速开始 (Quick Start)

### 1. Run Tests
```bash
python test_mrn.py
```

### 2. Train Model
```bash
# Basic training with sample data
python train.py --output-dir outputs

# Advanced training
python train.py \
  --embedding-dim 64 \
  --modulus 97 \
  --history-size 128 \
  --parallel-window 5 \
  --num-epochs 100 \
  --output-dir outputs/my_model
```

### 3. Generate Sequences
```bash
# Sample generation
python inference.py \
  --model-dir outputs \
  --prompt "<START>,1,+,1" \
  --length 10 \
  --num-samples 5

# Deterministic generation
python inference.py \
  --model-dir outputs \
  --prompt "theorem,:" \
  --length 20 \
  --deterministic

# Beam search
python inference.py \
  --model-dir outputs \
  --prompt "proof,:" \
  --length 15 \
  --beam-search \
  --beam-width 5
```

## 架构 (Architecture)

```
mathlib-mrn-training/
├── core/                      # Core MRN components
│   ├── symbol_embedding.py    # Symbol embeddings (Section 1.1)
│   ├── modular_space.py       # Modular arithmetic operations
│   ├── matching.py            # Exact matching functions (Section 2)
│   ├── memory.py              # History + context memory (Section 1.2-1.3)
│   ├── derivation.py          # Two-step derivation (Section 3)
│   ├── sequence_generator.py  # Sequence generation (Section 4)
│   └── training.py            # Training framework (Section 5)
├── train.py                   # Main training script
├── inference.py               # Inference/generation script
├── test_mrn.py               # Comprehensive tests
├── MRN_DOCS.md               # Complete documentation
└── requirements.txt           # Dependencies
```

## 性能特点 (Performance)

| Feature | Value |
|---------|-------|
| **Complexity** | O(K·N·n) per step |
| **Parameters** | N×n (embeddings only) |
| **Context** | Theoretically infinite |
| **Hardware** | Integer ops, ASIC-friendly |
| **Interpretability** | Full path tracing |

## 与其他模型对比 (Model Comparison)

| Model | Context | Parameters | Trainable | Interpretability |
|-------|---------|------------|-----------|------------------|
| **MRN** | Infinite | O(N·n) | Embeddings only | High |
| Transformer | Fixed window | O(d²·L) | All layers | Low |
| RNN/LSTM | Infinite | O(d²) | All layers | Low |

## 使用示例 (Examples)

### Programmatic Usage

```python
from core import (
    SymbolEmbedding, ModularSpace, MemoryManager,
    SequenceGenerator, MRNTrainer
)

# Initialize
vocabulary = ['0', '1', '+', '=', '<START>', '<END>']
embedding = SymbolEmbedding(vocabulary, embedding_dim=32, modulus=97)
mod_space = ModularSpace(modulus=97, dimension=32)
memory = MemoryManager(history_size=64, dimension=32, modulus=97)

# Create generator
generator = SequenceGenerator(
    symbol_embedding=embedding,
    modular_space=mod_space,
    memory_manager=memory,
    parallel_window_size=3
)

# Generate
generated, _ = generator.generate_sequence(
    initial_sequence=['<START>', '1', '+'],
    length=5
)
print(' '.join(generated))

# Train
trainer = MRNTrainer(generator)
sequences = [['<START>', '1', '+', '1', '=', '2', '<END>']]
stats = trainer.train_epoch(sequences)
```

## 配置建议 (Recommended Configurations)

**Small** (quick experiments):
```
embedding_dim=16, modulus=31, history_size=32, K=2
→ ~500-1000 parameters
```

**Medium** (balanced):
```
embedding_dim=32, modulus=97, history_size=64, K=3
→ ~2000-5000 parameters
```

**Large** (high capacity):
```
embedding_dim=64, modulus=251, history_size=128, K=5
→ ~10000-50000 parameters
```

## 理论基础 (Theoretical Foundation)

This implementation is based on the complete mathematical framework covering:

1. **基础定义** - Symbol tables, embedding spaces, modular arithmetic
2. **精确匹配** - Exact matching functions with mathematical properties
3. **推导图** - Two-step derivation with scoring mechanisms
4. **序列生成** - Complete generation algorithm with state updates
5. **训练原理** - Loss functions, continuous relaxation, curriculum learning
6. **理论性质** - Expressiveness, memory capacity, anti-crystallization

See the problem statement and [MRN_DOCS.md](MRN_DOCS.md) for detailed proofs and derivations.

## 测试 (Testing)

Comprehensive test suite covering:
- ✓ Symbol embedding and discretization
- ✓ Exact matching functions (symmetry, bounds, decomposition)
- ✓ Memory management (history window, context vector)
- ✓ Two-step derivation paths
- ✓ Sequence generation algorithm
- ✓ Training with teacher forcing

Run: `python test_mrn.py`

## 贡献 (Contributing)

Contributions welcome! Areas for improvement:
- Additional training algorithms (RL, meta-learning)
- More sophisticated curriculum strategies
- Integration with formal theorem provers (Lean, Coq)
- Benchmark datasets for mathematical reasoning
- Performance optimizations (C++/Rust implementations)

## 许可 (License)

MIT License - see LICENSE file for details

## 致谢 (Acknowledgements)

Special thanks to:
- The Mathlib project for foundational mathematical tools
- The symbolic reasoning and formal verification communities
- Contributors to modular arithmetic and discrete mathematics research

## 引用 (Citation)

```bibtex
@software{mrn2026,
  title={Modular Reasoning Network: Complete Implementation},
  author={MRN Team},
  year={2026},
  url={https://github.com/hzc1234520/mathlib-mrn-training},
  note={Based on exact matching in modular space}
}
```

---

**Status**: ✅ Complete implementation with all components functional

**Last Updated**: 2026-01-22