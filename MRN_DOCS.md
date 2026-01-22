# Modular Reasoning Network (MRN) - Complete Implementation

## 概述 (Overview)

This is a complete implementation of the **Modular Reasoning Network (MRN)** based on the mathematical framework described in the problem statement. The system implements a novel approach to sequence generation using exact matching in modular space with trainable symbol embeddings.

## 核心原理 (Core Principles)

### 1. 符号嵌入与模空间 (Symbol Embeddings and Modular Space)

- **Vocabulary**: V = {v₁, v₂, ..., vₙ} - finite symbol set
- **Embeddings**: Each symbol v has a trainable embedding eᵥ ∈ ℝⁿ
- **Modular Space**: Zₘⁿ where vectors are discretized via xᵥ = round(eᵥ) mod m
- **Training Principle**: "Train vocabulary, not model" - only embeddings are trainable

### 2. 精确匹配机制 (Exact Matching)

```
match(q, x) = Σᵢ₌₁ⁿ 1(qᵢ = xᵢ)
```

- Counts number of exactly matching dimensions
- Symmetric, bounded [0, n], and decomposable
- Simpler than continuous distance metrics

### 3. 两步推导 (Two-Step Derivation)

**Step 1**: Intermediate vector
```
z₁ = (xₜ + h) mod m
xᵥ = best_match(z₁)  # score m₁
```

**Step 2**: Final output
```
z₂ = (xᵥ + h) mod m
xw = best_match(z₂)  # score m₂
```

**Scoring**: score(h → xw) = α·m₁ + β·m₂

### 4. 双通道记忆 (Dual-Channel Memory)

- **History Window**: Hₜ = [xₜ, xₜ₋₁, ..., xₜ₋ₗ₊₁] - short-term precise memory
- **Context Vector**: cₜ₊₁ = (A·cₜ + B·xₜ₊₁) mod m - long-term compressed memory
- Combined for theoretically infinite context

### 5. 并行推导 (Parallel Derivation)

- Considers K adjacent history states simultaneously
- Aggregates scores across all derivation paths
- Softmax over aggregated scores for selection

## 架构组件 (Architecture Components)

```
core/
├── symbol_embedding.py    # Symbol embeddings in modular space
├── modular_space.py       # Modular arithmetic operations
├── matching.py            # Exact matching functions
├── memory.py              # History window + context vector
├── derivation.py          # Two-step derivation logic
├── sequence_generator.py  # Complete generation algorithm
└── training.py            # Training framework
```

## 安装 (Installation)

```bash
# Clone repository
git clone https://github.com/hzc1234520/mathlib-mrn-training.git
cd mathlib-mrn-training

# Install dependencies
pip install -r requirements.txt
```

## 使用方法 (Usage)

### 训练模型 (Training)

Basic training:
```bash
python train.py --output-dir outputs
```

Advanced training with custom parameters:
```bash
python train.py \
  --embedding-dim 64 \
  --modulus 97 \
  --history-size 128 \
  --parallel-window 5 \
  --learning-rate 0.01 \
  --num-epochs 200 \
  --temperature 1.0 \
  --use-curriculum \
  --output-dir outputs/my_model
```

Training with custom vocabulary and data:
```bash
python train.py \
  --vocab-file data/vocab.txt \
  --data-file data/sequences.json \
  --output-dir outputs/custom
```

### 推理生成 (Inference)

Generate sequences with trained model:
```bash
python inference.py \
  --model-dir outputs \
  --prompt "<START>,1,+,1" \
  --length 10 \
  --temperature 1.0 \
  --num-samples 5
```

Deterministic generation:
```bash
python inference.py \
  --model-dir outputs \
  --prompt "theorem,:" \
  --length 20 \
  --deterministic
```

Beam search:
```bash
python inference.py \
  --model-dir outputs \
  --prompt "proof,:" \
  --length 15 \
  --beam-search \
  --beam-width 5
```

### 程序化使用 (Programmatic Usage)

```python
from core import (
    SymbolEmbedding, ModularSpace, MemoryManager,
    SequenceGenerator, MRNTrainer
)

# Initialize components
vocabulary = ['a', 'b', 'c', 'd', 'e', '+', '=', '<START>', '<END>']
embedding = SymbolEmbedding(vocabulary, embedding_dim=32, modulus=97)
mod_space = ModularSpace(modulus=97, dimension=32)
memory = MemoryManager(history_size=64, dimension=32, modulus=97)

# Create generator
generator = SequenceGenerator(
    symbol_embedding=embedding,
    modular_space=mod_space,
    memory_manager=memory,
    parallel_window_size=3,
    temperature=1.0
)

# Generate sequence
initial = ['<START>', 'a', 'b']
generated, _ = generator.generate_sequence(
    initial_sequence=initial,
    length=10,
    deterministic=False
)
print("Generated:", ' '.join(initial + generated))

# Train model
trainer = MRNTrainer(generator, learning_rate=0.01)
sequences = [['a', 'b', 'c', 'd'], ['b', 'c', 'd', 'e']]
stats = trainer.train_epoch(sequences)
print("Loss:", stats['total_loss'])
```

## 超参数 (Hyperparameters)

### 架构超参数 (Architecture)

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `embedding_dim` | n | 32 | Embedding dimension |
| `modulus` | m | 97 | Modulus for modular space |
| `history_size` | L | 64 | History window size |
| `parallel_window` | K | 3 | Parallel derivation window |

### 训练超参数 (Training)

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `learning_rate` | η | 0.01 | Learning rate |
| `temperature` | τ | 1.0 | Softmax temperature |
| `lambda_contrastive` | λc | 1.0 | Contrastive loss weight |
| `lambda_reg` | λr | 1e-5 | Regularization weight |
| `alpha` | α | 0.5 | First step weight |
| `beta` | β | 0.5 | Second step weight |

### 推荐配置 (Recommended Configurations)

**Small Model** (for quick experiments):
```
embedding_dim=16, modulus=31, history_size=32, parallel_window=2
Parameters: ~500-1000
```

**Medium Model** (balanced):
```
embedding_dim=32, modulus=97, history_size=64, parallel_window=3
Parameters: ~2000-5000
```

**Large Model** (high capacity):
```
embedding_dim=64, modulus=251, history_size=128, parallel_window=5
Parameters: ~10000-50000
```

## 性能特点 (Performance Characteristics)

### 计算复杂度 (Computational Complexity)

- **Per-step generation**: O(K·N·n)
  - K: parallel window size
  - N: vocabulary size
  - n: embedding dimension

- **Training**: O(T·K·N·n)
  - T: sequence length

### 内存使用 (Memory Usage)

- **Parameters**: N×n (embedding matrix only)
- **State**: L×n (history window) + n (context vector)
- **Extremely parameter-efficient** compared to transformers

### 优势 (Advantages)

1. **Parameter Efficiency**: Only train embeddings, not model
2. **Infinite Context**: Dual-channel memory (history + context)
3. **Interpretability**: Clear derivation paths and matching scores
4. **Diversity**: Multiple mechanisms prevent "固化" (crystallization)
5. **Hardware Friendly**: Integer operations suitable for ASIC

## 理论性质 (Theoretical Properties)

### 表达能力 (Expressiveness)

- **State Space**: mⁿ possible discrete states
- **Parameters**: Only N×n trainable values
- **Universal Approximation**: Can approximate arbitrary functions with sufficient m, n

### 记忆能力 (Memory Capacity)

- **Short-term**: Exact L-step history
- **Long-term**: Recursive compression via context vector
- **Infinite Context**: Theoretical ability to process unlimited history

### 抗固化 (Anti-Crystallization)

Four layers of diversity:
1. Parallel derivation over K states
2. Random tie-breaking in matching
3. Stochastic sampling
4. Modular arithmetic cycles

## 与现有模型对比 (Comparison with Existing Models)

| Feature | Transformer | RNN/LSTM | MRN |
|---------|-------------|----------|-----|
| Context Length | Fixed window | Theoretically infinite | Infinite (dual-channel) |
| Complexity | O(L²·d) | O(L·d²) | O(K·N·n) |
| Parameters | O(d²·L) | O(d²) | O(N·n) |
| Training | All layers | All layers | Embeddings only |
| Interpretability | Low | Low | High |
| Hardware | GPU (parallel) | Sequential | ASIC-friendly (integer) |

## 测试 (Testing)

Run comprehensive tests:
```bash
python test_mrn.py
```

Tests cover:
- Symbol embedding and discretization
- Exact matching functions
- Memory management (history + context)
- Two-step derivation
- Sequence generation
- Training framework

## 示例 (Examples)

See `train.py` and `inference.py` for complete examples of:
- Training on mathematical sequences
- Proof generation
- Symbolic reasoning

## 数学框架 (Mathematical Framework)

This implementation follows the complete mathematical specification provided in the problem statement, including:

- Section 1: Symbol embeddings and modular space
- Section 2: Exact matching functions
- Section 3: Two-step derivation graph
- Section 4: Sequence generation algorithm
- Section 5: Training principles and loss functions
- Section 6: Theoretical properties
- Section 7: Model comparisons

## 引用 (Citation)

If you use this implementation, please cite:

```
@software{mrn2026,
  title={Modular Reasoning Network: Complete Implementation},
  author={MRN Team},
  year={2026},
  url={https://github.com/hzc1234520/mathlib-mrn-training}
}
```

## 许可 (License)

MIT License - see LICENSE file

## 贡献 (Contributing)

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## 致谢 (Acknowledgements)

Special thanks to the mathematical reasoning and symbolic AI communities for inspiration and foundational work.
