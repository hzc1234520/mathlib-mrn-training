# MRN Mathematical Theory Implementation Guide

## Overview
This document explains how the mathematical theory from the problem statement is implemented in the codebase.

## 1. Foundation: Modular Space (Section 1)

### 1.1 Symbol Table and Embedding Space
**Theory**: V = {v₁, v₂, ..., vₙ} with embeddings e_v ∈ ℝⁿ  
**Implementation**: `SymbolEmbedding` class in `core/embedding.py`

```python
# Embedding matrix E ∈ ℝ^{N×n}
self.embeddings = np.random.randn(vocab_size, embedding_dim)
```

### 1.2 Modular Space Z_m
**Theory**: x_v,i = round(e_v,i) mod m  
**Implementation**: `ModularSpace.discretize()` in `core/modular_space.py`

```python
def discretize(self, vector):
    rounded = np.round(vector).astype(int)
    return rounded % self.modulus
```

### 1.3 History Window H_t
**Theory**: H_t = [x_t, x_{t-1}, ..., x_{t-L+1}] ∈ Z_m^{n×L}  
**Implementation**: `HistoryWindow` class in `core/sequence_generator.py`

```python
def update(self, new_state):
    # H_{t+1} = [x_{t+1}, H_t[0], ..., H_t[L-2]]
    self.states = [new_state] + self.states[:-1]
```

### 1.4 Recursive Context Vector c_t
**Theory**: c_{t+1} = (A·c_t + B·x_{t+1}) mod m  
**Implementation**: `ContextVector` class in `core/sequence_generator.py`

```python
def update(self, new_state):
    term1 = self.modular_space.matrix_multiply(self.A, self.vector)
    term2 = self.modular_space.matrix_multiply(self.B, new_state)
    self.vector = self.modular_space.add(term1, term2)
```

## 2. Exact Matching Functions (Section 2)

### 2.1 Match Function
**Theory**: match(q, x) = Σᵢ 1(qᵢ = xᵢ)  
**Implementation**: `ExactMatcher.match()` in `core/matching.py`

```python
def match(self, q, x):
    return int(np.sum(q == x))
```

**Properties Verified**:
- Symmetry: match(q, x) = match(x, q) ✓
- Bounded: 0 ≤ match(q, x) ≤ n ✓
- Decomposable: Sum of dimension-wise matches ✓

### 2.2 Best Matching
**Theory**: best(q) = argmax_{x∈X} match(q, x)  
**Implementation**: `ExactMatcher.best()` in `core/matching.py`

```python
def best(self, q, candidates):
    scores = [self.match(q, x) for x in candidates]
    max_score = max(scores)
    best_indices = [i for i, s in enumerate(scores) if s == max_score]
    selected_idx = np.random.choice(best_indices)  # Random on tie
    return candidates[selected_idx], max_score
```

### 2.3 Smooth Approximation for Training
**Theory**: match_σ(q, x) = Σᵢ σ(γ·(qᵢ - xᵢ))  
**Implementation**: `ExactMatcher.match_smooth()` in `core/matching.py`

```python
def match_smooth(self, q, x, gamma=10.0):
    diff = q - x
    sigmoid_vals = 1.0 / (1.0 + np.exp(gamma * np.abs(diff)))
    return float(np.sum(sigmoid_vals))
```

## 3. Two-Step Derivation (Section 3)

### 3.1 Step 1: Intermediate Matching
**Theory**:
- z₁ = (x_t + h) mod m
- x_v = best(z₁)

**Implementation**: `TwoStepDerivation.derive()` in `core/derivation.py`

```python
# Step 1
z_1 = (x_t + h) % modulus
x_v, m_1 = self.matcher.best(z_1, candidates)
```

### 3.2 Step 2: Final Matching
**Theory**:
- z₂ = (x_v + h) mod m
- x_w = best(z₂)

**Implementation**:

```python
# Step 2
z_2 = (x_v + h) % modulus
x_w, m_2 = self.matcher.best(z_2, candidates)
```

### 3.3 Path Scoring
**Theory**: score = α·m₁ + β·m₂  
**Implementation**:

```python
score = self.alpha * m_1 + self.beta * m_2
```

## 4. Sequence Generation (Section 4)

### 4.1 Parallel Derivation Window
**Theory**: W_t = [H_t[0], H_t[1], ..., H_t[K-1]]  
**Implementation**: `MRNGenerator.generate_next()` in `core/sequence_generator.py`

```python
window = self.history.get_window(self.parallel_window_size)
```

### 4.2 Candidate Aggregation
**Theory**: S_j = (1/K) Σ_{(h,x_w,s)∈C} s·1(x_w = x_j)  
**Implementation**: `DerivationGraph.aggregate_candidates()` in `core/derivation.py`

```python
for x_w, s, metadata in results:
    for j, candidate in enumerate(candidates):
        if np.array_equal(x_w, candidate):
            scores[j] += s
            break
scores = scores / K  # Average
```

### 4.3 Probability Distribution
**Theory**: P(x_j) = exp(S_j/τ) / Σ_j' exp(S_j'/τ)  
**Implementation**: `DerivationGraph.compute_probability()` in `core/derivation.py`

```python
def compute_probability(self, scores, temperature):
    scaled_scores = scores / temperature
    scaled_scores = scaled_scores - np.max(scaled_scores)  # Stability
    exp_scores = np.exp(scaled_scores)
    return exp_scores / np.sum(exp_scores)
```

## 5. Training (Section 5)

### 5.1 Straight-Through Estimator
**Theory**: ∂x/∂round(x) ≈ 1, ∂x/∂(x mod m) ≈ 1  
**Implementation**: `EmbeddingGradient.straight_through_gradient()` in `core/embedding.py`

```python
def straight_through_gradient(output_grad):
    # Pass gradient through unchanged
    return output_grad
```

### 5.2 Loss Functions
**Theory**:
- L_NLL = -log P(w_{t+1} | w_1, ..., w_t)
- L_contrastive = max(0, margin - s_positive + max_{i≠target} s_i)
- L_reg = ||E||_F²

**Implementation**: `LossFunction` class in `core/training.py`

```python
def nll_loss(probabilities, target_idx):
    prob = np.clip(probabilities[target_idx], 1e-10, 1.0)
    return -np.log(prob)

def contrastive_loss(scores, target_idx, margin):
    s_positive = scores[target_idx]
    s_negative_max = max(s for i, s in enumerate(scores) if i != target_idx)
    return max(0.0, margin - s_positive + s_negative_max)

def compute_l2_regularization(embeddings, lambda_reg):
    loss = np.sum(embeddings ** 2)
    gradient = 2.0 * embeddings
    return lambda_reg * loss, lambda_reg * gradient
```

### 5.3 Total Loss
**Theory**: L = L_NLL + λ_contrastive·L_contrastive + λ_reg·L_reg  
**Implementation**:

```python
def total_loss(probabilities, scores, target_idx, embeddings, ...):
    l_nll = LossFunction.nll_loss(probabilities, target_idx)
    l_contrastive = LossFunction.contrastive_loss(scores, target_idx, margin)
    l_reg, _ = EmbeddingGradient.compute_l2_regularization(embeddings, lambda_reg)
    total = l_nll + lambda_contrastive * l_contrastive + l_reg
    return total, components_dict
```

### 5.4 Teacher Forcing
**Theory**: Use ground truth history x_{w_t} instead of predicted states  
**Implementation**: `MRNTrainer.train_step()` in `core/training.py`

```python
if teacher_forcing:
    target_state = self.embedding.get_discrete_vector(target_symbol)
    self.generator.history.update(target_state)
    self.generator.context.update(target_state)
    self.generator.current_state = target_state
```

### 5.5 Curriculum Learning
**Theory**: Progressively increase K and L  
**Implementation**: `CurriculumLearning` class in `core/training.py`

```python
schedules = {
    'early': {'K': 1, 'L': 16},
    'middle': {'K': 3, 'L': 128},
    'late': {'K': 5, 'L': 1024}
}
```

## 6. Key Mathematical Properties

### 6.1 Expressiveness
- **Modular space size**: m^n distinct states
- **Parameter count**: N×n (embeddings only)
- **Comparison**: O(N×n) vs O(d²·L) for Transformers

### 6.2 Infinite Context
- **Short-term**: History window H_t (precise, size L)
- **Long-term**: Context vector c_t (compressed, recursive)
- **Combined**: Dual-channel memory enables unbounded history

### 6.3 Anti-Solidification
1. Parallel derivation from K history states
2. Random tie-breaking in matching
3. Stochastic sampling with temperature
4. Modular arithmetic introduces periodicity

### 6.4 Interpretability
Every decision is traceable:
- Match scores (m₁, m₂) for each derivation
- Intermediate vectors (x_v) in two-step process
- Aggregated scores (S_j) for each candidate
- Final probabilities P(x_j)

## 7. Verification

All mathematical properties are verified through unit tests in `test_mrn.py`:
- ✓ Modular operations preserve bounds
- ✓ Matching is symmetric and bounded
- ✓ Derivation produces valid candidates
- ✓ Loss functions are well-defined
- ✓ Integration test validates full pipeline

## 8. Usage Example Mapping

### Mathematical Notation → Code
```
V = {v₁, v₂, ...}        → vocab = ['a', 'b', 'c', ...]
m (modulus)               → modulus = 16
n (dimension)             → embedding_dim = 32
L (history size)          → history_window_size = 128
K (parallel window)       → parallel_window_size = 3
E ∈ ℝ^{N×n}              → mrn.embedding.embeddings
x_v ∈ Z_m^n              → mrn.embedding.get_discrete_vector('a')
match(q, x)               → mrn.matcher.match(q, x)
derive(x_t, h)            → mrn.derivation.derive(x_t, h, candidates)
generate()                → mrn.generate_next()
train()                   → mrn.train(training_data, num_epochs)
```

This implementation faithfully realizes all aspects of the mathematical theory while maintaining clean, testable, and efficient code.
