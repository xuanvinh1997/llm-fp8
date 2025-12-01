# LLM Optimization for Edge Devices: A Methodology-Focused Survey

## Abstract

This survey provides an in-depth examination of the algorithmic methodologies underlying Large Language Model (LLM) optimization for deployment on edge and consumer devices. Rather than focusing solely on results and benchmarks, we dissect the mathematical foundations, algorithmic procedures, and implementation details of key techniques including quantization (GPTQ, AWQ, QLoRA), pruning (SparseGPT), attention optimization (FlashAttention, PagedAttention), speculative decoding, and parameter-efficient fine-tuning (LoRA). For each technique, we present the core mathematical formulation, step-by-step algorithmic procedures, computational complexity analysis, and implementation considerations.

---

## 1. Quantization Methodologies

### 1.1 Mathematical Foundation of Quantization

Quantization maps high-precision floating-point values to lower-precision representations. For a weight tensor **W**, the fundamental quantization operation is:

```
Q(w) = Δ · round(w/Δ + z)
```

Where:
- Δ (delta) is the scale factor: `Δ = (max(W) - min(W)) / (2^b - 1)`
- z is the zero-point for asymmetric quantization
- b is the target bit-width

The layer-wise quantization objective for LLMs minimizes reconstruction error:

```
argmin_Ŵ ||WX - ŴX||²₂
```

Where W is the original weight matrix, Ŵ is the quantized weight matrix, and X represents layer inputs from calibration data.

### 1.2 GPTQ Algorithm: Optimal Brain Quantization for LLMs

**Theoretical Foundation:**
GPTQ builds on the Optimal Brain Quantization (OBQ) framework, which itself derives from Optimal Brain Surgeon (OBS). The key insight is that quantizing a weight introduces error that can be partially compensated by adjusting remaining unquantized weights.

**The OBQ Update Formula:**
When quantizing weight w_q, the optimal update δ_F for remaining weights F is:

```
δ_F = -w_q · (H_F)⁻¹ · h_Fq / [H⁻¹]_qq
```

Where:
- H = XX^T + λI is the Hessian matrix (with regularization λ)
- h_Fq is the column of H corresponding to the quantized weight
- [H⁻¹]_qq is the diagonal element of the inverse Hessian

**GPTQ Key Innovations:**

1. **Arbitrary Order Quantization:** Unlike OBQ which greedily selects weights minimizing immediate error, GPTQ quantizes all rows in the same column order. Empirically, this produces equivalent results for large models while enabling massive parallelization.

2. **Lazy Batch Updates:** Instead of updating the Hessian after each weight, GPTQ processes blocks of B columns (typically B=128):
   - Quantize B columns using current Hessian information
   - Batch-update remaining columns after processing the block
   - This improves GPU utilization by converting memory-bound operations to compute-bound

3. **Cholesky Reformulation:** For numerical stability, GPTQ uses Cholesky decomposition of the inverse Hessian:
   ```
   H⁻¹ = (Chol(H⁻¹))^T · Chol(H⁻¹)
   ```
   This eliminates accumulating numerical errors during sequential processing.

**Algorithm Pseudocode:**
```
Input: Weight matrix W, inverse Hessian H⁻¹, sparsity/quantization target
Output: Quantized weight matrix Ŵ

1. Compute Cholesky decomposition of H⁻¹
2. For j = 0 to d_col in blocks of size B:
   a. For each column in block:
      - Quantize column j: ŵ_j = quant(w_j)
      - Compute quantization error: δ_j = (w_j - ŵ_j) / [H⁻¹]_jj
      - Update remaining columns in block: W[:,j+1:] -= δ_j · H⁻¹[j,j+1:]
   b. Apply lazy batch update to columns beyond current block
3. Return Ŵ
```

**Complexity:** O(d_col · d_row · d_col) = O(d²_col · d_row) per layer, with the inverse Hessian computation dominating at O(d³_col).

### 1.3 AWQ: Activation-Aware Weight Quantization

**Core Observation:**
Not all weights contribute equally to model outputs. AWQ identifies that approximately 0.1-1% of weight channels—those corresponding to large activation magnitudes—are "salient" and protecting them significantly reduces quantization error.

**Salient Channel Identification:**
Unlike magnitude-based selection, AWQ uses activation statistics:
```
Saliency(channel_i) = mean(|X[:,i]|) · ||W[i,:]||
```
Channels with high saliency scores process more important features.

**Scaling-Based Protection:**
Instead of mixed-precision (which is hardware-inefficient), AWQ scales salient channels before quantization:

For weight w and input activation x, the output y = w · x. If we scale:
- w' = s · w (scale up weight)
- x' = x / s (scale down activation)

The output remains unchanged: y' = w' · x' = (s·w) · (x/s) = w · x = y

**Quantization Error Analysis:**
For a weight w with scale factor s and quantization step Δ:
```
Error(w) = |w - Q(w)| ≤ Δ/2

Error(s·w) scaled back = |w - Q(s·w)/s| ≤ Δ/(2s)
```

By choosing s > 1 for salient channels, the effective quantization error decreases proportionally.

**Optimal Scale Search:**
AWQ searches for optimal per-channel scales using grid search:
```
s* = argmin_s ||Q(s·W)·(X/s) - W·X||²
```

The search space is typically s ∈ {1, 1.25, 1.5, ..., 4} for salient channels.

**Algorithm:**
```
1. Collect activation statistics from calibration data
2. Identify salient channels (top 1% by activation magnitude)
3. For each salient channel i:
   a. Search for optimal scale s_i minimizing reconstruction error
   b. Apply equivalent transformation: W[i,:] *= s_i
4. Fuse inverse scales into adjacent layer (absorbed during inference)
5. Perform group-wise quantization on transformed weights
```

**Key Advantage:** AWQ requires no backpropagation or weight reconstruction, making it faster than GPTQ while achieving comparable or better quality.

### 1.4 NF4 Quantization: Information-Theoretically Optimal Data Type

**Motivation:**
Pre-trained neural network weights follow approximately normal distributions. Standard uniform quantization levels waste representation capacity on regions with low probability density.

**NF4 Construction:**
NF4 creates 16 quantization levels (4 bits) that are information-theoretically optimal for N(0,1):

1. Compute quantiles of the standard normal distribution:
   ```
   q_i = Φ⁻¹(i/(2^k + 1)) for i = 1, ..., 2^k - 1
   ```
   Where Φ⁻¹ is the inverse CDF of N(0,1)

2. Center quantization bins between adjacent quantiles:
   ```
   c_i = (q_i + q_{i+1}) / 2
   ```

3. Ensure exact zero representation by creating asymmetric positive/negative ranges

**NF4 Quantization Levels (normalized to [-1, 1]):**
```
Negative: -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0
Positive: 0.0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
```

**Quantization Procedure:**
```
1. Normalize weights to [-1, 1] using absolute maximum: w_norm = w / max(|W|)
2. Find nearest NF4 level: q = argmin_i |w_norm - nf4_levels[i]|
3. Store 4-bit index q and scale factor max(|W|)
```

---

## 2. Pruning Methodologies

### 2.1 SparseGPT: One-Shot Unstructured Pruning

**Problem Formulation:**
Given a fixed pruning mask M (indicating which weights to zero), find optimal values for remaining weights that minimize layer output reconstruction error:

```
argmin_{W_M} ||W·X - W_M·X||²₂
```

Where W_M has zeros at positions specified by M.

**The Row-Hessian Challenge:**
Each row of W can have a different sparsity pattern, meaning each row requires a different masked Hessian inverse (H_F)⁻¹. Computing separate inverses for each row is prohibitively expensive for billion-parameter models.

**SparseGPT Solution:**
The key insight is to process columns left-to-right with partial updates:

1. For column j being pruned in row i:
   - The optimal update for weights in columns > j depends only on the sub-Hessian for those columns
   - This sub-Hessian is shared across all rows, regardless of their sparsity patterns

2. By updating only "rightward" weights after each column, SparseGPT reuses Hessian information across rows with different masks.

**Algorithm:**
```
Input: Weight matrix W ∈ R^{d_row × d_col}, Hessian H⁻¹, target sparsity p%
Output: Sparse weight matrix W_sparse

1. Precompute Cholesky decomposition of H⁻¹
2. For j = 0 to d_col:
   For each row i:
     a. Determine if w_{ij} should be pruned (via mask selection)
     b. If pruned: Set w_{ij} = 0
        Else: Keep current value
     c. Compute error: err = (w_{ij,original} - w_{ij}) / [H⁻¹]_{jj}
     d. Update remaining weights: W[i, j+1:] -= err · H⁻¹[j, j+1:]
3. Return W_sparse
```

**Adaptive Mask Selection:**
Rather than pre-determining the mask, SparseGPT selects which weights to prune adaptively:

```
For each block of B_s columns:
  1. Compute OBS pruning criterion: ε_ij = |w_{ij}|² / [H⁻¹]_{jj}
  2. Select lowest-error weights to achieve target sparsity
  3. Prune selected weights and update remaining
```

**Semi-Structured Sparsity (N:M Patterns):**
For 2:4 sparsity (2 zeros per 4 consecutive weights), SparseGPT modifies selection:
```
For each group of 4 consecutive weights:
  Select 2 weights with lowest pruning error ε to preserve
  Zero the other 2 weights
```

**Complexity:** O(d³_col) for Hessian preparation + O(d_col · d_row · d_col) for pruning pass.

### 2.2 Wanda: Pruning Without Weight Updates

**Simplified Criterion:**
Wanda (Weights AND Activations) prunes based on:
```
Importance(w_{ij}) = |w_{ij}| · ||X[:,j]||₂
```

Where X[:,j] is the j-th input feature's activation across calibration samples.

**No Reconstruction:**
Unlike SparseGPT, Wanda performs no weight updates after pruning—simply zeroing low-importance weights. This makes it significantly faster (minutes vs. hours) while achieving comparable results at 50% sparsity.

---

## 3. Attention Optimization Methodologies

### 3.1 FlashAttention: IO-Aware Exact Attention

**Memory Hierarchy Context:**
- HBM (High Bandwidth Memory): 40-80GB, ~1.5-2 TB/s bandwidth
- SRAM (On-chip): ~20MB total (192KB × 108 SMs on A100), ~19 TB/s bandwidth

Standard attention materializes N×N intermediate matrices in HBM, causing memory-bound operations.

**Standard Attention Memory Pattern:**
```
1. S = Q @ K^T          → Write N×N to HBM
2. P = softmax(S)       → Read N×N, Write N×N to HBM  
3. O = P @ V            → Read N×N from HBM
Total HBM access: Θ(N² + Nd)
```

**FlashAttention Tiling Strategy:**
Divide Q, K, V into blocks that fit in SRAM and compute attention incrementally:

```
Block sizes: B_r (row blocks), B_c (column blocks)
Constraint: B_r · B_c ≤ M (SRAM size)

For each K,V block j:
  For each Q block i:
    Load Q_i, K_j, V_j to SRAM
    Compute local attention: S_ij = Q_i @ K_j^T
    Apply local softmax with running statistics
    Update output accumulator
```

**Online Softmax Algorithm:**
The challenge: softmax requires the full row to compute the normalization constant.

Solution: Track running maximum m and sum l, rescale incrementally:

```
Initialize: m = -∞, l = 0, O = 0

For each block j:
  1. Compute S_ij = Q_i @ K_j^T / √d
  2. Compute block maximum: m̃ = rowmax(S_ij)
  3. Update global maximum: m_new = max(m, m̃)
  4. Compute local softmax: P̃_ij = exp(S_ij - m̃)
  5. Update running sum: l_new = e^(m - m_new) · l + rowsum(P̃_ij) · e^(m̃ - m_new)
  6. Update output: O = e^(m - m_new) · O + e^(m̃ - m_new) · P̃_ij @ V_j
  7. m = m_new, l = l_new

Final: O = O / l
```

**IO Complexity:**
```
Standard attention: Θ(Nd + N²) HBM accesses
FlashAttention: Θ(N²d²/M) HBM accesses
```

For typical M (SRAM size), this is a significant reduction.

**Backward Pass:**
FlashAttention recomputes S and P during backward pass rather than storing them:
- Saves O(N²) memory
- Additional FLOPs are offset by reduced memory traffic
- Stores only O (output) and softmax normalization statistics (m, l)

### 3.2 PagedAttention: Virtual Memory for KV-Cache

**Problem:**
KV-cache grows dynamically and varies per request. Traditional pre-allocation wastes 60-80% of memory through:
- **Internal fragmentation:** Pre-allocated slots never filled
- **External fragmentation:** Gaps between variable-size allocations
- **Redundant duplication:** Shared prefixes stored multiple times

**Core Concept:**
Apply OS virtual memory principles:
- **Pages** → KV blocks (fixed size, e.g., 16 tokens)
- **Virtual addresses** → Logical block indices
- **Physical addresses** → GPU memory locations
- **Page table** → Block table mapping logical to physical

**Data Structures:**
```python
class BlockTable:
    # Maps logical_block_idx → (physical_block_idx, num_filled_slots)
    entries: List[Tuple[int, int]]

class KVCacheManager:
    physical_blocks: Tensor  # [num_blocks, num_heads, head_dim, block_size]
    free_blocks: List[int]   # Available physical block indices
    block_tables: Dict[request_id, BlockTable]
```

**Attention Computation:**
```python
def paged_attention(query, block_table, k_cache, v_cache):
    output = zeros_like(query)
    
    for logical_idx, (physical_idx, filled) in enumerate(block_table):
        # Fetch non-contiguous KV blocks
        k_block = k_cache[physical_idx, :, :, :filled]
        v_block = v_cache[physical_idx, :, :, :filled]
        
        # Compute attention for this block
        scores = query @ k_block.T / sqrt(head_dim)
        # Accumulate with proper softmax normalization
        output += softmax(scores) @ v_block
    
    return output
```

**Memory Sharing (Copy-on-Write):**
```
Scenario: Multiple outputs from same prompt

1. All sequences share physical blocks for prompt KV-cache
2. Reference count tracks sharing: ref_count[physical_block] = num_sharers
3. On modification:
   if ref_count > 1:
       new_block = allocate_free_block()
       copy(physical_blocks[original], physical_blocks[new_block])
       update block_table entry
       ref_count[original] -= 1
```

**Block Size Trade-offs:**
- Larger blocks → Better GPU utilization, more internal fragmentation
- Smaller blocks → Less waste, more overhead
- Typical: 16 tokens (balances ~4% waste vs good parallelism)

---

## 4. Speculative Decoding Methodology

### 4.1 Draft-Verify Framework

**Motivation:**
Autoregressive generation is memory-bandwidth bound. A single forward pass reads all model weights (~TB) to generate one token. The GPU compute is vastly underutilized.

**Core Algorithm:**
```
Input: Draft model M_d (small, fast), Target model M_t (large, accurate)
       Speculation length γ

1. Draft phase: Generate γ candidate tokens using M_d
   x̃_1, x̃_2, ..., x̃_γ ~ M_d(prefix)

2. Verify phase: Run M_t on all γ+1 positions in parallel
   p_1, p_2, ..., p_{γ+1} = M_t(prefix, x̃_1, ..., x̃_γ)

3. Accept/Reject via rejection sampling:
   For i = 1 to γ:
     q_i = M_d(x̃_i | prefix, x̃_1, ..., x̃_{i-1})  # Draft probability
     p_i = M_t(x̃_i | prefix, x̃_1, ..., x̃_{i-1})  # Target probability
     
     if random() < min(1, p_i/q_i):
       Accept x̃_i, continue
     else:
       Reject x̃_i, resample from residual distribution
       Break (discard remaining draft tokens)

4. Return accepted tokens + one target-sampled token
```

**Rejection Sampling Guarantee:**
The accepted sequence follows the exact target model distribution:
```
P(accept x) = min(1, p(x)/q(x)) · q(x) = min(q(x), p(x))
P(resample from residual) = Σ_x max(0, p(x) - q(x))
```

**Expected Accepted Tokens:**
```
E[accepted] = Σ_{i=1}^{γ} Π_{j=1}^{i} α_j

Where α_j = Σ_x min(p_j(x), q_j(x)) is the acceptance rate at position j
```

**Speedup Analysis:**
Let c = cost(M_d)/cost(M_t) be the relative draft model cost.

```
Standard decoding: γ target forward passes
Speculative decoding: γ draft passes + 1 target pass = γc + 1

Speedup = γ / (γc + 1) · E[accepted] / γ
        ≈ (1 - c) · acceptance_rate / (1 + c·γ)
```

### 4.2 EAGLE: Feature-Level Speculation

**Innovation:**
Instead of using a separate draft model, EAGLE predicts from the target model's own hidden states.

**Architecture:**
```
1. Extract feature from second-to-last layer: f_t = M_t.layer[-2](x)
2. Lightweight autoregressive head predicts next features:
   f̃_{t+1} = EAGLE_head(f_t, embedding(x_t))
3. Project to vocabulary: logits = M_t.lm_head(f̃_{t+1})
```

**Training:**
```
Loss = Σ_t ||f_{t+1} - EAGLE_head(f_t, emb(x_t))||² + CE(logits, x_{t+1})
```

**EAGLE-2 Dynamic Trees:**
Rather than linear speculation, EAGLE-2 builds draft trees based on confidence:
```
1. Generate multiple candidates per position based on top-k
2. Expand high-confidence branches more deeply
3. Verify entire tree in single target pass using tree attention mask
4. Accept longest valid path
```

---

## 5. Parameter-Efficient Fine-Tuning Methodologies

### 5.1 LoRA: Low-Rank Adaptation

**Mathematical Foundation:**
The intrinsic dimensionality hypothesis suggests weight updates during fine-tuning lie in a low-dimensional subspace.

**Formulation:**
For a pre-trained weight matrix W₀ ∈ R^{d×k}, LoRA parameterizes the update as:
```
W = W₀ + ΔW = W₀ + B·A

Where: B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d,k)
```

**Forward Pass:**
```
h = W₀·x + (B·A)·x = W₀·x + B·(A·x)
```

**Initialization:**
- A: Random Gaussian initialization
- B: Zero initialization (ΔW = 0 at start, preserving pre-trained behavior)

**Scaling Factor:**
```
h = W₀·x + (α/r)·B·A·x
```

Where α is a hyperparameter controlling adaptation strength. Setting α = r at the start of experiments allows changing r without re-tuning learning rate.

**Parameter Reduction:**
```
Full fine-tuning: d × k parameters
LoRA: r × (d + k) parameters

Reduction factor: dk / (r(d+k)) ≈ d/(2r) for d ≈ k
Example: d=k=4096, r=8 → 256x reduction
```

**Target Modules:**
Typically applied to attention projections:
- Query (W_q): Most impactful
- Value (W_v): Second most impactful  
- Key (W_k): Less impact
- Output (W_o): Moderate impact
- FFN layers: Useful for some tasks

**Merging for Inference:**
```
W_deployed = W₀ + (α/r)·B·A
```
No additional inference latency—adapters are absorbed into base weights.

### 5.2 QLoRA: Quantized LoRA

**Three Key Innovations:**

**1. NF4 Base Model Quantization:**
Store frozen base model in 4-bit NF4, dequantize on-the-fly for computation:
```
Compute: Y = dequant(W_NF4) · X + B·A·X
         = (scale · nf4_lookup[indices]) · X + B·A·X

Storage: 4 bits/weight + 32-bit scale per group (e.g., 64 weights)
Compute: BF16 precision
```

**2. Double Quantization:**
Quantize the quantization constants themselves:
```
Standard: 32-bit scale per group of 64 weights → 32/64 = 0.5 bits/weight overhead
Double quant: 8-bit scale with 256-group second-level quantization
             → 8/64 + 32/256 = 0.125 + 0.125 = 0.25 bits/weight
Savings: ~0.37 bits/weight (≈3GB for 65B model)
```

**3. Paged Optimizers:**
Handle memory spikes during gradient checkpointing:
```
When GPU memory exhausted:
1. CUDA unified memory pages optimizer states to CPU
2. Computation continues with automatic page faults
3. States paged back on access

No code changes required—transparent to training loop
```

**Memory Breakdown (65B model):**
```
Component              Full FT    QLoRA
Base model weights     130GB      ~16GB (4-bit)
Gradients              130GB      0 (frozen base)
Optimizer states       260GB      ~0.5GB (LoRA only)
Activations            Variable   ~4GB (checkpointing)
LoRA adapters          N/A        ~0.5GB
Total                  >520GB     ~21GB
```

---

## 6. Complexity Summary and Trade-offs

| Technique | Time Complexity | Space Reduction | Quality Impact |
|-----------|----------------|-----------------|----------------|
| GPTQ (4-bit) | O(d³) per layer | 4× | ~0.1-0.5 perplexity |
| AWQ (4-bit) | O(calibration) | 4× | <GPTQ degradation |
| SparseGPT 50% | O(d³) per layer | ~2× | ~0.05 perplexity |
| FlashAttention | Same FLOPs | O(N) vs O(N²) | Exact (none) |
| PagedAttention | Slight overhead | ~4% waste vs 60-80% | None |
| Speculative (γ=5) | 5c + 1 target calls | None | Exact (none) |
| LoRA (r=8) | <1% params trained | 256× checkpoint | Task-dependent |
| QLoRA | Same as LoRA | 10-20× memory | Same as LoRA |

---

## 7. Implementation Guidelines

### 7.1 Framework Selection by Target

| Target Platform | Recommended Stack | Key Techniques |
|----------------|-------------------|----------------|
| NVIDIA Datacenter | TensorRT-LLM + vLLM | FP8, PagedAttention, continuous batching |
| NVIDIA Consumer | ExLlamaV2 or llama.cpp | GPTQ/EXL2, Flash-Attn |
| Apple Silicon | MLX | 4-bit native, unified memory |
| CPU-only | llama.cpp | GGUF Q4_K_M, AVX-512/AMX |
| Mobile | MLC-LLM | INT4, NPU delegation |

### 7.2 Quantization Selection

```
Decision tree:
1. Is accuracy critical? 
   → Yes: Q8_0 or AWQ INT4
   → No: Continue

2. Is memory severely constrained?
   → Yes: Q3_K_M or IQ3_XXS
   → No: Q4_K_M or Q5_K_M

3. Is serving throughput priority?
   → Yes: AWQ (fast quant) + vLLM
   → No: GPTQ (slightly better accuracy)
```

### 7.3 Fine-tuning Decision

```
Memory available vs Model size:
- >4× model size: Full fine-tuning feasible
- 2-4× model size: LoRA with BF16 base
- 1-2× model size: QLoRA required
- <1× model size: Gradient checkpointing + QLoRA
```

---

## 8. References

1. Frantar, E. et al. "GPTQ: Accurate Post-training Quantization for Generative Pre-trained Transformers." ICLR 2023.
2. Lin, J. et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." MLSys 2024.
3. Dettmers, T. et al. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023.
4. Frantar, E. & Alistarh, D. "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot." ICML 2023.
5. Dao, T. et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022.
6. Kwon, W. et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.
7. Leviathan, Y. et al. "Fast Inference from Transformers via Speculative Decoding." ICML 2023.
8. Hu, E. et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
9. Ma, S. et al. "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits." arXiv 2024.
