# CS5760 – Natural Language Processing | Homework 4

**University of Central Missouri – Department of Computer Science & Cybersecurity**
**Course:** CS5760 – Natural Language Processing | Spring 2026
**Student Name:** [Your Name]
**Student ID:** [Your ID]

---

## Submission

- Source code pushed to this GitHub repository
- Short-answer solutions submitted separately via Bright Space
- All code is commented as required

---

## Repository Structure

```
├── Q1_RNN.ipynb              # Q1 – Character-Level RNN Language Model
├── Q2_transformer.ipynb      # Q2 – Mini Transformer Encoder
├── Q3_attention.ipynb        # Q3 – Scaled Dot-Product Attention
└── README.md
```

---

## Part II – Programming

---

### Q1: Character-Level RNN Language Model (`Q1_RNN.ipynb`)

**Goal:** Train a character-level RNN to predict the next character given previous characters.

**Dataset:**
- **Phase 1 (toy):** Small corpus of `hello`, `help`, short sentences (~5,985 chars, repeated ×15 for training volume)
- **Phase 2 (large):** *Pride & Prejudice* auto-downloaded from Project Gutenberg (~150,000 chars, public domain)

**Model Architecture:**
```
Embedding(vocab_size → 64)
  → GRU(hidden_size=256, num_layers=2, dropout=0.3, batch_first=True)
  → Linear(256 → vocab_size)
  → Softmax (at generation time via temperature scaling)
```

**Training Setup:**

| Hyperparameter | Value |
|---|---|
| Sequence length | 75 |
| Batch size | 64 |
| Embedding dim | 64 |
| Hidden size | 256 |
| GRU layers | 2 |
| Epochs | 15 |
| Learning rate | 1e-3 (Adam) |
| LR scheduler | StepLR (γ=0.5, step=5 epochs) |
| Gradient clipping | 5.0 |
| Train/Val split | 90% / 10% |

- **Teacher forcing:** At every training step the true previous character is fed as input (not the model's own prediction)
- **Loss:** Cross-entropy; **Optimizer:** Adam

**Report:**
1. Training/validation loss curves saved as `q1_loss_curves_toy.png` (toy) and `q1_loss_curves_large.png` (large corpus)
2. Temperature-controlled text generation — 3 samples per run (τ = 0.7, 1.0, 1.2), ~350 chars each:
   - **τ = 0.7** → sharpened softmax, confident and repetitive output
   - **τ = 1.0** → samples from the true learned distribution
   - **τ = 1.2** → flattened softmax, more diverse and noisy output
3. **Reflection:** Longer sequence lengths (e.g., 75 vs 50) give the GRU more context for learning word boundaries but deepen the gradient path and increase memory. Increasing hidden size from 64 to 256 boosts capacity but risks overfitting on small corpora without sufficient dropout. Temperature directly controls the sampling loop: τ < 1 sharpens the distribution (exploit), τ > 1 flattens it (explore). Teacher forcing during training creates a train/inference mismatch — at inference the model feeds its own output back, and high temperature amplifies this drift.

---

### Q2: Mini Transformer Encoder (`Q2_transformer.ipynb`)

**Goal:** Build a Transformer Encoder from scratch to process a batch of sentences and produce contextual embeddings.

**Dataset:** 10 custom short sentences (e.g., `"transformers use self attention"`, `"attention is all you need"`), tokenized at the word level. Vocabulary size: 44 tokens. Max sequence length: 6 (with `<PAD>` for shorter sentences).

**Architecture:**

| Component | Detail |
|---|---|
| Word Embedding | `nn.Embedding(44, 64)`, padding_idx=0 |
| Positional Encoding | Sinusoidal — `PE(pos,2i) = sin(pos/10000^(2i/d))`, `PE(pos,2i+1) = cos(...)` |
| Encoder Layers | 2 layers |
| Self-Attention | Multi-head, **4 heads**, d_k = 64/4 = 16 per head |
| Feed-Forward | Linear(64→128) → ReLU → Dropout → Linear(128→64) |
| Add & Norm | Residual connection + LayerNorm after each sub-layer |
| Final Norm | LayerNorm on encoder output |

**Step-by-step flow:**
1. Tokenize sentences → build vocabulary → pad to max length
2. Embed tokens + add sinusoidal positional encoding
3. Pass through 2× EncoderLayer (MHA → Add&Norm → FFN → Add&Norm)
4. Apply final LayerNorm

**Output shown:**
- Input token IDs for all 10 sentences (printed)
- Final contextual embedding tensor: shape `(10, 6, 64)`
- Attention weight matrix for sentence `"transformers use self attention"` (Layer 2, Head 1) — printed as a formatted table
- Attention heatmap saved as `q2_attention_heatmap.png`

---

### Q3: Scaled Dot-Product Attention (`Q3_attention.ipynb`)

**Goal:** Implement the attention function from the slides and validate it with multiple tests.

**Formula:**
```
Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) ) * V
```

**Implementation:** `scaled_dot_product_attention(Q, K, V, mask=None)` returns:
- `output` — weighted sum of values `(..., T, d_v)`
- `weights` — attention probabilities after softmax `(..., T, T)`
- `raw_scores` — unscaled `QK^T`
- `scaled_scores` — `QK^T / sqrt(d_k)`

**Tests:**

| Test | Setup | What is verified |
|---|---|---|
| Test 1 | T=4, d_k=4, deterministic Q/K/V | Raw scores, scaled scores, full weight matrix, output vectors, row sums = 1.0 |
| Test 2 | T=8, d_k=64, random N(0,1) inputs | Full attention matrix + output vectors printed; softmax stability check |
| Test 3 | B=2, H=4, T=6, d_k=16 (batched multi-head) | Shape assertions + row-sum assertions pass |
| Test 4 | T=5, d_k=8, causal upper-triangular mask | Blocked (upper-triangle) positions sum to ~0 |

**Softmax stability check (Test 2):** Without scaling, score std is large → softmax saturates (avg max-prob near 1.0, near-zero gradients). After dividing by √64 = 8, std drops to ≈ 1.0 → softmax weights are balanced → healthy gradient flow during training.

---

## How to Run

```bash
pip install torch numpy matplotlib
jupyter notebook
```

Run each notebook top-to-bottom. Q1 auto-downloads *Pride & Prejudice* (internet required for Phase 2 only). Q2 and Q3 are fully self-contained.

All random seeds are fixed (`torch.manual_seed(42)`, `np.random.seed(42)`) for reproducibility.
