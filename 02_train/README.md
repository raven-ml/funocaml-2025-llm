## 02 — Train an MLP Baseline (Pipeline Track)

High‑level task
- Train a next‑token model from a fixed context window of length T=16 using a small MLP.
- Use Kaun and the dataset API from step 01 to learn short‑range patterns.

Mental model
- Look at the last T characters, embed each into a vector, flatten to one feature vector, predict the next token.
- Pros: simple, fast; learns local patterns. Cons: mixing is fixed to positions (1..T).

Architecture (implemented in code)
- `Embedding(vocab, 64)` → `Flatten` → `Linear(T*64 → 64)` → `Tanh` → `Linear(64 → vocab)`
- Loss: `Loss.softmax_cross_entropy_with_indices` (average NLL in nats)
- Optimizer: AdamW by default (`lr=2e-3`, `weight_decay=0.01`) or Adam

Inputs/outputs
- Input `x:[B;T]`: last 16 token ids, left‑padded with `'.'`
- Target `y:[B]`: next token id
- Output logits `:[B; vocab]`

Training config
- Split: deterministic 90/10 train/eval on names
- Data: `Dataset.sliding_window` → `batch 256` → `shuffle`
- Epochs: 1 (CPU‑friendly); RNG seed 42

Run
- `dune exec ./02_train/solution.exe`

Expected
- Prints training progress, then eval NLL and PPL
- With T=16, MLP should beat bigram (lower NLL/PPL). Expect NLL ≈ 2.2–2.6 depending on seed and machine. Runtime: seconds to ~1–2 minutes on CPU.

NLL and PPL
- NLL: surprise at the true next token — lower is better
- PPL = `exp(NLL)` ≈ average branching factor

Troubleshooting
- NaN loss: ensure dataset isn’t empty; labels are integer ids (encoded chars)
- Loss flat: lower LR (e.g., `1e-3`) or increase batch size
- Slow first run: dependency fetch during first `dune build` is expected

Try next
- Tweak `block_size`, `n_embd`, or epochs in `02_train/solution.ml`
- Save `state.params` for reuse when sampling (step 03)
