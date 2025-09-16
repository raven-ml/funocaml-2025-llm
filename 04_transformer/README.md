## 04 — Upgrade to a Transformer Decoder (Pipeline Track)

High‑level task
- Replace the MLP with a small GPT‑2–style decoder using Kaun primitives.

{pause up}
---

Why this matters
- Attention can focus by content and capture longer dependencies more flexibly than a fixed MLP.

{pause up}
---

What we’ll compose (matches the code)
- Token embedding: `Embedding(vocab, C)` → `[B;T;C]`
- Learned positional embedding: `positional_embedding_learned(T, C)`
- Stacked decoder blocks: `transformer_decoder(n_layer, n_head, mlp_hidden=4*C)`
- Final layer norm → take last time step `[B;T;C]→[B;C]` → `Linear(C→vocab)`

{pause up}
---

Shapes (tiny config)
- Use `T=16`, `C=64`, `n_layer=2`, `n_head=4`, batch `256`, epochs `1`, `lr=2e-3`, `weight_decay=0.01` for quick CPU runs.

{pause up}
---

What you’ll do
- Swap the model builder to the decoder composition above.
- Train 1 epoch (same dataset) and report eval NLL + PPL.
- Reuse the sampler from step 03 — only the `model_fn` changes (pads to T and returns logits).

{pause up}
---

Run
- `dune exec ./04_transformer/main.exe`

{pause up}
---

Expected
- Eval NLL ≤ the MLP baseline with the same T (varies by seed/config).
- Generated names often look sharper/more varied at similar temperatures.

{pause up}
---

Troubleshooting
- Slow on CPU: reduce `T` to 8 or `n_layer` to 1; keep batch 128–256.
- Chaotic generation: lower temperature (e.g., 0.8) or set top‑p.
