## 03 — Sample from the Model (Pipeline Track)

High‑level task
- Turn a trained model into a generator: given a history of token ids, predict the next‑token distribution and sample repeatedly to produce text.

Why this matters
- Validates the pipeline end‑to‑end with human‑readable output.
- Lets you compare different checkpoints (MLP vs transformer) by swapping the `model_fn`.

What this step does (in this repo)
- Rebuilds a small MLP (same as step 02), trains briefly, then samples names.
- Defines a `model_fn` that pads the history on the left to length T=16 and returns logits for the next token.

Inputs/outputs (for `model_fn`)
- Input: list of previous token ids, e.g., `[0; 1; 14]`
- Output: 1D float array of logits with length `vocab_size`

How sampling works (intuition)
1) Start from `"."` (start token)
2) Left‑pad history to T=16 with `'.'`
3) Get logits from `model_fn`, apply temperature, sample next id
4) Append and repeat until `'.'` (EOS) or `max_new` tokens

Run
- `dune exec ./03_sample/solution.exe`

Expected
- Prints eval NLL/PPL for the quick MLP run, then ~10 sampled names
- Temperature around `0.7–1.0` gives a good balance of coherence/novelty

Troubleshooting
- Repetitive or empty outputs: lower temperature or `max_new_tokens`
- Exceptions: ensure `names.txt` exists; first build may take minutes to fetch deps

Notes
- The code extracts the text between leading and trailing `'.'` to present clean names
- To sample from another model, swap the `model_fn` implementation and params
