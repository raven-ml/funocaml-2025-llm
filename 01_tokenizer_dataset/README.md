# 01 — Tokenizer & Dataset

High‑level task
- Next‑token prediction: given the last T characters (context), predict the next one.

What training data must look like
- Inputs `x:[N; T]` (contexts of exactly T token ids)
- Targets `y:[N]` (the id of the next token for each context)
 - If a context is shorter than `T`: left‑pad with the start token id (the `'.'` id, 0).
   Example (T=4): ids so far `[0,1]` → context becomes `[0,0,0,1]`.

What we’ll implement now
- A character vocabulary and encode/decode functions
  - `encode: string -> int list` maps characters to token ids
  - `decode: int list -> string` maps ids back to characters
- A dataset maker that turns raw names into (context → next token) pairs.

Outcome
- By the end, `dune exec .../01_tokenizer_dataset/exercise/exercise.exe` should print basic sanity info and not fail.

{pause up}
---

## Why this matters

- We predict the next token, so we need many (context → next) examples.
- Models consume numbers, not characters. Vocabulary maps each symbol to a stable id.
- Embedding layers are lookup tables `[vocab_size x n_embd]`; token ids index rows. Consistent ids are critical.
- The start/stop token `'.'` teaches beginnings and endings so we can sample valid names.

Mental model
- Vocabulary = alphabet with ids; Encode text → ids; Decode ids → text.
- Dataset = many tiny rows of fixed length: X is context `[T]`, y is next id.

{pause up}
---

## From raw text to supervised pairs (the pipeline)

1) Read raw names (strings)
2) Build a vocabulary (characters ↔ ids)
3) Encode text into ids; wrap each name with `'.'` on both sides
4) Slide a fixed‑size window (“take the last T ids”) to make (context → next) pairs
   - A “window” is the last `T` tokens right before the next token
   - Why: fixed shapes for batching and to let models learn local context
5) Pack into tensors and a Kaun `Dataset` for batching/shuffling

Tiny visual
```
"anna"         →  ".anna."             →  [0,1,14,14,1,0]
raw text          with boundaries          encoded ids

windows (T=4):
[0,0,0,0] → 1    [0,0,0,1] → 14    [0,0,1,14] → 14    [0,1,14,14] → 1    [1,14,14,1] → 0
```

{pause up}
---

## Step 1 — Build the vocabulary

Task
- Scan `names.txt`, collect unique chars; always include `'.'` mapped to 0.
- Produce:
  - `vocab_size:int`
  - `ch2idx: char -> int`
  - `encode: string -> int list`
  - `decode: int list -> string`

Visual example
- Suppose observed chars are `{'.','a','b','n'}`
- ids: `'.'→0, 'a'→1, 'b'→2, 'n'→3`
- Encode `.anna.` → `[0,1,3,3,1,0]` → Decode back `.anna.`

Key API
- `Saga.read_lines` (reads file into list of strings)

Gotchas
- Lowercase everything for stability.
- Ensure `ch2idx` returns 0 (the dot) for unknown chars (defensive default).

{pause up}
---

## Step 2 — Build context windows (block_size = T)

We turn every wrapped name `"." ^ name ^ "."` into (context → next) pairs.

Example with T=4, name `anna`
- Wrapped: `.anna.` → ids `[0,1,14,14,1,0]`
- Pairs:
  - Context `[0,0,0,0]` → Next `1`
  - Context `[0,0,0,1]` → Next `14`
  - Context `[0,0,1,14]` → Next `14`
  - Context `[0,1,14,14]` → Next `1`
  - Context `[1,14,14,1]` → Next `0`

Rules
- Left‑pad with the first id (the leading dot) so context has exactly T tokens.
- Next is the token immediately after the current position.

{pause up}
---

## Step 3 — Materialize a Rune dataset

Data layout
- Flatten all examples into two flat arrays:
  - `x_data: float[]` of length `N*T`
  - `y_idx: float[]` of length `N` (class indices as floats)
- Then create tensors and a dataset:
  - `x: [N; T]`, `y: [N]`, dtype `float32`
  - `Dataset.from_tensors (x,y) |> Dataset.batch batch_size |> Dataset.shuffle ~buffer_size:200`

Key APIs
- `Rune.create`, `Rune.float32`
- `Dataset.from_tensors`, `Dataset.batch`, `Dataset.shuffle`

Notes
- Using floats for labels is OK here because the loss expects integer‑encoded labels in a float tensor.

{pause up}
---

## Sanity checks you should see

- Vocab size: a small integer (~27–35 for Latin names)
- Encode/decode roundtrip matches length of input string
- Building dataset on a subset (e.g., first 50 names) prints the approximate window count without errors

How to run
- `dune exec ./01_tokenizer_dataset/solution.exe`

{pause up}
---

## Common pitfalls (and fixes)

- Forgot to include `'.'`? Start/stop won’t work — add it with id 0.
- Mismatch lengths? Check left‑padding; every context must have exactly T tokens.
- Off‑by‑one windows? Number of pairs is `len(ids) - 1` per wrapped string.
- Using raw strings in tensors? Always encode to ids; tensors hold numbers.
