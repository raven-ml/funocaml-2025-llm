# FunOCaml 2025 — Building an LLM with Raven

Welcome! This workshop walks you through building a tiny end‑to‑end next‑token model in OCaml using Raven’s libraries (Rune, Kaun, Saga):
- Tokenize a names dataset and build (context → next) pairs
- Train a simple MLP baseline, then a small Transformer
- Sample generated names from your trained model

- This workshop is inspired by Andrej Karpathy’s makemore: https://github.com/karpathy/makemore/

## Prerequisites
- Have LLVM Version 19 installed (Debian/Ubuntu: https://apt.llvm.org/) 
- Have Dune installed, either from opam, or `curl -fsSL https://get.dune.build/install | sh`
- Join the workshop chat on Discord: https://discord.gg/vKPEXGye

## Setup
We use Dune package management (no opam):
- Lock dependencies: `dune pkg lock`
- Build everything: `dune build`

Tip: The first dependency lock/build will download pinned packages and may take a few minutes.

## Run The Steps
- 01 — Tokenizer & Dataset: `dune exec ./01_tokenizer_dataset/solution.exe`
- 02 — Train MLP Baseline: `dune exec ./02_train/solution.exe`
- 03 — Sample From Model: `dune exec ./03_sample/solution.exe`
- 04 — Transformer Decoder: `dune exec ./04_transformer/solution.exe`

Each step prints progress and basic metrics or samples to stdout. See the per‑step READMEs for details:
- `01_tokenizer_dataset/README.md`
- `02_train/README.md`
- `03_sample/README.md`
- `04_transformer/README.md`

## Data
- Input file: `names.txt` (plain text, one name per line)
- Context length (T) and other knobs are set in the step sources; feel free to tweak.

Have fun, and happy hacking!
