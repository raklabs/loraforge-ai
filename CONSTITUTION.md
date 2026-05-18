# Engineering & ML Security Constitution — LoraForge AI

> **Project**: LoraForge AI — Auditable LoRA Fine-Tuning, Verification & Export Pipeline (`loraforge-ai`)  
> **Organization**: IntelligentStack AI (`raklabs`)  
> **Methodology**: Spec-Driven Development (SDD)  
> **Standard**: Production-Grade Python & Reproducible ML Engineering

This document defines the non-negotiable architectural, scientific reproducibility, software quality, and ML security standards for **LoraForge AI**. Every capability must trace to a numbered specification in `specs/` and pass automated verification gates before merging.

---

## 1. Spec-Driven ML Engineering Workflow

1. **Specification First (`specs/NNN-feature-name.md`)**:
   - Every training workflow, dataset contract, evaluation metric, or export gate is specified before implementation.
2. **Reproducibility & Cryptographic Provenance**:
   - **Immutable Base Model Revision**: Base models are pinned to an exact 40-character hexadecimal commit SHA from Hugging Face Hub—never a mutable branch tag (`main`).
   - **Frozen Dataset Split Hashing**: `train.jsonl`, `validation.jsonl`, and `test.jsonl` are hashed via SHA-256 during preparation. Any post-preparation modification to the dataset invalidates the lock (`model-lock.json`) and halts training/evaluation.
   - **Adapter-to-Evaluation Binding**: Every evaluation artifact records the SHA-256 digest of `adapter_model.safetensors` and verifies exact agreement of `config`, `model_id`, `revision`, `data_hashes`, and `system_prompt_sha256` before generating comparison reports.

---

## 2. Core ML Correctness & Scientific Rigor Principles

### 2.1 Scenario-Level Split Isolation (Zero Data Leakage)
- Every dataset record includes an `id`, `scenario_id`, `text`, and `category`.
- All paraphrases of a given `scenario_id` MUST reside in the same split (`train`, `validation`, or `test`). Cross-split scenario leakage or case/whitespace-normalized duplicate text is rejected at load time.

### 2.2 Assistant-Only Loss Masking & Non-Truncation
- **Loss Masking**: Prompt tokens (`system` + `user` + generation prefix) are masked with `-100` so cross-entropy loss is computed exclusively over the assistant's structured JSON target tokens and end-of-turn marker.
- **Prefix Invariance Check**: The tokenized prompt prefix must match `full_sequence[:len(prompt)]` byte-for-token before masking is applied.
- **Zero Target Truncation**: Any training or evaluation example exceeding `max_length` (or leaving insufficient budget for `max_new_tokens`) fails fast rather than silently truncating the target JSON.

### 2.3 Strict Test-Set Quarantine
- Validation loss selects the best checkpoint; the frozen test split (`--split test --final`) is locked behind an explicit `--final` flag to prevent iterative hyperparameter tuning on the test set.
- Raw model outputs are scored without post-hoc regex repair; malformed JSON or extra keys (`unique_pairs` check) are scored as invalid.

### 2.4 Numerically Verified Standalone Model Export
- Merging a LoRA adapter (`W_merged = W_base + (alpha / r) * B @ A`) via `merge_and_unload(safe_merge=True)` must pass two automated gates before marking `export.json` complete:
  1. **Logit Equivalence Gate**: Next-token logits on a held-out validation example must match between the unmerged PEFT model and the merged standalone model within `rtol=1e-4, atol=1e-3`, with zero `NaN`/`Inf` values.
  2. **Fresh-Process Standalone Reload Gate**: A separate OS process loads *only* the exported standalone directory (`local_files_only=True`, no base snapshot or PEFT adapter) and verifies byte-identical greedy generation.

---

## 3. ML Security & Supply Chain Controls (OWASP LLM Top 10)

### 3.1 Safe Deserialization (`safetensors` Only — OWASP LLM05)
- All base models, LoRA adapters, and merged checkpoints MUST load and save exclusively with `use_safetensors=True` / `safe_serialization=True`. Legacy Python `pickle` weights (`.bin`, `.pt`) are prohibited to eliminate arbitrary code execution during model loading.

### 3.2 Training Data Integrity & Poisoning Defense (OWASP LLM03)
- Dataset validators enforce strict schema keys, non-empty strings, balanced class distributions, control-character sanitization, and SHA-256 lock verification prior to training and evaluation.

### 3.3 Inference Boundary Safety (OWASP LLM01 / LLM02)
- The production inference API (`loraforge/serve.py`) enforces request payload limits, strips control characters, bounds `max_new_tokens`, and validates model output through a strict duplicate-key-rejecting JSON parser (`parse_prediction`) before returning structured classifications.
- Zero hardcoded secrets or local developer filesystem paths are permitted in committed code or experiment reports.

