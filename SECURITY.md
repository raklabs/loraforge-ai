# Security Policy & ML Supply-Chain Controls

> **Maintained by**: IntelligentStack AI (`raklabs`)

**LoraForge AI** implements defense-in-depth security controls for Supervised Fine-Tuning (SFT), LoRA adapter management, standalone model merging, and production inference aligned with the **OWASP Top 10 for Large Language Model Applications**.

---

## 1. Active ML Security Controls

| Threat Category | Implementation in LoraForge AI | OWASP Reference |
| :--- | :--- | :--- |
| **Unsafe Model Deserialization (Pickle Exploit Prevention)** | All base models, LoRA adapters, and merged standalone checkpoints enforce `use_safetensors=True` and `safe_serialization=True` ([`loraforge/engine.py`](loraforge/engine.py)). Legacy `.bin`/`.pt` pickle checkpoints are rejected. | **OWASP LLM05 (Supply Chain Vulnerabilities)** |
| **Mutable Upstream Model Tampering** | [`loraforge/engine.py`](loraforge/engine.py) pins the exact 40-character hexadecimal Git commit SHA from Hugging Face Hub in `model-lock.json` and loads strictly with `local_files_only=True`. | **OWASP LLM05 (Supply Chain Vulnerabilities)** |
| **Training Data Poisoning & Split Leakage** | [`loraforge/dataset.py`](loraforge/dataset.py) verifies SHA-256 dataset split digests against `model-lock.json`, enforces balanced classes, rejects duplicate text, and prevents cross-split `scenario_id` leakage. | **OWASP LLM03 (Training Data Poisoning)** |
| **Silent Target Truncation & Prompt Masking Drift** | [`encode_example`](loraforge/dataset.py) verifies byte-for-token chat-template prefix invariance and rejects any example exceeding `max_length` rather than truncating the target JSON. | **OWASP LLM04 / Model Integrity** |
| **Strict Output Schema & Duplicate-Key Rejection** | [`parse_prediction`](loraforge/evaluation.py) uses `object_pairs_hook=unique_pairs` to reject duplicate JSON keys (`{"category":"account","category":"billing"}`), extra fields, prose wrappers, or non-string labels. | **OWASP LLM02 (Insecure Output Handling)** |
| **Hardened FastAPI Inference Boundary** | [`loraforge/serve.py`](loraforge/serve.py) enforces payload length bounds, control-character sanitization, constant-time `X-API-Key` authentication, and strict security response headers. | **OWASP LLM01 / API Security** |

---

## 2. Reporting a Vulnerability

Please report any security vulnerabilities privately via GitHub Security Advisories on `github.com/raklabs/loraforge-ai`.

