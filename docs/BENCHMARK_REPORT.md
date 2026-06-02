# Empirical Benchmark Report — Qwen2.5-0.5B-Instruct LoRA Fine-Tuning

> **Engineered by**: IntelligentStack AI (`raklabs`)  
> **Pipeline**: LoraForge AI (`loraforge-ai`)  
> **Release Date**: `2026-06-04`  
> **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct`  
> **Pinned Revision**: `7ae557604adf67be50417f59c2c2f167def9a775`

---

## 1. Executive Summary

Using **LoraForge AI**, we fine-tuned `Qwen/Qwen2.5-0.5B-Instruct` (`494.57M` parameters) via Low-Rank Adaptation (**LoRA**, training just `540,672` parameters or **`0.109%`** of the network) for 1 epoch (`20` optimizer steps, effective batch size `8`) on a balanced 160-example support ticket dataset.

On the **frozen 40-example test set**, strict JSON classification accuracy increased from **22.5% (9/40)** on the zero-shot baseline to **80.0% (32/40)**—an improvement of **+57.5 percentage points**—while reaching **100.0% (40/40) strict JSON schema validity**.

---

## 2. Quantitative Test-Set Results (`n = 40` Frozen Test Split)

| Metric | Pre-Trained Baseline (`Qwen2.5-0.5B-Instruct`) | Fine-Tuned LoRA Adapter (`r=8, alpha=16`) | Delta |
| :--- | :---: | :---: | :---: |
| **Strict JSON Schema Validity** | `39/40` (**97.5%**) | `40/40` (**100.0%**) | **+2.5 pp** |
| **Overall Category Accuracy** | `9/40` (**22.5%**) | `32/40` (**80.0%**) | **+57.5 pp** |
| **`billing` Accuracy** | `1/10` (**10.0%**) | `7/10` (**70.0%**) | **+60.0 pp** |
| **`account` Accuracy** | `0/10` (**0.0%**) | `8/10` (**80.0%**) | **+80.0 pp** |
| **`technical` Accuracy** | `0/10` (**0.0%**) | `9/10` (**90.0%**) | **+90.0 pp** |
| **`other` Accuracy** | `8/10` (**80.0%**) | `8/10` (**80.0%**) | **0.0 pp** |

---

## 3. Training & Standalone Merge Telemetry

| Telemetry Dimension | Recorded Value |
| :--- | :--- |
| **Total Model Parameters** | `494,573,440` |
| **Trainable LoRA Parameters (`q_proj`, `v_proj`)** | `540,672` (`0.1093%`) |
| **Epoch 1 Weighted Training Loss (Assistant Tokens)** | `0.3953` |
| **Epoch 1 Validation Loss (Assistant Tokens)** | `0.0765` |
| **Optimizer Steps (`micro_batch=1, grad_accum=8`)** | `20` steps (`160` examples) |
| **Wall-Clock Training Time (CPU `float32`, 4 threads)** | `238.85 seconds` (~3.98 minutes) |
| **Peak Process Memory (RSS)** | `3.91 GiB` |
| **Standalone Merge Max Logit Difference (`W + B@A`)** | `2.079e-4` (within `atol=1e-3`) |
| **Standalone Merge Mean Logit Difference** | `2.276e-5` |
| **Fresh-Process Standalone Reload Check (`account-21-1`)** | `PASSED` (`{"category":"account"}`) |

### Cryptographic Provenance Digests (SHA-256)
- `train.jsonl`: `22e975fad325e95963bbf0264ab25ca46769f8836a799db9b9639f768d453df4`
- `validation.jsonl`: `81fcb3c4635954172804af514ff5eaec59591a0d7b9990bf3a7b49bbe655451b`
- `test.jsonl`: `829e4e942f60a3ac52c88bb8bc9a59b2deb7cd5f1db9d964d43a2237c5c64b04`
- `system_prompt`: `73cefde4905abfbb15633a0135ec752c1ae0dd2923fdb6bf0b4325332091778d`
- `adapter_model.safetensors`: `0cf7957eaa6f5c4153595218328aef3e1fa78473ed3c9a3fb9efa5a5e515762c`

---

## 4. Error Analysis & Boundary Case Deep-Dive

Why did the zero-shot baseline score `22.5%`? Prior to LoRA SFT, `Qwen2.5-0.5B-Instruct` exhibited a strong prior bias toward predicting `{"category": "other"}` across almost all inputs (scoring `80%` on `other`, `10%` on `billing`, and `0%` on `account` and `technical`).

After 1 epoch of LoRA SFT, the model resolved single-domain tickets accurately (`90%` on `technical`, `80%` on `account`), while the remaining errors concentrated in **cross-domain lexical overlap** where surface nouns from one category collide with the intent of another:

### Case 1: `technical-26-1` (Lexical Overlap: `"invoice"` vs. Sorting Bug)
- **Input**: *"Sorting invoice records by date puts them in alphabetical order."*
- **Ground Truth**: `technical` (software UI sorting bug)
- **Baseline Output**: `{"category": "other"}`
- **LoRA Output**: `{"category":"other"}`
- **Root-Cause Analysis**: The ticket contains the billing noun `"invoice"` alongside UI behavior (`"sorting... alphabetical order"`). Competing activations between `billing` and `technical` caused the 0.5B model to fall back to `other`.
- **Remediation**: Add contrastive training pairs where billing objects (`invoice`, `receipt`) experience UI/rendering defects explicitly labeled `technical`.

### Case 2: `account-27-1` & `account-27-2` (Enterprise Workspace Administration)
- **Input 1**: *"I need to remove a former employee's access to our workspace."*
- **Input 2**: *"How do I revoke workspace membership for someone who left the company?"*
- **Ground Truth**: `account`
- **Baseline Output**: `{"category": "other"}`
- **LoRA Output**: `{"category":"technical"}`
- **Root-Cause Analysis**: Training scenarios `account-01` through `account-20` focused heavily on first-person self-service (`forgot my password`, `my 2FA phone`, `my profile`), whereas `account-27` involves multi-user workspace administration (`revoke workspace membership`), which the adapter associated with system configuration (`technical`).

### Case 3: `billing-28-1` & `billing-28-2` (Departmental Invoice Consolidation)
- **Input 1**: *"Please combine the two department invoices into one bill."*
- **Input 2**: *"Can finance receive a single invoice covering both departments?"*
- **Ground Truth**: `billing`
- **Baseline Output**: `{"category": "other"}`
- **LoRA Output**: `{"category":"account"}`
- **Root-Cause Analysis**: The phrase `"combine... into one"` closely mirrors training scenario `account-15` (*"merge duplicate user accounts into one login"*), causing the attention projections (`q_proj, v_proj`) to map consolidation phrasing to `account`.

---

## 5. Next Engineering Iteration Plan

Following strict test-set quarantine (never tuning directly on `test.jsonl`), the next controlled experiment on `validation.jsonl` should test **one variable at a time**:
1. **Target Module Expansion**: Expand LoRA target modules from `["q_proj", "v_proj"]` to all attention and MLP projections (`["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`) to improve disambiguation on cross-domain nouns.
2. **2-Epoch Schedule with Cosine Warmup**: Evaluate whether 2 epochs (`40` optimizer steps) reduces validation loss below `0.0765` without overfitting.

