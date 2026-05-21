# Dataset Provenance, Split Isolation & Governance

> **Curated by**: IntelligentStack AI (`raklabs`)  
> **Total Examples**: `240` English support tickets (`120` distinct scenarios $\times$ `2` paraphrases)  
> **PII Status**: `100% Synthetic` — Zero customer or personal data

---

## 1. Split Architecture & Scenario Isolation

Each of the four target categories (`billing`, `account`, `technical`, `other`) contains **30 distinct scenarios**, and each scenario has **2 manually authored paraphrases**:

| Split | Scenarios per Category | Examples per Category | Total Examples | SHA-256 Digest |
| :--- | :---: | :---: | :---: | :--- |
| `train.jsonl` | `01–20` (`20`) | `40` | **`160`** | `22e975fad325e95963bbf0264ab25ca46769f8836a799db9b9639f768d453df4` |
| `validation.jsonl` | `21–25` (`5`) | `10` | **`40`** | `81fcb3c4635954172804af514ff5eaec59591a0d7b9990bf3a7b49bbe655451b` |
| `test.jsonl` | `26–30` (`5`) | `10` | **`40`** | `829e4e942f60a3ac52c88bb8bc9a59b2deb7cd5f1db9d964d43a2237c5c64b04` |

### Why Scenario-Level Split Isolation Matters
In standard random split datasets, two paraphrases of the same underlying issue often land in both `train` and `test`, inflating test accuracy through near-duplicate memorization. In **LoraForge AI**, both paraphrases of a `scenario_id` (e.g., `billing-01-1` and `billing-01-2`) are strictly confined to the same split, and `loraforge validate` verifies zero cross-split scenario leakage at load time.

---

## 2. Dataset Quality & Governance Checklist

- [x] **Single Clear Intent**: Every ticket expresses one primary support intent matching its category definition in `specs/001-reproducible-lora-sft.md`.
- [x] **Self-Contained Labeling**: Each label is verifiable from the ticket text alone without external context.
- [x] **Paraphrase Equivalence**: Both variants in each scenario share identical semantic intent while varying vocabulary and syntax.
- [x] **Strict Split Isolation**: Zero `scenario_id` overlap across `train`, `validation`, and `test`.
- [x] **Boundary Case Inclusion**: Includes realistic cross-domain lexical distractors (e.g., `technical-26`: sorting invoice dates alphabetically is a `technical` UI bug rather than a `billing` dispute).
- [x] **Zero PII**: Contains no customer names, emails, or real organizational identifiers.

**Human Review Status**: Completed & Verified (`raklabs`, IntelligentStack AI).

