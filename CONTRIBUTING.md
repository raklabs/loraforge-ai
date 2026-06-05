# Contributing to LoraForge AI

Thank you for your interest in **LoraForge AI**, built and maintained by **IntelligentStack AI (`@raklabs`)**.

To protect repository integrity, supply-chain security, and deterministic reproducibility, all contributions are governed by strict guardrails.

---

## 1. Spec-Driven Development Workflow

We do not accept unrequested code dumps or ad-hoc changes. Every change must follow our **Spec-Driven Development** lifecycle:

1. **Specification Alignment**: All changes must trace to [`CONSTITUTION.md`](CONSTITUTION.md) and a formal specification in [`specs/`](specs/).
2. **Fork & Feature Branch**: Create a dedicated branch (`feat/<spec-id>-<short-name>` or `fix/<issue-id>-<short-name>`). Direct pushes to `main` are blocked.
3. **Code Owner Review**: All files are protected by [`.github/CODEOWNERS`](.github/CODEOWNERS). Every Pull Request requires explicit review and approval from `@raklabs` before merging.
4. **Automated CI & Security Gate**: All Pull Requests must pass:
   - Secret & token hygiene scan
   - Deterministic dataset & scenario-isolation validation (`python scripts/poc.py validate`)
   - Runtime & CPU autograd verification (`python scripts/poc.py doctor`)
   - Full unit, API security, and offline Qwen2 LoRA integration test suite (`pytest -v`)

---

## 2. Security & Anti-Abuse Policy

- **Zero Secrets & Zero Pickle Weights**: Never commit API keys, `.env` files, or `.bin`/`.pt`/`.ckpt` pickle weight files. Only `safetensors` serialization is permitted by code, and binary weights are excluded from git.
- **Workflow Protection**: External fork Pull Requests require explicit maintainer approval before GitHub Actions workflows execute, and all workflows run with read-only `GITHUB_TOKEN` permissions (`contents: read`), concurrency cancellation, and a strict execution timeout.
- **Spam & Abuse**: Unsolicited promotional PRs, automated typo-farming PRs, or issues that bypass the structured issue templates will be closed and locked.
- **Vulnerability Disclosure**: Do **not** report security vulnerabilities in public issues or PRs. Follow [`SECURITY.md`](SECURITY.md) to submit a private report.

---

## 3. Local Verification Before Opening a PR

Run the full verification suite locally before submitting a Pull Request:

```bash
python scripts/poc.py validate
python scripts/poc.py doctor
pytest -v
```
