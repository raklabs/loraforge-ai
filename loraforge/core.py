"""Core auditable LoRA SFT engine, dataset integrity verifier, and merge validator."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# Keep downloads inside this project. Set before importing any Hugging Face modules.
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

LABELS: tuple[str, ...] = ("billing", "account", "technical", "other")
SYSTEM: str = (
    "Classify the support ticket. billing: payments, invoices, refunds, subscription charges. "
    "account: login, password, profile, account access. "
    "technical: product errors, broken features, performance. "
    "other: unrelated requests or insufficient information. "
    'Return only a JSON object with exactly one key, "category", whose value is '
    "billing, account, technical, or other."
)


def read_json(path: str | Path) -> Any:
    """Deserialize JSON from disk using UTF-8 encoding."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    """Serialize JSON to disk with deterministic 2-space indentation."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(path: str | Path) -> str:
    """Compute the lowercase hexadecimal SHA-256 digest of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config() -> dict[str, Any]:
    """Load and validate hyperparameter configuration from config.json."""
    cfg = read_json(ROOT / "config.json")
    for key in ("threads", "max_length", "max_new_tokens", "gradient_accumulation", "epochs", "lora_rank"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["micro_batch_size"] != 1:
        raise ValueError("This deterministic loop supports micro_batch_size=1 only")
    if not math.isfinite(cfg["learning_rate"]) or cfg["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")
    return cfg


def load_data() -> dict[str, list[dict[str, str]]]:
    """Load and cryptographically/structurally validate train, validation, and test splits."""
    data: dict[str, list[dict[str, str]]] = {}
    ids: set[str] = set()
    scenarios: dict[str, str] = {}
    texts: set[str] = set()
    for split, size in (("train", 160), ("validation", 40), ("test", 40)):
        rows = [
            json.loads(line)
            for line in (ROOT / "data" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != size:
            raise ValueError(f"{split}: expected {size} rows, got {len(rows)}")
        for row in rows:
            if set(row) != {"id", "scenario_id", "text", "category"}:
                raise ValueError(f"{split}: incorrect fields")
            if any(not isinstance(row[k], str) or not row[k].strip() for k in row):
                raise ValueError(f"{split}: all fields must be nonempty strings")
            if row["category"] not in LABELS:
                raise ValueError(f"Unknown label in {row['id']}")
            normalized = " ".join(row["text"].casefold().split())
            if row["id"] in ids or normalized in texts:
                raise ValueError(f"Duplicate ID or text: {row['id']}")
            if row["scenario_id"] in scenarios and scenarios[row["scenario_id"]] != split:
                raise ValueError(f"Scenario leakage: {row['scenario_id']}")
            ids.add(row["id"])
            texts.add(normalized)
            scenarios[row["scenario_id"]] = split
        if Counter(r["category"] for r in rows) != Counter({k: size // 4 for k in LABELS}):
            raise ValueError(f"{split}: categories must be balanced")
        data[split] = rows
    return data


def data_hashes() -> dict[str, str]:
    """Return SHA-256 digests for all three dataset splits."""
    return {split: digest(ROOT / "data" / f"{split}.jsonl") for split in ("train", "validation", "test")}


def lock_model(cfg: dict[str, Any]) -> dict[str, Any]:
    """Verify artifacts/model-lock.json against active config and dataset SHA-256 digests."""
    path = ROOT / "artifacts" / "model-lock.json"
    lock = read_json(path)
    if lock["model_id"] != cfg["model_id"]:
        raise ValueError("Model ID differs from prepared model; use a new experiment directory")
    if lock["data_hashes"] != data_hashes():
        raise ValueError("Data changed after preparation; restore the frozen dataset or start a new experiment")
    return lock


def runtime(cfg: dict[str, Any]) -> Any:
    """Seed Python and PyTorch RNGs and configure CPU thread count."""
    import torch

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["threads"])
    return torch


def versions() -> dict[str, str]:
    """Return pinned versions of core ML dependencies."""
    return {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "numpy",
            "transformers",
            "tokenizers",
            "peft",
            "accelerate",
            "huggingface-hub",
            "safetensors",
        )
    }


def all_packages() -> dict[str, str]:
    """Return sorted map of all installed Python distributions."""
    return dict(sorted((dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions()))


def _detect_hardware() -> dict[str, Any]:
    """Detect processor and memory without hardcoding machine-specific user strings."""
    processor = platform.processor() or platform.machine() or "unknown"
    ram_gib: float | None = None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        if isinstance(page_size, int) and isinstance(phys_pages, int) and page_size > 0 and phys_pages > 0:
            ram_gib = round((page_size * phys_pages) / (1024**3), 1)
    except (AttributeError, ValueError, OSError):
        ram_gib = None
    return {"processor": processor, "ram_gib": ram_gib, "source": "auto-detected"}


def metadata(cfg: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    """Build full provenance metadata dictionary for audit artifacts."""
    return {
        "config": cfg,
        "model_id": lock["model_id"],
        "revision": lock["revision"],
        "data_hashes": data_hashes(),
        "system_prompt_sha256": hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": "cpu",
        "precision": "float32",
        "hardware": _detect_hardware(),
        "packages": versions(),
        "all_packages": all_packages(),
    }


def load_model(cfg: dict[str, Any], lock: dict[str, Any], adapter: str | Path | None = None) -> tuple[Any, Any, Any]:
    """Load frozen base model (and optional LoRA adapter) strictly from local safetensors."""
    torch = runtime(cfg)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = lock["snapshot"]
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
        use_safetensors=True,
    ).to("cpu")
    if adapter:
        from peft import PeftModel

        adapter_path = Path(adapter).resolve()
        run = read_json(adapter_path.parent / "run.json")
        expected = metadata(cfg, lock)
        # Generation/evaluation settings and data must stay identical for a fair comparison.
        for key in ("config", "model_id", "revision", "data_hashes", "system_prompt_sha256"):
            if run[key] != expected[key]:
                raise ValueError(f"Adapter experiment mismatch: {key}")
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    return torch, tokenizer, model


def messages(text: str, category: str | None = None) -> list[dict[str, str]]:
    """Build chat template messages for prompt or supervised training pair."""
    result = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    if category is not None:
        result.append({"role": "assistant", "content": json.dumps({"category": category}, separators=(",", ":"))})
    return result


def encode_example(tokenizer: Any, row: dict[str, str], max_length: int) -> dict[str, list[int]]:
    """Encode a ticket and mask prompt tokens with -100 so only assistant response tokens are supervised."""
    prompt = tokenizer.apply_chat_template(messages(row["text"]), tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        messages(row["text"], row["category"]), tokenize=True, add_generation_prompt=False
    )
    if full[: len(prompt)] != prompt:
        raise ValueError("Chat template prefix mismatch; cannot safely mask assistant tokens")
    if len(full) > max_length:
        raise ValueError(f"{row['id']}: {len(full)} tokens exceed {max_length}; shorten example, never truncate target")
    if len(full) <= len(prompt):
        raise ValueError("Example has no assistant target")
    return {"input_ids": full, "labels": [-100] * len(prompt) + full[len(prompt) :]}


def tensor_batch(torch: Any, example: dict[str, list[int]]) -> dict[str, Any]:
    """Convert an encoded example into a micro-batch of batch size 1."""
    ids = torch.tensor([example["input_ids"]], dtype=torch.long)
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": torch.tensor([example["labels"]], dtype=torch.long),
    }


def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """JSON object_pairs_hook that rejects duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def parse_prediction(text: str) -> str | None:
    """Strictly validate model output as '{"category": "<label>"}' with no duplicate keys or prose."""
    try:
        value = json.loads(text, object_pairs_hook=unique_pairs)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or set(value) != {"category"}:
        return None
    category = value["category"]
    return category if isinstance(category, str) and category in LABELS else None


def score(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute schema validity, overall accuracy, and per-category accuracy."""
    if not predictions:
        raise ValueError("Cannot score an empty set")
    valid = sum(p["predicted"] is not None for p in predictions)
    correct = sum(p["predicted"] == p["expected"] for p in predictions)
    by_category: dict[str, dict[str, Any]] = {}
    for label in LABELS:
        rows = [p for p in predictions if p["expected"] == label]
        n = sum(p["predicted"] == label for p in rows)
        by_category[label] = {"correct": n, "total": len(rows), "accuracy": n / len(rows) if rows else None}
    return {
        "total": len(predictions),
        "valid": valid,
        "correct": correct,
        "schema_validity": valid / len(predictions),
        "accuracy": correct / len(predictions),
        "per_category": by_category,
    }


def generate(torch: Any, tokenizer: Any, model: Any, cfg: dict[str, Any], text: str) -> str:
    """Run deterministic greedy decoding (do_sample=False) for a single support ticket."""
    prompt = tokenizer.apply_chat_template(messages(text), tokenize=True, add_generation_prompt=True)
    if len(prompt) + cfg["max_new_tokens"] > cfg["max_length"]:
        raise ValueError("Prompt plus generation budget exceeds max_length; shorten the ticket")
    ids = torch.tensor([prompt], dtype=torch.long)
    model.eval()
    with torch.inference_mode():
        output = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            max_new_tokens=cfg["max_new_tokens"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    return tokenizer.decode(output[0, len(prompt) :], skip_special_tokens=True)


def evaluate_rows(
    torch: Any, tokenizer: Any, model: Any, cfg: dict[str, Any], rows: list[dict[str, str]]
) -> dict[str, Any]:
    """Evaluate a split of rows and return metrics, predictions, and elapsed time."""
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    for i, row in enumerate(rows, 1):
        raw = generate(torch, tokenizer, model, cfg, row["text"])
        predictions.append(
            {
                "id": row["id"],
                "text": row["text"],
                "expected": row["category"],
                "raw": raw,
                "predicted": parse_prediction(raw),
            }
        )
        print(f"Generated {i}/{len(rows)}", flush=True)
    return {
        "metrics": score(predictions),
        "predictions": predictions,
        "elapsed_seconds": time.perf_counter() - started,
    }


def validation_loss(torch: Any, model: Any, examples: list[dict[str, list[int]]]) -> float:
    """Compute token-weighted cross-entropy loss across assistant tokens on validation examples."""
    model.eval()
    loss_sum, token_count = 0.0, 0
    with torch.inference_mode():
        for example in examples:
            output = model(**tensor_batch(torch, example), use_cache=False)
            loss = output.loss.item()
            if not math.isfinite(loss):
                raise ValueError("Non-finite validation loss")
            n = sum(token != -100 for token in example["labels"][1:])
            loss_sum += loss * n
            token_count += n
    return loss_sum / token_count


def doctor(_: Any = None) -> None:
    """Verify Python runtime, package versions, and CPU autograd forward/backward pass."""
    print(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "device": "cpu",
                "macos": platform.mac_ver()[0],
                "config": config(),
            },
            indent=2,
        )
    )
    try:
        print(json.dumps(versions(), indent=2))
        import torch

        a = torch.tensor([2.0], requires_grad=True)
        a.square().sum().backward()
        if a.grad is None or a.grad.item() != 4.0:
            raise ValueError("CPU autograd verification failed")
        print("CPU forward/backward check passed. Model compatibility is checked by prepare/smoke.")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise ValueError(f"Install requirements using Python 3.11+ first: {exc}") from exc


def validate(_: Any = None) -> None:
    """Validate dataset schema, split counts, label balance, and scenario isolation."""
    data = load_data()
    print(
        json.dumps(
            {
                split: {"count": len(rows), "categories": dict(Counter(r["category"] for r in rows))}
                for split, rows in data.items()
            },
            indent=2,
        )
    )
    print("Structural checks passed. Semantic overlap and label quality require human review; see data/README.md.")


def prepare(_: Any = None) -> None:
    """Resolve immutable 40-char Hugging Face commit SHA, download safetensors, and write model-lock.json."""
    cfg, data = config(), load_data()
    from huggingface_hub import HfApi, snapshot_download

    lock_path = ROOT / "artifacts" / "model-lock.json"
    if lock_path.exists():
        lock = lock_model(cfg)
    else:
        revision = HfApi().model_info(cfg["model_id"]).sha
        if not revision or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("The Hub did not return a full immutable model commit")
        source = snapshot_download(
            cfg["model_id"],
            revision=revision,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        )
        lock = {
            "model_id": cfg["model_id"],
            "revision": revision,
            "snapshot": source,
            "data_hashes": data_hashes(),
        }
    torch, tokenizer, model = load_model(cfg, lock)
    lengths: dict[str, dict[str, int]] = {}
    for split, rows in data.items():
        examples = [encode_example(tokenizer, r, cfg["max_length"]) for r in rows]
        for row in rows:
            prompt = tokenizer.apply_chat_template(messages(row["text"]), tokenize=True, add_generation_prompt=True)
            if len(prompt) + cfg["max_new_tokens"] > cfg["max_length"]:
                raise ValueError(f"{row['id']}: insufficient generation budget")
        lengths[split] = {
            "min": min(len(e["input_ids"]) for e in examples),
            "max": max(len(e["input_ids"]) for e in examples),
        }
    example = encode_example(tokenizer, data["train"][0], cfg["max_length"])
    target = [token for token in example["labels"] if token != -100]
    write_json(
        ROOT / "artifacts" / "formatted-example.json",
        {
            "messages": messages(data["train"][0]["text"], data["train"][0]["category"]),
            **example,
            "supervised_text": tokenizer.decode(target),
            "lengths": lengths,
            "note": "-100 labels are ignored by cross-entropy; only assistant response and its end marker contribute.",
        },
    )
    write_json(lock_path, lock)
    write_json(ROOT / "artifacts" / "environment.json", metadata(cfg, lock))
    print(
        json.dumps(
            {
                "revision": lock["revision"],
                "lengths": lengths,
                "supervised_text": tokenizer.decode(target),
            },
            indent=2,
        )
    )


def train(args: argparse.Namespace) -> None:
    """Run smoke or full LoRA supervised fine-tuning with explicit token-weighted loss and reload checks."""
    cfg, data = config(), load_data()
    lock = lock_model(cfg)
    name = "smoke" if args.smoke else "train"
    if not args.smoke:
        smoke = read_json(ROOT / "artifacts" / "smoke" / "run.json")
        if smoke.get("status") != "complete" or not smoke.get("fresh_reload_passed"):
            raise ValueError("Complete a successful smoke run before full training")
        for key in ("config", "revision", "data_hashes", "system_prompt_sha256"):
            if smoke[key] != metadata(cfg, lock)[key]:
                raise ValueError(f"Smoke run differs from current experiment: {key}; archive it and rerun smoke")
    out = ROOT / "artifacts" / name
    if out.exists():
        raise ValueError(f"{out} already exists; archive it before a new run")
    torch, tokenizer, model = load_model(cfg, lock)
    from peft import LoraConfig, TaskType, get_peft_model

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg["lora_rank"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["target_modules"],
            bias="none",
        ),
    )
    model.config.use_cache = False
    params = [p for p in model.parameters() if p.requires_grad]
    count = sum(p.numel() for p in params)
    total = sum(p.numel() for p in model.parameters())
    if not 0 < count < total:
        raise ValueError("Expected a nonzero strict subset of trainable parameters")
    print(f"Trainable parameters: {count:,}/{total:,} ({100 * count / total:.3f}%)", flush=True)
    train_rows = data["train"][:3] if args.smoke else data["train"]
    val_rows = data["validation"][:4] if args.smoke else data["validation"]
    examples = [encode_example(tokenizer, row, cfg["max_length"]) for row in train_rows]
    val_examples = [encode_example(tokenizer, row, cfg["max_length"]) for row in val_rows]
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=0.0)
    out.mkdir(parents=True)
    run = metadata(cfg, lock)
    run.update(
        {
            "smoke": args.smoke,
            "trainable_parameters": count,
            "total_parameters": total,
            "effective_batch_size": cfg["gradient_accumulation"],
            "status": "running",
        }
    )
    write_json(out / "run.json", run)
    history: list[dict[str, Any]] = []
    micro_seconds: list[float] = []
    best = math.inf
    updates = 0
    started = time.perf_counter()
    result = None
    try:
        for epoch in range(1 if args.smoke else cfg["epochs"]):
            order = list(range(len(examples)))
            random.Random(cfg["seed"] + epoch).shuffle(order)
            model.train()
            weighted_loss, tokens = 0.0, 0
            for start in range(0, len(order), cfg["gradient_accumulation"]):
                group = order[start : start + cfg["gradient_accumulation"]]
                optimizer.zero_grad(set_to_none=True)
                for index in group:
                    tick = time.perf_counter()
                    example = examples[index]
                    result = model(**tensor_batch(torch, example), use_cache=False)
                    value = result.loss.item()
                    if not math.isfinite(value):
                        raise ValueError("Non-finite training loss")
                    (result.loss / len(group)).backward()
                    n = sum(t != -100 for t in example["labels"][1:])
                    weighted_loss += value * n
                    tokens += n
                    micro_seconds.append(time.perf_counter() - tick)
                    print(f"Epoch {epoch + 1}, example {len(micro_seconds)}, loss {value:.4f}", flush=True)
                norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0, error_if_nonfinite=True)
                if updates == 0 and norm.item() == 0:
                    raise ValueError("Adapters received zero gradient")
                optimizer.step()
                updates += 1
            val_started = time.perf_counter()
            val_loss = validation_loss(torch, model, val_examples)
            history.append(
                {
                    "epoch": epoch + 1,
                    "training_loss": weighted_loss / tokens,
                    "validation_loss": val_loss,
                    "validation_seconds": time.perf_counter() - val_started,
                }
            )
            if val_loss < best:
                best = val_loss
                model.save_pretrained(out / "adapter", safe_serialization=True)
                tokenizer.save_pretrained(out / "adapter")
            write_json(out / "history.json", history)
        run.update(
            {
                "status": "complete",
                "optimizer_steps": updates,
                "best_validation_loss": best,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_process_memory_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024 ** (3 if sys.platform == "darwin" else 2)),
                "microstep_seconds": micro_seconds,
            }
        )
        if args.smoke:
            average = sum(micro_seconds[1:]) / len(micro_seconds[1:])
            generate(torch, tokenizer, model, cfg, data["validation"][0]["text"])
            gen_started = time.perf_counter()
            generate(torch, tokenizer, model, cfg, data["validation"][0]["text"])
            generate(torch, tokenizer, model, cfg, data["validation"][1]["text"])
            seconds_per_request = (time.perf_counter() - gen_started) / 2
            run["timing_estimate"] = {
                "training_microstep_seconds": average,
                "one_epoch_training_seconds_excluding_validation": average * len(data["train"]),
                "one_validation_loss_pass_seconds": history[-1]["validation_seconds"]
                * len(data["validation"])
                / len(val_rows),
                "generation_seconds_per_request": seconds_per_request,
                "final_comparison_generation_seconds": seconds_per_request * 2 * len(data["test"]),
                "note": "Tiny sample; includes no model load time and is sensitive to CPU heat, lengths, and background work.",
            }
        write_json(out / "run.json", run)
    except Exception:
        run["status"] = "failed"
        write_json(out / "run.json", run)
        raise
    print(json.dumps(run, indent=2), flush=True)
    del optimizer, params, model, result
    gc.collect()
    if args.smoke:
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "poc.py"),
                    "infer",
                    "--adapter",
                    str(out / "adapter"),
                    "--text",
                    "The login page rejects my password.",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            (out / "reload-check.txt").write_text(completed.stdout, encoding="utf-8")
            run["fresh_reload_passed"] = True
        except subprocess.CalledProcessError as exc:
            run.update({"status": "failed", "fresh_reload_passed": False, "reload_error": exc.stderr})
            raise
        finally:
            write_json(out / "run.json", run)
        print("Fresh-process adapter reload passed.", flush=True)


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate baseline or LoRA-adapted model on validation or held-out test split."""
    if args.split == "test" and not args.final:
        raise ValueError(
            "Use --final only after selecting the checkpoint from validation data; do not tune on test results"
        )
    cfg, data = config(), load_data()
    lock = lock_model(cfg)
    variant = "adapted" if args.adapter else "baseline"
    path = ROOT / "artifacts" / f"{variant}-{args.split}.json"
    if path.exists():
        raise ValueError(f"{path} exists; preserve the previous evaluation before rerunning")
    torch, tokenizer, model = load_model(cfg, lock, args.adapter)
    result = evaluate_rows(torch, tokenizer, model, cfg, data[args.split])
    result.update(
        {
            "metadata": metadata(cfg, lock),
            "variant": variant,
            "split": args.split,
            "adapter": str(Path(args.adapter).resolve()) if args.adapter else None,
        }
    )
    if args.adapter:
        result["adapter_sha256"] = digest(Path(args.adapter) / "adapter_model.safetensors")
    write_json(path, result)
    print(json.dumps(result["metrics"], indent=2))


def infer(args: argparse.Namespace) -> None:
    """Run single-ticket inference against base model, LoRA adapter, or merged standalone export."""
    cfg = config()
    if getattr(args, "merged", None):
        torch, tokenizer, model = load_merged(cfg, args.merged)
    else:
        lock = lock_model(cfg)
        torch, tokenizer, model = load_model(cfg, lock, args.adapter)
    raw = generate(torch, tokenizer, model, cfg, args.text)
    print(json.dumps({"raw": raw, "parsed_category": parse_prediction(raw)}, indent=2))


def load_merged(cfg: dict[str, Any], path: str | Path) -> tuple[Any, Any, Any]:
    """Load only the exported standalone directory, without the base snapshot or PEFT adapter."""
    resolved = Path(path).resolve()
    export = read_json(resolved / "export.json")
    if export["config"] != cfg or export["system_prompt_sha256"] != hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest():
        raise ValueError("Merged model configuration/prompt differs from this experiment")
    torch = runtime(cfg)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(resolved, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            resolved,
            torch_dtype=torch.float32,
            attn_implementation="eager",
            local_files_only=True,
            use_safetensors=True,
        )
        .to("cpu")
        .eval()
    )
    return torch, tokenizer, model


def verify_merge_logits(torch: Any, actual: Any, expected: Any) -> dict[str, float]:
    """Verify numerical parity between PEFT adapter logits and merged standalone model logits."""
    rtol, atol = 1e-4, 1e-3
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("Merge logit verification found non-finite values. No model has been exported.")
    difference = (actual - expected).abs()
    stats = {
        "logits_rtol": rtol,
        "logits_atol": atol,
        "max_logit_difference": difference.max().item(),
        "mean_logit_difference": difference.mean().item(),
    }
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
    except AssertionError as exc:
        raise ValueError(
            f"Merge logit verification failed (max absolute difference "
            f"{stats['max_logit_difference']:.6g}, rtol={rtol}, atol={atol}). "
            f"No model has been exported. Details: {exc}"
        ) from exc
    return stats


def merge(args: argparse.Namespace) -> None:
    """Fold LoRA weights into base weights, verify logit parity, and validate fresh-process reload."""
    cfg = config()
    output, adapter = args.output.resolve(), args.adapter.resolve()
    if output.exists():
        raise ValueError(f"{output} exists; choose a new output directory")
    run = read_json(adapter.parent / "run.json")
    if run.get("status") != "complete" or run.get("smoke"):
        raise ValueError("Merge requires an adapter from a complete full training run")
    lock = lock_model(cfg)
    torch, tokenizer, model = load_model(cfg, lock, adapter)
    model.eval()
    # Validate conversion on a validation ticket, never use test data for export checks.
    row = load_data()["validation"][0]
    prompt = tokenizer.apply_chat_template(messages(row["text"]), tokenize=True, add_generation_prompt=True)
    ids = torch.tensor([prompt], dtype=torch.long)
    expected_raw = generate(torch, tokenizer, model, cfg, row["text"])
    with torch.inference_mode():
        expected_logits = model(input_ids=ids, use_cache=False).logits[:, -1, :].clone()
        merged = model.merge_and_unload(safe_merge=True).eval()
        actual_logits = merged(input_ids=ids, use_cache=False).logits[:, -1, :]
        logit_check = verify_merge_logits(torch, actual_logits, expected_logits)
    merged.config.use_cache = True
    merged.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    export = {
        **metadata(cfg, lock),
        "system_prompt": SYSTEM,
        "status": "pending_reload",
        "adapter_sha256": digest(adapter / "adapter_model.safetensors"),
        "check_ticket_id": row["id"],
        **logit_check,
    }
    write_json(output / "export.json", export)
    del model, merged
    gc.collect()
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "poc.py"),
                "infer",
                "--merged",
                str(output),
                "--text",
                row["text"],
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        if result["raw"] != expected_raw:
            raise ValueError(
                "Merged reload generated a different response; inspect numerical differences before using the export"
            )
        export.update(
            {
                "status": "complete",
                "fresh_reload_passed": True,
                "expected_raw": expected_raw,
                "reloaded_raw": result["raw"],
            }
        )
    except Exception as exc:
        export.update({"status": "failed", "error": str(exc)})
        raise
    finally:
        write_json(output / "export.json", export)
    print(f"Standalone model saved and fresh-process reload verified: {output}")


def report(_: Any = None) -> None:
    """Verify baseline vs adapted provenance parity and generate artifacts/report.md."""
    base = read_json(ROOT / "artifacts" / "baseline-test.json")
    adapted = read_json(ROOT / "artifacts" / "adapted-test.json")
    run = read_json(ROOT / "artifacts" / "train" / "run.json")
    if run["status"] != "complete" or run["smoke"]:
        raise ValueError("A complete full training run is required")
    for key in ("config", "model_id", "revision", "data_hashes", "system_prompt_sha256", "packages"):
        if not base["metadata"][key] == adapted["metadata"][key] == run[key]:
            raise ValueError(f"Comparison metadata mismatch: {key}")
    if adapted["adapter_sha256"] != digest(ROOT / "artifacts" / "train" / "adapter" / "adapter_model.safetensors"):
        raise ValueError("Adapted evaluation did not use the saved training adapter")
    if [p["id"] for p in base["predictions"]] != [p["id"] for p in adapted["predictions"]]:
        raise ValueError("Prediction order/IDs differ")
    lines = [
        "# Experiment results",
        "",
        f"Model: `{run['model_id']}` at `{run['revision']}`.",
        "",
        "| Metric | Original | LoRA adapter |",
        "| --- | --- | --- |",
    ]
    for metric, count in (("schema_validity", "valid"), ("accuracy", "correct")):
        a, b = base["metrics"], adapted["metrics"]
        lines.append(f"| {metric} | {a[count]}/{a['total']} ({a[metric]:.1%}) | {b[count]}/{b['total']} ({b[metric]:.1%}) |")
    for label in LABELS:
        a, b = base["metrics"]["per_category"][label], adapted["metrics"]["per_category"][label]
        lines.append(
            f"| {label} accuracy | {a['correct']}/{a['total']} ({a['accuracy']:.1%}) | "
            f"{b['correct']}/{b['total']} ({b['accuracy']:.1%}) |"
        )
    delta = adapted["metrics"]["accuracy"] - base["metrics"]["accuracy"]
    lines += [
        "",
        f"Observed accuracy change: {delta * 100:+.1f} percentage points on 40 synthetic test tickets.",
        "This small test does not establish production quality or statistical significance.",
        "",
        f"Training time: {run['elapsed_seconds']:.1f} seconds. Peak process memory: {run['peak_process_memory_gib']:.2f} GiB.",
        "",
        "## Examples to discuss",
        "",
    ]
    pairs = list(zip(base["predictions"], adapted["predictions"]))
    pairs.sort(key=lambda p: (p[1]["predicted"] == p[1]["expected"], p[0]["predicted"] == p[0]["expected"]))
    for a, b in pairs[:5]:
        lines += [
            f"### {a['id']}",
            "",
            a["text"],
            "",
            f"Expected: `{a['expected']}`.",
            f"Original raw output: `{json.dumps(a['raw'])}`.",
            f"Adapter raw output: `{json.dumps(b['raw'])}`.",
            "",
            "Learner notes: explain the prediction, any format failure, and what training evidence might help.",
            "",
        ]
    lines += [
        "## Reflection to complete",
        "",
        "- Did validation loss and test accuracy tell the same story?",
        "- Which category or wording caused errors? Could label ambiguity explain them?",
        "- Did the model already solve this task before training?",
        "- Choose one next experiment using validation results; reserve a new test set if you tune after inspecting this one.",
        "",
    ]
    path = ROOT / "artifacts" / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)


def _serve_cmd(args: argparse.Namespace) -> None:
    """Launch the FastAPI inference microservice."""
    import uvicorn
    from loraforge.serve import create_app_from_paths

    app = create_app_from_paths(adapter=args.adapter, merged=args.merged)
    uvicorn.run(app, host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point supporting doctor, validate, prepare, train, evaluate, infer, merge, report, and serve."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, function in (("doctor", doctor), ("validate", validate), ("prepare", prepare), ("report", report)):
        sub.add_parser(name).set_defaults(function=function)

    p = sub.add_parser("train")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="3 training examples, 4 validation examples, benchmark and fresh reload",
    )
    p.set_defaults(function=train)

    p = sub.add_parser("evaluate")
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--adapter", type=Path)
    p.add_argument("--final", action="store_true")
    p.set_defaults(function=evaluate)

    p = sub.add_parser("infer")
    p.add_argument("--text", required=True)
    sources = p.add_mutually_exclusive_group()
    sources.add_argument("--adapter", type=Path)
    sources.add_argument("--merged", type=Path, help="Standalone exported model directory")
    p.set_defaults(function=infer)

    p = sub.add_parser("merge", help="Export original model plus trained LoRA as a standalone model")
    p.add_argument("--adapter", type=Path, default=ROOT / "artifacts" / "train" / "adapter")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts" / "merged-model")
    p.set_defaults(function=merge)

    p = sub.add_parser("serve", help="Run the FastAPI support ticket classification microservice")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    serve_sources = p.add_mutually_exclusive_group()
    serve_sources.add_argument("--adapter", type=Path)
    serve_sources.add_argument("--merged", type=Path, help="Standalone exported model directory")
    p.set_defaults(function=_serve_cmd)

    args = parser.parse_args(argv)
    try:
        args.function(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
