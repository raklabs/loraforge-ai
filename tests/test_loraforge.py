"""Comprehensive unit, security, API, and offline LoRA integration test suite for LoraForge AI."""
from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT_DIR / "scripts"
for p in (str(ROOT_DIR), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import create_dataset  # noqa: E402
import poc  # noqa: E402

FASTAPI_AVAILABLE = all(importlib.util.find_spec(m) is not None for m in ("fastapi", "pydantic", "httpx"))


class FakeTokenizer:
    """A tiny deterministic template: prompt [1, 2, 3], target [4, 5]."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [1, 2, 3] if add_generation_prompt else [1, 2, 3, 4, 5]


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        create_dataset.build(self.root / "data")
        self.root_patch = patch.object(poc, "ROOT", self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def mutate(self, split, edit):
        path = self.root / "data" / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        edit(rows)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_committed_data_and_reproduction(self):
        rows = poc.load_data()
        self.assertEqual([len(rows[k]) for k in ("train", "validation", "test")], [160, 40, 40])
        for split in rows:
            self.assertEqual(
                (self.root / "data" / f"{split}.jsonl").read_bytes(),
                (SCRIPTS.parent / "data" / f"{split}.jsonl").read_bytes(),
            )

    def test_scenario_leakage(self):
        scenario = poc.load_data()["train"][0]["scenario_id"]
        self.mutate("test", lambda rows: rows[0].update(scenario_id=scenario))
        with self.assertRaisesRegex(ValueError, "leakage"):
            poc.load_data()

    def test_duplicate_text_case_and_whitespace(self):
        text = poc.load_data()["train"][0]["text"]
        self.mutate("validation", lambda rows: rows[0].update(text="  " + text.upper() + "  "))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            poc.load_data()

    def test_invalid_label(self):
        self.mutate("train", lambda rows: rows[0].update(category="urgent"))
        with self.assertRaisesRegex(ValueError, "Unknown label"):
            poc.load_data()

    def test_empty_text(self):
        self.mutate("train", lambda rows: rows[0].update(text=" "))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            poc.load_data()

    def test_balance(self):
        self.mutate("train", lambda rows: rows[0].update(category="billing"))
        with self.assertRaisesRegex(ValueError, "balanced"):
            poc.load_data()

    def test_frozen_data_tampering(self):
        poc.write_json(
            self.root / "artifacts" / "model-lock.json",
            {"model_id": "fixture", "data_hashes": poc.data_hashes()},
        )
        self.mutate("test", lambda rows: rows[0].update(text="An edited ticket"))
        with self.assertRaisesRegex(ValueError, "Data changed"):
            poc.lock_model({"model_id": "fixture"})


class ScoringTests(unittest.TestCase):
    def test_strict_json(self):
        self.assertEqual(poc.parse_prediction(' {"category": "billing"}\n'), "billing")
        for raw in (
            '```json\n{"category":"billing"}\n```',
            '{"category":"billing","reason":"x"}',
            '{"category":"BILLING"}',
            '{"category":null}',
            '{"category":[]}',
            '[{"category":"billing"}]',
            'Here: {"category":"billing"}',
            '{"category":"account","category":"billing"}',
            '{"category":"billing"} trailing',
        ):
            with self.subTest(raw=raw):
                self.assertIsNone(poc.parse_prediction(raw))

    def test_invalid_output_counts_as_incorrect(self):
        result = poc.score(
            [
                {"predicted": "billing", "expected": "billing"},
                {"predicted": None, "expected": "account"},
                {"predicted": "other", "expected": "technical"},
            ]
        )
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["total"], 3)
        self.assertAlmostEqual(result["schema_validity"], 2 / 3)
        self.assertEqual(result["per_category"]["account"]["accuracy"], 0)

    def test_empty_scores_fail(self):
        with self.assertRaises(ValueError):
            poc.score([])


class MaskTests(unittest.TestCase):
    row = {"id": "example", "text": "Hi", "category": "other"}

    def test_only_response_tokens_supervised(self):
        encoded = poc.encode_example(FakeTokenizer(), self.row, 10)
        self.assertEqual(encoded["labels"], [-100, -100, -100, 4, 5])
        self.assertEqual(encoded["input_ids"], [1, 2, 3, 4, 5])

    def test_truncation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "never truncate"):
            poc.encode_example(FakeTokenizer(), self.row, 4)

    def test_mismatched_template_is_rejected(self):
        tokenizer = FakeTokenizer()
        tokenizer.apply_chat_template = (
            lambda messages, tokenize, add_generation_prompt: [1, 2, 3] if add_generation_prompt else [9, 4, 5]
        )
        with self.assertRaisesRegex(ValueError, "prefix mismatch"):
            poc.encode_example(tokenizer, self.row, 10)

    def test_test_split_requires_final_flag(self):
        with self.assertRaisesRegex(ValueError, "--final"):
            poc.evaluate(argparse.Namespace(split="test", final=False, adapter=None))


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.root_patch = patch.object(poc, "ROOT", self.root)
        self.root_patch.start()
        create_dataset.build(self.root / "data")
        metadata = {
            "config": {"seed": 42},
            "model_id": "test-fixture",
            "revision": "a" * 40,
            "data_hashes": poc.data_hashes(),
            "system_prompt_sha256": "fixture",
            "packages": {},
        }
        rows = poc.load_data()["test"]
        base_predictions = [
            {
                "id": r["id"],
                "text": r["text"],
                "expected": r["category"],
                "raw": "invalid fixture output",
                "predicted": None,
            }
            for r in rows
        ]
        adapted_predictions = [
            {
                "id": r["id"],
                "text": r["text"],
                "expected": r["category"],
                "raw": json.dumps({"category": r["category"]}),
                "predicted": r["category"],
            }
            for r in rows
        ]
        adapter = self.root / "artifacts" / "train" / "adapter" / "adapter_model.safetensors"
        adapter.parent.mkdir(parents=True)
        adapter.write_bytes(b"test fixture, not model weights")
        self.adapted = {
            "metadata": metadata,
            "metrics": poc.score(adapted_predictions),
            "predictions": adapted_predictions,
            "adapter_sha256": poc.digest(adapter),
        }
        poc.write_json(
            self.root / "artifacts" / "baseline-test.json",
            {"metadata": metadata, "metrics": poc.score(base_predictions), "predictions": base_predictions},
        )
        poc.write_json(self.root / "artifacts" / "adapted-test.json", self.adapted)
        poc.write_json(
            self.root / "artifacts" / "train" / "run.json",
            {**metadata, "status": "complete", "smoke": False, "elapsed_seconds": 1, "peak_process_memory_gib": 1},
        )

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def test_report_comparison_counts_and_reflection(self):
        with contextlib.redirect_stdout(io.StringIO()):
            poc.report(None)
        result = (self.root / "artifacts" / "report.md").read_text(encoding="utf-8")
        self.assertIn("0/40 (0.0%)", result)
        self.assertIn("40/40 (100.0%)", result)
        self.assertEqual(result.count("Learner notes:"), 5)
        self.assertIn("Reflection to complete", result)

    def test_mismatched_evaluations_rejected(self):
        self.adapted["metadata"]["revision"] = "b" * 40
        poc.write_json(self.root / "artifacts" / "adapted-test.json", self.adapted)
        with self.assertRaisesRegex(ValueError, "metadata mismatch: revision"):
            poc.report(None)

    def test_different_adapter_rejected(self):
        self.adapted["adapter_sha256"] = "not-the-training-adapter"
        poc.write_json(self.root / "artifacts" / "adapted-test.json", self.adapted)
        with self.assertRaisesRegex(ValueError, "saved training adapter"):
            poc.report(None)


@unittest.skipUnless(FASTAPI_AVAILABLE, "Install fastapi, pydantic, and httpx to run API tests")
class ServingAndSecurityTests(unittest.TestCase):
    """Tests for the FastAPI inference microservice, input sanitization, and API key auth."""

    def test_control_character_sanitization(self):
        from loraforge.serve import sanitize_ticket_text

        dirty = "My card\x00 was \x07charged\x1f   twice\nfor the monthly plan."
        self.assertEqual(sanitize_ticket_text(dirty), "My card was charged twice for the monthly plan.")

    def test_healthz_and_classify_endpoints(self):
        from fastapi.testclient import TestClient
        from loraforge.serve import create_app

        app = create_app(
            predictor=lambda text: '{"category":"billing"}',
            model_variant="test-merged-qwen",
        )
        client = TestClient(app)

        health = client.get("/healthz")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")
        self.assertEqual(health.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(health.headers["X-Frame-Options"], "DENY")

        resp = client.post("/v1/classify", json={"text": "My card was charged twice."})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["category"], "billing")
        self.assertTrue(body["schema_valid"])
        self.assertEqual(body["model_variant"], "test-merged-qwen")

    def test_api_key_authentication_enforced_when_configured(self):
        from fastapi.testclient import TestClient
        from loraforge.serve import create_app

        app = create_app(
            predictor=lambda text: '{"category":"account"}',
            model_variant="secured-model",
            api_key="test-secret-key-999",
        )
        client = TestClient(app)

        unauth = client.post("/v1/classify", json={"text": "I forgot my password."})
        self.assertEqual(unauth.status_code, 401)

        wrong = client.post(
            "/v1/classify",
            json={"text": "I forgot my password."},
            headers={"X-API-Key": "wrong-key"},
        )
        self.assertEqual(wrong.status_code, 401)

        valid = client.post(
            "/v1/classify",
            json={"text": "I forgot my password."},
            headers={"X-API-Key": "test-secret-key-999"},
        )
        self.assertEqual(valid.status_code, 200)
        self.assertEqual(valid.json()["category"], "account")

    def test_rejects_empty_or_extra_fields(self):
        from fastapi.testclient import TestClient
        from loraforge.serve import create_app

        app = create_app()
        client = TestClient(app)

        empty_resp = client.post("/v1/classify", json={"text": "   \x00\x07  "})
        self.assertEqual(empty_resp.status_code, 422)

        extra_resp = client.post("/v1/classify", json={"text": "Hello", "injected": "field"})
        self.assertEqual(extra_resp.status_code, 422)


ML_AVAILABLE = all(importlib.util.find_spec(module) is not None for module in ("torch", "transformers", "peft"))


@unittest.skipUnless(ML_AVAILABLE, "Install requirements to run the tiny offline LoRA integration test")
class LoraIntegrationTests(unittest.TestCase):
    def test_merge_tolerates_small_rounding_but_rejects_corruption(self):
        import torch

        expected = torch.tensor([[0.0, 0.01, -0.01, 25.0]])
        actual = expected + 0.000208
        stats = poc.verify_merge_logits(torch, actual, expected)
        self.assertLess(stats["max_logit_difference"], stats["logits_atol"])
        for bad in (
            expected + 0.1,
            torch.full_like(expected, float("nan")),
            torch.full_like(expected, float("inf")),
        ):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "No model has been exported"):
                    poc.verify_merge_logits(torch, bad, expected)
        infinite = torch.full_like(expected, float("inf"))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            poc.verify_merge_logits(torch, infinite, infinite)

    def test_frozen_base_adapter_update_and_reload(self):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(42)
        cfg = Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
            attention_dropout=0.0,
        )
        original = Qwen2ForCausalLM(cfg)
        original_state = copy.deepcopy(original.state_dict())
        model = get_peft_model(
            original,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=2,
                lora_alpha=4,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.0,
            ),
        )
        frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
        params = [p for p in model.parameters() if p.requires_grad]
        initial = [p.detach().clone() for p in params]
        batch = poc.tensor_batch(torch, {"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, -100, 4, 5]})
        loss = model(**batch).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in params if p.grad is not None), 0)
        torch.optim.AdamW(params, lr=0.01).step()
        self.assertTrue(any(not torch.equal(before, after) for before, after in zip(initial, params)))
        for name, p in model.named_parameters():
            if name in frozen:
                self.assertTrue(torch.equal(p, frozen[name]), name)
        model.eval()
        with torch.no_grad():
            expected = model(input_ids=batch["input_ids"]).logits
        with tempfile.TemporaryDirectory() as path:
            model.save_pretrained(path, safe_serialization=True)
            base = Qwen2ForCausalLM(cfg)
            base.load_state_dict(original_state)
            restored = PeftModel.from_pretrained(base, path).eval()
            with torch.no_grad():
                actual = restored(input_ids=batch["input_ids"]).logits
            torch.testing.assert_close(actual, expected)
            merged = restored.merge_and_unload(safe_merge=True).eval()
            with torch.no_grad():
                merged_logits = merged(input_ids=batch["input_ids"]).logits
            torch.testing.assert_close(merged_logits, expected, rtol=1e-4, atol=1e-4)
            export = str(Path(path) / "standalone")
            merged.save_pretrained(export, safe_serialization=True)
            reloaded = Qwen2ForCausalLM.from_pretrained(export, local_files_only=True).eval()
            with torch.no_grad():
                reloaded_logits = reloaded(input_ids=batch["input_ids"]).logits
            torch.testing.assert_close(reloaded_logits, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
