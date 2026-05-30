"""Auditable CPU-only Qwen LoRA fine-tuning, verification, and standalone export CLI."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import loraforge.core as _core  # noqa: E402
from loraforge.core import (  # noqa: E402, F401
    LABELS,
    SYSTEM,
    all_packages,
    digest,
    encode_example,
    evaluate_rows,
    generate,
    load_merged,
    load_model,
    messages,
    metadata,
    parse_prediction,
    read_json,
    runtime,
    score,
    tensor_batch,
    unique_pairs,
    validation_loss,
    verify_merge_logits,
    versions,
    write_json,
)

ROOT = _core.ROOT


def _sync_root() -> None:
    _core.ROOT = ROOT


def config():
    _sync_root()
    return _core.config()


def load_data():
    _sync_root()
    return _core.load_data()


def data_hashes():
    _sync_root()
    return _core.data_hashes()


def lock_model(cfg):
    _sync_root()
    return _core.lock_model(cfg)


def doctor(args=None):
    _sync_root()
    return _core.doctor(args)


def validate(args=None):
    _sync_root()
    return _core.validate(args)


def prepare(args=None):
    _sync_root()
    return _core.prepare(args)


def train(args):
    _sync_root()
    return _core.train(args)


def evaluate(args):
    _sync_root()
    return _core.evaluate(args)


def infer(args):
    _sync_root()
    return _core.infer(args)


def merge(args):
    _sync_root()
    return _core.merge(args)


def report(args=None):
    _sync_root()
    return _core.report(args)


def main(argv=None):
    _sync_root()
    return _core.main(argv)


if __name__ == "__main__":
    main()
