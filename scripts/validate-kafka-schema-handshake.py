#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Cross-repo Kafka schema handshake validator (OMN-3411).

Validates that every registered producer→consumer boundary pair can
successfully round-trip across the Kafka wire format.  Uses the same
``BoundaryPair`` registry as the pytest suite in
``tests/integration/event_bus/test_kafka_boundary_compat.py``.

Usage
-----
Full scan (all pairs):
    uv run python scripts/validate-kafka-schema-handshake.py

Changed-only scan (fast path for CI):
    uv run python scripts/validate-kafka-schema-handshake.py --changed-only

JSON output (machine-readable, also prints human output):
    uv run python scripts/validate-kafka-schema-handshake.py --format json

Exit codes
----------
0  All validated pairs passed (xfail pairs are skipped, not failed).
1  One or more pairs failed validation.
2  Sibling packages not installed (skipped with warning).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from typing import Any

# ---------------------------------------------------------------------------
# Sibling package availability check
# ---------------------------------------------------------------------------

try:
    from omniintelligence.nodes.node_intent_classifier_compute.models.enum_intent_class import (
        EnumIntentClass,
    )
    from omniintelligence.nodes.node_intent_classifier_compute.models.model_intent_classified_event import (
        ModelIntentClassifiedEvent as _OmniIntelligenceIntentClassifiedEvent,
    )
    from omnimemory.models.events.model_intent_classified_event import (
        ModelIntentClassifiedEvent as _OmniMemoryIntentClassifiedEvent,
    )
except ImportError as exc:
    print(
        f"WARNING: sibling packages not installed ({exc}).\n"
        "Install with: uv pip install -e ../omniintelligence -e ../omnimemory\n"
        "Skipping schema handshake validation.",
        file=sys.stderr,
    )
    sys.exit(2)

# ---------------------------------------------------------------------------
# Import the BoundaryPair registry from the test module.
# This is the canonical source of truth — never re-implement it here.
# ---------------------------------------------------------------------------

# The test module does a pytest.importorskip at collection time.  We can't
# import it via the normal import path in a standalone script because pytest
# is not running.  We load it directly with importlib so the pytest calls are
# visible but we skip the xfail markers at test-parametrize time.
import importlib.util
from pathlib import Path

_TEST_MODULE_PATH = str(
    Path(__file__).resolve().parent.parent
    / "tests"
    / "integration"
    / "event_bus"
    / "test_kafka_boundary_compat.py"
)


def _load_boundary_pairs() -> list[Any]:
    """Load BOUNDARY_PAIRS from the test module without running pytest."""
    spec = importlib.util.spec_from_file_location(
        "test_kafka_boundary_compat", _TEST_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load test module from {_TEST_MODULE_PATH}")

    # Patch pytest.importorskip to be a no-op for import-time side effects.
    from unittest.mock import patch

    import pytest

    # importorskip returns the imported module when it succeeds.
    # We patch it to import normally so the module-level imports run fine.
    original_importorskip = pytest.importorskip

    def _passthrough_importorskip(modname: str, **kwargs: Any) -> Any:
        return original_importorskip(modname, **kwargs)

    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec so @dataclass can resolve __module__.
    sys.modules["test_kafka_boundary_compat"] = module
    try:
        with patch.object(pytest, "importorskip", _passthrough_importorskip):
            spec.loader.exec_module(module)  # type: ignore[union-attr]
    finally:
        # Clean up to avoid polluting sys.modules for any subsequent imports.
        sys.modules.pop("test_kafka_boundary_compat", None)

    return module.BOUNDARY_PAIRS  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# --changed-only: determine which Python module paths changed vs origin/main
# ---------------------------------------------------------------------------


def _changed_module_prefixes() -> set[str]:
    """Return a set of dotted module prefixes for files changed vs origin/main.

    Uses ``git diff --name-only origin/main`` to detect touched Python files,
    then converts them to dotted module paths (``src/foo/bar.py`` → ``foo.bar``).
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "origin/main"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        # Fall back to HEAD~1 diff if origin/main is not available (shallow clones).
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD~1"],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            return set()

    prefixes: set[str] = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.endswith(".py"):
            continue
        # Strip leading src/ or src/<pkg>/ prefix and convert to dotted path.
        parts = line.replace("\\", "/").split("/")
        if parts and parts[0] == "src":
            parts = parts[1:]
        dotted = ".".join(p.rstrip(".py") if p.endswith(".py") else p for p in parts)
        # Normalise: strip trailing .py segment if present.
        if dotted.endswith(".py"):
            dotted = dotted[:-3]
        prefixes.add(dotted)

    return prefixes


def _pair_touches_changed(pair: Any, changed: set[str]) -> bool:
    """Return True if the pair involves a module that appears in *changed*."""
    producer_mod = pair.producer_cls.__module__
    consumer_mod = pair.consumer_cls.__module__
    for prefix in changed:
        if producer_mod.startswith(prefix) or consumer_mod.startswith(prefix):
            return True
    return False


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


class PairResult:
    def __init__(
        self,
        topic: str,
        producer: str,
        consumer: str,
        passed: bool,
        skipped: bool,
        xfail: bool,
        error: str | None,
    ) -> None:
        self.topic = topic
        self.producer = producer
        self.consumer = consumer
        self.passed = passed
        self.skipped = skipped
        self.xfail = xfail
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "producer": self.producer,
            "consumer": self.consumer,
            "passed": self.passed,
            "skipped": self.skipped,
            "xfail": self.xfail,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------


def _validate_pair(pair: Any) -> PairResult:
    """Run one boundary pair through the three-step round-trip check."""
    topic = pair.topic
    producer_name = f"{pair.producer_cls.__module__}.{pair.producer_cls.__qualname__}"
    consumer_name = f"{pair.consumer_cls.__module__}.{pair.consumer_cls.__qualname__}"
    xfail = bool(pair.xfail_reason)

    try:
        produced_model = pair.producer_cls.model_validate(pair.sample)
        produced_json = produced_model.model_dump(mode="json")
        pair.consumer_cls.model_validate(produced_json)
        return PairResult(
            topic=topic,
            producer=producer_name,
            consumer=consumer_name,
            passed=True,
            skipped=False,
            xfail=xfail,
            error=None,
        )
    except Exception:
        err = traceback.format_exc().strip()
        return PairResult(
            topic=topic,
            producer=producer_name,
            consumer=consumer_name,
            passed=False,
            skipped=False,
            xfail=xfail,
            error=err,
        )


# ---------------------------------------------------------------------------
# Human-readable output formatting
# ---------------------------------------------------------------------------


def _print_human(results: list[PairResult], *, changed_only: bool) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.passed and not r.xfail)
    xfail_pass = sum(1 for r in results if r.passed and r.xfail)
    xfail_fail = sum(1 for r in results if not r.passed and r.xfail)
    failed = sum(1 for r in results if not r.passed and not r.xfail)
    skipped = sum(1 for r in results if r.skipped)

    mode_label = " [--changed-only]" if changed_only else ""
    print(f"\nKafka Schema Handshake{mode_label}")
    print("=" * 60)

    for r in results:
        if r.skipped:
            status = "SKIP"
        elif r.xfail and not r.passed:
            status = "XFAIL"
        elif r.xfail and r.passed:
            status = "XPASS"
        elif r.passed:
            status = "PASS"
        else:
            status = "FAIL"

        print(f"{status:6}  {r.topic}")

        if not r.passed and not r.skipped:
            print(f"         Producer: {r.producer}")
            print(f"         Consumer: {r.consumer}")
            if r.xfail_reason if hasattr(r, "xfail_reason") else False:
                pass
            if r.error:
                # Show first meaningful lines of the traceback.
                lines = [ln for ln in r.error.splitlines() if ln.strip()]
                for ln in lines[-6:]:
                    print(f"         {ln}")
            print()

    print("-" * 60)
    print(
        f"Total: {total}  |  "
        f"Pass: {passed}  |  "
        f"Fail: {failed}  |  "
        f"XFail: {xfail_fail}  |  "
        f"XPass: {xfail_pass}  |  "
        f"Skip: {skipped}"
    )

    if failed:
        print(
            f"\n{'ERROR':>6}: {failed} pair(s) FAILED — Kafka boundary contract broken"
        )
    else:
        print("\nAll validated pairs passed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Kafka schema handshake boundary pairs (OMN-3411)."
    )
    parser.add_argument(
        "--changed-only",
        action="store_true",
        help=(
            "Only validate pairs whose models were touched "
            "(compares against git diff origin/main)."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["human", "json"],
        default="human",
        help="Output format. 'json' also always prints human output.",
    )
    args = parser.parse_args()

    pairs = _load_boundary_pairs()

    if args.changed_only:
        changed = _changed_module_prefixes()
        pairs_to_validate = [p for p in pairs if _pair_touches_changed(p, changed)]
        skipped_pairs = [p for p in pairs if not _pair_touches_changed(p, changed)]
    else:
        pairs_to_validate = list(pairs)
        skipped_pairs = []

    results: list[PairResult] = []

    # Add skipped results first (for ordering in output).
    for pair in skipped_pairs:
        producer_name = (
            f"{pair.producer_cls.__module__}.{pair.producer_cls.__qualname__}"
        )
        consumer_name = (
            f"{pair.consumer_cls.__module__}.{pair.consumer_cls.__qualname__}"
        )
        results.append(
            PairResult(
                topic=pair.topic,
                producer=producer_name,
                consumer=consumer_name,
                passed=True,
                skipped=True,
                xfail=bool(pair.xfail_reason),
                error=None,
            )
        )

    for pair in pairs_to_validate:
        results.append(_validate_pair(pair))

    # Always print human output.
    _print_human(results, changed_only=args.changed_only)

    if args.format == "json":
        payload = {
            "changed_only": args.changed_only,
            "total": len(results),
            "pairs": [r.to_dict() for r in results],
        }
        print("\n--- JSON ---")
        print(json.dumps(payload, indent=2))

    failed = [r for r in results if not r.passed and not r.skipped and not r.xfail]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
