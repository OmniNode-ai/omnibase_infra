# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for measured pricing recomputation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "recompute_measured_pricing.py"


def _load_recompute_module() -> object:
    spec = importlib.util.spec_from_file_location("recompute_measured_pricing", _SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["recompute_measured_pricing"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_manifest(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0.0",
                "models": {
                    "measured-model": {
                        "input_cost_per_1k": 0.0,
                        "output_cost_per_1k": 0.0,
                        "effective_date": "2026-04-29",
                        "confidence": "LOW_CONFIDENCE",
                        "source": "FALLBACK_PROVIDER_DOCUMENTATION",
                        "sample_count": 0,
                        "evidence": {"authoritative": False},
                    },
                    "thin-model": {
                        "input_cost_per_1k": 0.0,
                        "output_cost_per_1k": 0.0,
                        "effective_date": "2026-04-29",
                        "confidence": "LOW_CONFIDENCE",
                        "source": "FALLBACK_PROVIDER_DOCUMENTATION",
                        "sample_count": 0,
                        "evidence": {"authoritative": False},
                    },
                    "fallback-model": {
                        "input_cost_per_1k": 0.001,
                        "output_cost_per_1k": 0.002,
                        "effective_date": "2026-04-29",
                        "confidence": "LOW_CONFIDENCE",
                        "source": "FALLBACK_PROVIDER_DOCUMENTATION",
                        "sample_count": 0,
                        "evidence": {"authoritative": False},
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_fixture(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for _ in range(10):
        rows.append(
            {
                "model_id": "measured-model",
                "prompt_tokens": 1000,
                "completion_tokens": 0,
                "estimated_cost_usd": 0.003,
                "usage_source": "API",
            }
        )
        rows.append(
            {
                "model_id": "measured-model",
                "prompt_tokens": 0,
                "completion_tokens": 1000,
                "estimated_cost_usd": 0.015,
                "usage_source": "API",
            }
        )
    for _ in range(19):
        rows.append(
            {
                "model_id": "thin-model",
                "prompt_tokens": 1000,
                "completion_tokens": 0,
                "estimated_cost_usd": 0.004,
                "usage_source": "API",
            }
        )
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


@pytest.mark.integration
def test_recompute_measured_pricing_updates_manifest_from_fixture(
    tmp_path: Path,
) -> None:
    module = _load_recompute_module()
    manifest = tmp_path / "pricing_manifest.yaml"
    output = tmp_path / "pricing_manifest.updated.yaml"
    fixture = tmp_path / "llm_call_metrics.jsonl"
    _write_manifest(manifest)
    _write_fixture(fixture)

    assert (
        module.main(
            [
                "--manifest",
                str(manifest),
                "--output",
                str(output),
                "--fixture-jsonl",
                str(fixture),
                "--generated-at",
                "2026-04-29T00:00:00+00:00",
            ]
        )
        == 0
    )

    data = yaml.safe_load(output.read_text(encoding="utf-8"))
    measured = data["models"]["measured-model"]
    thin = data["models"]["thin-model"]
    fallback = data["models"]["fallback-model"]

    assert measured["input_cost_per_1k"] == 0.003
    assert measured["output_cost_per_1k"] == 0.015
    assert measured["sample_count"] == 20
    assert measured["confidence"] == "MEASURED"
    assert measured["source"] == "API_REPORTED_COST_ROLLING_7D"
    assert measured["evidence"]["authoritative"] is True

    assert thin["input_cost_per_1k"] == 0.004
    assert thin["sample_count"] == 19
    assert thin["confidence"] == "LOW_CONFIDENCE"
    assert thin["source"] == "FALLBACK_PROVIDER_DOCUMENTATION"
    assert thin["evidence"]["authoritative"] is False

    assert fallback["input_cost_per_1k"] == 0.001
    assert fallback["output_cost_per_1k"] == 0.002
    assert fallback["source"] == "FALLBACK_PROVIDER_DOCUMENTATION"
    assert fallback["evidence"]["authoritative"] is False


@pytest.mark.integration
def test_load_dotenv_sets_database_url_without_overriding_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_recompute_module()
    env_file = tmp_path / ".env"
    env_file.write_text(
        'DATABASE_URL="postgresql://from-file"\nOTHER_VALUE=from-file\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("OTHER_VALUE", "from-env")

    module.load_dotenv(env_file)

    assert module.os.environ["DATABASE_URL"] == "postgresql://from-file"
    assert module.os.environ["OTHER_VALUE"] == "from-env"
