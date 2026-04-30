#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Recompute pricing manifest token rates from API-reported cost evidence."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import yaml

DEFAULT_MANIFEST = Path("src/omnibase_infra/configs/pricing_manifest.yaml")
DEFAULT_ENV_FILE = Path.home() / ".omnibase" / ".env"
LOW_CONFIDENCE = "LOW_CONFIDENCE"
MEASURED = "MEASURED"
MEASURED_SOURCE = "API_REPORTED_COST_ROLLING_7D"
MEASURED_USAGE_SOURCE = "measured"
LEGACY_API_USAGE_SOURCE = "API"
FALLBACK_SOURCE = "FALLBACK_PROVIDER_DOCUMENTATION"
LOCAL_SOURCE = "LOCAL_ZERO_API_COST_POLICY"
MANIFEST_HEADER = (
    "# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.\n# SPDX-License-Identifier: MIT\n"
)


@dataclass(frozen=True)
class PricingSample:
    """One API-reported cost sample from llm_call_metrics."""

    model_id: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@dataclass(frozen=True)
class ComputedPricing:
    """Computed pricing result for one model."""

    input_cost_per_1k: float
    output_cost_per_1k: float
    sample_count: int
    confidence: str


class DbConnection(Protocol):
    """Minimal asyncpg connection protocol used by this script."""

    async def fetch(self, query: str) -> list[object]:
        """Fetch query rows."""

    async def close(self) -> None:
        """Close the connection."""


def _as_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Expected YAML mapping")
    return value


def _sample_from_mapping(data: dict[str, object]) -> PricingSample | None:
    model_id = data.get("model_id")
    cost = data.get("estimated_cost_usd", data.get("cost_usd"))
    prompt_tokens = data.get("prompt_tokens", data.get("input_tokens", 0))
    completion_tokens = data.get(
        "completion_tokens",
        data.get("output_tokens", 0),
    )
    usage_source = data.get("usage_source", LEGACY_API_USAGE_SOURCE)

    if (
        not isinstance(model_id, str)
        or usage_source not in {MEASURED_USAGE_SOURCE, LEGACY_API_USAGE_SOURCE}
        or cost is None
    ):
        return None

    prompt = int(cast("int | float | str", prompt_tokens or 0))
    completion = int(cast("int | float | str", completion_tokens or 0))
    if prompt < 0 or completion < 0 or (prompt == 0 and completion == 0):
        return None

    return PricingSample(
        model_id=model_id,
        prompt_tokens=prompt,
        completion_tokens=completion,
        cost_usd=float(cast("int | float | str", cost)),
    )


def load_fixture_samples(path: Path) -> list[PricingSample]:
    """Load deterministic pricing samples from JSONL fixture rows."""
    samples: list[PricingSample] = []
    for line_no, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line:
            continue
        decoded = json.loads(line)
        sample = _sample_from_mapping(_as_mapping(decoded))
        if sample is not None:
            samples.append(sample)
        elif decoded:
            continue
        else:
            raise ValueError(f"{path}:{line_no}: fixture row must be a JSON object")
    return samples


async def fetch_db_samples(database_url: str) -> list[PricingSample]:
    """Fetch the last 7 days of API-reported llm_call_metrics rows."""
    import asyncpg

    query = """
        SELECT
            model_id,
            COALESCE(prompt_tokens, 0) AS prompt_tokens,
            COALESCE(completion_tokens, 0) AS completion_tokens,
            estimated_cost_usd
        FROM llm_call_metrics
        WHERE created_at >= NOW() - INTERVAL '7 days'
          AND usage_source::text IN ('measured', 'API')
          AND estimated_cost_usd IS NOT NULL
          AND COALESCE(prompt_tokens, 0) + COALESCE(completion_tokens, 0) > 0
        ORDER BY model_id, created_at
    """
    connection = cast("DbConnection", await asyncpg.connect(database_url))
    try:
        rows = await connection.fetch(query)
    finally:
        await connection.close()

    samples: list[PricingSample] = []
    for row in rows:
        sample = _sample_from_mapping(dict(cast("object", row)))
        if sample is not None:
            samples.append(sample)
    return samples


def compute_pricing_by_model(
    samples: list[PricingSample],
    *,
    min_samples: int,
) -> dict[str, ComputedPricing]:
    """Compute per-1k input/output rates per model via least squares."""
    grouped: dict[str, list[PricingSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.model_id].append(sample)

    results: dict[str, ComputedPricing] = {}
    for model_id, model_samples in grouped.items():
        sample_count = len(model_samples)
        input_rate, output_rate = _solve_token_rates(model_samples)
        results[model_id] = ComputedPricing(
            input_cost_per_1k=input_rate,
            output_cost_per_1k=output_rate,
            sample_count=sample_count,
            confidence=MEASURED if sample_count >= min_samples else LOW_CONFIDENCE,
        )
    return results


def _solve_token_rates(samples: list[PricingSample]) -> tuple[float, float]:
    prompt_units = [sample.prompt_tokens / 1000.0 for sample in samples]
    completion_units = [sample.completion_tokens / 1000.0 for sample in samples]
    costs = [sample.cost_usd for sample in samples]

    sum_prompt_sq = sum(value * value for value in prompt_units)
    sum_cross = sum(
        prompt * completion
        for prompt, completion in zip(prompt_units, completion_units, strict=True)
    )
    sum_completion_sq = sum(value * value for value in completion_units)
    sum_prompt_cost = sum(
        prompt * cost for prompt, cost in zip(prompt_units, costs, strict=True)
    )
    sum_completion_cost = sum(
        completion * cost
        for completion, cost in zip(completion_units, costs, strict=True)
    )

    determinant = sum_prompt_sq * sum_completion_sq - sum_cross * sum_cross
    if abs(determinant) > 1e-12:
        input_rate = (
            sum_prompt_cost * sum_completion_sq - sum_cross * sum_completion_cost
        ) / determinant
        output_rate = (
            sum_prompt_sq * sum_completion_cost - sum_cross * sum_prompt_cost
        ) / determinant
        return round(max(input_rate, 0.0), 10), round(max(output_rate, 0.0), 10)

    prompt_total = sum(prompt_units)
    completion_total = sum(completion_units)
    cost_total = sum(costs)
    if prompt_total > 0 and completion_total == 0:
        return round(max(cost_total / prompt_total, 0.0), 10), 0.0
    if completion_total > 0 and prompt_total == 0:
        return 0.0, round(max(cost_total / completion_total, 0.0), 10)
    if cost_total == 0:
        return 0.0, 0.0
    blended = cost_total / max(prompt_total + completion_total, 1e-12)
    return round(max(blended, 0.0), 10), round(max(blended, 0.0), 10)


def update_manifest(
    manifest_path: Path,
    output_path: Path,
    computed: dict[str, ComputedPricing],
    *,
    min_samples: int,
    generated_at: str,
) -> None:
    """Write updated pricing metadata to a manifest."""
    manifest = _as_mapping(yaml.safe_load(manifest_path.read_text(encoding="utf-8")))
    models = _as_mapping(manifest.get("models", {}))

    for model_id, raw_entry in models.items():
        entry = _as_mapping(raw_entry)
        result = computed.get(str(model_id))
        if result is not None:
            entry["input_cost_per_1k"] = result.input_cost_per_1k
            entry["output_cost_per_1k"] = result.output_cost_per_1k
            entry["confidence"] = result.confidence
            entry["source"] = (
                MEASURED_SOURCE
                if result.confidence == MEASURED
                else _fallback_source(entry)
            )
            entry["sample_count"] = result.sample_count
        else:
            entry.setdefault("confidence", LOW_CONFIDENCE)
            entry.setdefault("source", _fallback_source(entry))
            entry.setdefault("sample_count", 0)

        entry["evidence"] = {
            "generated_at": generated_at,
            "sample_window_days": 7,
            "sample_count": int(entry["sample_count"]),
            "min_samples": min_samples,
            "query": "llm_call_metrics usage_source=measured created_at>=now-7d",
            "authoritative": entry["confidence"] == MEASURED,
        }

    rendered = yaml.safe_dump(
        manifest,
        sort_keys=False,
        default_flow_style=False,
    )
    output_path.write_text(f"{MANIFEST_HEADER}{rendered}", encoding="utf-8")


def _fallback_source(entry: dict[str, object]) -> str:
    note = str(entry.get("note", "")).lower()
    if "local" in note:
        return LOCAL_SOURCE
    source = entry.get("source")
    if isinstance(source, str) and source != MEASURED_SOURCE:
        return source
    return FALLBACK_SOURCE


def load_dotenv(path: Path = DEFAULT_ENV_FILE) -> None:
    """Load KEY=VALUE pairs from a dotenv file without overriding the environment."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--fixture-jsonl", type=Path)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--generated-at")
    return parser.parse_args(argv)


async def async_main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_path = args.output or args.manifest
    generated_at = args.generated_at or dt.datetime.now(dt.UTC).isoformat()

    if args.fixture_jsonl is not None:
        samples = load_fixture_samples(args.fixture_jsonl)
    elif args.database_url:
        samples = await fetch_db_samples(str(args.database_url))
    else:
        raise SystemExit("--database-url or --fixture-jsonl is required")

    computed = compute_pricing_by_model(samples, min_samples=int(args.min_samples))
    update_manifest(
        args.manifest,
        output_path,
        computed,
        min_samples=int(args.min_samples),
        generated_at=str(generated_at),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
