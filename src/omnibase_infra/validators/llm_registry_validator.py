# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""LLM model registry drift validator (OMN-11926).

Validates the model_registry_v1.yaml (or equivalent) for schema completeness
and internal consistency. Fails (exit 1) on any violation.

Checks performed:
  - Required top-level keys: schema_version, model_registry_version,
    pricing_manifest_version, observed_at, models
  - Every model entry has required pricing manifest fields
  - model_id field matches the registry key
  - valid cost_basis values
  - context_window is a positive integer
  - observed_at is present
  - Cloud providers require requires_api_key_env
  - No duplicate endpoint_env across local/same-provider models

Usage:
    python -m omnibase_infra.validators.llm_registry_validator \\
        path/to/model_registry_v1.yaml
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

VALID_COST_BASES = frozenset(
    {
        "zero_marginal_api_cost",
        "cloud_api_cost",
    }
)

# Providers that must supply a requires_api_key_env
CLOUD_PROVIDERS = frozenset({"anthropic", "openai", "openrouter", "z.ai"})

REQUIRED_TOP_LEVEL = (
    "schema_version",
    "model_registry_version",
    "pricing_manifest_version",
    "observed_at",
    "models",
)

REQUIRED_MODEL_FIELDS = (
    "model_id",
    "provider",
    "endpoint_env",
    "cost_basis",
    "pricing_per_1m_input",
    "pricing_per_1m_output",
    "context_window",
    "observed_at",
)


@dataclass(frozen=True, slots=True)  # internal-dataclass-ok: validator-internal finding
class Finding:
    """A single registry violation."""

    model_key: str
    rule: str
    message: str

    def format(self) -> str:
        return f"[{self.rule}] {self.message}"


def _str_field(entry: dict[str, object], key: str) -> str:
    """Return entry[key] as a string, or '' if absent/None."""
    val = entry.get(key)
    return str(val) if val is not None else ""


def validate_registry_yaml(path: Path) -> list[Finding]:
    """Validate a model registry YAML file. Returns findings (empty = clean)."""
    if not path.exists():
        return [
            Finding(
                model_key="<file>",
                rule="file_exists",
                message=f"Registry file not found: {path}",
            )
        ]

    if yaml is None:
        return [
            Finding(
                model_key="<file>",
                rule="yaml_available",
                message="PyYAML is not installed; cannot parse model registry",
            )
        ]

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return [
            Finding(
                model_key="<file>",
                rule="yaml_parse",
                message=f"Failed to parse YAML registry: {exc}",
            )
        ]

    if not isinstance(raw, dict):
        return [
            Finding(
                model_key="<file>",
                rule="yaml_parse",
                message="Registry YAML root must be a mapping",
            )
        ]

    findings: list[Finding] = []

    for key in REQUIRED_TOP_LEVEL:
        if key not in raw:
            findings.append(
                Finding(
                    model_key="<file>",
                    rule=f"missing_{key}",
                    message=f"Registry is missing required top-level key: {key}",
                )
            )

    if "models" not in raw:
        return findings

    models = raw["models"]
    if not isinstance(models, dict):
        findings.append(
            Finding(
                model_key="<file>",
                rule="models_type",
                message="'models' must be a mapping of model_key → model entry",
            )
        )
        return findings

    seen_endpoint_envs: dict[str, str] = {}

    for model_key, entry in models.items():
        if not isinstance(entry, dict):
            findings.append(
                Finding(
                    model_key=str(model_key),
                    rule="entry_type",
                    message=f"Model entry '{model_key}' is not a mapping",
                )
            )
            continue
        findings.extend(
            _validate_model_entry(str(model_key), entry, seen_endpoint_envs)
        )

    return findings


def _validate_model_entry(
    model_key: str,
    entry: dict[str, object],
    seen_endpoint_envs: dict[str, str],
) -> list[Finding]:
    findings: list[Finding] = []

    # Required fields
    for field in REQUIRED_MODEL_FIELDS:
        if field not in entry:
            findings.append(
                Finding(
                    model_key=model_key,
                    rule=f"missing_{field}",
                    message=f"Model '{model_key}' is missing required field: {field}",
                )
            )

    # model_id must match the registry key
    model_id = _str_field(entry, "model_id")
    if model_id and model_id != model_key:
        findings.append(
            Finding(
                model_key=model_key,
                rule="model_id_key_mismatch",
                message=(
                    f"Model '{model_key}' has model_id='{model_id}' which does not match "
                    "the registry key. They must be identical."
                ),
            )
        )

    # cost_basis validation
    cost_basis = _str_field(entry, "cost_basis")
    if cost_basis and cost_basis not in VALID_COST_BASES:
        findings.append(
            Finding(
                model_key=model_key,
                rule="invalid_cost_basis",
                message=(
                    f"Model '{model_key}' has invalid cost_basis='{cost_basis}'. "
                    f"Valid values: {', '.join(sorted(VALID_COST_BASES))}"
                ),
            )
        )

    # context_window must be a positive integer
    context_window = entry.get("context_window")
    if context_window is not None:
        if not isinstance(context_window, int) or context_window <= 0:
            findings.append(
                Finding(
                    model_key=model_key,
                    rule="invalid_context_window",
                    message=(
                        f"Model '{model_key}' has invalid context_window='{context_window}'. "
                        "Must be a positive integer."
                    ),
                )
            )

    # Cloud providers require requires_api_key_env
    provider = _str_field(entry, "provider")
    if provider in CLOUD_PROVIDERS and "requires_api_key_env" not in entry:
        findings.append(
            Finding(
                model_key=model_key,
                rule="cloud_missing_api_key_env",
                message=(
                    f"Model '{model_key}' uses cloud provider='{provider}' but is missing "
                    "required field: requires_api_key_env"
                ),
            )
        )

    # No duplicate endpoint_env across same-provider local models
    endpoint_env = _str_field(entry, "endpoint_env")
    if endpoint_env:
        dupe_key = f"{provider}:{endpoint_env}"
        if dupe_key in seen_endpoint_envs:
            findings.append(
                Finding(
                    model_key=model_key,
                    rule="duplicate_endpoint_env",
                    message=(
                        f"Duplicate endpoint_env '{endpoint_env}' for provider '{provider}' "
                        f"on model '{model_key}'. Already claimed by '{seen_endpoint_envs[dupe_key]}'."
                    ),
                )
            )
        else:
            seen_endpoint_envs[dupe_key] = model_key

    return findings


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate model registry YAML for completeness and consistency (OMN-11926)."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("src/omnimarket/data/model_registry/model_registry_v1.yaml"),
        help="Path to model_registry_v1.yaml",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    findings = validate_registry_yaml(args.path)
    if not findings:
        return 0

    sys.stderr.write(f"LLM registry validator found {len(findings)} violation(s):\n")
    for f in findings:
        sys.stderr.write(f"  {f.format()}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
