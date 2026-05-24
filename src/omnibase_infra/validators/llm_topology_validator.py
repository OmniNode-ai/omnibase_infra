# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""LLM topology contract drift validator (OMN-11926).

Validates contracts/llm_endpoints.yaml for shape correctness. Fails (exit 1)
on any violation — this is enforcement, not detection.

Checks performed:
  - Required top-level key: endpoints
  - Every entry has slot_id, role, status
  - Valid role taxonomy (closed set)
  - Valid status values (running | disabled | on_demand | planned)
  - Status-specific nullability: running slots require host/port/endpoint_url/model_hf_id
  - Planned slots must have null host/port/endpoint_url/model_hf_id
  - Disabled slots must not own a url_env_var (runtime env var ownership)
  - endpoint_url must be consistent with host:port when all three are non-null
  - No duplicate slot_id values
  - No duplicate url_env_var values across running slots

Usage:
    python -m omnibase_infra.validators.llm_topology_validator contracts/llm_endpoints.yaml
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

VALID_ROLES = frozenset(
    {
        "coder_slow",
        "coder_fast",
        "embedding",
        "reasoning",
        "reasoning_fast",
        "reasoning_lightweight",
        "reasoning_transient",
        "vision",
        "stt",
        "tts",
        "reranker",
    }
)

VALID_STATUSES = frozenset({"running", "disabled", "on_demand", "planned"})

# Fields that must be non-null when status=running
RUNNING_REQUIRED_FIELDS = ("host", "port", "endpoint_url", "model_hf_id")

# Fields that must be null when status=planned
PLANNED_NULL_FIELDS = ("host", "port", "endpoint_url", "model_hf_id")


@dataclass(frozen=True, slots=True)
class Finding:
    """A single topology violation."""

    slot_id: str
    rule: str
    message: str

    def format(self) -> str:
        return f"[{self.rule}] {self.message}"


def _str_field(entry: dict[str, object], key: str) -> str:
    """Return entry[key] as a string, or '' if absent/None."""
    val = entry.get(key)
    return str(val) if val is not None else ""


def validate_topology_yaml(path: Path) -> list[Finding]:
    """Validate an llm_endpoints.yaml file. Returns findings (empty = clean)."""
    if not path.exists():
        return [
            Finding(
                slot_id="<file>",
                rule="file_exists",
                message=f"Contract file not found: {path}",
            )
        ]

    if yaml is None:
        return [
            Finding(
                slot_id="<file>",
                rule="yaml_available",
                message="PyYAML is not installed; cannot parse topology contract",
            )
        ]

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return [
            Finding(
                slot_id="<file>",
                rule="yaml_parse",
                message=f"Failed to parse YAML contract: {exc}",
            )
        ]

    if not isinstance(raw, dict):
        return [
            Finding(
                slot_id="<file>",
                rule="yaml_parse",
                message="Contract YAML root must be a mapping",
            )
        ]

    findings: list[Finding] = []

    if "endpoints" not in raw:
        findings.append(
            Finding(
                slot_id="<file>",
                rule="missing_endpoints_key",
                message="Contract is missing required top-level key: endpoints",
            )
        )
        return findings

    endpoints = raw["endpoints"]
    if not isinstance(endpoints, list):
        findings.append(
            Finding(
                slot_id="<file>",
                rule="endpoints_type",
                message="'endpoints' must be a list",
            )
        )
        return findings

    seen_slot_ids: dict[str, int] = {}
    seen_url_env_vars: dict[str, str] = {}

    for idx, entry in enumerate(endpoints):
        if not isinstance(entry, dict):
            findings.append(
                Finding(
                    slot_id=f"<index {idx}>",
                    rule="entry_type",
                    message=f"Endpoint at index {idx} is not a mapping",
                )
            )
            continue

        slot_id = _str_field(entry, "slot_id")
        label = slot_id or f"<index {idx}>"

        if not slot_id:
            findings.append(
                Finding(
                    slot_id=label,
                    rule="missing_slot_id",
                    message=f"Endpoint at index {idx} is missing required field: slot_id",
                )
            )

        findings.extend(_validate_entry(label, entry, seen_slot_ids, seen_url_env_vars))

    return findings


def _validate_entry(
    label: str,
    entry: dict[str, object],
    seen_slot_ids: dict[str, int],
    seen_url_env_vars: dict[str, str],
) -> list[Finding]:
    findings: list[Finding] = []
    slot_id = _str_field(entry, "slot_id")
    status = _str_field(entry, "status")
    role = _str_field(entry, "role")

    # Duplicate slot_id
    if slot_id:
        if slot_id in seen_slot_ids:
            findings.append(
                Finding(
                    slot_id=slot_id,
                    rule="duplicate_slot_id",
                    message=f"Duplicate slot_id '{slot_id}' — already seen at index {seen_slot_ids[slot_id]}",
                )
            )
        else:
            seen_slot_ids[slot_id] = len(seen_slot_ids)

    # role validation
    if not role:
        findings.append(
            Finding(
                slot_id=label,
                rule="missing_role",
                message=f"Endpoint '{label}' is missing required field: role",
            )
        )
    elif role not in VALID_ROLES:
        findings.append(
            Finding(
                slot_id=label,
                rule="invalid_role",
                message=(
                    f"Endpoint '{label}' has invalid role '{role}'. "
                    f"Valid roles: {', '.join(sorted(VALID_ROLES))}"
                ),
            )
        )

    # status validation
    if not status:
        findings.append(
            Finding(
                slot_id=label,
                rule="missing_status",
                message=f"Endpoint '{label}' is missing required field: status",
            )
        )
    elif status not in VALID_STATUSES:
        findings.append(
            Finding(
                slot_id=label,
                rule="invalid_status",
                message=(
                    f"Endpoint '{label}' has invalid status '{status}'. "
                    f"Valid statuses: {', '.join(sorted(VALID_STATUSES))}"
                ),
            )
        )

    if status not in VALID_STATUSES:
        return findings

    # Status-specific nullability checks
    if status == "running":
        for field in RUNNING_REQUIRED_FIELDS:
            if entry.get(field) is None:
                findings.append(
                    Finding(
                        slot_id=label,
                        rule="running_field_null",
                        message=(
                            f"Endpoint '{label}' has status=running but '{field}' is null. "
                            f"All of {RUNNING_REQUIRED_FIELDS} must be non-null when running."
                        ),
                    )
                )

    if status == "planned":
        for field in PLANNED_NULL_FIELDS:
            if entry.get(field) is not None:
                findings.append(
                    Finding(
                        slot_id=label,
                        rule="planned_field_non_null",
                        message=(
                            f"Endpoint '{label}' has status=planned but '{field}' is non-null. "
                            f"Planned slots must have null {', '.join(PLANNED_NULL_FIELDS)}."
                        ),
                    )
                )

    # Disabled slots must not own a url_env_var
    if status == "disabled" and entry.get("url_env_var") is not None:
        url_env_var_val = _str_field(entry, "url_env_var")
        findings.append(
            Finding(
                slot_id=label,
                rule="disabled_owns_url_env_var",
                message=(
                    f"Endpoint '{label}' is disabled but has url_env_var='{url_env_var_val}'. "
                    "Disabled slots must not own runtime env vars."
                ),
            )
        )

    # endpoint_url consistency check (host:port vs endpoint_url)
    host = entry.get("host")
    port = entry.get("port")
    endpoint_url = entry.get("endpoint_url")
    if host is not None and port is not None and endpoint_url is not None:
        expected_url = f"http://{host}:{port}"
        if endpoint_url != expected_url:
            findings.append(
                Finding(
                    slot_id=label,
                    rule="endpoint_url_mismatch",
                    message=(
                        f"Endpoint '{label}' endpoint_url '{endpoint_url}' does not match "
                        f"derived URL from host+port: '{expected_url}'"
                    ),
                )
            )

    # Duplicate url_env_var across running/on_demand slots
    url_env_var = _str_field(entry, "url_env_var")
    if url_env_var and status in ("running", "on_demand"):
        if url_env_var in seen_url_env_vars:
            findings.append(
                Finding(
                    slot_id=label,
                    rule="duplicate_url_env_var",
                    message=(
                        f"Duplicate url_env_var '{url_env_var}' on slot '{label}'. "
                        f"Already claimed by slot '{seen_url_env_vars[url_env_var]}'."
                    ),
                )
            )
        else:
            seen_url_env_vars[url_env_var] = label

    return findings


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate llm_endpoints.yaml topology contract (OMN-11926)."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("contracts/llm_endpoints.yaml"),
        help="Path to llm_endpoints.yaml (default: contracts/llm_endpoints.yaml)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    findings = validate_topology_yaml(args.path)
    if not findings:
        return 0

    sys.stderr.write(f"LLM topology validator found {len(findings)} violation(s):\n")
    for f in findings:
        sys.stderr.write(f"  {f.format()}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
