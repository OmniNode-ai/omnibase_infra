#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Compose config lint: detect dangling credential/address env vars for disabled services.

Retro B-9 / OMN-13037.

A "dangling dependency" is when a compose overlay simultaneously:
  1. Disables a service (profiles: !override ["*-disabled"])
  2. Injects an env var for that service with a non-empty concrete value that
     references the disabled service by hostname

Static analysis only: PyYAML parsing, no `docker compose config` execution.
This means env var interpolation (${VAR:-}) is parsed syntactically, not resolved.

Usage:
    uv run python scripts/lint_compose_dangling_deps.py docker/docker-compose.stability-test.yml
    uv run python scripts/lint_compose_dangling_deps.py docker/docker-compose.prod.yml
    uv run python scripts/lint_compose_dangling_deps.py docker/docker-compose.judge.yml

Exit codes:
    0 — no violations found
    1 — one or more violations found, or file not found / parse error
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# YAML loader with !override support
# ---------------------------------------------------------------------------


def _override_constructor(
    loader: yaml.SafeLoader,
    node: yaml.Node,
) -> Any:
    """Handle Docker Compose !override tag by returning the underlying value."""
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return loader.construct_scalar(node)  # type: ignore[arg-type]


def _make_loader() -> type[yaml.SafeLoader]:
    """Create a SafeLoader with !override support registered."""
    loader = yaml.SafeLoader
    loader.add_constructor("!override", _override_constructor)
    return loader


# ---------------------------------------------------------------------------
# Core detection logic
# ---------------------------------------------------------------------------

# Regex to match empty-default interpolation: ${VAR:-} or ${VAR:-""} → safe
_EMPTY_DEFAULT_RE = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*:-[\"']?\s*[\"']?\}$")

# Pattern: value contains service_name as a hostname component
# Matches: http://infisical:8080, https://infisical/path, infisical:8080,
#          bare "infisical" (exact match or as hostname segment)
_HOSTNAME_PATTERNS = (
    # Scheme://hostname or scheme://hostname:port
    r"https?://{service}(:[0-9]+)?(/|$)",
    # hostname:port
    r"(^|[^a-zA-Z0-9_-]){service}:[0-9]",
    # bare hostname match (value is exactly the service name)
    r"^{service}$",
)


def _value_references_hostname(value: str, service_name: str) -> bool:
    """Return True if value references service_name as a Docker hostname."""
    for pat in _HOSTNAME_PATTERNS:
        compiled = re.compile(pat.format(service=re.escape(service_name)))
        if compiled.search(value):
            return True
    return False


def _is_empty_or_interpolated_empty(value: Any) -> bool:
    """Return True if the env var value is empty or resolves to empty by default.

    Safe values:
    - None / null
    - Empty string ""
    - Interpolated with empty default: ${VAR:-} or ${VAR:-""}
    """
    if value is None:
        return True
    s = str(value).strip()
    if not s:
        return True
    return bool(_EMPTY_DEFAULT_RE.match(s))


def _find_disabled_services(services: dict[str, Any]) -> set[str]:
    """Return the set of service names whose profiles are overridden to *-disabled.

    A service is considered disabled if its 'profiles' list contains only
    entries matching the '*-disabled' naming convention.
    """
    disabled: set[str] = set()
    for name, definition in services.items():
        if not isinstance(definition, dict):
            continue
        profiles = definition.get("profiles")
        if not isinstance(profiles, list) or not profiles:
            continue
        # All profile entries must match '*-disabled' pattern
        if all(isinstance(p, str) and p.endswith("-disabled") for p in profiles):
            disabled.add(name)
    return disabled


def _collect_violations(
    overlay_path: Path,
    parsed: dict[str, Any],
) -> list[str]:
    """Collect all dangling-dependency violations from a parsed overlay.

    Returns a list of human-readable violation strings.
    """
    services: dict[str, Any] = parsed.get("services", {}) or {}
    disabled = _find_disabled_services(services)

    if not disabled:
        return []

    violations: list[str] = []

    for svc_name, definition in services.items():
        if svc_name in disabled:
            continue
        if not isinstance(definition, dict):
            continue
        env_block = definition.get("environment")
        if not env_block:
            continue

        # environment can be a dict (key: value) or a list (KEY=value)
        if isinstance(env_block, dict):
            env_items: list[tuple[str, Any]] = list(env_block.items())
        elif isinstance(env_block, list):
            env_items = []
            for entry in env_block:
                if isinstance(entry, str) and "=" in entry:
                    k, _, v = entry.partition("=")
                    env_items.append((k, v))
                # Entries without '=' have no value (picked from env) — skip
        else:
            continue

        for var_name, var_value in env_items:
            if _is_empty_or_interpolated_empty(var_value):
                continue
            # Check if this non-empty value references any disabled service
            value_str = str(var_value)
            for disabled_svc in disabled:
                if _value_references_hostname(value_str, disabled_svc):
                    violations.append(
                        f"  container={svc_name!r}  env={var_name!r}"
                        f"  value={value_str!r}"
                        f"  disabled_service={disabled_svc!r}"
                    )

    return violations


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def lint_file(path: Path) -> int:
    """Lint a single compose overlay YAML file.

    Returns 0 if clean, 1 if violations found or file cannot be processed.
    """
    if not path.exists():
        print(f"ERROR: file not found: {path}", flush=True)
        return 1

    raw = path.read_text(encoding="utf-8")
    try:
        parsed: dict[str, Any] = yaml.load(raw, Loader=_make_loader())  # noqa: S506
    except yaml.YAMLError as exc:
        print(f"ERROR: failed to parse {path}: {exc}", flush=True)
        return 1

    if not isinstance(parsed, dict):
        # Empty or non-mapping YAML — nothing to check
        return 0

    violations = _collect_violations(path, parsed)

    if violations:
        print(
            f"FAIL — dangling service credential/address env vars detected in {path}:",
            flush=True,
        )
        for v in violations:
            print(v, flush=True)
        print(
            "\nFix: remove the env var or set it to an empty value "
            "(e.g. ${VAR:-}) when the referenced service is disabled "
            "in the same overlay.",
            flush=True,
        )
        return 1

    print(
        f"OK — {path}: no dangling service credential/address env vars found.",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print(
            "Usage: lint_compose_dangling_deps.py <overlay.yml> [<overlay.yml> ...]",
            flush=True,
        )
        return 1

    exit_code = 0
    for arg in args:
        result = lint_file(Path(arg))
        if result != 0:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
