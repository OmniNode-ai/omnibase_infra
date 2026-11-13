# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read a host settings store (``.env`` file) into a name -> value mapping (OMN-24017).

Moved here unchanged from ``scripts/check_llm_endpoint_env_contract.py`` (the env
file reader, then OMN-24025 the contract's endpoint reader, the derived env var set
and the validation) so that script, whose code lines the file-shape ratchet only
lets shrink, carries the overlay resolution and the skip option without growing.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml

__all__ = ["contract_env_vars", "load_endpoints", "load_env_file", "validate"]


def load_env_file(path: Path) -> dict[str, str]:
    """Parse ``NAME=value`` lines; blank lines, comments and lines without ``=`` are skipped."""
    env: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("'\"")
    return env


def contract_env_vars(endpoints: list[dict[str, object]]) -> tuple[str, ...]:
    """Return the env vars the contract itself assigns to endpoint slots.

    OMN-16442: this set is DERIVED, never hardcoded. A slot released from its
    variable (``url_env_var: null``, as ``coder-fast-4090`` was when its GPU was
    physically removed) drops out of the checked set automatically, so a release
    can never leave an orphaned assertion behind that fires against a host which
    still — correctly — seeds the variable for another consumer.

    Order follows the contract file so the reported errors read in slot order.
    """
    env_vars: dict[str, None] = {}
    for endpoint in endpoints:
        url_env_var = endpoint.get("url_env_var")
        if url_env_var:
            env_vars[str(url_env_var)] = None
    return tuple(env_vars)


def load_endpoints(path: Path) -> list[dict[str, object]]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("endpoints"), list):
        msg = f"{path} must contain a top-level endpoints list"
        raise ValueError(msg)
    return cast("list[dict[str, object]]", data["endpoints"])


def validate(
    env: dict[str, str],
    endpoints: list[dict[str, object]],
    env_vars: tuple[str, ...],
) -> list[str]:
    """Return validation errors for configured endpoint env vars."""
    by_env = {
        str(endpoint["url_env_var"]): endpoint
        for endpoint in endpoints
        if endpoint.get("url_env_var")
    }
    by_url = {
        str(endpoint["endpoint_url"]).rstrip("/"): endpoint
        for endpoint in endpoints
        if endpoint.get("endpoint_url")
    }

    errors: list[str] = []
    for env_var in env_vars:
        value = env.get(env_var, "").strip().rstrip("/")
        if not value:
            continue

        expected = by_env.get(env_var)
        if expected is None:
            errors.append(f"{env_var} is not assigned to a canonical endpoint slot")
            continue

        if expected.get("status") != "running":
            errors.append(
                f"{env_var} is assigned to non-running slot {expected.get('slot_id')!r}"
            )
            continue

        expected_url = str(expected.get("endpoint_url", "")).rstrip("/")
        if value != expected_url:
            actual = by_url.get(value)
            actual_status = actual.get("status") if actual else "unknown"
            actual_slot = actual.get("slot_id") if actual else "not in contract"
            errors.append(
                f"{env_var}={value} does not match canonical running slot "
                f"{expected['slot_id']!r} ({expected_url}); actual slot="
                f"{actual_slot!r} status={actual_status!r}"
            )
            continue

        actual = by_url.get(value)
        if actual and actual.get("status") != "running":
            errors.append(
                f"{env_var} points at non-running slot {actual.get('slot_id')!r}"
            )

    return errors
