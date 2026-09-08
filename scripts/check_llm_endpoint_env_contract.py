#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Validate LLM endpoint env values against the canonical endpoint contract."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, cast

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONTRACT = _REPO_ROOT / "contracts" / "llm_endpoints.yaml"


def _load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("'\"")
    return env


def contract_env_vars(endpoints: list[dict[str, Any]]) -> tuple[str, ...]:
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


def _load_endpoints(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("endpoints"), list):
        msg = f"{path} must contain a top-level endpoints list"
        raise ValueError(msg)
    return cast("list[dict[str, Any]]", data["endpoints"])


def validate(
    env: dict[str, str],
    endpoints: list[dict[str, Any]],
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=_DEFAULT_CONTRACT,
        help="Path to contracts/llm_endpoints.yaml",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Optional .env file to validate instead of the process environment",
    )
    parser.add_argument(
        "--env-var",
        action="append",
        dest="env_vars",
        help=(
            "Env var to validate; repeatable. Defaults to every url_env_var the "
            "contract assigns to a slot."
        ),
    )
    args = parser.parse_args(argv)

    env = _load_env_file(args.env_file) if args.env_file else dict(os.environ)
    endpoints = _load_endpoints(args.contract)
    env_vars = tuple(args.env_vars) if args.env_vars else contract_env_vars(endpoints)
    errors = validate(env, endpoints, env_vars)

    if errors:
        print("LLM endpoint env contract check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("LLM endpoint env contract check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
