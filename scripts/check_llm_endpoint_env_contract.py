#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Validate LLM endpoint env values against the canonical endpoint contract."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from omnibase_infra.models.contracts.model_llm_endpoint_env_file import (
    contract_env_vars,
    load_endpoints,
    load_env_file,
    validate,
)
from omnibase_infra.models.contracts.model_llm_endpoint_overlay import (
    resolve_overlay_endpoints,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONTRACT = _REPO_ROOT / "contracts" / "llm_endpoints.yaml"


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
    parser.add_argument(
        "--skip-env-var",
        action="append",
        dest="skip_env_vars",
        default=[],
        help=(
            "Env var to leave unchecked; repeatable. For a name the caller renders "
            "itself and checks on the environment it renders."
        ),
    )
    args = parser.parse_args(argv)

    env = load_env_file(args.env_file) if args.env_file else dict(os.environ)
    endpoints = load_endpoints(args.contract)
    try:
        endpoints = resolve_overlay_endpoints(endpoints, os.environ, env)
    except ValueError as exc:
        print(f"LLM endpoint env contract check failed:\n  - {exc}", file=sys.stderr)
        return 1
    env_vars = tuple(args.env_vars) if args.env_vars else contract_env_vars(endpoints)
    env_vars = tuple(name for name in env_vars if name not in args.skip_env_vars)
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
