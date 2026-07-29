# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Render or parity-check secret-free application database topology projections."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from omnibase_infra.topology import (
    SUPPORTED_ENVIRONMENTS,
    render_database_projection,
    validate_database_projection,
    validate_docker_catalog_parity,
)
from omnibase_infra.topology.application_database import write_database_projection


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        choices=sorted(SUPPORTED_ENVIRONMENTS),
        required=True,
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output", type=Path)
    output_group.add_argument("--check", type=Path)
    parser.add_argument(
        "--validate-docker",
        action="store_true",
        help="Also parity-check Docker catalog consumers (local only).",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    environment = str(args.environment)
    output = args.output
    check = args.check

    if output is not None:
        write_database_projection(environment, output)
    elif check is not None:
        validate_database_projection(environment, check)
    else:
        print(
            yaml.safe_dump(
                render_database_projection(environment),
                sort_keys=True,
            ),
            end="",
        )

    if args.validate_docker:
        if environment != "local":
            raise ValueError("Docker catalog parity is defined only for local topology")
        validate_docker_catalog_parity()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
