# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The command line of the dev-lane branch installer (OMN-15920)."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omnibase_infra.handlers.handler_dev_lane_branch_install",
        description="Declare, check, apply and release dev-lane branch installs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("register", "release", "list", "check", "apply", "release-due"):
        p = sub.add_parser(name)
        p.add_argument("--state-dir", required=True, type=Path)
        if name in ("check", "apply", "release-due"):
            p.add_argument("--compose-project", required=True)
        if name in ("apply", "release-due"):
            p.add_argument(
                "--clone", action="append", default=[], help="REPO=PATH of a clone"
            )
            p.add_argument(
                "--running-ref",
                action="append",
                default=[],
                help="REPO=SHA the running image was built from",
            )
    reg = sub.choices["register"]
    reg.add_argument("--lane", required=True)
    reg.add_argument("--id", dest="install_id")
    reg.add_argument("--repo", required=True)
    reg.add_argument("--pr", type=int)
    reg.add_argument("--branch", required=True)
    reg.add_argument("--head", required=True)
    reg.add_argument("--ticket", required=True)
    reg.add_argument("--registered-by", required=True)
    reg.add_argument("--service", required=True, action="append", dest="services")
    reg.add_argument("--file", required=True, action="append", dest="files")
    reg.add_argument(
        "--entry-point",
        action="append",
        default=[],
        dest="entry_points",
        help="DIST:GROUP:NAME=MODULE[:ATTR] to write into the installed dist-info",
    )
    reg.add_argument("--expect-sha256", action="append", default=[], dest="expect")
    reg.add_argument("--expires-in-hours", required=True, type=float)
    reg.add_argument("--no-restart", action="store_true")
    reg.add_argument("--replace", action="store_true")
    reg.add_argument("--compose-project", help="apply at once to this project")
    rel = sub.choices["release"]
    rel.add_argument("--id", required=True, dest="install_id")
    rel.add_argument("--reason", required=True)
    return parser


__all__ = ["build_parser"]
