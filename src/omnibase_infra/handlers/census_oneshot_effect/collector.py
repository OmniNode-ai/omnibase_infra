# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Filesystem, process and clock boundaries for the census EFFECT node."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from subprocess import TimeoutExpired
from tempfile import TemporaryDirectory
from time import monotonic, sleep

import yaml

__all__ = [
    "TimeoutExpired",
    "has_plan",
    "monotonic",
    "read_policy",
    "pending_seconds",
    "sample",
    "sleep",
    "workspace",
]


def read_policy(path: Path) -> dict[str, object]:
    """Decode the explicitly supplied policy document at the I/O boundary."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("invalid census policy document")
    return data


def pending_seconds(path: Path) -> list[float]:
    """Decode the collector's remaining completion times."""
    plan = json.loads(path.read_text(encoding="utf-8"))
    return [row["remaining_seconds"] for row in plan["pending_oneshots"]]


def has_plan(path: Path) -> bool:
    """Observe whether the collector supplied its completion plan."""
    return path.is_file()


@contextmanager
def workspace() -> Iterator[Path]:
    """Keep a sample's temporary plan scoped to this invocation."""
    with TemporaryDirectory(prefix="census-oneshot-") as scratch:
        yield Path(scratch) / "plan.json"


def sample(
    script_directory: str,
    arguments: tuple[str, ...],
    plan_path: Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Invoke one collector sample with its own plan and unchanged environment."""
    plan_path.unlink(missing_ok=True)
    environment = {
        **os.environ,
        "ONEX_CENSUS_SINGLE_SAMPLE": "1",
        "ONEX_CENSUS_PLAN_OUT": str(plan_path),
    }
    return subprocess.run(
        ["bash", str(Path(script_directory) / "lane-census-check.sh"), *arguments],
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
