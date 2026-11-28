# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Reading and writing branch-install entry points in a container (OMN-15920)."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable, Sequence

from omnibase_infra.enums.enum_dev_lane_branch_install_file_state import (
    EnumDevLaneBranchInstallFileState as State,
)
from omnibase_infra.models.model_dev_lane_branch_install import (
    ModelDevLaneBranchInstall,
)
from omnibase_infra.models.model_dev_lane_branch_install_entry_point import (
    ModelDevLaneBranchInstallEntryPoint,
)
from omnibase_infra.models.model_dev_lane_branch_install_file_readback import (
    ModelDevLaneBranchInstallFileReadback,
)

Runner = Callable[..., subprocess.CompletedProcess[str]]


def parse_entry_points(
    values: Sequence[str],
) -> tuple[ModelDevLaneBranchInstallEntryPoint, ...]:
    """Parse ``DIST:GROUP:NAME=VALUE`` flags; ValueError names a malformed one."""
    out: list[ModelDevLaneBranchInstallEntryPoint] = []
    for raw in values:
        left, sep, value = raw.partition("=")
        parts = left.split(":")
        if not sep or len(parts) != 3:
            raise ValueError(f"--entry-point {raw!r} must be DIST:GROUP:NAME=VALUE")
        out.append(
            ModelDevLaneBranchInstallEntryPoint(
                dist=parts[0], group=parts[1], name=parts[2], value=value
            )
        )
    return tuple(out)


def put_entry_point(
    cmd_ok: Callable[..., object],
    container: str,
    ep: ModelDevLaneBranchInstallEntryPoint,
) -> str:
    """Write one entry point into the installed dist-info; "" is success."""
    if cmd_ok(ep.upsert_argv(container)) is None:
        return "the entry point could not be written into the dist-info"
    return ""


def _read(
    run: Runner,
    container: str,
    ep: ModelDevLaneBranchInstallEntryPoint,
    timeout: float,
) -> tuple[str, State, str]:
    try:
        result = run(ep.read_argv(container), timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "", State.UNREADABLE, type(exc).__name__
    if result.returncode != 0:
        return "", State.UNREADABLE, f"entry point read exited {result.returncode}"
    line = result.stdout.strip()
    if not line:
        return "", State.MISSING, "entry point absent"
    return hashlib.sha256(line.encode("utf-8")).hexdigest(), State.PRESENT, ""


def entry_point_rows(
    run: Runner,
    service: str,
    container: str,
    install: ModelDevLaneBranchInstall,
    timeout: float,
    no_container: str = "",
) -> list[ModelDevLaneBranchInstallFileReadback]:
    """One readback row per declared entry point; ``container`` "" means none ran."""
    rows: list[ModelDevLaneBranchInstallFileReadback] = []
    for ep in install.entry_points:
        if container:
            observed, state, detail = _read(run, container, ep, timeout)
            if state is State.PRESENT and observed != ep.sha256:
                state, detail = State.DRIFTED, "entry point line differs"
        else:
            observed, state, detail = "", State.NO_CONTAINER, no_container
        rows.append(
            ModelDevLaneBranchInstallFileReadback(
                service=service,
                container=container,
                path=ep.path,
                expected_sha256=ep.sha256,
                observed_sha256=observed,
                state=state,
                detail=detail,
            )
        )
    return rows


__all__ = ["entry_point_rows", "parse_entry_points", "put_entry_point"]
