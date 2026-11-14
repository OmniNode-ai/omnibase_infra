# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime image dependency pin tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]


def test_runtime_image_includes_session_orchestrator_probe_toolchain(
    dockerfile_path: Path,
) -> None:
    """The deployed runtime owns dry-run health and repo probes."""
    dockerfile = dockerfile_path.read_text(encoding="utf-8")
    runtime_match = re.search(
        r"^FROM\s+.+\s+AS\s+runtime\s*$",
        dockerfile,
        flags=re.MULTILINE,
    )
    assert runtime_match is not None, "Runtime stage not found in Dockerfile.runtime"
    runtime_stage = dockerfile[runtime_match.start() :]
    runtime_marker_length = len(runtime_match.group(0))
    next_stage_match = re.search(
        r"^FROM\s+",
        runtime_stage[runtime_marker_length:],
        flags=re.MULTILINE,
    )
    if next_stage_match is not None:
        runtime_stage = runtime_stage[
            : runtime_marker_length + next_stage_match.start()
        ]

    assert "OMNI_HOME=/app" in runtime_stage
    assert "COPY --from=uv-bin /uv /uvx /usr/local/bin/" in runtime_stage
    assert re.search(r"^\s+git \\\s*$", runtime_stage, flags=re.MULTILINE) is not None
    assert (
        re.search(r"^\s+openssh-client \\\s*$", runtime_stage, flags=re.MULTILINE)
        is not None
    )
