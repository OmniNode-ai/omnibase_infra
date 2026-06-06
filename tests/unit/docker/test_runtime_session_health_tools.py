# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime image tool contract for session health probes."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

DOCKERFILE = Path(__file__).resolve().parents[3] / "docker" / "Dockerfile.runtime"


def test_runtime_image_installs_session_health_probe_tools() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "# git is required by gh and session health repo probes." in dockerfile
    assert "\n    git \\\n" in dockerfile
    assert "COPY --from=uv-bin /uv /uvx /usr/local/bin/" in dockerfile
