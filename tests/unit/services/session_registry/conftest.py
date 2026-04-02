# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Session registry test fixtures (OMN-7227)."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _set_memgraph_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide OMNIMEMORY_MEMGRAPH_* env vars for graph projector config."""
    monkeypatch.setenv("OMNIMEMORY_MEMGRAPH_HOST", "localhost")
    monkeypatch.setenv("OMNIMEMORY_MEMGRAPH_PORT", "7687")
