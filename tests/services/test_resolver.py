# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for `resolve_project_tracker()` — central tracker DI authority.

Three required branches (per plan Task 4.1):
    1. No token → LocalStubProjectTracker
    2. LINEAR_TOKEN present → AdapterLinearProjectTracker
    3. Construction failure → fail-soft to LocalStubProjectTracker (never raises)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from omnibase_infra.adapters.project_tracker.local_stub_project_tracker import (
    LocalStubProjectTracker,
)
from omnibase_infra.services.project_tracker.resolver import resolve_project_tracker

pytestmark = pytest.mark.unit


class TestResolveProjectTracker:
    def test_returns_local_stub_when_no_token(self, tmp_path: Path) -> None:
        with patch.dict("os.environ", {}, clear=True):
            tracker = resolve_project_tracker(state_root=tmp_path)
            assert isinstance(tracker, LocalStubProjectTracker)

    def test_returns_linear_adapter_when_token_present(self, tmp_path: Path) -> None:
        pytest.importorskip(
            "omnibase_infra.adapters.project_tracker.linear_project_tracker_adapter",
            reason="AdapterLinearProjectTracker ships in OMN-8816; skip until merged",
        )
        from omnibase_infra.adapters.project_tracker.linear_project_tracker_adapter import (
            AdapterLinearProjectTracker,
        )

        with patch.dict("os.environ", {"LINEAR_TOKEN": "fake-token"}, clear=True):
            tracker = resolve_project_tracker(state_root=tmp_path)
            assert isinstance(tracker, AdapterLinearProjectTracker)

    def test_never_raises_on_construction_failure(self, tmp_path: Path) -> None:
        with patch.dict("os.environ", {"LINEAR_TOKEN": "bad-token"}, clear=True):
            tracker = resolve_project_tracker(
                state_root=tmp_path,
                _force_construction_error=True,
            )
            assert isinstance(tracker, LocalStubProjectTracker)

    def test_api_key_env_var_also_selects_linear(self, tmp_path: Path) -> None:
        """LINEAR_API_KEY must be honored in addition to LINEAR_TOKEN."""
        pytest.importorskip(
            "omnibase_infra.adapters.project_tracker.linear_project_tracker_adapter",
            reason="AdapterLinearProjectTracker ships in OMN-8816; skip until merged",
        )
        from omnibase_infra.adapters.project_tracker.linear_project_tracker_adapter import (
            AdapterLinearProjectTracker,
        )

        with patch.dict("os.environ", {"LINEAR_API_KEY": "fake-token"}, clear=True):
            tracker = resolve_project_tracker(state_root=tmp_path)
            assert isinstance(tracker, AdapterLinearProjectTracker)
