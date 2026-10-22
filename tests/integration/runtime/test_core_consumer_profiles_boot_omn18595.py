# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Every consumer-attached profile installed core blesses boots in infra (OMN-18595).

Core 0.47.50 registers ``projection-writer-run-trace`` (OMN-19972) as a
standalone consumer-attached profile. A process started with that
``RUNTIME_PROFILE`` resolves its role through the same two readers the kernel
uses; an unregistered name makes both refuse and the process wire nothing.
"""

from __future__ import annotations

import pytest

from omnibase_core.constants.constants_runtime_profiles import (
    CONSUMER_ATTACHED_RUNTIME_PROFILES,
)
from omnibase_infra.runtime.runtime_profile import (
    load_runtime_profile,
    resolve_runtime_profile_name,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("profile_name", sorted(CONSUMER_ATTACHED_RUNTIME_PROFILES))
def test_consumer_attached_profile_resolves_from_environment(
    profile_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNTIME_PROFILE", profile_name)

    assert resolve_runtime_profile_name() == profile_name
    assert load_runtime_profile().name == profile_name


def test_run_trace_writer_profile_is_consumer_attached() -> None:
    assert "projection-writer-run-trace" in CONSUMER_ATTACHED_RUNTIME_PROFILES
    assert load_runtime_profile("projection-writer-run-trace").prefetch_policy == (
        "disabled"
    )
