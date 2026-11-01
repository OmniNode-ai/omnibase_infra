# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration test: call sites carry the timeouts declared on disk (OMN-21719)."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.models.runtime_timeouts import (
    RUNTIME_TIMEOUTS,
    ModelRuntimeTimeouts,
    load_runtime_timeouts,
    model_runtime_timeouts,
)
from omnibase_infra.probes import capability_probe


@pytest.mark.integration
def test_shipped_file_loads_and_binds_capability_probe() -> None:
    """The file shipped beside the model loads and the probe module binds its values."""
    shipped = Path(model_runtime_timeouts.__file__).parent / "runtime_timeouts.yaml"
    loaded = ModelRuntimeTimeouts.from_file(shipped)
    assert loaded == load_runtime_timeouts() == RUNTIME_TIMEOUTS
    assert loaded.capability_probe_ttl_seconds == capability_probe.PROBE_TTL_SECONDS
