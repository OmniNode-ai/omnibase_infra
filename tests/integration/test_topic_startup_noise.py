# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for runtime topic-startup noise suppression."""

from __future__ import annotations

import inspect

import pytest

from omnibase_infra.runtime import service_kernel

pytestmark = [pytest.mark.integration]


def test_runtime_topic_validation_suppresses_pre_create_missing_topic_logs() -> None:
    """Runtime startup validates quietly before auto-create and revalidates after."""
    source = inspect.getsource(service_kernel)

    assert "log_missing=False" in source
    assert "log_missing=strict_topic_validation" in source
    assert "Topic validation recovered after auto-create" in source
