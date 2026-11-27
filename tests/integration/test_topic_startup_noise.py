# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for runtime topic-startup noise suppression."""

from __future__ import annotations

import inspect

import pytest

from omnibase_infra.handlers import boot_topic_provisioning
from omnibase_infra.runtime import service_kernel

pytestmark = [pytest.mark.integration]


def test_runtime_topic_validation_suppresses_pre_create_missing_topic_logs() -> None:
    """Runtime startup validates quietly before auto-create and revalidates after."""
    kernel_source = inspect.getsource(service_kernel)
    # OMN-26359: the auto-create pass moved out of the kernel into
    # boot_topic_provisioning; the kernel passes strict_topic_validation in.
    provisioning_source = inspect.getsource(boot_topic_provisioning)

    assert "log_missing=False" in kernel_source
    assert "strict_validation=strict_topic_validation" in kernel_source
    assert "log_missing=strict_validation" in provisioning_source
    assert "Topic validation recovered after auto-create" in provisioning_source
