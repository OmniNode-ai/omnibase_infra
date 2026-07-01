# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Readiness-config knob tests (OMN-13237, §3.7/§3.9).

Named knobs, not magic numbers. The model itself is pure (no env reads); the
kernel ``resolve_topic_readiness_config`` is the env-resolution boundary and
falls back to the field default for any malformed override so a bad value never
wedges boot.
"""

from __future__ import annotations

import pytest

from omnibase_infra.event_bus.model_topic_readiness_config import (
    DEFAULT_MAX_CONCURRENT_CONTRACT_ATTACH,
    DEFAULT_READINESS_MAX_ATTEMPTS,
    DEFAULT_READINESS_POLL_INTERVAL_MS,
    DEFAULT_READINESS_TIMEOUT_SECONDS,
    ModelTopicReadinessConfig,
)
from omnibase_infra.runtime.service_kernel import resolve_topic_readiness_config


def test_defaults_are_conservative() -> None:
    cfg = ModelTopicReadinessConfig()
    assert cfg.readiness_timeout_seconds == DEFAULT_READINESS_TIMEOUT_SECONDS
    assert cfg.readiness_poll_interval_ms == DEFAULT_READINESS_POLL_INTERVAL_MS
    assert cfg.max_attempts == DEFAULT_READINESS_MAX_ATTEMPTS
    assert cfg.max_concurrent_contract_attach == DEFAULT_MAX_CONCURRENT_CONTRACT_ATTACH


def test_config_is_frozen() -> None:
    cfg = ModelTopicReadinessConfig()
    with pytest.raises(Exception):
        cfg.max_attempts = 99  # type: ignore[misc]


def test_resolve_reads_valid_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONEX_TOPIC_READINESS_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("ONEX_TOPIC_READINESS_POLL_INTERVAL_MS", "250")
    monkeypatch.setenv("ONEX_TOPIC_READINESS_MAX_ATTEMPTS", "10")
    monkeypatch.setenv("ONEX_MAX_CONCURRENT_CONTRACT_ATTACH", "8")
    cfg = resolve_topic_readiness_config()
    assert cfg.readiness_timeout_seconds == 12.5
    assert cfg.readiness_poll_interval_ms == 250
    assert cfg.max_attempts == 10
    assert cfg.max_concurrent_contract_attach == 8


@pytest.mark.parametrize("bad", ["nan", "inf", "-1", "0", "abc", ""])
def test_resolve_rejects_invalid_timeout(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    monkeypatch.setenv("ONEX_TOPIC_READINESS_TIMEOUT_SECONDS", bad)
    cfg = resolve_topic_readiness_config()
    assert cfg.readiness_timeout_seconds == DEFAULT_READINESS_TIMEOUT_SECONDS


@pytest.mark.parametrize("bad", ["0", "-3", "abc", ""])
def test_resolve_rejects_invalid_int_knobs(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    monkeypatch.setenv("ONEX_MAX_CONCURRENT_CONTRACT_ATTACH", bad)
    cfg = resolve_topic_readiness_config()
    assert cfg.max_concurrent_contract_attach == DEFAULT_MAX_CONCURRENT_CONTRACT_ATTACH
