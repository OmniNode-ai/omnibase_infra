# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26311: an unmapped logical name is warned about once, not once per call.

A poller that asks for ``GITHUB_TOKEN`` every tick on a runtime with no mapping
for it logged a ``no_mapping`` warning on every call (2,639 in 30 minutes). The
miss is still counted every call, and every call still returns None; only the
WARNING is emitted once per logical name, later misses go to DEBUG.
"""

from __future__ import annotations

import logging

import pytest

from omnibase_infra.runtime.models.model_secret_resolver_config import (
    ModelSecretResolverConfig,
)
from omnibase_infra.runtime.secret_resolver import SecretResolver

pytestmark = pytest.mark.unit

_LOGGER = "omnibase_infra.runtime.secret_resolver"


def _resolver() -> SecretResolver:
    return SecretResolver(
        config=ModelSecretResolverConfig(enable_convention_fallback=False)
    )


def _warnings(caplog: pytest.LogCaptureFixture, name: str) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and getattr(r, "logical_name", "") == name
    ]


def test_unmapped_name_sync_warns_once_across_many_calls(
    caplog: pytest.LogCaptureFixture,
) -> None:
    resolver = _resolver()
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        for _ in range(50):
            assert resolver.get_secret("GITHUB_TOKEN", required=False) is None
    assert len(_warnings(caplog, "GITHUB_TOKEN")) == 1


@pytest.mark.asyncio
async def test_unmapped_name_async_warns_once_across_many_calls(
    caplog: pytest.LogCaptureFixture,
) -> None:
    resolver = _resolver()
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        for _ in range(50):
            assert (
                await resolver.get_secret_async("GITHUB_TOKEN", required=False) is None
            )
    assert len(_warnings(caplog, "GITHUB_TOKEN")) == 1


def test_each_distinct_unmapped_name_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    resolver = _resolver()
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        for _ in range(5):
            resolver.get_secret("GITHUB_TOKEN", required=False)
            resolver.get_secret("LINEAR_API_KEY", required=False)
    assert len(_warnings(caplog, "GITHUB_TOKEN")) == 1
    assert len(_warnings(caplog, "LINEAR_API_KEY")) == 1
