# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20173: Coding Plan URLs cannot activate system callers."""

import logging
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.adapters.llm.plugin_llm import _LLM_URL_ENV_VARS, PluginLlm
from omnibase_infra.runtime.models import ModelDomainPluginConfig

pytestmark = pytest.mark.unit


def _config() -> ModelDomainPluginConfig:
    return ModelDomainPluginConfig(
        container=MagicMock(),
        event_bus=MagicMock(),  # transport-mock-ok: should_activate never touches the bus
        correlation_id=uuid4(),
        input_topic="requests",
        output_topic="responses",
        consumer_group="test",
    )


@pytest.mark.parametrize(
    ("url", "blocked"),
    [
        ("https://api.z.ai/api/coding/paas/v4", True),
        ("http://other.example/api/coding/paas/v4", True),
        ("https://api.z.ai/api/coding/", True),
        ("https://api.z.ai/api/paas/v4", False),
        ("http://localhost:8000/v1", False),
        ("https://example.org/v1?redirect=/api/coding/", False),
        ("https://example.org/api/coding-tools/v1", False),
        ("", False),
    ],
)
def test_coding_plan_url_predicate(url: str, blocked: bool) -> None:
    from omnibase_infra.adapters.llm.coding_plan_endpoint import is_coding_plan_endpoint

    assert is_coding_plan_endpoint(url) is blocked


@pytest.mark.parametrize("variable", _LLM_URL_ENV_VARS)
@pytest.mark.parametrize("blocked", [True, False])
def test_plugin_filters_coding_plan_urls(
    variable: str,
    blocked: bool,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    for name in _LLM_URL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    url = (
        "https://api.z.ai/api/coding/paas/v4"
        if blocked
        else "https://api.z.ai/api/paas/v4"
    )
    monkeypatch.setenv(variable, url)
    plugin = PluginLlm()
    with caplog.at_level(logging.WARNING):
        assert plugin.should_activate(_config()) is (not blocked)
    assert plugin._endpoints == ({} if blocked else {variable: url})
    warnings = [
        record for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert len(warnings) == int(blocked)
    if blocked:
        assert variable in warnings[0].message
        assert url not in warnings[0].message


def test_plugin_removes_previously_configured_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _LLM_URL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    plugin = PluginLlm()
    monkeypatch.setenv("LLM_GLM_URL", "https://api.z.ai/api/paas/v4")
    assert plugin.should_activate(_config())
    monkeypatch.setenv("LLM_GLM_URL", "https://api.z.ai/api/coding/paas/v4")
    assert not plugin.should_activate(_config())
    assert plugin._endpoints == {}
