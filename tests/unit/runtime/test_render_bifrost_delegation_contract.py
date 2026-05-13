# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for runtime Bifrost delegation contract rendering."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.render_bifrost_delegation_contract import (
    render_bifrost_delegation_contract,
)


def _http_url(authority: str) -> str:
    return "http" + "://" + authority


def _source_contract(path: Path, *, required: bool = False) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "config_version": "1.1.0",
                "schema_version": "bifrost_delegation.v1",
                "backends": [
                    {
                        "backend_id": "local-qwen-coder-30b",
                        "base_url_env": "LLM_CODER_URL",
                        "required": required,
                        "endpoint_url": "",
                        "model_name": "qwen-test",
                        "tier": "local",
                        "timeout_ms": 30000,
                        "capabilities": ["code_generation"],
                    }
                ],
                "routing_rules": [
                    {
                        "rule_id": "d4e5f6a7-0001-4000-8000-000000000001",
                        "priority": 10,
                        "task_class": "test",
                        "task_class_contract_version": "1.0.0",
                        "backend_policy_version": "1.0.0",
                        "match_operation_types": ["chat_completion"],
                        "match_capabilities": ["code_generation"],
                        "latency_sla_ms": 30000,
                        "backend_ids": ["local-qwen-coder-30b"],
                        "fallback_policy": {
                            "action": "return_error",
                            "max_retries": 0,
                            "on_exhaust": "return_error",
                        },
                        "shadow_policy_id": "e5f6a7b8-0001-4000-8000-000000000001",
                    }
                ],
                "default_backends": ["local-qwen-coder-30b"],
                "circuit_breaker": {"failure_threshold": 5, "window_seconds": 30},
                "failover": {"max_attempts": 3, "backoff_base_ms": 500},
                "shadow_mode": {
                    "enabled": False,
                    "policy_version": "unknown",
                    "log_sample_rate": 1.0,
                    "comparison_logging_enabled": True,
                    "max_shadow_latency_ms": 5.0,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _source_contract_with_optional_backend(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "config_version": "1.1.0",
                "schema_version": "bifrost_delegation.v1",
                "backends": [
                    {
                        "backend_id": "local-qwen-coder-30b",
                        "base_url_env": "LLM_CODER_URL",
                        "required": True,
                        "endpoint_url": "",
                        "model_name": "qwen-test",
                        "tier": "local",
                        "timeout_ms": 30000,
                        "capabilities": ["code_generation"],
                    },
                    {
                        "backend_id": "local-deepseek-r1-14b",
                        "base_url_env": "LLM_DEEPSEEK_R1_URL",
                        "endpoint_url": "",
                        "model_name": "deepseek-test",
                        "tier": "local",
                        "timeout_ms": 30000,
                        "capabilities": ["research"],
                    },
                ],
                "routing_rules": [
                    {
                        "rule_id": "d4e5f6a7-0001-4000-8000-000000000001",
                        "priority": 10,
                        "task_class": "test",
                        "task_class_contract_version": "1.0.0",
                        "backend_policy_version": "1.0.0",
                        "match_operation_types": ["chat_completion"],
                        "match_capabilities": ["code_generation"],
                        "latency_sla_ms": 30000,
                        "backend_ids": ["local-qwen-coder-30b"],
                        "fallback_policy": {
                            "action": "return_error",
                            "max_retries": 0,
                            "on_exhaust": "return_error",
                        },
                        "shadow_policy_id": "e5f6a7b8-0001-4000-8000-000000000001",
                    }
                ],
                "default_backends": ["local-qwen-coder-30b"],
                "circuit_breaker": {"failure_threshold": 5, "window_seconds": 30},
                "failover": {"max_attempts": 3, "backoff_base_ms": 500},
                "shadow_mode": {
                    "enabled": False,
                    "policy_version": "unknown",
                    "log_sample_rate": 1.0,
                    "comparison_logging_enabled": True,
                    "max_shadow_latency_ms": 5.0,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.unit
def test_render_populates_endpoint_from_declared_provider_env(tmp_path: Path) -> None:
    source = _source_contract(tmp_path / "source.yaml")
    target = tmp_path / "rendered" / "bifrost_delegation.yaml"

    rendered = render_bifrost_delegation_contract(
        source_path=source,
        target_path=target,
        environ={"LLM_CODER_URL": _http_url("coder.local:8000")},
    )

    assert rendered == target
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert data["backends"][0]["endpoint_url"] == _http_url("coder.local:8000")


@pytest.mark.unit
def test_render_fails_for_missing_required_provider_env(tmp_path: Path) -> None:
    source = _source_contract(tmp_path / "source.yaml", required=True)
    target = tmp_path / "rendered.yaml"

    with pytest.raises(ProtocolConfigurationError, match="LLM_CODER_URL"):
        render_bifrost_delegation_contract(
            source_path=source,
            target_path=target,
            environ={},
        )


@pytest.mark.unit
def test_existing_populated_target_is_reused(tmp_path: Path) -> None:
    source = _source_contract(tmp_path / "source.yaml", required=True)
    target = _source_contract(tmp_path / "target.yaml")
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    data["backends"][0]["endpoint_url"] = _http_url("pre-rendered.local:8000")
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    rendered = render_bifrost_delegation_contract(
        source_path=source,
        target_path=target,
        environ={},
    )

    assert rendered == target
    loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert loaded["backends"][0]["endpoint_url"] == _http_url("pre-rendered.local:8000")


@pytest.mark.unit
def test_required_endpoint_probe_failure_blocks_render(tmp_path: Path) -> None:
    source = _source_contract(tmp_path / "source.yaml", required=True)
    target = tmp_path / "rendered.yaml"

    with pytest.raises(ProtocolConfigurationError, match="qwen-test"):
        render_bifrost_delegation_contract(
            source_path=source,
            target_path=target,
            environ={"LLM_CODER_URL": _http_url("coder.local:8000")},
            verify_endpoints=True,
            endpoint_probe=lambda _url, model, _timeout: f"missing model {model}",
        )


@pytest.mark.unit
def test_optional_endpoint_probe_failure_leaves_backend_unpopulated(
    tmp_path: Path,
) -> None:
    source = _source_contract_with_optional_backend(tmp_path / "source.yaml")
    target = tmp_path / "rendered.yaml"

    def probe(url: str, _model: str, _timeout: float) -> str | None:
        if url == _http_url("deepseek.local:8101"):
            return "model endpoint timed out"
        return None

    render_bifrost_delegation_contract(
        source_path=source,
        target_path=target,
        environ={
            "LLM_CODER_URL": _http_url("coder.local:8000"),
            "LLM_DEEPSEEK_R1_URL": _http_url("deepseek.local:8101"),
        },
        verify_endpoints=True,
        endpoint_probe=probe,
    )

    loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert loaded["backends"][0]["endpoint_url"] == _http_url("coder.local:8000")
    assert loaded["backends"][1]["endpoint_url"] == ""
