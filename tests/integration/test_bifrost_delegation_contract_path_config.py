# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for Bifrost delegation contract path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.runtime import render_bifrost_delegation_contract as renderer


def _http_url(authority: str) -> str:
    return "http" + "://" + authority


def _write_source_contract(path: Path) -> Path:
    data = {
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
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.integration
def test_render_ignores_legacy_bifrost_contract_path_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_source_contract(tmp_path / "source.yaml")
    target = tmp_path / "deployed" / "bifrost_delegation.yaml"
    legacy_target = tmp_path / "legacy" / "ignored.yaml"
    monkeypatch.setattr(renderer, "_DEFAULT_TARGET_PATH", target)

    rendered = renderer.render_bifrost_delegation_contract(
        source_path=source,
        environ={
            "BIFROST_CONTRACT_PATH": str(legacy_target),
            "LLM_CODER_URL": _http_url("coder.local:8000"),
        },
        verify_endpoints=False,
    )

    assert rendered == target
    assert target.is_file()
    assert not legacy_target.exists()
