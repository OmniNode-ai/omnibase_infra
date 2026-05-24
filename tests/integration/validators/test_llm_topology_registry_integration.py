# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for the LLM topology and registry drift validators (OMN-11926).

These tests validate that:
- The real contracts/llm_endpoints.yaml passes topology validation
- The validators correctly reject known-bad inputs end-to-end
- The main() CLI entry points work against real and fixture files
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.validators.llm_registry_validator import (
    main as registry_main,
)
from omnibase_infra.validators.llm_registry_validator import (
    validate_registry_yaml,
)
from omnibase_infra.validators.llm_topology_validator import (
    main as topology_main,
)
from omnibase_infra.validators.llm_topology_validator import (
    validate_topology_yaml,
)

_REPO_ROOT = Path(__file__).parents[3]
_TOPOLOGY_CONTRACT = _REPO_ROOT / "contracts" / "llm_endpoints.yaml"


@pytest.mark.integration
def test_canonical_topology_contract_is_valid() -> None:
    """The committed contracts/llm_endpoints.yaml must have zero findings."""
    assert _TOPOLOGY_CONTRACT.exists(), (
        f"Topology contract not found: {_TOPOLOGY_CONTRACT}"
    )
    findings = validate_topology_yaml(_TOPOLOGY_CONTRACT)
    assert findings == [], "\n".join(f.format() for f in findings)


@pytest.mark.integration
def test_topology_main_exits_zero_on_canonical_contract() -> None:
    """CLI entry point exits 0 for the real contract file."""
    assert _TOPOLOGY_CONTRACT.exists(), (
        f"Topology contract not found: {_TOPOLOGY_CONTRACT}"
    )
    assert topology_main([str(_TOPOLOGY_CONTRACT)]) == 0


@pytest.mark.integration
def test_topology_validator_rejects_invalid_role_end_to_end(tmp_path: Path) -> None:
    data = {
        "endpoints": [
            {
                "slot_id": "bad-role-slot",
                "host": "192.168.86.201",
                "port": 8000,
                "endpoint_url": "http://192.168.86.201:8000",
                "url_env_var": "LLM_CODER_URL",
                "role_env_alias": None,
                "model_hf_id": "vendor/Model",
                "role": "not_a_real_role",
                "status": "running",
            }
        ]
    }
    p = tmp_path / "llm_endpoints.yaml"
    p.write_text(yaml.dump(data), encoding="utf-8")

    findings = validate_topology_yaml(p)
    exit_code = topology_main([str(p)])

    assert any("not_a_real_role" in f.message for f in findings)
    assert exit_code == 1


@pytest.mark.integration
def test_topology_validator_rejects_running_slot_with_null_host(tmp_path: Path) -> None:
    data = {
        "endpoints": [
            {
                "slot_id": "no-host-slot",
                "host": None,
                "port": 8000,
                "endpoint_url": "http://192.168.86.201:8000",
                "url_env_var": "LLM_CODER_URL",
                "role_env_alias": None,
                "model_hf_id": "vendor/Model",
                "role": "coder_slow",
                "status": "running",
            }
        ]
    }
    p = tmp_path / "llm_endpoints.yaml"
    p.write_text(yaml.dump(data), encoding="utf-8")

    findings = validate_topology_yaml(p)
    assert any("host" in f.message and "no-host-slot" in f.message for f in findings)
    assert topology_main([str(p)]) == 1


@pytest.mark.integration
def test_registry_validator_rejects_missing_cost_basis_end_to_end(
    tmp_path: Path,
) -> None:
    data = {
        "schema_version": "1.0.0",
        "model_registry_version": "1.0.0",
        "pricing_manifest_version": "2026-05-23-initial",
        "observed_at": "2026-05-23T00:00:00Z",
        "models": {
            "test-model": {
                "model_id": "test-model",
                "provider": "local",
                "endpoint_env": "LLM_CODER_URL",
                "pricing_per_1m_input": "0.00",
                "pricing_per_1m_output": "0.00",
                "context_window": 8192,
                "observed_at": "2026-05-23T00:00:00Z",
                # cost_basis intentionally omitted
            }
        },
    }
    p = tmp_path / "model_registry_v1.yaml"
    p.write_text(yaml.dump(data), encoding="utf-8")

    findings = validate_registry_yaml(p)
    exit_code = registry_main([str(p)])

    assert any("cost_basis" in f.message for f in findings)
    assert exit_code == 1


@pytest.mark.integration
def test_registry_validator_accepts_valid_local_model(tmp_path: Path) -> None:
    data = {
        "schema_version": "1.0.0",
        "model_registry_version": "1.0.0",
        "pricing_manifest_version": "2026-05-23-initial",
        "observed_at": "2026-05-23T00:00:00Z",
        "models": {
            "qwen3-coder-30b": {
                "model_id": "qwen3-coder-30b",
                "provider": "local",
                "endpoint_env": "LLM_CODER_URL",
                "cost_basis": "zero_marginal_api_cost",
                "pricing_per_1m_input": "0.00",
                "pricing_per_1m_output": "0.00",
                "context_window": 112000,
                "observed_at": "2026-05-23T00:00:00Z",
            }
        },
    }
    p = tmp_path / "model_registry_v1.yaml"
    p.write_text(yaml.dump(data), encoding="utf-8")

    findings = validate_registry_yaml(p)
    exit_code = registry_main([str(p)])

    assert findings == []
    assert exit_code == 0
