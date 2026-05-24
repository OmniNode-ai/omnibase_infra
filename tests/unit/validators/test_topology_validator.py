# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the LLM topology contract drift validator (OMN-11926)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from omnibase_infra.validators.llm_topology_validator import (
    Finding,
    validate_topology_yaml,
)


def _write_yaml(tmp_path: Path, data: object, name: str = "llm_endpoints.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.dump(data), encoding="utf-8")
    return p


@pytest.mark.unit
def test_valid_running_endpoint_passes(tmp_path: Path) -> None:
    data = {
        "endpoints": [
            {
                "slot_id": "coder-5090",
                "host": "192.168.86.201",
                "port": 8000,
                "endpoint_url": "http://192.168.86.201:8000",
                "url_env_var": "LLM_CODER_URL",
                "role_env_alias": "LLM_CODER_SLOW_URL",
                "model_hf_id": "vendor/SomeModel-7B",
                "role": "coder_slow",
                "status": "running",
            }
        ]
    }
    p = _write_yaml(tmp_path, data)
    findings = validate_topology_yaml(p)
    assert findings == []


@pytest.mark.unit
def test_valid_planned_endpoint_passes(tmp_path: Path) -> None:
    data = {
        "endpoints": [
            {
                "slot_id": "vision-planned",
                "host": None,
                "port": None,
                "endpoint_url": None,
                "url_env_var": None,
                "role_env_alias": "LLM_VISION_URL",
                "model_hf_id": None,
                "role": "vision",
                "status": "planned",
            }
        ]
    }
    p = _write_yaml(tmp_path, data)
    assert validate_topology_yaml(p) == []


@pytest.mark.unit
def test_valid_disabled_endpoint_passes(tmp_path: Path) -> None:
    data = {
        "endpoints": [
            {
                "slot_id": "embeddings-200",
                "host": "192.168.86.200",
                "port": 8100,
                "endpoint_url": "http://192.168.86.200:8100",
                "url_env_var": None,
                "role_env_alias": None,
                "model_hf_id": "vendor/Embed-8B",
                "role": "embedding",
                "status": "disabled",
            }
        ]
    }
    p = _write_yaml(tmp_path, data)
    assert validate_topology_yaml(p) == []


@pytest.mark.unit
def test_running_endpoint_missing_host_fails(tmp_path: Path) -> None:
    data = {
        "endpoints": [
            {
                "slot_id": "coder-5090",
                "host": None,
                "port": 8000,
                "endpoint_url": "http://192.168.86.201:8000",
                "url_env_var": "LLM_CODER_URL",
                "role_env_alias": None,
                "model_hf_id": "vendor/SomeModel-7B",
                "role": "coder_slow",
                "status": "running",
            }
        ]
    }
    p = _write_yaml(tmp_path, data)
    findings = validate_topology_yaml(p)
    assert any("host" in f.message and "coder-5090" in f.message for f in findings)


@pytest.mark.unit
def test_running_endpoint_missing_port_fails(tmp_path: Path) -> None:
    data = {
        "endpoints": [
            {
                "slot_id": "coder-5090",
                "host": "192.168.86.201",
                "port": None,
                "endpoint_url": "http://192.168.86.201:8000",
                "url_env_var": "LLM_CODER_URL",
                "role_env_alias": None,
                "model_hf_id": "vendor/SomeModel-7B",
                "role": "coder_slow",
                "status": "running",
            }
        ]
    }
    p = _write_yaml(tmp_path, data)
    findings = validate_topology_yaml(p)
    assert any("port" in f.message and "coder-5090" in f.message for f in findings)


@pytest.mark.unit
def test_running_endpoint_missing_endpoint_url_fails(tmp_path: Path) -> None:
    data = {
        "endpoints": [
            {
                "slot_id": "coder-5090",
                "host": "192.168.86.201",
                "port": 8000,
                "endpoint_url": None,
                "url_env_var": "LLM_CODER_URL",
                "role_env_alias": None,
                "model_hf_id": "vendor/SomeModel-7B",
                "role": "coder_slow",
                "status": "running",
            }
        ]
    }
    p = _write_yaml(tmp_path, data)
    findings = validate_topology_yaml(p)
    assert any(
        "endpoint_url" in f.message and "coder-5090" in f.message for f in findings
    )


@pytest.mark.unit
def test_running_endpoint_missing_model_hf_id_fails(tmp_path: Path) -> None:
    data = {
        "endpoints": [
            {
                "slot_id": "coder-5090",
                "host": "192.168.86.201",
                "port": 8000,
                "endpoint_url": "http://192.168.86.201:8000",
                "url_env_var": "LLM_CODER_URL",
                "role_env_alias": None,
                "model_hf_id": None,
                "role": "coder_slow",
                "status": "running",
            }
        ]
    }
    p = _write_yaml(tmp_path, data)
    findings = validate_topology_yaml(p)
    assert any(
        "model_hf_id" in f.message and "coder-5090" in f.message for f in findings
    )


@pytest.mark.unit
def test_endpoint_url_host_port_mismatch_fails(tmp_path: Path) -> None:
    data = {
        "endpoints": [
            {
                "slot_id": "coder-5090",
                "host": "192.168.86.201",
                "port": 9999,
                "endpoint_url": "http://192.168.86.201:8000",
                "url_env_var": "LLM_CODER_URL",
                "role_env_alias": None,
                "model_hf_id": "vendor/SomeModel-7B",
                "role": "coder_slow",
                "status": "running",
            }
        ]
    }
    p = _write_yaml(tmp_path, data)
    findings = validate_topology_yaml(p)
    assert any(
        "endpoint_url" in f.message.lower() or "mismatch" in f.message.lower()
        for f in findings
    )


@pytest.mark.unit
def test_invalid_role_fails(tmp_path: Path) -> None:
    data = {
        "endpoints": [
            {
                "slot_id": "coder-5090",
                "host": "192.168.86.201",
                "port": 8000,
                "endpoint_url": "http://192.168.86.201:8000",
                "url_env_var": "LLM_CODER_URL",
                "role_env_alias": None,
                "model_hf_id": "vendor/SomeModel-7B",
                "role": "invalid_role_xyz",
                "status": "running",
            }
        ]
    }
    p = _write_yaml(tmp_path, data)
    findings = validate_topology_yaml(p)
    assert any("role" in f.message and "coder-5090" in f.message for f in findings)


@pytest.mark.unit
def test_invalid_status_fails(tmp_path: Path) -> None:
    data = {
        "endpoints": [
            {
                "slot_id": "coder-5090",
                "host": "192.168.86.201",
                "port": 8000,
                "endpoint_url": "http://192.168.86.201:8000",
                "url_env_var": "LLM_CODER_URL",
                "role_env_alias": None,
                "model_hf_id": "vendor/SomeModel-7B",
                "role": "coder_slow",
                "status": "bogus_status",
            }
        ]
    }
    p = _write_yaml(tmp_path, data)
    findings = validate_topology_yaml(p)
    assert any("status" in f.message and "coder-5090" in f.message for f in findings)


@pytest.mark.unit
def test_missing_slot_id_fails(tmp_path: Path) -> None:
    data = {
        "endpoints": [
            {
                "host": "192.168.86.201",
                "port": 8000,
                "endpoint_url": "http://192.168.86.201:8000",
                "url_env_var": "LLM_CODER_URL",
                "role_env_alias": None,
                "model_hf_id": "vendor/SomeModel-7B",
                "role": "coder_slow",
                "status": "running",
            }
        ]
    }
    p = _write_yaml(tmp_path, data)
    findings = validate_topology_yaml(p)
    assert any("slot_id" in f.message for f in findings)


@pytest.mark.unit
def test_duplicate_url_env_var_fails(tmp_path: Path) -> None:
    data = {
        "endpoints": [
            {
                "slot_id": "slot-a",
                "host": "192.168.86.201",
                "port": 8000,
                "endpoint_url": "http://192.168.86.201:8000",
                "url_env_var": "LLM_CODER_URL",
                "role_env_alias": None,
                "model_hf_id": "vendor/ModelA",
                "role": "coder_slow",
                "status": "running",
            },
            {
                "slot_id": "slot-b",
                "host": "192.168.86.201",
                "port": 8001,
                "endpoint_url": "http://192.168.86.201:8001",
                "url_env_var": "LLM_CODER_URL",
                "role_env_alias": None,
                "model_hf_id": "vendor/ModelB",
                "role": "coder_fast",
                "status": "running",
            },
        ]
    }
    p = _write_yaml(tmp_path, data)
    findings = validate_topology_yaml(p)
    assert any(
        "LLM_CODER_URL" in f.message and "duplicate" in f.message.lower()
        for f in findings
    )


@pytest.mark.unit
def test_duplicate_slot_id_fails(tmp_path: Path) -> None:
    data = {
        "endpoints": [
            {
                "slot_id": "same-slot",
                "host": "192.168.86.201",
                "port": 8000,
                "endpoint_url": "http://192.168.86.201:8000",
                "url_env_var": "LLM_CODER_URL",
                "role_env_alias": None,
                "model_hf_id": "vendor/ModelA",
                "role": "coder_slow",
                "status": "running",
            },
            {
                "slot_id": "same-slot",
                "host": "192.168.86.201",
                "port": 8001,
                "endpoint_url": "http://192.168.86.201:8001",
                "url_env_var": "LLM_CODER_FAST_URL",
                "role_env_alias": None,
                "model_hf_id": "vendor/ModelB",
                "role": "coder_fast",
                "status": "running",
            },
        ]
    }
    p = _write_yaml(tmp_path, data)
    findings = validate_topology_yaml(p)
    assert any(
        "same-slot" in f.message and "duplicate" in f.message.lower() for f in findings
    )


@pytest.mark.unit
def test_disabled_endpoint_with_url_env_var_fails(tmp_path: Path) -> None:
    """Disabled slots must not own runtime env vars per contract comment."""
    data = {
        "endpoints": [
            {
                "slot_id": "embeddings-200",
                "host": "192.168.86.200",
                "port": 8100,
                "endpoint_url": "http://192.168.86.200:8100",
                "url_env_var": "LLM_EMBEDDING_URL",
                "role_env_alias": None,
                "model_hf_id": "vendor/Embed",
                "role": "embedding",
                "status": "disabled",
            }
        ]
    }
    p = _write_yaml(tmp_path, data)
    findings = validate_topology_yaml(p)
    assert any(
        "disabled" in f.message.lower() and "url_env_var" in f.message for f in findings
    )


@pytest.mark.unit
def test_planned_endpoint_with_host_fails(tmp_path: Path) -> None:
    """Planned slots must have null host/port/endpoint_url/model_hf_id."""
    data = {
        "endpoints": [
            {
                "slot_id": "vision-planned",
                "host": "192.168.86.201",
                "port": None,
                "endpoint_url": None,
                "url_env_var": None,
                "role_env_alias": "LLM_VISION_URL",
                "model_hf_id": None,
                "role": "vision",
                "status": "planned",
            }
        ]
    }
    p = _write_yaml(tmp_path, data)
    findings = validate_topology_yaml(p)
    assert any(
        "planned" in f.message.lower() and "vision-planned" in f.message
        for f in findings
    )


@pytest.mark.unit
def test_missing_endpoints_key_fails(tmp_path: Path) -> None:
    p = tmp_path / "llm_endpoints.yaml"
    p.write_text("not_endpoints:\n  - foo: bar\n", encoding="utf-8")
    findings = validate_topology_yaml(p)
    assert any("endpoints" in f.message for f in findings)


@pytest.mark.unit
def test_non_yaml_file_fails(tmp_path: Path) -> None:
    p = tmp_path / "llm_endpoints.yaml"
    p.write_text(": invalid: yaml: [unclosed", encoding="utf-8")
    findings = validate_topology_yaml(p)
    assert any(
        "parse" in f.message.lower() or "yaml" in f.message.lower() for f in findings
    )


@pytest.mark.unit
def test_missing_file_fails(tmp_path: Path) -> None:
    p = tmp_path / "nonexistent.yaml"
    findings = validate_topology_yaml(p)
    assert any(
        "not found" in f.message.lower() or "exist" in f.message.lower()
        for f in findings
    )


@pytest.mark.unit
def test_canonical_contract_file_is_valid() -> None:
    """The actual contract file ships with zero findings."""
    contract = Path(__file__).parents[4] / "contracts" / "llm_endpoints.yaml"
    if not contract.exists():
        pytest.skip("Contract file not found in worktree")
    findings = validate_topology_yaml(contract)
    assert findings == [], "\n".join(f.message for f in findings)


@pytest.mark.unit
def test_finding_has_slot_and_rule(tmp_path: Path) -> None:
    data = {
        "endpoints": [
            {
                "slot_id": "bad-slot",
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
    p = _write_yaml(tmp_path, data)
    findings = validate_topology_yaml(p)
    assert len(findings) >= 1
    f = findings[0]
    assert isinstance(f, Finding)
    assert f.slot_id
    assert f.rule
    assert f.message


@pytest.mark.unit
def test_main_exits_nonzero_on_violation(tmp_path: Path) -> None:
    from omnibase_infra.validators.llm_topology_validator import main

    data = {
        "endpoints": [
            {
                "slot_id": "bad",
                "host": None,
                "port": None,
                "endpoint_url": None,
                "url_env_var": None,
                "role_env_alias": None,
                "model_hf_id": None,
                "role": "invalid_role",
                "status": "running",
            }
        ]
    }
    p = _write_yaml(tmp_path, data)
    assert main([str(p)]) != 0


@pytest.mark.unit
def test_main_exits_zero_on_clean_file(tmp_path: Path) -> None:
    from omnibase_infra.validators.llm_topology_validator import main

    data = {
        "endpoints": [
            {
                "slot_id": "clean-slot",
                "host": "192.168.86.201",
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
    p = _write_yaml(tmp_path, data)
    assert main([str(p)]) == 0
