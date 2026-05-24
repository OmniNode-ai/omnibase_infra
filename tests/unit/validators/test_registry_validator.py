# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the LLM model registry drift validator (OMN-11926)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.validators.llm_registry_validator import (
    Finding,
    validate_registry_yaml,
)


def _write_yaml(
    tmp_path: Path, data: object, name: str = "model_registry_v1.yaml"
) -> Path:
    p = tmp_path / name
    p.write_text(yaml.dump(data), encoding="utf-8")
    return p


def _minimal_model(
    model_id: str = "test-model",
    provider: str = "local",
    endpoint_env: str = "LLM_CODER_URL",
    cost_basis: str = "zero_marginal_api_cost",
    pricing_in: str = "0.00",
    pricing_out: str = "0.00",
    context_window: int = 8192,
    observed_at: str = "2026-05-23T00:00:00Z",
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "provider": provider,
        "endpoint_env": endpoint_env,
        "cost_basis": cost_basis,
        "pricing_per_1m_input": pricing_in,
        "pricing_per_1m_output": pricing_out,
        "context_window": context_window,
        "observed_at": observed_at,
    }


def _minimal_registry(models: dict[str, object] | None = None) -> dict[str, object]:
    if models is None:
        models = {"test-model": _minimal_model()}
    return {
        "schema_version": "1.0.0",
        "model_registry_version": "1.0.0",
        "pricing_manifest_version": "2026-05-23-initial",
        "observed_at": "2026-05-23T00:00:00Z",
        "models": models,
    }


@pytest.mark.unit
def test_valid_registry_passes(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path, _minimal_registry())
    findings = validate_registry_yaml(p)
    assert findings == []


@pytest.mark.unit
def test_missing_models_key_fails(tmp_path: Path) -> None:
    data = {
        "schema_version": "1.0.0",
        "model_registry_version": "1.0.0",
        "pricing_manifest_version": "2026-05-23-initial",
        "observed_at": "2026-05-23T00:00:00Z",
    }
    p = _write_yaml(tmp_path, data)
    findings = validate_registry_yaml(p)
    assert any("models" in f.message for f in findings)


@pytest.mark.unit
def test_missing_schema_version_fails(tmp_path: Path) -> None:
    data = _minimal_registry()
    del data["schema_version"]
    p = _write_yaml(tmp_path, data)
    findings = validate_registry_yaml(p)
    assert any("schema_version" in f.message for f in findings)


@pytest.mark.unit
def test_missing_pricing_manifest_version_fails(tmp_path: Path) -> None:
    data = _minimal_registry()
    del data["pricing_manifest_version"]
    p = _write_yaml(tmp_path, data)
    findings = validate_registry_yaml(p)
    assert any("pricing_manifest_version" in f.message for f in findings)


@pytest.mark.unit
def test_model_missing_model_id_fails(tmp_path: Path) -> None:
    model = _minimal_model()
    del model["model_id"]
    data = _minimal_registry({"test-model": model})
    p = _write_yaml(tmp_path, data)
    findings = validate_registry_yaml(p)
    assert any("model_id" in f.message for f in findings)


@pytest.mark.unit
def test_model_id_key_mismatch_fails(tmp_path: Path) -> None:
    """The model_id field inside the entry must match the registry key."""
    model = _minimal_model(model_id="wrong-id")
    data = _minimal_registry({"test-model": model})
    p = _write_yaml(tmp_path, data)
    findings = validate_registry_yaml(p)
    assert any("test-model" in f.message for f in findings)


@pytest.mark.unit
def test_model_missing_provider_fails(tmp_path: Path) -> None:
    model = _minimal_model()
    del model["provider"]
    data = _minimal_registry({"test-model": model})
    p = _write_yaml(tmp_path, data)
    findings = validate_registry_yaml(p)
    assert any("provider" in f.message for f in findings)


@pytest.mark.unit
def test_model_missing_endpoint_env_fails(tmp_path: Path) -> None:
    model = _minimal_model()
    del model["endpoint_env"]
    data = _minimal_registry({"test-model": model})
    p = _write_yaml(tmp_path, data)
    findings = validate_registry_yaml(p)
    assert any("endpoint_env" in f.message for f in findings)


@pytest.mark.unit
def test_model_missing_cost_basis_fails(tmp_path: Path) -> None:
    model = _minimal_model()
    del model["cost_basis"]
    data = _minimal_registry({"test-model": model})
    p = _write_yaml(tmp_path, data)
    findings = validate_registry_yaml(p)
    assert any("cost_basis" in f.message for f in findings)


@pytest.mark.unit
def test_invalid_cost_basis_fails(tmp_path: Path) -> None:
    model = _minimal_model(cost_basis="free_money")
    data = _minimal_registry({"test-model": model})
    p = _write_yaml(tmp_path, data)
    findings = validate_registry_yaml(p)
    assert any("cost_basis" in f.message for f in findings)


@pytest.mark.unit
def test_model_missing_pricing_input_fails(tmp_path: Path) -> None:
    model = _minimal_model()
    del model["pricing_per_1m_input"]
    data = _minimal_registry({"test-model": model})
    p = _write_yaml(tmp_path, data)
    findings = validate_registry_yaml(p)
    assert any("pricing_per_1m_input" in f.message for f in findings)


@pytest.mark.unit
def test_model_missing_pricing_output_fails(tmp_path: Path) -> None:
    model = _minimal_model()
    del model["pricing_per_1m_output"]
    data = _minimal_registry({"test-model": model})
    p = _write_yaml(tmp_path, data)
    findings = validate_registry_yaml(p)
    assert any("pricing_per_1m_output" in f.message for f in findings)


@pytest.mark.unit
def test_model_missing_context_window_fails(tmp_path: Path) -> None:
    model = _minimal_model()
    del model["context_window"]
    data = _minimal_registry({"test-model": model})
    p = _write_yaml(tmp_path, data)
    findings = validate_registry_yaml(p)
    assert any("context_window" in f.message for f in findings)


@pytest.mark.unit
def test_model_missing_observed_at_fails(tmp_path: Path) -> None:
    model = _minimal_model()
    del model["observed_at"]
    data = _minimal_registry({"test-model": model})
    p = _write_yaml(tmp_path, data)
    findings = validate_registry_yaml(p)
    assert any("observed_at" in f.message for f in findings)


@pytest.mark.unit
def test_cloud_provider_missing_api_key_env_fails(tmp_path: Path) -> None:
    model = _minimal_model(
        model_id="claude-test",
        provider="anthropic",
        cost_basis="cloud_api_cost",
        pricing_in="3.00",
        pricing_out="15.00",
    )
    # No requires_api_key_env set
    data = _minimal_registry({"claude-test": model})
    p = _write_yaml(tmp_path, data)
    findings = validate_registry_yaml(p)
    assert any("requires_api_key_env" in f.message for f in findings)


@pytest.mark.unit
def test_cloud_provider_with_api_key_env_passes(tmp_path: Path) -> None:
    model = _minimal_model(
        model_id="claude-test",
        provider="anthropic",
        cost_basis="cloud_api_cost",
        pricing_in="3.00",
        pricing_out="15.00",
    )
    model["requires_api_key_env"] = "ANTHROPIC_API_KEY"
    data = _minimal_registry({"claude-test": model})
    p = _write_yaml(tmp_path, data)
    findings = validate_registry_yaml(p)
    assert findings == []


@pytest.mark.unit
def test_duplicate_endpoint_env_across_local_models_fails(tmp_path: Path) -> None:
    """Two local models sharing the same endpoint_env is an orphaned ref."""
    m1 = _minimal_model(model_id="model-a", endpoint_env="LLM_CODER_URL")
    m2 = _minimal_model(model_id="model-b", endpoint_env="LLM_CODER_URL")
    data = _minimal_registry({"model-a": m1, "model-b": m2})
    p = _write_yaml(tmp_path, data)
    findings = validate_registry_yaml(p)
    assert any(
        "LLM_CODER_URL" in f.message and "duplicate" in f.message.lower()
        for f in findings
    )


@pytest.mark.unit
def test_non_positive_context_window_fails(tmp_path: Path) -> None:
    model = _minimal_model(context_window=0)
    data = _minimal_registry({"test-model": model})
    p = _write_yaml(tmp_path, data)
    findings = validate_registry_yaml(p)
    assert any("context_window" in f.message for f in findings)


@pytest.mark.unit
def test_non_yaml_file_fails(tmp_path: Path) -> None:
    p = tmp_path / "model_registry_v1.yaml"
    p.write_text(": invalid: yaml: [unclosed", encoding="utf-8")
    findings = validate_registry_yaml(p)
    assert any(
        "parse" in f.message.lower() or "yaml" in f.message.lower() for f in findings
    )


@pytest.mark.unit
def test_missing_file_fails(tmp_path: Path) -> None:
    p = tmp_path / "nonexistent.yaml"
    findings = validate_registry_yaml(p)
    assert any(
        "not found" in f.message.lower() or "exist" in f.message.lower()
        for f in findings
    )


@pytest.mark.unit
def test_finding_has_model_key_and_rule(tmp_path: Path) -> None:
    model = _minimal_model()
    del model["cost_basis"]
    data = _minimal_registry({"test-model": model})
    p = _write_yaml(tmp_path, data)
    findings = validate_registry_yaml(p)
    assert len(findings) >= 1
    f = findings[0]
    assert isinstance(f, Finding)
    assert f.model_key
    assert f.rule
    assert f.message


@pytest.mark.unit
def test_main_exits_nonzero_on_violation(tmp_path: Path) -> None:
    from omnibase_infra.validators.llm_registry_validator import main

    model = _minimal_model()
    del model["cost_basis"]
    data = _minimal_registry({"test-model": model})
    p = _write_yaml(tmp_path, data)
    assert main([str(p)]) != 0


@pytest.mark.unit
def test_main_exits_zero_on_clean_file(tmp_path: Path) -> None:
    from omnibase_infra.validators.llm_registry_validator import main

    p = _write_yaml(tmp_path, _minimal_registry())
    assert main([str(p)]) == 0
