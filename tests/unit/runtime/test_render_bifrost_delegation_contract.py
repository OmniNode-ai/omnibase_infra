# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for rendering Bifrost delegation solely from a typed overlay."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.render_bifrost_delegation_contract import (
    render_bifrost_delegation_contract,
)

_ROOT = Path(__file__).resolve().parents[3]
_OVERLAY = _ROOT / "docker" / "lane-overlays" / "dev.bifrost.yaml"
_ENDPOINT = "http://192.168.86.201:8000/v1/chat/completions"


@pytest.mark.unit
def test_renderer_has_only_the_overlay_contract_arguments() -> None:
    assert tuple(inspect.signature(render_bifrost_delegation_contract).parameters) == (
        "source_path",
        "overlay_path",
        "target_path",
        "environ",
        "verify_endpoints",
        "endpoint_probe",
    )


def _write_base_contract(path: Path, *, coder_model: str | None = "qwen3.8") -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "backends": [
                    {
                        "backend_id": "local-coder",
                        "model_name": coder_model,
                        "endpoint_url_env": "LLM_CODER_URL",
                        "required": True,
                    },
                    {
                        "backend_id": "local-heavy-reasoning",
                        "model_name": "qwen3.8",
                        "endpoint_url_env": "BIFROST_LOCAL_REASONER_ENDPOINT_URL",
                        "required": True,
                    },
                    {
                        "backend_id": "local-reasoner",
                        "model_name": "retired",
                        "endpoint_url_env": "BIFROST_LOCAL_REASONER_ENDPOINT_URL",
                    },
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


@pytest.mark.unit
def test_typed_overlay_wins_over_poisoned_model_and_endpoint_environment(
    tmp_path: Path,
) -> None:
    source = tmp_path / "base.yaml"
    target = tmp_path / "rendered.yaml"
    _write_base_contract(source)

    rendered = render_bifrost_delegation_contract(
        source_path=source,
        overlay_path=_OVERLAY,
        target_path=target,
        environ={
            "LLM_CODER_URL": "http://cloud.invalid/v1/chat/completions",
            "LLM_CODER_MODEL_NAME": "poisoned",
            "BIFROST_LOCAL_CODER_ENDPOINT_URL": "http://192.168.86.201:8001/v1/chat/completions",
            "BIFROST_LOCAL_REASONER_ENDPOINT_URL": "http://cloud.invalid/v1/chat/completions",
        },
    )

    assert rendered == target
    contract = yaml.safe_load(target.read_text(encoding="utf-8"))
    by_id = {backend["backend_id"]: backend for backend in contract["backends"]}
    for backend_id in ("local-coder", "local-heavy-reasoning"):
        assert by_id[backend_id]["endpoint_url"] == _ENDPOINT
        assert by_id[backend_id]["model_name"] == "qwen3.8"
        assert by_id[backend_id]["max_tokens"] == 65_536
        assert by_id[backend_id]["timeout_ms"] == 300_000
    assert all("endpoint_url_env" not in backend for backend in by_id.values())
    assert all("required" not in backend for backend in by_id.values())


@pytest.mark.unit
def test_missing_or_malformed_overlay_fails_before_dispatch(tmp_path: Path) -> None:
    source = tmp_path / "base.yaml"
    invalid_overlay = tmp_path / "invalid-overlay.yaml"
    target = tmp_path / "rendered.yaml"
    _write_base_contract(source)
    invalid_overlay.write_text(
        "schema_version: bifrost_lane_overlay.v2\nlane: dev\nbackends: []\n",
        encoding="utf-8",
    )

    with pytest.raises(ProtocolConfigurationError, match="overlay is invalid"):
        render_bifrost_delegation_contract(
            source_path=source,
            overlay_path=invalid_overlay,
            target_path=target,
        )
    assert not target.exists()

    with pytest.raises(ProtocolConfigurationError, match="overlay not found"):
        render_bifrost_delegation_contract(
            source_path=source,
            overlay_path=tmp_path / "missing.yaml",
            target_path=target,
        )


@pytest.mark.unit
def test_base_model_mismatch_fails_instead_of_dropping_served_id(
    tmp_path: Path,
) -> None:
    source = tmp_path / "base.yaml"
    _write_base_contract(source, coder_model="qwen3.8-27b")

    with pytest.raises(ProtocolConfigurationError, match="does not match overlay"):
        render_bifrost_delegation_contract(
            source_path=source,
            overlay_path=_OVERLAY,
            target_path=tmp_path / "rendered.yaml",
        )


@pytest.mark.unit
def test_unbound_base_model_is_materialized_from_the_typed_served_id(
    tmp_path: Path,
) -> None:
    source = tmp_path / "base.yaml"
    target = tmp_path / "rendered.yaml"
    _write_base_contract(source, coder_model=None)

    render_bifrost_delegation_contract(
        source_path=source,
        overlay_path=_OVERLAY,
        target_path=target,
    )

    contract = yaml.safe_load(target.read_text(encoding="utf-8"))
    by_id = {backend["backend_id"]: backend for backend in contract["backends"]}
    assert by_id["local-coder"]["model_name"] == "qwen3.8"


@pytest.mark.unit
def test_endpoint_probe_requires_advertised_served_id(tmp_path: Path) -> None:
    source = tmp_path / "base.yaml"
    _write_base_contract(source)

    def rejected_probe(
        endpoint_url: str, model_name: str, timeout: float
    ) -> str | None:
        assert endpoint_url == _ENDPOINT
        assert model_name == "qwen3.8"
        assert timeout > 0
        return "model endpoint did not advertise qwen3.8"

    with pytest.raises(ProtocolConfigurationError, match="failed verification"):
        render_bifrost_delegation_contract(
            source_path=source,
            overlay_path=_OVERLAY,
            target_path=tmp_path / "rendered.yaml",
            verify_endpoints=True,
            endpoint_probe=rejected_probe,
        )
