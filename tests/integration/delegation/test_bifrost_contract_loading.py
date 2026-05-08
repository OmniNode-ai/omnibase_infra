# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test: routing reducer reads on-disk bifrost contract (OMN-10657).

Exercises the bifrost contract loading path against the real
configs/bifrost_delegation.yaml shipped in the repo. Unit tests already
cover individual model-selection helpers with synthetic bifrost YAMLs; this
module proves the reducer wires to the source bifrost contract file at its
installed location and that endpoint resolution falls back to the source
contract when the deployed contract is absent.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

import omnibase_infra.nodes.node_delegation_routing_reducer.handlers.handler_delegation_routing as _handler_mod

pytestmark = [pytest.mark.integration]

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_BIFROST_PATH = (
    REPO_ROOT / "src" / "omnibase_infra" / "configs" / "bifrost_delegation.yaml"
)


@pytest.fixture(autouse=True)
def reset_singletons() -> Iterator[None]:
    _handler_mod._config = None
    _handler_mod._load_bifrost_endpoints.cache_clear()
    yield
    _handler_mod._config = None
    _handler_mod._load_bifrost_endpoints.cache_clear()


def test_source_bifrost_contract_exists_at_canonical_path() -> None:
    """The source bifrost contract must exist on disk for development wiring."""
    assert SOURCE_BIFROST_PATH.exists(), (
        f"Source bifrost contract missing at {SOURCE_BIFROST_PATH}"
    )


def test_source_bifrost_contract_parses_to_expected_shape() -> None:
    """Loaded bifrost contract must declare expected backend IDs."""
    raw = yaml.safe_load(SOURCE_BIFROST_PATH.read_text())
    assert isinstance(raw, dict)
    backends = raw.get("backends")
    assert isinstance(backends, list)
    backend_ids = {b.get("backend_id") for b in backends if isinstance(b, dict)}
    for required_id in (
        "local-qwen-coder-30b",
        "local-deepseek-r1-14b",
        "cloud-sonnet",
        "cloud-haiku",
        "cloud-glm",
        "cloud-gemini-flash",
        "cli-claude",
        "cli-opencode",
    ):
        assert required_id in backend_ids, f"Missing backend_id: {required_id}"


def test_bifrost_endpoints_load_from_source_contract_when_deployed_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When deployed contract is absent, handler falls back to source bifrost contract."""
    # Write a sentinel source contract with a known populated backend.
    sentinel_source = tmp_path / "sentinel_source.yaml"
    sentinel_source.write_text(
        "config_version: '1.1.0'\n"
        "schema_version: bifrost_delegation.v1\n"
        "backends:\n"
        "  - backend_id: sentinel-fallback-backend\n"
        '    endpoint_url: "http://sentinel-fallback:8000"\n'
        "    model_name: sentinel-model\n"
        "    tier: local\n"
        "    timeout_ms: 30000\n"
        "    capabilities: []\n"
    )
    # Absent deployed contract triggers fallback to the sentinel source.
    # No BIFROST_CONTRACT_PATH env var, and _DEFAULT_BIFROST_CONTRACT_PATH points nowhere.
    monkeypatch.delenv("BIFROST_CONTRACT_PATH", raising=False)
    monkeypatch.setattr(
        _handler_mod, "_DEFAULT_BIFROST_CONTRACT_PATH", tmp_path / "absent.yaml"
    )
    monkeypatch.setattr(_handler_mod, "_SOURCE_BIFROST_CONTRACT_PATH", sentinel_source)

    backends = _handler_mod._load_bifrost_endpoints()
    assert isinstance(backends, dict)
    assert "sentinel-fallback-backend" in backends, (
        "Fallback to source bifrost contract did not load sentinel backend"
    )
    assert (
        backends["sentinel-fallback-backend"].endpoint_url
        == "http://sentinel-fallback:8000"
    )


def test_bifrost_endpoints_load_from_env_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """BIFROST_CONTRACT_PATH env var redirects loading to the chosen file."""
    # Write a sentinel override contract with a known populated backend.
    sentinel_override = tmp_path / "sentinel_override.yaml"
    sentinel_override.write_text(
        "config_version: '1.1.0'\n"
        "schema_version: bifrost_delegation.v1\n"
        "backends:\n"
        "  - backend_id: sentinel-override-backend\n"
        '    endpoint_url: "http://sentinel-override:9000"\n'
        "    model_name: sentinel-override-model\n"
        "    tier: cloud\n"
        "    timeout_ms: 10000\n"
        "    capabilities: []\n"
    )
    monkeypatch.setenv("BIFROST_CONTRACT_PATH", str(sentinel_override))

    backends = _handler_mod._load_bifrost_endpoints()
    assert isinstance(backends, dict)
    assert "sentinel-override-backend" in backends, (
        "BIFROST_CONTRACT_PATH env override did not load sentinel backend"
    )
    assert (
        backends["sentinel-override-backend"].endpoint_url
        == "http://sentinel-override:9000"
    )


def test_bifrost_contract_schema_version_is_correct() -> None:
    """Source bifrost contract must declare the expected schema version."""
    raw = yaml.safe_load(SOURCE_BIFROST_PATH.read_text())
    assert isinstance(raw, dict)
    assert raw.get("schema_version") == "bifrost_delegation.v1"
    assert raw.get("config_version") == "1.1.0"
