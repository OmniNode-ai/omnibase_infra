# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Production composition-root wiring for typed database contracts (OMN-15418)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from omnibase_infra.runtime.service_kernel import _load_runtime_database_topology

pytestmark = pytest.mark.unit

SERVICE_KERNEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "runtime"
    / "service_kernel.py"
)


def _keyword_for_call(call_name: str, keyword: str) -> ast.keyword | None:
    tree = ast.parse(SERVICE_KERNEL_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == call_name:
            return next((item for item in node.keywords if item.arg == keyword), None)
    return None


def test_event_namespace_never_selects_database_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ONEX_DATABASE_TOPOLOGY_PROFILE", raising=False)
    monkeypatch.setenv("ONEX_ENVIRONMENT", "test-env")
    monkeypatch.setenv("KAFKA_ENVIRONMENT", "test-env")

    assert _load_runtime_database_topology() is None


def test_explicit_database_topology_profile_loads_checked_in_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONEX_DATABASE_TOPOLOGY_PROFILE", "test")
    monkeypatch.setenv("ONEX_ENVIRONMENT", "test-env")

    topology = _load_runtime_database_topology()

    assert topology is not None
    assert topology.databases["application"].physical_name == "omnidash_analytics"


def test_unknown_database_topology_profile_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONEX_DATABASE_TOPOLOGY_PROFILE", "test-env")

    with pytest.raises(ValueError, match="Unsupported database topology profile"):
        _load_runtime_database_topology()


def test_cold_boot_wiring_receives_resolved_topology() -> None:
    keyword = _keyword_for_call("wire_from_manifest", "topology")

    assert keyword is not None
    assert isinstance(keyword.value, ast.Name)
    assert keyword.value.id == "deployment_topology"


def test_dynamic_runtime_receives_the_same_resolved_topology() -> None:
    keyword = _keyword_for_call("RuntimeHostProcess", "deployment_topology")

    assert keyword is not None
    assert isinstance(keyword.value, ast.Name)
    assert keyword.value.id == "deployment_topology"


def test_db_io_boot_requires_explicit_profile() -> None:
    source = SERVICE_KERNEL_PATH.read_text(encoding="utf-8")

    assert "if db_io_contracts and deployment_topology is None:" in source
    assert "ONEX_DATABASE_TOPOLOGY_PROFILE is not set" in source
