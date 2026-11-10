# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Hermetic coverage of the publisher's optional OMN-23613 lane tenant key."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import yaml
from pydantic import ValidationError

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "trigger_rebuild_on_merge.py"
)

pytestmark = pytest.mark.unit


def _import_trigger_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "trigger_rebuild_on_merge_tenant_key", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_declared_tenant_loads_without_changing_transport(tmp_path: Path) -> None:
    """A declared tenant key loads and leaves the publisher's routing alone."""
    lane: dict[str, object] = {
        "broker": "example.invalid:19092",
        "security_protocol": "PLAINTEXT",
        "tenant_id": "820272f9-4aaf-5add-a2df-0af942852ab2",
    }
    overlay_path = tmp_path / "ci_bus_lanes.yaml"
    overlay_path.write_text(
        yaml.safe_dump({"default": "inmemory", "lanes": {"dev": lane}}),
        encoding="utf-8",
    )
    mod = _import_trigger_module()

    model = mod.load_ci_bus_overlay(overlay_path)

    parsed = model.lanes["dev"]
    assert parsed.tenant_id == "820272f9-4aaf-5add-a2df-0af942852ab2"
    assert (
        mod.resolve_ci_bus_broker(overlay=model, lane="dev", injected_broker="")
        == "example.invalid:19092"
    )


def test_tenant_declaration_remains_optional() -> None:
    mod = _import_trigger_module()

    assert mod.ModelCiBusLane.model_validate({"broker": "inmemory"}).tenant_id is None


def test_an_unknown_lane_key_is_still_refused() -> None:
    mod = _import_trigger_module()

    with pytest.raises(ValidationError, match="tenent_id"):
        mod.ModelCiBusLane.model_validate({"broker": "inmemory", "tenent_id": "x"})
