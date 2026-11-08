# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-to-handler and command-line integration for the dev-lane code-ref pin (OMN-23039).

Loads node_dev_lane_code_ref_pin_effect's contract from disk, resolves each routed
operation to its declared handler class and method, then drives the node's command line
as a subprocess against a temporary agent state directory: register a pin, read it back
through ``followed_branch`` the way ``self_update`` does, refuse a bad record, release it.
No Docker daemon, broker or deploy agent is touched.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.handlers.handler_dev_lane_code_ref_pin import (
    ENV_STATE_DIR,
    PIN_FILE_NAME,
    HandlerDevLaneCodeRefPin,
    followed_branch,
)
from omnibase_infra.models.model_dev_lane_code_ref_pin import ModelDevLaneCodeRefPin
from omnibase_infra.models.model_dev_lane_code_ref_pin_request import (
    ModelDevLaneCodeRefPinRequest,
)

pytestmark = pytest.mark.integration

_NODE_DIR = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_dev_lane_code_ref_pin_effect"
)
_RUNTIME = "omnibase_infra.nodes.node_dev_lane_code_ref_pin_effect.runtime_code_ref_pin"
_HEAD = "58b95d2b3314fe801b22a0b70a04102d3e4fec29"
_BRANCH = "omn-22930-dev-effects-fold"


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", _RUNTIME, *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _register(state: Path, **override: str) -> subprocess.CompletedProcess[str]:
    values = {
        "--branch": _BRANCH,
        "--head": _HEAD,
        "--pr": "5772",
        "--ticket": "OMN-23039",
        "--registered-by": "oneruntime-t8-2340",
        "--expires-in-hours": "6",
    }
    values.update(override)
    flat = [part for pair in values.items() for part in pair]
    return _cli("register", "--state-dir", str(state), *flat)


def test_contract_routes_each_operation_to_a_handler_method() -> None:
    contract = yaml.safe_load((_NODE_DIR / "contract.yaml").read_text(encoding="utf-8"))
    routed = contract["handler_routing"]["handlers"]
    assert {entry["operation"] for entry in routed} == {"register", "active", "release"}
    for entry in routed:
        module = importlib.import_module(entry["handler"]["module"])
        handler = getattr(module, entry["handler"]["name"])
        assert callable(getattr(handler, entry["operation"]))


def test_registered_pin_is_what_self_update_follows(tmp_path: Path) -> None:
    result = _register(tmp_path)
    assert result.returncode == 0, result.stderr
    assert f"branch={_BRANCH}" in result.stdout
    record = json.loads((tmp_path / PIN_FILE_NAME).read_text(encoding="utf-8"))
    assert record["head"] == _HEAD
    assert (
        followed_branch("dev", {ENV_STATE_DIR: str(tmp_path)}, "idle_heartbeat")
        == _BRANCH
    )
    assert _cli("show", "--state-dir", str(tmp_path)).returncode == 0


def test_released_pin_returns_self_update_to_the_tracking_ref(tmp_path: Path) -> None:
    assert _register(tmp_path).returncode == 0
    assert _cli("release", "--state-dir", str(tmp_path)).returncode == 0
    assert not (tmp_path / PIN_FILE_NAME).exists()
    assert (
        followed_branch("dev", {ENV_STATE_DIR: str(tmp_path)}, "idle_heartbeat")
        == "dev"
    )
    assert _cli("show", "--state-dir", str(tmp_path)).returncode == 1


def test_expired_pin_returns_self_update_to_the_tracking_ref(tmp_path: Path) -> None:
    assert _register(tmp_path).returncode == 0
    path = tmp_path / PIN_FILE_NAME
    record = json.loads(path.read_text(encoding="utf-8"))
    record["expires_at"] = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    path.write_text(json.dumps(record), encoding="utf-8")
    assert (
        followed_branch("dev", {ENV_STATE_DIR: str(tmp_path)}, "idle_heartbeat")
        == "dev"
    )


@pytest.mark.parametrize(
    "override",
    [
        {"--branch": "origin/dev"},
        {"--branch": "main"},
        {"--head": "58b95d2b"},
        {"--ticket": "23039"},
        {"--expires-in-hours": "200"},
    ],
)
def test_bad_record_is_refused_and_writes_nothing(
    tmp_path: Path, override: dict[str, str]
) -> None:
    result = _register(tmp_path, **override)
    assert result.returncode == 2
    assert "code-ref-pin refused" in result.stderr
    assert not (tmp_path / PIN_FILE_NAME).exists()


# -- the single typed entry point the runtime dispatches through -----------------


def _pin() -> ModelDevLaneCodeRefPin:
    now = datetime.now(UTC)
    return ModelDevLaneCodeRefPin(
        branch=_BRANCH,
        head=_HEAD,
        pr=5772,
        ticket="OMN-23039",
        registered_by="oneruntime-t8-2340",
        registered_at=now,
        expires_at=now + timedelta(hours=6),
    )


def test_handle_registers_reads_and_releases_through_one_entry_point(
    tmp_path: Path,
) -> None:
    handler = HandlerDevLaneCodeRefPin()
    pin = _pin()
    registered = handler.handle(
        ModelDevLaneCodeRefPinRequest(operation="register", state_dir=tmp_path, pin=pin)
    )
    assert registered.operation == "register"
    assert registered.path == tmp_path / PIN_FILE_NAME
    assert registered.path.exists()
    live = handler.handle(
        ModelDevLaneCodeRefPinRequest(
            operation="active", state_dir=tmp_path, now=datetime.now(UTC)
        )
    )
    assert live.pin == pin
    released = handler.handle(
        ModelDevLaneCodeRefPinRequest(operation="release", state_dir=tmp_path)
    )
    assert released.pin is None
    assert not (tmp_path / PIN_FILE_NAME).exists()
    after = handler.handle(
        ModelDevLaneCodeRefPinRequest(
            operation="active", state_dir=tmp_path, now=datetime.now(UTC)
        )
    )
    assert after.pin is None


def test_handle_reports_no_live_pin_once_it_has_expired(tmp_path: Path) -> None:
    handler = HandlerDevLaneCodeRefPin()
    pin = _pin()
    handler.handle(
        ModelDevLaneCodeRefPinRequest(operation="register", state_dir=tmp_path, pin=pin)
    )
    expired = handler.handle(
        ModelDevLaneCodeRefPinRequest(
            operation="active",
            state_dir=tmp_path,
            now=pin.expires_at + timedelta(seconds=1),
        )
    )
    assert expired.pin is None


@pytest.mark.parametrize(
    "fields",
    [
        {"operation": "register"},
        {"operation": "active"},
        {"operation": "active", "now": datetime(2026, 1, 1)},
        {"operation": "restart"},
    ],
)
def test_handle_refuses_a_request_missing_what_its_operation_needs(
    tmp_path: Path, fields: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        ModelDevLaneCodeRefPinRequest.model_validate({"state_dir": tmp_path, **fields})
    assert not (tmp_path / PIN_FILE_NAME).exists()
