# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A runtime resolves its lane from the deployment's overlay, or refuses to start.

OMN-19747, plan task LO2 of knowledge-base-internal
``beta/plans/2026-09-26-runtime-lane-overlays-plan.md``; RULING
2026-09-26T14:31:29Z lane=orchestrator-83. Every refusal names what is
missing. None of them is a discovery error or a DEGRADED dimension.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from omnibase_core.enums.enum_runtime_lane_role import EnumRuntimeLaneRole
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.health import runtime_lane_identity
from omnibase_infra.runtime.health.runtime_lane_identity import (
    ENV_RUNTIME_ENVIRONMENT,
    ENV_RUNTIME_LANE,
    resolve_runtime_lane,
    resolve_runtime_lane_declaration,
)

pytestmark = pytest.mark.unit

# A lane id and environment no deployment of ours has ever used: nothing in
# any package may need to know them.
_ENV = "acme-prod"
_LANE = "customer-edge-7"


def _home(tmp_path: Path, *, local_home: bool = True) -> Path:
    home = tmp_path / "home"
    (home / ".onex").mkdir(parents=True)
    if local_home:
        (home / ".onex" / "config.yaml").write_text("config_source: local-home\n")
    return home


def _write_document(
    home: Path,
    body: dict[str, object] | str,
    *,
    environment: str = _ENV,
    lane: str = _LANE,
    mode: int = 0o600,
) -> Path:
    path = home / ".omninode" / "config" / environment / lane / "runtime.lane.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body if isinstance(body, str) else json.dumps(body))
    path.chmod(mode)
    return path


def _body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": "runtime_lane.v1",
        "lane_id": _LANE,
        "roles": ["lab"],
        "description": "A customer's edge deployment",
    }
    body.update(overrides)
    return body


def _env(**overrides: str) -> dict[str, str]:
    env = {ENV_RUNTIME_ENVIRONMENT: _ENV, ENV_RUNTIME_LANE: _LANE}
    env.update(overrides)
    return env


def test_a_declared_lane_resolves_with_its_provenance(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    home = _home(tmp_path)
    path = _write_document(home, _body())

    resolution = resolve_runtime_lane_declaration(environ=_env(), home=home)

    assert resolution.declaration.lane_id == _LANE
    assert resolution.declaration.roles == (EnumRuntimeLaneRole.LAB,)
    assert resolution.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert resolution.location == str(path)
    import logging

    with caplog.at_level(logging.INFO):
        runtime_lane_identity.establish_runtime_lane(resolution)
    assert resolution.log_line() in caplog.text
    runtime_lane_identity.clear_established_runtime_lane()
    line = resolution.log_line()
    for part in (
        f"lane={_LANE}",
        "roles=lab",
        "source=local-home",
        resolution.sha256,
    ):
        assert part in line


def test_an_unset_environment_refuses(tmp_path: Path) -> None:
    home = _home(tmp_path)
    env = _env()
    del env[ENV_RUNTIME_ENVIRONMENT]
    with pytest.raises(ProtocolConfigurationError, match=ENV_RUNTIME_ENVIRONMENT):
        resolve_runtime_lane_declaration(environ=env, home=home)


@pytest.mark.parametrize("value", [None, "", "   "])
def test_an_unset_lane_refuses_naming_the_variable(
    tmp_path: Path, value: str | None
) -> None:
    home = _home(tmp_path)
    env = _env()
    if value is None:
        del env[ENV_RUNTIME_LANE]
    else:
        env[ENV_RUNTIME_LANE] = value
    with pytest.raises(
        ProtocolConfigurationError, match=f"{ENV_RUNTIME_LANE} is not set"
    ):
        resolve_runtime_lane_declaration(environ=env, home=home)


def test_a_malformed_lane_refuses_naming_the_value(tmp_path: Path) -> None:
    home = _home(tmp_path)
    with pytest.raises(ProtocolConfigurationError, match="'Edge_7' is not a lane id"):
        resolve_runtime_lane_declaration(
            environ=_env(**{ENV_RUNTIME_LANE: "Edge_7"}), home=home
        )


def test_no_overlay_source_refuses_naming_both(tmp_path: Path) -> None:
    home = _home(tmp_path, local_home=False)
    with pytest.raises(ProtocolConfigurationError) as info:
        resolve_runtime_lane_declaration(environ=_env(), home=home)
    message = str(info.value)
    assert "no config overlay source is configured" in message
    assert "INFISICAL_ADDR" in message
    assert "config_source: local-home" in message


def test_both_overlay_sources_refuse(tmp_path: Path) -> None:
    home = _home(tmp_path)
    with pytest.raises(ProtocolConfigurationError, match="both config overlay sources"):
        resolve_runtime_lane_declaration(
            environ=_env(INFISICAL_ADDR="http://store:8080"), home=home
        )


def test_the_store_source_is_refused_not_fallen_back_from(tmp_path: Path) -> None:
    home = _home(tmp_path, local_home=False)
    _write_document(home, _body())  # present, and still not read
    with pytest.raises(ProtocolConfigurationError, match="store source is the"):
        resolve_runtime_lane_declaration(
            environ=_env(INFISICAL_ADDR="http://store:8080"), home=home
        )


def test_an_undeclared_lane_refuses_naming_lane_key_and_scope(tmp_path: Path) -> None:
    home = _home(tmp_path)
    _write_document(home, _body())  # declares customer-edge-7 only
    with pytest.raises(ProtocolConfigurationError) as info:
        resolve_runtime_lane_declaration(
            environ=_env(**{ENV_RUNTIME_LANE: "customer-edge-8"}), home=home
        )
    message = str(info.value)
    assert "runtime lane 'customer-edge-8' is not declared" in message
    assert "runtime.lane" in message
    assert re.search(r"acme-prod/customer-edge-8/runtime\.lane\.json", message)


def test_an_invalid_document_refuses_naming_the_field(tmp_path: Path) -> None:
    home = _home(tmp_path)
    _write_document(home, _body(roles=["labb"]))
    with pytest.raises(
        ProtocolConfigurationError, match=r"roles: .*'labb' is not a runtime lane role"
    ):
        resolve_runtime_lane_declaration(environ=_env(), home=home)


def test_a_document_for_another_lane_refuses_naming_both(tmp_path: Path) -> None:
    home = _home(tmp_path)
    _write_document(home, _body(lane_id="customer-edge-9"))
    with pytest.raises(ProtocolConfigurationError) as info:
        resolve_runtime_lane_declaration(environ=_env(), home=home)
    assert "'customer-edge-7'" in str(info.value)
    assert "'customer-edge-9'" in str(info.value)


def test_a_document_readable_by_others_refuses(tmp_path: Path) -> None:
    home = _home(tmp_path)
    _write_document(home, _body(), mode=0o644)
    with pytest.raises(ProtocolConfigurationError, match="must be 0600"):
        resolve_runtime_lane_declaration(environ=_env(), home=home)


def test_a_document_that_is_not_json_refuses(tmp_path: Path) -> None:
    home = _home(tmp_path)
    _write_document(home, "{not json")
    with pytest.raises(ProtocolConfigurationError, match="not UTF-8 JSON"):
        resolve_runtime_lane_declaration(environ=_env(), home=home)


def test_only_a_lab_lane_on_the_main_profile_keys_health(tmp_path: Path) -> None:
    home = _home(tmp_path)
    _write_document(home, _body(roles=["lab"]))
    lab = resolve_runtime_lane_declaration(environ=_env(), home=home).declaration
    _write_document(home, _body(roles=[]))
    dev = resolve_runtime_lane_declaration(environ=_env(), home=home).declaration

    assert resolve_runtime_lane(lab) == _LANE
    assert resolve_runtime_lane(lab, runtime_profile="main") == _LANE
    assert resolve_runtime_lane(lab, runtime_profile="effects") is None
    assert resolve_runtime_lane(dev) is None


def test_health_carries_no_lane_before_one_is_established() -> None:
    runtime_lane_identity.clear_established_runtime_lane()
    assert resolve_runtime_lane() is None


@pytest.mark.parametrize(
    "case",
    [
        "unset",
        "malformed",
        "no-source",
        "both-sources",
        "undeclared",
        "invalid",
        "mismatch",
    ],
)
async def test_the_kernel_refuses_before_discovery(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from unittest.mock import patch

    from omnibase_infra.runtime.service_kernel import bootstrap

    home = _home(tmp_path, local_home=case != "no-source")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv(ENV_RUNTIME_ENVIRONMENT, _ENV)
    monkeypatch.setenv(ENV_RUNTIME_LANE, _LANE)
    monkeypatch.delenv("INFISICAL_ADDR", raising=False)
    monkeypatch.setenv("RUNTIME_PROFILE", "main")
    expected = {
        "unset": ENV_RUNTIME_LANE,
        "malformed": "'Edge_7'",
        "no-source": "no config overlay source",
        "both-sources": "both config overlay sources",
        "undeclared": "runtime.lane",
        "invalid": "labb",
        "mismatch": "customer-edge-9",
    }[case]
    if case == "unset":
        monkeypatch.delenv(ENV_RUNTIME_LANE)
    elif case == "malformed":
        monkeypatch.setenv(ENV_RUNTIME_LANE, "Edge_7")
    elif case == "both-sources":
        monkeypatch.setenv("INFISICAL_ADDR", "http://store:8080")
    elif case == "invalid":
        _write_document(home, _body(roles=["labb"]))
    elif case == "mismatch":
        _write_document(home, _body(lane_id="customer-edge-9"))
    runtime_lane_identity.clear_established_runtime_lane()
    with (
        patch("omnibase_infra.runtime.auto_wiring.discover_contracts") as discover,
        patch("omnibase_infra.runtime.service_kernel.RuntimeHostProcess") as host,
    ):
        result = await bootstrap()
    assert result == 1
    assert expected in caplog.text
    assert "DEGRADED" not in caplog.text
    discover.assert_not_called()
    host.assert_not_called()
    assert runtime_lane_identity.established_runtime_lane() is None


async def test_real_startup_contract_wires_and_its_handler_resolves(
    tmp_path: Path,
) -> None:
    import importlib

    from omnibase_core.container import ModelONEXContainer
    from omnibase_infra.models.model_runtime_lane_resolution_request import (
        ModelRuntimeLaneResolutionRequest,
    )
    from omnibase_infra.runtime.auto_wiring.discovery import _parse_contract
    from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
    from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
    from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

    root = Path(runtime_lane_identity.__file__).resolve().parents[2]
    path = root / "nodes/node_runtime_lane_resolution_effect/contract.yaml"
    contract = _parse_contract(
        contract_path=path,
        entry_point_name="node_runtime_lane_resolution_effect",
        package_name="omnibase-infra",
        package_version="test",
    )
    report = await wire_from_manifest(
        manifest=ModelAutoWiringManifest(contracts=(contract,)),
        dispatch_engine=MessageDispatchEngine(),
        event_bus=None,
        environment=_ENV,
        container=ModelONEXContainer(),
        subscribe_immediately=False,
        result_appliers_by_contract={},
        materialized_explicit_dependencies={},
        topology=None,
    )
    assert report.total_failed == 0, report.model_dump()
    assert len(report.results) == 1
    # Startup handlers have no bus subscription: exercise the real handler from
    # the contract as well, so the zero-failure wiring assertion is not vacuous.
    ref = contract.handler_routing.handlers[0].handler
    cls = getattr(importlib.import_module(ref.module), ref.name)
    home = _home(tmp_path)
    _write_document(home, _body())
    result = await cls().handle(
        ModelRuntimeLaneResolutionRequest(environ=_env(), home=home)
    )
    assert result.declaration.lane_id == _LANE
    assert result.declaration.roles == (EnumRuntimeLaneRole.LAB,)


def test_an_unreadable_document_is_a_named_configuration_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path)
    path = _write_document(home, _body())

    def unreadable(self: Path) -> bytes:
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    with pytest.raises(
        ProtocolConfigurationError, match="cannot read overlay document"
    ) as info:
        resolve_runtime_lane_declaration(environ=_env(), home=home)
    assert str(path) in str(info.value)
    assert "runtime.lane" in str(info.value)
