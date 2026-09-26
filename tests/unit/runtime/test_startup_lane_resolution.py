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


def test_a_declared_lane_resolves_with_its_provenance(tmp_path: Path) -> None:
    home = _home(tmp_path)
    path = _write_document(home, _body())

    resolution = resolve_runtime_lane_declaration(environ=_env(), home=home)

    assert resolution.declaration.lane_id == _LANE
    assert resolution.declaration.roles == (EnumRuntimeLaneRole.LAB,)
    assert resolution.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert resolution.location == str(path)
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


def test_the_kernel_resolves_the_lane_before_it_discovers_contracts() -> None:
    """The refusal must come before discovery, or it is a discovery error again."""
    kernel = (
        Path(runtime_lane_identity.__file__).resolve().parents[1] / "service_kernel.py"
    ).read_text(encoding="utf-8")
    resolve_at = kernel.index(
        "establish_runtime_lane(resolve_runtime_lane_declaration())"
    )
    discover_at = kernel.index("manifest = discover_contracts()")
    profile_at = kernel.index("kernel_profile = load_runtime_profile()")
    assert profile_at < resolve_at < discover_at
