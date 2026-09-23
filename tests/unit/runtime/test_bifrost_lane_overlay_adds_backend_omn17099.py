# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17099: a lane overlay may ADD a backend, validated as a contract.

Before this ticket the lane overlay was an authorization table in disguise: the
binding model refused any ``backend_id`` outside a hardcoded
``_AUTHORIZED_BINDINGS`` map (lab hosts, ports and served ids written into the
shipped product), the overlay demanded SET EQUALITY with that map, and the
renderer refused any backend the base contract did not already declare. So a lab
host could not be registered without a product release, and the product carried
lab values.

The replacement is schema validation. A backend the base contract does not
declare may be added when it is FULLY specified — id, endpoint, served model,
provider kind, tier, and a credential reference or an explicit ``none``. A
partial declaration fails loudly naming the backend and the missing field; it is
never dropped and never defaulted.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.models.model_bifrost_lane_overlay import (
    ModelBifrostLaneOverlay,
)
from omnibase_infra.runtime.render_bifrost_delegation_contract import (
    render_bifrost_delegation_contract,
)

_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME_SRC = _ROOT / "src" / "omnibase_infra" / "runtime"

# Fixture endpoints are documentation-range addresses (RFC 5737), deliberately
# NOT lab hosts: the point of this file is that the product accepts any
# well-formed endpoint a lane declares, not a list it ships.
_BASE_ENDPOINT = "http://192.0.2.10:8000/v1/chat/completions"
_ADDED_ENDPOINT = "http://192.0.2.20:8000/v1/chat/completions"
_ADDED_TLS_ENDPOINT = "https://inference.example.test/v1/chat/completions"


def _base_contract(path: Path, *, routed_local: tuple[str, ...] = ()) -> Path:
    """A base contract with one local backend and one cloud backend."""
    path.write_text(
        yaml.safe_dump(
            {
                "backends": [
                    {
                        "backend_id": "local-coder",
                        "provider": "local",
                        "endpoint_url_env": "BIFROST_LOCAL_CODER_ENDPOINT_URL",
                        "endpoint_url": None,
                        "model_name": "served-model",
                        "tier": "local",
                        "timeout_ms": 300_000,
                    },
                    {
                        "backend_id": "cloud-frontier",
                        "provider": "gemini",
                        "endpoint_url": "https://api.example.test/v1/chat/completions",
                        "model_name": "frontier-model",
                        "secret_ref": "llm.frontier.api_key",
                        "tier": "frontier_api",
                        "timeout_ms": 60_000,
                    },
                ],
                "routing_rules": [
                    {"task_class": "code_generation", "backend_ids": list(routed_local)}
                ],
                "default_backends": ["cloud-frontier"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _base_binding(**overrides: object) -> dict[str, object]:
    binding: dict[str, object] = {
        "backend_id": "local-coder",
        "endpoint_url": _BASE_ENDPOINT,
        "served_model_id": "served-model",
        "parameter_count": "27B",
        "context_window": 131_072,
        "max_tokens": 65_536,
        "timeout_ms": 300_000,
    }
    binding.update(overrides)
    return binding


def _added_binding(**overrides: object) -> dict[str, object]:
    binding: dict[str, object] = {
        "backend_id": "local-second-host",
        "endpoint_url": _ADDED_ENDPOINT,
        "served_model_id": "served-model",
        "parameter_count": "27B",
        "context_window": 32_768,
        "max_tokens": 16_384,
        "timeout_ms": 300_000,
        "provider": "local",
        "tier": "local",
        "credential": {"kind": "none"},
    }
    binding.update(overrides)
    return binding


def _overlay_file(path: Path, backends: list[dict[str, object]]) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "bifrost_lane_overlay.v3",
                "lane": "dev",
                "locale": "lab",
                "backends": backends,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _render(tmp_path: Path, backends: list[dict[str, object]], **kw: object) -> dict:
    target = tmp_path / "rendered.yaml"
    render_bifrost_delegation_contract(
        source_path=_base_contract(tmp_path / "base.yaml", **kw),  # type: ignore[arg-type]
        overlay_path=_overlay_file(tmp_path / "overlay.yaml", backends),
        target_path=target,
        environ={},
        verify_endpoints=False,
    )
    rendered = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert isinstance(rendered, dict)
    return rendered


def _by_id(rendered: dict) -> dict[str, dict]:
    return {backend["backend_id"]: backend for backend in rendered["backends"]}


# --- the freedom ------------------------------------------------------------


@pytest.mark.unit
def test_a_fully_specified_new_backend_is_added_to_the_rendered_contract(
    tmp_path: Path,
) -> None:
    rendered = _render(tmp_path, [_base_binding(), _added_binding()])

    added = _by_id(rendered)["local-second-host"]
    assert added == {
        "backend_id": "local-second-host",
        "provider": "local",
        "endpoint_url": _ADDED_ENDPOINT,
        "model_name": "served-model",
        "tier": "local",
        "timeout_ms": 300_000,
        "max_tokens": 16_384,
        "secret_ref": None,
        "capabilities": [],
    }
    # The base backend is still bound and the base ordering is kept: an added
    # backend is appended, so it never displaces a rung the base declared.
    assert [b["backend_id"] for b in rendered["backends"]] == [
        "local-coder",
        "cloud-frontier",
        "local-second-host",
    ]
    assert _by_id(rendered)["local-coder"]["endpoint_url"] == _BASE_ENDPOINT


@pytest.mark.unit
def test_an_added_backend_declared_not_serving_renders_the_disabled_shape(
    tmp_path: Path,
) -> None:
    rendered = _render(tmp_path, [_base_binding(), _added_binding(serving=False)])

    added = _by_id(rendered)["local-second-host"]
    assert added["endpoint_url"] is None
    assert added["model_name"] == "served-model"


@pytest.mark.unit
def test_an_added_backend_with_a_secret_ref_renders_that_reference(
    tmp_path: Path,
) -> None:
    rendered = _render(
        tmp_path,
        [
            _base_binding(),
            _added_binding(
                backend_id="cloud-added",
                endpoint_url=_ADDED_TLS_ENDPOINT,
                provider="openrouter",
                tier="cheap_cloud",
                credential={"kind": "secret_ref", "secret_ref": "llm.added.api_key"},
                capabilities=["code_generation"],
            ),
        ],
    )

    added = _by_id(rendered)["cloud-added"]
    assert added["secret_ref"] == "llm.added.api_key"
    assert added["capabilities"] == ["code_generation"]


@pytest.mark.unit
def test_the_endpoint_is_whatever_the_lane_declares_not_a_shipped_host(
    tmp_path: Path,
) -> None:
    """No host, port or served id is authorized by the product any more."""
    rendered = _render(
        tmp_path,
        [_base_binding(endpoint_url="http://198.51.100.7:9123/v1/chat/completions")],
    )

    assert (
        _by_id(rendered)["local-coder"]["endpoint_url"]
        == "http://198.51.100.7:9123/v1/chat/completions"
    )


# --- loud failure -------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("missing", ["provider", "tier", "credential"])
def test_an_added_backend_missing_a_declaration_fails_naming_it(
    tmp_path: Path, missing: str
) -> None:
    binding = _added_binding()
    del binding[missing]

    with pytest.raises(ProtocolConfigurationError) as excinfo:
        _render(tmp_path, [_base_binding(), binding])

    message = str(excinfo.value)
    assert "local-second-host" in message
    assert missing in message


@pytest.mark.unit
def test_an_added_backend_with_no_declaration_at_all_fails_rather_than_dropping(
    tmp_path: Path,
) -> None:
    binding = {
        key: value
        for key, value in _added_binding().items()
        if key not in {"provider", "tier", "credential"}
    }

    with pytest.raises(ProtocolConfigurationError) as excinfo:
        _render(tmp_path, [_base_binding(), binding])

    message = str(excinfo.value)
    assert "local-second-host" in message
    assert "not declared by the base contract" in message


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider", "gemini"),
        ("tier", "cheap_cloud"),
        ("credential", {"kind": "secret_ref", "secret_ref": "llm.other.api_key"}),
    ],
)
def test_a_base_backend_cannot_have_its_identity_rewritten_by_a_lane(
    tmp_path: Path, field: str, value: object
) -> None:
    """The base contract owns provider, tier and credential for its backends."""
    with pytest.raises(ProtocolConfigurationError) as excinfo:
        _render(tmp_path, [_base_binding(**{field: value})])

    message = str(excinfo.value)
    assert "local-coder" in message
    assert field in message


@pytest.mark.unit
def test_a_lab_lane_omitting_a_routed_local_backend_fails_naming_it(
    tmp_path: Path,
) -> None:
    """The OMN-16833 silent-degradation guard, now derived from the contract.

    The required set is every local backend the BASE contract routes to, not a
    list shipped in the product.
    """
    with pytest.raises(ProtocolConfigurationError) as excinfo:
        _render(tmp_path, [_added_binding()], routed_local=("local-coder",))

    message = str(excinfo.value)
    assert "local-coder" in message
    assert "'dev'" in message


@pytest.mark.unit
@pytest.mark.parametrize(
    ("credential", "reason"),
    [
        ({"kind": "secret_ref"}, "must name the secret_ref"),
        ({"kind": "none", "secret_ref": "llm.added.api_key"}, "must not carry"),
        ({"kind": "bearer"}, "'none' or 'secret_ref'"),
    ],
)
def test_an_inconsistent_credential_is_rejected(
    credential: object, reason: str
) -> None:
    with pytest.raises(ValidationError, match=reason):
        ModelBifrostLaneOverlay.model_validate(
            {
                "schema_version": "bifrost_lane_overlay.v3",
                "lane": "dev",
                "locale": "lab",
                "backends": [_added_binding(credential=credential)],
            }
        )


@pytest.mark.unit
def test_a_house_credential_is_never_sent_over_plaintext_http() -> None:
    with pytest.raises(ValidationError, match="must use https"):
        ModelBifrostLaneOverlay.model_validate(
            {
                "schema_version": "bifrost_lane_overlay.v3",
                "lane": "dev",
                "locale": "lab",
                "backends": [
                    _added_binding(
                        credential={
                            "kind": "secret_ref",
                            "secret_ref": "llm.added.api_key",
                        }
                    )
                ],
            }
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "endpoint_url",
    [
        "http://192.0.2.20:8000/v1",
        "http://192.0.2.20:8000/v1/chat/completions?model=x",
        "http://user@192.0.2.20:8000/v1/chat/completions",
        "http://192.0.2.20:8000/v1/chat/completions#frag",
        "ftp://192.0.2.20:8000/v1/chat/completions",
        "http:///v1/chat/completions",
    ],
)
def test_a_malformed_endpoint_is_still_rejected(endpoint_url: str) -> None:
    with pytest.raises(ValidationError, match="endpoint_url"):
        ModelBifrostLaneOverlay.model_validate(
            {
                "schema_version": "bifrost_lane_overlay.v3",
                "lane": "dev",
                "locale": "lab",
                "backends": [_added_binding(endpoint_url=endpoint_url)],
            }
        )


@pytest.mark.unit
def test_a_cloud_lane_still_cannot_add_any_backend() -> None:
    with pytest.raises(ValidationError, match="locale 'cloud'"):
        ModelBifrostLaneOverlay.model_validate(
            {
                "schema_version": "bifrost_lane_overlay.v3",
                "lane": "onex-dev",
                "locale": "cloud",
                "backends": [_added_binding()],
            }
        )


# --- the product ships no lab values -----------------------------------------


_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


@pytest.mark.unit
@pytest.mark.parametrize(
    "module",
    [
        "models/model_bifrost_lane_backend_binding.py",
        "models/model_bifrost_lane_backend_credential.py",
        "models/model_bifrost_lane_overlay.py",
        "render_bifrost_delegation_contract.py",
    ],
)
def test_the_shipped_overlay_code_carries_no_host_address(module: str) -> None:
    """Lab hosts live in the lane overlays, which are lane config, not product."""
    source = (_RUNTIME_SRC / module).read_text(encoding="utf-8")
    assert _IPV4.findall(source) == []
