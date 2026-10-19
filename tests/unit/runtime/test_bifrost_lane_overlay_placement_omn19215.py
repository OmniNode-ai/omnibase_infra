# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19215: a lane-added backend may carry a tier placement to the routing authority.

OMN-17099 lets a lane overlay ADD a backend, but routing only offers a backend
the packaged routing ladder names, so an added backend was reachable only by a
per-run pin. The routing authority (omnimarket) now reads an optional
``placement`` on a rendered backend entry and mirrors the backend into the named
tier after the rungs it backs. The renderer validates the context window and,
when the runtime pins its ladder path, the tier and rung names before passing
the placement through unchanged.

A placement may not offer the backend a larger prompt than the backend says it
holds. The routing authority also checks tier and rung names at its own load.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.models.model_bifrost_lane_backend_binding import (
    ModelBifrostLaneBackendBinding,
)
from omnibase_infra.runtime.render_bifrost_delegation_contract import (
    render_bifrost_delegation_contract,
)

# RFC 5737 documentation-range addresses, never lab hosts.
_BASE_ENDPOINT = "http://192.0.2.10:8000/v1/chat/completions"
_ADDED_ENDPOINT = "http://192.0.2.20:8000/v1/chat/completions"

_PLACEMENT = {
    "tier": "local",
    "fallback_for": ["local-coder"],
    "max_context_tokens": 32_768,
}


def _base_contract(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "backends": [
                    {
                        "backend_id": "local-coder",
                        "provider": "local",
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
                    {"task_class": "code_generation", "backend_ids": ["local-coder"]}
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
        "placement": dict(_PLACEMENT),
    }
    binding.update(overrides)
    return binding


def _render(
    tmp_path: Path,
    backends: list[dict[str, object]],
    *,
    extra_environ: dict[str, str] | None = None,
) -> dict[str, object]:
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
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
    target = tmp_path / "rendered.yaml"
    render_bifrost_delegation_contract(
        source_path=_base_contract(tmp_path / "base.yaml"),
        overlay_path=overlay,
        target_path=target,
        environ=extra_environ if extra_environ is not None else {},
        verify_endpoints=False,
    )
    rendered = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert isinstance(rendered, dict)
    return rendered


def _added(rendered: dict[str, object]) -> dict[str, object]:
    backends = rendered["backends"]
    assert isinstance(backends, list)
    return next(b for b in backends if b["backend_id"] == "local-second-host")


@pytest.mark.unit
def test_a_placement_is_passed_through_to_the_rendered_backend(tmp_path: Path) -> None:
    rendered = _render(tmp_path, [_base_binding(), _added_binding()])
    assert _added(rendered)["placement"] == _PLACEMENT


@pytest.mark.unit
def test_an_added_backend_without_a_placement_renders_no_placement_key(
    tmp_path: Path,
) -> None:
    binding = _added_binding()
    del binding["placement"]
    rendered = _render(tmp_path, [_base_binding(), binding])
    assert "placement" not in _added(rendered)


@pytest.mark.unit
def test_a_placement_larger_than_the_context_window_fails_naming_the_backend() -> None:
    with pytest.raises(ValidationError, match="local-second-host"):
        ModelBifrostLaneBackendBinding.model_validate(
            _added_binding(placement={**_PLACEMENT, "max_context_tokens": 65_536})
        )


@pytest.mark.unit
def test_a_base_backend_cannot_be_given_a_placement_by_a_lane(tmp_path: Path) -> None:
    # A base-declared rung is already in the ladder; a lane placing it again
    # would duplicate it. Only a backend the lane adds carries a placement.
    with pytest.raises(
        (ValidationError, ProtocolConfigurationError), match="local-coder"
    ):
        _render(tmp_path, [_base_binding(placement=dict(_PLACEMENT))])


@pytest.mark.unit
@pytest.mark.parametrize(
    "placement",
    [
        {"tier": "local", "fallback_for": [], "max_context_tokens": 1024},
        {"tier": "", "fallback_for": ["local-coder"], "max_context_tokens": 1024},
        {"tier": "local", "fallback_for": ["local-coder"], "max_context_tokens": 0},
        {
            "tier": "local",
            "fallback_for": ["local-coder"],
            "priority": 2,
            "max_context_tokens": 1,
        },
    ],
    ids=["no-rungs", "no-tier", "no-context", "unknown-field"],
)
def test_a_malformed_placement_is_refused(placement: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ModelBifrostLaneBackendBinding.model_validate(
            _added_binding(placement=placement)
        )


@pytest.mark.unit
def test_the_render_refuses_an_oversized_placement_with_a_typed_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(ProtocolConfigurationError, match="local-second-host"):
        _render(
            tmp_path,
            [
                _base_binding(),
                _added_binding(placement={**_PLACEMENT, "max_context_tokens": 65_536}),
            ],
        )


def _ladder(
    tmp_path: Path,
    *,
    rung_key: str = "backend_id",
    local_rungs: tuple[str, ...] = ("local-writer", "local-coder"),
) -> Path:
    path = tmp_path / "routing_tiers.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "tiers": [
                    {
                        "name": "local",
                        "models": [{rung_key: rung} for rung in local_rungs],
                    },
                    {
                        "name": "frontier_api",
                        "models": [{rung_key: "cloud-frontier"}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.unit
@pytest.mark.parametrize("rung_key", ["backend_id", "backend_ref"])
def test_a_valid_placement_renders_against_the_pinned_ladder(
    tmp_path: Path, rung_key: str
) -> None:
    rendered = _render(
        tmp_path,
        [_base_binding(), _added_binding()],
        extra_environ={
            "DELEGATION_ROUTING_TIERS_PATH": str(_ladder(tmp_path, rung_key=rung_key))
        },
    )
    assert _added(rendered)["placement"] == _PLACEMENT


@pytest.mark.unit
def test_an_unknown_placement_tier_is_refused_naming_the_backend(
    tmp_path: Path,
) -> None:
    with pytest.raises(ProtocolConfigurationError, match="local-second-host") as exc:
        _render(
            tmp_path,
            [
                _base_binding(),
                _added_binding(placement={**_PLACEMENT, "tier": "unknown-tier"}),
            ],
            extra_environ={"DELEGATION_ROUTING_TIERS_PATH": str(_ladder(tmp_path))},
        )
    assert "unknown-tier" in str(exc.value)
    assert "['frontier_api', 'local']" in str(exc.value)
    assert str(exc.value).endswith("(OMN-19215)")
    assert not (tmp_path / "rendered.yaml").exists()


@pytest.mark.unit
@pytest.mark.parametrize("rung", ["unknown-rung", "cloud-frontier"])
def test_a_fallback_rung_outside_the_tier_is_refused_naming_the_backend(
    tmp_path: Path, rung: str
) -> None:
    with pytest.raises(ProtocolConfigurationError, match="local-second-host") as exc:
        _render(
            tmp_path,
            [
                _base_binding(),
                _added_binding(placement={**_PLACEMENT, "fallback_for": [rung]}),
            ],
            extra_environ={"DELEGATION_ROUTING_TIERS_PATH": str(_ladder(tmp_path))},
        )
    assert rung in str(exc.value)
    assert "['local-coder', 'local-writer']" in str(exc.value)
    assert str(exc.value).endswith("(OMN-19215)")


@pytest.mark.unit
@pytest.mark.parametrize("rung_key", ["backend_id", "backend_ref"])
def test_a_tier_already_carrying_the_added_backend_is_refused(
    tmp_path: Path, rung_key: str
) -> None:
    ladder = _ladder(
        tmp_path,
        rung_key=rung_key,
        local_rungs=("local-coder", "local-second-host"),
    )
    with pytest.raises(ProtocolConfigurationError, match="local-second-host") as exc:
        _render(
            tmp_path,
            [_base_binding(), _added_binding()],
            extra_environ={"DELEGATION_ROUTING_TIERS_PATH": str(ladder)},
        )
    assert "already declares" in str(exc.value)
    assert str(exc.value).endswith("(OMN-19215)")


@pytest.mark.unit
def test_a_missing_ladder_is_refused_when_a_placement_exists(tmp_path: Path) -> None:
    with pytest.raises(ProtocolConfigurationError, match="local-second-host") as exc:
        _render(
            tmp_path,
            [_base_binding(), _added_binding()],
            extra_environ={
                "DELEGATION_ROUTING_TIERS_PATH": str(tmp_path / "missing.yaml")
            },
        )
    assert "missing.yaml" in str(exc.value)
    assert str(exc.value).endswith("(OMN-19215)")


@pytest.mark.unit
@pytest.mark.parametrize(
    "document",
    ["[", "- local", "{}", "tiers: {}", "tiers: null", "\xff"],
    ids=["malformed", "non-mapping", "no-tiers", "non-list", "null", "non-utf8"],
)
def test_an_invalid_ladder_is_refused_naming_the_backend(
    tmp_path: Path, document: str
) -> None:
    ladder = tmp_path / "routing_tiers.yaml"
    ladder.write_bytes(document.encode("latin-1"))
    with pytest.raises(ProtocolConfigurationError, match="local-second-host") as exc:
        _render(
            tmp_path,
            [_base_binding(), _added_binding()],
            extra_environ={"DELEGATION_ROUTING_TIERS_PATH": str(ladder)},
        )
    assert str(exc.value).endswith("(OMN-19215)")


@pytest.mark.unit
def test_an_unreadable_ladder_is_refused_naming_the_backend(tmp_path: Path) -> None:
    # A directory reliably raises an OSError even when tests run as root.
    with pytest.raises(ProtocolConfigurationError, match="local-second-host") as exc:
        _render(
            tmp_path,
            [_base_binding(), _added_binding()],
            extra_environ={"DELEGATION_ROUTING_TIERS_PATH": str(tmp_path)},
        )
    assert str(exc.value).endswith("(OMN-19215)")


@pytest.mark.unit
@pytest.mark.parametrize(
    "extra_environ", [{}, {"DELEGATION_ROUTING_TIERS_PATH": " \t"}]
)
def test_an_invalid_tier_passes_through_without_a_ladder_pin(
    tmp_path: Path, extra_environ: dict[str, str]
) -> None:
    placement = {**_PLACEMENT, "tier": "unknown-tier"}
    rendered = _render(
        tmp_path,
        [_base_binding(), _added_binding(placement=placement)],
        extra_environ=extra_environ,
    )
    assert _added(rendered)["placement"] == placement


@pytest.mark.unit
def test_a_render_without_placements_does_not_read_the_ladder(tmp_path: Path) -> None:
    rendered = _render(
        tmp_path,
        [_base_binding(), _added_binding(placement=None)],
        extra_environ={"DELEGATION_ROUTING_TIERS_PATH": str(tmp_path / "missing.yaml")},
    )
    assert "placement" not in _added(rendered)


# --- AC4: placement mode (RULING ledger:4257) --------------------------------


@pytest.mark.unit
def test_a_spread_placement_passes_its_mode_through(tmp_path: Path) -> None:
    spread = {**_PLACEMENT, "mode": "spread"}
    rendered = _render(tmp_path, [_base_binding(), _added_binding(placement=spread)])
    assert _added(rendered)["placement"] == spread


@pytest.mark.unit
def test_a_fallback_placement_renders_without_a_mode_key(tmp_path: Path) -> None:
    # The default renders the pre-mode bytes, so a routing authority that does
    # not know the field still accepts every fallback placement.
    explicit = {**_PLACEMENT, "mode": "fallback"}
    rendered = _render(tmp_path, [_base_binding(), _added_binding(placement=explicit)])
    placement = _added(rendered)["placement"]
    assert placement == _PLACEMENT
    assert isinstance(placement, dict)
    assert "mode" not in placement


@pytest.mark.unit
def test_an_unknown_placement_mode_is_refused() -> None:
    with pytest.raises(ValidationError, match="mode"):
        ModelBifrostLaneBackendBinding.model_validate(
            _added_binding(placement={**_PLACEMENT, "mode": "round_robin"})
        )
