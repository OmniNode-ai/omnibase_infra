# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Hermetic coverage of the publisher's optional OMN-20275 lane declarations."""

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
        "trigger_rebuild_on_merge_binding_fallback", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("declaration", ["binding_fallback", "secret_resolver", "both"])
def test_declared_binding_shape_loads_without_changing_transport(
    tmp_path: Path, declaration: str
) -> None:
    """Either new block, and both together, preserve the publisher's routing."""
    lane: dict[str, object] = {
        "broker": "example.invalid:19092",
        "security_protocol": "SASL_PLAINTEXT",
        "sasl_mechanism": "SCRAM-SHA-256",
    }
    if declaration in {"binding_fallback", "both"}:
        lane["binding_fallback"] = ["primary", "secondary"]
    if declaration in {"secret_resolver", "both"}:
        lane["secret_resolver"] = {
            "enable_convention_fallback": True,
            "mappings": [
                {
                    "logical_name": "KAFKA_SASL_PASSWORD",
                    "source": {
                        "source_type": "env",
                        "source_path": "CI_KAFKA_PASSWORD",
                    },
                }
            ],
        }
    overlay_path = tmp_path / "ci_bus_lanes.yaml"
    overlay_path.write_text(
        yaml.safe_dump({"default": "inmemory", "lanes": {"dev": lane}}),
        encoding="utf-8",
    )
    mod = _import_trigger_module()

    model = mod.load_ci_bus_overlay(overlay_path)

    parsed = model.lanes["dev"]
    if declaration in {"binding_fallback", "both"}:
        assert parsed.binding_fallback == ("primary", "secondary")
    if declaration in {"secret_resolver", "both"}:
        resolver = parsed.secret_resolver
        assert resolver is not None
        assert resolver.enable_convention_fallback is True
        assert resolver.mappings[0].logical_name == "KAFKA_SASL_PASSWORD"
        assert resolver.mappings[0].source.source_type == "env"
        assert resolver.mappings[0].source.source_path == "CI_KAFKA_PASSWORD"
    assert (
        mod.resolve_ci_bus_broker(overlay=model, lane="dev", injected_broker="")
        == "example.invalid:19092"
    )
    assert mod.resolve_ci_bus_security(overlay=model, lane="dev") == (
        "SASL_PLAINTEXT",
        "SCRAM-SHA-256",
    )


def test_binding_declarations_remain_optional() -> None:
    mod = _import_trigger_module()

    lane = mod.ModelCiBusLane.model_validate({"broker": "inmemory"})

    assert lane.binding_fallback == ()
    assert lane.secret_resolver is None
    resolver = mod.ModelCiBusLane.model_validate(
        {"broker": "inmemory", "secret_resolver": {}}
    ).secret_resolver
    assert resolver is not None
    assert resolver.enable_convention_fallback is False
    assert resolver.mappings == ()


@pytest.mark.parametrize(
    "declaration",
    [
        {"binding_falback": []},
        {"secret_resolver": {"enable_convention_falback": True}},
        {
            "secret_resolver": {
                "mappings": [
                    {
                        "logical_name": "KAFKA_SASL_PASSWORD",
                        "source": {
                            "source_type": "env",
                            "source_path": "CI_KAFKA_PASSWORD",
                        },
                        "logical_nam": "typo",
                    }
                ]
            }
        },
        {
            "secret_resolver": {
                "mappings": [
                    {
                        "logical_name": "KAFKA_SASL_PASSWORD",
                        "source": {
                            "source_type": "env",
                            "source_path": "CI_KAFKA_PASSWORD",
                            "path": "typo",
                        },
                    }
                ]
            }
        },
    ],
)
def test_unknown_binding_keys_are_refused(declaration: dict[str, object]) -> None:
    mod = _import_trigger_module()

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        mod.ModelCiBusLane.model_validate({"broker": "inmemory", **declaration})


@pytest.mark.parametrize(
    ("declaration", "expected_loc"),
    [
        ({"binding_fallback": "primary"}, ("binding_fallback",)),
        (
            {
                "secret_resolver": {
                    "mappings": [{"logical_name": "KAFKA_SASL_PASSWORD"}]
                }
            },
            ("secret_resolver", "mappings", 0, "source"),
        ),
        (
            {
                "secret_resolver": {
                    "mappings": [
                        {
                            "logical_name": "KAFKA_SASL_PASSWORD",
                            "source": {"source_type": "env"},
                        }
                    ]
                }
            },
            ("secret_resolver", "mappings", 0, "source", "source_path"),
        ),
    ],
)
def test_malformed_binding_shapes_are_refused(
    declaration: dict[str, object], expected_loc: tuple[str, ...]
) -> None:
    mod = _import_trigger_module()

    with pytest.raises(ValidationError) as excinfo:
        mod.ModelCiBusLane.model_validate({"broker": "inmemory", **declaration})

    locs = [tuple(err["loc"]) for err in excinfo.value.errors()]
    assert any(loc[: len(expected_loc)] == expected_loc for loc in locs), locs


def test_binding_fallback_edge_values_are_kept_verbatim() -> None:
    """The model is shape-only: empty and duplicate entries are not policed here."""
    mod = _import_trigger_module()

    empty = mod.ModelCiBusLane.model_validate(
        {"broker": "inmemory", "binding_fallback": []}
    )
    assert empty.binding_fallback == ()
    odd = mod.ModelCiBusLane.model_validate(
        {"broker": "inmemory", "binding_fallback": ["primary", "primary", ""]}
    )
    assert odd.binding_fallback == ("primary", "primary", "")


def test_secret_resolver_edge_values() -> None:
    mod = _import_trigger_module()

    explicit_none = mod.ModelCiBusLane.model_validate(
        {"broker": "inmemory", "secret_resolver": None}
    )
    assert explicit_none.secret_resolver is None
    resolver = mod.ModelCiBusLane.model_validate(
        {
            "broker": "inmemory",
            "secret_resolver": {"enable_convention_fallback": True, "mappings": []},
        }
    ).secret_resolver
    assert resolver is not None
    assert resolver.enable_convention_fallback is True
    assert resolver.mappings == ()
