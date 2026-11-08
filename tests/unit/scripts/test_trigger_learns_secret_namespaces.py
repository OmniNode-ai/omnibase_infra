# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Hermetic coverage of the publisher's model of ``secret_resolver.namespaces``.

The CI overlay declares pattern-scoped secret sources under a lane's
``secret_resolver.namespaces`` (the runtime model is
``ModelSecretNamespaceRule``). The publisher's consumer model is
``extra="forbid"``, so a lane declaring that key failed the live binding job
until the publisher learned the shape. The publisher never resolves a secret:
these tests pin that it loads the declaration and keeps its routing unchanged,
and that a typo in the block is still refused.
"""

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

_NAMESPACE_RULE: dict[str, object] = {
    "namespace": "ilo_accounts",
    "ref_pattern": "^ILO_[A-Z0-9_]+_(READ|WRITE)_(USERNAME|PASSWORD)$",
    "source_type": "infisical",
    "source_path_template": "/shared/env/{ref}",
}


def _import_trigger_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "trigger_rebuild_on_merge_secret_namespaces", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_namespaces_load_without_changing_transport(tmp_path: Path) -> None:
    """Golden: a lane declaring namespaces loads and routes as before."""
    lane: dict[str, object] = {
        "broker": "example.invalid:19092",
        "security_protocol": "SASL_PLAINTEXT",
        "sasl_mechanism": "SCRAM-SHA-256",
        "secret_resolver": {
            "enable_convention_fallback": False,
            "mappings": [],
            "namespaces": [dict(_NAMESPACE_RULE, ttl_seconds=300)],
        },
    }
    overlay_path = tmp_path / "ci_bus_lanes.yaml"
    overlay_path.write_text(
        yaml.safe_dump({"default": "inmemory", "lanes": {"dev": lane}}),
        encoding="utf-8",
    )
    mod = _import_trigger_module()

    model = mod.load_ci_bus_overlay(overlay_path)

    resolver = model.lanes["dev"].secret_resolver
    assert resolver is not None
    assert len(resolver.namespaces) == 1
    rule = resolver.namespaces[0]
    assert rule.namespace == "ilo_accounts"
    assert rule.ref_pattern == _NAMESPACE_RULE["ref_pattern"]
    assert rule.source_type == "infisical"
    assert rule.source_path_template == "/shared/env/{ref}"
    assert rule.ttl_seconds == 300
    assert (
        mod.resolve_ci_bus_broker(overlay=model, lane="dev", injected_broker="")
        == "example.invalid:19092"
    )
    assert mod.resolve_ci_bus_security(overlay=model, lane="dev") == (
        "SASL_PLAINTEXT",
        "SCRAM-SHA-256",
    )


def test_namespaces_remain_optional() -> None:
    mod = _import_trigger_module()

    resolver = mod.ModelCiBusLane.model_validate(
        {"broker": "inmemory", "secret_resolver": {}}
    ).secret_resolver

    assert resolver is not None
    assert resolver.namespaces == ()


@pytest.mark.parametrize(
    "namespace_rule",
    [
        dict(_NAMESPACE_RULE, source_path_templat="/shared/env/{ref}"),
        {k: v for k, v in _NAMESPACE_RULE.items() if k != "source_path_template"},
        {k: v for k, v in _NAMESPACE_RULE.items() if k != "ref_pattern"},
    ],
    ids=["unknown-key", "missing-source-path-template", "missing-ref-pattern"],
)
def test_malformed_namespace_rule_is_refused(
    namespace_rule: dict[str, object],
) -> None:
    """Error: the block stays strict, so a typo is named, never silently dropped."""
    mod = _import_trigger_module()

    with pytest.raises(ValidationError):
        mod.ModelCiBusLane.model_validate(
            {
                "broker": "inmemory",
                "secret_resolver": {"namespaces": [namespace_rule]},
            }
        )


def test_misspelled_namespaces_key_is_refused() -> None:
    mod = _import_trigger_module()

    with pytest.raises(ValidationError):
        mod.ModelCiBusLane.model_validate(
            {
                "broker": "inmemory",
                "secret_resolver": {"namespace": [_NAMESPACE_RULE]},
            }
        )
