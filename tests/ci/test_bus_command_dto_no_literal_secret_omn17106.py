# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17106 (DR-12): no node's bus DTO may hold a literal secret value.

A node's ``handler_routing[].event_model`` is the typed shape of a message the
runtime consumes off a Kafka topic. A field on it that holds a credential VALUE
-- ``SecretStr`` hides the value from ``repr`` but ``model_dump_json`` still
puts it on the wire for the topic's full retention -- is the defect. The
platform pattern is a reference (``api_key_ref``) resolved at the effect
boundary.

The gate walks every node contract under ``src/omnibase_infra/nodes`` and
fails a DTO that has a ``SecretStr``-typed field or a credential-named field
(``util_dlq_credential_redaction.is_credential_field_name``, which already
exempts ``*_ref``). There is no exemption list: a node that needs one has to
justify it in a reviewed change to this file.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel

from omnibase_infra.utils.util_dlq_credential_redaction import is_credential_field_name

pytestmark = pytest.mark.unit

NODES_ROOT = Path(__file__).resolve().parents[2] / "src" / "omnibase_infra" / "nodes"


def _contract_event_models() -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for contract_path in sorted(NODES_ROOT.glob("*/contract.yaml")):
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
        routing = contract.get("handler_routing") or {}
        for entry in routing.get("handlers") or []:
            ref = entry.get("event_model")
            if isinstance(ref, dict) and ref.get("module") and ref.get("name"):
                found.append((contract_path.parent.name, ref["module"], ref["name"]))
    return found


def literal_secret_fields(model: type[BaseModel]) -> list[str]:
    """Names of the fields on ``model`` that are typed to hold a secret value."""
    return [
        name
        for name, field in model.model_fields.items()
        if "SecretStr" in repr(field.annotation) or is_credential_field_name(name)
    ]


def test_gate_sees_the_declared_event_models() -> None:
    # Positive control: an empty walk would pass vacuously.
    assert (
        "node_llm_inference_effect",
        "omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_inference_command",
        "ModelLlmInferenceCommand",
    ) in _contract_event_models()


def test_gate_flags_a_secretstr_field() -> None:
    from pydantic import SecretStr

    class _Leaky(BaseModel):
        api_key: SecretStr | None = None
        api_key_ref: str | None = None

    assert literal_secret_fields(_Leaky) == ["api_key"]


def test_gate_flags_a_plain_str_credential_field() -> None:
    class _Leaky(BaseModel):
        access_token: str = ""

    assert literal_secret_fields(_Leaky) == ["access_token"]


@pytest.mark.parametrize(
    ("node", "module", "name"),
    _contract_event_models(),
    ids=lambda v: v if isinstance(v, str) and "." not in v else None,
)
def test_bus_command_dto_holds_no_literal_secret(
    node: str, module: str, name: str
) -> None:
    model = getattr(importlib.import_module(module), name)
    assert literal_secret_fields(model) == [], (
        f"{node}: {name} declares a field typed to hold a literal secret value; "
        "carry an *_ref and resolve it at the effect boundary"
    )
