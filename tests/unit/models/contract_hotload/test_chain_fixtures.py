# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The golden and error chain fixtures of contract hot loading (OMN-21226, H1).

Plan section 4 names the chains G1 to G8 and E1 to E17. Each is a fixture under
``tests/fixtures/contract_hotload/``, written before the code that makes it pass.

This file checks the fixture shape. It does not import the record models: they are in
omnibase_core's ``models.events.contract_hotload``, and this repository pins a core
release that does not carry them yet. The follow-up that moves the pin adds the test
that parses each ``record`` with its named model and refuses an undeclared field; the
planted-violation test below is the control that this shape check refuses one at the
step level in the meantime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "contract_hotload"
_GOLDEN = tuple(f"G{n}" for n in range(1, 9))
_ERROR = tuple(f"E{n}" for n in range(1, 18))
_CHAIN_KEYS = {"chain", "kind", "title", "steps", "ends_in", "note"}
_REQUIRED_CHAIN_KEYS = {"chain", "kind", "title", "steps", "ends_in"}
_STEP_KEYS = {
    "step",
    "actor",
    "action",
    "topic_role",
    "model",
    "record",
    "key",
    "result",
    "note",
}
_REQUIRED_STEP_KEYS = {"step", "actor", "action"}
_ACTIONS = {"publish", "commit", "observe", "replay"}
_MODELS = {
    "ModelContractLoadRequest",
    "ModelContractHotloadAccepted",
    "ModelContractHotloadRemoved",
    "ModelContractLoadReply",
    "ModelContractRevision",
    "ModelRegistryWatermark",
    "ModelContractHotloadApplied",
    "ModelRuntimeHotloadState",
}
_ROLES = {
    "sender_request",
    "accepted",
    "replies",
    "revisions",
    "watermark",
    "applied",
    "state",
}


def _load(chain: str) -> dict[str, Any]:
    loaded = json.loads((_FIXTURES / f"{chain}.json").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def shape_errors(fixture: dict[str, Any], chain: str) -> list[str]:
    """Every way a fixture departs from the shape; empty when it conforms."""
    errors: list[str] = []
    extra = set(fixture) - _CHAIN_KEYS
    missing = _REQUIRED_CHAIN_KEYS - set(fixture)
    if extra:
        errors.append(f"undeclared chain fields {sorted(extra)}")
    if missing:
        errors.append(f"missing chain fields {sorted(missing)}")
    if fixture.get("chain") != chain:
        errors.append(f"chain id {fixture.get('chain')!r} is not {chain!r}")
    steps = fixture.get("steps")
    if not isinstance(steps, list) or not steps:
        return [*errors, "steps must be a non-empty list"]
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            errors.append(f"step {index} is not a mapping")
            continue
        if set(step) - _STEP_KEYS:
            errors.append(
                f"step {index} has undeclared fields {sorted(set(step) - _STEP_KEYS)}"
            )
        if _REQUIRED_STEP_KEYS - set(step):
            errors.append(
                f"step {index} lacks {sorted(_REQUIRED_STEP_KEYS - set(step))}"
            )
        if step.get("step") != index:
            errors.append(f"step {index} is numbered {step.get('step')!r}")
        if step.get("action") not in _ACTIONS:
            errors.append(f"step {index} action {step.get('action')!r} is not declared")
        if ("model" in step) != ("record" in step):
            errors.append(f"step {index} names a model without a record or the reverse")
        if "model" in step and step["model"] not in _MODELS:
            errors.append(
                f"step {index} model {step['model']!r} is not a hot-loading record"
            )
        if "model" in step and not isinstance(step.get("record"), dict):
            errors.append(f"step {index} record is not a mapping")
        if "model" in step and "topic_role" not in step:
            errors.append(f"step {index} carries a record without a topic role")
        if "topic_role" in step and step["topic_role"] not in _ROLES:
            errors.append(
                f"step {index} topic role {step['topic_role']!r} is not declared"
            )
    return errors


def test_the_plan_names_eight_golden_and_seventeen_error_chains() -> None:
    assert len(_GOLDEN) == 8
    assert len(_ERROR) == 17


def test_the_fixture_directory_holds_exactly_the_planned_chains() -> None:
    present = {path.stem for path in _FIXTURES.glob("*.json")}
    assert present == set(_GOLDEN) | set(_ERROR)


@pytest.mark.parametrize("chain", _GOLDEN + _ERROR)
def test_every_fixture_conforms_to_the_shape(chain: str) -> None:
    assert shape_errors(_load(chain), chain) == []


@pytest.mark.parametrize("chain", _GOLDEN)
def test_a_golden_chain_is_kind_golden(chain: str) -> None:
    assert _load(chain)["kind"] == "golden"


@pytest.mark.parametrize("chain", _ERROR)
def test_an_error_chain_is_kind_error(chain: str) -> None:
    assert _load(chain)["kind"] == "error"


@pytest.mark.parametrize("chain", _GOLDEN + _ERROR)
def test_a_content_hash_is_the_canonical_form(chain: str) -> None:
    for step in _load(chain)["steps"]:
        record = step.get("record", {})
        hash_value = record.get("content_hash")
        if hash_value is not None:
            assert len(hash_value) == 64
            assert hash_value == hash_value.lower()
            assert set(hash_value) <= set("0123456789abcdef")


def test_planted_violation_an_undeclared_step_field_is_refused() -> None:
    fixture = _load("G1")
    fixture["steps"][0]["undeclared_field"] = "x"
    assert any("undeclared" in error for error in shape_errors(fixture, "G1"))


def test_planted_violation_an_undeclared_chain_field_is_refused() -> None:
    fixture = _load("G1")
    fixture["undeclared_field"] = "x"
    assert any("undeclared chain" in error for error in shape_errors(fixture, "G1"))


def test_planted_violation_a_record_of_an_unknown_model_is_refused() -> None:
    fixture = _load("G1")
    fixture["steps"][0]["model"] = "ModelSomethingElse"
    assert any("not a hot-loading record" in e for e in shape_errors(fixture, "G1"))
