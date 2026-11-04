# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The golden and error chain fixtures of contract hot loading (OMN-21226, H1).

Plan section 4 names the chains G1 to G8 and E1 to E17. Each is a fixture under
``tests/fixtures/contract_hotload/``, written before the code that makes it pass.

Records are parsed by the hot-loading models in the pinned core dependency,
including their Python validators. No schema copy or sibling checkout can drift
from the models the runtime uses.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from omnibase_core.models.events.contract_hotload.model_contract_hotload_accepted import (
    ModelContractHotloadAccepted,
)
from omnibase_core.models.events.contract_hotload.model_contract_hotload_applied import (
    ModelContractHotloadApplied,
)
from omnibase_core.models.events.contract_hotload.model_contract_hotload_removed import (
    ModelContractHotloadRemoved,
)
from omnibase_core.models.events.contract_hotload.model_contract_load_reply import (
    ModelContractLoadReply,
)
from omnibase_core.models.events.contract_hotload.model_contract_load_request import (
    ModelContractLoadRequest,
)
from omnibase_core.models.events.contract_hotload.model_contract_revision import (
    ModelContractRevision,
)
from omnibase_core.models.events.contract_hotload.model_registry_watermark import (
    ModelRegistryWatermark,
)
from omnibase_core.models.events.contract_hotload.model_runtime_hotload_state import (
    ModelRuntimeHotloadState,
)

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
_MODELS: dict[str, type[BaseModel]] = {
    model.__name__: model
    for model in (
        ModelContractLoadRequest,
        ModelContractHotloadAccepted,
        ModelContractHotloadRemoved,
        ModelContractLoadReply,
        ModelContractRevision,
        ModelRegistryWatermark,
        ModelContractHotloadApplied,
        ModelRuntimeHotloadState,
    )
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
        if step.get("model") in _MODELS and isinstance(step.get("record"), dict):
            try:
                _MODELS[step["model"]].model_validate_json(
                    json.dumps(step["record"]), strict=True
                )
            except ValidationError as error:
                errors.append(f"step {index} record: {error}")
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


def test_planted_violation_an_undeclared_record_field_is_refused() -> None:
    fixture = _load("G1")
    fixture["steps"][0]["record"]["undeclared_field"] = "x"
    assert any("undeclared_field" in error for error in shape_errors(fixture, "G1"))


def _record_objects(
    record: dict[str, Any], path: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], dict[str, Any]]]:
    # `names` is a map keyed by contract name; its values and every other
    # object are closed models, including observed and served entries.
    if path != ("names",):
        yield path, record
    for key, value in record.items():
        if isinstance(value, dict):
            yield from _record_objects(value, (*path, key))


@pytest.mark.parametrize("chain", _GOLDEN + _ERROR)
def test_every_record_refuses_unknown_and_missing_fields(chain: str) -> None:
    fixture = _load(chain)
    for step in fixture["steps"]:
        if "record" not in step:
            continue
        model = _MODELS[step["model"]]
        model.model_validate_json(json.dumps(step["record"]), strict=True)
        for path, obj in _record_objects(step["record"]):
            for field in ("undeclared_field", *obj):
                mutated = deepcopy(step["record"])
                target = mutated
                for key in path:
                    target = target[key]
                if field == "undeclared_field":
                    target[field] = "x"
                else:
                    del target[field]
                with pytest.raises(ValidationError):
                    model.model_validate_json(json.dumps(mutated), strict=True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("operation", "unknown"),
        ("registration_request_id", "not-a-uuid"),
        ("content_hash", "A" * 64),
        ("expected_version", -1),
        ("expected_version", "1"),
        ("owner", ""),
    ],
)
def test_planted_violation_invalid_record_values_are_refused(
    field: str, value: Any
) -> None:
    fixture = _load("G1")
    fixture["steps"][0]["record"][field] = value
    assert shape_errors(fixture, "G1")


def test_fixture_validation_follows_core_validator_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate core tightening a Python validator without changing its JSON Schema.
    # A copied schema would keep accepting this record and hide the drift.
    def tightened_validator(*args: Any, **kwargs: Any) -> Any:
        raise ValidationError.from_exception_data(
            "ModelContractLoadRequest",
            [
                {
                    "type": "value_error",
                    "loc": ("owner",),
                    "input": "lab-deploy",
                    "ctx": {"error": ValueError("owner rejected by core validator")},
                }
            ],
        )

    monkeypatch.setattr(
        ModelContractLoadRequest, "model_validate_json", tightened_validator
    )
    assert any(
        "owner rejected by core validator" in error
        for error in shape_errors(_load("G1"), "G1")
    )
