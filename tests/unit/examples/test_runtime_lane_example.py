# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The public ``runtime.lane`` example document (OMN-19749, runtime lane overlays LO5).

A runtime learns which lane it is, and what the lane is for, only from a
``runtime.lane`` overlay document that whoever runs it supplies. This package
ships one example of that document, for a single-machine install, so that
``onex local init`` can write it and an operator running their own deployment
can copy it. These tests pin the example to the schema core publishes for the
key, and keep it free of anything that would attach a role-gated contract on
the machine it is copied to.
"""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files

import jsonschema
import pytest

from omnibase_core.enums.enum_config_overlay_key import EnumConfigOverlayKey
from omnibase_core.enums.enum_config_overlay_source import EnumConfigOverlaySource
from omnibase_core.models.config_overlay import (
    ModelConfigOverlayDocument,
    ModelRuntimeLaneDeclaration,
)
from omnibase_infra.examples.config_overlays import (
    RUNTIME_LANE_EXAMPLE_NAME,
    read_runtime_lane_example,
)

pytestmark = pytest.mark.unit


def _example() -> dict[str, object]:
    document = json.loads(read_runtime_lane_example())
    assert isinstance(document, dict)
    return document


def test_example_is_named_for_its_key() -> None:
    assert (
        f"{EnumConfigOverlayKey.RUNTIME_LANE.value}.example.json"
    ) == RUNTIME_LANE_EXAMPLE_NAME


def test_example_validates_against_the_core_model() -> None:
    declaration = ModelRuntimeLaneDeclaration.model_validate(_example())
    assert declaration.lane_id == _example()["lane_id"]


def test_example_validates_against_the_published_json_schema() -> None:
    """The schema core exports for the key's ``schema_ref``, read from the wheel."""
    package, name = EnumConfigOverlayKey.RUNTIME_LANE.schema_ref.split(":")
    schema_text = (
        files(package).joinpath("schemas", "config_overlay", f"{name}.schema.json")
    ).read_text()
    jsonschema.validate(_example(), json.loads(schema_text))


def test_example_claims_no_role() -> None:
    """A role admits role-gated contracts; a first lane needs none, and a copied
    example must not attach any on the machine it lands on."""
    assert _example()["roles"] == []


def test_example_is_what_a_runtime_resolves() -> None:
    """The example resolves through core's own startup rules for its lane."""
    raw = read_runtime_lane_example()
    key = EnumConfigOverlayKey.RUNTIME_LANE
    document = ModelConfigOverlayDocument(
        key=key,
        schema_ref=key.schema_ref,
        schema_version=str(_example()["schema_version"]),
        source=EnumConfigOverlaySource.LOCAL_HOME,
        sha256=hashlib.sha256(raw).hexdigest(),
        content=_example(),
    )
    lane = str(_example()["lane_id"])
    resolved = ModelRuntimeLaneDeclaration.resolve(
        declared_lane_id=lane, document=document, where="the shipped example"
    )
    assert resolved.lane_id == lane
    assert resolved.roles == ()
