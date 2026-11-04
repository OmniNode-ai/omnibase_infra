# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20821 AC4: the local grants declaration is a subset of the lab declaration.

``deploy/local/developer-principal-grants.yaml`` is the developer-run broker's
counterpart of ``deploy/lab/developer-principal-grants.yaml``. Every local grant
must be a lab grant with the same resource, name and pattern and a subset of its
operations: a developer's own broker never grants more than the lab would. Both
files are read through the lab issuer's own ``load_declaration``, so each is held
to the issuer's rules as well.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from omnibase_infra.handlers.handler_broker_grant_derive import HandlerBrokerGrantDerive
from omnibase_infra.nodes.node_broker_grant_derive_compute.models.model_broker_grant_bindings import (
    ModelBrokerGrantBindings,
)
from scripts import lab_principal_issuer as lpi

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
LAB = REPO_ROOT / "deploy" / "lab" / "developer-principal-grants.yaml"
LOCAL = REPO_ROOT / "deploy" / "local" / "developer-principal-grants.yaml"


def grants_outside_the_lab(local: lpi.Declaration, lab: lpi.Declaration) -> list[str]:
    """Each local grant with no lab grant of the same resource, name and pattern
    whose operations include all of its own."""
    lab_operations = {
        (g.resource, g.name, g.pattern): set(g.operations) for g in lab.grants
    }
    outside = []
    for grant in local.grants:
        held = lab_operations.get((grant.resource, grant.name, grant.pattern))
        if held is None or not set(grant.operations) <= held:
            outside.append(
                f"{grant.resource} {grant.pattern} {grant.name} {list(grant.operations)}"
            )
    return outside


def _local_base(tmp_path: Path) -> Path:
    doc = yaml.safe_load(LOCAL.read_text())
    doc.pop("machine_grants_from", None)
    target = tmp_path / "developer-principal-grants.yaml"
    target.write_text(yaml.safe_dump(doc, sort_keys=False))
    return target


def test_every_local_grant_is_a_lab_grant_with_a_subset_of_its_operations(
    tmp_path: Path,
) -> None:
    local, lab = lpi.load_declaration(_local_base(tmp_path)), lpi.load_declaration(LAB)
    assert local.grants, "the local declaration grants nothing"
    assert grants_outside_the_lab(local, lab) == []


@pytest.mark.parametrize(
    "extra",
    [
        {
            "resource": "topic",
            "name": "onex.cmd.omnibase-infra.not-in-the-lab.v1",
            "pattern": "literal",
            "operations": ["write"],
        },
        {
            "resource": "topic",
            "name": "onex.cmd.omnimarket.delegate-skill.v1",
            "pattern": "literal",
            "operations": ["read"],
        },
        {
            "resource": "group",
            "name": "local.omnibase_core.runtime_local_",
            "pattern": "literal",
            "operations": ["read"],
        },
    ],
    ids=["unknown_topic", "wider_operations", "other_pattern"],
)
def test_a_local_grant_the_lab_lacks_is_refused(
    tmp_path: Path, extra: dict[str, object]
) -> None:
    """Positive control: a grant the lab file does not hold is named."""
    doc = yaml.safe_load(_local_base(tmp_path).read_text())
    doc["grants"] = [*doc["grants"], extra]
    widened = tmp_path / "developer-principal-grants.yaml"
    widened.write_text(yaml.safe_dump(doc, sort_keys=False))

    outside = grants_outside_the_lab(
        lpi.load_declaration(widened), lpi.load_declaration(LAB)
    )
    assert len(outside) == 1
    assert str(extra["name"]) in outside[0]


def test_local_machine_supplement_matches_lab_and_derivation() -> None:
    lab_dir = REPO_ROOT / "deploy" / "lab"
    local_dir = REPO_ROOT / "deploy" / "local"
    lab = yaml.safe_load((lab_dir / "machine-principal-bindings.yaml").read_text())
    local = yaml.safe_load((local_dir / "machine-principal-bindings.yaml").read_text())
    expected = {
        **lab,
        "bindings": [{**binding, "broker": "local"} for binding in lab["bindings"]],
    }
    assert local == expected
    request = ModelBrokerGrantBindings.model_validate(local)
    derived = HandlerBrokerGrantDerive().handle(request)
    assert (
        local_dir / "machine-principal-derived-grants.json"
    ).read_text() == derived.model_dump_json(indent=2) + "\n"
    lab_derived = json.loads(
        (lab_dir / "machine-principal-derived-grants.json").read_text()
    )
    expected_rows = {
        tuple(sorted({**row, "broker": "local"}.items()))
        for row in lab_derived["grants"]
    }
    actual_rows = {
        tuple(sorted(grant.model_dump().items())) for grant in derived.grants
    }
    assert actual_rows == expected_rows
    assert derived.principal == lab_derived["principal"]
