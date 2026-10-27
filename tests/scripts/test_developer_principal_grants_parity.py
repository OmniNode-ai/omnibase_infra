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

from pathlib import Path

import pytest
import yaml

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


def test_every_local_grant_is_a_lab_grant_with_a_subset_of_its_operations() -> None:
    local, lab = lpi.load_declaration(LOCAL), lpi.load_declaration(LAB)
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
    doc = yaml.safe_load(LOCAL.read_text())
    doc["grants"] = [*doc["grants"], extra]
    widened = tmp_path / "developer-principal-grants.yaml"
    widened.write_text(yaml.safe_dump(doc, sort_keys=False))

    outside = grants_outside_the_lab(
        lpi.load_declaration(widened), lpi.load_declaration(LAB)
    )
    assert len(outside) == 1
    assert str(extra["name"]) in outside[0]
