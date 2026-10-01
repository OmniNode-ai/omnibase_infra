# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The usage-by-model-day shipped instances carry the writer grant (OMN-19978).

The interim supplemental bridges were retired by the omnimarket contract pin
advance (OMN-17292), so the node contract now derives both relations. This
proves each shipped instance still gives tenant_projection_writer exactly
SELECT, INSERT and UPDATE on each relation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RELATIONS = ("usage_by_model_day", "usage_by_model_day_calls")
_SCHEMA = "public"
_PRINCIPAL = "tenant_projection_writer"
_REQUIRED_PRIVILEGES = frozenset({"INSERT", "SELECT", "UPDATE"})
_SHIPPED_INSTANCES = ("local", "onex-dev", "onex-prod")


def _shipped_grants(profile: str, relation: str) -> list[dict[str, Any]]:
    path = (
        _REPO_ROOT
        / "src"
        / "omnibase_infra"
        / "topology"
        / "instances"
        / f"{profile}.yaml"
    )
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    principal = document["databases"]["application"]["principals"][_PRINCIPAL]
    return [
        grant
        for grant in principal["grants"]
        if grant.get("object_type") == "TABLE"
        and grant.get("schema") == _SCHEMA
        and relation in tuple(grant.get("objects") or ())
    ]


@pytest.mark.parametrize("relation", _RELATIONS)
@pytest.mark.parametrize("profile", _SHIPPED_INSTANCES)
class TestTheUsageByModelDayShippedGrant:
    def test_shipped_instance_grants_exact_writer_privileges(
        self, profile: str, relation: str
    ) -> None:
        matching = _shipped_grants(profile, relation)
        assert len(matching) == 1, (
            f"{profile} ships {len(matching)} {_PRINCIPAL} TABLE grants for "
            f"{_SCHEMA}.{relation}, not one. Regenerate the instance."
        )
        assert frozenset(matching[0]["privileges"]) == _REQUIRED_PRIVILEGES
