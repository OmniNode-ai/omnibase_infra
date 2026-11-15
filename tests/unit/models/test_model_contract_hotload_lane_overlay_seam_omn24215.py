# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Seam test: the runtime's lane overlay reader against the overlay home's checker (OMN-24215).

The overlays repository validates ``overlays/runtime_contract_hotload/overlay.yaml`` through
``omnimarket.models.overlay_documents.model_contract_hotload_lane_overlay``; the runtime reads the same
document through ``ModelContractHotloadLaneOverlay``. The two models live in two repositories, so the corpus
below is spelled here and again in omnimarket's
``tests/unit/models/test_contract_hotload_lane_overlay_document_omn24215.py``, and each side asserts its own
model against its own copy. Nothing here imports the other repository, so the test runs everywhere and is
never skipped. A key added to or dropped from this reader changes a verdict in the corpus or the field-set
assertions and fails here; the same change on the omnimarket side fails there.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.models.model_contract_hotload_config import (
    ModelContractHotloadConfig,
)
from omnibase_infra.models.model_contract_hotload_lane_overlay import (
    ModelContractHotloadLaneOverlay,
)

pytestmark = pytest.mark.unit

ENABLED_ON_DEMAND_LANE = {
    "enabled": True,
    "mode": "on_demand",
    "allowed_environments": ["env_a"],
}

ACCEPTED_DOCUMENTS: list[dict[str, object]] = [
    {"schema_version": 1, "lanes": {"lane_a": ENABLED_ON_DEMAND_LANE}},
    {
        "schema_version": 1,
        "lanes": {
            "lane_a": {
                "enabled": True,
                "mode": "full_load",
                "allowed_environments": ["env_a"],
            }
        },
    },
    {"schema_version": 1, "lanes": {"lane_a": {"enabled": False}}},
    {"schema_version": 1, "lanes": {}},
    {"schema_version": 1, "lanes": {"": {"enabled": False}}},
    {
        "schema_version": 1,
        "lanes": {"lane_a": ENABLED_ON_DEMAND_LANE, "lane_b": {"enabled": False}},
    },
]

REFUSED_DOCUMENTS: list[dict[str, object]] = [
    {"schema_version": 1, "lanes": {"lane_a": {"enabled": False}}, "stray": 1},
    {"schema_version": 2, "lanes": {"lane_a": {"enabled": False}}},
    {"schema_version": 1, "lanes": {"lane_a": {"enabled": False, "limit": 5}}},
    {"schema_version": 1, "lanes": {"lane_a": {"enabled": False, "mode": "always"}}},
    {"schema_version": 1, "lanes": {"lane_a": {"enabled": True}}},
    {
        "schema_version": 1,
        "lanes": {
            "lane_a": {
                "enabled": True,
                "mode": "on_demand",
                "allowed_environments": [],
            },
        },
    },
    {
        "schema_version": 1,
        "lanes": {
            "lane_a": {
                "enabled": True,
                "mode": "on_demand",
                "allowed_environments": ["  "],
            },
        },
    },
]


@pytest.mark.parametrize("document", ACCEPTED_DOCUMENTS)
def test_runtime_reader_accepts_the_seam_corpus(document: dict[str, object]) -> None:
    overlay = ModelContractHotloadLaneOverlay.model_validate(document)
    assert overlay.schema_version == 1


@pytest.mark.parametrize("document", REFUSED_DOCUMENTS)
def test_runtime_reader_refuses_the_seam_corpus(document: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ModelContractHotloadLaneOverlay.model_validate(document)


def test_runtime_reader_normalizes_environment_names_like_the_checker() -> None:
    overlay = ModelContractHotloadLaneOverlay.model_validate(
        {
            "schema_version": 1,
            "lanes": {
                "lane_a": {
                    "enabled": True,
                    "mode": "on_demand",
                    "allowed_environments": ["  ENV_A  "],
                },
            },
        }
    )
    assert overlay.lanes["lane_a"].allowed_environments == ("env_a",)


def test_runtime_reader_holds_exactly_the_keys_the_checker_holds() -> None:
    """A key added to or dropped from the runtime reader fails here (the checker holds these three row keys)."""
    assert set(ModelContractHotloadLaneOverlay.model_fields) == {
        "schema_version",
        "lanes",
    }
    assert set(ModelContractHotloadConfig.model_fields) == {
        "enabled",
        "mode",
        "allowed_environments",
    }
