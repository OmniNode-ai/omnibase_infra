# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: a recorded real resolver answer passes the dev-merge tag decision.

The artifact is the typed answer ``resolve_dev_merge_tag_hermetic`` returned for
a real core dev-merge tag (mirror first, GitHub REST second), kept with its
provenance. The decision that judges it is the production one, so a change to
the resolver's answer shape or to the decision fails here against real bytes.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from scripts.check_dep_provenance import (
    DevMergeLockEntry,
    DevMergeTagFacts,
    dev_merge_tag_violations,
)

pytestmark = pytest.mark.unit

GIT_URL = "https://github.com/OmniNode-ai/omnibase_core.git"


@pytest.mark.live_contact("tests/fixtures/dep_provenance_dev_tag_live_omn17427.json")
def test_recorded_tag_response_passes_lineage_and_off_dev_twin_is_refused(
    recorded_response: dict[str, object],
) -> None:
    tag = recorded_response["tag"]
    response = recorded_response["response"]
    assert isinstance(tag, str)
    assert isinstance(response, dict)
    facts = DevMergeTagFacts(**response)
    assert facts.tag_exists is True
    assert facts.on_dev is True
    assert facts.commit is not None
    assert facts.version is not None
    lock = DevMergeLockEntry(facts.version, f"{GIT_URL}?tag={tag}#{facts.commit}")
    assert (
        dev_merge_tag_violations(
            "omnibase-core", tag, facts.version, facts, lock, lock_required=True
        )
        == []
    )
    rejected = dev_merge_tag_violations(
        "omnibase-core",
        tag,
        facts.version,
        replace(facts, on_dev=False),
        lock,
        lock_required=True,
    )
    assert len(rejected) == 1
    assert "tag commit is reachable from dev" in rejected[0]
