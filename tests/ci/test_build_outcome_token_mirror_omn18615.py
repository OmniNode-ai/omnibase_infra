# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The build-outcome token is one vocabulary across a package boundary (OMN-18615).

``deploy_agent.build_budget.EnumBuildOutcome`` is written by the deploy agent
on the lab host. ``check_dev_lane_staleness.EnumLaneBuildOutcome`` is read by
CI, which cannot import the agent's package. Two copies of one vocabulary drift
silently and the reader simply stops recognising the token -- the failure would
be a receipt that no longer says why a build died, reported as a clean receipt.

So the copies are pinned against each other here. This test fails if either
side adds, removes or renames a value, which makes a rename a red test rather
than a review catch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "deploy-agent"))

from deploy_agent.build_budget import EnumBuildOutcome

from scripts.ci.check_dev_lane_staleness import EnumLaneBuildOutcome


@pytest.mark.unit
class TestTheTwoCopiesAreOneVocabulary:
    def test_member_names_match(self) -> None:
        assert {member.name for member in EnumBuildOutcome} == {
            member.name for member in EnumLaneBuildOutcome
        }

    def test_member_values_match(self) -> None:
        assert {member.name: member.value for member in EnumBuildOutcome} == {
            member.name: member.value for member in EnumLaneBuildOutcome
        }

    def test_neither_value_is_a_substring_of_the_other(self) -> None:
        """A reader classifying by substring must not match both."""
        values = [member.value for member in EnumLaneBuildOutcome]
        for one in values:
            for other in values:
                if one is not other:
                    assert one not in other

    def test_the_reader_recognises_a_token_the_writer_produces(self) -> None:
        """The two halves agree on a real agent error string, not only on names."""
        written = (
            f"{EnumBuildOutcome.BUDGET_EXHAUSTED.value}: runtime image build for "
            "profile 'runtime' exceeded its 2177s ceiling and was killed."
        )
        assert (
            EnumLaneBuildOutcome.classify([written])
            is EnumLaneBuildOutcome.BUDGET_EXHAUSTED
        )

    def test_an_unrelated_error_list_classifies_to_none(self) -> None:
        assert EnumLaneBuildOutcome.classify(["lane lock contended"]) is None
        assert EnumLaneBuildOutcome.classify([]) is None
        assert EnumLaneBuildOutcome.classify(None) is None
