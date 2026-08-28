# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Incident replay: the release train left dev sitting on the published version.

Registry case ``omn13912-release-left-dev-at-the-published-version``
(``tests/incident_replays/registry.yaml``).

The artifact is not a constructed example. It is the verbatim ``pyproject.toml``
of ``4529c3486`` — the commit the ``v0.38.11`` tag points at — read out of the
git object and committed under a ``.captured`` suffix so no formatter can touch
it. It declares ``version = "0.38.11"``.

That single fact is the whole incident: at the instant ``v0.38.11`` was
published, dev's ``[project].version`` was **equal to** the newly published
version, which is exactly the state ``scripts/check_release_identity.py``
(OMN-13412) rejects for any diff touching ``src/**``. The release train's
verdict on that tree was "done" — it published, tagged, and exited — so the
gate stayed armed against dev for ~1h38m until an unrelated feature PR
(OMN-16769, ``93c42ada4``) happened to bump the version. The same shape is
recoverable one release earlier for ``v0.38.10`` (~2h06m, disarmed by
OMN-16536).

The test drives the real ``decide()`` over the captured bytes and requires it to
call that tree ARMED. A world with this guard could not have shipped v0.38.10 or
v0.38.11 into a wedged dev; the world without it did, twice.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.ci.post_release_dev_bump import (
    ACTION_BUMP,
    decide,
    parse_final_version,
    read_project_version,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "omn13912" / "pyproject-4529c348.toml.captured"
)

# The tag under replay, and the sha the registry case pins.
RELEASED_TAG = "v0.38.11"
TAG_COMMIT = "4529c3486a8bb49a45a623a008e4bbfa2a3a3148"
FIXTURE_SHA256 = "ae1bf5904ad8a7adfb8fc84c8f9783a1326257473b00562d69bf23b2ab8824d7"


def test_the_captured_artifact_is_the_bytes_the_registry_pins() -> None:
    # R1: if the fixture is ever reformatted it stops being the artifact that
    # failed, and this replay silently becomes decorative.
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert digest == FIXTURE_SHA256, (
        f"{FIXTURE} no longer matches the sha256 recorded in "
        f"tests/incident_replays/registry.yaml — re-fetch it with "
        f"`git cat-file blob {TAG_COMMIT}:pyproject.toml`"
    )


def test_the_real_tagged_tree_carried_the_published_version() -> None:
    # The precondition that makes this an incident rather than a hypothetical.
    assert read_project_version(FIXTURE) == RELEASED_TAG.lstrip("v")


def test_the_real_v0_38_11_dev_tree_is_flagged_as_armed() -> None:
    # The verdict the release train got wrong: this tree is NOT fine.
    decision = decide(
        dev_version=read_project_version(FIXTURE), released_version=RELEASED_TAG
    )
    assert decision.action == ACTION_BUMP
    assert decision.target_version == "0.38.12"


def test_the_replayed_remedy_actually_disarms_the_release_identity_gate() -> None:
    # Not merely "different from published" — strictly greater, which is the
    # invariant check_release_identity.py enforces.
    decision = decide(
        dev_version=read_project_version(FIXTURE), released_version=RELEASED_TAG
    )
    assert parse_final_version(
        decision.target_version, label="target"
    ) > parse_final_version(RELEASED_TAG, label="released")
