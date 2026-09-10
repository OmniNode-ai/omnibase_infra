# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15547 incident replay for ``scripts/ci/lab_pass_receipt.py`` (OMN-17530).

THE REGRESSION BEING REPLAYED IS THE ABSENCE OF A DURABLE LAB PREMISE. Before
this guard, a dev candidate was announced to ``omninode_infra`` on the strength
of an in-run ``needs:`` edge and nothing else. That edge is a fact about one
workflow run: it cannot be read by the deploy that follows in another
repository, by a later run, or by anybody asking afterwards which shas had been
exercised on a lab lane. On 2026-09-08 that question mattered and could not be
answered -- twelve staging runs failed on two defects (a tenant-form split
between the ``onex-api`` reader and the projection writer, and a quality-gate
``NOT NULL`` timestamp) that nothing had exercised on a lab lane first, which is
the incident ``omni_home`` ``CLAUDE.md`` Operating Rule 24 was written for.

THE ARTIFACT is the surface this guard actually reads, captured live: the
GitHub artifact listing for the lab-pass receipt of a sha that WAS announced to
staging that morning. ``omnibase_infra`` dev commit
``10abdcc418654b1c8047ed802c3f465888cdb108``, delivered by
``deliver-dev-candidate-to-staging.yml`` run 34199710333 (2026-09-08T07:30:58Z),
whose four jobs all report ``success`` -- including ``Announce the bundle to
omninode_infra``. The listing for that sha's receipt is, verbatim:

    {"total_count":0,"artifacts":[]}

Zero. The candidate went to staging with no durable, sha-keyed record that any
lab lane had exercised it. That is the verdict the guard did not exist to
give, and this case pins it.

WHY AN ARTIFACT LISTING AND NOT A JOB LOG. R5 requires the case to drive the
REAL guard over the REAL bytes and pin the verdict it got wrong. The gate's
input is not a log; it is exactly this REST response. A log capture would prove
the announcement happened but would never reach the code path that decides.

THE ZERO IS PROVEN REAL, NOT ASSUMED (repo rule 16: an empty result is not
evidence of absence). ``test_the_empty_listing_is_a_real_zero`` re-runs the same
exact-name query shape against a name that DOES exist and requires rows back, so
"no receipt" is distinguishable from "the query never worked". Measured live at
capture time: ``name=saturation-record`` returned 69 artifacts through the same
``list_artifacts`` call that returned 0 for the receipt name.

THE DISCRIMINATOR. An accept-only or reject-only proof cannot tell a correct
guard from one stuck on a single answer. ``test_the_same_guard_accepts_a_sha
_that_does_carry_a_pass_receipt`` drives the identical code path over the same
captured listing shape with one receipt artifact present, and requires the gate
to pass -- so a guard that simply rejected everything could not satisfy this
module.

NOTHING WAS MODIFIED. The fixture is the 32 bytes ``gh api`` returned, byte for
byte. It carries no identifier, credential, host or account value, so unlike the
``omn17534`` case in this registry it needed no redaction at all. The
registry's locator is the RUN-SCOPED listing for run 34199710333, because
R2's grammar requires a locator that pins one resource and the
repository-wide ``?name=`` form pins none; the two responses were measured
byte-identical at capture time.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from scripts.ci.lab_pass_receipt import (
    EnumLabLane,
    ModelLabPassCheck,
    artifact_name,
    build_receipt,
    evaluate_gate,
    list_artifacts,
)

pytestmark = pytest.mark.unit

#: The dev commit that was announced to onex-dev by run 34199710333.
ANNOUNCED_SHA = "10abdcc418654b1c8047ed802c3f465888cdb108"

REPO = "OmniNode-ai/omnibase_infra"

#: Named literally, not built from ANNOUNCED_SHA: the registry's R4 check reads
#: this module's source for the fixture it claims to replay, and an f-string
#: assembles a name no reader (human or machine) can grep for.
FIXTURE_NAME = "run-34199710333-lab-pass-listing.json.captured"

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "omn17530" / FIXTURE_NAME


def _captured_listing() -> str:
    return FIXTURE.read_text(encoding="utf-8")


class _SurfaceFromCapture:
    """Serves the captured REST bytes for every lab lane's receipt name.

    ``extra`` optionally injects one real receipt so the discriminator below can
    drive the same code path to the opposite verdict.
    """

    def __init__(self, extra: dict[str, str] | None = None) -> None:
        self.extra = extra or {}

    def __call__(self, path: str) -> bytes:
        if "/actions/artifacts?name=" in path:
            name = path.split("name=")[1].split("&")[0]
            if name in self.extra:
                return json.dumps(
                    {
                        "total_count": 1,
                        "artifacts": [
                            {
                                "id": 4242,
                                "name": name,
                                "expired": False,
                                "created_at": "2026-09-08T07:45:00Z",
                            }
                        ],
                    }
                ).encode()
            # The captured bytes, unmodified.
            return _captured_listing().encode()
        name = next(iter(self.extra))
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("receipt.json", self.extra[name])
        return buffer.getvalue()


def _run(surface: _SurfaceFromCapture, monkeypatch: Any) -> tuple[int, str]:
    monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", surface)
    out = io.StringIO()
    code = evaluate_gate(REPO, ANNOUNCED_SHA, list(EnumLabLane), out)
    return code, out.getvalue()


def test_the_captured_listing_is_the_bytes_github_returned() -> None:
    """R1: the fixture is the response, not a reconstruction of it."""
    payload = json.loads(_captured_listing())
    assert payload == {"total_count": 0, "artifacts": []}


def test_the_real_guard_refuses_the_sha_that_was_announced_with_no_receipt(
    monkeypatch: Any,
) -> None:
    """R5: the verdict the absent guard could not give.

    Run 34199710333 announced this sha to omninode_infra with every job green.
    Driven over the same surface, the shipped gate refuses it and names the
    commit -- which is what makes the refusal actionable rather than a red X.
    """
    code, output = _run(_SurfaceFromCapture(), monkeypatch)
    assert code == 1
    assert f"lab-pass gate FAILED for {ANNOUNCED_SHA}" in output
    assert "no receipt artifact named" in output
    # Not a skip, and not silently downgraded to a warning.
    assert "This is not a skip" in output


def test_the_same_guard_accepts_a_sha_that_does_carry_a_pass_receipt(
    monkeypatch: Any,
) -> None:
    """The discriminator: a guard that rejects everything fails this test."""
    receipt = build_receipt(
        sha=ANNOUNCED_SHA,
        lane=EnumLabLane.COMPOSE_DEV,
        started_at=datetime(2026, 9, 8, 7, 31, tzinfo=UTC),
        finished_at=datetime(2026, 9, 8, 7, 45, tzinfo=UTC),
        checks=[
            ModelLabPassCheck(
                name="deployed_revision",
                ok=True,
                evidence=(
                    "check_dev_lane_staleness.py --expect-revision "
                    f"{ANNOUNCED_SHA} against the omnibase-infra compose project"
                ),
            )
        ],
        agent_command_id=None,
    )
    surface = _SurfaceFromCapture(
        {artifact_name(EnumLabLane.COMPOSE_DEV, ANNOUNCED_SHA): (receipt.to_json())}
    )
    code, output = _run(surface, monkeypatch)
    assert code == 0
    assert "lab-pass gate PASSED" in output


@pytest.mark.integration
def test_the_empty_listing_is_a_real_zero() -> None:
    """Rule 16 positive control: prove the query returns rows when rows exist.

    Without this, a listing call that silently failed and a genuinely empty
    surface are the same reading -- which is the class of false clean bill of
    health this repository has paid for repeatedly. Marked integration because
    it makes a live API call; the replay above needs no network.
    """
    rows = list_artifacts(REPO, "saturation-record")
    assert rows, (
        "the exact-name artifact query returned nothing for a name known to "
        "exist; the empty result in the fixture cannot be read as absence "
        "until this control fires"
    )
    assert not list_artifacts(
        REPO, artifact_name(EnumLabLane.COMPOSE_DEV, ANNOUNCED_SHA)
    )
