# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15547 incident replay for ``scripts/ci/boot_gate.py`` (OMN-17534).

THE REGRESSION BEING REPLAYED IS THE ABSENCE OF THE GUARD. There was no
pre-announcement boot check at all, so a candidate reached onex-dev and took the
runtime plane down before anything looked at it. The artifact is the deploy that
did it: ``Deploy onex-staging`` run 33609666720, job 100181796582, fired by the
merge of omninode_infra#1140 (``e113446ab``) at 2026-09-02T08:37:04Z. Its
rollout-wait step reports, verbatim:

    error: deployment "omninode-runtime" exceeded its progress deadline
    error: deployment "omninode-runtime-effects" exceeded its progress deadline
    error: deployment "omnimarket-projection-api" exceeded its progress deadline

and, further down, the cause:

    ModelOnexError: Auto-wiring failed for 2 contract(s): node_hook_event_capture
    ... ValueError: Projection handler requires topology bindings with configured
    DSNs: tenant_projection:ONEX_TENANT_DB...

That is OMN-17519. It is a BOOT failure under strict wiring mode against the real
manifests, which is exactly and only what the new gate reproduces -- so the
verdict this case pins is the one that did not exist when the incident happened.

WHY THIS ARTIFACT AND NOT A ``kubectl get deployments -o json`` CAPTURE. The
cluster-side JSON would be the more direct input, but onex-dev is reachable only
through the SSM port-forward path and this lane holds no live AWS session; a
capture that cannot be taken must not be invented, and R2 exists precisely to
make that distinction machine-visible. The job log IS the surface that recorded
the incident, it is re-fetchable by anyone with ``gh api``, and it carries the
roster the gate consumes -- which deployments were waited on, and which of them
never became Ready.

WHAT WAS MODIFIED, because R1 is about honesty and not about ceremony:

* EXCERPT. Lines 2550-3400 of the job log (the rollout-wait step through the
  runtime pod's diagnostic block). The full 349,601-byte log is
  sha256 a481eb9a4b7111291b9723ad9a174383cd96718a72c4d7857481f4f72d8caa5a and
  the un-redacted excerpt is
  sha256 9c7683190c225681deb4efa2a2e7391fe5f487c96ee0cbc9281bcbe4bb9060af, both
  recorded so anyone re-fetching can verify exactly what was taken. The full log
  is not committed because this repository is PUBLIC and the remainder carries a
  live RDS endpoint, MSK broker DNS and 32 occurrences of an EC2 instance id --
  a disclosure the repo's own guards exist to prevent.
* FOUR LENGTH-PRESERVING REDACTIONS inside the excerpt: one EC2 instance id and
  three UUIDs. Every byte offset in the file is therefore unchanged, and none of
  them appears in any line this test reads. Same precedent and same reasoning as
  the omn17320 case in this registry.

Nothing else was touched. The deployment names, the ``--tail=30`` invocation and
the auto-wiring traceback are the bytes GitHub recorded.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from scripts.ci import boot_gate

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "omn17534"
    / "deploy-onex-staging-33609666720.rollout-diagnostics.log.captured"
)
FIXTURE_SHA256 = "4dd041f6654137e47a6d65b3775b05ee57536a765fb4a89e6c2748654ddf881c"

_FAILED_RE = re.compile(r'deployment "([a-z0-9-]+)" exceeded its progress deadline')
_WAITED_RE = re.compile(r"Checking rollout state for: ([a-z0-9-]+)")


@pytest.fixture(scope="module")
def captured() -> str:
    assert FIXTURE.is_file(), f"missing captured artifact {FIXTURE}"
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert digest == FIXTURE_SHA256, (
        f"{FIXTURE.name} has been edited since capture ({digest}). A captured "
        "artifact that changed is no longer the thing that failed."
    )
    return FIXTURE.read_text(errors="surrogateescape")


def _roster_from(captured: str) -> tuple[list[str], list[str]]:
    """Read the real roster out of the artifact rather than restating it here.

    Hardcoding the deployment names would make this a test of my typing. Parsing
    them means a re-capture that recorded a different roster changes what the
    guard is driven with, which is the point of a replay.
    """
    waited = sorted(set(_WAITED_RE.findall(captured)))
    failed = sorted(set(_FAILED_RE.findall(captured)))
    assert waited, "the artifact names no rollout-waited Deployment"
    assert failed, "the artifact records no failed rollout"
    assert set(failed) <= set(waited)
    return waited, failed


def test_the_artifact_records_the_incident_it_claims_to(captured: str) -> None:
    """Guard against a re-capture that quietly stops containing the failure."""
    _waited, failed = _roster_from(captured)
    assert failed == [
        "omnimarket-projection-api",
        "omninode-runtime",
        "omninode-runtime-effects",
    ]
    assert "Auto-wiring failed for 2 contract(s)" in captured
    assert "tenant_projection:ONEX_TENANT_DB" in captured


def test_the_real_guard_rejects_the_roster_the_incident_produced(
    captured: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R5, reject direction: the verdict that did not exist on 2026-09-02.

    ``boot_gate.wait_for_boot`` is driven with the artifact's own roster -- every
    Deployment the deploy waited on, with exactly the ones it recorded as failed
    marked 0/1. It must fail closed, and it must name ALL THREE. A gate that
    named one would have cost a second run to find the rest, which is the cost
    OMN-17519 actually paid.
    """
    waited, failed = _roster_from(captured)
    rows = [
        (name, 0 if name in failed else 1, 1, "MinimumReplicasUnavailable")
        for name in waited
    ]
    monkeypatch.setattr(boot_gate, "_deployment_rows", lambda namespace: rows)
    # The projection snapshot topic WAS absent during the incident (the MSK
    # admin probe from inside the namespace found 1887 topics and not this one),
    # but it is pinned True here on purpose: the case must prove the Deployment
    # roster alone is sufficient to reject. Letting the topic carry the verdict
    # would leave the readiness half unproven.
    monkeypatch.setattr(boot_gate, "_topic_exists", lambda *a, **k: True)

    verdict = boot_gate.wait_for_boot(
        namespace="onex-dev",
        timeout_seconds=0,
        poll_seconds=1,
        lane_prefix="onex-lab-",
        broker_deployment="onex-lab-redpanda",
        require_topic=True,
    )
    assert verdict == 1, (
        "the boot gate accepted the exact rollout state that took onex-dev down"
    )
    reported = "".join(capsys.readouterr())
    for name in failed:
        assert name in reported, (
            f"{name} failed in the incident and the gate's report does not name it"
        )


def test_the_same_guard_accepts_the_healthy_roster(
    captured: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The accept control.

    Without it, a guard that rejected every input would satisfy the case above
    while enforcing nothing -- the exact false-enforcement shape OMN-15547 exists
    to stop. Same roster, same code path, every Deployment Ready.
    """
    waited, _failed = _roster_from(captured)
    rows = [(name, 1, 1, "") for name in waited]
    monkeypatch.setattr(boot_gate, "_deployment_rows", lambda namespace: rows)
    monkeypatch.setattr(boot_gate, "_topic_exists", lambda *a, **k: True)

    assert (
        boot_gate.wait_for_boot(
            namespace="onex-dev",
            timeout_seconds=60,
            poll_seconds=1,
            lane_prefix="onex-lab-",
            broker_deployment="onex-lab-redpanda",
            require_topic=True,
        )
        == 0
    )


def test_the_artifact_shows_why_the_new_gate_forbids_tail(captured: str) -> None:
    """OMN-17534 AC-2, evidenced from the incident rather than asserted.

    The deploy's own rollout diagnostics read ``kubectl logs ... --tail=30``. In
    this artifact the runtime pod's fatal traceback is preceded by hundreds of
    lines of contract-discovery and aiokafka noise, so thirty lines could not
    reach it -- which is why OMN-17519 needed a SECOND deploy run just to read
    its own error message. The new gate takes untruncated logs plus
    ``--previous``, and ``test_log_capture_is_complete_and_not_truncated`` in
    tests/ci/test_candidate_boot_gate_omn17534.py holds it there.
    """
    assert "--tail=30" in captured
    lines = captured.splitlines()
    fatal = next(
        index
        for index, line in enumerate(lines)
        if "Auto-wiring failed for 2 contract(s)" in line
    )
    truncation_marker = next(
        index for index, line in enumerate(lines) if "--tail=30" in line
    )
    assert fatal - truncation_marker > 30, (
        "the fatal line is within 30 lines of the truncated invocation, so this "
        "artifact no longer demonstrates the truncation this AC is about"
    )
