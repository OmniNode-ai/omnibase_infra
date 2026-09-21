# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Incident replay for the fleet failure sink (OMN-18942, OMN-15547 rules R1-R5).

THE FALSE GREEN BEING REPLACED IS A FLEET NOBODY IS TOLD ABOUT.

The evaluator behind these bytes has been correct and running every ten minutes
since OMN-18322. Its last log line, verbatim, on every run:

    [nonrequired-checks] Slack not configured; annotation + artifact only

No chat secret of any name exists at GitHub Actions organisation scope or in any
repository scope, so twenty-nine above-threshold scheduled workflows -- five of
which have not succeeded ONCE in the trailing window -- were named into an
annotation on a job nobody opens. The detection was never the defect. The hop
was, and this probe is the hop.

THE ARTIFACT IS NOT A RECONSTRUCTION. It is the byte-for-byte report the shared
evaluator wrote on the `.201` host at 2026-09-20T22:53Z, running as root out of
the host env file, over the thirteen-repository fleet list. It is re-fetchable
by re-running the same command on that host, and it carries repository names,
workflow paths, run counts and public run URLs only -- no token, no identity, no
address.

IT ALSO CARRIES THE SECOND INCIDENT, which is why this one capture proves both.
`OmniNode-ai/omni_home` is present with an `error` rather than a result: the
host token reads twelve of the thirteen fleet repositories and returns HTTP 404
on the registry repository -- the one whose scheduled-gap detector had failed 12
of 12 daily runs since 2026-09-09 and which this ticket names by hand. Before
the per-repository split, that single 404 aborted the whole sweep, so the first
live run on `.201` produced NO rows for ANY repository. The measured evidence is
in the refresh log from that run:

    RuntimeError: gh api repos/OmniNode-ai/omni_home/actions/workflows?per_page=100
        failed: gh: Not Found (HTTP 404)

That is the OMN-18254 failure recurring on a second surface: "the repo loop hits
omniweb second, so omnimarket was never reached either -- one private repo took
the whole sweep down."
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE = REPO_ROOT / "scripts" / "omninode-fleet-failure-probe.py"
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "omn18942"
    / "fleet-scheduled-sweep-201-2026-09-20.json.captured"
)
FIXTURE_SHA256 = "5dff47a214fdef2d90276986dcee3265e7e584be8006e0c0d862f9a05ee7cf4e"

#: The policy's own `failure_threshold`. Passed explicitly here rather than read
#: from the policy file so this replay pins the behaviour at the number that was
#: in force when the incident was captured; the parity of the two is asserted in
#: test_fleet_failure_sink_omn18942.py.
MIN_RUNS_FOR_CRITICAL = 3


def _load_probe() -> Any:
    spec = importlib.util.spec_from_file_location("fleet_failure_probe_replay", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe_module = _load_probe()


def _report() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_the_fixture_is_the_bytes_that_were_captured() -> None:
    """R2. A fixture nobody pinned is a fixture somebody can quietly edit."""
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert digest == FIXTURE_SHA256, (
        "the captured sweep report has changed on disk; a reformatted artifact "
        "is no longer the artifact that failed"
    )


def test_the_real_probe_names_every_never_green_surface_on_the_captured_fleet() -> None:
    """The refusal that never happened: these five reached nobody for a week."""
    rows, _ = probe_module._rows_from_report(_report(), MIN_RUNS_FOR_CRITICAL)
    critical = sorted(r["key"] for r in rows if r["status"] == "CRITICAL")
    assert critical == [
        "sched/omnibase_infra/nightly-integration.yml",
        "sched/omniclaude/public-repo-hygiene-audit.yml",
        "sched/omninode_infra/m3-tenant-attribution-chain.yml",
        "sched/omninode_infra/publish-unshipped-board.yml",
        "sched/onex_change_control/staleness-monitor.yml",
    ], critical
    for row in rows:
        if row["status"] == "CRITICAL":
            assert "has not succeeded once" in row["detail"]
            # Actionable at 3am or it is not an alert: the rate, the counts and
            # a run URL, not just a workflow name.
            assert "https://github.com/OmniNode-ai/" in row["detail"]


def test_the_real_probe_names_the_repository_it_could_not_read() -> None:
    """The second incident in the same bytes.

    A repository the sweep could not read must be a standing row of its own. The
    registry repository was invisible to the fleet's only alerter BY
    CONSTRUCTION before this, and nothing on any board could tell that apart
    from a clean result.
    """
    rows, heartbeat = probe_module._rows_from_report(_report(), MIN_RUNS_FOR_CRITICAL)
    unreadable = [r for r in rows if r["key"] == "sched/omni_home/source-unreadable"]
    assert len(unreadable) == 1, [r["key"] for r in rows]
    assert "UNKNOWN, not clean" in unreadable[0]["detail"]
    assert "404" in unreadable[0]["detail"], (
        "the row must carry the reason it could not look, or the next reader "
        "cannot tell a permissions hole from an outage"
    )
    # And the denominator is stated rather than implied.
    assert "12/13 repos" in heartbeat, heartbeat
    assert "1 repo(s) unreadable" in heartbeat, heartbeat


def test_one_unreadable_repository_does_not_blank_the_twelve_that_were_read() -> None:
    """The abort this replaces. The first live `.201` sweep produced no rows."""
    rows, _ = probe_module._rows_from_report(_report(), MIN_RUNS_FOR_CRITICAL)
    repos_with_findings = {
        r["key"].split("/")[1] for r in rows if r["key"].startswith("sched/")
    }
    assert "omni_home" in repos_with_findings
    assert len(repos_with_findings) >= 7, sorted(repos_with_findings)


def test_the_same_probe_stays_silent_on_the_workflows_that_are_fine() -> None:
    """THE DISCRIMINATOR, and it is load-bearing rather than a formality.

    A probe that emitted a row for every workflow it saw would replay the
    incident above perfectly and be useless: seventy-four rows every tick is a
    channel nobody reads, which is the failure mode this ticket's own third
    defect names -- roughly 2,160 annotations a day, muted on day one.

    Driven over the SAME captured bytes, the probe must produce a row for the
    twenty-nine above threshold and for the one unreadable repository, and
    nothing for the other forty-five workflows that ran and were fine.
    """
    report = _report()
    rows, _ = probe_module._rows_from_report(report, MIN_RUNS_FOR_CRITICAL)

    observed = sum(
        len(block.get("scheduled", {}).get("workflows", {}))
        for block in report["repos"].values()
        if "error" not in block
    )
    alerting = sum(
        len(block.get("scheduled", {}).get("alerts", []))
        for block in report["repos"].values()
        if "error" not in block
    )
    assert observed == 74, observed
    assert alerting == 29, alerting
    # 29 findings + 1 unreadable repository, and not one row more.
    assert len(rows) == alerting + 1, [r["key"] for r in rows]
    assert observed > alerting, (
        "the fixture must contain workflows that are BELOW threshold, or this "
        "discriminator asserts nothing"
    )


def test_a_single_observed_run_at_a_hundred_percent_does_not_escalate() -> None:
    """The narrow half of the discriminator, from the same bytes.

    The capture contains a workflow that fired once in the whole window and
    failed. A rate with no denominator is not evidence of a dead verification
    surface, and escalating it would page on a workflow that has run once.
    """
    report = _report()
    thin = [
        (slug, alert)
        for slug, block in report["repos"].items()
        if "error" not in block
        for alert in block["scheduled"]["alerts"]
        if alert["observed"] == alert["failures"] < MIN_RUNS_FOR_CRITICAL
    ]
    assert thin, "the fixture no longer contains a thin-denominator alert"
    rows, _ = probe_module._rows_from_report(report, MIN_RUNS_FOR_CRITICAL)
    by_key = {r["key"]: r for r in rows}
    for slug, alert in thin:
        key = f"sched/{slug.split('/')[-1]}/{alert['workflow'].rsplit('/', 1)[-1]}"
        assert by_key[key]["status"] == "WARNING", by_key[key]
