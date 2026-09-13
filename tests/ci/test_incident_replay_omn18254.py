# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Incident replay: a non-required check failing on every pull request (OMN-18254).

THE INCIDENT
    `occ-companion-effect / Publish occ-companion-effect command` failed on the
    heads of three consecutive pull requests on the web repository, on missing
    broker credentials. That repository's `dev` requires exactly one context,
    `merge-hold-gate / evaluate`, so the failing check blocked nothing. It was
    reported red on three separate pull requests before anyone asked whether it
    meant companions were never publishing (OMN-18217).

    The fixtures are the verbatim `commits/{sha}/check-runs` responses for those
    three heads, plus the head of the pull request that fixed it, where the same
    check concludes `success`.

THE REPLAY
    The real evaluator is driven over the three failing captures with the
    repository's real required set, and must raise exactly one alert naming the
    check and its count.

THE DISCRIMINATOR IS MANDATORY
    An alerter that is always silent and an alerter that is correct look
    identical from a green run -- the ticket says so in as many words. The same
    evaluator is driven over two of the failing heads plus the fixed one, two
    failures against a threshold of three, and must raise nothing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD = REPO_ROOT / "scripts" / "ci" / "nonrequired_check_failure_rate.py"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "omn18254"

INCIDENT_CAPTURE = FIXTURES / "omniweb-checkruns-8024bb78.json.captured"
SECOND_FAILING = FIXTURES / "omniweb-checkruns-87adf60a.json.captured"
THIRD_FAILING = FIXTURES / "omniweb-checkruns-930fcc5d.json.captured"
FIXED_HEAD = FIXTURES / "omniweb-checkruns-1ea89df5.json.captured"

CHECK = "occ-companion-effect / Publish occ-companion-effect command"
# Read live from omniweb's branch protection: one context, and it is not this.
OMNIWEB_REQUIRED = {"merge-hold-gate / evaluate"}
THRESHOLD = 3


def _guard() -> Any:
    spec = importlib.util.spec_from_file_location(
        "nonrequired_check_failure_rate_replay", GUARD
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _page(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_captures_record_the_incident_before_anything_is_replayed() -> None:
    """Without this the fixtures are only a quotation.

    Prove from the bytes that the check failed on three heads and passed on the
    fourth, and that none of those heads carried it as a required context.
    """
    for capture in (INCIDENT_CAPTURE, SECOND_FAILING, THIRD_FAILING):
        page = _page(capture)
        assert [r["conclusion"] for r in page["check_runs"] if r["name"] == CHECK] == [
            "failure"
        ], f"{capture.name} no longer records the failure"
    assert [
        r["conclusion"] for r in _page(FIXED_HEAD)["check_runs"] if r["name"] == CHECK
    ] == ["success"]
    assert CHECK not in OMNIWEB_REQUIRED


def test_the_real_evaluator_raises_one_alert_naming_the_check_and_its_count() -> None:
    """R5, false_green: nothing was watching, so nothing said anything."""
    module = _guard()
    alerts = module.evaluate(
        "OmniNode-ai/omniweb",
        [_page(INCIDENT_CAPTURE), _page(SECOND_FAILING), _page(THIRD_FAILING)],
        OMNIWEB_REQUIRED,
        THRESHOLD,
    )
    assert len(alerts) == 1, [a.check for a in alerts]
    assert alerts[0].check == CHECK
    assert alerts[0].failures == 3
    assert CHECK in alerts[0].detail
    assert "3 of 3" in alerts[0].detail


def test_the_same_evaluator_stays_silent_below_threshold() -> None:
    """The discriminator. Two failing heads and the head that fixed it."""
    module = _guard()
    alerts = module.evaluate(
        "OmniNode-ai/omniweb",
        [_page(INCIDENT_CAPTURE), _page(SECOND_FAILING), _page(FIXED_HEAD)],
        OMNIWEB_REQUIRED,
        THRESHOLD,
    )
    assert alerts == [], [a.detail for a in alerts]
