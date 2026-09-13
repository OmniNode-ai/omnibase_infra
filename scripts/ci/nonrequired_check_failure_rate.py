# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Alert on a non-required check that keeps failing (OMN-18254).

THE PROBLEM
    A check that is not required can fail indefinitely and nobody learns. The
    companion-effect job on the web repository failed 9 of 10 recent pull-request
    runs on missing broker credentials, and it was reported red on three separate
    pull requests before anyone asked whether it meant companions were never
    publishing. Nothing was watching, because nothing blocks.

WHAT "NON-REQUIRED" MEANS HERE, and why the naive reading is wrong
    "Absent from `required_status_checks`" is NOT the same as "cannot block a
    merge". `omnibase_infra` requires exactly ONE context on `dev`, the
    `CI Summary` umbrella, and that poller fails on any job in its own run that
    concludes badly. Dozens of checks are therefore enforced while appearing
    nowhere in branch protection.

    Reporting those as unwatched would bury the real finding in noise, and an
    alerter nobody reads is the failure mode OMN-16030 already paid for once. So
    the required set is branch protection UNION the contexts the policy declares
    umbrella-enforced, and the policy carries the reason for each.

    The honest limit, stated rather than implied: this reads DECLARED
    enforcement. A context enforced by a mechanism nobody wrote down still reads
    as non-required here, and the fix is to write it down.

THE ALERT PATH IS INHERITED, NOT CHOSEN
    Annotations, the step summary, a JSON report artifact, and a best-effort
    Slack post through the same two secrets the fleet canary already uses. No
    new channel and no new scheduler: this rides an existing scheduled workflow.
    A missing Slack token degrades to annotation plus artifact rather than
    failing the monitor, for the same reason `runner_saturation_record.py` does
    it that way -- a watcher that goes red because Slack was down stops being
    read.

EXIT CODE IS ALWAYS 0 ON A SUCCESSFUL EVALUATION
    Raising an alert is not this job failing. The job fails only when it could
    not evaluate, which keeps "the alerter is broken" distinguishable from "the
    alerter fired" -- the same distinction phase 3 exists to enforce everywhere
    else.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FAILING_CONCLUSIONS = frozenset({"failure", "timed_out", "action_required"})


@dataclass(frozen=True)
class Alert:
    repo: str
    check: str
    failures: int
    observed: int

    @property
    def detail(self) -> str:
        return (
            f"{self.repo}: non-required check '{self.check}' failed "
            f"{self.failures} of {self.observed} recent pull-request heads"
        )


def required_contexts(payload: dict[str, Any]) -> set[str]:
    """The contexts branch protection itself declares required."""
    contexts = payload.get("contexts")
    if isinstance(contexts, list):
        return {str(c) for c in contexts}
    checks = payload.get("checks")
    if isinstance(checks, list):
        return {str(c.get("context")) for c in checks if isinstance(c, dict)}
    return set()


def count_failures(
    check_run_pages: list[dict[str, Any]], required: set[str]
) -> tuple[Counter[str], int]:
    """Count failures per non-required check name across one page per head.

    Each element of ``check_run_pages`` is one head's verbatim
    ``commits/{sha}/check-runs`` response. Returns the counter and the number of
    heads observed, because a count without its denominator is not a rate.
    """
    failures: Counter[str] = Counter()
    for page in check_run_pages:
        runs = page.get("check_runs")
        if not isinstance(runs, list):
            continue
        seen: set[str] = set()
        for run in runs:
            if not isinstance(run, dict):
                continue
            name = str(run.get("name") or "")
            if not name or name in required or name in seen:
                continue
            if str(run.get("conclusion") or "") in FAILING_CONCLUSIONS:
                failures[name] += 1
                # One head contributes at most one failure per check name, so a
                # re-run storm on a single pull request cannot manufacture a
                # rate on its own.
                seen.add(name)
    return failures, len(check_run_pages)


def evaluate(
    repo: str,
    check_run_pages: list[dict[str, Any]],
    required: set[str],
    threshold: int,
) -> list[Alert]:
    """Return one alert per non-required check at or above ``threshold``."""
    failures, observed = count_failures(check_run_pages, required)
    return [
        Alert(repo=repo, check=name, failures=count, observed=observed)
        for name, count in sorted(failures.items())
        if count >= threshold
    ]


def _gh_json(path: str, token: str | None) -> Any:
    env = dict(os.environ)
    if token:
        env["GH_TOKEN"] = token
    result = subprocess.run(
        ["gh", "api", path],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        # NEVER suppressed: a sweep whose stderr is discarded returns zero rows
        # and reads exactly like a clean bill of health.
        raise RuntimeError(f"gh api {path} failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


# The two shapes are separated deliberately rather than left to duck typing.
# `/pulls` returns a LIST and `/check-runs` returns an OBJECT; iterating the
# wrong one silently yields dict KEYS, which are strings, and every `.get` on
# them would raise far from the cause -- or, worse, an empty list would read as
# "no pull requests" rather than "I read the wrong shape". Asserting the shape
# at the boundary is the same fail-loud posture the rest of this module takes.
def _gh_object(path: str, token: str | None) -> dict[str, Any]:
    payload = _gh_json(path, token)
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"gh api {path} returned {type(payload).__name__}, expected an object"
        )
    return payload


def _gh_array(path: str, token: str | None) -> list[Any]:
    payload = _gh_json(path, token)
    if not isinstance(payload, list):
        raise RuntimeError(
            f"gh api {path} returned {type(payload).__name__}, expected an array"
        )
    return payload


def post_slack_alert(alerts: list[Alert]) -> None:
    """Post through the SAME channel secret the fleet canary already uses."""
    token = os.environ.get("SLACK_BOT_TOKEN")
    channel = os.environ.get("SLACK_CHANNEL_ID")
    if not token or not channel:
        print("[nonrequired-checks] Slack not configured; annotation + artifact only")
        return
    detail = " | ".join(a.detail for a in alerts)
    payload = json.dumps(
        {"channel": channel, "text": f"*[NON-REQUIRED CHECK FAILING]* {detail}"}
    ).encode("utf-8")
    # fmt: off
    slack_url = "https://slack.com/api/chat.postMessage"  # url-authority-ok: fixed public Slack Web API method, no ONEX routing authority -- same contract as runner_saturation_record.py's existing chat.postMessage call
    # fmt: on
    request = urllib.request.Request(  # noqa: S310 -- fixed https Slack literal above
        slack_url,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            response.read()
    except Exception as exc:  # noqa: BLE001 -- evidence survives a dead webhook
        print(f"[nonrequired-checks] Slack post failed, continuing: {exc}")


def render(alerts: list[Alert]) -> None:
    for alert in alerts:
        print(f"::warning title=Non-required check failing::{alert.detail}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return
    with open(summary, "a", encoding="utf-8") as handle:
        handle.write("## Non-required check failure rate (OMN-18254)\n\n")
        if not alerts:
            handle.write("No non-required check is failing above threshold.\n")
            return
        handle.write("| repo | check | failures | heads observed |\n")
        handle.write("|------|-------|----------|----------------|\n")
        for alert in alerts:
            handle.write(
                f"| {alert.repo} | {alert.check} | {alert.failures} | {alert.observed} |\n"
            )


def load_policy(path: Path) -> dict[str, Any]:
    """Read the threshold and the umbrella-enforced contexts.

    Fails rather than defaulting (rule 8): a silently-defaulted threshold is how
    an alerter ends up either permanently silent or permanently noisy.
    """
    import yaml

    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    route = (doc or {}).get("route")
    if not isinstance(route, dict):
        raise RuntimeError(f"{path} is missing the required 'route:' section")
    block = route.get("nonrequired_check_alert")
    if not isinstance(block, dict):
        raise RuntimeError(
            f"{path} is missing the required 'route.nonrequired_check_alert:' section"
        )
    for key in ("failure_threshold", "heads_observed"):
        if key not in block:
            raise KeyError(
                f"{path} route.nonrequired_check_alert is missing required key {key!r}"
            )
    return block


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default="OmniNode-ai")
    parser.add_argument("--repo", action="append", default=[], required=True)
    parser.add_argument("--branch", default="dev")
    parser.add_argument("--heads", type=int, default=10)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument(
        "--report", type=Path, default=Path("nonrequired-check-report.json")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    policy = load_policy(args.policy)
    threshold = int(policy["failure_threshold"])
    umbrella: dict[str, list[str]] = policy.get("umbrella_enforced_contexts") or {}

    token = os.environ.get("GH_TOKEN") or os.environ.get("CROSS_REPO_PAT")
    all_alerts: list[Alert] = []
    report: dict[str, Any] = {"schema": "nonrequired_check_report/v1", "repos": {}}

    for repo in args.repo:
        slug = f"{args.owner}/{repo}"
        protection = _gh_object(
            f"repos/{slug}/branches/{args.branch}/protection/required_status_checks",
            token,
        )
        required = required_contexts(protection) | set(umbrella.get(repo) or [])
        pulls = _gh_array(
            f"repos/{slug}/pulls?state=all&sort=updated&direction=desc&per_page={args.heads}",
            token,
        )
        heads = [
            str(p.get("head", {}).get("sha")) for p in pulls if isinstance(p, dict)
        ]
        pages = [
            _gh_object(f"repos/{slug}/commits/{sha}/check-runs?per_page=100", token)
            for sha in heads
            if sha and sha != "None"
        ]
        alerts = evaluate(slug, pages, required, threshold)
        all_alerts.extend(alerts)
        report["repos"][slug] = {
            "heads_observed": len(pages),
            "required_contexts": sorted(required),
            "alerts": [
                {"check": a.check, "failures": a.failures, "observed": a.observed}
                for a in alerts
            ],
        }

    render(all_alerts)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    if all_alerts and not args.dry_run:
        post_slack_alert(all_alerts)

    # Raising an alert is not this job failing. See the module docstring.
    return 0


if __name__ == "__main__":
    sys.exit(main())
