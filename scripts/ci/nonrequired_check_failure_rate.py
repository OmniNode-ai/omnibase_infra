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

SCHEDULED RUNS, ADDED IN OMN-18322
    The reads above are all pull-request-shaped: a check-run on a head. A
    scheduled workflow has no head and no pull request, so nothing above
    watches it -- and the friction trend report (2026-09-13) measured scheduled
    runs failing at 15.4% over 9,746 runs against 2.9% on pull requests, with
    nothing triaging the difference.

    For each repo already watched, this lists every ACTIVE workflow, reads its
    `event == schedule` runs created within the trailing window (policy:
    `scheduled_window_days`), and alerts per workflow when the failure rate
    exceeds `scheduled_failure_threshold_pct`. Same conclusions count as a
    failure as the check-run path (`FAILING_CONCLUSIONS`), same policy file,
    same report artifact, same annotation/summary/Slack destinations -- no new
    channel, no new scheduler, per the plan this rides.

    A workflow with zero scheduled runs in the window is silent, not zero
    percent: a rate with no denominator is not evidence of anything, so it is
    excluded rather than reported as a 0% or 100% control it did not earn.
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
from datetime import UTC, datetime, timedelta, timezone
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


@dataclass(frozen=True)
class ScheduledAlert:
    repo: str
    workflow: str
    failures: int
    observed: int
    rate_pct: float
    last_failure_url: str

    @property
    def detail(self) -> str:
        return (
            f"{self.repo}: scheduled workflow '{self.workflow}' failed "
            f"{self.rate_pct:.1f}% ({self.failures}/{self.observed}) over the "
            f"trailing window -- last failure {self.last_failure_url}"
        )


def evaluate_scheduled(
    repo: str,
    workflow_path: str,
    runs: list[dict[str, Any]],
    threshold_pct: float,
) -> ScheduledAlert | None:
    """One workflow's scheduled-run failure rate over an already-windowed list.

    ``runs`` is one workflow's ``event == schedule`` runs, already filtered to
    the trailing window by the caller (the GitHub `created` query qualifier).
    Silent (returns ``None``) on zero observed runs -- a rate with no
    denominator is not evidence, not a 0% control it did not earn -- and on a
    rate at or below ``threshold_pct``.
    """
    observed = len(runs)
    if observed == 0:
        return None
    failing = [
        run
        for run in runs
        if isinstance(run, dict)
        and str(run.get("conclusion") or "") in FAILING_CONCLUSIONS
    ]
    if not failing:
        return None
    rate_pct = 100.0 * len(failing) / observed
    if rate_pct <= threshold_pct:
        return None
    last = max(failing, key=lambda run: str(run.get("created_at") or ""))
    return ScheduledAlert(
        repo=repo,
        workflow=workflow_path,
        failures=len(failing),
        observed=observed,
        rate_pct=rate_pct,
        last_failure_url=str(last.get("html_url") or ""),
    )


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


def active_workflows(slug: str, token: str | None) -> list[dict[str, Any]]:
    """Every ACTIVE workflow declared in the repo (OMN-18322).

    Disabled workflows (manually, or by inactivity) are excluded: a `created`
    filter against one returns zero runs anyway, so including it would only
    spend an API call for no possible finding.
    """
    payload = _gh_object(f"repos/{slug}/actions/workflows?per_page=100", token)
    workflows = payload.get("workflows")
    if not isinstance(workflows, list):
        raise RuntimeError(
            f"repos/{slug}/actions/workflows returned no 'workflows' array"
        )
    return [w for w in workflows if isinstance(w, dict) and w.get("state") == "active"]


def scheduled_runs_for_workflow(
    slug: str,
    workflow_id: int,
    since_date: str,
    token: str | None,
    max_pages: int = 15,
) -> list[dict[str, Any]]:
    """One workflow's ``event == schedule`` runs since ``since_date`` (YYYY-MM-DD).

    The `created=>=<date>` qualifier filters server-side, so this scopes to one
    workflow's own run history rather than paging through the repo's combined
    run list -- which for a repo running a 10-minute cron hits GitHub's ~1000-
    result pagination ceiling in under two days and can never reach a 7-day
    window (measured live on omnibase_infra's own zombie-detector schedule
    while building this).
    """
    runs: list[dict[str, Any]] = []
    page = 1
    while page <= max_pages:
        payload = _gh_object(
            f"repos/{slug}/actions/workflows/{workflow_id}/runs"
            f"?event=schedule&created=>={since_date}&per_page=100&page={page}",
            token,
        )
        batch = payload.get("workflow_runs")
        if not isinstance(batch, list):
            raise RuntimeError(
                f"repos/{slug}/actions/workflows/{workflow_id}/runs returned no "
                "'workflow_runs' array"
            )
        runs.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return runs


def post_slack_alert(alerts: list[Any]) -> None:
    """Post through the SAME channel secret the fleet canary already uses.

    ``alerts`` mixes ``Alert`` (non-required check) and ``ScheduledAlert``
    (OMN-18322) instances -- both expose ``.detail`` and nothing else here
    depends on which kind it is.
    """
    token = os.environ.get("SLACK_BOT_TOKEN")
    channel = os.environ.get("SLACK_CHANNEL_ID")
    if not token or not channel:
        # OMN-18942: this is NOT the fleet going dark. No chat secret exists at
        # organisation or repository scope and deliberately none is being added
        # -- a fourth copy of a token that already works elsewhere. The
        # delivering surface for these same findings is the `.201` system
        # reporter, which runs this evaluator hourly through
        # scripts/omninode-fleet-failure-probe.py and posts with the bot token
        # that host already holds. Say where delivery happens, so this line is
        # not read as "nobody is being told".
        print(
            "[nonrequired-checks] Slack not configured here; annotation + "
            "artifact only. Delivery of these findings is the .201 system "
            "reporter (omninode-fleet-failure-probe.py, OMN-18942), not this job."
        )
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


def render_scheduled(alerts: list[ScheduledAlert]) -> None:
    """OMN-18322: the scheduled-run counterpart of :func:`render`."""
    for alert in alerts:
        print(f"::warning title=Scheduled workflow failing::{alert.detail}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return
    with open(summary, "a", encoding="utf-8") as handle:
        handle.write("\n## Scheduled workflow failure rate (OMN-18322)\n\n")
        if not alerts:
            handle.write(
                "No scheduled workflow is failing above the trailing-window "
                "threshold.\n"
            )
            return
        handle.write(
            "| repo | workflow | fail % | failures/observed | last failure |\n"
        )
        handle.write(
            "|------|----------|--------|--------------------|--------------|\n"
        )
        for alert in alerts:
            handle.write(
                f"| {alert.repo} | {alert.workflow} | {alert.rate_pct:.1f}% | "
                f"{alert.failures}/{alert.observed} | {alert.last_failure_url} |\n"
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
    for key in (
        "failure_threshold",
        "heads_observed",
        "scheduled_failure_threshold_pct",
        "scheduled_window_days",
        # OMN-18942. The fleet repository list is a policy fact, not a run-step
        # literal. It is REQUIRED rather than defaulted for the same reason as
        # every key above it: a silently-defaulted list is how this alerter read
        # three repositories for a month while reporting on "the fleet".
        "fleet_repos",
    ):
        if key not in block:
            raise KeyError(
                f"{path} route.nonrequired_check_alert is missing required key {key!r}"
            )
    return block


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default="OmniNode-ai")
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        help="repeatable; mutually exclusive with --repos-from-policy",
    )
    parser.add_argument(
        "--repos-from-policy",
        action="store_true",
        help=(
            "read the repository list from the policy's route."
            "nonrequired_check_alert.fleet_repos, so every caller watches the "
            "same fleet (OMN-18942)"
        ),
    )
    parser.add_argument(
        "--no-scheduled",
        action="store_true",
        help=(
            "evaluate the pull-request check-run half only, skipping the "
            "scheduled-run sweep. The scheduled sweep has exactly ONE caller "
            "on purpose (the `.201` system reporter): two surfaces evaluating "
            "the same scheduled runs at different cadences into different "
            "destinations is the divergence a shared evaluator exists to "
            "prevent, and only one of them has somewhere to deliver "
            "(OMN-18942)."
        ),
    )
    parser.add_argument(
        "--scheduled-only",
        action="store_true",
        help=(
            "evaluate scheduled-run failure rates only, skipping the "
            "pull-request check-run half. The check-run half reads branch "
            "protection, which needs an Administration-scoped token; the "
            "`.201` reporter that consumes this has no such token and no use "
            "for pull-request heads, and requiring one there would widen a "
            "host credential for a result that caller discards (OMN-18942)."
        ),
    )
    parser.add_argument("--branch", default="dev")
    parser.add_argument("--heads", type=int, default=10)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument(
        "--report", type=Path, default=Path("nonrequired-check-report.json")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    policy = load_policy(args.policy)
    if args.scheduled_only and args.no_scheduled:
        parser.error("--scheduled-only and --no-scheduled are mutually exclusive")
    if args.repos_from_policy:
        if args.repo:
            parser.error("--repo and --repos-from-policy are mutually exclusive")
        repos = [str(r) for r in policy["fleet_repos"]]
    else:
        repos = list(args.repo)
    if not repos:
        parser.error("no repositories selected: pass --repo or --repos-from-policy")
    threshold = int(policy["failure_threshold"])
    umbrella: dict[str, list[str]] = policy.get("umbrella_enforced_contexts") or {}
    scheduled_threshold_pct = float(policy["scheduled_failure_threshold_pct"])
    scheduled_window_days = int(policy["scheduled_window_days"])
    scheduled_since = (
        datetime.now(UTC) - timedelta(days=scheduled_window_days)
    ).strftime("%Y-%m-%d")

    token = os.environ.get("GH_TOKEN") or os.environ.get("CROSS_REPO_PAT")
    all_alerts: list[Alert] = []
    all_scheduled_alerts: list[ScheduledAlert] = []
    report: dict[str, Any] = {"schema": "nonrequired_check_report/v1", "repos": {}}

    unreadable: list[str] = []
    for repo in repos:
        slug = f"{args.owner}/{repo}"
        try:
            _evaluate_one_repo(
                args=args,
                repo=repo,
                slug=slug,
                token=token,
                threshold=threshold,
                umbrella=umbrella,
                scheduled_threshold_pct=scheduled_threshold_pct,
                scheduled_window_days=scheduled_window_days,
                scheduled_since=scheduled_since,
                all_alerts=all_alerts,
                all_scheduled_alerts=all_scheduled_alerts,
                report=report,
            )
        except Exception as exc:  # noqa: BLE001 -- see _evaluate_one_repo's docstring
            unreadable.append(repo)
            report["repos"][slug] = {"error": str(exc)[:400]}

    render(all_alerts)
    render_scheduled(all_scheduled_alerts)
    for repo in unreadable:
        print(
            f"::warning title=Fleet repository unreadable::{args.owner}/{repo} "
            "could not be read; its scheduled workflows were NOT evaluated"
        )
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    combined_alerts: list[Any] = [*all_alerts, *all_scheduled_alerts]
    if combined_alerts and not args.dry_run:
        post_slack_alert(combined_alerts)

    # Raising an alert is not this job failing (see the module docstring), but
    # failing to READ a repository is: an unevaluated repository must never be
    # indistinguishable from a clean one. The report still carries every repo
    # that WAS evaluated, so a caller that can render the partial result --
    # the `.201` probe does, as a named row per unreadable repo -- reads the
    # file rather than the exit code.
    return 1 if unreadable else 0


def _evaluate_one_repo(
    *,
    args: argparse.Namespace,
    repo: str,
    slug: str,
    token: str | None,
    threshold: int,
    umbrella: dict[str, list[str]],
    scheduled_threshold_pct: float,
    scheduled_window_days: int,
    scheduled_since: str,
    all_alerts: list[Alert],
    all_scheduled_alerts: list[ScheduledAlert],
    report: dict[str, Any],
) -> None:
    """One repository's evaluation, extracted so ONE can fail without the rest.

    WHY THIS IS A FUNCTION (OMN-18942). Until this split, any read failure
    anywhere in the loop aborted the whole sweep, so a single repository the
    token cannot see blanked every other repository's findings. That is not
    hypothetical twice over: OMN-18254 spent three days on exactly it -- "the
    repo loop hits omniweb second, so omnimarket was never reached either" --
    and it recurred on `.201` on 2026-09-20, where the host token reads twelve
    of the thirteen fleet repositories and returns 404 on `omni_home`, the one
    repository whose red scheduled workflow this ticket names.

    The caller records the failure under `repos[slug]["error"]` and exits
    non-zero, so an unreadable repository is loud in both directions: named in
    the report for a renderer, and a red exit for CI.
    """
    required: set[str] = set()
    pages: list[dict[str, Any]] = []
    alerts: list[Alert] = []
    if not args.scheduled_only:
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

    scheduled_report: dict[str, Any] = {}
    workflows_iter = [] if args.no_scheduled else active_workflows(slug, token)
    for workflow in workflows_iter:
        workflow_id = workflow.get("id")
        workflow_path = str(workflow.get("path") or workflow.get("name") or "")
        if not isinstance(workflow_id, int) or not workflow_path:
            continue
        runs = scheduled_runs_for_workflow(slug, workflow_id, scheduled_since, token)
        scheduled_alert = evaluate_scheduled(
            slug, workflow_path, runs, scheduled_threshold_pct
        )
        if runs:
            failing = sum(
                1
                for run in runs
                if isinstance(run, dict)
                and str(run.get("conclusion") or "") in FAILING_CONCLUSIONS
            )
            scheduled_report[workflow_path] = {
                "observed": len(runs),
                "failures": failing,
                "rate_pct": round(100.0 * failing / len(runs), 1),
            }
        if scheduled_alert is not None:
            all_scheduled_alerts.append(scheduled_alert)

    report["repos"][slug] = {
        "heads_observed": len(pages),
        "required_contexts": sorted(required),
        "alerts": [
            {"check": a.check, "failures": a.failures, "observed": a.observed}
            for a in alerts
        ],
        "scheduled": {
            "window_days": scheduled_window_days,
            "threshold_pct": scheduled_threshold_pct,
            "workflows": scheduled_report,
            "alerts": [
                {
                    "workflow": a.workflow,
                    "failures": a.failures,
                    "observed": a.observed,
                    "rate_pct": a.rate_pct,
                    "last_failure_url": a.last_failure_url,
                }
                for a in all_scheduled_alerts
                if a.repo == slug
            ],
        },
    }


if __name__ == "__main__":
    sys.exit(main())
