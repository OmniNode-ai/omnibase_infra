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
    Annotations, the step summary, a JSON report artifact, and a Slack post
    through the same two secrets the fleet canary already uses. No new channel
    and no new scheduler: this rides an existing scheduled workflow.

    A missing Slack credential USED to degrade to annotation plus artifact and
    leave the run green, on the argument that a watcher which reddens because
    Slack was down stops being read. Measured 2026-09-21: no chat secret of any
    name exists at OmniNode-ai organisation scope or in this repository's
    scope, so that branch was not a rare outage path -- it was the only path,
    and every finding this job produced was delivered to nobody while the job
    reported success. That is the defect class the ticket is about, reproduced
    inside its own fix. The delivery path now REFUSES: findings plus no
    destination exits non-zero with a one-line reason.

    The distinction the original argument was protecting is kept exactly. A
    transient Slack failure WITH a credential present still logs and carries
    on, unchanged. A clean sweep with no credential still exits 0, because
    nothing failed to reach anybody. Only "there is a finding and there is
    nowhere to send it" is red.

RAISING AN ALERT IS NOT THIS JOB FAILING
    The job fails when it could not EVALUATE, and when it could not DELIVER a
    finding it did raise. Both are "the alerter is broken"; neither is "the
    alerter fired", which stays exit 0 -- the same distinction phase 3 exists
    to enforce everywhere else.

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
import io
import json
import os
import subprocess
import sys
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass, field
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

    @property
    def key(self) -> str:
        """The dedup identity (OMN-18942): one finding per (repo, check)."""
        return f"check|{self.repo}|{self.check}"


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

    @property
    def key(self) -> str:
        """The dedup identity (OMN-18942): one finding per (repo, workflow)."""
        return f"scheduled|{self.repo}|{self.workflow}"


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


def _gh_bytes(path: str, token: str | None) -> bytes:
    """Raw ``gh api`` response body. Used for the artifact zip (OMN-18942)."""
    env = dict(os.environ)
    if token:
        env["GH_TOKEN"] = token
    result = subprocess.run(
        ["gh", "api", path],
        capture_output=True,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        # NEVER suppressed: a sweep whose stderr is discarded returns zero rows
        # and reads exactly like a clean bill of health.
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"gh api {path} failed: {stderr}")
    return bytes(result.stdout)


def _gh_json(path: str, token: str | None) -> Any:
    return json.loads(_gh_bytes(path, token).decode("utf-8"))


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


class AlertDestinationMissingError(RuntimeError):
    """Findings were raised and no destination credential resolves.

    Carried as an exception rather than a return code so it cannot be dropped
    at the call site by a caller that ignores the value -- which is how the
    branch it replaces stayed invisible for as long as it did. Its message is
    ONE line and names the credential REFERENCE only; no value of either
    secret is ever read into it.
    """


def resolve_destination(finding_count: int) -> tuple[str, str]:
    """The channel credential pair, or a refusal naming what is missing.

    Split out of the post (OMN-18942) so the refusal still fires when every
    finding is suppressed by the dedup window: a destination that disappears
    must stay red whatever the prior alert state says.
    """
    token = os.environ.get("SLACK_BOT_TOKEN")
    channel = os.environ.get("SLACK_CHANNEL_ID")
    if token and channel:
        return token, channel
    # OMN-18942. The line this replaced said delivery happens on the `.201`
    # system reporter instead, and returned. That sentence is true of the
    # SCHEDULED half, which moved there behind `--no-scheduled`. It is not
    # true of the pull-request check-run half, which still raises findings
    # HERE and had nowhere to put them -- so the job went green having told
    # nobody. A comment naming another surface is not a destination.
    #
    # Which reference is missing is named, because "not configured" sends a
    # reader to look at both. No value is read into the message.
    missing = ", ".join(
        name
        for name, present in (
            ("SLACK_BOT_TOKEN", bool(token)),
            ("SLACK_CHANNEL_ID", bool(channel)),
        )
        if not present
    )
    raise AlertDestinationMissingError(
        f"{finding_count} non-required check finding(s) raised and no "
        f"destination resolves: {missing} is unset or empty in this job's "
        "environment; the findings were written to the report artifact but "
        "delivered to nobody"
    )


def post_slack_alert(alerts: list[Any], *, text: str | None = None) -> bool:
    """Post through the SAME channel secret the fleet canary already uses.

    ``alerts`` mixes ``Alert`` (non-required check) and ``ScheduledAlert``
    (OMN-18322) instances -- both expose ``.detail`` and nothing else here
    depends on which kind it is. ``text`` overrides the rendered message.

    Returns True only when Slack acknowledged the post (OMN-18942). Slack
    answers a refused post with HTTP 200 and ``"ok": false``; counting that as
    delivered would record a post nobody received, and the dedup window would
    then suppress the finding for a whole window. A failed post is logged and
    the run carries on, exactly as before -- it is just not recorded as sent.
    """
    token, channel = resolve_destination(len(alerts))
    if text is None:
        detail = " | ".join(a.detail for a in alerts)
        text = f"*[NON-REQUIRED CHECK FAILING]* {detail}"
    payload = json.dumps({"channel": channel, "text": text}).encode("utf-8")
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
            body = response.read()
    except Exception as exc:  # noqa: BLE001 -- evidence survives a dead webhook
        print(f"[nonrequired-checks] Slack post failed, continuing: {exc}")
        return False
    try:
        answer = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        answer = {}
    if not isinstance(answer, dict) or answer.get("ok") is not True:
        error = answer.get("error") if isinstance(answer, dict) else None
        print(
            "[nonrequired-checks] Slack post not acknowledged, continuing: "
            f"{error or 'no ok:true in the response'}"
        )
        return False
    return True


# ---------------------------------------------------------------------------
# ALERT DEDUP (OMN-18942)
#
# Measured 2026-09-23: once the channel secrets existed (10:43Z), every
# scheduled run re-posted the same unchanged findings -- runs 35850665037,
# 35851297482, 35852673260 -- because nothing remembered what had been sent.
#
# THE RULE. A finding (keyed by repo and check, or repo and workflow) whose
# failure count is unchanged or lower than the count last POSTED, inside the
# window measured from that post, is suppressed. A new key, a higher count, or
# an expired window posts. A key that stops appearing is resolved: recorded,
# not posted, and dropped from the state so a recurrence reads as new.
#
# THE STATE LIVES IN THE REPORT ARTIFACT. Each run's report carries
# `alert_state`, and the next run downloads the most recent earlier run's
# report through the Actions API with the job's already-minted App token (it
# carries `actions: read` on this repository; nothing new is minted or
# widened). So the thing that suppresses is the same file that records the
# suppression -- there is no second store to drift from the evidence.
#
# FAILS OPEN TO THE CHANNEL. Prior state that is absent or unreadable
# suppresses NOTHING: every finding posts, the message carries a marker, and
# the summary says why. A duplicate post is noise; a wrongly suppressed one is
# the silent-gate defect this ticket exists to remove.
# ---------------------------------------------------------------------------

ALERT_STATE_SCHEMA = "nonrequired_check_alert_state/v1"
REPORT_ARTIFACT = "nonrequired-check-report"
REPORT_MEMBER = "nonrequired-check-report.json"
# How many of this workflow's most recent completed runs are searched for a
# report. A run that died before its upload has none, and the next-older one
# still holds correct state about what was posted. 20 runs is roughly three
# hours at the ten-minute cadence; finding nothing in that span reads as
# ABSENT, which posts.
PRIOR_RUN_SEARCH_LIMIT = 20

_STAMP = "%Y-%m-%dT%H:%M:%SZ"


def _now() -> datetime:
    return datetime.now(UTC)


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime(_STAMP)


def _parse_stamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"timestamp {value!r} is not a string")
    return datetime.strptime(value, _STAMP).replace(tzinfo=UTC)


@dataclass(frozen=True)
class PriorAlertState:
    """What the previous run says was posted, and whether it could be read.

    ``status`` is ``read``, ``absent`` (no earlier state exists, e.g. the first
    run after this change), ``unreadable`` (it exists and could not be read),
    or ``not_configured`` (no prior-state source was requested). Only ``read``
    can suppress anything.
    """

    status: str
    reason: str
    entries: dict[str, dict[str, Any]]
    source_run_id: int | None = None


def parse_prior_report(
    payload: Any, source_run_id: int | None = None
) -> PriorAlertState:
    """Read ``alert_state`` out of one earlier report. Never raises."""
    if not isinstance(payload, dict):
        return PriorAlertState(
            "unreadable",
            f"prior report is {type(payload).__name__}, not an object",
            {},
            source_run_id,
        )
    if "alert_state" not in payload:
        return PriorAlertState(
            "absent",
            "prior report predates alert state (no alert_state key)",
            {},
            source_run_id,
        )
    state = payload["alert_state"]
    if not isinstance(state, dict) or state.get("schema") != ALERT_STATE_SCHEMA:
        schema = state.get("schema") if isinstance(state, dict) else None
        return PriorAlertState(
            "unreadable",
            f"prior alert_state schema is {schema!r}, expected {ALERT_STATE_SCHEMA!r}",
            {},
            source_run_id,
        )
    entries = state.get("entries")
    if not isinstance(entries, dict):
        return PriorAlertState(
            "unreadable", "prior alert_state has no entries object", {}, source_run_id
        )
    clean: dict[str, dict[str, Any]] = {}
    for key, entry in entries.items():
        try:
            if not isinstance(entry, dict):
                raise ValueError("entry is not an object")
            posted = entry["posted_failures"]
            if not isinstance(posted, int) or isinstance(posted, bool):
                raise ValueError(f"posted_failures {posted!r} is not an integer")
            _parse_stamp(entry["last_posted_at"])
        except (KeyError, ValueError) as exc:
            return PriorAlertState(
                "unreadable",
                f"prior alert_state entry {key!r} is malformed: {exc}",
                {},
                source_run_id,
            )
        clean[str(key)] = {
            "posted_failures": posted,
            "last_posted_at": entry["last_posted_at"],
        }
    return PriorAlertState("read", "", clean, source_run_id)


def fetch_prior_alert_state(
    repo_slug: str | None,
    workflow_file: str,
    current_run_id: str | None,
    token: str | None,
) -> PriorAlertState:
    """The newest earlier run's report, from this workflow's own run history.

    Never raises: every failure is an ``unreadable`` state carrying its reason,
    and an unreadable state posts every finding.
    """
    if not repo_slug:
        return PriorAlertState(
            "unreadable", "GITHUB_REPOSITORY is unset; cannot locate prior runs", {}
        )
    try:
        runs_payload = _gh_json(
            f"repos/{repo_slug}/actions/workflows/{workflow_file}/runs"
            f"?status=completed&per_page={PRIOR_RUN_SEARCH_LIMIT}",
            token,
        )
        runs = (
            runs_payload.get("workflow_runs")
            if isinstance(runs_payload, dict)
            else None
        )
        if not isinstance(runs, list):
            raise RuntimeError(
                f"{workflow_file} runs listing has no workflow_runs array"
            )
        for run in runs:
            if not isinstance(run, dict):
                continue
            run_id = run.get("id")
            if not isinstance(run_id, int) or str(run_id) == str(current_run_id):
                continue
            listing = _gh_json(
                f"repos/{repo_slug}/actions/runs/{run_id}/artifacts"
                f"?name={REPORT_ARTIFACT}",
                token,
            )
            artifacts = listing.get("artifacts") if isinstance(listing, dict) else None
            if not isinstance(artifacts, list):
                raise RuntimeError(
                    f"run {run_id} artifact listing has no artifacts array"
                )
            usable = [
                a
                for a in artifacts
                if isinstance(a, dict)
                and a.get("name") == REPORT_ARTIFACT
                and not a.get("expired")
                and isinstance(a.get("id"), int)
            ]
            if not usable:
                continue
            archive = _gh_bytes(
                f"repos/{repo_slug}/actions/artifacts/{usable[0]['id']}/zip", token
            )
            try:
                with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
                    raw = bundle.read(REPORT_MEMBER)
                payload = json.loads(raw.decode("utf-8"))
            except (
                zipfile.BadZipFile,
                KeyError,
                UnicodeDecodeError,
                ValueError,
            ) as exc:
                return PriorAlertState(
                    "unreadable",
                    f"run {run_id} report could not be read: {str(exc)[:200]}",
                    {},
                    run_id,
                )
            return parse_prior_report(payload, run_id)
    except Exception as exc:  # noqa: BLE001 -- unreadable posts, it never suppresses
        return PriorAlertState(
            "unreadable", f"prior state fetch failed: {str(exc)[:300]}", {}
        )
    return PriorAlertState(
        "absent",
        f"no {REPORT_ARTIFACT} artifact in the last {PRIOR_RUN_SEARCH_LIMIT} "
        f"completed runs of {workflow_file}",
        {},
    )


@dataclass(frozen=True)
class AlertDecision:
    """Per-finding outcome of one run's dedup pass."""

    to_post: list[tuple[Any, str, int | None]]  # (alert, reason, previous_failures)
    suppressed: list[tuple[Any, dict[str, Any], datetime]]  # (alert, entry, expires)
    resolved: list[str]
    # Prior keys for a repository this run could NOT read: neither resolved nor
    # re-evaluated, carried unchanged, so a transient read failure does not
    # drop their state and re-post them as new once the repository reads again.
    carried: list[str] = field(default_factory=list)


def decide_alerts(
    alerts: list[Any],
    prior: PriorAlertState,
    now: datetime,
    window: timedelta,
    unreadable_repos: frozenset[str] = frozenset(),
) -> AlertDecision:
    """Which findings post and which are suppressed, and why.

    ``unreadable_repos`` holds ``owner/repo`` slugs this run could not read.
    """
    to_post: list[tuple[Any, str, int | None]] = []
    suppressed: list[tuple[Any, dict[str, Any], datetime]] = []
    if prior.status != "read":
        reason = (
            "prior_state_unreadable"
            if prior.status == "unreadable"
            else f"prior_state_{prior.status}"
        )
        return AlertDecision([(a, reason, None) for a in alerts], [], [])
    current_keys = {a.key for a in alerts}
    for alert in alerts:
        entry = prior.entries.get(alert.key)
        if entry is None:
            to_post.append((alert, "new", None))
            continue
        posted = int(entry["posted_failures"])
        posted_at = _parse_stamp(entry["last_posted_at"])
        expires = posted_at + window
        if alert.failures > posted:
            to_post.append((alert, "worsened", posted))
        elif posted_at > now:
            # A post recorded in the future cannot be trusted to bound a
            # window; fail open to the channel rather than suppress past it.
            to_post.append((alert, "prior_entry_in_future", posted))
        elif now >= expires:
            to_post.append((alert, "window_expired", posted))
        else:
            suppressed.append((alert, entry, expires))
    absent = [k for k in prior.entries if k not in current_keys]
    carried = sorted(k for k in absent if _key_repo(k) in unreadable_repos)
    resolved = sorted(k for k in absent if _key_repo(k) not in unreadable_repos)
    return AlertDecision(to_post, suppressed, resolved, carried)


def _key_repo(key: str) -> str:
    parts = key.split("|")
    return parts[1] if len(parts) >= 3 else ""


def next_alert_state(
    decision: AlertDecision,
    prior: PriorAlertState,
    now: datetime,
    delivered: bool,
) -> dict[str, dict[str, Any]]:
    """The state the NEXT run reads.

    A suppressed finding carries its entry unchanged, so the window stays
    anchored to the last post and a standing finding re-surfaces once per
    window. A finding due to post is recorded only when the post was
    acknowledged; otherwise its previous entry (if any) is carried, so it is
    still due -- and retried -- next run. Resolved keys are dropped.
    """
    entries: dict[str, dict[str, Any]] = {
        key: dict(prior.entries[key]) for key in decision.carried
    }
    for alert, entry, _expires in decision.suppressed:
        entries[alert.key] = dict(entry)
    for alert, _reason, _previous in decision.to_post:
        if delivered:
            entries[alert.key] = {
                "posted_failures": int(alert.failures),
                "last_posted_at": _stamp(now),
            }
        elif alert.key in prior.entries:
            entries[alert.key] = dict(prior.entries[alert.key])
    return entries


def dedup_report(
    decision: AlertDecision,
    prior: PriorAlertState,
    window: timedelta,
    delivered: bool | None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """The per-run record: what was due, suppressed and resolved, and why.

    ``due`` findings are the ones this run decided to post; whether the post
    happened is ``delivered`` (None: no post was attempted -- nothing was due,
    or a dry run).
    """
    return {
        "window_hours": window.total_seconds() / 3600,
        "dry_run": dry_run,
        "prior_state": {
            "status": prior.status,
            "reason": prior.reason,
            "source_run_id": prior.source_run_id,
        },
        "due": [
            {
                "key": alert.key,
                "failures": alert.failures,
                "reason": reason,
                "previous_failures": previous,
            }
            for alert, reason, previous in decision.to_post
        ],
        "suppressed": [
            {
                "key": alert.key,
                "failures": alert.failures,
                "posted_failures": entry["posted_failures"],
                "last_posted_at": entry["last_posted_at"],
                "window_expires_at": _stamp(expires),
            }
            for alert, entry, expires in decision.suppressed
        ],
        "resolved": list(decision.resolved),
        "carried_unreadable": list(decision.carried),
        "delivered": delivered,
    }


def dedup_message(
    decision: AlertDecision, prior: PriorAlertState, window: timedelta
) -> str:
    """The channel text for the findings that are due, with their reasons."""
    parts = []
    for alert, reason, previous in decision.to_post:
        if reason == "worsened":
            label = f"worsened {previous}->{alert.failures}"
        elif reason == "window_expired":
            label = "still failing, re-surfaced after the dedup window"
        elif reason == "new":
            label = "new"
        else:
            label = reason
        parts.append(f"{alert.detail} ({label})")
    text = "*[NON-REQUIRED CHECK FAILING]* " + " | ".join(parts)
    if prior.status == "unreadable":
        text += (
            f" -- prior alert state unreadable ({prior.reason}); posting every "
            "finding rather than risk suppressing one"
        )
    if decision.suppressed:
        hours = window.total_seconds() / 3600
        text += (
            f" -- {len(decision.suppressed)} unchanged finding(s) suppressed "
            f"inside the {hours:g}h window"
        )
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if run_id:
        text += f" -- run {run_id}, report artifact {REPORT_ARTIFACT}"
    return text


def _cell(value: str) -> str:
    """A markdown table cell; the key separator and check names may hold '|'."""
    return value.replace("|", "\\|")


def render_dedup(record: dict[str, Any]) -> None:
    """Log lines and the job-summary section for the dedup pass (OMN-18942).

    Suppression is written out per finding, in the log and in the summary:
    suppressed is a recorded outcome, never a silent drop.
    """
    prior = record["prior_state"]
    if prior["status"] == "unreadable":
        print(
            "::warning title=Alert dedup state unreadable::"
            f"{prior['reason']}; every finding is posted this run"
        )
    for item in record["suppressed"]:
        print(
            f"[nonrequired-checks] SUPPRESSED {item['key']} failures="
            f"{item['failures']} (posted {item['posted_failures']} at "
            f"{item['last_posted_at']}; window until {item['window_expires_at']})"
        )
    for item in record["due"]:
        print(f"[nonrequired-checks] DUE {item['key']} reason={item['reason']}")
    for key in record["resolved"]:
        print(f"[nonrequired-checks] RESOLVED {key} (not posted)")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return
    with open(summary, "a", encoding="utf-8") as handle:
        handle.write(
            f"\n## Alert dedup (OMN-18942), window {record['window_hours']:g}h\n\n"
        )
        status = prior["status"]
        if status == "read":
            handle.write(
                f"Prior alert state read from run {prior['source_run_id']}.\n\n"
            )
        elif status == "unreadable":
            handle.write(
                f"**prior alert state UNREADABLE** ({prior['reason']}): every "
                "finding was posted, none suppressed.\n\n"
            )
        else:
            handle.write(
                f"Prior alert state {status} ({prior['reason']}): every finding "
                "was posted, none suppressed.\n\n"
            )
        delivered = record["delivered"]
        if record["dry_run"]:
            outcome = "dry run, nothing sent and no state advanced"
        elif delivered is None:
            outcome = "nothing was due, no post made"
        elif delivered:
            outcome = "yes"
        else:
            outcome = "NO (state not advanced; retried next run)"
        handle.write(f"Delivered: {outcome}\n\n")
        if record["due"]:
            handle.write("| DUE | failures | reason |\n|---|---|---|\n")
            for item in record["due"]:
                handle.write(
                    f"| {_cell(item['key'])} | {item['failures']} | "
                    f"{item['reason']} |\n"
                )
            handle.write("\n")
        if record["suppressed"]:
            handle.write(
                "| SUPPRESSED | failures | last posted | posted at | window until |\n"
                "|---|---|---|---|---|\n"
            )
            for item in record["suppressed"]:
                handle.write(
                    f"| {_cell(item['key'])} | {item['failures']} | "
                    f"{item['posted_failures']} | {item['last_posted_at']} | "
                    f"{item['window_expires_at']} |\n"
                )
            handle.write("\n")
        if record["resolved"]:
            handle.write(
                "Resolved since the last run (not posted): "
                + ", ".join(record["resolved"])
                + "\n"
            )


def render(alerts: list[Alert], suppressed: frozenset[str] = frozenset()) -> None:
    for alert in alerts:
        # A finding suppressed by the dedup window (OMN-18942) raises no
        # annotation; its suppression is logged and tabled by render_dedup.
        if alert.key in suppressed:
            continue
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


def render_scheduled(
    alerts: list[ScheduledAlert], suppressed: frozenset[str] = frozenset()
) -> None:
    """OMN-18322: the scheduled-run counterpart of :func:`render`."""
    for alert in alerts:
        if alert.key in suppressed:
            continue
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
        # OMN-18942. The window an unchanged finding is suppressed for after it
        # was posted. Required, not defaulted, like every key above it.
        "alert_dedup_window_hours",
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
    parser.add_argument(
        "--prior-state-workflow",
        default="",
        help=(
            "workflow file (e.g. pr-ci-zombie-detector.yml) whose most recent "
            "earlier run's report artifact holds the prior alert state; read "
            "from GITHUB_REPOSITORY via the Actions API with GH_TOKEN. Without "
            "it no finding is suppressed (OMN-18942)."
        ),
    )
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
    dedup_window = timedelta(hours=float(policy["alert_dedup_window_hours"]))
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

    combined_alerts: list[Any] = [*all_alerts, *all_scheduled_alerts]

    # OMN-18942: decide what is due BEFORE rendering, so a suppressed finding
    # raises no annotation and its suppression is recorded instead.
    now = _now()
    if args.prior_state_workflow:
        prior = fetch_prior_alert_state(
            os.environ.get("GITHUB_REPOSITORY"),
            args.prior_state_workflow,
            os.environ.get("GITHUB_RUN_ID"),
            token,
        )
    else:
        prior = PriorAlertState(
            "not_configured", "no --prior-state-workflow given; nothing suppressed", {}
        )
    decision = decide_alerts(
        combined_alerts,
        prior,
        now,
        dedup_window,
        frozenset(f"{args.owner}/{repo}" for repo in unreadable),
    )
    suppressed_keys = frozenset(a.key for a, _entry, _exp in decision.suppressed)

    render(all_alerts, suppressed_keys)
    render_scheduled(all_scheduled_alerts, suppressed_keys)
    for repo in unreadable:
        print(
            f"::warning title=Fleet repository unreadable::{args.owner}/{repo} "
            "could not be read; its scheduled workflows were NOT evaluated"
        )

    # Written BEFORE the post with the not-delivered state, so the evidence
    # survives a refusal or a crash mid-post -- and a crash re-posts next run
    # rather than suppressing something that may never have been sent.
    def write_report(delivered: bool | None) -> dict[str, Any]:
        record = dedup_report(
            decision, prior, dedup_window, delivered, dry_run=args.dry_run
        )
        report["dedup"] = record
        report["alert_state"] = {
            "schema": ALERT_STATE_SCHEMA,
            "window_hours": dedup_window.total_seconds() / 3600,
            "entries": next_alert_state(decision, prior, now, bool(delivered)),
        }
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        return record

    write_report(None)
    undeliverable = False
    delivered: bool | None = None
    if combined_alerts and not args.dry_run:
        try:
            # Checked whenever there is a finding, suppressed or not, so the
            # fail-closed refusal is exactly what it was before the dedup.
            resolve_destination(len(combined_alerts))
            if decision.to_post:
                delivered = post_slack_alert(
                    [a for a, _reason, _prev in decision.to_post],
                    text=dedup_message(decision, prior, dedup_window),
                )
        except AlertDestinationMissingError as exc:
            # The report artifact is written ABOVE this point, so the evidence
            # survives the refusal: a reader gets both the red and the findings.
            print(f"::error title=Alert destination missing::{exc}")
            undeliverable = True
            delivered = False
    render_dedup(write_report(delivered))

    # Raising an alert is not this job failing (see the module docstring), but
    # failing to READ a repository is: an unevaluated repository must never be
    # indistinguishable from a clean one. The report still carries every repo
    # that WAS evaluated, so a caller that can render the partial result --
    # the `.201` probe does, as a named row per unreadable repo -- reads the
    # file rather than the exit code.
    #
    # Failing to DELIVER a raised finding is the same class: a finding nobody
    # receives and a sweep that found nothing are indistinguishable from
    # outside, which is the whole defect. It is deliberately the same exit code
    # as an unreadable repo -- both mean "do not read this run as clean" -- and
    # the annotation above says which occurred.
    return 1 if (unreadable or undeliverable) else 0


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
