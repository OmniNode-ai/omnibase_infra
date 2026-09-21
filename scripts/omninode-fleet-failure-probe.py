#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Fleet failure rows for the `.201` system reporter (OMN-18942).

WHAT THIS IS
    A probe that emits ``fleet|STATUS|key|detail`` rows on stdout, read by
    ``check_fleet_failures()`` in ``omninode-system-slack-report.sh``. It is the
    same row contract ``omninode-ci-required-context-probe.py`` already uses:
    ``row_status()`` reads the status at column 2 and ``row_key()`` de-duplicates
    on ``fleet|<key>``.

WHY IT LIVES HERE RATHER THAN IN A NEW ALERTER
    The net-negative-surface rule, which is the argument the two checks already
    folded into that reporter are here on. The fleet's failure findings inherit
    that script's Slack poster, its per-key state-change de-duplication, its
    hysteresis, its resolved-notification and its ``*/15`` cron. No new cron
    unit, no new Slack app, no new credential, no new Actions secret.

    That mattered more than usual here. The 2026-09-20 silent-gates review
    measured that NO chat secret of any name exists at GitHub Actions
    organisation scope or in any repository scope -- so the fleet's only
    failure-rate alerter (OMN-18254 / OMN-18322) runs correctly every ten
    minutes, names fifteen continuously-red scheduled workflows, and delivers
    them nowhere. The mechanism was never the problem. The last hop was.
    The `.201` host already holds a working bot token and already posts to the
    channel every fifteen minutes, so the shortest correct path is to give the
    existing evaluator a destination that exists rather than to mint a
    credential into a fourth store.

WHAT IT REPORTS

    ``scheduled-workflows``
        Every repository-scheduled workflow whose trailing-window failure rate
        is above the policy threshold, ONE row per workflow, evaluated by
        ``scripts/ci/nonrequired_check_failure_rate.py`` -- the SAME evaluator
        the GitHub Actions job runs, invoked with ``--dry-run`` so this script's
        state machine owns delivery and the two surfaces cannot diverge on what
        counts as failing. The repository list is read from the policy file by
        both callers for the same reason.

    ``lab-pass/<lane>``
        The lab-pass receipt (rule 24(b), OMN-17530) for the current ``dev``
        head, per lane, read through ``scripts/ci/lab_pass_receipt.py``'s own
        artifact reader rather than a second implementation of it. A FAIL
        receipt is the finding: the delivery gate already refuses on it, but it
        refuses silently, on a workflow run nobody opens.

WHAT IT DELIBERATELY DOES NOT REPORT
    The morning-tick fire receipts under ``.onex_state/morning-workflows/``.
    They are written on the operator's Mac and this probe runs on `.201`, which
    cannot read them; nothing syncs that directory to this host. Emitting a row
    that can only ever say "absent" would be worse than emitting none -- a check
    that cannot pass stops being read. The honest home for that finding is the
    Mac-side sender that already exists and already holds the same bot token
    (``omniclaude/plugins/onex/hooks/scripts/alert-channel.sh``, OMN-15600),
    driven by a Mac-local timer beside the ticks it watches. That is a separate
    surface on a separate host and is deliberately out of this probe's scope.

FAIL-CLOSED, AND WHAT THAT MEANS FOR A CACHE
    Every "we could not look" outcome is a row naming the reason, never
    silence, and never an empty green -- an absent finding set and an
    unevaluated finding set are the two states this whole family exists to tell
    apart.

    The scheduled sweep is CACHED AND REFRESHED OUT OF BAND, because it is the
    expensive one: one API call per active workflow per repository, measured at
    434 seconds across the thirteen-repository fleet on 2026-09-20. The
    reporter holds an exclusive lock for its whole tick and runs every fifteen
    minutes, so an inline sweep would hold that lock for seven minutes and the
    next tick would exit on it -- a health reporter skipping a tick every hour
    to look at CI. So a stale cache starts a DETACHED refresh and the tick
    serves the rows it already has, marked with their age; the next tick folds
    the finished report in. Staleness is bounded at one refresh interval plus
    one tick, against a rate computed over a seven-day window.

    A cron unit of its own would have been the other way to do it; it is not
    taken, because the reason this probe exists at all is that the fleet has
    too many alerting surfaces and not enough delivery.

    When a re-evaluation fails, the last good rows are RE-EMITTED with their age
    in the detail, alongside a ``source-unreadable`` row. Dropping them would be
    worse than stale: ``row_key()`` keys on the label, so a dropped key reads to
    the reporter's state machine as a RECOVERY and posts "resolved" for a
    finding nobody fixed. The age is in the detail rather than the key so the
    staleness is visible without re-arming the alert every tick.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess  # fixed argv, no shell
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

#: How long a scheduled-workflow sweep result is served from cache before it is
#: re-evaluated. One hour against a */15 reporter tick means three ticks in four
#: cost no API calls at all. It is not shorter because a scheduled workflow's
#: trailing-window failure RATE cannot meaningfully move inside an hour: the
#: window is seven days, so one more run shifts a rate by at most a few points.
SWEEP_INTERVAL_SECONDS = int(os.environ.get("OMNINODE_FLEET_SWEEP_INTERVAL", "3600"))

#: When the newest successful sweep is older than this, the `source-unreadable`
#: row escalates from WARNING to CRITICAL. A probe that has not managed to look
#: for a whole day is no longer a network blip.
SWEEP_STALE_CRITICAL_SECONDS = int(
    os.environ.get("OMNINODE_FLEET_SWEEP_STALE_CRITICAL", "86400")
)

#: Minimum age of the `dev` head before a MISSING lab-pass receipt may alarm.
#: Below it, "absent" and "the delivery run has not finished yet" are
#: indistinguishable -- the candidate boot gate's own rollout waits are 900s and
#: 300s and the delivery workflow runs several jobs around them, so a receipt
#: that is going to appear has appeared well inside this. Same posture, and the
#: same reason, as MIN_HEAD_AGE in omninode-ci-required-context-probe.py.
RECEIPT_ABSENCE_GRACE_SECONDS = int(
    os.environ.get("OMNINODE_FLEET_RECEIPT_GRACE", "5400")
)

#: Bounded so a hung sweep cannot wedge the 15-minute health tick. The reporter
#: bounds this probe too; this is the inner bound on the child it spawns.
SWEEP_TIMEOUT_SECONDS = int(os.environ.get("OMNINODE_FLEET_SWEEP_TIMEOUT", "900"))

# There is deliberately NO per-half kill switch here. Each would be a bare
# feature-flag environment variable, which the `no-bare-feature-flags` gate
# refuses, and neither is needed: the reporter already switches this probe as a
# whole (OMNINODE_FLEET_PROBE_ENABLED, read in bash where that gate does not
# apply), and the tests drive `check_scheduled_workflows` and
# `check_lab_pass_receipts` directly. A half-disabled probe would also be the
# worst shape to leave reachable -- it renders as a clean fleet section while
# one of its two sources is simply not looked at.
DEFAULT_REPO = "OmniNode-ai/omnibase_infra"
DEV_BRANCH = os.environ.get("OMNINODE_FLEET_DEV_BRANCH", "dev")


def emit(status: str, key: str, detail: str) -> None:
    """One `fleet|STATUS|key|detail` row.

    Pipes are stripped from the detail rather than escaped: the row format is
    pipe-delimited and the reporter's awk selectors count fields, so a detail
    carrying a pipe would silently shift every column after it.
    """
    print(f"fleet|{status}|{key}|{detail.replace('|', '/')}")


def _resolve_sibling(filename: str, env_override: str) -> Path | None:
    """Find a companion module in either layout, or return None.

    Two layouts are real and both must work. In the repository this file sits in
    ``scripts/`` and its companions in ``scripts/ci/``; on the host the
    maintenance sync installs every manifest entry FLAT into
    ``/data/maintenance/bin``. Guessing one and failing on the other is how a
    probe reports "missing" on a host where the file is present.
    """
    override = os.environ.get(env_override)
    if override:
        candidate = Path(override)
        return candidate if candidate.is_file() else None
    for candidate in (HERE / filename, HERE / "ci" / filename):
        if candidate.is_file():
            return candidate
    return None


#: The environment names that may carry a GitHub token, in the order `gh`
#: itself resolves them, with GH_PAT appended because the host env file uses
#: that name and `gh` has never heard of it.
GH_TOKEN_NAMES = ("GH_TOKEN", "GH_PAT", "GITHUB_TOKEN")


def _gh_token_name() -> str | None:
    """WHICH name supplied the token. Reported; the value never is.

    Worth a function of its own because of a measured 2026-09-20 confusion on
    the host: `/home/jonah/.omnibase/.env` carries a GH_PAT that authenticates
    HTTP 401 next to a GITHUB_TOKEN that authenticates HTTP 200, while
    `/data/omninode/omnibase_infra/.env` -- the file this reporter actually
    sources -- carries a valid GH_PAT and no GITHUB_TOKEN at all. A row that
    says only "Bad credentials" sends the next reader to whichever file they
    happen to think of. Naming the variable does not disclose anything: it is
    a variable name, and no value is read, printed or logged anywhere here.
    """
    for name in GH_TOKEN_NAMES:
        if os.environ.get(name):
            return name
    return None


def _gh_token() -> str | None:
    """The token `gh` will use, normalised across the names the host sets.

    The host env file carries GH_PAT and GITHUB_TOKEN; `gh` itself reads GH_TOKEN
    first and GITHUB_TOKEN second and has never heard of GH_PAT. Normalising here
    means a host that sets only GH_PAT is not silently unauthenticated -- which
    presents as a 404 on a private repository, i.e. as an absence rather than as
    a refusal.
    """
    for name in GH_TOKEN_NAMES:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _normalise_gh_token() -> None:
    """Publish the resolved token under the names `gh` actually reads.

    THIS IS A PROCESS-WIDE NORMALISATION ON PURPOSE, not just a child env.
    ``lab_pass_receipt`` is imported and calls ``gh api`` through its own
    ``subprocess.run`` with no ``env=``, so it inherits THIS process's
    environment. The host env file names the token ``GH_PAT``, which `gh` has
    never heard of, and under `sudo` there is no `gh` config to fall back on --
    so without this the receipt reads fail with "please run gh auth login"
    while the head-sha read one line above them succeeds. Measured on `.201`,
    2026-09-20, which is the whole reason this function exists rather than a
    child-env dict.

    No value is printed, logged or returned; only the names are ever reported.
    """
    token = _gh_token()
    if token:
        os.environ["GH_TOKEN"] = token
        os.environ["GITHUB_TOKEN"] = token


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    # The evaluator posts to Slack itself when these are set and it is not in
    # dry-run. It IS run in dry-run below, so this is belt-and-braces for the one
    # shape that would produce a duplicate message: a future caller dropping the
    # flag. Delivery belongs to the reporter's state machine, which is the only
    # thing here that de-duplicates.
    env.pop("SLACK_BOT_TOKEN", None)
    env.pop("SLACK_CHANNEL_ID", None)
    # An annotation writer with nowhere to write is noise on stdout that the row
    # parser would have to skip.
    env.pop("GITHUB_STEP_SUMMARY", None)
    return env


# ---------------------------------------------------------------------------
# Scheduled-workflow sweep
# ---------------------------------------------------------------------------


def _sweep_argv(evaluator: Path, policy: Path, report_path: Path) -> list[str]:
    return [
        sys.executable,
        str(evaluator),
        "--repos-from-policy",
        "--scheduled-only",
        "--policy",
        str(policy),
        "--report",
        str(report_path),
        # Delivery belongs to the reporter's state machine, which is the only
        # thing in this path that de-duplicates. A second poster here would send
        # every finding every hour with no hysteresis.
        "--dry-run",
    ]


def _spawn_refresh(evaluator: Path, policy: Path, cache_path: Path) -> str:
    """Start the sweep DETACHED and return a one-line status for the row.

    WHY DETACHED. The sweep is one API call per active workflow per repository
    and measured 434 seconds across the thirteen-repository fleet on
    2026-09-20. The reporter holds an exclusive flock for the whole of its tick
    and its cron cadence is */15, so running this inline would hold the lock
    for seven minutes and make the NEXT tick exit on the lock -- a health
    reporter that skips a tick every hour to look at CI. Detaching keeps every
    tick fast and bounds the cache at roughly one refresh interval plus one
    tick.

    A cron unit of its own was the other way to do this and is not taken: the
    finding behind this ticket is that the fleet has too many alerting surfaces
    and not enough delivery, so adding a scheduler to fix an alerting gap would
    be answering the problem with more of its cause.
    """
    marker = cache_path.with_suffix(".refreshing")
    try:
        if marker.is_file():
            age = time.time() - marker.stat().st_mtime
            if age < SWEEP_TIMEOUT_SECONDS:
                return f"a refresh started {int(age // 60)}m ago is still running"
            # Older than the sweep's own bound: the previous one died without
            # clearing its marker. Fall through and start a new one rather than
            # wedging forever on a stale file.
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        report_path = cache_path.with_suffix(".inflight.json")
        log_path = cache_path.with_suffix(".refresh.log")
        marker.write_text(json.dumps({"started_at": time.time()}), encoding="utf-8")
        with open(log_path, "w", encoding="utf-8") as log:
            subprocess.Popen(  # fixed argv, no shell
                _sweep_argv(evaluator, policy, report_path),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=_child_env(),
                start_new_session=True,
                cwd=str(cache_path.parent),
            )
    except OSError as exc:
        return f"refresh could not be started: {exc}"
    return "a refresh was started in the background"


def _harvest_refresh(cache_path: Path, min_runs_for_critical: int) -> str | None:
    """Fold a finished detached sweep into the cache. Returns an error or None.

    Returns None when there was nothing to harvest, which is not an error: the
    common case is that no refresh is in flight at all.
    """
    marker = cache_path.with_suffix(".refreshing")
    report_path = cache_path.with_suffix(".inflight.json")
    if not marker.is_file():
        return None
    if not report_path.is_file():
        age = time.time() - marker.stat().st_mtime
        if age >= SWEEP_TIMEOUT_SECONDS:
            log_path = cache_path.with_suffix(".refresh.log")
            tail = ""
            try:
                tail = log_path.read_text(encoding="utf-8").strip()[-300:]
            except OSError:
                tail = "(no refresh log)"
            marker.unlink(missing_ok=True)
            return f"the background refresh produced no report in {int(age // 60)}m: {tail}"
        return None
    # THE REPORT IS THE CONTRACT, NOT THE CHILD'S EXIT CODE. The evaluator
    # exits non-zero when ANY repository could not be read, which is right for
    # CI -- an unevaluated repository must never read as a clean one. The
    # report still carries every repository that WAS evaluated plus an `error`
    # for each that was not, and `_rows_from_report` turns each error into its
    # own standing row, so the partial result is rendered rather than
    # discarded. Discarding it is the whole-sweep blanking this avoids.
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows, heartbeat = _rows_from_report(report, min_runs_for_critical)
    except Exception as exc:  # noqa: BLE001 -- rule 16: a discarded error reads as a clean bill of health
        marker.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        return f"the background refresh wrote an unusable report: {str(exc)[:300]}"
    _write_cache(
        cache_path,
        {"evaluated_at": time.time(), "rows": rows, "heartbeat": heartbeat},
    )
    marker.unlink(missing_ok=True)
    report_path.unlink(missing_ok=True)
    return None


def _rows_from_report(
    report: dict[str, Any], min_runs_for_critical: int
) -> tuple[list[dict[str, str]], str]:
    """Turn one evaluator report into rows, plus the heartbeat detail.

    Severity is DERIVED from the report's own counters and the policy's own
    threshold, not from a second tuned number. A workflow above the policy's
    rate threshold is a WARNING. It escalates to CRITICAL only when it has not
    succeeded ONCE in the window AND has failed at least `failure_threshold`
    times -- the count the policy already declares as the point at which
    repeated failure stops being noise.

    The denominator clause is load-bearing. Without it a workflow with a single
    observed run in the whole window reports 100% and pages as a dead
    verification surface; two such rows were in the first live sweep. The
    evaluator's own docstring makes the same point about rates: a rate with no
    denominator is not evidence of anything.
    """
    rows: list[dict[str, str]] = []
    repos = report.get("repos")
    if not isinstance(repos, dict):
        msg = "report carries no 'repos' object"
        raise RuntimeError(msg)

    observed_workflows = 0
    unreadable = 0
    for slug, block in sorted(repos.items()):
        if not isinstance(block, dict):
            continue
        repo_name = slug.split("/")[-1]
        error = block.get("error")
        if error:
            # A repository the sweep could not read gets its OWN standing row.
            # Silence here would be the original defect in a new place: the
            # registry repository was invisible to the fleet's only alerter by
            # construction, and nobody could tell that from a clean board.
            unreadable += 1
            rows.append(
                {
                    "status": "WARNING",
                    "key": f"sched/{repo_name}/source-unreadable",
                    "detail": (
                        f"source unreadable: {slug} could not be read, so its "
                        "scheduled workflows were NOT evaluated and its state "
                        f"is UNKNOWN, not clean -- {str(error)[:220]}"
                    ),
                }
            )
            continue
        scheduled = block.get("scheduled")
        if not isinstance(scheduled, dict):
            continue
        workflows = scheduled.get("workflows")
        if isinstance(workflows, dict):
            observed_workflows += len(workflows)
        window_days = scheduled.get("window_days")
        for alert in scheduled.get("alerts") or []:
            if not isinstance(alert, dict):
                continue
            workflow = str(alert.get("workflow") or "unknown")
            # The evaluator reports a workflow by its repo-relative path; the
            # leading directory is the same on every row and only costs width
            # in a chat message.
            short = workflow.rsplit("/", 1)[-1]
            failures = int(alert.get("failures") or 0)
            total = int(alert.get("observed") or 0)
            rate = float(alert.get("rate_pct") or 0.0)
            never_green = (
                total > 0 and failures == total and failures >= min_runs_for_critical
            )
            rows.append(
                {
                    "status": "CRITICAL" if never_green else "WARNING",
                    "key": f"sched/{repo_name}/{short}",
                    "detail": (
                        f"scheduled workflow failed {rate:.1f}% ({failures}/{total}) "
                        f"over {window_days}d"
                        + (" and has not succeeded once" if never_green else "")
                        + f"; last failure {alert.get('last_failure_url') or 'unknown'}"
                    ),
                }
            )

    # The heartbeat states the DENOMINATOR, not just the findings. "scanned 13"
    # and "scanned 12 of 13" are different claims, and a reader must not have to
    # infer which one they are looking at.
    readable = len(repos) - unreadable
    heartbeat = (
        f"scanned {readable}/{len(repos)} repos, {observed_workflows} workflows "
        f"with scheduled runs in the window; {len(rows) - unreadable} above "
        f"threshold, {unreadable} repo(s) unreadable"
    )
    return rows, heartbeat


def _read_cache(cache_path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Fail-closed to "no cache": a corrupt cache re-evaluates rather than
        # being trusted. Losing the cache costs API calls; trusting a bad one
        # costs a wrong verdict.
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        return None
    return payload


def _write_cache(cache_path: Path, payload: dict[str, Any]) -> None:
    """Atomic swap: a crash mid-write must not leave a truncated cache."""
    tmp = cache_path.with_suffix(".tmp")
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(cache_path)
    except OSError as exc:
        emit(
            "WARNING",
            "scheduled-workflows/source-unreadable",
            f"source unreadable: cache could not be written to {cache_path}: {exc}",
        )


def _resolve_policy() -> Path | None:
    override = os.environ.get("OMNINODE_FLEET_POLICY_FILE")
    if override:
        candidate = Path(override)
        return candidate if candidate.is_file() else None
    for candidate in (
        HERE / "runner_routing_policy.yaml",
        HERE.parent / "config" / "runner_routing_policy.yaml",
    ):
        if candidate.is_file():
            return candidate
    return None


def _min_runs_for_critical(policy: Path) -> int:
    """The policy's own `failure_threshold`, reused rather than re-invented."""
    import yaml

    doc = yaml.safe_load(policy.read_text(encoding="utf-8"))
    block = ((doc or {}).get("route") or {}).get("nonrequired_check_alert") or {}
    return int(block["failure_threshold"])


def check_scheduled_workflows(state_dir: Path) -> None:
    cache_path = Path(
        os.environ.get(
            "OMNINODE_FLEET_SWEEP_CACHE", str(state_dir / "fleet-scheduled-sweep.json")
        )
    )
    evaluator = _resolve_sibling(
        "nonrequired_check_failure_rate.py", "OMNINODE_FLEET_EVALUATOR"
    )
    policy = _resolve_policy()

    if evaluator is None or policy is None:
        missing = (
            "the shared failure-rate evaluator"
            if evaluator is None
            else "the routing policy that declares the fleet repo list"
        )
        _emit_unreadable(
            cache_path, f"{missing} is not installed beside this probe", None
        )
        return

    try:
        min_runs = _min_runs_for_critical(policy)
    except Exception as exc:  # noqa: BLE001 -- see rule 16
        _emit_unreadable(cache_path, f"policy unreadable: {str(exc)[:200]}", None)
        return

    harvest_error = _harvest_refresh(cache_path, min_runs)
    cached = _read_cache(cache_path)
    age = time.time() - float(cached.get("evaluated_at", 0)) if cached else None

    if cached is None:
        status_line = _spawn_refresh(evaluator, policy, cache_path)
        emit(
            "WARNING",
            "scheduled-workflows/source-unreadable",
            "source unreadable: no fleet sweep has completed yet"
            + (f" ({harvest_error})" if harvest_error else "")
            + f"; {status_line}. Fleet scheduled-workflow state is UNKNOWN, not clean",
        )
        return

    stale = age is not None and age >= SWEEP_INTERVAL_SECONDS
    refresh_note = ""
    if stale:
        refresh_note = "; " + _spawn_refresh(evaluator, policy, cache_path)

    if harvest_error:
        status = (
            "CRITICAL"
            if age is not None and age >= SWEEP_STALE_CRITICAL_SECONDS
            else "WARNING"
        )
        emit(
            status,
            "scheduled-workflows/source-unreadable",
            f"source unreadable: {harvest_error}; serving the sweep from a "
            f"{int((age or 0) // 60)}m-old cache{refresh_note}",
        )

    _emit_cached(cached, stale=stale, age=age or 0.0)


def _emit_unreadable(cache_path: Path, reason: str, age: float | None) -> None:
    cached = _read_cache(cache_path)
    if cached is None:
        emit(
            "WARNING",
            "scheduled-workflows/source-unreadable",
            f"source unreadable: {reason}; no previous sweep to fall back on, "
            "so fleet scheduled-workflow state is UNKNOWN, not clean",
        )
        return
    cache_age = (
        age if age is not None else time.time() - float(cached.get("evaluated_at", 0))
    )
    status = "CRITICAL" if cache_age >= SWEEP_STALE_CRITICAL_SECONDS else "WARNING"
    emit(
        status,
        "scheduled-workflows/source-unreadable",
        f"source unreadable: {reason}; serving the sweep from a "
        f"{int(cache_age // 60)}m-old cache",
    )
    _emit_cached(cached, stale=True, age=cache_age)


def _emit_cached(cached: dict[str, Any], *, stale: bool, age: float) -> None:
    suffix = f" [cached {int(age // 60)}m ago]" if stale else ""
    heartbeat = str(cached.get("heartbeat") or "cached sweep")
    emit("OK", "scheduled-workflows", f"{heartbeat}{suffix}")
    for row in cached.get("rows") or []:
        if not isinstance(row, dict):
            continue
        emit(
            str(row.get("status") or "WARNING"),
            str(row.get("key") or "sched/unknown"),
            f"{row.get('detail') or 'no detail'}{suffix}",
        )


# ---------------------------------------------------------------------------
# Lab-pass receipts on the dev head
# ---------------------------------------------------------------------------


def _load_receipt_module() -> Any:
    path = _resolve_sibling("lab_pass_receipt.py", "OMNINODE_FLEET_RECEIPT_MODULE")
    if path is None:
        msg = "lab_pass_receipt.py is not installed beside this probe"
        raise RuntimeError(msg)
    spec = importlib.util.spec_from_file_location("omninode_lab_pass_receipt", path)
    if spec is None or spec.loader is None:
        msg = f"lab_pass_receipt.py at {path} could not be loaded as a module"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE execution, not after. `lab_pass_receipt` declares frozen
    # dataclasses, and `dataclasses._is_type` resolves a field's annotation
    # through `sys.modules[cls.__module__]` while the decorator runs -- so a
    # module executed outside `sys.modules` dies at its first `@dataclass` with
    # `'NoneType' object has no attribute '__dict__'`, which reads as a
    # corrupt file rather than as a loader mistake.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _gh_json(path: str) -> Any:
    completed = subprocess.run(
        ["gh", "api", path],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=_child_env(),
    )
    if completed.returncode != 0:
        msg = f"`gh api {path}` exited {completed.returncode}: {completed.stderr.strip()[:200]}"
        raise RuntimeError(msg)
    return json.loads(completed.stdout)


def check_lab_pass_receipts() -> None:
    """Rule 24(b) lab-pass state for the current `dev` head.

    THE SHAPE OF THE FINDING IS THE GATE'S OWN, NOT ONE INVENTED HERE.
    ``evaluate_gate`` passes a sha when ANY lane carries a PASS receipt for it,
    so a lane with no receipt is ordinary -- the three emitters do not each fire
    on every head, and reporting per-lane absence would stand a WARNING up
    permanently against a fleet that is working. Two things are findings:

      * a FAIL receipt on a lane. The delivery gate already refuses on it and
        says so on a workflow run nobody opens; this is the row that makes the
        refusal reach a person.
      * no PASS on ANY lane once the head is past the grace window. That is the
        delivery gate held closed, which is the silent-gate case this ticket
        exists for -- the candidate is undeliverable and nothing says so.
    """
    repo = os.environ.get("OMNINODE_FLEET_RECEIPT_REPO", DEFAULT_REPO)
    try:
        module = _load_receipt_module()
        head = _gh_json(f"repos/{repo}/commits/{DEV_BRANCH}")
        sha = str(head.get("sha") or "")
        committed = str(
            (head.get("commit") or {}).get("committer", {}).get("date") or ""
        )
    except Exception as exc:  # noqa: BLE001 -- see rule 16
        token_name = _gh_token_name() or "no GitHub token variable is set"
        emit(
            "WARNING",
            "lab-pass/source-unreadable",
            f"source unreadable: {str(exc)[:250]} (authenticating with "
            f"{token_name}); lab-pass state for the {DEV_BRANCH} head is "
            "UNKNOWN, not clean",
        )
        return

    if len(sha) != 40:
        emit(
            "WARNING",
            "lab-pass/source-unreadable",
            f"source unreadable: {repo}@{DEV_BRANCH} resolved to {sha!r}, "
            "which is not a 40-character sha",
        )
        return

    head_age = _age_seconds(committed)
    head_age_text = f"{int(head_age // 60)}m" if head_age is not None else "unknown"
    lanes = list(module.EnumLabLane)
    passing: list[str] = []
    unreadable: list[str] = []
    absent: list[str] = []

    for lane in lanes:
        key = f"lab-pass/{lane.value}"
        name = module.artifact_name(lane, sha)
        try:
            artifacts = module.list_artifacts(repo, name)
        except Exception as exc:  # noqa: BLE001 -- see rule 16
            unreadable.append(lane.value)
            emit(
                "WARNING", key, f"source unreadable: {str(exc)[:250]} (sha {sha[:12]})"
            )
            continue
        if not artifacts:
            absent.append(lane.value)
            continue
        # Newest first: a re-run of the emitting job supersedes an earlier
        # attempt for the same sha and lane. Same rule as the gate's own reader.
        artifacts.sort(key=lambda a: str(a.get("created_at", "")), reverse=True)
        try:
            receipt = module.download_receipt(repo, int(artifacts[0]["id"]))
        except Exception as exc:  # noqa: BLE001 -- see rule 16
            unreadable.append(lane.value)
            emit(
                "WARNING",
                key,
                f"source unreadable: receipt for {sha[:12]} could not be read: "
                f"{str(exc)[:200]}",
            )
            continue

        if receipt.result.value == "PASS":
            passing.append(lane.value)
            emit("OK", key, f"PASS for {sha[:12]} ({len(receipt.checks)} checks)")
            continue

        failed = [c.name for c in receipt.checks if not c.ok]
        emit(
            "CRITICAL",
            key,
            f"lab pass FAILED for {DEV_BRANCH} head {sha[:12]} (head age "
            f"{head_age_text}); failing checks: {', '.join(failed) or 'unnamed'}",
        )

    if passing:
        emit(
            "OK",
            "lab-pass/dev-head",
            f"{DEV_BRANCH} head {sha[:12]} is deliverable: PASS on "
            f"{', '.join(passing)}",
        )
        return

    if head_age is not None and head_age < RECEIPT_ABSENCE_GRACE_SECONDS:
        emit(
            "OK",
            "lab-pass/dev-head",
            f"no PASS yet for {sha[:12]}, head is only {head_age_text} old "
            "(inside the grace window)",
        )
        return

    # Fail-closed on an unknown age: a head whose timestamp did not parse is not
    # evidence that it is young.
    detail = (
        f"NO passing lab-pass receipt on any lane for {DEV_BRANCH} head "
        f"{sha[:12]} (head age {head_age_text}). Delivery to staging is gated "
        f"closed on this sha. absent={', '.join(absent) or 'none'} "
        f"unreadable={', '.join(unreadable) or 'none'}"
    )
    emit("CRITICAL", "lab-pass/dev-head", detail)


def _age_seconds(iso: str) -> float | None:
    from datetime import UTC, datetime

    if not iso:
        return None
    try:
        when = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (datetime.now(UTC) - when).total_seconds()


def main() -> int:
    state_dir = Path(
        os.environ.get("OMNINODE_ALERT_STATE_DIR", "/data/maintenance/state")
    )
    _normalise_gh_token()
    check_scheduled_workflows(state_dir)
    check_lab_pass_receipts()
    # Emitting a row is not this probe failing. The reporter reads rows, and a
    # non-zero exit would make it report "the probe broke" over rows it in fact
    # produced -- the same distinction the evaluator's own docstring draws.
    return 0


if __name__ == "__main__":
    sys.exit(main())
