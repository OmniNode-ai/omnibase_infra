# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19894 -- pick the lab lane a probe grades from the overlay's ordered list.

WHY THIS EXISTS
    Operator rulings 2026-09-28T01:58:06Z ("it's a problem if checks are reliant
    on a specific machine") and 01:58:17Z ("chain of responders right?"). Before
    this, every lab probe in this repository ran only on the .201 verify runner
    and reached the lane through that host's ``host.docker.internal`` alias, so
    the probe queued forever when .201 was down and could only ever grade .201.

    Now the deployment overlay -- the repository variable ``LAB_LANES_JSON``,
    supplied by whoever runs the lab, never a literal in this repository --
    declares the lanes a probe may grade, in order, cheapest first. The probe
    grades the FIRST lane whose required surfaces all answer. The probe job
    itself runs on any runner of the pool the overlay names.

THE OVERLAY SHAPE
    A JSON list of objects. Each object has a ``name`` and any of:

      ``<surface>_url``   an ``http(s)://host:port`` base a runner on ANY host can
                          reach. A loopback address or a Docker host-gateway
                          alias is refused: it names a different machine on
                          every runner, which is the defect this file removes.
      ``*_runs_on``       a runner label list, for a step that must run beside
                          the lane (reading the lane's own container). This is
                          the only way a probe may name a host, and it is data.
      anything else       a string, exported as-is.

    ``LANE_REQUIRE`` names the fields a lane must declare to be eligible. Each
    required ``*_url`` must answer ``GET <url>/health`` with a 2xx; a lane with
    one silent surface is skipped, and the next lane is tried.

    ``LANE_MATCH`` (optional, ``field=value`` pairs) narrows eligibility to the
    lanes a probe knows how to read: a probe that reads containers of the
    compose project ``omnibase-infra`` asks for ``compose_project=omnibase-infra``
    and is then placed beside that lane, wherever the overlay says it runs.

WHAT IT WRITES
    ``GITHUB_ENV``: ``LANE_NAME`` and ``LANE_<FIELD>`` for every declared field
    of the chosen lane (lists as JSON). ``GITHUB_OUTPUT``: ``name``, ``lane``
    (the whole entry as JSON) and every field under its own name.

DEPLOY WAIT (OMN-19811, OMN-20509)
    ``LANE_DEPLOY_WAIT_SECONDS`` opts into a bounded wait when no lane answers.
    Eligible lanes' declared deploy agents must report readable, busy state;
    idle or unreadable agents alone never justify waiting. Poll every
    ``LANE_DEPLOY_POLL_SECONDS`` (default 15), then try the ordered lanes again.
    A successful wait exports ``LANE_WAITED_SECONDS``, rounded up, so the probe
    can share the same budget. The default budget of 0 keeps a single shot.

    The agents are read once BEFORE the first lane check as well as after it:
    the check spends its probe timeouts first, and a settle that ends inside
    them (the agent re-execs, then reports idle) would otherwise leave nothing
    to see. A busy pre-read counts as busy evidence for the wait. When an agent
    then reads idle or unreadable, the lane gets ``SETTLE_GRACE_SECONDS`` more
    to come back before the wait ends RED; the budget still bounds all of it.

NO RESPONDER IS RED, NEVER A SKIP
    An empty or unset list, a malformed entry, or no eligible lane answering
    exits 1 naming every lane tried and why. There is no default lane.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any

URL_RE = re.compile(r"^https?://(?P<host>[A-Za-z0-9.-]+):(?P<port>[0-9]{1,5})/?$")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
FIELD_RE = re.compile(r"^[a-z][a-z0-9_]*$")
# A name that resolves to a different machine on every runner.
HOST_LOCAL_NAMES = frozenset(
    {"localhost", "host.docker.internal", "gateway.docker.internal"}
)
# OMN-20509: how long a lane keeps being waited for once its deploy agent stops
# reading busy. The agent leaves ``settling`` a moment before the lane's
# surfaces answer again (run 37179200279: the settle ended 05:12:58Z, after
# the lane had spent the preceding minutes timing out under image imports).
SETTLE_GRACE_SECONDS = 60.0


class OverlayError(ValueError):
    """The overlay itself is unusable; no lane was tried."""


def _check_url(lane: str, field: str, value: Any) -> str:
    if not isinstance(value, str):
        raise OverlayError(f"lane {lane!r}: {field} must be a string")
    match = URL_RE.match(value)
    if match is None:
        raise OverlayError(
            f"lane {lane!r}: {field}={value!r} is not http(s)://host:port"
        )
    host = match.group("host").lower()
    if host in HOST_LOCAL_NAMES:
        raise OverlayError(
            f"lane {lane!r}: {field}={value!r} names a host-local alias, which "
            "reaches a different machine from every runner"
        )
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None  # a hostname, not an address
    if address is not None and address.is_loopback:
        raise OverlayError(f"lane {lane!r}: {field}={value!r} is a loopback address")
    return value.rstrip("/")


def parse_lanes(raw: str) -> list[dict[str, Any]]:
    """Validate the overlay list. Raises OverlayError; never returns a default."""
    if not raw or not raw.strip():
        raise OverlayError(
            "no lab lane is declared; set the repository variable LAB_LANES_JSON"
        )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OverlayError(f"LAB_LANES_JSON is not JSON: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise OverlayError("LAB_LANES_JSON must be a non-empty JSON list")
    lanes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise OverlayError(f"LAB_LANES_JSON[{index}] is not an object")
        name = entry.get("name")
        if not isinstance(name, str) or not NAME_RE.match(name):
            raise OverlayError(f"LAB_LANES_JSON[{index}] has no valid name")
        if name in seen:
            raise OverlayError(f"lane {name!r} is declared twice")
        seen.add(name)
        lane: dict[str, Any] = {"name": name}
        for field, value in entry.items():
            if field == "name":
                continue
            if not FIELD_RE.match(field):
                raise OverlayError(f"lane {name!r}: field {field!r} is not snake_case")
            if field.endswith("_url"):
                lane[field] = _check_url(name, field, value)
            elif field.endswith("_runs_on"):
                if (
                    not isinstance(value, list)
                    or not value
                    or not all(isinstance(v, str) and v for v in value)
                ):
                    raise OverlayError(
                        f"lane {name!r}: {field} must be a non-empty list of labels"
                    )
                lane[field] = list(value)
            elif isinstance(value, str):
                lane[field] = value
            else:
                raise OverlayError(f"lane {name!r}: {field} must be a string")
        lanes.append(lane)
    return lanes


def _open(url: str, timeout: float) -> Any:
    """The one urlopen in this file; every caller has checked the scheme."""
    return urllib.request.urlopen(url, timeout=timeout)  # noqa: S310 -- scheme checked by the caller


def http_health(url: str, timeout: float) -> str | None:
    """None when ``GET <url>/health`` answers 2xx, else why not. Never raises.

    ``url`` has already passed ``_check_url``: its scheme is http or https.
    """
    if not url.startswith(("http://", "https://")):
        return f"refused scheme ({url})"
    try:
        with _open(f"{url}/health", timeout) as response:
            if 200 <= response.status < 300:
                return None
            return f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code}"
    except (urllib.error.URLError, OSError, ValueError) as exc:
        reason = getattr(exc, "reason", exc)
        return f"no answer ({type(exc).__name__}: {reason})"


def read_deploy_agent(url: str, timeout: float) -> dict[str, Any] | None:
    """Read /health even on HTTP 503, and best-effort /queue. Never raises."""
    if not url.startswith(("http://", "https://")):
        return None
    base = url.rstrip("/")

    def read_object(surface: str) -> dict[str, Any] | None:
        # An unreadable agent is never deploy evidence: every way a read can
        # fail below is None, and nothing else is swallowed.
        try:
            try:
                response = _open(f"{base}/{surface}", timeout)
            except urllib.error.HTTPError as exc:
                response = exc  # An error status can still carry the agent's state.
            with response:
                body = json.load(response)
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError):
            return None
        return body if isinstance(body, dict) else None

    health = read_object("health")
    if health is None:
        return None
    return {"health": health, "queue": read_object("queue")}


def deploy_agent_busy_reason(snapshot: Mapping[str, Any]) -> str | None:
    """Pure busy test, matching the canary snapshot and its queue freshness bound."""
    health = snapshot.get("health")
    if not isinstance(health, dict):
        return None
    state = health.get("state")
    if state and state != "idle":
        return f"state={state}"
    if health.get("active_job"):
        return "active_job"
    last = health.get("last_result")
    if isinstance(last, dict) and last.get("settling"):
        return "settling"
    queue = snapshot.get("queue")
    if not isinstance(queue, dict):
        return None
    ahead = queue.get("commands_ahead")
    if not isinstance(ahead, int) or isinstance(ahead, bool) or ahead <= 0:
        return None
    age = queue.get("control_topic_lag_age_seconds")
    if age is not None and (
        isinstance(age, bool) or not isinstance(age, (int, float)) or age > 120
    ):
        return None
    return f"{ahead} command(s) queued"


def _busy_agents(
    candidates: Sequence[Mapping[str, Any]], timeout: float, note: str = ""
) -> dict[str, str]:
    """Lane name -> busy detail for every candidate whose agent reads busy now."""
    busy: dict[str, str] = {}
    for candidate in candidates:
        snapshot = read_deploy_agent(candidate["deploy_agent_url"], timeout)
        reason = deploy_agent_busy_reason(snapshot) if snapshot is not None else None
        if reason is not None:
            busy[candidate["name"]] = (
                f"{candidate['name']} agent {candidate['deploy_agent_url']}: "
                f"{reason}{note}"
            )
    return busy


def choose_lane(
    lanes: Sequence[Mapping[str, Any]],
    require: Sequence[str],
    probe: Callable[[str], str | None],
    match: Mapping[str, str] | None = None,
) -> tuple[Mapping[str, Any] | None, list[str]]:
    """First matching lane that declares every required field and whose required URLs answer."""
    tried: list[str] = []
    for lane in lanes:
        unmatched = [
            f"{field}={lane.get(field, '<undeclared>')}"
            for field, want in (match or {}).items()
            if lane.get(field) != want
        ]
        if unmatched:
            tried.append(f"{lane['name']}: does not match ({', '.join(unmatched)})")
            continue
        missing = [field for field in require if field not in lane]
        if missing:
            tried.append(f"{lane['name']}: does not declare {', '.join(missing)}")
            continue
        silent = []
        for field in require:
            if field.endswith("_url"):
                why = probe(lane[field])
                if why is not None:
                    silent.append(f"{field} {lane[field]} {why}")
        if silent:
            tried.append(f"{lane['name']}: " + "; ".join(silent))
            continue
        return lane, tried
    return None, tried


def _say(line: str) -> None:
    """One line to the job log (workflow commands such as ::error:: included)."""
    sys.stdout.write(f"{line}\n")


def _env_value(value: Any) -> str:
    return (
        json.dumps(value, separators=(",", ":"))
        if isinstance(value, list)
        else str(value)
    )


def main(environ: Mapping[str, str] | None = None) -> int:
    env = os.environ if environ is None else environ
    try:
        wait_budget = float(env.get("LANE_DEPLOY_WAIT_SECONDS", "0"))
        if not math.isfinite(wait_budget) or wait_budget < 0:
            raise ValueError
    except ValueError:
        _say("::error::LANE_DEPLOY_WAIT_SECONDS must be a non-negative finite number")
        return 1
    try:
        poll = float(env.get("LANE_DEPLOY_POLL_SECONDS", "15"))
        if not math.isfinite(poll) or poll <= 0:
            raise ValueError
    except ValueError:
        _say("::error::LANE_DEPLOY_POLL_SECONDS must be a positive finite number")
        return 1
    require = [f for f in re.split(r"[\s,]+", env.get("LANE_REQUIRE", "")) if f]
    match: dict[str, str] = {}
    for pair in (p for p in re.split(r"[\s,]+", env.get("LANE_MATCH", "")) if p):
        field, sep, want = pair.partition("=")
        if not sep or not FIELD_RE.match(field) or not want:
            _say(f"::error::LANE_MATCH entry {pair!r} is not field=value")
            return 1
        match[field] = want
    timeout = float(env.get("LANE_PROBE_TIMEOUT_SECONDS", "10"))
    try:
        lanes = parse_lanes(env.get("LAB_LANES_JSON", ""))
    except OverlayError as exc:
        _say(f"::error::{exc}")
        return 1
    candidates = [
        candidate
        for candidate in lanes
        if all(candidate.get(field) == want for field, want in match.items())
        and all(field in candidate for field in require)
        and "deploy_agent_url" in candidate
    ]
    # Before the lane check: it spends its probe timeouts first, and a settle
    # that ends inside them leaves an idle agent behind (OMN-20509).
    pre_busy = (
        _busy_agents(candidates, timeout, " (read before the lane check)")
        if wait_budget > 0
        else {}
    )
    lane, tried = choose_lane(
        lanes, require, lambda url: http_health(url, timeout), match
    )
    waited: float | None = None
    wait_detail = ""
    if lane is None and wait_budget > 0:
        started = time.monotonic()
        last_busy = dict(pre_busy)
        saw_busy = bool(pre_busy)
        not_busy_since: float | None = None
        end_reason = "no readable busy deploy agent"
        while lane is None:
            busy = _busy_agents(candidates, timeout)
            last_busy.update(busy)
            now = time.monotonic()
            if busy:
                saw_busy = True
                not_busy_since = None
            elif not saw_busy:
                break
            else:
                if not_busy_since is None:
                    not_busy_since = now
                if now - not_busy_since >= SETTLE_GRACE_SECONDS:
                    end_reason = f"deploy agent not busy for {SETTLE_GRACE_SECONDS:g}s"
                    break
            if waited is None:
                _say(
                    "lab lane deploy wait started -- "
                    + " | ".join((busy or pre_busy).values())
                )
                waited = 0.0
            if now - started + poll > wait_budget:
                end_reason = "budget exhausted"
                break
            time.sleep(poll)
            lane, tried = choose_lane(
                lanes, require, lambda url: http_health(url, timeout), match
            )
        if waited is not None:
            waited = time.monotonic() - started
            if lane is not None:
                end_reason = f"lane {lane['name']} answers"
            wait_detail = (
                "; deploy wait: "
                + " | ".join(last_busy.values())
                + f"; waited {waited:g}s of {wait_budget:g}s budget ({end_reason})"
            )
            _say(f"lab lane deploy wait ended -- {wait_detail.removeprefix('; ')}")
    for line in tried:
        _say(f"lane skipped -- {line}")
    if lane is None:
        _say(
            "::error::no declared lab lane answers on "
            f"{', '.join(require) or '(nothing required)'}; tried: "
            + " | ".join(tried)
            + ". That is RED, not a skip."
            + wait_detail
        )
        return 1
    _say(f"lab lane: {lane['name']} (after {len(tried)} earlier)")
    env_path = env.get("GITHUB_ENV")
    out_path = env.get("GITHUB_OUTPUT")
    if env_path:
        with open(env_path, "a", encoding="utf-8") as handle:
            handle.write(f"LANE_NAME={lane['name']}\n")
            for field, value in lane.items():
                if field != "name":
                    handle.write(f"LANE_{field.upper()}={_env_value(value)}\n")
            if waited is not None:
                handle.write(f"LANE_WAITED_SECONDS={math.ceil(waited)}\n")
    if out_path:
        with open(out_path, "a", encoding="utf-8") as handle:
            handle.write(f"lane={json.dumps(dict(lane), separators=(',', ':'))}\n")
            for field, value in lane.items():
                handle.write(f"{field}={_env_value(value)}\n")
    summary = env.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(
                f"lab lane: `{lane['name']}`, first of the overlay's ordered list "
                f"to answer, after {len(tried)} earlier\n\n"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
