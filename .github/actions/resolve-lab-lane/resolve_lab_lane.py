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

WHAT IT WRITES
    ``GITHUB_ENV``: ``LANE_NAME`` and ``LANE_<FIELD>`` for every declared field
    of the chosen lane (lists as JSON). ``GITHUB_OUTPUT``: ``name``, ``lane``
    (the whole entry as JSON) and every field under its own name.

NO RESPONDER IS RED, NEVER A SKIP
    An empty or unset list, a malformed entry, or no eligible lane answering
    exits 1 naming every lane tried and why. There is no default lane.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import sys
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


def http_health(url: str, timeout: float) -> str | None:
    """None when ``GET <url>/health`` answers 2xx, else why not. Never raises.

    ``url`` has already passed ``_check_url``: its scheme is http or https.
    """
    if not url.startswith(("http://", "https://")):
        return f"refused scheme ({url})"
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=timeout) as response:  # noqa: S310 -- scheme checked above
            if 200 <= response.status < 300:
                return None
            return f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code}"
    except (urllib.error.URLError, OSError, ValueError) as exc:
        reason = getattr(exc, "reason", exc)
        return f"no answer ({type(exc).__name__}: {reason})"


def choose_lane(
    lanes: Sequence[Mapping[str, Any]],
    require: Sequence[str],
    probe: Callable[[str], str | None],
) -> tuple[Mapping[str, Any] | None, list[str]]:
    """First lane that declares every required field and whose required URLs answer."""
    tried: list[str] = []
    for lane in lanes:
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
    require = [f for f in re.split(r"[\s,]+", env.get("LANE_REQUIRE", "")) if f]
    timeout = float(env.get("LANE_PROBE_TIMEOUT_SECONDS", "10"))
    try:
        lanes = parse_lanes(env.get("LAB_LANES_JSON", ""))
    except OverlayError as exc:
        _say(f"::error::{exc}")
        return 1
    lane, tried = choose_lane(lanes, require, lambda url: http_health(url, timeout))
    for line in tried:
        _say(f"lane skipped -- {line}")
    if lane is None:
        _say(
            "::error::no declared lab lane answers on "
            f"{', '.join(require) or '(nothing required)'}; tried: "
            + " | ".join(tried)
            + ". That is RED, not a skip."
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
