#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18200 AC5 -- read the deploy agent's onex-lab overlay record for one sha.

WHY THIS EXISTS
---------------
The k3s ``onex-lab`` overlay can only be applied from the lab host: the k3s admin
kubeconfig is ``0600 root``, the containerd socket is root-owned, and the runtime
image for the merged sha lives on that host's Docker daemon because the deploy
agent just built it. The runner fleet holds a READ-ONLY ``lab-ci-reader``
ServiceAccount kubeconfig (OMN-18188), which can read the lane and not apply to
it. So the apply belongs to the agent.

A GitHub Actions artifact, on the other hand, can only be created by a job inside
a run, and the lab-pass receipt surface is an artifact by deliberate design (see
``lab_pass_receipt.py``'s header: an artifact needs no new credential, no bucket
and no cluster access). So the receipt belongs to CI.

This script is the join: it polls the agent's own HTTP surface for the record
keyed by the merged sha and prints the record's checks as the JSON array
``lab_pass_receipt.py emit --checks-json`` consumes.

IT ALWAYS PRINTS A CHECK LIST, and that is the point. A receipt that exists only
when the apply worked cannot distinguish "the lab apply failed" from "nobody ran
it", which is the distinction rule 24(a)'s ``if: always()`` emitters exist for. A
timeout, an unreachable agent, a 404, a 500 and a malformed body are each turned
into a NAMED FAILING CHECK carrying what was actually read, so the receipt is
emitted either way and the reason is legible in it.

STDLIB ONLY. The emitting job runs on a bare self-hosted runner and the receipt
module it feeds is stdlib-plus-hand-written-validation for a measured reason: the
first live run of that emitter died at import on a missing third-party package
after all its checks had already passed, and took the whole delivery with it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any

#: Only an exact commit. An abbreviated ref could resolve to more than one
#: commit, and a record fetched by prefix is a record about the wrong change.
SHA_RE = re.compile(r"^[0-9a-f]{40}$")

#: The check name that carries a transport or absence failure. One name, so a
#: reader grepping a run for the lab-overlay verdict finds it whatever went
#: wrong, instead of having to know the failure taxonomy first.
RECORD_CHECK = "lab_overlay_record"


def _truncate(text: str, limit: int = 400) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + "..."


def fetch_once(url: str, timeout_seconds: float) -> tuple[int, str]:
    """GET the record. Never raises -- the failure IS the evidence."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:  # noqa: S310
            return int(response.status), response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 - any transport failure is evidence
        return 0, f"{type(exc).__name__}: {exc}"


def checks_from_record(payload: Any, *, sha: str, url: str) -> list[dict[str, Any]]:
    """Validate one record and return its checks, or one failing check saying why.

    A record whose ``sha`` disagrees with the one requested is REFUSED rather than
    accepted: the name and the payload disagreeing is itself a finding, and it is
    the same check ``lab_pass_receipt.evaluate_gate`` makes on an artifact for the
    same reason.
    """
    if not isinstance(payload, dict):
        return [
            {
                "name": RECORD_CHECK,
                "ok": False,
                "evidence": f"GET {url} returned a {type(payload).__name__}, not an object",
            }
        ]
    if payload.get("sha") != sha:
        return [
            {
                "name": RECORD_CHECK,
                "ok": False,
                "evidence": (
                    f"GET {url} returned a record for sha {payload.get('sha')!r}; "
                    f"the request named {sha}. The reader refuses a record whose "
                    "payload and key disagree."
                ),
            }
        ]
    checks = payload.get("checks")
    if not isinstance(checks, list) or not checks:
        return [
            {
                "name": RECORD_CHECK,
                "ok": False,
                "evidence": (
                    f"GET {url} returned a record carrying no checks. A record with "
                    "no checks asserts that nothing was verified."
                ),
            }
        ]
    for check in checks:
        if not isinstance(check, dict) or set(check) != {"name", "ok", "evidence"}:
            return [
                {
                    "name": RECORD_CHECK,
                    "ok": False,
                    "evidence": (
                        f"GET {url} returned a check that is not "
                        f"{{name, ok, evidence}}: {_truncate(json.dumps(check))}"
                    ),
                }
            ]
    return checks


def poll(
    *,
    base_url: str,
    sha: str,
    wait_seconds: int,
    poll_interval_seconds: int,
    request_timeout_seconds: float,
    out: Any,
    sleep: Any = time.sleep,
    now: Any = time.monotonic,
) -> list[dict[str, Any]]:
    """Poll until a record exists, the window closes, or a hard failure occurs.

    A ``404`` is the only retryable answer: it means the apply has not finished
    writing yet, which is expected -- the agent writes the record after a
    multi-minute apply that begins only once the compose lane has converged. A
    ``400``, a ``500`` or a malformed body is terminal, because retrying cannot
    change any of them and spending the window on a certainty is time the run
    could have spent reporting it.
    """
    url = f"{base_url.rstrip('/')}/lab-overlay/{sha}"
    deadline = now() + wait_seconds
    attempts = 0
    last = ""
    while True:
        attempts += 1
        status, body = fetch_once(url, request_timeout_seconds)
        if status == 200:
            try:
                payload = json.loads(body)
            except json.JSONDecodeError as exc:
                return [
                    {
                        "name": RECORD_CHECK,
                        "ok": False,
                        "evidence": f"GET {url} -> 200 but not JSON: {exc}",
                    }
                ]
            print(f"record found after {attempts} attempt(s)", file=out)
            return checks_from_record(payload, sha=sha, url=url)
        if status in (400, 500):
            return [
                {
                    "name": RECORD_CHECK,
                    "ok": False,
                    "evidence": f"GET {url} -> {status} {_truncate(body)}",
                }
            ]
        last = f"{status} {_truncate(body, 160)}"
        remaining = deadline - now()
        if remaining <= 0:
            return [
                {
                    "name": RECORD_CHECK,
                    "ok": False,
                    "evidence": (
                        f"GET {url} never returned a record within {wait_seconds}s "
                        f"({attempts} attempt(s), last: {last}). The lab overlay "
                        "was not re-applied for this sha, or the deploy agent did "
                        "not reach the step that records it."
                    ),
                }
            ]
        print(f"attempt {attempts}: {last}; {int(remaining)}s left", file=out)
        sleep(min(poll_interval_seconds, max(1, int(remaining))))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha", required=True, help="the merged 40-character sha")
    parser.add_argument(
        "--agent-url",
        required=True,
        help="base URL of the deploy agent's HTTP surface on the lab host",
    )
    parser.add_argument("--wait-seconds", type=int, default=900)
    parser.add_argument("--poll-interval-seconds", type=int, default=30)
    parser.add_argument("--request-timeout-seconds", type=float, default=10.0)
    parser.add_argument(
        "--out",
        required=True,
        help="path the checks JSON array is written to",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not SHA_RE.match(args.sha):
        print(
            f"::error::{args.sha!r} is not a 40-character lowercase commit sha. "
            "Refusing to resolve an abbreviated ref.",
            file=sys.stderr,
        )
        return 1
    checks = poll(
        base_url=args.agent_url,
        sha=args.sha,
        wait_seconds=args.wait_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
        out=sys.stderr,
    )
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(checks, handle, indent=2)
        handle.write("\n")
    failing = [check for check in checks if not check["ok"]]
    for check in checks:
        print(
            f"  [{'ok  ' if check['ok'] else 'FAIL'}] {check['name']}: "
            f"{check['evidence']}",
            file=sys.stderr,
        )
    # Exit 0 even when checks fail. The verdict is DERIVED from the checks by
    # `lab_pass_receipt.py emit`, which is the one place a verdict is computed, and
    # it exits non-zero on a FAIL receipt. A non-zero exit here would stop the
    # emit step from running at all and lose the very record this script fetched.
    print(
        f"wrote {len(checks)} check(s) to {args.out} ({len(failing)} failing)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
