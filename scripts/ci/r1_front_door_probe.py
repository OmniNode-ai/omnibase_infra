#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# onex-allow-file OMN-19445 reason="R1 front-door probe -- the lab lane address is resolved by the CLI's own config/ci_bus_lanes.py, never passed as a literal here."

"""OMN-19445 -- the R1 front-door probe, run daily from the lab host.

WHAT THIS IS
    "PLAN-R1 / MD-14: a good local answer ends the chain, and the decision is
    in the log." The probe text carried in the beta rebaseline and GOAL
    documents was wrong for months -- `onex delegate ... --bus kafka
    --kafka-bootstrap <lab broker literal host:port>`, which the dev-lane
    broker refuses (SASL required since OMN-18012 Phase B) -- and NOTHING ran
    it on a schedule, so the front door could regress for days with no signal. The
    corrected form, `--lane dev` (the CLI resolves the address and transport
    from `config/ci_bus_lanes.yaml`, the same resolver the OCC publishers and
    the delegation-nightly runner use), was proven once (2026-09-21, run
    `4a6254d5`) and then the daily rebaseline snapshots silently reverted to
    citing the stale, refused form again on the next three runs. A probe that
    is proven once by hand and never re-run is not a regression signal.

    This script is the producer: one `onex delegate` call, graded, with its
    EXIT CODE as the verdict -- the C11/C12/C15/provider-rung-canary shape
    already used in this repo.

    ``0``  the CLI accepted the prompt and exited 0
    ``1``  the CLI refused or exited non-zero (the record names why)
    ``2``  the probe itself could not run (the `onex` binary is not on PATH,
           or the subprocess call raised before producing a returncode).
           Deliberately distinct from ``1``: "I could not run the probe" is
           not "the front door is broken".

WHY THIS RUNNER, AND WHY OUTSIDE THE RUNTIME CONTAINER
    AC2 asks for the probe to run "outside the runtime container": unlike
    provider-rung-canary (which execs INTO omninode-runtime-effects to reach
    the RUNTIME's own resolved routing), this probe exercises the FRONT DOOR
    -- the CLI a customer or a lane operator runs directly, on its own host,
    dispatching over the bus to the deployed lane. Running it inside the
    runtime container would prove the runtime can reach itself, not that the
    documented customer-facing command works. `host-201` (the lab-host runner
    label already used by provider-rung-canary and C12) is required for the
    same reason as those: the lab lane's broker is not reachable from a
    GitHub-hosted runner.

WHY THE LANE IS `dev`, NOT `stability-test`
    PLAN-R1/MD-14 measure the FRONT DOOR a person actually uses, and `dev` is
    the lane a lane operator's ordinary `onex delegate` targets (D11's
    stability-test-only nightly, OMN-19311, is a different measurement: the
    Layer-2 regression corpus against the designated proof lane). Conflating
    the two is exactly the MD-14/two-meanings drift this ticket settles one
    half of (the other half, D11, already only ever meant stability-test).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

# The exact prompt the ticket's own live evidence used (run `4a6254d5`,
# 2026-09-24): a two-word answer, cheap, and unambiguous to grade by exit
# code alone -- this probe does not parse or score the response text.
PROBE_PROMPT = "Reply with exactly the word: ok"
PROBE_LANE = "dev"
DEFAULT_TIMEOUT_S = 90.0
_TAIL_CHARS = 2000


class _CompletedLike(Protocol):
    returncode: int
    stdout: str
    stderr: str


RunnerFn = Callable[..., _CompletedLike]


def build_argv(onex_bin: str = "onex") -> list[str]:
    """The corrected R1/MD-14 probe command.

    No `--kafka-bootstrap` literal: the CLI resolves the lab lane's broker
    address and transport from the checked-in `config/ci_bus_lanes.yaml`
    (OMN-18349's fix, the same one `tests/delegation_golden/runner.py` in
    omnimarket relies on). `--lane dev` is what makes that resolution correct
    for THIS probe -- the front door, not the stability-test proof lane.
    """
    return [onex_bin, "delegate", PROBE_PROMPT, "--bus", "kafka", "--lane", PROBE_LANE]


class ModelR1ProbeResult(BaseModel):
    """One R1 front-door probe run: the command, its verdict, and its evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    argv: list[str]
    lane: str
    exit_code: int
    ok: bool
    duration_s: float = Field(..., ge=0.0)
    stdout_tail: str
    stderr_tail: str
    timeout_reason: str | None = None


def _default_runner(
    argv: list[str], *, timeout: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,  # the probe's own exit code is the verdict, not an exception
    )


def run_probe(
    *,
    onex_bin: str = "onex",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    runner: RunnerFn = _default_runner,
) -> ModelR1ProbeResult:
    """Run one probe and grade it. Never raises: every outcome is a result."""
    argv = build_argv(onex_bin=onex_bin)
    started = time.monotonic()
    try:
        completed = runner(argv, timeout=timeout_s)
    except (subprocess.TimeoutExpired, OSError, TimeoutError) as exc:
        # TimeoutExpired/TimeoutError: the CLI hung past --timeout-s.
        # OSError (covers FileNotFoundError): no `onex` on PATH.
        duration = time.monotonic() - started
        return ModelR1ProbeResult(
            argv=argv,
            lane=PROBE_LANE,
            exit_code=2,
            ok=False,
            duration_s=duration,
            stdout_tail="",
            stderr_tail="",
            timeout_reason=str(exc),
        )
    duration = time.monotonic() - started
    return ModelR1ProbeResult(
        argv=argv,
        lane=PROBE_LANE,
        exit_code=completed.returncode,
        ok=completed.returncode == 0,
        duration_s=duration,
        stdout_tail=(completed.stdout or "")[-_TAIL_CHARS:],
        stderr_tail=(completed.stderr or "")[-_TAIL_CHARS:],
        timeout_reason=None,
    )


def record_result(result: ModelR1ProbeResult, out_path: Path) -> int:
    """Write the typed record and return the probe's own exit code (the verdict)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.model_dump(), indent=2) + "\n")
    return result.exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onex-bin", default="onex")
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--record", type=Path, default=Path("r1-front-door-probe.json"))
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    result = run_probe(onex_bin=args.onex_bin, timeout_s=args.timeout_s)
    exit_code = record_result(result, args.record)

    if result.ok:
        print(
            f"[r1-front-door-probe] OK: exit=0 in {result.duration_s:.1f}s (lane={result.lane})"
        )
    else:
        print(
            f"[r1-front-door-probe] FAIL: exit={result.exit_code} in "
            f"{result.duration_s:.1f}s (lane={result.lane}); "
            f"timeout_reason={result.timeout_reason!r}; "
            f"stderr_tail={result.stderr_tail!r}",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
