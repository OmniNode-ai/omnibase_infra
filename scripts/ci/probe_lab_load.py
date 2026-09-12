# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Write the lab-load record and report honestly whether it is a measurement (OMN-18253).

WHAT WAS WRONG, precisely
    The ``lab-load-probe`` job ran its probe from an inline heredoc and had two
    failure shapes, both of which read as success:

    1. ``probe_lab_saturation_from_fleet`` NEVER raises. Every failure --
       missing token, HTTP error, malformed JSON, unreadable memory -- comes
       back as ``{"ok": false, "error": ...}``. The heredoc printed that to the
       record and exited 0, so a probe that measured NOTHING produced a green
       step and a green run. Only someone opening the artifact could tell.
    2. When the interpreter itself died -- the ``ModuleNotFoundError: No module
       named 'yaml'`` that killed 10 of the job's first 12 runs -- the shell's
       ``> lab-load.json`` redirect had already created the file, so the
       artifact was ZERO BYTES. The job's ``continue-on-error: true`` then kept
       even that out of the run's conclusion.

    Between them: a job that could not distinguish "the lab is quiet", "I could
    not reach the API", and "I crashed before writing anything".

WHAT THIS DOES
    One record is always written, and the exit status states which of the three
    happened:

    * a measurement -> ``ok: true`` with hosts, exit 0;
    * a probe that ran and could not measure -> ``ok: false`` with the probe's
      own ``error``, exit 1;
    * an unexpected exception -> ``ok: false`` with ``unhandled:<type>``, exit 1,
      and the record is still written, so the artifact is never zero bytes.

    The exit status is what reaches the job's conclusion once the job-level
    suppression is gone. The record is what reaches the artifact. Both now say
    the same thing, which is the whole of this ticket's AC3.

WHAT THIS DELIBERATELY DOES NOT DO: make a busy lab an outage
    A saturated lab is a MEASUREMENT, not a failure. ``probe_lab_saturation_
    from_fleet`` returns ``ok: true`` with a high ``ratio`` when every runner is
    busy, and this module passes that through as exit 0. The suppression it
    replaces existed for a stated reason -- "this job failing IS a data point,
    not an outage" -- and that reason survives: what fails here is the probe
    being unable to answer, never the answer being bad news. A regression test
    pins exactly that.

    The OMN-18247 artifact assertion stays in the job as a backstop. This module
    makes a zero-byte record nearly unreachable; the assertion is what still
    catches the case where even this writer never ran.
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
import sys
from pathlib import Path
from typing import Any


def build_record(
    probe: Any, token: str | None, runner_group: str, api_url: str
) -> dict[str, Any]:
    """Call ``probe`` and return its record, converting any escape into a record.

    ``probe`` is injected rather than imported at module scope so the regression
    tests can drive every branch -- including the raising one -- without a
    network, and so a failure to import the probe itself is reported as a record
    rather than as a traceback with no artifact behind it.
    """
    try:
        record = probe(token, runner_group, api_url)
    except Exception as exc:  # noqa: BLE001 -- every escape must still leave a record
        return {"ok": False, "error": f"unhandled:{type(exc).__name__}: {exc}"}
    if not isinstance(record, dict):
        return {"ok": False, "error": f"unexpected_return_type:{type(record).__name__}"}
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write the lab-load record and exit non-zero when it is not a measurement.",
    )
    parser.add_argument("--out", type=Path, default=Path("lab-load.json"))
    parser.add_argument("--runner-group", default="omnibase-ci")
    # fmt: off
    parser.add_argument("--api-url", default="https://api.github.com")  # url-authority-ok: Actions-injected GitHub API base, passed through to the same probe that already pins this literal in runner_route_decision.py; it addresses no ONEX service and carries no routing authority.
    # fmt: on
    parser.add_argument("--token", default="")
    parser.add_argument(
        "--probe-module",
        default="runner_route_decision",
        help=(
            "Module supplying probe_lab_saturation_from_fleet. The seam exists so "
            "the regression tests can drive EVERY branch of this module -- the "
            "raising one and the unimportable one included -- with no network. A "
            "module whose failure paths can only be exercised by a live workflow "
            "run is the shape that let this job stay dead for its whole life."
        ),
    )
    args = parser.parse_args(argv)

    # Imported here, not at module scope: an ImportError on this line is itself
    # one of the failures this module exists to record rather than crash on.
    try:
        sys.path.append(str(Path(__file__).resolve().parent))
        module = importlib.import_module(args.probe_module)
        probe = module.probe_lab_saturation_from_fleet
    except Exception as exc:  # noqa: BLE001 -- see above
        record: dict[str, Any] = {
            "ok": False,
            "error": f"probe_unimportable:{type(exc).__name__}: {exc}",
        }
    else:
        record = build_record(
            probe, args.token or None, args.runner_group, args.api_url
        )

    record["sampled_at"] = datetime.datetime.now(datetime.UTC).isoformat()
    serialised = json.dumps(record)
    args.out.write_text(serialised + "\n", encoding="utf-8")
    print(serialised)

    if record.get("ok"):
        return 0
    print(
        f"lab-load probe did not measure: {record.get('error', 'unknown')}. "
        "The record is written and uploaded either way; this non-zero exit is "
        "what makes the job's own conclusion say so (OMN-18253). A BUSY lab is "
        "not this case -- it returns ok:true with a high ratio and exits 0.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
