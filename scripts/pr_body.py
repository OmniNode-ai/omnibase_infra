#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Set or append a PR body via the REST API, with a read-back assertion (OMN-16839).

WHY THIS EXISTS RATHER THAN ``gh pr edit``
------------------------------------------
``gh pr edit <n> --body ...`` / ``--body-file ...`` exits **1** on OmniNode-ai
repositories: the edit path calls the deprecated **Projects-classic** GraphQL
API, which the org no longer serves. The PR body is **not written**.

That failure is in the dangerous direction. There is no output a casual
reader notices, the exit code is easy to lose (inside a ``&&`` chain, a
``|| true``, a pipeline whose last stage succeeds, or a caller that simply
does not check), and the next step -- a Receipt-Gate run, a merge, a closeout
report -- proceeds believing an ``Evidence-Source:`` / ``Evidence-Ticket:``
line is published when nothing changed. It has now been rediscovered three
separate times and worked around privately each time with an ad-hoc
``gh api --method PATCH``. This is the one committed copy of that workaround,
with the two assertions the ad-hoc form never had.

Note the exit code is not evidence in *either* direction: for ``--body`` the
write does not land (the three observations behind this ticket), while the
2026-05-25 main-cutover report recorded ``gh pr edit --base`` exiting 1 on the
same deprecation with the mutation **applied**. So "it exited 1" does not mean
nothing happened, and "it exited 0" would not mean the write landed. Only a
read-back settles it, which is why this helper performs one.

THE TWO ASSERTIONS
------------------
1. **Identity, before writing.** The caller names ``--repo OWNER/NAME --pr N``.
   The helper GETs that PR and refuses to write unless the PR the API serves
   is that PR (owner, repo, and number all matching what came back). Body
   writing automation with no target-identity check is how OCC#5621's body was
   overwritten with omnibase_infra#2589's (OMN-15564) -- cross-repo and
   unattributable.

2. **Read-back, after writing.** Success is never concluded from the mutation
   call returning. The helper re-GETs the PR and compares what the API now
   serves against what it intended to write. A write that was accepted and
   changed nothing -- precisely the ``gh pr edit`` behaviour -- exits non-zero
   and says so. **This helper never exits 0 without having confirmed the
   published body.**

APPEND IS IDEMPOTENT
--------------------
``--append`` reads the current body, adds the text separated by a blank line,
and **no-ops if the text is already present verbatim**. Evidence appends get
retried (a lane re-runs a closeout, a sweep passes twice); two
``Evidence-Source:`` lines in one body is the over-match class OMN-14675
describes, so appending the same stamp twice must not stack it.

EXIT CODES
----------
  0     the body was written AND confirmed by read-back (or --dry-run printed
        the intended body without writing)
  2     usage error (argparse)
  65    EX_DATAERR -- target-identity mismatch. Nothing was written.
  69    EX_UNAVAILABLE -- ``gh`` could not be started, or the API call failed.
        Deliberately not 127: a bare 127 with no ``pr_body:`` marker means
        *this helper* is missing, and those two must stay mechanically
        distinguishable (the lesson OMN-16822 encoded when ``flock(1)``'s
        absence produced exactly that markerless 127).
  70    EX_SOFTWARE -- the mutation was accepted but the read-back does NOT
        match. The body on GitHub is not what you asked for. This is the
        ``gh pr edit`` defect, caught.

Every line this helper originates carries a ``pr_body:`` marker on stderr.

USAGE
-----
    scripts/pr_body.py --repo OmniNode-ai/omnibase_infra --pr 2960 \
        --append --body 'Evidence-Source: OCC#5678'

    scripts/pr_body.py --repo OmniNode-ai/omnibase_infra --pr 2960 \
        --set --body-file /path/to/body.md

    ... --append --body-file -        # read the text from stdin
    ... --dry-run                     # print the intended body, write nothing
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from typing import Any

EXIT_IDENTITY_MISMATCH = 65  # EX_DATAERR
EXIT_GH_UNAVAILABLE = 69  # EX_UNAVAILABLE
EXIT_READBACK_MISMATCH = 70  # EX_SOFTWARE

_MARKER = "pr_body:"


def _log(message: str) -> None:
    """Every helper-originated line carries the marker.

    A failure WITHOUT this marker means the helper never ran -- which is the
    exact ambiguity that made the ``gh pr edit`` breakage survive three
    rediscoveries.
    """
    print(f"{_MARKER} {message}", file=sys.stderr, flush=True)


class GhError(Exception):
    """``gh`` could not be started, or returned a failure/unparseable body."""


class IdentityMismatchError(Exception):
    """The PR the API served is not the PR the caller named. Refuse to write."""


def compose_append(existing: str | None, text: str) -> str:
    """Append ``text`` to ``existing``, separated by one blank line.

    Idempotent: if ``text`` is already present verbatim, the body is returned
    unchanged, so a retried evidence append never stacks a duplicate stamp.
    """
    current = existing or ""
    if text in current:
        return current
    if not current.strip():
        return text
    return current.rstrip("\n") + "\n\n" + text


def run_gh(gh_bin: list[str], args: list[str], stdin_text: str | None = None) -> Any:
    """Invoke ``gh`` and parse its JSON stdout."""
    try:
        completed = subprocess.run(
            [*gh_bin, *args],
            capture_output=True,
            text=True,
            input=stdin_text,
            check=False,
        )
    except (FileNotFoundError, NotADirectoryError, PermissionError) as exc:
        raise GhError(
            f"could not start {gh_bin[0]!r}: {exc}. Nothing was written. "
            f"(This is exit {EXIT_GH_UNAVAILABLE}, NOT 127 -- a bare 127 with "
            f"no {_MARKER!r} line means this helper itself is missing.)"
        ) from exc
    except OSError as exc:  # pragma: no cover - defensive
        raise GhError(f"could not start {gh_bin[0]!r}: {exc}") from exc

    if completed.returncode != 0:
        raise GhError(
            f"`gh {' '.join(args)}` exited {completed.returncode}: "
            f"{completed.stderr.strip() or '(no stderr)'}"
        )
    try:
        return json.loads(completed.stdout)
    except ValueError as exc:
        raise GhError(
            f"`gh {' '.join(args)}` returned unparseable output: "
            f"{completed.stdout[:400]!r}"
        ) from exc


def fetch_pull(gh_bin: list[str], repo: str, number: int) -> dict[str, Any]:
    payload = run_gh(gh_bin, ["api", f"repos/{repo}/pulls/{number}"])
    if not isinstance(payload, dict):
        raise GhError(f"expected an object for repos/{repo}/pulls/{number}")
    return payload


def assert_identity(payload: dict[str, Any], repo: str, number: int) -> None:
    """Refuse to write unless the served PR is the one the caller named.

    ``html_url`` is used as the identity of record because it carries owner,
    repo, AND number in one un-spoofable-by-accident string; a body write
    aimed by number alone is how a body reached the wrong repository's PR.
    """
    served_url = str(payload.get("html_url", ""))
    expected_suffix = f"/{repo}/pull/{number}"
    served_number = payload.get("number")
    if not served_url.endswith(expected_suffix) or served_number != number:
        raise IdentityMismatchError(
            f"REFUSING TO WRITE: asked for {repo}#{number} but the API served "
            f"number={served_number!r} url={served_url!r}. Nothing was written. "
            f"Check the --repo/--pr you passed before retrying."
        )


def patch_body(gh_bin: list[str], repo: str, number: int, body: str) -> None:
    """PATCH the body via REST.

    The payload goes over stdin as JSON rather than ``--field body=...`` so
    that newlines, backticks, and leading dashes in an evidence block survive
    untouched.
    """
    run_gh(
        gh_bin,
        [
            "api",
            "--method",
            "PATCH",
            f"repos/{repo}/pulls/{number}",
            "--input",
            "-",
        ],
        stdin_text=json.dumps({"body": body}),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pr_body.py",
        description=(
            "Set or append a PR body through the REST API with a read-back "
            "assertion. Replaces `gh pr edit --body/--body-file`, which exits 1 "
            "on the deprecated Projects-classic GraphQL call and writes NOTHING."
        ),
        epilog=(
            "Examples:\n"
            "  scripts/pr_body.py --repo OmniNode-ai/omnibase_infra --pr 2960 \\\n"
            "      --append --body 'Evidence-Source: OCC#5678'\n"
            "  scripts/pr_body.py --repo OmniNode-ai/omnibase_infra --pr 2960 \\\n"
            "      --set --body-file body.md\n"
            "\n"
            "Exit codes: 0 written AND confirmed | 2 usage | 65 identity "
            "mismatch (nothing written) | 69 gh unavailable | 70 read-back "
            "mismatch -- the write silently did nothing.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo", required=True, help="OWNER/NAME, e.g. OmniNode-ai/omnibase_infra"
    )
    parser.add_argument("--pr", required=True, type=int, help="pull request number")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--set", action="store_true", help="replace the body entirely")
    mode.add_argument(
        "--append",
        action="store_true",
        help="append to the existing body (idempotent: no-op if already present)",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--body", help="the literal text")
    source.add_argument(
        "--body-file", help="read the text from this path, or '-' for stdin"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the body that WOULD be published; write nothing",
    )
    parser.add_argument(
        "--gh-bin",
        default="gh",
        help="the gh executable (shell-split; for tests/wrappers). Default: gh",
    )
    return parser


def read_text_argument(args: argparse.Namespace) -> str:
    body: str | None = args.body
    if body is not None:
        return body
    body_file: str = args.body_file
    if body_file == "-":
        return sys.stdin.read()
    with open(body_file, encoding="utf-8") as handle:
        return handle.read()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    gh_bin = shlex.split(args.gh_bin)
    if not gh_bin:
        parser.error("--gh-bin cannot be empty")

    repo: str = args.repo
    number: int = args.pr
    text = read_text_argument(args)

    try:
        current = fetch_pull(gh_bin, repo, number)
        assert_identity(current, repo, number)
    except IdentityMismatchError as exc:
        _log(str(exc))
        return EXIT_IDENTITY_MISMATCH
    except GhError as exc:
        _log(f"{exc}")
        return EXIT_GH_UNAVAILABLE

    existing = current.get("body") or ""
    intended = text if args.set else compose_append(existing, text)

    if args.dry_run:
        _log(f"--dry-run: nothing written to {repo}#{number}")
        sys.stdout.write(intended)
        if not intended.endswith("\n"):
            sys.stdout.write("\n")
        return 0

    if intended == existing:
        _log(f"{repo}#{number} already carries this body verbatim; nothing to do")
        return 0

    try:
        patch_body(gh_bin, repo, number, intended)
    except GhError as exc:
        _log(f"the PATCH failed: {exc}. The body was NOT changed.")
        return EXIT_GH_UNAVAILABLE

    # The whole point of this helper: do not believe the write.
    try:
        served = fetch_pull(gh_bin, repo, number)
        assert_identity(served, repo, number)
    except IdentityMismatchError as exc:  # pragma: no cover - defensive
        _log(f"read-back identity check failed AFTER writing: {exc}")
        return EXIT_IDENTITY_MISMATCH
    except GhError as exc:
        _log(
            f"the PATCH was accepted but the read-back could not be performed: "
            f"{exc}. Treat the body as UNCONFIRMED and re-run."
        )
        return EXIT_READBACK_MISMATCH

    if (served.get("body") or "") != intended:
        _log(
            f"FAILED: the PATCH to {repo}#{number} was accepted but the read-back "
            f"does NOT match what was requested -- the published body is unchanged "
            f"or different. This is the `gh pr edit` silent-no-op class. Do NOT "
            f"report the body as written."
        )
        return EXIT_READBACK_MISMATCH

    _log(f"wrote and confirmed the body of {repo}#{number} ({len(intended)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
