#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Mint a lane-unique scratch path so parallel lanes stop clobbering each other (OMN-16842).

THE COLLISIONS THIS EXISTS FOR
------------------------------
Parallel lanes in one session are handed the **same** scratchpad directory.
Left to themselves, lanes independently converge on the same obvious
basenames, and three collisions were observed in a single day (2026-08-27):

* a shared log filename -- one lane's run log overwritten by a peer's;
* a shared ``msg.txt`` -- a lane read a peer's message as its own;
* six lanes each writing a private ``heavy.py`` wrapper to the same name
  (the friction that produced OMN-16822).

A collision here does not crash. It produces a *plausible* artifact: the read
succeeds, the content is well-formed, and it belongs to somebody else. Two of
the three were caught by a human noticing the result was wrong, not by any
mechanism. Cf. OMN-15678, where a peer's ``pr_body.md`` was published into
another lane's PR and a peer overwrote a gate lane's RED-before capture.

WHY A HELPER RATHER THAN THE STANDING PROSE
-------------------------------------------
The dispatch brief already said "use lane-unique scratchpad filenames". That
is a rule, not a mechanism (memory ``feedback_a_rule_is_not_a_mechanism``): a
forgotten prefix silently reinstates the bug, and forgetting is exactly what
happened three times. Uniqueness here is **structural**, not remembered:

* the basename carries ``<label>-<pid>-<random>``, with the random component
  drawn from ``secrets`` -- so two lanes asking for the identical label at the
  identical moment still get distinct paths;
* the path is created with ``O_EXCL``, so a collision is a hard ``FileExists``
  error and can never present as a successful reuse of a peer's file;
* an omitted or empty ``--label`` still yields an isolated path. **A forgotten
  scope degrades to isolated, never to shared** -- the scaffolding requirement
  OMN-15678 AC3 states -- because a bare shared default is the original bug;
* the label survives in the filename, so a stray artifact is attributable to
  the lane that wrote it. That attribution is precisely what was missing when
  the three incidents had to be diagnosed by hand.

SCOPE -- WHAT THIS DOES **NOT** DO
----------------------------------
This makes collisions structurally impossible and stray artifacts
attributable. It does **not** make a peer's path unreadable: a lane that
deliberately opens another lane's file still can. Harness-side lane-scoped
scratchpad hand-off (correct by construction) remains open on OMN-15678.

EXIT CODES
----------
  0     a path was minted; it is printed to **stdout** and nothing else is
  2     usage error (argparse)
  73    EX_CANTCREAT -- the path could not be created (``--exact`` target
        already taken, or the root is not writable). Nothing was reused.

Every line this helper originates carries a ``lane_scratch:`` marker on
stderr, so "the helper ran and refused" stays distinguishable from "the
helper is missing" (the lesson OMN-16822 encoded).

USAGE
-----
    LOG=$(scripts/lane_scratch.py --label "OMN-16842 pytest" --suffix .log)
    uv run pytest -q > "$LOG" 2>&1

    DIR=$(scripts/lane_scratch.py --label friction-pair --dir)

The root defaults to ``$CLAUDE_SCRATCHPAD_DIR`` when set, else
``$OMNI_HOME/.onex_state/lane_scratch``. It is never ``/tmp`` (memory
``feedback_no_tmp_use_workspace``).
"""

from __future__ import annotations

import argparse
import os
import re
import secrets
import sys
from pathlib import Path

EXIT_PATH_TAKEN = 73  # EX_CANTCREAT

_MARKER = "lane_scratch:"

#: Enough entropy that same-label, same-pid, same-instant mints do not collide
#: in practice; ``O_EXCL`` is what makes it safe rather than merely unlikely.
_SUFFIX_BYTES = 5

_UNLABELLED = "lane"

_SAFE_LABEL = re.compile(r"[^A-Za-z0-9._-]+")

#: ``<label>-<pid>-<random>``: the two trailing fields are the uniqueness, so
#: stripping them recovers the label for attribution.
_MINTED = re.compile(r"^(?P<label>.+)-(?P<pid>\d+)-(?P<rand>[0-9a-f]+)$")


def _log(message: str) -> None:
    """Marker-carrying stderr line; stdout stays reserved for the path alone."""
    print(f"{_MARKER} {message}", file=sys.stderr, flush=True)


def sanitize_label(label: str | None) -> str:
    """Make ``label`` filename-safe while keeping it recognisable.

    A missing, empty, or all-whitespace label does NOT collapse to a shared
    bare name -- it becomes a generic stem that still carries pid+random, so
    a forgotten label is isolated rather than shared.
    """
    cleaned = _SAFE_LABEL.sub("-", (label or "").strip()).strip("-")
    return cleaned or _UNLABELLED


def default_root() -> Path:
    """The scratch root. Deliberately never ``/tmp``."""
    from_env = os.environ.get("CLAUDE_SCRATCHPAD_DIR")
    if from_env:
        return Path(from_env)
    omni_home = os.environ.get("OMNI_HOME")
    if omni_home:
        return Path(omni_home) / ".onex_state" / "lane_scratch"
    return Path.home() / ".onex_state" / "lane_scratch"


def mint_name(label: str | None, suffix: str = "") -> str:
    return (
        f"{sanitize_label(label)}-{os.getpid()}-{secrets.token_hex(_SUFFIX_BYTES)}"
        f"{suffix}"
    )


def label_of(path: Path) -> str:
    """Recover the label from a minted path, for attributing a stray artifact.

    Returns an empty string for a path this helper did not mint, rather than
    guessing -- an un-attributable artifact must read as un-attributable.
    """
    stem = path.name
    while True:
        root, dot, _ = stem.rpartition(".")
        if not dot:
            break
        stem = root
    match = _MINTED.match(stem)
    return match.group("label") if match else ""


def create_exclusive(path: Path, *, as_dir: bool = False) -> Path:
    """Create ``path``, failing loudly if it already exists.

    ``O_EXCL`` (and ``mkdir``'s equivalent) is the point: a collision must
    never be observable as a successful reuse of somebody else's file, which
    is the exact shape of all three recorded incidents.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if as_dir:
        path.mkdir()  # raises FileExistsError if taken
        return path
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    os.close(fd)
    return path


def mint_path(
    *,
    root: Path,
    label: str | None,
    suffix: str = "",
    as_dir: bool = False,
) -> Path:
    """Mint and create a lane-unique path under ``root``."""
    return create_exclusive(root / mint_name(label, suffix), as_dir=as_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lane_scratch.py",
        description=(
            "Mint a lane-unique scratch file or directory and print its path. "
            "Structural uniqueness (label + pid + secrets suffix, created "
            "O_EXCL) so parallel lanes cannot clobber each other's artifacts."
        ),
        epilog=(
            "Examples:\n"
            '  LOG=$(scripts/lane_scratch.py --label "OMN-1234 pytest" '
            "--suffix .log)\n"
            "  DIR=$(scripts/lane_scratch.py --label my-lane --dir)\n"
            "\n"
            "Exit codes: 0 minted (path on stdout) | 2 usage | 73 could not "
            "create (nothing reused).\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--label",
        default=None,
        help=(
            "short lane identifier kept in the filename for attribution; "
            "omitting it still yields an isolated path, never a shared one"
        ),
    )
    parser.add_argument("--suffix", default="", help="filename suffix, e.g. .log")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="directory to mint under (default: $CLAUDE_SCRATCHPAD_DIR, else "
        "$OMNI_HOME/.onex_state/lane_scratch)",
    )
    parser.add_argument(
        "--dir",
        action="store_true",
        dest="as_dir",
        help="mint a lane-private directory instead of a file",
    )
    parser.add_argument(
        "--exact",
        default=None,
        help=(
            "create this exact basename under the root instead of minting one. "
            "Escape hatch for a caller that must control the name: it FAILS "
            "(exit 73) if the name is taken rather than reusing a peer's file"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.exact and (args.label or args.suffix):
        parser.error("--exact cannot be combined with --label/--suffix")

    root: Path = args.root if args.root is not None else default_root()

    try:
        if args.exact:
            minted = create_exclusive(root / args.exact, as_dir=args.as_dir)
        else:
            minted = mint_path(
                root=root,
                label=args.label,
                suffix=args.suffix,
                as_dir=args.as_dir,
            )
    except FileExistsError:
        _log(
            f"REFUSING TO REUSE: {root / (args.exact or '')} already exists. "
            f"Nothing was written and no peer's file was touched. Drop --exact "
            f"and let this helper mint a unique name."
        )
        return EXIT_PATH_TAKEN
    except OSError as exc:
        _log(f"could not create a scratch path under {root}: {exc}")
        return EXIT_PATH_TAKEN

    print(minted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
