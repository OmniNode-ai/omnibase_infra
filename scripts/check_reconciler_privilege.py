#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Gate: a reconciler resolves its tools explicitly and writes as the surface owner (OMN-17335).

CLAUDE.md rule 5 -- a check that is not wired as a gate is advisory and gets
ignored -- so this ships in the same PR as the fix it protects.

It closes two defects that were found together on ``.201``, both of which are
structural and neither of which a comment can satisfy.

**1. Tool resolution may not be PATH-only.**

The first live run of ``reconcile-host.sh`` on ``.201`` refused with "``uv`` is
not on PATH", and the ticket filed against it concluded that uv was not
installed on the host. It was installed -- under the operator's ``~/.local/bin``
-- and a non-interactive shell simply never sources the profile that puts that
directory on PATH. The cron.d unit pins an explicit minimal PATH, so a
user-local install is unreachable *by construction*.

That is an empty result being read as evidence of absence, inside the very
family of scripts built to stop exactly that. So the venv reconciler must
resolve ``uv`` from an ordered candidate list -- not from PATH alone -- and this
gate pins the two properties that make the list real: the environment override
exists, and a ``resolve_uv`` function exists to consult it.

**2. Package operations may not write as the wrong user.**

The ``.201`` cron unit runs as root; the venv and every file in it are owned by
the operator. "Fixing" defect 1 by putting uv on root's PATH would have had root
write root-owned files into a user-owned venv, after which the owner's own
reconcile fails on permissions -- trading a loud, correct failure for a quiet,
latent one.

So every command that installs or syncs packages must go through the
``as_owner`` helper. This half is fully structural: it reads the actual
invocation lines, not a declaration about them. Adding a new ``uv sync`` without
``as_owner`` fails the build.

**3. What the gate SCANS may not be a filename pattern (OMN-17443).**

Both rules above were enforced over scripts discovered by the glob
``scripts/reconcile*.sh``. On 2026-09-01 the `.201` clone surface was repaired
(2010 root-owned paths, 0 -> 0 on the next root tick) and re-broke thirty
minutes later: 32 root-owned objects, all stamped ``:37``, deposited by
``deploy/maintenance/omninode-host-maintenance-sync.sh`` -- which fetched as
root into an operator-owned clone for as long as it existed, and which this
gate had never opened, because it is not called ``reconcile*.sh``.

That is the OMN-17383 lesson recurring inside the same epic: a gate whose scope
stops at a filename pattern gives false assurance about the file it never
scanned. The answer is not a second pattern. Discovery now follows the
INVOCATION -- every command a shipped ``deploy/**/cron.d/`` unit runs, resolved
through the host-artifact manifest to the repo file it is installed from -- and
a scheduled command that resolves to no repo file is a FAILURE, not a skip, so
"install an unlisted script and schedule it" is not a way back to green.

Usage:
    python scripts/check_reconciler_privilege.py [--repo-root PATH]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Same glob as check_reconciler_movement_proof.py, and for the same reason: a
# hand-maintained list is a thing you forget to update, and an artifact absent
# from a manifest is unguarded with nobody finding out (OMN-15525).
#
# NOT the whole discovery surface -- see discover_scheduled_host_scripts below.
# A glob over `scripts/` was the ENTIRE scope until OMN-17443, and the file that
# still carried the defect was the one file it never opened.
RECONCILER_GLOBS = ("scripts/reconcile*.sh", "scripts/**/reconcile*.sh")

# Cron units this repo ships for the `.201` host, and the manifest that maps the
# paths they invoke back to the repo files those paths are installed from.
CRON_UNIT_GLOB = "deploy/**/cron.d/*"
HOST_ARTIFACT_MANIFEST = "deploy/maintenance/omninode-host-maintenance-sync.sh"

# A crontab command line: five schedule fields, a user, then the command.
# The user field is captured but NOT filtered on. The defect is "a scheduled job
# writes into a tree it may not own"; narrowing to `root` would leave a dodge
# (schedule it as any other user) for the sake of a distinction that does not
# change the damage.
_CRON_COMMAND = re.compile(
    r"^\s*(?:@\w+|(?:[-\dA-Za-z*/,]+\s+){5})(?P<user>[A-Za-z_][-\w]*)\s+(?P<command>\S+)"
)

# A `relpath|hostpath|mode` row of the host-artifact MANIFEST array.
_MANIFEST_ENTRY = re.compile(r'"([^"|]+)\|([^"|]+)\|[0-7]{3,4}"')

VENV_RECONCILER = "scripts/reconcile-workspace-venvs.sh"
OWNER_HELPER = "as_owner"
# The one file allowed to define OWNER_HELPER. Everything else sources it.
PRIVILEGE_LIB = "reconcile_privilege_lib.sh"

# A git subcommand that WRITES into the repository it is pointed at. `fetch`
# earns its place here despite reading from the remote: it writes objects, refs
# and reflogs locally, which is exactly what accumulated as root on `.201`.
# Read-only plumbing (`rev-parse`, `cat-file`, `config --get`) is deliberately
# absent -- guarding a read would be noise, and noise is what gets allowlisted.
#
# The intervening-token allowance matters: this runs on the UNQUOTED line, so
# `git -C "$OMNI_HOME/$repo" fetch` arrives as `git -C   fetch` with the path
# already stripped. A pattern that insisted on a concrete argument after `-C`
# matched nothing at all -- the gate reported OK on the exact line it exists to
# catch.
_GIT_WRITE = re.compile(
    r"(?<![\w./$-])git\s+(?:-[A-Za-z-]+\s+|\S+\s+){0,2}?"
    r"(?:fetch|pull|checkout|reset|clone|switch|merge|gc|prune|remote)\b"
)

# Wherever the clone delegate is EXECUTED. Followed by invocation rather than by
# filename, the OMN-17383 lesson: a gate whose scope stops at a filename pattern
# gives false assurance about the file it never scanned.
#
# An interpreter must appear before the variable, because the script also tests
# the delegate for existence (`[[ ! -f "$CLONE_DELEGATE" ]]`) and names it in the
# UNCOVERED record. Neither runs anything, and flagging them would train the next
# reader to reach for an allowlist -- which is how a gate stops meaning anything.
_CLONE_DELEGATE_INVOCATION = re.compile(
    r"(?:bash|sh|exec|source)\s[^\n]*\$\{?CLONE_DELEGATE\}?"
)

_RESOLVE_UV_DEF = re.compile(r"^\s*resolve_uv\s*\(\s*\)\s*\{", re.MULTILINE)
_UV_OVERRIDE = "ONEX_RECONCILE_UV_BIN"

# A line that actually RUNS a package operation. Both forms reference the
# resolved variable, which is what distinguishes an invocation from the many
# `say`/`trace`/comment lines that merely quote the command for a human.
_UV_INVOCATION = re.compile(r"\$\{?UV_BIN\}?")
_INSTALL_INVOCATION = re.compile(r"\$\{?INSTALL_SCRIPT\}?")

# `uv` reached by name rather than through the resolved path -- the PATH-only
# defect re-entering through a back door.
_BARE_UV = re.compile(r"(?<![\w./$-])uv\s+(?:sync|pip|run|venv|tool)\b")

_QUOTED = re.compile(r"\"[^\"]*\"|'[^']*'")

# Every refusal in this family names the exact command to run by hand, so the
# forbidden invocation appears verbatim inside message strings. Those lines are
# documentation and must not be flagged. They are identifiable by position, not
# by content: a message is an ARGUMENT to one of these helpers, or a quoted
# continuation of one, so it starts with the helper name or with a quote --
# while a real invocation starts with `if`, `bash`, `(cd`, `env`, and friends.
_MESSAGE_LINE = re.compile(r"^\s*(?:say|trace|fail|echo|printf|record)\b|^\s*[\"']")


def discover_reconcilers(repo_root: Path) -> list[Path]:
    found: set[Path] = set()
    for pattern in RECONCILER_GLOBS:
        found.update(p for p in repo_root.glob(pattern) if p.is_file())
    return sorted(found)


def host_artifact_map(repo_root: Path) -> dict[str, str]:
    """installed host path -> repo-relative path, read from the sync manifest.

    That manifest already exists and is already load-bearing: an artifact absent
    from it is not installed, not checked, and drifts invisibly (OMN-15525). So
    it is the honest place to learn which repo file a scheduled command is
    actually running -- rather than a second list here, which would drift from
    it and be the same defect one layer up.
    """
    manifest = repo_root / HOST_ARTIFACT_MANIFEST
    if not manifest.is_file():
        return {}
    text = manifest.read_text(encoding="utf-8")
    return {host: rel for rel, host in _MANIFEST_ENTRY.findall(text)}


def discover_scheduled_host_scripts(repo_root: Path) -> tuple[list[Path], list[str]]:
    """Repo files that a shipped cron unit executes, plus discovery failures.

    OMN-17443. `discover_reconcilers` finds scripts by NAME. The `:37` host
    maintenance sync fetched as root into an operator-owned clone for as long as
    it existed and was never once scanned, because it lives in
    `deploy/maintenance/` and is not called `reconcile*.sh` -- the OMN-17383
    lesson (a gate whose scope stops at a filename pattern gives false
    assurance) recurring inside the same epic.

    So discovery follows the INVOCATION: whatever a cron unit in this repo runs
    is in scope, resolved through the host-artifact manifest to the repo file it
    is installed from.

    Failures are returned rather than raised, and an unresolvable command is a
    FAILURE rather than a skip. Giving up quietly on a command that maps to no
    repo file would reopen the original hole through a new door -- schedule an
    unlisted script and the gate is once again reporting OK on a file it never
    opened.
    """
    mapping = host_artifact_map(repo_root)
    found: set[Path] = set()
    failures: list[str] = []

    for unit in sorted(p for p in repo_root.glob(CRON_UNIT_GLOB) if p.is_file()):
        rel_unit = unit.relative_to(repo_root)
        for number, raw in enumerate(
            unit.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw.strip()
            if not line or line.startswith("#") or "=" in line.split()[0]:
                continue
            match = _CRON_COMMAND.match(line)
            if not match:
                failures.append(
                    f"{rel_unit}:{number}: cannot parse the scheduled command out of "
                    f"{line!r}. An unparseable unit is an unscanned script, which is "
                    "the OMN-17443 condition."
                )
                continue
            command = match.group("command")
            relpath = mapping.get(command)
            if relpath is None:
                failures.append(
                    f"{rel_unit}:{number}: runs {command} as '{match.group('user')}', "
                    f"but no entry in {HOST_ARTIFACT_MANIFEST} maps that path to a "
                    "repo file. An unmapped host artifact is neither installed by a "
                    "sanctioned path nor scanned by this gate -- exactly the file "
                    "that carried the defect in OMN-17443."
                )
                continue
            target = repo_root / relpath
            if not target.is_file():
                failures.append(
                    f"{rel_unit}:{number}: {command} maps to {relpath}, which does not "
                    "exist in this repository. The mapping must resolve to something "
                    "this gate can actually read."
                )
                continue
            found.add(target)

    return sorted(found), failures


def discover_guarded_scripts(repo_root: Path) -> tuple[list[Path], list[str]]:
    """Every script whose writes this gate governs: reconcilers AND scheduled hosts."""
    scheduled, failures = discover_scheduled_host_scripts(repo_root)
    return sorted(set(discover_reconcilers(repo_root)) | set(scheduled)), failures


def logical_lines(text: str) -> list[tuple[int, str]]:
    """(starting line number, joined text) for each backslash-continued command.

    Analysing physical lines gets this wrong in both directions. A command split
    across a continuation puts ``as_owner`` on the first line and the tool on the
    second, so the second reads as an unguarded invocation; and a multi-line
    ``fail`` block puts the quoted example command on its own line, where it
    reads as an invocation rather than as the message text it is.
    """
    joined: list[tuple[int, str]] = []
    buffer = ""
    start = 0
    for number, line in enumerate(text.splitlines(), start=1):
        if not buffer:
            start = number
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            buffer += stripped[:-1] + " "
            continue
        joined.append((start, buffer + line))
        buffer = ""
    if buffer:
        joined.append((start, buffer))
    return joined


def _code(line: str) -> str:
    """The line as bash would run it, minus whole-line comments.

    Quoted strings are deliberately KEPT here: the invocations this gate looks
    for reference their tool as ``"$UV_BIN"`` -- quoted -- so stripping literals
    would erase the exact thing being detected. (It did, in the first draft, and
    Part 2 silently matched nothing.)
    """
    if line.strip().startswith("#"):
        return ""
    if _MESSAGE_LINE.search(line):
        return ""
    return line


def _unquoted(line: str) -> str:
    """``_code`` with string literals removed, for finding a tool called by name.

    Every refusal in these scripts prints the exact command to run by hand, so
    ``fail "... uv sync --frozen ..."`` appears verbatim in message strings.
    Those are documentation, not invocations, and must not be flagged.
    """
    return _QUOTED.sub(" ", line)


def check(repo_root: Path) -> list[str]:
    failures: list[str] = []

    # -- Part 0: what this gate is allowed to say it scanned ---------------- #
    #
    # OMN-17443. Resolved once, up front, so every part below governs the same
    # set and a discovery hole cannot be a silent pass in one part and not
    # another. `guarded` is reconcilers PLUS whatever a shipped cron unit
    # actually runs; `discovery_failures` are commands this gate could not
    # resolve to a file, which are reported rather than skipped.
    guarded, discovery_failures = discover_guarded_scripts(repo_root)
    failures.extend(discovery_failures)

    # -- Part 1: explicit tool resolution ----------------------------------- #
    venv_reconciler = repo_root / VENV_RECONCILER
    if not venv_reconciler.is_file():
        failures.append(
            f"{VENV_RECONCILER} is missing. It is the only script that installs "
            "packages into the governed venvs."
        )
    else:
        source = venv_reconciler.read_text(encoding="utf-8")
        if not _RESOLVE_UV_DEF.search(source):
            failures.append(
                f"{VENV_RECONCILER}: no resolve_uv() function. `uv` must be "
                "resolved from an ordered candidate list. PATH-only resolution "
                "cannot see a user-local install from a cron PATH, which is how "
                "OMN-17335 concluded a tool that was installed was missing."
            )
        if _UV_OVERRIDE not in source:
            failures.append(
                f"{VENV_RECONCILER}: no {_UV_OVERRIDE} candidate. Without an "
                "explicit override there is no way to point the reconciler at a "
                "uv that PATH does not reach, and no way to test the resolution "
                "order hermetically."
            )

    # -- Part 2: every package operation writes as the surface owner -------- #
    for script in guarded:
        rel = script.relative_to(repo_root)
        text = script.read_text(encoding="utf-8")
        for number, line in logical_lines(text):
            executable = _code(line)
            if not executable.strip():
                continue

            bare = _BARE_UV.search(_unquoted(executable))
            if bare:
                failures.append(
                    f"{rel}:{number}: invokes `{bare.group(0).strip()}` by name "
                    "instead of through the resolved $UV_BIN. Whichever uv PATH "
                    "happens to hold is not necessarily the one resolve_uv() "
                    "proved usable."
                )

            runs_uv = bool(_UV_INVOCATION.search(executable)) and " sync" in executable
            runs_install = (
                bool(_INSTALL_INVOCATION.search(executable))
                and "--execute" in executable
            )
            if not (runs_uv or runs_install):
                continue
            if OWNER_HELPER in executable:
                continue

            what = "a uv sync" if runs_uv else "the provider co-install"
            failures.append(
                f"{rel}:{number}: runs {what} without {OWNER_HELPER}. Package "
                "operations must run as the owner of the venv they write. A root "
                "cron writing into a user-owned venv leaves files its owner "
                "cannot manage, turning a loud failure into a latent one "
                "(OMN-17335)."
            )

    # -- Part 3: the resolved uv reaches the delegates too ------------------ #
    #
    # OMN-17383. Parts 1 and 2 govern the reconciler's OWN uv calls. They said
    # nothing about the script it shells out to, and `install-node-skill-package.sh`
    # calls bare `uv` -- so on `.201` the co-install died with "uv: command not
    # found" after resolve_uv() had already located the binary one process up.
    #
    # The discovery rule is the real lesson: Part 2 finds scripts by the glob
    # `scripts/reconcile*.sh`, and the co-install does not match it, so the file
    # that still had the defect was the one file never scanned. A gate whose
    # scope stops at a filename pattern gives false assurance (the OMN-15525
    # shape). This part follows the INVOCATION instead: wherever the reconciler
    # executes the co-install, that line must hand the child the resolved uv.
    if venv_reconciler.is_file():
        source = venv_reconciler.read_text(encoding="utf-8")
        for number, line in logical_lines(source):
            executable = _code(line)
            if (
                not _INSTALL_INVOCATION.search(executable)
                or "--execute" not in executable
            ):
                continue
            if "PATH=" in executable and "UV_BIN" in executable:
                continue
            failures.append(
                f"{VENV_RECONCILER}:{number}: invokes the provider co-install "
                "without putting the resolved uv on the child's PATH. That "
                "child calls bare `uv` and inherits the caller's PATH, which "
                "under cron cannot reach a user-local install -- so the parent "
                "resolves uv successfully and the child still fails "
                '(OMN-17383). Pass PATH="$(dirname "$UV_BIN"):$PATH".'
            )

    # -- Part 4: the clone surface obeys the same rule (OMN-17366) ---------- #
    #
    # Parts 1-3 govern package operations. They said nothing about `git`, and the
    # clone surface had the identical defect for as long as it existed: the root
    # `:19` cron fetched and checked out into operator-owned clones, leaving 1118
    # root-owned paths across `.201`'s five deploy-source trees. A plain operator
    # `git fetch` then fails intermittently with "insufficient permission for
    # adding an object to repository database".
    #
    # Two write paths, and guarding either alone fixes nothing:
    #   * the reconciler's own `git fetch` (it fetches to establish targets, in
    #     BOTH modes -- so `--check` is a writer here too);
    #   * the clone delegate, which fetches AND checks out, and is the larger of
    #     the two.
    for script in guarded:
        rel = script.relative_to(repo_root)
        text = script.read_text(encoding="utf-8")
        for number, line in logical_lines(text):
            executable = _code(line)
            if not executable.strip():
                continue

            writes_a_clone = bool(_GIT_WRITE.search(_unquoted(executable)))
            runs_clone_delegate = bool(_CLONE_DELEGATE_INVOCATION.search(executable))
            if not (writes_a_clone or runs_clone_delegate):
                continue
            if OWNER_HELPER in executable:
                continue

            what = (
                "a git fetch/checkout into a clone"
                if writes_a_clone
                else "the clone delegate"
            )
            failures.append(
                f"{rel}:{number}: runs {what} without {OWNER_HELPER}. `git fetch` "
                "writes objects, refs and reflogs, so a root cron writing into "
                "an operator-owned clone leaves objects its owner cannot write "
                "past -- an intermittent, later failure instead of a loud one "
                "(OMN-17366)."
            )

    # -- Part 5: the owner helper is defined exactly once, and shared ------- #
    #
    # OMN-17366 moved the mechanics into reconcile_privilege_lib.sh precisely so
    # there would be one of them. A script that defines its own `as_owner` again
    # has re-created the second implementation, and the copy that drifts is the
    # one nobody is watching.
    for script in guarded:
        rel = script.relative_to(repo_root)
        text = script.read_text(encoding="utf-8")
        if OWNER_HELPER not in text:
            continue
        defines = bool(
            re.search(rf"^\s*{OWNER_HELPER}\s*\(\s*\)\s*\{{", text, re.MULTILINE)
        )
        if script.name == PRIVILEGE_LIB:
            if not defines:
                failures.append(
                    f"{rel}: is the privilege library but does not define "
                    f"{OWNER_HELPER}."
                )
            continue
        if defines:
            failures.append(
                f"{rel}: defines its own {OWNER_HELPER} instead of sourcing "
                f"{PRIVILEGE_LIB}. Two copies of a privilege rule drift, and the "
                "half that drifts is the half nobody is looking at (OMN-17366)."
            )
        elif PRIVILEGE_LIB not in text:
            # Used, not defined, and not sourced: bash would only discover that
            # on the host, inside cron, at 03:00.
            failures.append(
                f"{rel}: uses {OWNER_HELPER} without sourcing {PRIVILEGE_LIB}."
            )

    # -- Part 6: library functions are called by their exported names -------- #
    #
    # OMN-17439. Moving the mechanics into the library renamed them with an
    # `rp_` prefix, and one call site kept the old spelling:
    #
    #     hook_owner="$(surface_owner "$project/.venv" || true)"
    #
    # Bash resolves function names at CALL time, so this parsed cleanly, passed
    # shellcheck, and only appeared as `command not found` on stderr of a live
    # `.201` root tick -- inside a run that exited 0 and reported IN_SYNC.
    #
    # It was not a cosmetic error. The call is wrapped in `|| true`, so the
    # failure was swallowed, `hook_owner` became empty, and the ownership guard
    # below it skipped itself on its own `-n` test. The guard stopped guarding
    # and said nothing, which is the defect class this whole family exists to
    # end -- reproduced inside its own fix.
    #
    # Stated as the general rule rather than as a check for that one name, so
    # the next rename cannot repeat it.
    lib_path = repo_root / "scripts" / PRIVILEGE_LIB
    if lib_path.is_file():
        exported = set(
            re.findall(
                r"^\s*(rp_[a-z_]+)\s*\(\s*\)\s*\{",
                lib_path.read_text(encoding="utf-8"),
                re.MULTILINE,
            )
        )
        for script in guarded:
            if script.name == PRIVILEGE_LIB:
                continue
            rel = script.relative_to(repo_root)
            text = script.read_text(encoding="utf-8")
            if PRIVILEGE_LIB not in text:
                continue
            for exported_name in sorted(exported):
                bare = exported_name[len("rp_") :]
                # A name the script defines itself is its own, not a leftover:
                # reconcile-workspace-venvs.sh legitimately wraps
                # rp_plan_privileges in a local plan_privileges policy function.
                if re.search(rf"^\s*{bare}\s*\(\s*\)\s*\{{", text, re.MULTILINE):
                    continue
                for number, line in logical_lines(text):
                    # `_code`, NOT `_unquoted(_code(...))`. The offending call is
                    # a command substitution INSIDE a quoted assignment:
                    #
                    #     hook_owner="$(surface_owner "$p/.venv" || true)"
                    #
                    # so the quote-stripper pairs `"$(surface_owner "` and
                    # deletes the very name being looked for. Written that way
                    # first, this part reported OK on the exact line it exists to
                    # catch -- the same trap `_GIT_WRITE` fell into in OMN-17366.
                    # Message lines are already excluded by `_code`.
                    executable = _code(line)
                    if not re.search(rf"(?<![\w.-]){bare}\b", executable):
                        continue
                    failures.append(
                        f"{rel}:{number}: calls `{bare}`, but the library "
                        f"exports it as `{exported_name}` and this script does "
                        "not define its own. bash resolves function names at "
                        "call time, so this is a `command not found` on a live "
                        "host inside a run that still exits 0 -- and where the "
                        "call is wrapped in `|| true`, the guard that reads its "
                        "result silently stops guarding (OMN-17439)."
                    )

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="repository root to scan (default: this script's repo)",
    )
    # pre-commit passes staged filenames; the check is whole-repo by design.
    parser.add_argument("filenames", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    failures = check(Path(args.repo_root))
    if not failures:
        print("reconciler privilege/tool-resolution gate: OK")
        return 0
    print("reconciler privilege/tool-resolution gate: FAILED", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
