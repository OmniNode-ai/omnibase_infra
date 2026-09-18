# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Plan-then-apply wrapper around ``uv pip install`` that refuses downgrades.

OMN-18675. ``scripts/install-node-skill-package.sh`` mutates a venv that is
SHARED by every lane on the host (the plugin CLI venv,
``$CLAUDE_PLUGIN_DATA/.venv`` — see omni_home/CLAUDE.md rule 11). On
2026-09-18 it silently downgraded ``omnibase-compat`` 0.5.7 -> 0.5.5 and
``omninode-memory`` 0.18.0 -> 0.15.0 while repairing an unrelated omnimarket
pin, which removed ``omnibase_compat.contracts.pr_occ_stamp`` and broke the
whole ``onex`` CLI for every command and every lane on the machine.

The failure was silent because ``uv pip install --no-deps pkg==OLD`` is a
perfectly successful install: an exact specifier is an instruction, not a
floor, and ``--no-deps`` suppresses the resolution that would have objected.
Nothing in the pipeline compared the requested version against the installed
one.

This module closes that: it runs the install as a ``--dry-run`` FIRST, parses
uv's own change plan, prints a before/after row for every package the plan
would touch, and REFUSES (exit 3, naming the packages) when any package would
move backwards or disappear. Only a plan with no downgrade and no removal is
applied.

There is deliberately no ``--force`` / ``--allow-downgrade`` escape hatch: a
downgrade of a shared venv is the defect this exists to stop, and a flag to
re-enable it would be the first thing a failing lane reached for
(omni_home/CLAUDE.md rule 10). A genuine downgrade is performed by naming the
older version to ``uv pip install`` directly, by hand, by a human who has
decided that is what they want.

Usage::

    python scripts/venv_install_plan.py --python <venv-python> \\
        [--apply] [--no-deps] [--label "step 1"] -- REQUIREMENT [REQUIREMENT ...]

Exit codes:
    0  plan is safe (and applied, when ``--apply`` was passed)
    2  bad invocation
    3  REFUSED — the plan would downgrade or remove an installed package
    4  uv failed, or its output could not be parsed
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_REFUSED = 3
EXIT_UV_ERROR = 4

# uv renders its install plan as one line per changed distribution:
#   " + omnibase-compat==0.5.5"
#   " - omnibase-compat==0.5.7"
# Local-version and direct-URL installs carry a trailing " (from ...)" or a
# "+local" segment, both of which are kept in the version string verbatim so a
# VCS reinstall is never mistaken for a version change.
_CHANGE_LINE = re.compile(
    r"^\s*(?P<sign>[-+])\s+(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>\S+)"
)


class PlanVerdict(str, Enum):
    """What the plan would do to one distribution."""

    INSTALL = "INSTALL"
    UPGRADE = "UPGRADE"
    REINSTALL = "REINSTALL"
    DOWNGRADE = "DOWNGRADE"
    REMOVE = "REMOVE"
    UNKNOWN = "UNKNOWN"


#: Verdicts that make the whole plan unsafe to apply to a shared venv.
REFUSING_VERDICTS = frozenset(
    {PlanVerdict.DOWNGRADE, PlanVerdict.REMOVE, PlanVerdict.UNKNOWN}
)


@dataclass(frozen=True)
class PlanChange:
    """One distribution's before/after state under a proposed install."""

    name: str
    before: str | None
    after: str | None
    verdict: PlanVerdict

    def render(self) -> str:
        before = self.before or "(absent)"
        after = self.after or "(removed)"
        return f"  {self.verdict.value:<9} {self.name}  {before} -> {after}"


def normalize_name(name: str) -> str:
    """Normalize a distribution name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_version(raw: str) -> tuple[object, ...] | None:
    """Parse a version into a comparable key, or None when it is not comparable.

    ``packaging`` is used when importable (it always is inside a built venv);
    an unparseable version deliberately yields ``None`` so the caller can fail
    CLOSED rather than guess an ordering.
    """
    try:
        from packaging.version import InvalidVersion, Version
    except ImportError:  # pragma: no cover - packaging ships with every venv here
        return None
    try:
        return (Version(raw),)
    except InvalidVersion:
        return None


def classify(name: str, before: str | None, after: str | None) -> PlanChange:
    """Classify one before/after pair into a :class:`PlanVerdict`."""
    if before is None and after is None:
        return PlanChange(name, before, after, PlanVerdict.UNKNOWN)
    if before is None:
        return PlanChange(name, before, after, PlanVerdict.INSTALL)
    if after is None:
        return PlanChange(name, before, after, PlanVerdict.REMOVE)
    if before == after:
        return PlanChange(name, before, after, PlanVerdict.REINSTALL)

    before_key = _parse_version(before)
    after_key = _parse_version(after)
    if before_key is None or after_key is None:
        # An unorderable version pair is not proof of safety. Fail closed.
        return PlanChange(name, before, after, PlanVerdict.UNKNOWN)
    if after_key < before_key:
        return PlanChange(name, before, after, PlanVerdict.DOWNGRADE)
    return PlanChange(name, before, after, PlanVerdict.UPGRADE)


def parse_plan(output: str) -> list[PlanChange]:
    """Parse uv's ``+``/``-`` change lines into classified per-package changes."""
    removed: dict[str, str] = {}
    added: dict[str, str] = {}
    order: list[str] = []
    for line in output.splitlines():
        match = _CHANGE_LINE.match(line)
        if match is None:
            continue
        name = normalize_name(match.group("name"))
        version = match.group("version")
        if name not in order:
            order.append(name)
        if match.group("sign") == "-":
            removed[name] = version
        else:
            added[name] = version
    return [classify(name, removed.get(name), added.get(name)) for name in order]


def _run_uv(
    python_bin: str, requirements: list[str], *, no_deps: bool, dry_run: bool
) -> subprocess.CompletedProcess[str]:
    argv = ["uv", "pip", "install", "--python", python_bin]
    if no_deps:
        argv.append("--no-deps")
    if dry_run:
        argv.append("--dry-run")
    argv.extend(requirements)
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def _describe(label: str, requirements: list[str], no_deps: bool) -> None:
    flags = " --no-deps" if no_deps else ""
    print(f"== {label}: plan (uv pip install{flags} --dry-run) ==")
    for requirement in requirements:
        print(f"     {requirement}")


def plan_and_apply(
    python_bin: str,
    requirements: list[str],
    *,
    no_deps: bool,
    apply: bool,
    label: str,
) -> int:
    """Dry-run the install, refuse a regressive plan, then optionally apply it."""
    _describe(label, requirements, no_deps)

    dry = _run_uv(python_bin, requirements, no_deps=no_deps, dry_run=True)
    combined = f"{dry.stdout}\n{dry.stderr}"
    if dry.returncode != 0:
        print(combined.strip(), file=sys.stderr)
        print(
            f"ERROR: uv could not resolve the plan for {label} (exit {dry.returncode}).",
            file=sys.stderr,
        )
        return EXIT_UV_ERROR

    changes = parse_plan(combined)
    if not changes:
        print(f"  no change — every requirement is already satisfied ({label}).")
        return EXIT_OK

    print(f"  {len(changes)} package(s) would change:")
    for change in changes:
        print(change.render())

    refused = [c for c in changes if c.verdict in REFUSING_VERDICTS]
    if refused:
        names = ", ".join(sorted(c.name for c in refused))
        print()
        print(
            f"REFUSED: {label} would move installed package(s) backwards.",
            file=sys.stderr,
        )
        for change in refused:
            print(
                f"  {change.name}: {change.before} -> {change.after} ({change.verdict.value})",
                file=sys.stderr,
            )
        print(file=sys.stderr)
        print(
            "  This venv may be shared by every lane on the host "
            "(omni_home/CLAUDE.md rule 11), so a downgrade here breaks commands\n"
            "  that have nothing to do with this install (OMN-18675). Nothing was\n"
            "  changed. Raise the requested pin so it is >= what is installed, or\n"
            f"  install the older {names} deliberately by hand if that is really\n"
            "  what you want.",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    if not apply:
        print(f"  PLAN ONLY — re-run with --apply to perform {label}.")
        return EXIT_OK

    print(f"== {label}: applying ==")
    real = _run_uv(python_bin, requirements, no_deps=no_deps, dry_run=False)
    real_combined = f"{real.stdout}\n{real.stderr}"
    if real.returncode != 0:
        print(real_combined.strip(), file=sys.stderr)
        print(f"ERROR: {label} failed (exit {real.returncode}).", file=sys.stderr)
        return EXIT_UV_ERROR

    applied = parse_plan(real_combined)
    print(f"  {len(applied)} package(s) changed:")
    for change in applied:
        print(change.render())
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run a uv install, print before/after for every changed package, "
            "and refuse any plan that downgrades or removes one."
        )
    )
    parser.add_argument("--python", required=True, help="target venv python")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the install when the plan is safe (default: plan only)",
    )
    parser.add_argument(
        "--no-deps", action="store_true", help="pass --no-deps through to uv"
    )
    parser.add_argument(
        "--label", default="install", help="human label for this step in the output"
    )
    parser.add_argument("requirements", nargs="*", help="requirements to install")
    args = parser.parse_args(argv)

    if not args.requirements:
        print("ERROR: no requirements given.", file=sys.stderr)
        return EXIT_USAGE

    return plan_and_apply(
        args.python,
        list(args.requirements),
        no_deps=args.no_deps,
        apply=args.apply,
        label=args.label,
    )


if __name__ == "__main__":
    sys.exit(main())
