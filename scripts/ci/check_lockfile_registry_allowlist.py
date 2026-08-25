#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lockfile registry-host allowlist gate: reject a committed ``uv.lock`` that
resolves any package from a non-public registry, git, or artifact host
(OMN-16516).

Born from the 2026-08-23 mirror-leak incident (OMN-16162, root-caused in
``docs/plans/2026-08-23-cloud-ci-offload-plan.md`` §1.1): a single commit
(``d0ada7dc7``, the sibling-lock-refresh dependency-cascade bot) baked 783
``source = { registry = ... }`` lines into ``onex_change_control/uv.lock``,
100% of them pointing at the Tailscale-only devpi mirror
``http://omninode-pc.tail75df5e.ts.net:3141`` and 0 at ``pypi.org``. Any
runner without tailnet access hard-fails ``uv sync --locked`` at resolution
time -- regardless of runner label, cache state, or workflow config. Four
downstream CI jobs failed the same day (OMN-16413/16427/16428/16431) and
three were "fixed" by routing work *toward* the LAN-bound fleet rather than
at the root cause.

Structural, not regex (plan-corrected, F5 / R2-round)
------------------------------------------------------
Every committed ``uv.lock`` is parsed with :mod:`tomllib` (the stdlib TOML
parser -- no third-party dependency, no line-oriented pattern matching that
a reformatted lockfile could dodge). Every host-bearing field this project's
own incident actually populated is checked:

- ``[[package]].source.registry`` -- the field 100% of the incident's 783
  poisoned lines lived in.
- ``[[package]].source.git`` -- a legitimate cross-repo dependency channel
  (``[tool.uv.sources]`` git pins resolve here); the *host*, not the
  mechanism, is what must be public.
- ``[[package]].sdist.url`` and ``[[package]].wheels[].url`` -- the actual
  artifact-fetch URLs uv resolves against once the index has been consulted.

``source.editable`` / ``source.directory`` / ``source.path`` / ``source.virtual``
are LOCAL sources with no network host to validate and are silently skipped
-- they are not a leak vector by construction.

This is deliberately the FAIL-CLOSED BACKSTOP, not the only fix. It catches
a re-leaked lock from ANY channel (a re-baked ambient ``UV_INDEX``, a
misconfigured ``uv.toml`` on a runner, a future generator regression) --
independent of, and in addition to, sanitizing the known generator
workflows (OMN-16517) and pinning a repo-level public index (OMN-16518). See
the plan's own layered-defense statement (§3 Stage 1): "(2) closes the
generator channel; (1) is the fail-closed backstop... Keep both. Neither
substitutes for the other."

Anti-vacuity floor (``--min-packages``)
-----------------------------------------
Mirrors ``check_pin_reachability.py``'s ``--min-pins`` precedent directly: a
gate that silently checks zero packages (a glob typo, a lockfile-schema
drift, ``tomllib`` returning an unexpectedly-shaped tree) reports green
forever and is *more dangerous* than no gate at all, because it looks like
coverage. The CI wiring passes a floor sized to the repo's own known package
count.

Exit codes: ``0`` no blocking findings (including: no lockfile present --
nothing to check is not a failure) | ``1`` at least one blocking finding, or
the anti-vacuity floor was not met | ``2`` misuse (malformed TOML).
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

# The public hosts this project's dependency graph is allowed to resolve
# from. Anything else -- a Tailscale mirror, a devpi instance, a corporate
# proxy -- fails the gate. Extend at the CLI with repeatable --allow-host,
# never by editing this set to permit a one-off private mirror.
_PUBLIC_HOST_ALLOWLIST: frozenset[str] = frozenset(
    {
        "pypi.org",
        "files.pythonhosted.org",
        "github.com",
        "raw.githubusercontent.com",
        "codeload.github.com",
        "objects.githubusercontent.com",
    }
)

# Source kinds that name a LOCAL path, not a network host. Nothing to check.
_LOCAL_SOURCE_KINDS: frozenset[str] = frozenset(
    {"editable", "directory", "path", "virtual"}
)

# uv.lock `[[package]].source` keys that carry a URL to validate.
_SOURCE_HOST_FIELDS: tuple[str, ...] = ("registry", "git")


@dataclass(frozen=True)
class ModelLockfileHostFinding:
    """One (package, field, url) tuple whose host is not on the allowlist."""

    package_name: str
    field_path: str
    url: str
    host: str


def _extract_host(url: str) -> str | None:
    """Return the lowercased hostname of ``url``, or ``None`` if it has none.

    A uv ``source.git`` value carries a query string and fragment
    (``https://host/repo.git?rev=<sha>#<sha>``) -- ``urlsplit`` strips both
    correctly and leaves ``netloc``, from which ``hostname`` drops any
    userinfo/port, exactly what a host-allowlist check needs.
    """
    parsed = urlsplit(url)
    return parsed.hostname.lower() if parsed.hostname else None


def _check_host_field(
    *,
    package_name: str,
    field_path: str,
    url: object,
    allowlist: frozenset[str],
    findings: list[ModelLockfileHostFinding],
) -> None:
    if not isinstance(url, str) or not url:
        return
    host = _extract_host(url)
    if host is None:
        # No parseable host (e.g. a bare local-looking string in a field
        # that is normally a URL) -- nothing to validate against a host
        # allowlist; not this gate's concern.
        return
    if host not in allowlist:
        findings.append(
            ModelLockfileHostFinding(
                package_name=package_name,
                field_path=field_path,
                url=url,
                host=host,
            )
        )


def check_lockfile(
    lockfile: Path, allowlist: frozenset[str] = _PUBLIC_HOST_ALLOWLIST
) -> tuple[ModelLockfileHostFinding, ...]:
    """Parse ``lockfile`` as TOML and validate every host-bearing field.

    Raises ``tomllib.TOMLDecodeError`` on malformed TOML -- the caller (the
    CLI) is responsible for turning that into the misuse exit code (2), a
    real parse failure being categorically different from "found a bad
    host" (1).
    """
    raw = lockfile.read_bytes()
    data = tomllib.loads(raw.decode("utf-8"))

    findings: list[ModelLockfileHostFinding] = []
    for package in data.get("package", []):
        if not isinstance(package, dict):
            continue
        name = str(package.get("name", "<unknown>"))

        source = package.get("source")
        if isinstance(source, dict):
            # A local source kind carries no network host; skip entirely
            # rather than mis-flagging a path string as a bad host.
            if not (_LOCAL_SOURCE_KINDS & source.keys()):
                for field in _SOURCE_HOST_FIELDS:
                    if field in source:
                        _check_host_field(
                            package_name=name,
                            field_path=f"source.{field}",
                            url=source[field],
                            allowlist=allowlist,
                            findings=findings,
                        )

        sdist = package.get("sdist")
        if isinstance(sdist, dict):
            _check_host_field(
                package_name=name,
                field_path="sdist.url",
                url=sdist.get("url"),
                allowlist=allowlist,
                findings=findings,
            )

        for i, wheel in enumerate(package.get("wheels", []) or []):
            if isinstance(wheel, dict):
                _check_host_field(
                    package_name=name,
                    field_path=f"wheels[{i}].url",
                    url=wheel.get("url"),
                    allowlist=allowlist,
                    findings=findings,
                )

    return tuple(findings)


def count_packages(lockfile: Path) -> int:
    raw = lockfile.read_bytes()
    data = tomllib.loads(raw.decode("utf-8"))
    packages = data.get("package", [])
    return len(packages) if isinstance(packages, list) else 0


_FIX_GUIDANCE = (
    "\nFix: the committed uv.lock must resolve every package from a public "
    "registry (pypi.org), a public artifact host (files.pythonhosted.org), "
    "or a public git host (github.com and its content mirrors). If this "
    "was generated with a private/tailnet index reachable, re-generate it "
    "in a sanitized environment: unset UV_INDEX/UV_DEFAULT_INDEX/"
    "UV_INDEX_URL/UV_EXTRA_INDEX_URL, pass --no-config, and pass "
    "--default-index for the public index (see OMN-16517/OMN-16518). This "
    "gate is the fail-closed backstop regardless of which channel produced "
    "the leak."
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reject a committed uv.lock that resolves any package from a "
            "non-public registry, git, or artifact host (OMN-16516)."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="uv.lock file(s) to check. Defaults to ./uv.lock.",
    )
    parser.add_argument(
        "--allow-host",
        action="append",
        default=None,
        metavar="HOST",
        help="Additional allowed hostname (repeatable). Extends, never replaces, the built-in public allowlist.",
    )
    parser.add_argument(
        "--min-packages",
        type=int,
        default=0,
        help=(
            "Fail if a checked lockfile has fewer than N [[package]] "
            "entries. Anti-vacuity floor -- guards the case where a schema "
            "drift or a wrong path makes the gate check (near-)nothing and "
            "report green forever."
        ),
    )
    args = parser.parse_args(argv)

    targets = [Path(p) for p in args.paths] if args.paths else [Path("uv.lock")]
    allowlist = _PUBLIC_HOST_ALLOWLIST | frozenset(args.allow_host or ())

    all_findings: list[tuple[Path, ModelLockfileHostFinding]] = []
    checked_any = False

    for lockfile in targets:
        if not lockfile.exists():
            # A repo with no committed uv.lock has nothing for this gate to
            # check. Missing is not malformed and is not a leak -- only a
            # PRESENT lockfile with a bad host, or (via --min-packages) a
            # present lockfile with implausibly few entries, fails.
            continue
        checked_any = True
        try:
            findings = check_lockfile(lockfile, allowlist=allowlist)
        except tomllib.TOMLDecodeError as exc:
            print(f"error: could not parse {lockfile} as TOML: {exc}", file=sys.stderr)
            return 2

        all_findings.extend((lockfile, f) for f in findings)

        if args.min_packages > 0:
            n = count_packages(lockfile)
            if n < args.min_packages:
                print(
                    f"FAIL: {lockfile} has {n} [[package]] entries, expected "
                    f"at least {args.min_packages}. The parser is more "
                    "likely broken (wrong path, schema drift) than the tree "
                    "is genuinely near-empty -- a gate that checks nothing "
                    "reports green forever.",
                    file=sys.stderr,
                )
                return 1

    if not checked_any:
        print("no uv.lock found in the given target(s) -- nothing to check")
        return 0

    if not all_findings:
        print(
            f"OK: {len(targets)} lockfile target(s) checked, all package "
            "sources resolve to allowlisted public hosts."
        )
        return 0

    print(
        f"FAIL: {len(all_findings)} lockfile entr{'y' if len(all_findings) == 1 else 'ies'} "
        "reference a non-allowlisted host:",
        file=sys.stderr,
    )
    for lockfile, f in all_findings:
        print(
            f"  {lockfile}:: package={f.package_name} field={f.field_path} "
            f"host={f.host!r} url={f.url}",
            file=sys.stderr,
        )
    print(_FIX_GUIDANCE, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
