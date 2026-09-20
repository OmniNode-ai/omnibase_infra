#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pre-merge twin of the OMN-14631 workspace content-parity gate.

WHY THIS EXISTS
---------------
`scripts/runtime_build/compute_workspace_provenance.py` proves, during the
`BUILD_SOURCE=workspace` runtime image build, that the INSTALLED package tree
under site-packages is byte-for-byte the staged source tree plus whatever the
repo declares force-included into that package. It is the right proof and it
is in the wrong place in time: it runs on the lab, after the change has
already merged in the package's own repository.

Twice on 2026-09-19 a change that was green in its own repository broke that
gate and was found only post-merge, each time costing hours of dev-lane
downtime:

  - omnibase_core#1710 added a hatch `force-include` the gate could not
    resolve, so the mapped file read as an extra installed file. Fixed in the
    GATE by omnibase_infra#3846 (OMN-18847).
  - omnimarket#2670 adopted a propagated `.gitignore` block declaring a BARE
    `merge-sweep/`. A bare directory pattern matches at any depth, so it also
    matched the real, git-tracked package directory
    `src/omnimarket/adapters/codex/skills/merge-sweep/`; hatchling applies the
    VCS ignore file as a build-time exclude, so the wheel silently lost a
    tracked source file. Fixed in the PACKAGES by omnibase_core#1718 and
    omnimarket#2694 (OMN-18859).

Both are decidable from the package repository alone, at pull-request time,
against the pull request's own head. That is what this script does.

WHEEL, NOT VENV -- AND WHY THAT IS THE SAME PROOF
-------------------------------------------------
The image gate compares the staged tree against the INSTALLED site-packages
tree. A non-editable install of a local path IS an unpack of the wheel that
path builds, so comparing the staged tree against the WHEEL's own
`<import_name>/` entries tests the identical property, at the cost of one
`uv build --wheel` (measured in single-digit seconds) instead of a venv, a
resolve and an install. `--wheel` accepts a prebuilt wheel for a caller that
already has one.

SINGLE SOURCE OF TRUTH
----------------------
Every comparison primitive is IMPORTED from the image gate's own module --
the exclusion rule, the force-include resolution, the digest and the diff.
Nothing is reimplemented here and nothing is copied into the package
repositories, because a second copy of this comparison is precisely the
failure this check exists to prevent: the 2026-09-19 window found three
copies of one wrong assumption about the staged tree.
`tests/scripts/test_wheel_content_parity_single_source.py` asserts the
imported objects ARE the gate's objects, so a copy is a red test.

Exit codes:
  0  the wheel's package tree matches the source tree
  1  content drift (a file missing from, extra in, or differing in the wheel)
  2  the comparison could not be resolved (fails closed, never a silent pass)
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# The comparison primitives live in the image gate and are imported, never
# reimplemented. See "SINGLE SOURCE OF TRUTH" above.
_RUNTIME_BUILD = Path(__file__).resolve().parent.parent / "runtime_build"
sys.path.insert(0, str(_RUNTIME_BUILD))
from compute_workspace_provenance import (
    _diff_file_maps,
    _digest_files,
    _force_included_files,
    _is_excluded_part,
    _tracked_files,
)

# Wheel members that are metadata rather than package content.
_WHEEL_METADATA_SUFFIXES = (".dist-info", ".data")


def assert_build_root_is_not_vcs_ignored(repo_root: Path) -> list[str]:
    """Refuse a build whose own absolute location the repo's .gitignore matches.

    MEASURED, and the reason this function exists rather than a comment.
    hatchling skips the VCS ignore file ENTIRELY when the build root itself
    resolves as ignored. Building this exact check under `/private/tmp/...`
    against a repository whose `.gitignore` carries a bare `tmp/` therefore
    produced a GREEN result on the very tree whose wheel the image gate had
    just refused -- a false pass, not a false failure, which is the dangerous
    direction.

    Reproduction, hatchling 1.32.3, one minimal project, one variable:

        .gitignore = ["merge-sweep/"]          built under /private/tmp -> excluded
        .gitignore = ["tmp/", "merge-sweep/"]  built under /private/tmp -> INCLUDED
        .gitignore = ["tmp/", "merge-sweep/"]  built under ~/.cache     -> excluded

    So the guard is not about a pattern being wrong; it is about the BUILD
    LOCATION silently disarming every pattern. It fails closed: an
    unreadable `.gitignore` is an error, never an assumed-clean root.
    """
    errors: list[str] = []
    gitignore = repo_root / ".gitignore"
    if not gitignore.exists():
        return errors
    try:
        lines = gitignore.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(
            f"Cannot read {gitignore} to verify the build root is not itself "
            f"VCS-ignored ({exc}). Failing closed: an unverified build root "
            "can silently disarm every ignore pattern (see this function's "
            "docstring for the measurement)."
        )
        return errors

    # Only an UNANCHORED, single-segment pattern can match an ANCESTOR of the
    # build root, which is the condition that disarms the spec. An anchored
    # pattern ('/x/') is relative to the project root and cannot.
    #
    # The trailing slash is OPTIONAL and leaving it out of this test is a real
    # bug this check had until a positive control caught it: omnibase_compat
    # ignores a bare `.cache` with NO trailing slash, which in git matches a
    # directory just as `.cache/` does. Judging that repository from a path
    # under `~/.cache/` produced a confident GREEN on a tree whose wheel drops
    # a real package directory.
    patterns: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "!", "/")):
            continue
        body = stripped.rstrip("/")
        if not body or "/" in body:
            continue
        patterns.add(body)

    ancestors = [part for part in repo_root.resolve().parts if part != "/"]
    collisions = sorted(
        {
            f"{pattern} (matches path component {part!r})"
            for pattern in patterns
            for part in ancestors
            if fnmatch.fnmatch(part, pattern)
        }
    )
    if collisions:
        errors.append(
            "REFUSING to judge wheel content parity from this location: the "
            f"build root {repo_root.resolve()} has a path component matching "
            f"this repository's own bare ignore pattern(s) {collisions}. "
            "hatchling skips the VCS ignore file entirely when the build root "
            "resolves as ignored, so the wheel would contain files a real "
            "build excludes and this check would pass a tree the runtime "
            "image build refuses. Build from a path with no such component."
        )
    return errors


def build_wheel(repo_root: Path, out_dir: Path) -> tuple[Path | None, list[str]]:
    """Build the project's wheel with uv and return its path."""
    cmd = ["uv", "build", "--wheel", "--out-dir", str(out_dir), str(repo_root)]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=repo_root, check=False
    )
    if proc.returncode != 0:
        return None, [
            f"Wheel build failed ({' '.join(cmd)}) with exit {proc.returncode}.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        ]
    wheels = sorted(out_dir.glob("*.whl"))
    if len(wheels) != 1:
        return None, [
            f"Expected exactly one wheel in {out_dir}, found {len(wheels)}: "
            f"{[w.name for w in wheels]}."
        ]
    return wheels[0], []


def wheel_package_files(
    wheel: Path, import_name: str
) -> tuple[dict[str, bytes], list[str]]:
    """Return {path relative to <import_name>/: content} for the wheel's package tree.

    Applies the image gate's own exclusion rule to the member paths so the two
    sides of the comparison are filtered identically.
    """
    out: dict[str, bytes] = {}
    errors: list[str] = []
    prefix = f"{import_name}/"
    with zipfile.ZipFile(wheel) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            top = name.split("/", 1)[0]
            if top.endswith(_WHEEL_METADATA_SUFFIXES):
                continue
            if not name.startswith(prefix):
                errors.append(
                    f"Wheel {wheel.name} carries a top-level member outside "
                    f"'{import_name}/' and outside wheel metadata: {name!r}. "
                    "The parity comparison only covers this package's own "
                    "directory, so an unexpected top-level member is not "
                    "judged rather than silently ignored."
                )
                continue
            rel = name[len(prefix) :]
            if any(_is_excluded_part(part) for part in Path(rel).parts):
                continue
            out[rel] = zf.read(info)
    return out, errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prove the built wheel's package tree is byte-for-byte the source "
            "tree plus declared force-includes (pre-merge twin of OMN-14631)."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(os.environ.get("GITHUB_WORKSPACE", Path.cwd())),
        help="repository root to judge (default: GITHUB_WORKSPACE, else cwd)",
    )
    parser.add_argument(
        "--package",
        required=True,
        help="the package's IMPORT name, e.g. omnibase_core",
    )
    parser.add_argument(
        "--wheel",
        type=Path,
        default=None,
        help="a prebuilt wheel to judge instead of building one",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="write a machine-readable result document here",
    )
    args = parser.parse_args()

    repo_root: Path = args.repo_root.resolve()
    import_name: str = args.package
    errors: list[str] = []

    src_dir = repo_root / "src" / import_name
    if not src_dir.is_dir():
        print(
            f"::error::Cannot judge content parity for '{import_name}': "
            f"expected source package directory {src_dir} does not exist.",
            file=sys.stderr,
        )
        return 2

    errors.extend(assert_build_root_is_not_vcs_ignored(repo_root))
    if errors:
        for err in errors:
            print(f"::error::{err}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="wheel-parity-") as tmp:
        wheel = args.wheel
        if wheel is None:
            wheel, build_errors = build_wheel(repo_root, Path(tmp))
            if wheel is None:
                for err in build_errors:
                    print(f"::error::{err}", file=sys.stderr)
                return 2

        wheel_files, wheel_errors = wheel_package_files(wheel, import_name)
        errors.extend(wheel_errors)

        # The expected content is the source package tree PLUS whatever the
        # repo declares force-included into it, resolved by the image gate's
        # own resolver so the two agree by construction (OMN-18033).
        expected = _tracked_files(src_dir)
        expected.update(_force_included_files(repo_root, import_name, errors))

        expected_digest = _digest_files(expected)
        wheel_digest = _digest_files(wheel_files)
        diff = (
            []
            if expected_digest == wheel_digest
            else _diff_file_maps(expected, wheel_files)
        )

        result: dict[str, object] = {
            "package": import_name,
            "repo_root": str(repo_root),
            "wheel": wheel.name,
            "source_digest": expected_digest,
            "wheel_digest": wheel_digest,
            "source_file_count": len(expected),
            "wheel_file_count": len(wheel_files),
            "differing_files": diff,
            "errors": errors,
            "status": "verified" if not diff and not errors else "content_mismatch",
        }
        if args.json is not None:
            args.json.write_text(json.dumps(result, indent=2, sort_keys=True))

        for err in errors:
            print(f"::error::{err}", file=sys.stderr)

        if diff:
            missing = sorted(set(expected) - set(wheel_files))
            extra = sorted(set(wheel_files) - set(expected))
            changed = sorted(
                rel
                for rel in set(expected) & set(wheel_files)
                if expected[rel] != wheel_files[rel]
            )
            print(
                f"::error::WHEEL CONTENT DRIFT for '{import_name}': the wheel "
                f"built from this tree does NOT match {src_dir}. "
                f"source_digest={expected_digest[:16]}... "
                f"wheel_digest={wheel_digest[:16]}... "
                f"differing files ({len(diff)} total): {diff[:10]}",
                file=sys.stderr,
            )
            if missing:
                print(
                    f"::error::  {len(missing)} tracked source file(s) the wheel "
                    f"DROPPED (usually a .gitignore pattern matching a real "
                    f"package directory -- check with "
                    f"`git check-ignore --no-index -v <path>`, because a plain "
                    f"check-ignore is silent on a TRACKED file): {missing[:10]}",
                    file=sys.stderr,
                )
            if extra:
                print(
                    f"::error::  {len(extra)} file(s) in the wheel with no source "
                    f"behind them (usually an unresolved force-include): "
                    f"{extra[:10]}",
                    file=sys.stderr,
                )
            if changed:
                print(
                    f"::error::  {len(changed)} file(s) whose content differs: "
                    f"{changed[:10]}",
                    file=sys.stderr,
                )
            return 1

        if errors:
            return 2

        print(
            f"wheel content parity OK: {import_name} "
            f"files={len(expected)} digest={expected_digest[:16]}... (match)"
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
