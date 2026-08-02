#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Assert every required config path exists INSIDE a built runtime image (OMN-15676).

This is the anti-recurrence mechanism for the "tracked in repo, absent from the
image" class. It reads the single typed source
(``omnibase_infra.runtime.required_image_config_paths``) and runs one ``test -f``
per entry *inside the image under test*, so a missing Dockerfile ``COPY`` fails
the build instead of reaching a registry and boot-crashing a deployed pod.

A repo-working-tree check cannot substitute for this: all three prior incidents
(grants fixture, routing_tiers.yaml, runner_fleet.yaml) had a perfectly correct
working tree.

Fail-closed in every direction:

* any declared path missing -> exit 1,
* the probe container failing to run at all -> exit 1,
* a path the probe returned no verdict for -> exit 1 (never treated as present),
* an empty registry -> exit 1 (a silently-emptied registry must not look green).

Usage::

    python scripts/ci/assert_image_config_paths.py --image <image-ref>
    python scripts/ci/assert_image_config_paths.py --image <ref> --docker-bin podman
"""

from __future__ import annotations

import argparse
import importlib.util
import shlex
import subprocess  # nosec B404 - deliberate: probing a container image requires the docker CLI
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Import-for-types only. At runtime the registry is loaded by file path
    # (see _load_registry_module) so an ambient/installed copy can never be
    # substituted for the one in this checkout.
    from omnibase_infra.runtime.required_image_config_paths import (
        ModelRequiredImageConfigPath,
    )

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY_PATH = (
    _REPO_ROOT / "src" / "omnibase_infra" / "runtime" / "required_image_config_paths.py"
)


def _load_registry_module() -> ModuleType:
    """Load the registry from THIS checkout by file path, never by package name.

    A plain ``from omnibase_infra.runtime... import`` resolves through whatever
    ``omnibase_infra`` the ambient environment already binds -- an installed
    wheel, or a canonical clone on ``PYTHONPATH``. Observed while building this
    guard: with ``PYTHONPATH`` pointing at the canonical clone, the import
    raised ModuleNotFoundError for a module that exists right here in the
    checkout. The failure mode that does NOT announce itself is the other one:
    an ambient copy resolving to an OLDER registry, so the guard silently
    asserts fewer paths than the branch declares and reports green.
    Path-loading pins the guard to the tree it ships in.
    """
    spec = importlib.util.spec_from_file_location(
        "_omn15676_required_image_config_paths", _REGISTRY_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load required-config registry: {_REGISTRY_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_registry = _load_registry_module()
REQUIRED_IMAGE_CONFIG_PATHS: tuple[ModelRequiredImageConfigPath, ...] = (
    _registry.REQUIRED_IMAGE_CONFIG_PATHS
)

_PRESENT = "PRESENT"
_MISSING = "MISSING"
_PROBE_TIMEOUT_SECONDS = 300


def _validate_paths(entries: Sequence[ModelRequiredImageConfigPath]) -> list[str]:
    """Reject entries that are not safe, absolute, unambiguous in-image paths."""
    problems: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        path = entry.image_path
        if not path.startswith("/"):
            problems.append(f"{path}: image_path must be absolute")
        if path != shlex.quote(path):
            problems.append(
                f"{path}: image_path contains characters requiring shell quoting"
            )
        if path in seen:
            problems.append(f"{path}: duplicate registry entry")
        seen.add(path)
    return problems


def _build_probe_script(paths: Sequence[str]) -> str:
    """Emit a POSIX-sh probe printing one `<VERDICT> <path>` line per path.

    The probe never exits non-zero on a missing file -- the verdict lines are the
    channel, so a non-zero container exit unambiguously means the probe itself
    could not run.
    """
    lines = ["set -u"]
    for path in paths:
        quoted = shlex.quote(path)
        lines.append(
            f"if [ -f {quoted} ]; then echo '{_PRESENT} {path}'; "
            f"else echo '{_MISSING} {path}'; fi"
        )
    return "\n".join(lines)


def _run_probe(*, docker_bin: str, image: str, script: str) -> tuple[int, str, str]:
    command = [
        docker_bin,
        "run",
        "--rm",
        "--entrypoint",
        "sh",
        image,
        "-c",
        script,
    ]
    try:
        completed = subprocess.run(  # nosec B603 - fixed argv, no shell, paths validated above
            command,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        return 127, "", f"{docker_bin}: not found on PATH"
    except subprocess.TimeoutExpired:
        return 124, "", f"probe timed out after {_PROBE_TIMEOUT_SECONDS}s"
    return completed.returncode, completed.stdout, completed.stderr


def _parse_verdicts(stdout: str) -> dict[str, str]:
    verdicts: dict[str, str] = {}
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        verdict, _, path = line.partition(" ")
        if verdict in (_PRESENT, _MISSING) and path:
            verdicts[path] = verdict
    return verdicts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Assert every path in the required-image-config registry exists "
            "inside the given built image (OMN-15676)."
        )
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Image ref to probe. Must already be present locally (docker build --load).",
    )
    parser.add_argument(
        "--docker-bin",
        default="docker",
        help="Container CLI to use (default: docker).",
    )
    args = parser.parse_args(argv)

    entries = REQUIRED_IMAGE_CONFIG_PATHS
    if not entries:
        print(
            "FAIL: required-image-config registry is EMPTY -- refusing to report "
            "green. An empty registry asserts nothing (OMN-15676).",
            file=sys.stderr,
        )
        return 1

    problems = _validate_paths(entries)
    if problems:
        for problem in problems:
            print(f"FAIL: invalid registry entry -- {problem}", file=sys.stderr)
        return 1

    paths = [entry.image_path for entry in entries]
    print(f"Asserting {len(paths)} required config path(s) inside image: {args.image}")

    returncode, stdout, stderr = _run_probe(
        docker_bin=args.docker_bin,
        image=args.image,
        script=_build_probe_script(paths),
    )
    if returncode != 0:
        print(
            f"FAIL: probe container exited {returncode} -- cannot prove any path "
            "is present, failing closed.",
            file=sys.stderr,
        )
        if stdout.strip():
            print(f"stdout: {stdout.strip()}", file=sys.stderr)
        if stderr.strip():
            print(f"stderr: {stderr.strip()}", file=sys.stderr)
        return 1

    verdicts = _parse_verdicts(stdout)
    missing: list[ModelRequiredImageConfigPath] = []
    unobserved: list[ModelRequiredImageConfigPath] = []
    for entry in entries:
        verdict = verdicts.get(entry.image_path)
        if verdict == _PRESENT:
            print(f"  OK       {entry.image_path}")
        elif verdict == _MISSING:
            print(f"  MISSING  {entry.image_path}")
            missing.append(entry)
        else:
            print(f"  NO-VERDICT {entry.image_path}")
            unobserved.append(entry)

    if not missing and not unobserved:
        print(f"PASS: all {len(paths)} required config path(s) present in the image.")
        return 0

    print("", file=sys.stderr)
    for entry in missing:
        print(
            f"FAIL: {entry.image_path} is NOT in the image.\n"
            f"      resolved by: {entry.resolved_by}\n"
            f"      impact:      {entry.why_required}\n"
            f"      ticket:      {entry.ticket}\n"
            f"      fix:         add a COPY for this path to docker/Dockerfile.runtime "
            f"(a bind-mount does not satisfy this -- the deployed pod has no such mount).",
            file=sys.stderr,
        )
    for entry in unobserved:
        print(
            f"FAIL: {entry.image_path} returned no verdict from the probe -- "
            "treated as absent (fail-closed).",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
