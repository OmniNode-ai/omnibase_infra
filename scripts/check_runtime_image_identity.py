#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Enforce runtime image identity stamping (OMN-12965).

The deep dive found the main runtime image stamped
``org.opencontainers.image.version=0.1.0`` with a blank
``org.opencontainers.image.revision`` after workspace rebuilds. A blank identity
degrades every proof packet — the runtime SHA + image digest are required
citations in accepted evidence.

This is the enforcement ratchet (pre-commit + CI) that prevents the build paths
from regressing to a blank/placeholder identity. It is a static check: no docker
build, no network. It verifies:

1. ``docker/Dockerfile.runtime`` labels ``org.opencontainers.image.revision``
   from ``VCS_REF`` and ``org.opencontainers.image.version`` from
   ``RUNTIME_VERSION``, and carries a workspace-mode guard that fails the build
   when either would be blank/``unknown``/placeholder ``0.1.0``.
2. The operator rebuild path ``cmd_up`` in
   ``src/omnibase_infra/docker/catalog/cli.py`` stamps the identity quad via
   ``_image_identity_build_args()`` rather than a bare ``GIT_SHA`` build-arg.

Usage::

    # From the repo root
    uv run python scripts/check_runtime_image_identity.py

Exit codes:
    0 — all checks passed
    1 — one or more checks failed (stderr has details)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DOCKERFILE = _REPO_ROOT / "docker" / "Dockerfile.runtime"
_CLI = _REPO_ROOT / "src" / "omnibase_infra" / "docker" / "catalog" / "cli.py"


def _collapse_continuations(text: str) -> str:
    """Collapse shell ``\\``-continued physical lines into single logical lines."""
    return re.sub(r"\\\n\s*", " ", text)


def _check_dockerfile() -> list[str]:
    errors: list[str] = []
    if not _DOCKERFILE.is_file():
        return [f"missing {_DOCKERFILE}"]
    collapsed = _collapse_continuations(_DOCKERFILE.read_text(encoding="utf-8"))

    if 'org.opencontainers.image.revision="${VCS_REF}"' not in collapsed:
        errors.append(
            "Dockerfile.runtime must label org.opencontainers.image.revision "
            'from "${VCS_REF}" (OMN-12965).'
        )
    if 'org.opencontainers.image.version="${RUNTIME_VERSION}"' not in collapsed:
        errors.append(
            "Dockerfile.runtime must label org.opencontainers.image.version "
            'from "${RUNTIME_VERSION}" (OMN-12965).'
        )

    guard = re.search(
        r'if \[ "\$\{BUILD_SOURCE\}" = "workspace" \];.*?VCS_REF.*?'
        r"exit 64.*?RUNTIME_VERSION.*?0\.1\.0.*?exit 64",
        collapsed,
        re.DOTALL,
    )
    if not guard:
        errors.append(
            "Dockerfile.runtime must guard workspace builds against blank VCS_REF "
            "and placeholder RUNTIME_VERSION=0.1.0 (OMN-12965)."
        )
    return errors


def _function_body(source: str, name: str) -> str | None:
    lines = source.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.startswith(f"def {name}(")),
        None,
    )
    if start is None:
        return None
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if lines[i] and not lines[i][0].isspace()
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def _check_cli() -> list[str]:
    errors: list[str] = []
    if not _CLI.is_file():
        return [f"missing {_CLI}"]
    source = _CLI.read_text(encoding="utf-8")

    body = _function_body(source, "cmd_up")
    if body is None:
        return ["cmd_up not found in cli.py"]

    if "_image_identity_build_args()" not in body:
        errors.append(
            "cmd_up --build must stamp the identity quad via "
            "_image_identity_build_args() (OMN-12965)."
        )
    if "GIT_SHA=" in body:
        errors.append(
            "cmd_up must not hand-roll a GIT_SHA build-arg; delegate to "
            "_image_identity_build_args() so VCS_REF + RUNTIME_VERSION are "
            "stamped (OMN-12965)."
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    errors = _check_dockerfile() + _check_cli()
    if errors:
        print("Runtime image identity check FAILED (OMN-12965):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print("Runtime image identity check passed (OMN-12965).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
