# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Compute the immutable key for the shared omnibase_infra CI environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

DEFAULT_ENV_INPUTS = (
    "pyproject.toml",
    "uv.lock",
    ".github/actions/setup-python-uv/action.yml",
    "scripts/ci/ci_env_digest.py",
    "scripts/ci/ensure_ci_env.sh",
)


def compute_digest(
    repo_root: Path,
    *,
    python_version: str,
    uv_version: str,
    install_args: str,
    extra: str = "",
    platform_id: str | None = None,
) -> str:
    """Return a short deterministic digest for the CI dependency environment."""
    root = repo_root.resolve()
    resolved_platform = platform_id or f"{platform.system()}-{platform.machine()}"
    payload = {
        "schema": 1,
        "repo": "omnibase_infra",
        "python_version": python_version,
        "uv_version": uv_version,
        "install_args": install_args,
        "platform": resolved_platform,
        "extra": extra,
    }

    digest = hashlib.sha256()
    digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    digest.update(b"\0")

    for relative in DEFAULT_ENV_INPUTS:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if path.exists():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
        digest.update(b"\0")

    return digest.hexdigest()[:24]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute the shared omnibase_infra CI environment digest."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--uv-version", required=True)
    parser.add_argument("--install-args", required=True)
    parser.add_argument("--extra", default="")
    parser.add_argument("--platform-id")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    digest = compute_digest(
        args.repo_root,
        python_version=args.python_version,
        uv_version=args.uv_version,
        install_args=args.install_args,
        extra=args.extra,
        platform_id=args.platform_id,
    )
    if args.json:
        print(json.dumps({"digest": digest}, sort_keys=True))
    else:
        print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
