# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Bind, verify, and emit the versioned omnibase_infra runner image identity.

OMN-12567. The runner image "version" is a *binding*, not a human label. A
bound identity folds together, into one reproducible digest:

* the pinned base image digest,
* the dependency manifest (``pyproject.toml`` + ``uv.lock``),
* the Python version,
* the uv version,
* the shared CI env (canary, OMN-12564) digest, and
* the integer image version.

``ci_env_digest.compute_digest`` (the OMN-12564 canary) supplies the shared-env
component, so an env rebuild forces a new runner identity. The lock file
``docker/runners/runner-image.lock.json`` records the bound identity; CI verifies
the recorded digest against the recomputed one, and every CI job emits the
identity as startup evidence so image-drift debugging is not guesswork.

Modes:
* ``generate`` — recompute the binding and rewrite the lock file in place.
* ``verify``   — fail (non-zero) if the recorded identity is stale or drifted.
* ``emit``     — print the machine-readable startup-evidence line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import ci_env_digest

DEFAULT_LOCK_FILE = Path("docker/runners/runner-image.lock.json")

# Manifest inputs whose bytes participate in the dependency-manifest portion of
# the binding. Changing any of these is a new runner image identity.
MANIFEST_INPUTS = (
    "pyproject.toml",
    "uv.lock",
)


def _read_manifest_digest(repo_root: Path) -> str:
    """Return a deterministic digest of the dependency manifest files."""
    digest = hashlib.sha256()
    for relative in MANIFEST_INPUTS:
        path = repo_root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes() if path.exists() else b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def compute_shared_env_digest(repo_root: Path, lock: dict[str, object]) -> str:
    """Return the OMN-12564 shared CI env digest for the lock's pinned inputs.

    This is the canary env "version": flipping the lockfile, Python, uv, or
    install args produces a new shared-env digest, which in turn re-binds the
    runner image identity.
    """
    digest: str = ci_env_digest.compute_digest(
        repo_root,
        python_version=str(lock["python_version"]),
        uv_version=str(lock["uv_version"]),
        install_args=str(lock["shared_env_install_args"]),
        platform_id="Linux-x86_64",
    )
    return str(digest)


def compute_identity(repo_root: Path, lock: dict[str, object]) -> str:
    """Return the bound runner image identity digest for ``lock``.

    The digest binds every version component. Any single change — base image
    digest, Python version, uv version, runner version, install args, manifest
    bytes, shared-env digest, or the integer image version — produces a new
    identity. That is the binding-not-label property.
    """
    payload = {
        "schema": 1,
        "repo": "omnibase_infra",
        "image_version": lock["image_version"],
        "base_image_digest": lock["base_image_digest"],
        "python_version": lock["python_version"],
        "uv_version": lock["uv_version"],
        "runner_version": lock["runner_version"],
        "gh_version": lock.get("gh_version", ""),
        "kubectl_version": lock.get("kubectl_version", ""),
        "shared_env_install_args": lock["shared_env_install_args"],
        "shared_env_digest": compute_shared_env_digest(repo_root, lock),
        "manifest_digest": _read_manifest_digest(repo_root),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:32]


def _load_lock(lock_path: Path) -> dict[str, object]:
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"lock file must be a JSON object: {lock_path}")
    return data


def generate_lock(repo_root: Path, lock_path: Path) -> int:
    """Recompute the bound identity and rewrite the lock file in place."""
    lock = _load_lock(lock_path)
    lock["shared_env_digest"] = compute_shared_env_digest(repo_root, lock)
    lock["identity_digest"] = compute_identity(repo_root, lock)
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"runner image identity: v{lock['image_version']} {lock['identity_digest']}")
    return 0


def verify_lock(repo_root: Path, lock_path: Path) -> int:
    """Return 0 when the recorded identity matches the recomputed binding."""
    lock = _load_lock(lock_path)
    recomputed_env = compute_shared_env_digest(repo_root, lock)
    if lock.get("shared_env_digest") != recomputed_env:
        print(
            "::error::runner image shared_env_digest is stale "
            f"(recorded={lock.get('shared_env_digest')!r}, "
            f"recomputed={recomputed_env!r}); run "
            "scripts/ci/runner_image_identity.py --mode generate"
        )
        return 1
    recomputed = compute_identity(repo_root, lock)
    if lock.get("identity_digest") != recomputed:
        print(
            "::error::runner image identity_digest is stale "
            f"(recorded={lock.get('identity_digest')!r}, "
            f"recomputed={recomputed!r}); run "
            "scripts/ci/runner_image_identity.py --mode generate"
        )
        return 1
    print(
        f"runner image identity verified: v{lock['image_version']} "
        f"{lock['identity_digest']}"
    )
    return 0


def format_startup_evidence(lock: dict[str, object]) -> str:
    """Return the machine-readable startup-evidence JSON line."""
    return json.dumps(
        {
            "runner_image_version": lock["image_version"],
            "runner_image_identity": lock["identity_digest"],
            "python_version": lock["python_version"],
            "uv_version": lock["uv_version"],
            "shared_env_digest": lock.get("shared_env_digest", ""),
        },
        sort_keys=True,
    )


def emit_startup_evidence(lock_path: Path) -> int:
    """Print the startup-evidence line for the recorded identity."""
    lock = _load_lock(lock_path)
    print(format_startup_evidence(lock))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bind, verify, or emit the omnibase_infra runner image identity."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK_FILE)
    parser.add_argument(
        "--mode",
        choices=("generate", "verify", "emit"),
        default="verify",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    lock_path = (
        args.lock_file if args.lock_file.is_absolute() else repo_root / args.lock_file
    )

    if args.mode == "generate":
        return generate_lock(repo_root, lock_path)
    if args.mode == "emit":
        return emit_startup_evidence(lock_path)
    return verify_lock(repo_root, lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
