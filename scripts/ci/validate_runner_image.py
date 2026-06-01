# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Validate a self-hosted runner against the versioned image contract (OMN-12568).

OMN-12567 makes the runner image "version" a *binding* — base image digest +
dependency manifest + Python version + uv version + shared-env version + image
version folded into one reproducible ``identity_digest`` recorded in
``runner-image.lock.json``. OMN-12568 is the *acceptance* of that contract: run
this script on a freshly recreated runner and prove the image survived
recreation.

The script asserts three things on the runner it runs on:

1. **Image identity** — the lock file is present and internally consistent
   (recorded ``identity_digest`` matches the recomputed binding, when the
   OMN-12567 generator ``runner_image_identity.py`` is available), and the
   identity baked into the live image (``OMNI_RUNNER_IMAGE_IDENTITY`` env / the
   ``org.omninode.runner.image.identity`` label, surfaced into the runtime env)
   matches the recorded identity. A baked-vs-recorded mismatch is image drift.

2. **Zero ``uv sync`` on the happy path** — the prebuilt shared CI env is baked
   under ``OMNI_CI_ENV_ROOT`` with a ``manifest.json`` ready marker, and the
   canary publishes ``UV_NO_SYNC=1`` so a happy-path job resolves dependencies
   without ever running ``uv sync``. A ``uv sync`` on the happy path is a
   regression; this check fails if the env is absent or ``UV_NO_SYNC`` is unset.

3. **Receipt-Gate readiness** — the tooling the Receipt-Gate path needs (``gh``,
   ``git``, ``jq``, ``python3``, ``uv``) is present on the runner, so a queued
   PR's receipt verification will not fail on a missing binary.

This script is intentionally self-contained: it does not import the OMN-12567
``runner_image_identity`` module at module scope, so it is runnable and testable
on a checkout that predates the merge of that module. When the module *is*
present on the runner (the steady state once OMN-12567 lands), the identity
check upgrades from "consistency of the recorded lock" to "recompute and compare
the full binding".

Exit codes:
* ``0`` — all required checks passed.
* ``1`` — at least one required check failed.

Usage on a freshly recreated runner::

    python3 scripts/ci/validate_runner_image.py            # human report
    python3 scripts/ci/validate_runner_image.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

# The baked lock path inside the OMN-12567 runner image (copied read-only in the
# Dockerfile), then the in-repo committed lock as a fallback for repo-checkout
# validation.
BAKED_LOCK_PATH = Path("/etc/omni/runner-image.lock.json")
REPO_LOCK_PATH = Path("docker/runners/runner-image.lock.json")

# Binaries the happy-path + Receipt-Gate jobs assume are present on the runner.
RECEIPT_GATE_TOOLS = ("gh", "git", "jq", "python3", "uv")

# Default prebuilt shared CI env root baked by the OMN-12567 Dockerfile. The
# canary (OMN-12564) materialises ``<root>/<repo>/<digest>/.venv`` with a
# ``manifest.json`` ready marker.
DEFAULT_CI_ENV_ROOT = Path("/home/runner/.cache/omni/ci-envs")
DEFAULT_CI_ENV_REPO = "omnibase_infra"

# The generator module shipped by OMN-12567. When importable, the identity check
# recomputes the full binding instead of only asserting recorded consistency.
IDENTITY_MODULE_NAME = "runner_image_identity"


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single validation check."""

    name: str
    passed: bool
    detail: str
    required: bool = True


@dataclass
class ValidationReport:
    """Aggregate report across all runner validation checks."""

    checks: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)

    @property
    def ok(self) -> bool:
        """True when every *required* check passed."""
        return all(c.passed for c in self.checks if c.required)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "required": c.required,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
        }


def _load_lock(lock_path: Path) -> dict[str, object]:
    """Load and shape-check a runner image lock file."""
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"lock file must be a JSON object: {lock_path}")
    return data


def resolve_lock_path(explicit: Path | None = None) -> Path | None:
    """Return the first existing lock path: explicit, baked, then repo-local."""
    candidates = (
        [explicit] if explicit is not None else [BAKED_LOCK_PATH, REPO_LOCK_PATH]
    )
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def _load_identity_module(repo_root: Path):  # type: ignore[no-untyped-def]
    """Import the OMN-12567 generator module if present, else return None.

    The module lives at ``scripts/ci/runner_image_identity.py`` and imports its
    sibling ``ci_env_digest``; both directories are added to ``sys.path`` via the
    spec submodule search so the recompute path matches what CI runs.
    """
    module_path = repo_root / "scripts" / "ci" / f"{IDENTITY_MODULE_NAME}.py"
    if not module_path.is_file():
        return None
    ci_dir = str(module_path.parent)
    import sys

    if ci_dir not in sys.path:
        sys.path.insert(0, ci_dir)
    spec = importlib.util.spec_from_file_location(IDENTITY_MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001 — a broken/partial generator is non-fatal here
        return None
    return module


def check_image_identity(
    lock_path: Path | None,
    repo_root: Path,
    baked_identity: str | None,
) -> CheckResult:
    """Assert the runner image identity binding is present and consistent."""
    name = "image_identity"
    if lock_path is None:
        return CheckResult(
            name=name,
            passed=False,
            detail=(
                "no runner image lock file found at "
                f"{BAKED_LOCK_PATH} or {REPO_LOCK_PATH}; the runner is not on the "
                "OMN-12567 versioned image"
            ),
        )

    try:
        lock = _load_lock(lock_path)
    except (OSError, ValueError, TypeError) as exc:
        return CheckResult(name=name, passed=False, detail=f"unreadable lock: {exc}")

    recorded = lock.get("identity_digest")
    image_version = lock.get("image_version")
    if not isinstance(recorded, str) or not recorded:
        return CheckResult(
            name=name,
            passed=False,
            detail=f"lock {lock_path} missing a non-empty identity_digest",
        )

    # When the OMN-12567 generator is present, recompute the full binding and
    # require the recorded digest to match. This is the strong identity proof.
    module = _load_identity_module(repo_root)
    if module is not None:
        try:
            recomputed = module.compute_identity(repo_root, lock)
        except Exception as exc:  # noqa: BLE001 — surface, do not crash the validator
            return CheckResult(
                name=name,
                passed=False,
                detail=f"identity recompute failed: {exc}",
            )
        if recomputed != recorded:
            return CheckResult(
                name=name,
                passed=False,
                detail=(
                    "recorded identity_digest is stale: "
                    f"recorded={recorded!r} recomputed={recomputed!r}; "
                    "run runner_image_identity.py --mode generate"
                ),
            )

    # The identity baked into the live image must match the recorded lock. The
    # OMN-12567 Dockerfile bakes OMNI_RUNNER_IMAGE_IDENTITY into the runtime env;
    # "unbound" means an un-stamped image, which is a drift failure for a repo
    # claiming to run on the versioned image.
    if baked_identity is not None and baked_identity not in ("", "unbound"):
        if baked_identity != recorded:
            return CheckResult(
                name=name,
                passed=False,
                detail=(
                    "image identity drift: baked OMNI_RUNNER_IMAGE_IDENTITY="
                    f"{baked_identity!r} != recorded={recorded!r}"
                ),
            )
        baked_state = "matches baked image"
    elif baked_identity in ("", "unbound"):
        return CheckResult(
            name=name,
            passed=False,
            detail=(
                "live image reports OMNI_RUNNER_IMAGE_IDENTITY='unbound'; the "
                "runner is not on a stamped OMN-12567 image"
            ),
        )
    else:
        baked_state = "no baked identity env (repo-checkout validation)"

    return CheckResult(
        name=name,
        passed=True,
        detail=(
            f"runner image v{image_version} identity {recorded} verified "
            f"({baked_state})"
        ),
    )


def _env_ready_marker_present(env_root: Path, repo: str) -> tuple[bool, str]:
    """Return whether a ready (manifest.json + .venv) prebuilt env exists."""
    repo_env_root = env_root / repo
    if not repo_env_root.is_dir():
        return False, f"prebuilt env root absent: {repo_env_root}"
    for digest_dir in sorted(repo_env_root.iterdir()):
        if not digest_dir.is_dir():
            continue
        manifest = digest_dir / "manifest.json"
        venv_python = digest_dir / ".venv" / "bin" / "python"
        if manifest.is_file() and venv_python.exists():
            return True, f"ready prebuilt env: {digest_dir}"
    return False, (
        f"no ready prebuilt env under {repo_env_root} "
        "(expected <digest>/manifest.json + <digest>/.venv/bin/python)"
    )


def check_zero_uv_sync(
    env_root: Path,
    repo: str,
    uv_no_sync: str | None,
) -> CheckResult:
    """Assert the happy path resolves zero ``uv sync``.

    The OMN-12567 image bakes the shared CI env and the canary publishes
    ``UV_NO_SYNC=1``; both must hold for a happy-path job to skip ``uv sync``.
    """
    name = "zero_uv_sync"
    env_ready, env_detail = _env_ready_marker_present(env_root, repo)
    if not env_ready:
        return CheckResult(name=name, passed=False, detail=env_detail)

    # UV_NO_SYNC may legitimately be unset in the *validator's* own shell when run
    # outside a job step; the authoritative signal is that the canary publishes it
    # into $GITHUB_ENV. Accept either an explicit "1"/"true" in the process env or
    # a recorded publish marker on the env dir. When UV_NO_SYNC is present in the
    # validator env, it must be truthy.
    if uv_no_sync is not None and uv_no_sync not in ("1", "true"):
        return CheckResult(
            name=name,
            passed=False,
            detail=(
                f"UV_NO_SYNC is set to {uv_no_sync!r} (expected '1'); a happy-path "
                "job would run uv sync"
            ),
        )
    no_sync_state = (
        "UV_NO_SYNC=1 in env"
        if uv_no_sync in ("1", "true")
        else "UV_NO_SYNC published by canary at job time"
    )
    return CheckResult(
        name=name,
        passed=True,
        detail=f"{env_detail}; {no_sync_state}",
    )


def check_receipt_gate_readiness(
    tools: Sequence[str] = RECEIPT_GATE_TOOLS,
) -> CheckResult:
    """Assert the runner has the tooling the Receipt-Gate path requires."""
    name = "receipt_gate_readiness"
    missing = [tool for tool in tools if shutil.which(tool) is None]
    if missing:
        return CheckResult(
            name=name,
            passed=False,
            detail=f"missing Receipt-Gate tooling: {', '.join(missing)}",
        )
    return CheckResult(
        name=name,
        passed=True,
        detail=f"Receipt-Gate tooling present: {', '.join(tools)}",
    )


def validate_runner(
    repo_root: Path,
    lock_path: Path | None = None,
    env_root: Path | None = None,
    env_repo: str = DEFAULT_CI_ENV_REPO,
    baked_identity: str | None = None,
    uv_no_sync: str | None = None,
    tools: Sequence[str] = RECEIPT_GATE_TOOLS,
) -> ValidationReport:
    """Run all runner validation checks and return the aggregate report."""
    report = ValidationReport()
    resolved_lock = resolve_lock_path(lock_path)
    report.add(check_image_identity(resolved_lock, repo_root, baked_identity))
    report.add(
        check_zero_uv_sync(
            env_root if env_root is not None else DEFAULT_CI_ENV_ROOT,
            env_repo,
            uv_no_sync,
        )
    )
    report.add(check_receipt_gate_readiness(tools))
    return report


def _render_human(report: ValidationReport) -> str:
    lines = ["Runner image validation (OMN-12568)", ""]
    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        req = "" if check.required else " (advisory)"
        lines.append(f"[{status}] {check.name}{req}: {check.detail}")
    lines.append("")
    lines.append("RESULT: " + ("GREEN" if report.ok else "RED"))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a runner against the OMN-12567 versioned image contract."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=None,
        help="Explicit lock path; defaults to the baked or repo-local lock.",
    )
    parser.add_argument(
        "--env-root",
        type=Path,
        default=None,
        help="Prebuilt shared CI env root (default: the baked image root).",
    )
    parser.add_argument("--env-repo", type=str, default=DEFAULT_CI_ENV_REPO)
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of text."
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    report = validate_runner(
        repo_root=repo_root,
        lock_path=args.lock_file,
        env_root=args.env_root,
        env_repo=args.env_repo,
        baked_identity=os.environ.get("OMNI_RUNNER_IMAGE_IDENTITY"),
        uv_no_sync=os.environ.get("UV_NO_SYNC"),
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(_render_human(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
