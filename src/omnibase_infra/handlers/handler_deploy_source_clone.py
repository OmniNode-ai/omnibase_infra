# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent's own deploy-source clone (OMN-24562).

Before this handler the agent's deploy-source clone defaulted to the host's CANONICAL
omnibase_infra clone, and every pinned job ran ``git reset --hard <ref>`` against it.
That clone belongs to other tools (the remote-lane runner, the clone-sync unit,
``refresh_dev_lane.sh``), which then refused a clone that sat on a pinned head 32
commits ahead of dev.

The rule is the class, not the instance: an agent never mutates a clone another tool
owns. Four settings, none with a hidden default (rule 8):

``DEPLOY_AGENT_REPO_DIR``
    The agent's OWN clone, a path under its state directory.
``DEPLOY_AGENT_CANONICAL_CLONE``
    The canonical clone. Never mutated: it seeds the first creation (its HEAD is the
    commit the agent was serving) and anchors the guard.
``DEPLOY_AGENT_LAB_OVERLAY_SOURCE``
    The overlay source clone, which used to be derived from the deploy source's parent.
``DEPLOY_AGENT_GIT_MIRROR_URL``
    The central git mirror (RULING 2026-11-15T19:17:55Z, OMN-23032). Optional: empty
    means no mirror is declared and the clone fetches from origin. No repository file
    declares a mirror URL, so none is invented here.

The deploy agent loads this module from its own checkout, so it imports the standard
library, pydantic and the model only.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

from omnibase_infra.models.model_deploy_source_clone import ModelDeploySourceClone

logger = logging.getLogger(__name__)

ENV_CLONE_DIR = "DEPLOY_AGENT_REPO_DIR"
ENV_CANONICAL_CLONE = "DEPLOY_AGENT_CANONICAL_CLONE"
ENV_LAB_OVERLAY_SOURCE = "DEPLOY_AGENT_LAB_OVERLAY_SOURCE"
ENV_GIT_MIRROR = "DEPLOY_AGENT_GIT_MIRROR_URL"
MIRROR_FALLBACK_COUNT_FILE = "mirror-fallbacks.count"


class DeploySourceCloneError(RuntimeError):
    """The deploy-source clone cannot be resolved or created."""


class DeploySourceIsCanonicalCloneError(DeploySourceCloneError):
    """The deploy-source path resolves to a clone another tool owns."""


def settings(env: Mapping[str, str]) -> ModelDeploySourceClone:
    """Read the declared settings; refuse a deploy source that is a canonical clone."""
    values = {
        name: env.get(name, "").strip()
        for name in (ENV_CLONE_DIR, ENV_CANONICAL_CLONE, ENV_LAB_OVERLAY_SOURCE)
    }
    for name, value in values.items():
        if not value:
            raise DeploySourceCloneError(
                f"{name} is not set: the deploy agent declares its own deploy-source "
                "clone, the canonical clone it never mutates and the overlay source "
                "clone; none has a default"
            )
    return ModelDeploySourceClone(
        operation="resolve",
        clone_dir=values[ENV_CLONE_DIR],
        canonical_clone=values[ENV_CANONICAL_CLONE],
        lab_overlay_source=values[ENV_LAB_OVERLAY_SOURCE],
        mirror_url=env.get(ENV_GIT_MIRROR, "").strip(),
        host_root=env.get("OMNI_HOME", "").strip(),
        registry_root=env.get("UNIT_DRIFT_REGISTRY_ROOT", "").strip(),
    )


def _git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *(["-C", str(cwd)] if cwd else []), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _git_ok(*args: str, cwd: Path | None = None) -> str:
    result = _git(*args, cwd=cwd)
    if result.returncode != 0:
        raise DeploySourceCloneError(
            f"git {args[0]} failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


def read_mirror_fallback_count(clone_dir: Path) -> int:
    """Fallbacks to origin counted so far, kept in a file beside the clone."""
    try:
        text = (clone_dir.parent / MIRROR_FALLBACK_COUNT_FILE).read_text("utf-8")
        return int(text.strip() or "0")
    except (FileNotFoundError, ValueError):
        return 0


def _count_fallback(clone_dir: Path, reason: str) -> int:
    count = read_mirror_fallback_count(clone_dir) + 1
    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    (clone_dir.parent / MIRROR_FALLBACK_COUNT_FILE).write_text(f"{count}\n", "utf-8")
    logger.warning(
        "OMN-24562 git mirror fallback to origin (count=%d): %s", count, reason
    )
    return count


def _refuse_canonical(request: ModelDeploySourceClone) -> None:
    clone_real = Path(os.path.realpath(request.clone_dir))
    candidates = [Path(request.canonical_clone)]
    for root in (request.host_root, request.registry_root):
        if root:
            candidates.append(Path(root) / Path(request.canonical_clone).name)
    for candidate in candidates:
        real = Path(os.path.realpath(candidate))
        if clone_real == real or real in clone_real.parents:
            raise DeploySourceIsCanonicalCloneError(
                f"{ENV_CLONE_DIR}={request.clone_dir} resolves to the canonical clone "
                f"{candidate}: the agent resets and cleans its deploy source, and that "
                "clone belongs to other tools"
            )


def _create(request: ModelDeploySourceClone) -> tuple[bool, int]:
    clone = Path(request.clone_dir)
    canonical = Path(request.canonical_clone)
    if (clone / ".git").exists():
        return False, read_mirror_fallback_count(clone)
    if not (canonical / ".git").exists():
        raise DeploySourceCloneError(
            f"cannot create {clone}: {canonical} is not a git clone, so there is "
            "neither an origin url nor a served commit to seed from"
        )
    origin_url = _git_ok("remote", "get-url", "origin", cwd=canonical)
    served_sha = _git_ok("rev-parse", "HEAD", cwd=canonical)
    branch = _git("symbolic-ref", "--short", "-q", "HEAD", cwd=canonical).stdout.strip()

    fetch_url = origin_url
    fallbacks = read_mirror_fallback_count(clone)
    if request.mirror_url:
        probe = _git("ls-remote", "--exit-code", request.mirror_url, "HEAD")
        if probe.returncode == 0:
            fetch_url = request.mirror_url
        else:
            fallbacks = _count_fallback(
                clone, f"mirror unreachable (rc={probe.returncode})"
            )

    clone.parent.mkdir(parents=True, exist_ok=True)
    partial = clone.with_name(clone.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)  # leftover of an interrupted creation, agent-owned
    _git_ok("clone", "--no-checkout", fetch_url, str(partial))
    if fetch_url != origin_url:
        _git_ok("remote", "set-url", "--push", "origin", origin_url, cwd=partial)
    if _git("cat-file", "-e", f"{served_sha}^{{commit}}", cwd=partial).returncode != 0:
        # An unpushed head: read it from the canonical clone, which a fetch only reads.
        _git_ok("fetch", str(canonical), served_sha, cwd=partial)
    if branch:
        _git_ok("checkout", "-B", branch, served_sha, cwd=partial)
    else:
        _git_ok("checkout", "--detach", served_sha, cwd=partial)
    partial.rename(clone)
    logger.info("OMN-24562 created deploy-source clone %s at %s", clone, served_sha)
    return True, fallbacks


def apply_request(request: ModelDeploySourceClone) -> ModelDeploySourceClone:
    """Guard the declared deploy source, then create the clone when asked to ensure it."""
    _refuse_canonical(request)
    if request.operation == "resolve":
        return request
    created, fallbacks = _create(request)
    return request.model_copy(
        update={"created": created, "mirror_fallbacks": fallbacks}
    )


class HandlerDeploySourceClone:
    """Resolve and, on ensure, create the agent's own deploy-source clone."""

    def handle(self, request: ModelDeploySourceClone) -> ModelDeploySourceClone:
        return apply_request(request)


def ensure_clone(settings: ModelDeploySourceClone) -> ModelDeploySourceClone:
    """Create the own clone when absent; an existing clone and its pin are untouched."""
    return apply_request(settings.model_copy(update={"operation": "ensure"}))
