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
    The canonical clone. Never mutated and never read for a commit: it anchors the
    guard. The first creation is seeded at the commit the agent was serving.
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
from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path
from types import ModuleType

from omnibase_infra.models.model_deploy_source_clone import ModelDeploySourceClone

logger = logging.getLogger(__name__)

ENV_CLONE_DIR = "DEPLOY_AGENT_REPO_DIR"
ENV_CANONICAL_CLONE = "DEPLOY_AGENT_CANONICAL_CLONE"
ENV_LAB_OVERLAY_SOURCE = "DEPLOY_AGENT_LAB_OVERLAY_SOURCE"
ENV_GIT_MIRROR = "DEPLOY_AGENT_GIT_MIRROR_URL"
ENV_TRACKING_REF = "DEPLOY_AGENT_TRACKING_REF"
ENV_OVERLAY_STORE = "DEV_LANE_OVERLAY_STORE_ROOTS_DIR"
ENV_FENCE = "DEPLOY_AGENT_ALLOWED_LANES"
MIRROR_FALLBACK_COUNT_FILE = "mirror-fallbacks.count"


class DeploySourceCloneError(RuntimeError):
    """The deploy-source clone cannot be resolved or created."""


class DeploySourceIsCanonicalCloneError(DeploySourceCloneError):
    """The deploy-source path resolves to a clone another tool owns."""


def settings(env: Mapping[str, str]) -> ModelDeploySourceClone:
    """Read the declared settings; refuse a deploy source that is a canonical clone."""
    values = {
        name: env.get(name, "").strip()
        for name in (
            ENV_CLONE_DIR,
            ENV_CANONICAL_CLONE,
            ENV_LAB_OVERLAY_SOURCE,
            ENV_TRACKING_REF,
        )
    }
    for name, value in values.items():
        if not value:
            raise DeploySourceCloneError(
                f"{name} is not set: the deploy agent declares its own deploy-source "
                "clone, the canonical clone it never mutates, the overlay source "
                "clone and the tracked ref; none has a default"
            )
    return ModelDeploySourceClone(
        operation="resolve",
        clone_dir=values[ENV_CLONE_DIR],
        canonical_clone=values[ENV_CANONICAL_CLONE],
        lab_overlay_source=values[ENV_LAB_OVERLAY_SOURCE],
        tracking_ref=values[ENV_TRACKING_REF],
        mirror_url=env.get(ENV_GIT_MIRROR, "").strip(),
        host_root=env.get("OMNI_HOME", "").strip(),
        registry_root=env.get("UNIT_DRIFT_REGISTRY_ROOT", "").strip(),
    )


def _git(
    *args: str, cwd: Path | None = None, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *(["-C", str(cwd)] if cwd else []), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
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


GitRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]
"""Runs ``git -C <clone> <args>``; the executor's own runner, so its seams stay."""


def local_runner(clone: Path, timeout: float | None = None) -> GitRunner:
    """A runner over real git in ``clone``."""
    return lambda args: _git(*args, cwd=clone, timeout=timeout)


def _origin_url(run: GitRunner) -> str:
    """The real origin: the push URL, which a mirror clone keeps pointed at origin."""
    result = run(["remote", "get-url", "--push", "origin"])
    if result.returncode != 0:
        raise DeploySourceCloneError(f"cannot read origin: {result.stderr.strip()}")
    return result.stdout.strip()


def _first_answering(
    request: ModelDeploySourceClone, origin_url: str, probe: Callable[[str], int]
) -> str:
    """The mirror when declared and answering, else origin; each fallback is counted."""
    if not request.mirror_url:
        return origin_url
    rc = probe(request.mirror_url)
    if rc == 0:
        return request.mirror_url
    _count_fallback(Path(request.clone_dir), f"mirror unreachable (rc={rc})")
    return origin_url


def remote_reachable(request: ModelDeploySourceClone, *, run: GitRunner) -> bool:
    """Whether the mirror, else origin, answers; a mirror miss is a counted fallback."""
    url = _first_answering(
        request,
        _origin_url(run),
        lambda target: run(["ls-remote", "--exit-code", target, "HEAD"]).returncode,
    )
    return run(["ls-remote", "--exit-code", url, "HEAD"]).returncode == 0


def execution_runner(execution: ModuleType, timeout: float) -> GitRunner:
    """``git -C <deploy source> <args>`` through the agent's own runner."""
    return lambda args: execution._run(
        ["git", "-C", execution.deploy_repo_dir(), *args], timeout=timeout
    )


def fetch(
    request: ModelDeploySourceClone, *, run: GitRunner
) -> subprocess.CompletedProcess[str]:
    """Fetch the clone from the mirror, falling back to origin on EVERY fetch.

    A mirror declared at creation can stop answering later; the clone's own fetch URL
    would then fail every deploy, so each fetch tries the mirror first and counts and
    logs the fallback when it does not answer.
    """
    if not request.mirror_url:
        return run(["fetch", "--all", "--prune"])
    refspecs = ["+refs/heads/*:refs/remotes/origin/*", "+refs/tags/*:refs/tags/*"]
    mirrored = run(["fetch", "--prune", request.mirror_url, *refspecs])
    if mirrored.returncode == 0:
        return mirrored
    _count_fallback(
        Path(request.clone_dir),
        f"mirror fetch failed (rc={mirrored.returncode}): {mirrored.stderr[:200]}",
    )
    return run(["fetch", "--prune", _origin_url(run), *refspecs])


def _create(request: ModelDeploySourceClone) -> tuple[bool, int]:
    clone = Path(request.clone_dir)
    canonical = Path(request.canonical_clone)
    if (clone / ".git").exists():
        return False, read_mirror_fallback_count(clone)
    if not (canonical / ".git").exists():
        raise DeploySourceCloneError(
            f"cannot create {clone}: {canonical} is not a git clone, so there is "
            "no origin url to clone from"
        )
    # The canonical clone is read for its origin url only, never for a commit: its
    # HEAD is whatever another tool left there, not what the agent was serving.
    origin_url = _git_ok("remote", "get-url", "origin", cwd=canonical)
    fetch_url = _first_answering(
        request,
        origin_url,
        lambda target: _git("ls-remote", "--exit-code", target, "HEAD").returncode,
    )
    clone.parent.mkdir(parents=True, exist_ok=True)
    partial = clone.with_name(clone.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)  # leftover of an interrupted creation, agent-owned
    _git_ok("clone", "--no-checkout", fetch_url, str(partial))
    if fetch_url != origin_url:
        _git_ok("remote", "set-url", "--push", "origin", origin_url, cwd=partial)
    served_sha = request.served_sha or _git_ok(
        "rev-parse",
        "--verify",
        f"refs/remotes/origin/{request.tracking_ref}^{{commit}}",
        cwd=partial,
    )
    if _git("cat-file", "-e", f"{served_sha}^{{commit}}", cwd=partial).returncode != 0:
        _git_ok("fetch", origin_url, served_sha, cwd=partial)
    _git_ok("checkout", "-B", request.tracking_ref, served_sha, cwd=partial)
    partial.rename(clone)
    logger.info("OMN-24562 created deploy-source clone %s at %s", clone, served_sha)
    return True, read_mirror_fallback_count(clone)


def overlay_store_root(request: ModelDeploySourceClone) -> str:
    """The populated overlay store the dev runtime mounts, beside the canonical clone.

    The dev compose files default this mount to ``../../overlays/overlays`` relative to
    their own ``docker/`` directory, which was the registry root's overlays clone while
    the agent built from the canonical clone and is an empty path beside the agent's own
    clone. The agent declares it instead, from the canonical clone's registry root, and
    refuses an absent or empty store rather than mounting nothing.
    """
    store = Path(request.canonical_root) / "overlays" / "overlays"
    if not store.is_dir() or not any(store.iterdir()):
        raise DeploySourceCloneError(
            f"the dev overlay store {store} is absent or empty: the dev runtime would "
            f"mount nothing at /etc/onex/skill-overlays-store; set {ENV_OVERLAY_STORE} "
            "or populate the overlays clone"
        )
    return str(store)


def declare_overlay_store(
    env: MutableMapping[str, str], request: ModelDeploySourceClone
) -> None:
    """Name the overlay store in a dev lane's compose environment unless it is declared.

    Only an agent whose fence names a lab dev lane and not prod mounts the store.
    """
    lanes = {token.strip() for token in env.get(ENV_FENCE, "").split(",")}
    if "dev" in lanes and "prod" not in lanes and not env.get(ENV_OVERLAY_STORE):
        env[ENV_OVERLAY_STORE] = overlay_store_root(request)


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


def ensure_clone(
    settings: ModelDeploySourceClone, *, served_sha: str
) -> ModelDeploySourceClone:
    """Create the own clone at ``served_sha`` when absent; an existing clone is untouched.

    ``served_sha`` is the commit the agent was serving, empty when unreadable, which
    seeds the clone at the tracked ref's head on origin.
    """
    return apply_request(
        settings.model_copy(update={"operation": "ensure", "served_sha": served_sha})
    )
