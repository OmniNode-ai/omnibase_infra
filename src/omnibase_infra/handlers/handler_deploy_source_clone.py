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
    guard, together with the agent's own code checkout (the repository root of
    ``DEPLOY_AGENT_DIR``), which the agent updates by pull and never by reset. The
    first creation is seeded at the commit the agent was serving.
``DEPLOY_AGENT_LAB_OVERLAY_SOURCE``
    The overlay source clone, which used to be derived from the deploy source's parent.
    Required only when the fence names the lab dev lane; a non-lab fence never reads
    it (OMN-24677).
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
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from pathlib import Path
from types import ModuleType

import yaml

from omnibase_infra.models.model_deploy_source_clone import ModelDeploySourceClone

logger = logging.getLogger(__name__)

ENV_CLONE_DIR = "DEPLOY_AGENT_REPO_DIR"
ENV_CANONICAL_CLONE = "DEPLOY_AGENT_CANONICAL_CLONE"
ENV_LAB_OVERLAY_SOURCE = "DEPLOY_AGENT_LAB_OVERLAY_SOURCE"
ENV_GIT_MIRROR = "DEPLOY_AGENT_GIT_MIRROR_URL"
ENV_TRACKING_REF = "DEPLOY_AGENT_TRACKING_REF"
ENV_OVERLAY_STORE = "DEV_LANE_OVERLAY_STORE_ROOTS_DIR"
ENV_FENCE = "DEPLOY_AGENT_ALLOWED_LANES"
ENV_AGENT_DIR = "DEPLOY_AGENT_DIR"
TIMED_OUT = 124
MIRROR_FALLBACK_COUNT_FILE = "mirror-fallbacks.count"


class DeploySourceCloneError(RuntimeError):
    """The deploy-source clone cannot be resolved or created."""


class DeploySourceIsCanonicalCloneError(DeploySourceCloneError):
    """The deploy-source path resolves to a clone another tool owns."""


def _fence(env: Mapping[str, str]) -> set[str]:
    return {token.strip() for token in env.get(ENV_FENCE, "").split(",")}


def _code_checkout(env: Mapping[str, str]) -> str:
    """The repository root of the agent's code (``<root>/scripts/deploy-agent``)."""
    agent_dir = env.get(ENV_AGENT_DIR, "").strip()
    return str(Path(agent_dir).parents[1]) if len(Path(agent_dir).parts) > 2 else ""


def settings(env: Mapping[str, str]) -> ModelDeploySourceClone:
    """Read the declared settings; refuse a deploy source that is a canonical clone.

    The overlay source is required only when the fence names the lab dev lane.
    """
    required = [ENV_CLONE_DIR, ENV_CANONICAL_CLONE, ENV_TRACKING_REF]
    if "dev" in _fence(env):
        required.append(ENV_LAB_OVERLAY_SOURCE)
    for name in required:
        if not env.get(name, "").strip():
            raise DeploySourceCloneError(
                f"{name} is not set: the deploy agent declares its own deploy-source "
                "clone, the canonical clone it never mutates, the tracked ref and, "
                "for a lab dev fence, the overlay source clone; none has a default"
            )
    declared = ModelDeploySourceClone(
        operation="resolve",
        clone_dir=env[ENV_CLONE_DIR].strip(),
        canonical_clone=env[ENV_CANONICAL_CLONE].strip(),
        lab_overlay_source=env.get(ENV_LAB_OVERLAY_SOURCE, "").strip(),
        tracking_ref=env[ENV_TRACKING_REF].strip(),
        mirror_url=env.get(ENV_GIT_MIRROR, "").strip(),
        host_root=env.get("OMNI_HOME", "").strip(),
        code_checkout=_code_checkout(env),
        registry_root=env.get("UNIT_DRIFT_REGISTRY_ROOT", "").strip(),
    )
    _refuse_canonical(declared)
    return declared


def _git(
    *args: str, cwd: Path | None = None, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    argv = ["git", *(["-C", str(cwd)] if cwd else []), *args]
    try:
        return subprocess.run(
            argv, capture_output=True, text=True, check=False, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            argv, TIMED_OUT, "", f"git {args[0]} timed out after {timeout}s"
        )


def _git_ok(*args: str, timeout: float, cwd: Path | None = None) -> str:
    result = _git(*args, cwd=cwd, timeout=timeout)
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
    if request.code_checkout:
        candidates.append(Path(request.code_checkout))
    for candidate in candidates:
        real = Path(os.path.realpath(candidate))
        if clone_real == real or real in clone_real.parents:
            raise DeploySourceIsCanonicalCloneError(
                f"{ENV_CLONE_DIR}={request.clone_dir} resolves to the canonical clone "
                f"or the agent's code checkout {candidate}: the agent resets and cleans "
                "its deploy source, and that clone belongs to other tools or is the "
                "code the agent runs"
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


def probe_remote(
    request: ModelDeploySourceClone, *, run: GitRunner
) -> subprocess.CompletedProcess[str]:
    """Probe the mirror, else origin; a mirror miss is a counted fallback.

    Returns the probe that decided the answer, so a refusal keeps git's own stderr.
    """
    origin_url = _origin_url(run)
    if request.mirror_url:
        mirrored = run(["ls-remote", "--exit-code", request.mirror_url, "HEAD"])
        if mirrored.returncode == 0:
            return mirrored
        _count_fallback(
            Path(request.clone_dir), f"mirror unreachable (rc={mirrored.returncode})"
        )
    return run(["ls-remote", "--exit-code", origin_url, "HEAD"])


def remote_reachable(request: ModelDeploySourceClone, *, run: GitRunner) -> bool:
    """Whether the mirror, else origin, answers."""
    return probe_remote(request, run=run).returncode == 0


def execution_runner(execution: ModuleType, timeout: float) -> GitRunner:
    """``git -C <deploy source> <args>`` through the agent's own runner."""
    return lambda args: execution._run(
        ["git", "-C", execution.deploy_repo_dir(), *args], timeout=timeout
    )


def fetch_deploy_source(
    execution: ModuleType, timeout: float
) -> subprocess.CompletedProcess[str]:
    """Fetch the agent's deploy source, mirror first, through the agent's runner."""
    return fetch(execution.deploy_source(), run=execution_runner(execution, timeout))


def probe_deploy_source(
    execution: ModuleType, timeout: float
) -> subprocess.CompletedProcess[str]:
    """Probe the mirror, else origin, for the agent's deploy source."""
    return probe_remote(
        execution.deploy_source(), run=execution_runner(execution, timeout)
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
    timeout = request.timeout_seconds
    if timeout <= 0:
        raise DeploySourceCloneError(
            f"cannot create {clone}: the request declares no timeout, and a clone "
            "that never finishes would hang the agent's start"
        )
    if not (canonical / ".git").exists():
        raise DeploySourceCloneError(
            f"cannot create {clone}: {canonical} is not a git clone, so there is "
            "no origin url to clone from"
        )
    # The canonical clone is read for its origin url only, never for a commit: its
    # HEAD is whatever another tool left there, not what the agent was serving.
    origin_url = _git_ok("remote", "get-url", "origin", cwd=canonical, timeout=timeout)
    fetch_url = _first_answering(
        request,
        origin_url,
        lambda target: _git(
            "ls-remote", "--exit-code", target, "HEAD", timeout=timeout
        ).returncode,
    )
    clone.parent.mkdir(parents=True, exist_ok=True)
    partial = clone.with_name(clone.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)  # leftover of an interrupted creation, agent-owned
    _git_ok("clone", "--no-checkout", fetch_url, str(partial), timeout=timeout)
    if fetch_url != origin_url:
        _git_ok(
            "remote",
            "set-url",
            "--push",
            "origin",
            origin_url,
            cwd=partial,
            timeout=timeout,
        )
    served_sha = request.served_sha or _git_ok(
        "rev-parse",
        "--verify",
        f"refs/remotes/origin/{request.tracking_ref}^{{commit}}",
        cwd=partial,
        timeout=timeout,
    )
    has_commit = _git("cat-file", "-e", f"{served_sha}^{{commit}}", cwd=partial)
    if has_commit.returncode != 0:
        _git_ok("fetch", origin_url, served_sha, cwd=partial, timeout=timeout)
    _git_ok(
        "checkout", "-B", request.tracking_ref, served_sha, cwd=partial, timeout=timeout
    )
    partial.rename(clone)
    logger.info("OMN-24562 created deploy-source clone %s at %s", clone, served_sha)
    return True, read_mirror_fallback_count(clone)


STORE_TARGET = "/etc/onex/skill-overlays-store"


class ComposeMergeTag:
    """A compose merge tag (``!override`` / ``!reset``) on a value."""

    def __init__(self, value: object) -> None:
        self.value = value


class ComposeTagLoader(yaml.SafeLoader):
    """Safe YAML that keeps compose's merge tags instead of refusing them."""


def _tagged(loader: yaml.SafeLoader, node: yaml.Node) -> ComposeMergeTag:
    if isinstance(node, yaml.SequenceNode):
        return ComposeMergeTag(loader.construct_sequence(node, deep=True))
    if isinstance(node, yaml.MappingNode):
        return ComposeMergeTag(loader.construct_mapping(node, deep=True))
    if isinstance(node, yaml.ScalarNode):
        return ComposeMergeTag(loader.construct_scalar(node))
    return ComposeMergeTag(None)


for _tag in ("!override", "!reset"):
    ComposeTagLoader.add_constructor(_tag, _tagged)


def composition_mounts_store(compose_files: Sequence[str]) -> bool:
    """Whether the merged composition mounts the overlay store in any service.

    Files merge in order: a service's ``volumes`` extend the earlier files' list, and
    an ``!override`` or ``!reset`` replaces it, as ``docker compose`` merges them.
    A file that cannot be read counts as mounting the store.
    """
    volumes: dict[str, list[object]] = {}
    for name in compose_files:
        try:
            text = Path(name).read_text("utf-8")
        except OSError:
            return True  # an unreadable composition keeps the refusal: fail closed
        loader = ComposeTagLoader(text)
        try:
            document = loader.get_single_data()
        finally:
            loader.dispose()
        services = (document or {}).get("services") or {}
        for service, spec in services.items():
            raw = spec.get("volumes") if isinstance(spec, dict) else None
            if raw is None:
                continue
            if isinstance(raw, ComposeMergeTag):
                volumes[service] = (
                    list(raw.value) if isinstance(raw.value, list) else []
                )
            else:
                volumes.setdefault(service, []).extend(raw)
    return any(
        _volume_target(volume) == STORE_TARGET
        for mounts in volumes.values()
        for volume in mounts
    )


def _volume_target(volume: object) -> str:
    """A compose volume's container path, short (``src:target[:mode]``) or long form."""
    if isinstance(volume, dict):
        return str(volume.get("target", ""))
    text = str(volume)
    if text.startswith("${"):
        text = text[text.index("}") + 1 :]
    parts = text.split(":")
    return parts[1] if len(parts) > 1 else parts[0]


def overlay_store_root(request: ModelDeploySourceClone) -> str:
    """The populated overlay store the dev runtime mounts, under the registry root.

    The dev compose files default this mount to ``../../overlays/overlays`` relative to
    their own ``docker/`` directory, which was the registry root's overlays clone while
    the agent built from the canonical clone and is an empty path beside the agent's own
    clone. The agent declares it instead, from the registry root (the canonical clone's
    parent when none is declared), and refuses an absent or empty store rather than
    mounting nothing.
    """
    root = request.registry_root or request.canonical_root
    store = Path(root) / "overlays" / "overlays"
    if not store.is_dir() or not any(store.iterdir()):
        raise DeploySourceCloneError(
            f"the dev overlay store {store} is absent or empty: the dev runtime would "
            f"mount nothing at {STORE_TARGET}; set {ENV_OVERLAY_STORE} "
            "or populate the overlays clone"
        )
    return str(store)


def declare_overlay_store(
    env: MutableMapping[str, str],
    request: ModelDeploySourceClone,
    compose_files: Sequence[str],
) -> None:
    """Name the overlay store in a dev lane's compose environment unless it is declared.

    Only an agent whose fence names a lab dev lane and not prod, and whose composition
    carries the store mount, needs the store; an instance whose overlay drops the
    mount starts with none.
    """
    lanes = _fence(env)
    if "dev" not in lanes or "prod" in lanes or env.get(ENV_OVERLAY_STORE):
        return
    if composition_mounts_store(compose_files):
        env[ENV_OVERLAY_STORE] = overlay_store_root(request)


def declare_agent_overlay_store(
    env: MutableMapping[str, str], execution: ModuleType
) -> None:
    """:func:`declare_overlay_store` for the agent's own selected dev composition."""
    configs = execution.dev_instance_lane_configs()
    selected = configs[execution._DevInstanceSelection.name]
    declare_overlay_store(env, execution.deploy_source(), selected.compose_files)


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
