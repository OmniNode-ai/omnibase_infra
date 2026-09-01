# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Collect this process's runtime-identity stamp (OMN-17310, epic OMN-17306).

The I/O half of the stamp. The model lives in ``omnibase_core``
(:class:`~omnibase_core.models.runtime.model_runtime_identity.ModelRuntimeIdentity`)
because the receipt that carries it lives there; collection lives HERE because
it reads ``importlib.metadata``, PEP 610 ``direct_url.json``, the container's
build-provenance manifest, the hostname and the interpreter path — none of
which belongs in core.

## What it answers

Four questions, and nothing else:

* **what code** — per distribution, the version AND the commit
* **on what host** — ``gethostname()``
* **out of what venv or container** — ``sys.prefix``, or the container id
* **against what config** — the resolved contract path, supplied by the caller

## Why the commit and not just the version

Every incident in the OMN-17306 class turned on reading a version label as a
statement about content. The two disagree routinely:

* the ``.201`` dev lane advertised registry ``0.38.16`` while its vendored
  ``omnimarket`` sat 11 commits behind ``origin/dev`` at ``05e3882f9``
  (OMN-17291);
* a local venv pinning ``omnimarket 0.4.11`` at ``66b7131a3`` produced a probe
  whose output was read as a statement about that same lane (OMN-16932 /
  OMN-17295).

Both are invisible in a version string and both are obvious in a commit.

## Where the commit comes from

Two sources, in order:

1. **PEP 610** ``direct_url.json`` — ``vcs_info.commit_id`` for a git install.
   This generalises ``cli/omnimarket_drift_guard.py::installed_omnimarket_commit``,
   which already did exactly this read for one package; the logic is lifted
   rather than duplicated in spirit.
2. **The container build-provenance manifest** — ``/app/build-provenance.json``,
   written at image build by ``scripts/runtime_build/compute_workspace_provenance.py``.
   A workspace-mode image installs its siblings from
   ``file:///workspace/sibling-repos/<repo>``, so ``direct_url.json`` records a
   local path and NO commit; the manifest's ``per_repo_vcs_provenance.siblings``
   block is the only place the vendored SHA survives (``.git`` is stripped from
   the staged tree). Shape verified live against ``omninode-runtime`` on
   ``.201``, 2026-08-31.

A sibling the manifest marks ``vcs_dirty: true`` is reported
:attr:`~omnibase_core.enums.enum_package_source_kind.EnumPackageSourceKind.UNKNOWN`
with no commit, not ``LOCAL_PATH`` with one: a dirty tree's HEAD does not
identify its content, and naming it anyway would be the same false precision
the stamp exists to remove.

## Cost

One ``gethostname()``, one ``sys.prefix`` read, at most one small JSON parse,
and one ``importlib.metadata`` lookup per package. The expensive parts are
memoised for the life of the process — the identity of a running interpreter
cannot change under it.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import socket
import sys
from datetime import UTC, datetime
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

from omnibase_core.enums.enum_execution_locus_kind import EnumExecutionLocusKind
from omnibase_core.enums.enum_package_source_kind import EnumPackageSourceKind
from omnibase_core.models.runtime.model_package_identity import ModelPackageIdentity
from omnibase_core.models.runtime.model_runtime_identity import ModelRuntimeIdentity

logger = logging.getLogger(__name__)

__all__ = [
    "BUILD_PROVENANCE_PATH",
    "STAMPED_PACKAGES",
    "collect_runtime_identity",
    "render_identity_line",
]

# The distributions whose identity decides whether a receipt describes a local
# venv or a deployed lane: omnimarket owns the node/handler set, omnibase_infra
# the runtime and CLI, omnibase_core the models and dispatch. The core gate's
# DEFAULT_REQUIRED_PACKAGES is those three — the MINIMUM. This set is wider on
# purpose: a runtime image's build-provenance manifest declares every vendored
# sibling (measured live on the .201 dev lane, 2026-08-31: omnibase_core,
# omnibase_compat, omnimarket), and a declared package the stamp is silent
# about is an UNKNOWN refusal in the probe-target assertion. Stamping the full
# layer stack (compat -> core -> spi -> infra, plus omnimarket) means the
# assertion compares content rather than refusing for lack of coverage.
STAMPED_PACKAGES: tuple[str, ...] = (
    "omnibase_compat",
    "omnibase_core",
    "omnibase_infra",
    "omnibase_spi",
    "omnimarket",
)

# Written at image build by compute_workspace_provenance.py. Absent outside a
# runtime image, which is not an error — it just means direct_url.json is the
# only commit source available.
BUILD_PROVENANCE_PATH = Path("/app/build-provenance.json")

# Container detection. `/.dockerenv` is present in every Docker container;
# /proc/self/cgroup is the Linux fallback that also covers containerd.
_DOCKERENV_PATH = Path("/.dockerenv")
_CGROUP_PATH = Path("/proc/self/cgroup")

_COMMIT_LENGTH = 40


@lru_cache(maxsize=1)
def _build_provenance_siblings() -> dict[str, tuple[str | None, bool]]:
    """Return ``{repo: (vcs_ref, vcs_dirty)}`` from the container manifest.

    Empty outside a runtime image, or when the manifest is unreadable. An
    unreadable manifest is logged and treated as ABSENT rather than raised:
    the stamp's job is to report what it can determine, and "no commit from
    this source" is then reported honestly by the caller rather than guessed.
    """
    if not BUILD_PROVENANCE_PATH.is_file():
        return {}
    try:
        parsed = json.loads(BUILD_PROVENANCE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "runtime identity: build-provenance manifest at %s is unreadable "
            "(%s); sibling commits will be reported as unresolved",
            BUILD_PROVENANCE_PATH,
            type(exc).__name__,
        )
        return {}
    if not isinstance(parsed, dict):
        return {}
    block = parsed.get("per_repo_vcs_provenance")
    if not isinstance(block, dict):
        return {}
    siblings = block.get("siblings")
    if not isinstance(siblings, dict):
        return {}
    resolved: dict[str, tuple[str | None, bool]] = {}
    for repo, entry in siblings.items():
        if not isinstance(entry, dict):
            continue
        ref = entry.get("vcs_ref")
        dirty = bool(entry.get("vcs_dirty", False))
        resolved[str(repo)] = (
            ref if isinstance(ref, str) and len(ref) == _COMMIT_LENGTH else None,
            dirty,
        )
    return resolved


def _direct_url_commit(dist_name: str) -> tuple[str | None, bool]:
    """Return ``(commit, is_local_path)`` from PEP 610 install metadata.

    ``(None, False)`` for a registry (PyPI) install: a wheel carries no
    ``direct_url.json`` at all, or one without ``vcs_info``. That is an honest
    "no commit exists", not a failure — the OMN-14064 case.
    """
    try:
        dist = distribution(dist_name)
    except PackageNotFoundError:
        return None, False
    raw = dist.read_text("direct_url.json")
    if not raw:
        return None, False
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, False
    if not isinstance(data, dict):
        return None, False
    vcs_info = data.get("vcs_info")
    if isinstance(vcs_info, dict):
        commit = vcs_info.get("commit_id")
        if isinstance(commit, str) and len(commit) == _COMMIT_LENGTH:
            return commit, False
    return None, "dir_info" in data or str(data.get("url", "")).startswith("file://")


def _declared_local_root(dist_name: str) -> Path | None:
    """Return the local tree a ``file://`` install DECLARES it came from.

    An editable or workspace install legitimately imports from outside
    site-packages, and that is not shadowing — it is the install working as
    declared. Distinguishing the two is the whole point: without this, every
    editable install would be reported SHADOWED and the signal would be noise
    within a day.
    """
    try:
        dist = distribution(dist_name)
    except PackageNotFoundError:
        return None
    raw = dist.read_text("direct_url.json")
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    url = data.get("url")
    if not isinstance(url, str) or not url.startswith("file://"):
        return None
    try:
        return Path(url[len("file://") :]).resolve()
    except (OSError, ValueError):
        return None


def _import_root(dist_name: str) -> Path | None:
    """Return the directory the interpreter ACTUALLY imports the module from.

    ``find_spec`` resolves through ``sys.path`` exactly as a real import would,
    including a ``PYTHONPATH`` entry or a ``.pth`` redirect that wins over
    site-packages, and does so without executing the module.
    """
    try:
        spec = importlib.util.find_spec(dist_name)
    except (ImportError, ValueError, AttributeError):
        # A namespace collision or a partially-initialised parent makes the
        # import path undeterminable. Reported as "cannot tell" by the caller
        # rather than assumed fine.
        return None
    if spec is None or spec.origin is None:
        return None
    try:
        return Path(spec.origin).resolve().parent
    except (OSError, ValueError):
        return None


def _shadowing_import_root(dist_name: str) -> Path | None:
    """Return the winning import root when it is NOT the installed one.

    ``None`` means metadata and the interpreter agree — the ordinary case.
    """
    actual = _import_root(dist_name)
    if actual is None:
        return None
    try:
        # ``locate_file`` is typed as ``SimplePath``, a structural protocol that
        # is NOT ``os.PathLike`` — every real implementation stringifies to a
        # filesystem path, so ``str()`` is the supported conversion and the one
        # ``mypy src/omnibase_infra`` accepts.
        expected = Path(str(distribution(dist_name).locate_file(dist_name))).resolve()
    except (PackageNotFoundError, OSError, ValueError):
        return None
    if actual == expected:
        return None
    declared_root = _declared_local_root(dist_name)
    if declared_root is not None and (
        actual == declared_root or declared_root in actual.parents
    ):
        # Imports from the tree the install itself points at: declared, not
        # shadowed.
        return None
    return actual


def _package_identity(dist_name: str) -> ModelPackageIdentity:
    """Resolve one distribution's identity, or record it ABSENT."""
    try:
        version = distribution(dist_name).version
    except PackageNotFoundError:
        # Recorded explicitly, not omitted: "omnimarket is not installed" is
        # an identity fact and the exact regression OMN-14060/OMN-14531 kept
        # re-discovering behind a fail-open None.
        return ModelPackageIdentity(
            name=dist_name,
            version=None,
            commit=None,
            source=EnumPackageSourceKind.ABSENT,
        )

    # Checked BEFORE any version/commit is attributed: when the interpreter
    # imports a different tree than the metadata describes, that version and
    # that commit both name code which will not execute. Reporting them under
    # any other source would be the OMN-17306 substitution in miniature --
    # reproduced live 2026-08-31 under PYTHONPATH, where the stamp said
    # `omnibase_core=0.47.1@registry` while 0.47.2 worktree source ran.
    shadowing_root = _shadowing_import_root(dist_name)
    if shadowing_root is not None:
        return ModelPackageIdentity(
            name=dist_name,
            version=version,
            commit=None,
            source=EnumPackageSourceKind.SHADOWED,
            import_path=str(shadowing_root),
        )

    commit, is_local_path = _direct_url_commit(dist_name)
    if commit is not None:
        return ModelPackageIdentity(
            name=dist_name,
            version=version,
            commit=commit,
            source=EnumPackageSourceKind.VCS,
        )

    if is_local_path:
        manifest_ref, dirty = _build_provenance_siblings().get(dist_name, (None, False))
        if manifest_ref is not None and not dirty:
            return ModelPackageIdentity(
                name=dist_name,
                version=version,
                commit=manifest_ref,
                source=EnumPackageSourceKind.LOCAL_PATH,
            )
        if dirty:
            # A dirty staged tree's HEAD does not identify its content.
            return ModelPackageIdentity(
                name=dist_name,
                version=version,
                commit=None,
                source=EnumPackageSourceKind.UNKNOWN,
            )
        return ModelPackageIdentity(
            name=dist_name,
            version=version,
            commit=None,
            source=EnumPackageSourceKind.LOCAL_PATH,
        )

    return ModelPackageIdentity(
        name=dist_name,
        version=version,
        commit=None,
        source=EnumPackageSourceKind.REGISTRY,
    )


@lru_cache(maxsize=1)
def _collect_packages(names: tuple[str, ...]) -> tuple[ModelPackageIdentity, ...]:
    return tuple(_package_identity(name) for name in names)


@lru_cache(maxsize=1)
def _execution_locus() -> tuple[EnumExecutionLocusKind, str]:
    """Return ``(kind, locus)`` for the interpreter this process is running in.

    A container id binds a receipt to a running container rather than to a
    mutable image tag — the distinction OMN-17291 turned on, where the tag was
    fresh and the content was not. A venv prefix distinguishes the CLI venv
    from the daemon venv from a worktree venv (OMN-17190).
    """
    container_id = _container_id()
    if container_id is not None:
        return EnumExecutionLocusKind.CONTAINER, container_id
    prefix = sys.prefix
    if prefix != sys.base_prefix:
        return EnumExecutionLocusKind.VENV, prefix
    return EnumExecutionLocusKind.SYSTEM, prefix


def _container_id() -> str | None:
    """Return this container's id, or ``None`` when not containerised."""
    if not _DOCKERENV_PATH.exists() and not _CGROUP_PATH.exists():
        return None
    try:
        cgroup = _CGROUP_PATH.read_text(encoding="utf-8")
    except OSError:
        cgroup = ""
    for line in cgroup.splitlines():
        # docker: .../docker-<64hex>.scope ; containerd: .../<64hex>
        for token in line.replace("/", " ").replace("-", " ").replace(".", " ").split():
            if len(token) == 64 and all(c in "0123456789abcdef" for c in token):
                return token
    if _DOCKERENV_PATH.exists():
        # Containerised but the id is not recoverable from cgroup v2's
        # namespaced view. The hostname is Docker's default container-id
        # prefix and is the best remaining binding.
        return socket.gethostname()
    return None


def collect_runtime_identity(
    *,
    config_source: str | None = None,
    packages: tuple[str, ...] = STAMPED_PACKAGES,
) -> ModelRuntimeIdentity:
    """Return this process's runtime-identity stamp.

    Args:
        config_source: the contract/config surface this execution resolved
            against, when there is one. A location only — never a value, and
            never a secret.
        packages: distributions to stamp. Defaults to :data:`STAMPED_PACKAGES`.

    The package, locus and interpreter reads are memoised for the process
    lifetime; ``stamped_at`` is fresh on every call so a stamp always dates the
    moment it was taken rather than the moment the process started.
    """
    locus_kind, locus = _execution_locus()
    entries = _collect_packages(packages)
    return ModelRuntimeIdentity(
        host=socket.gethostname(),
        locus_kind=locus_kind,
        execution_locus=locus,
        interpreter=sys.executable,
        packages={entry.name: entry for entry in entries},
        config_source=config_source,
        stamped_at=datetime.now(UTC),
    )


def render_identity_line(identity: ModelRuntimeIdentity) -> str:
    """Render the stamp as ONE terminal line.

    Deliberately one line and deliberately lossy: the full block travels in the
    receipt, and this is the glance-check that would have made the OMN-17295
    probe self-evidently local ("locus=venv:...") instead of ambiguous. Never
    written to stdout — stdout carries exactly one receipt JSON.
    """
    parts = [
        f"host={identity.host}",
        f"locus={identity.locus_kind.value}:{_short(identity.execution_locus)}",
    ]
    for name in sorted(identity.packages):
        entry = identity.packages[name]
        version = entry.version or "absent"
        commit = entry.commit[:7] if entry.commit else entry.source.value
        parts.append(f"{name}={version}@{commit}")
    return "identity: " + " ".join(parts)


def _short(locus: str) -> str:
    """Trim a locus for the one-line render, keeping the distinguishing tail."""
    if len(locus) <= 32:
        return locus
    return "..." + locus[-29:]
