# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Canonical-venv purity fitness assertion (OMN-15620).

## Why this exists

Node auto-wiring discovery (``omnibase_infra.runtime.auto_wiring.discovery``)
walks the ``onex.nodes`` entry point of *every* installed distribution in the
current interpreter, unconditionally. When a sibling ONEX repo is
hand-installed into the canonical clone's venv (``uv pip install omnimarket``
straight into ``omni_home/omnibase_infra/.venv``, bypassing ``uv sync``) it
starts shipping its own ``onex.nodes`` entries alongside the declared
providers. Two providers of the same node identity collide, discovery logs
``DUPLICATE_REGISTRATION`` and skips the second one, and any test that
asserts on manifest completeness or a specific pass count fails -- not
because the tree changed, but because the gate host's site-packages did.

Measured live on ``.200`` (OMN-15620): the same dev HEAD scored 25 failed /
33 passed on ``tests/unit/runtime/test_kernel.py`` with ``omnimarket`` and
``omninode-memory`` hand-installed and undeclared, versus 58 passed / 0
failed once ``uv sync`` removed them. The repair does not stay durable on its
own: ``uv pip install <anything>`` re-pollutes the venv permanently and
``uv run`` does not self-heal, so the exact same regression reappears the
first time any lane hand-installs into the canonical clone -- which is
measured to have actually happened (the .200 canonical venv was found
re-polluted with the identical 25/33 signature three weeks after the first
repair). This module is the fix for THAT: a fitness assertion that turns the
next recurrence into one named refusal at gate/test-session start instead of
N unexplained test failures.

## What it checks

Enumerates installed distributions that expose an ``onex.nodes`` entry point
(the same entry-point group ``auto_wiring.discovery`` walks) and compares
that set against the distributions ``uv.lock`` actually resolved for this
project. ``uv.lock`` is the authoritative "declared" set: it is the full,
transitively-resolved graph rooted at ``pyproject.toml``, and ``uv sync``
(exact mode, the repo-standard sync command) installs exactly and only what
it lists. Any ONEX-node-providing distribution present in the environment but
absent from the lockfile was not put there by ``uv sync`` -- it was
hand-installed.

## Fails open, not closed, when it cannot determine

Same philosophy as ``omnimarket_drift_guard`` (OMN-14060): this check only
raises when it CAN locate a declared set (``uv.lock`` findable and parseable)
and finds a real discrepancy against it. It returns silently -- never raises
-- when the lockfile cannot be found or parsed, so it stays inert on
pip-installed deployments with no source tree present and never blocks an
environment it cannot reason about.

Related: OMN-15620 (this ticket), OMN-14060 (the co-installed-omnimarket
freshness ticket this is deliberately scoped apart from -- see the written
settlement on both tickets: OMN-14060's drift guard governs the on-demand
``onex skill`` dispatch path and already fails loudly and by name when
omnimarket is absent there; this module governs the canonical gate venv used
for the general test suite, which must stay free of ANY undeclared ONEX-node
provider, omnimarket included).
"""

from __future__ import annotations

import logging
import re
import tomllib
from dataclasses import dataclass
from importlib.metadata import distributions
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "ENTRY_POINT_GROUP",
    "UndeclaredProvider",
    "VenvPurityError",
    "assert_venv_purity",
    "find_undeclared_onex_providers",
]

# Mirrors omnibase_infra.runtime.auto_wiring.discovery.ENTRY_POINT_GROUP.
# Duplicated, not imported: this module exists to run BEFORE anything touches
# the auto-wiring subsystem it protects the test suite for, so it must not
# depend on it. If that constant ever changes, update both.
ENTRY_POINT_GROUP = "onex.nodes"


class VenvPurityError(RuntimeError):
    """Raised when the current interpreter's venv carries an ONEX-node-
    providing distribution that ``uv.lock`` does not declare."""


@dataclass(frozen=True)  # internal-dataclass-ok: module-internal fitness-check result
class UndeclaredProvider:
    """One installed distribution that provides ``onex.nodes`` entry points
    but is absent from ``uv.lock``."""

    name: str
    version: str
    entry_point_names: tuple[str, ...]


def _canonicalize(name: str) -> str:
    """PEP 503 style normalization: 'omnibase_infra', 'omnibase-infra', and
    'Omnibase.Infra' all compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _locate_uv_lock() -> Path | None:
    """Walk up from this file's location to find uv.lock next to
    pyproject.toml.

    Mirrors ``version_compatibility._locate_pyproject``'s walk-up strategy
    and its "None means cannot determine, fail open" contract.
    """
    candidate = Path(__file__).resolve()
    for _ in range(10):
        candidate = candidate.parent
        lock_path = candidate / "uv.lock"
        if lock_path.is_file() and (candidate / "pyproject.toml").is_file():
            return lock_path
    return None


def _declared_package_names(lock_path: Path) -> set[str] | None:
    try:
        with open(lock_path, "rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    packages = data.get("package", [])
    return {
        _canonicalize(pkg["name"])
        for pkg in packages
        if isinstance(pkg, dict) and "name" in pkg
    }


def find_undeclared_onex_providers(
    *,
    lock_path: Path | None = None,
    search_paths: list[str] | None = None,
) -> tuple[UndeclaredProvider, ...]:
    """Return every installed distribution that provides an ``onex.nodes``
    entry point but is absent from ``uv.lock``.

    Args:
        lock_path: Explicit path to uv.lock. Auto-located from this file's
            position in the source tree when omitted (every production call
            site omits it; tests pass an explicit scratch lockfile).
        search_paths: Explicit search list forwarded to
            ``importlib.metadata.distributions(path=...)``. Omitted means
            "search the real, current interpreter's installed distributions"
            (production default); the falsification test passes a scratch
            directory to inject a synthetic undeclared distribution without
            touching the real venv.

    Returns:
        Empty tuple when the lockfile cannot be located/parsed (fail open --
        see module docstring) or when every ``onex.nodes`` provider is
        declared. Otherwise one ``UndeclaredProvider`` per undeclared
        distribution, sorted by name for deterministic output.
    """
    resolved_lock_path = lock_path if lock_path is not None else _locate_uv_lock()
    if resolved_lock_path is None:
        logger.debug("venv_purity: uv.lock not found, skipping (fail open)")
        return ()
    declared = _declared_package_names(resolved_lock_path)
    if declared is None:
        logger.debug(
            "venv_purity: uv.lock at %s unreadable/unparseable, skipping (fail open)",
            resolved_lock_path,
        )
        return ()

    all_distributions = (
        distributions() if search_paths is None else distributions(path=search_paths)
    )
    undeclared: dict[str, UndeclaredProvider] = {}
    for dist in all_distributions:
        name = dist.metadata.get("Name") if dist.metadata else None
        if not name:
            continue
        node_entry_points = tuple(
            ep.name for ep in dist.entry_points if ep.group == ENTRY_POINT_GROUP
        )
        if not node_entry_points:
            continue
        if _canonicalize(name) in declared:
            continue
        undeclared[name] = UndeclaredProvider(
            name=name,
            version=dist.version or "unknown",
            entry_point_names=node_entry_points,
        )
    return tuple(sorted(undeclared.values(), key=lambda p: p.name))


def assert_venv_purity(
    *,
    lock_path: Path | None = None,
    search_paths: list[str] | None = None,
) -> None:
    """Fail loudly and immediately if the venv carries an undeclared
    ONEX-node-providing distribution.

    This is the mechanism half of OMN-15620 AC4. Call it once, as early as
    possible (test-session start, runtime boot) -- see ``tests/conftest.py``
    (``pytest_configure``) and ``RuntimeHostProcess.start()`` for the two
    wired call sites.

    Raises:
        VenvPurityError: one or more undeclared ONEX-node-providing
            distributions are installed. The message names every offender
            and the repair command.
    """
    undeclared = find_undeclared_onex_providers(
        lock_path=lock_path, search_paths=search_paths
    )
    if not undeclared:
        return
    offenders = "; ".join(
        f"{p.name}=={p.version} (entry points: {', '.join(p.entry_point_names)})"
        for p in undeclared
    )
    raise VenvPurityError(
        f"Canonical venv is IMPURE: {len(undeclared)} distribution(s) provide "
        f"'{ENTRY_POINT_GROUP}' entry points but are not declared in uv.lock: "
        f"{offenders}. These collide with declared providers of the same node "
        "identity and manufacture DUPLICATE_REGISTRATION false REDs across "
        "the whole test suite (OMN-15620) rather than a real defect in the "
        "tree. Repair with `uv sync` from the repo root (exact mode removes "
        "anything uv.lock does not declare). If a distribution is needed for "
        "on-demand skill dispatch rather than the gate venv, install it "
        "immediately before that dispatch via "
        "scripts/install-node-skill-package.sh, not by hand into the "
        "canonical clone (see OMN-14060)."
    )
