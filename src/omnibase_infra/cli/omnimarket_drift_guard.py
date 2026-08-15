# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pre-flight drift guard for the current interpreter's omnimarket install
(OMN-14060).

## Why this exists

``onex skill <name>`` dispatches to nodes provided by ``omnimarket``, co-installed
into the omnibase_infra venv via ``scripts/install-node-skill-package.sh``
(OMN-13829). That install silently reverts to a stale state whenever something
re-installs ``omnimarket`` from PyPI instead of the canonical git-source
co-install — the OMN-13829 -> OMN-14060 recurrence. Compounding factor
(OMN-14064): PyPI's last published omnimarket release predates the fix that
recurrence exposed by weeks and the newest PyPI release is flat-out
uninstallable (pins a sibling version that was never published), so there is
no PyPI version that would ever be "correct" here.

## Detect vs. repair split

This module only DETECTS drift, cheaply and entirely LOCALLY (no network):
it compares the commit the current interpreter's omnimarket was installed
from against the HEAD of the already-checked-out canonical clone at
``$OMNI_HOME/omnimarket``. It never re-installs anything -- that is
``scripts/install-node-skill-package.sh`` (via
``scripts/check-omnimarket-venv-drift.sh --repair``, meant to run on a
session/cron tick, NOT inline on every dispatch).

The check fails OPEN (no-op, never raises) **only** when the canonical local
clone itself cannot be determined -- e.g. ``OMNI_HOME`` unset, or no
``$OMNI_HOME/omnimarket`` clone present. That keeps the guard silent on CI
runners and fresh machines where the ``$OMNI_HOME/omnimarket`` convention
does not apply, and it never blocks in an environment it cannot reason
about.

On a machine that DOES have the canonical clone, "omnimarket is not
installed from git" (absent entirely, or installed from PyPI/a non-VCS
source) is now a DETERMINABLE, actionable state, not an indeterminate one --
it now raises with a repair pointer instead of failing open (OMN-14531).
Before this, ``installed_omnimarket_commit() is None`` unconditionally
short-circuited to a silent no-op, so the exact regression this module
exists to catch -- ``omnimarket`` silently reverting from a git co-install to
completely absent -- fell through the guard undetected. The only symptom was
a generic, unhelpful ``onex skill``/``onex run`` "Unknown node" error with no
pointer back to this module or the repair command (the OMN-13829 ->
OMN-14060 -> OMN-14531 recurrence: each time, the venv drifted from
"installed" to "absent", not merely "stale", and the pre-flight guard's
fail-open-on-None path let it pass silently).
"""

from __future__ import annotations

import json
import logging
import subprocess
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

__all__ = [
    "DRIFT_OVERRIDE_ENV",
    "OmnimarketDriftError",
    "canonical_local_omnimarket_commit",
    "check_omnimarket_drift",
    "installed_omnimarket_commit",
]

logger = logging.getLogger(__name__)

# The single supported way past a drift refusal (OMN-13930). Named in every
# refusal message so the escape hatch is discoverable from the failure alone.
#
# This module NEVER reads it itself: the value arrives as the ``allow_drift``
# argument, bound at the CLI boundary by click's ``envvar=`` (the same
# mechanism ``--omni-home`` uses). That keeps this module a pure function of
# its arguments and keeps the read out of ``src/`` where the
# ``check-env-reads`` hook (correctly) forbids raw ``os.environ`` access.
# Click's BOOL conversion is what makes the override fail closed: ``0`` /
# ``false`` parse as False, and an unparseable value is a hard usage error,
# so neither one silently disables the guard.
DRIFT_OVERRIDE_ENV = "ONEX_ALLOW_OMNIMARKET_DRIFT"

# Local `git rev-parse HEAD` only -- this never touches the network, so a
# generous timeout still keeps the hot path fast.
_GIT_TIMEOUT_SECONDS = 2


class OmnimarketDriftError(RuntimeError):
    """Raised when the installed omnimarket commit diverges from canonical."""


def installed_omnimarket_commit() -> str | None:
    """Return the git commit SHA the CURRENT interpreter's omnimarket was
    installed from.

    Returns ``None`` when omnimarket is absent, or installed from something
    other than the canonical git+URL co-install (e.g. a PyPI wheel --
    OMN-14064 is exactly this case: PyPI installs carry no ``vcs_info``).
    """
    try:
        dist = distribution("omnimarket")
    except PackageNotFoundError:
        return None
    direct_url_text = dist.read_text("direct_url.json")
    if not direct_url_text:
        return None
    try:
        data = json.loads(direct_url_text)
    except json.JSONDecodeError:
        return None
    commit_id = data.get("vcs_info", {}).get("commit_id")
    return commit_id if isinstance(commit_id, str) and len(commit_id) == 40 else None


def canonical_local_omnimarket_commit(omni_home: str | None = None) -> str | None:
    """Return the checked-out HEAD commit of the canonical local omnimarket
    clone at ``$OMNI_HOME/omnimarket``, or ``None`` when it cannot be
    determined.

    Deliberately a LOCAL ``git rev-parse HEAD`` -- never a live ``git
    ls-remote``. Keeping the canonical clone itself current is the job of
    ``pull-all.sh`` / the repair tick (OMN-14060), not every skill dispatch;
    this function only reads whatever is already checked out.
    """
    if not omni_home:
        return None
    omnimarket_root = Path(omni_home) / "omnimarket"
    if not (omnimarket_root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(omnimarket_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    sha = result.stdout.strip()
    return sha if len(sha) == 40 else None


def check_omnimarket_drift(
    omni_home: str | None = None, *, allow_drift: bool = False
) -> None:
    """Fail fast if the current venv's omnimarket is missing or has drifted
    from the canonical local clone.

    Refusal is the DEFAULT and is never silently skipped. Two ways past it,
    both deliberate:

    * Fails OPEN (returns silently) when the canonical local clone cannot be
      determined -- see the module docstring for why.
    * Downgrades to a loud WARNING when ``allow_drift`` is True -- the
      operator's explicit opt-out, bound at the CLI boundary to
      ``ONEX_ALLOW_OMNIMARKET_DRIFT`` (:data:`DRIFT_OVERRIDE_ENV`,
      OMN-13930). Every refusal message names that variable, so the escape
      hatch is discoverable from the failure itself rather than requiring a
      source read. Before it existed the only workaround was unsetting
      ``$OMNI_HOME``, which disables the guard globally and SILENTLY --
      strictly worse than a named, logged override.

    Never performs network I/O.

    Args:
        omni_home: Canonical workspace root to resolve the reference clone
            from. ``None`` (no ``$OMNI_HOME``) means "cannot determine" and
            fails open.
        allow_drift: Explicit operator opt-out. Keyword-only and defaulting
            to False so refusal stays the default at EVERY call site,
            including ones added later -- a forgotten argument fails closed.

    Raises:
        OmnimarketDriftError: a canonical clone IS present locally,
            ``allow_drift`` is False, and either (a) omnimarket is not
            installed from git in the current interpreter at all (absent, or
            a non-VCS/PyPI install), or (b) its installed commit does not
            match the canonical local clone's HEAD commit.
    """
    canonical = canonical_local_omnimarket_commit(omni_home=omni_home)
    if canonical is None:
        return
    installed = installed_omnimarket_commit()
    if installed == canonical:
        return

    if installed is None:
        detail = (
            "omnimarket is NOT INSTALLED from git in this interpreter "
            "(absent, or installed from PyPI/a non-VCS source), but a "
            f"canonical clone exists at $OMNI_HOME/omnimarket (HEAD "
            f"{canonical[:12]}). 'onex skill'/'onex run'/'onex delegate' "
            "dispatch for market-provided nodes (e.g. node_aislop_sweep) "
            "will fail with 'Unknown node'. Repair with: "
            "scripts/install-node-skill-package.sh --execute (or "
            "scripts/check-omnimarket-venv-drift.sh --repair)."
        )
    else:
        detail = (
            f"omnimarket venv is STALE: installed commit {installed[:12]} != "
            f"canonical $OMNI_HOME/omnimarket HEAD {canonical[:12]}. Repair with: "
            "scripts/check-omnimarket-venv-drift.sh --repair (or re-run "
            "scripts/install-node-skill-package.sh --execute directly)."
        )

    if allow_drift:
        # Loud on every dispatch, by design: a silent bypass would recreate
        # the invisible-drift failure this guard exists to end.
        logger.warning(
            "%s DISPATCHING ANYWAY because %s is set -- results from "
            "market-provided nodes come from an UNVERIFIED omnimarket build "
            "and must not be treated as evidence.",
            detail,
            DRIFT_OVERRIDE_ENV,
        )
        return

    raise OmnimarketDriftError(
        f"{detail} To dispatch anyway despite the drift (results are NOT "
        f"evidence), set {DRIFT_OVERRIDE_ENV}=1. "
        "See docs/runbooks/node-skill-package-install.md."
    )
