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

The check fails OPEN (no-op, never raises) whenever either side cannot be
determined locally -- e.g. ``OMNI_HOME`` unset, no canonical clone present,
or omnimarket not installed from git (PyPI/no install). This keeps the guard
silent on CI runners and fresh machines where the ``$OMNI_HOME/omnimarket``
convention does not apply, and it never blocks in an environment it cannot
reason about.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

__all__ = [
    "OmnimarketDriftError",
    "canonical_local_omnimarket_commit",
    "check_omnimarket_drift",
    "installed_omnimarket_commit",
]

logger = logging.getLogger(__name__)

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
    resolved_home = (
        omni_home if omni_home is not None else os.environ.get("OMNI_HOME", "")
    )
    if not resolved_home:
        return None
    omnimarket_root = Path(resolved_home) / "omnimarket"
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


def check_omnimarket_drift() -> None:
    """Fail fast if the current venv's omnimarket has drifted from the
    canonical local clone.

    Fails OPEN (returns silently) whenever either side cannot be determined
    locally -- see the module docstring for why. Never performs network I/O.

    Raises:
        OmnimarketDriftError: installed commit does not match the canonical
            local clone's HEAD commit.
    """
    installed = installed_omnimarket_commit()
    if installed is None:
        return
    canonical = canonical_local_omnimarket_commit()
    if canonical is None:
        return
    if installed != canonical:
        raise OmnimarketDriftError(
            f"omnimarket venv is STALE: installed commit {installed[:12]} != "
            f"canonical $OMNI_HOME/omnimarket HEAD {canonical[:12]}. Repair with: "
            "scripts/check-omnimarket-venv-drift.sh --repair (or re-run "
            "scripts/install-node-skill-package.sh --execute directly). "
            "See docs/runbooks/node-skill-package-install.md."
        )
