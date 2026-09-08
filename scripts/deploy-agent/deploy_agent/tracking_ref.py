# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The branch this deploy agent tracks and deploys (OMN-16442).

Operator ruling, in-session 2026-09-08, firm: **the deploy agent tracks the
lane's deploy branch, never ``main``.** Any deploy-path reference that resolves
``main`` is a defect.

WHAT THE HARDCODED ``main`` COST, MEASURED
------------------------------------------

``executor.self_update()`` fetched ``origin/main`` and re-execed only when HEAD
differed from it. The .201 dev agent's code clone
(``/data/omninode/omnibase_infra``) sits on ``dev``, which is hundreds of
commits ahead of ``main`` — ``main`` on this repo is release-synced and is by
definition the last release, not a PR-merge target. So the dev agent's
self-update either found itself "already at" a ref it was never on, or would
have reset it backwards onto a released commit. Either way it never picked up
its own fixes (#3305, #3306, #3316), and the manual deploy path was the only
dev deploy path for a whole night (ledger FRICTION row 2026-09-08T09:20:51Z).

The same literal ``origin/main`` was also the default ``git_ref`` on
``ModelRebuildRequested`` and the default recorded as ``requested_git_ref`` on
every completion event — so a command that omitted the ref asked the agent to
``git reset --hard origin/main`` the deploy-source clone, and the completion
event then reported that ref as the deployed lineage.

WHY THIS IS A DECLARED SETTING AND NOT A NEW LITERAL
----------------------------------------------------

Swapping ``"origin/main"`` for ``"origin/dev"`` would move the defect rather
than remove it: the next lane whose deploy branch is neither would inherit a
wrong default silently, exactly as the dev lane just did. The branch a process
deploys is a property of the deployment, not of this source file — the same
argument ``DEPLOY_AGENT_ALLOWED_LANES`` (OMN-16939) and
``KAFKA_SECURITY_PROTOCOL`` (OMN-18012) already carry in this package.

So ``DEPLOY_AGENT_TRACKING_REF`` is REQUIRED and has no default (rule 8:
fail-fast on missing env, never a silent fallback). Both tracked systemd units
declare ``dev``.

THIS IS NOT A PROD PROMOTION PATH
---------------------------------

Setting the base unit's tracking ref to ``dev`` does not change what prod runs.
Prod deploys a pinned, stability-proven **digest** — ``ModelRebuildRequested``
refuses a prod command without ``image_digest`` and prod never rebuilds from a
ref — and prod image promotion is governed by the OMN-13418 grant path, not by
which branch this agent's own code tracks. What this variable governs is which
commit of the *agent* runs and which ref a ref-mode rebuild resets to.
"""

from __future__ import annotations

import os

ENV_TRACKING_REF = "DEPLOY_AGENT_TRACKING_REF"


def load_tracking_ref_from_env() -> str:
    """Return the branch name this agent tracks, e.g. ``dev``.

    Raises ``RuntimeError`` when unset or empty. There is intentionally no
    default: an undeclared tracking ref is how the dev agent spent a night
    comparing itself against a branch it was not on.
    """
    raw = os.environ.get(ENV_TRACKING_REF, "").strip()
    if not raw:
        raise RuntimeError(
            f"{ENV_TRACKING_REF} is required for deploy-agent; it declares the "
            "branch this process tracks for its own self-update and deploys as "
            "the default git ref. There is no default: hardcoding one is the "
            "defect this variable removes (the dev agent compared itself "
            "against a release-synced branch it was never on). Declare it on "
            f"the systemd unit, e.g. {ENV_TRACKING_REF} with the value 'dev'."
        )
    if raw.startswith("origin/"):
        raise RuntimeError(
            f"{ENV_TRACKING_REF}={raw!r} must be a bare branch name, not a "
            "remote-tracking ref; the remote is added where it is needed "
            f"(e.g. {ENV_TRACKING_REF}=dev)."
        )
    return raw


def load_tracking_remote_ref_from_env() -> str:
    """Return the remote-tracking form of the tracking ref, e.g. ``origin/dev``."""
    return f"origin/{load_tracking_ref_from_env()}"
