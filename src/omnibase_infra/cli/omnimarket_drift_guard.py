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

## Detect, then heal (OMN-17190)

DETECTION is cheap and entirely LOCAL (no network): compare the commit the
current interpreter's omnimarket was installed from against the HEAD of the
already-checked-out canonical clone at ``$OMNI_HOME/omnimarket``.

REPAIR is not this module's policy and never has been -- it belongs to
``scripts/reconcile-workspace-venvs.sh``. What changed in OMN-17190 is *when*
that repair runs. It used to run only when a human read a refusal and typed the
command; now the CLI boundary passes a bound reconciler as ``reconcile=`` and
this module invokes it ONCE on detected drift, re-checks, and continues if the
re-check passes. Operator direction, 2026-08-30: "Why is anything hand built?"

The split is therefore unchanged in substance -- this module still owns no
install logic and still knows nothing about layers, locks, or uv -- and only
the trigger moved from a human to the guard itself. Callers that want the old
pure detect-and-refuse behaviour simply omit ``reconcile``, which remains the
default.

## Which interpreter is this, anyway (OMN-17190 follow-up)

The guard probes THE CURRENT INTERPRETER; the reconciler repairs exactly one
venv (``$OMNI_HOME/omnibase_infra/.venv``). Those coincide only when ``onex``
was resolved to that venv's entry point. On a developer machine it frequently
is not: ``onex`` is documented as an interactive shell alias, aliases do not
exist in non-interactive shells, and PATH there resolves whatever sibling
install came first. Measured on this Mac 2026-08-30, ``bash -lc 'onex ...'``
resolved a ``uv tool`` shim carrying omnibase_infra 0.38.11 and a PyPI
omnimarket, and refused with the pre-self-heal message while the workspace
venv was confirmed ``IN_SYNC``.

Two consequences are handled here. An EDITABLE omnimarket installed from the
canonical clone is not drift and never was -- it is the clone (see
:func:`installed_omnimarket_editable_root`). And when a reconciler is bound but
the running interpreter is not the workspace CLI venv, the guard refuses
DETERMINISTICALLY and names the sanctioned entry point
(``omnibase_infra/scripts/onex``) instead of running a repair that provably
cannot converge.

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
a generic, unhelpful ``onex skill``/``onex node`` "Unknown node" error with no
pointer back to this module or the repair command (the OMN-13829 ->
OMN-14060 -> OMN-14531 recurrence: each time, the venv drifted from
"installed" to "absent", not merely "stale", and the pre-flight guard's
fail-open-on-None path let it pass silently).
"""

from __future__ import annotations

import importlib
import json
import logging
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from urllib.parse import unquote, urlparse

from omnibase_infra.cli.workspace_reconcile import ReconcileFn

__all__ = [
    "DRIFT_OVERRIDE_ENV",
    "OmnimarketDriftError",
    "canonical_local_omnimarket_commit",
    "check_omnimarket_drift",
    "installed_omnimarket_commit",
    "installed_omnimarket_editable_root",
    "running_interpreter_prefix",
    "workspace_cli_prefix",
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


def installed_omnimarket_editable_root() -> Path | None:
    """Return the local source directory an EDITABLE omnimarket was installed
    from, or ``None`` when the install is not editable (or absent).

    An editable install records ``dir_info.editable`` and a ``file://`` URL in
    ``direct_url.json`` and carries NO ``vcs_info``, so
    :func:`installed_omnimarket_commit` reports ``None`` for it -- the same
    answer it gives for "absent" and for "a PyPI wheel". Those three states are
    not the same thing, and conflating them is a live defect (OMN-17190
    follow-up): an interpreter with omnimarket installed EDITABLE from
    ``$OMNI_HOME/omnimarket`` imports that clone's working tree directly, so it
    is at the clone's HEAD by construction and can never drift from it. The
    guard used to call that "NOT INSTALLED from git" and refuse, then hand the
    refusal to a reconciler that repairs a *different* venv -- a refusal no
    amount of reconciling could ever clear.
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
    if not data.get("dir_info", {}).get("editable"):
        return None
    url = data.get("url")
    if not isinstance(url, str) or not url.startswith("file://"):
        return None
    return Path(unquote(urlparse(url).path))


def running_interpreter_prefix() -> str:
    """Return the prefix of the interpreter this process is actually running in.

    A named function rather than an inline ``sys.prefix`` so a test can state
    which interpreter it is modelling. Reading ``sys.prefix`` is not
    configuration -- it is the same kind of live interpreter fact this module
    already reads via ``importlib.metadata``, and the whole guard is a
    statement about "this interpreter".
    """
    return sys.prefix


def workspace_cli_prefix(omni_home: str) -> Path:
    """Return the ONE venv the workspace reconciler repairs.

    ``scripts/reconcile-workspace-venvs.sh`` reconciles
    ``$OMNI_HOME/omnibase_infra/.venv`` and nothing else on the CLI surface. A
    guard running in some OTHER interpreter can therefore detect drift it has
    no way to repair, which is the whole reason the identity check below
    exists.
    """
    return Path(omni_home) / "omnibase_infra" / ".venv"


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
    omni_home: str | None = None,
    *,
    allow_drift: bool = False,
    reconcile: ReconcileFn | None = None,
    running_prefix: str | None = None,
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

    Performs no network I/O of its own. A supplied ``reconcile`` may (it
    installs packages); that is the caller's explicit choice, made by binding
    one, and it happens only after drift has already been detected locally.

    Args:
        omni_home: Canonical workspace root to resolve the reference clone
            from. ``None`` (no ``$OMNI_HOME``) means "cannot determine" and
            fails open.
        allow_drift: Explicit operator opt-out. Keyword-only and defaulting
            to False so refusal stays the default at EVERY call site,
            including ones added later -- a forgotten argument fails closed.
        reconcile: Optional zero-argument repair. When supplied and drift is
            found, it is invoked exactly once and the check is re-run against
            the same canonical clone; the dispatch proceeds only if that
            re-check passes. ``None`` (the default) preserves the pure
            detect-and-refuse behaviour, which is what every non-CLI caller
            and every unit test wants -- a guard that silently shells out
            would be an astonishing default.
        running_prefix: The prefix of the interpreter this dispatch is running
            in, bound at the CLI boundary as :func:`running_interpreter_prefix`
            exactly the way ``allow_drift`` and ``reconcile`` are bound there.
            When supplied together with ``reconcile``, a mismatch against
            :func:`workspace_cli_prefix` is a hard, deterministic refusal
            BEFORE any repair runs -- see the "Foreign interpreter" section
            below. ``None`` (the default) means the caller made no claim about
            its interpreter, so no identity check is performed and the pure
            behaviour every library/test caller relies on is unchanged.

    Raises:
        OmnimarketDriftError: a canonical clone IS present locally, no
            ``reconcile`` repaired the drift, ``allow_drift`` is False, and
            either (a) omnimarket is not installed from git in the current
            interpreter at all (absent, or a non-VCS/PyPI install), or (b) its
            installed commit does not match the canonical local clone's HEAD
            commit. Also raised when a supplied ``reconcile`` FAILED, or ran
            successfully and left the venv still drifted -- in both cases the
            message names the exact command to reproduce.
    """
    canonical = canonical_local_omnimarket_commit(omni_home=omni_home)
    if canonical is None:
        return
    installed = installed_omnimarket_commit()
    if installed == canonical:
        return

    # An EDITABLE install of the canonical clone is not drift -- it IS the
    # clone. `import omnimarket` in this interpreter loads files straight out of
    # $OMNI_HOME/omnimarket, so its code is whatever that working tree currently
    # holds, at whatever HEAD it currently sits on. There is no commit to
    # compare and nothing a reinstall could move it closer to. Treating it as
    # "NOT INSTALLED from git" (which is what the commit probe alone reports,
    # because an editable install records dir_info and no vcs_info) produced a
    # refusal that was both wrong and unclearable -- reproduced live on this
    # Mac 2026-08-30 via /opt/homebrew/bin/onex, whose interpreter carries
    # omnimarket installed editable from the canonical clone.
    if installed is None and omni_home:
        editable_root = installed_omnimarket_editable_root()
        if (
            editable_root is not None
            and editable_root.resolve() == (Path(omni_home) / "omnimarket").resolve()
        ):
            logger.debug(
                "omnimarket is installed EDITABLE from the canonical clone at %s; "
                "it is at clone HEAD by construction.",
                editable_root,
            )
            return

    # Name the exact repair command with its FULL path (not a cwd-relative
    # one) so the message is copy-pasteable from any working directory --
    # the refusal is what an operator sees mid-dispatch, not necessarily
    # from inside $OMNI_HOME/omnibase_infra. Falls back to the relative form
    # only when omni_home itself could not be resolved (should not happen on
    # this branch in production -- canonical is non-None here only when a
    # real omni_home resolved it -- but keeps the message sane if a caller
    # ever reaches this branch without one, e.g. a direct unit test).
    if omni_home:
        infra_scripts = Path(omni_home) / "omnibase_infra" / "scripts"
        repair_cmd = str(infra_scripts / "check-omnimarket-venv-drift.sh")
        install_cmd = str(infra_scripts / "install-node-skill-package.sh")
    else:
        repair_cmd = "scripts/check-omnimarket-venv-drift.sh"
        install_cmd = "scripts/install-node-skill-package.sh"

    if installed is None:
        detail = (
            "omnimarket is NOT INSTALLED from git in this interpreter "
            "(absent, or installed from PyPI/a non-VCS source), but a "
            f"canonical clone exists at $OMNI_HOME/omnimarket (HEAD "
            f"{canonical[:12]}). 'onex skill'/'onex node'/'onex delegate' "
            "dispatch for market-provided nodes (e.g. node_aislop_sweep) "
            f"will fail with 'Unknown node'. Repair with: {install_cmd} "
            f"--execute (or {repair_cmd} --repair)."
        )
    else:
        detail = (
            f"omnimarket venv is STALE: installed commit {installed[:12]} != "
            f"canonical $OMNI_HOME/omnimarket HEAD {canonical[:12]}. Repair with: "
            f"{repair_cmd} --repair (or re-run {install_cmd} --execute directly)."
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

    # ------------------------------------------------------------------ #
    # Self-heal (OMN-17190)
    # ------------------------------------------------------------------ #
    # Drift used to end here, in a refusal that told a human to run a repair
    # command by hand. The refusal was right; the hand-run repair was the
    # defect ("Why is anything hand built?", operator, 2026-08-30). So when a
    # reconciler is bound, run it ONCE and re-check. This is not a bypass: the
    # re-check below is the same comparison, against the same canonical clone,
    # and it still has to pass.
    #
    # This sits AFTER the ``allow_drift`` branch above on purpose: an operator
    # who explicitly accepted this build asked to run against it, not to have
    # it silently replaced underneath them mid-command.
    #
    # Exactly once, deliberately. A reconcile that ran and left the venv still
    # drifted is reporting something the next identical attempt will not fix,
    # and a retry loop on the CLI hot path would turn a clear refusal into a
    # hang.
    # ------------------------------------------------------------------ #
    # Foreign interpreter: refuse deterministically, never reconcile
    # (OMN-17190 follow-up)
    # ------------------------------------------------------------------ #
    # The reconciler repairs exactly ONE venv --
    # $OMNI_HOME/omnibase_infra/.venv. This guard, by contrast, probes
    # whichever interpreter happens to be executing. Those are the same thing
    # only when `onex` was resolved to that venv's entry point, and on a
    # developer machine that is routinely NOT what happens: `onex` is
    # documented as a zsh alias, aliases do not exist in non-interactive
    # shells, and PATH there resolves to whatever sibling install came first
    # (measured on this Mac 2026-08-30: `/opt/homebrew/bin/onex` and a
    # `uv tool` shim at `~/.local/bin/onex`, both of them different
    # interpreters with their own omnimarket state).
    #
    # Reconciling from a foreign interpreter is worse than refusing twice
    # over: the repair CANNOT converge, because the re-check below re-probes
    # this interpreter while the reconciler mutated a different one -- so the
    # dispatch fails anyway, having silently rewritten a venv the operator
    # never named. Refuse first, name the interpreter, and say which entry
    # point is the workspace CLI.
    if reconcile is not None and running_prefix is not None and omni_home:
        workspace_prefix = workspace_cli_prefix(omni_home)
        if Path(running_prefix).resolve() != workspace_prefix.resolve():
            raise OmnimarketDriftError(
                f"{detail} REFUSING to reconcile: this is NOT the workspace "
                f"onex. It is running in {running_prefix} (interpreter "
                f"{sys.executable}), while the reconciler repairs only "
                f"{workspace_prefix}. A repair from here would rewrite a venv "
                f"this process is not running in and STILL leave this dispatch "
                f"drifted. Run the workspace CLI instead:\n"
                f"  {Path(omni_home) / 'omnibase_infra' / 'scripts' / 'onex'} "
                f"<args>\n"
                f"(that wrapper always execs {workspace_prefix / 'bin' / 'onex'} "
                f"and is safe from any shell, aliased or not). If you meant to "
                f"dispatch from THIS interpreter anyway (results are NOT "
                f"evidence), set {DRIFT_OVERRIDE_ENV}=1."
            )

    if reconcile is not None:
        outcome = reconcile()
        if not outcome.ok:
            # The original diagnosis is carried through, not replaced. A
            # refusal that says only "the reconcile failed" has thrown away
            # the two things a reader needs -- WHAT drifted, and the repair
            # command for it -- and left them with a second-order failure to
            # debug instead of the first-order one. The override is named for
            # the same reason it is named everywhere else in this module: it
            # is checked BEFORE the reconcile, so it genuinely works here, and
            # a documented escape hatch withheld from the message does not stop
            # being used -- it just makes the failure a dead end (the exact
            # argument in this module's docstring for naming it at all).
            raise OmnimarketDriftError(
                f"{detail} A reconcile was attempted and FAILED: "
                f"{outcome.detail}. That makes this a BROKEN venv, not merely a "
                f"stale one, so fix the reconcile rather than working around it "
                f"-- re-run it directly and read the error:\n"
                f"  {outcome.command}\n"
                f"To dispatch anyway despite the drift (results are NOT "
                f"evidence), set {DRIFT_OVERRIDE_ENV}=1."
            )

        # The reconcile mutated site-packages out of process. importlib caches
        # directory listings per sys.path entry, so without this the re-probe
        # would faithfully report the pre-repair state and refuse a venv that
        # was just fixed.
        importlib.invalidate_caches()
        installed = installed_omnimarket_commit()
        if installed == canonical:
            logger.info(
                "omnimarket drift reconciled in-flight to %s; continuing.",
                canonical[:12],
            )
            return

        raise OmnimarketDriftError(
            f"{detail} A reconcile ran, reported SUCCESS, and the venv is "
            f"STILL drifted: installed {(installed or 'ABSENT')[:12]} != "
            f"canonical $OMNI_HOME/omnimarket HEAD {canonical[:12]}. The "
            f"reconciler and this guard therefore disagree about what "
            f"'reconciled' means, which no retry will resolve. Reproduce "
            f"with:\n"
            f"  {outcome.command}\n"
            f"To dispatch anyway despite the drift (results are NOT "
            f"evidence), set {DRIFT_OVERRIDE_ENV}=1."
        )

    raise OmnimarketDriftError(
        f"{detail} To dispatch anyway despite the drift (results are NOT "
        f"evidence), set {DRIFT_OVERRIDE_ENV}=1. "
        "See docs/runbooks/node-skill-package-install.md."
    )
