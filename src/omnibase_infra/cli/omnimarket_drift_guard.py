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
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

from omnibase_infra.cli.workspace_reconcile import ReconcileFn

__all__ = [
    "DRIFT_OVERRIDE_ENV",
    "CanonicalCloneAttachment",
    "OmnimarketDriftError",
    "canonical_clone_attachment",
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


@dataclass(frozen=True)
class PathOnexIdentity:
    status: PathOnexResolutionStatus
    executable: str | None = None
    identity: Path | None = None
    resolution_error: str | None = None

    def __post_init__(self) -> None:
        if self.status is PathOnexResolutionStatus.RESOLVED:
            if self.executable is None or self.identity is None:
                raise ValueError(
                    "resolved PATH onex identity requires executable and identity"
                )
            if self.resolution_error is not None:
                raise ValueError("resolved PATH onex identity cannot carry an error")
        elif self.status is PathOnexResolutionStatus.LOOKUP_FAILED:
            if self.executable is not None or self.identity is not None:
                raise ValueError(
                    "failed PATH lookup cannot carry an executable identity"
                )
            if self.resolution_error is None:
                raise ValueError("failed PATH lookup requires an error")
        elif self.status is PathOnexResolutionStatus.RESOLVE_FAILED:
            if self.executable is None or self.resolution_error is None:
                raise ValueError(
                    "failed PATH onex resolution requires executable and error"
                )
            if self.identity is not None:
                raise ValueError("failed PATH onex resolution cannot carry identity")


class PathOnexResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    LOOKUP_FAILED = "lookup_failed"
    RESOLVE_FAILED = "resolve_failed"


_DIAGNOSTIC_EXCEPTIONS = (OSError, RuntimeError, ValueError)


def _diagnostic_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


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


def _path_onex_identity() -> PathOnexIdentity | None:
    """Return PATH's ``onex`` entry and its normalized filesystem identity.

    A drift refusal from a foreign interpreter is usually caused by a second
    ``onex`` entrypoint preceding the sanctioned wrapper. The raw PATH entry is
    the actionable location to inspect; the resolved path is only for a
    symlink-safe filesystem identity comparison with the canonical wrapper. A
    diagnostic helper must not replace the drift error with an unrelated PATH
    or filesystem exception.
    """
    try:
        path_onex = shutil.which("onex")
    except _DIAGNOSTIC_EXCEPTIONS as exc:
        return PathOnexIdentity(
            status=PathOnexResolutionStatus.LOOKUP_FAILED,
            resolution_error=_diagnostic_error(exc),
        )
    if not path_onex:
        return None
    try:
        return PathOnexIdentity(
            status=PathOnexResolutionStatus.RESOLVED,
            executable=path_onex,
            identity=Path(path_onex).resolve(),
        )
    except _DIAGNOSTIC_EXCEPTIONS as exc:
        return PathOnexIdentity(
            status=PathOnexResolutionStatus.RESOLVE_FAILED,
            executable=path_onex,
            resolution_error=_diagnostic_error(exc),
        )


class CanonicalCloneAttachment(StrEnum):
    """Whether the canonical clone has a branch checked out.

    Three states, kept distinct on purpose. Collapsing DETACHED into
    UNDETERMINED is the exact defect this enum exists to prevent: the guard
    fails OPEN on UNDETERMINED, so a detached clone folded into that value
    would keep reporting clean -- which is the OMN-17313 bug, reintroduced one
    layer down.
    """

    ATTACHED = "attached"
    DETACHED = "detached"
    UNDETERMINED = "undetermined"


def canonical_clone_attachment(
    omni_home: str | None = None,
) -> CanonicalCloneAttachment:
    """Report whether the canonical local omnimarket clone is on a branch.

    ``git symbolic-ref --quiet HEAD`` is the probe: it succeeds with the full
    ref name on an attached HEAD and exits non-zero on a detached one. That is
    the only signal that distinguishes the two -- ``rev-parse HEAD``, which
    :func:`canonical_local_omnimarket_commit` uses, returns a valid sha in both
    states, which is precisely why detachment was invisible to this guard.

    Returns :attr:`CanonicalCloneAttachment.UNDETERMINED` when the clone cannot
    be reached at all (no ``$OMNI_HOME``, no clone, git unavailable, timeout).
    A git invocation that RAN and reported "not a symbolic ref" is not
    undetermined: it is DETACHED, and is reported as such.
    """
    if not omni_home:
        return CanonicalCloneAttachment.UNDETERMINED
    omnimarket_root = Path(omni_home) / "omnimarket"
    if not (omnimarket_root / ".git").exists():
        return CanonicalCloneAttachment.UNDETERMINED
    try:
        result = subprocess.run(
            ["git", "-C", str(omnimarket_root), "symbolic-ref", "--quiet", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return CanonicalCloneAttachment.UNDETERMINED
    symbolic_ref = result.stdout.strip()
    if result.returncode == 0 and symbolic_ref.startswith("refs/"):
        return CanonicalCloneAttachment.ATTACHED
    # git ran and declined to resolve HEAD to a branch. Exit status 1 is the
    # documented "not a symbolic ref" answer; anything else here is a repo git
    # could not read as a branch either, and a canonical clone in that state is
    # no more usable as a reference point than a detached one.
    return CanonicalCloneAttachment.DETACHED


def check_omnimarket_drift(
    omni_home: str | None = None,
    *,
    allow_drift: bool = False,
    reconcile: ReconcileFn | None = None,
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

    # OMN-17313: a DETACHED canonical clone is drift in its own right, and it
    # has to be judged BEFORE the commit comparison below -- because that
    # comparison PASSES on exactly this fault. The venv is pinned to the local
    # clone HEAD by reconcile-workspace-venvs.sh (OMN-16366), so when the clone
    # detaches the venv faithfully reproduces the frozen commit and the two
    # sides agree. Drift then reads as zero while both are arbitrarily stale
    # relative to the upstream branch. Live case, 2026-08-31: the clone sat
    # detached at a commit on an unmerged PR branch that existed on no remote
    # for two days, this guard reported clean the whole time, and the routing
    # authority resolved pre-fix content from that tree (OMN-6790 regression
    # re-appearing on the client path).
    #
    # No `reconcile` is attempted on this branch. The bound reconciler installs
    # packages into the venv; it cannot re-attach a git clone or make an
    # unreadable clone trustworthy, so invoking it here would burn an install
    # and then refuse anyway with a message about the wrong subsystem. The
    # sanctioned repair is the converge script, which accepts a detached HEAD
    # as of OMN-17313.
    attachment = CanonicalCloneAttachment.ATTACHED
    omni_home_path = Path(omni_home) if omni_home else None
    if omni_home_path and (omni_home_path / "omnimarket" / ".git").exists():
        attachment = canonical_clone_attachment(omni_home=omni_home)
    if attachment is not CanonicalCloneAttachment.ATTACHED:
        assert omni_home_path is not None
        converge_cmd = str(
            omni_home_path / "omniclaude" / "scripts" / "converge-canonical-clone.sh"
        )
        if attachment is CanonicalCloneAttachment.DETACHED:
            clone_detail = (
                f"canonical $OMNI_HOME/omnimarket clone is on a DETACHED HEAD "
                f"at {canonical[:12]} -- it tracks no branch, so it can no "
                f"longer follow its upstream"
            )
        else:
            clone_detail = (
                f"canonical $OMNI_HOME/omnimarket clone HEAD is {canonical[:12]}, "
                "but the guard could not prove that HEAD is attached to a ref"
            )
        attachment_detail = (
            f"{clone_detail}, and every consumer pinned to it (this venv, "
            f"BIFROST_CONTRACT_PATH, any contract path resolved from that tree) "
            f"is frozen with it. The installed-commit comparison CANNOT see "
            f"this: the venv is pinned to the clone HEAD, so both sides agree "
            f"while both are stale. Repair with: {converge_cmd} omnimarket "
            f"--execute (add --to-branch <name> if the re-attachment target "
            f"cannot be derived from the reflog)."
        )
        if allow_drift:
            logger.warning(
                "%s DISPATCHING ANYWAY because %s is set -- results from "
                "market-provided nodes come from an UNVERIFIED omnimarket build "
                "and must not be treated as evidence.",
                attachment_detail,
                DRIFT_OVERRIDE_ENV,
            )
            return
        else:
            raise OmnimarketDriftError(
                f"{attachment_detail} To dispatch anyway despite the drift "
                f"(results are NOT evidence), set {DRIFT_OVERRIDE_ENV}=1."
            )

    installed = installed_omnimarket_commit()
    if installed == canonical:
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
        # Name the interpreter (OMN-17190). "omnimarket is not installed" is
        # ambiguous between two very different faults: the CLI venv genuinely
        # lost its provider layer, or this is not the CLI venv at all. The
        # second is what actually happened -- `uv run --project X onex` falls
        # back to the first `onex` on PATH whenever the project entrypoint is
        # not resolvable, and that other interpreter refuses identically
        # regardless of the real venv's state. Printing sys.executable turns
        # the next occurrence into a one-line diagnosis instead of a session.
        canonical_wrapper_path = (
            Path(omni_home) / "omnibase_infra" / "scripts" / "onex"
            if omni_home
            else None
        )
        canonical_wrapper = (
            str(canonical_wrapper_path)
            if canonical_wrapper_path is not None
            else "$OMNI_HOME/omnibase_infra/scripts/onex"
        )
        path_onex_identity = _path_onex_identity()
        canonical_wrapper_resolution_error = None
        try:
            canonical_wrapper_identity = (
                canonical_wrapper_path.resolve()
                if canonical_wrapper_path is not None
                else None
            )
        except _DIAGNOSTIC_EXCEPTIONS as exc:
            canonical_wrapper_identity = None
            canonical_wrapper_resolution_error = _diagnostic_error(exc)

        if path_onex_identity is None:
            path_diagnosis = (
                "PATH did not resolve an 'onex' executable for this process."
            )
        elif path_onex_identity.resolution_error is not None:
            if path_onex_identity.status is PathOnexResolutionStatus.LOOKUP_FAILED:
                path_diagnosis = (
                    "PATH lookup for 'onex' failed before a candidate could be "
                    f"resolved: {path_onex_identity.resolution_error}."
                )
            else:
                assert path_onex_identity.executable is not None
                path_diagnosis = (
                    "PATH resolves 'onex' to "
                    f"{path_onex_identity.executable}, but that entry's "
                    "filesystem identity cannot be compared: "
                    f"{path_onex_identity.resolution_error}."
                )
        elif canonical_wrapper_identity is None:
            if canonical_wrapper_path is None:
                canonical_detail = "no OMNI_HOME was provided"
            else:
                canonical_detail = (
                    "canonical wrapper resolution failed: "
                    f"{canonical_wrapper_resolution_error}"
                )
            assert path_onex_identity.executable is not None
            path_diagnosis = (
                "PATH resolves 'onex' to "
                f"{path_onex_identity.executable}, but the canonical wrapper's "
                f"filesystem identity cannot be compared ({canonical_detail})."
            )
        elif path_onex_identity.identity == canonical_wrapper_identity:
            path_diagnosis = (
                "PATH resolves 'onex' through the canonical wrapper: "
                f"{path_onex_identity.executable}."
            )
        else:
            assert path_onex_identity.executable is not None
            path_diagnosis = (
                "PATH resolves 'onex' to an entry that is not the canonical "
                f"wrapper by filesystem identity: {path_onex_identity.executable}. "
                f"Canonical wrapper: {canonical_wrapper}."
            )
        detail = (
            "omnimarket is NOT INSTALLED from git in this interpreter "
            f"({sys.executable}) (absent, or installed from PyPI/a non-VCS "
            "source), but a "
            f"canonical clone exists at $OMNI_HOME/omnimarket (HEAD "
            f"{canonical[:12]}). 'onex skill'/'onex node'/'onex delegate' "
            "dispatch for market-provided nodes (e.g. node_aislop_sweep) "
            f"will fail with 'Unknown node'. {path_diagnosis} If that interpreter "
            f"is not $OMNI_HOME/omnibase_infra/.venv/bin/python, invoke the "
            f"canonical wrapper directly: {canonical_wrapper} (see "
            f"knowledge-base-internal:runbooks/omnibase-infra-onex-cli-invocation.md). Otherwise repair with: "
            f"{install_cmd} --execute (or {repair_cmd} --repair)."
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
        "See knowledge-base:runbooks/node-skill-package-install.md."
    )
