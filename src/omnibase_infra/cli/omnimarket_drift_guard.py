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

## Off registry (OMN-17255)

The canonical clone is a REGISTRY-machine convention. A customer has no
``$OMNI_HOME`` and no clone, and until OMN-17255 that case returned
**silently**: exit 0, no drift line, no skip line. Measured 2026-09-18 (row L2
of the local-path ground truth): ``env -u OMNI_HOME onex delegate "say ok"``
exited 0 with nothing at all on stderr. A silent pass is indistinguishable
from a check that never ran, which is the wrong half of fail-open
(``feedback_no_defensive_no_defaults``; omni_home/CLAUDE.md rule 8).

So when no canonical clone resolves, the guard now runs OFF-REGISTRY instead
of returning. It compares the installed omni-internal layer against the pins
the installer PACKAGED INSIDE the installed artifacts -- a distribution's
``Requires-Dist`` metadata is its ``[project].dependencies``, it travels in
the wheel, and reading it needs no clone, no network and no ``OMNI_HOME``.

Resolution order for "expected", in one place so it is not re-derived:

1. the installed ``omnimarket`` distribution's own packaged requirements,
   filtered to the omni-internal layer. omnimarket sits ABOVE infra and its
   requirements are the tightest statement of what it needs underneath it;
2. failing that (omnimarket absent, or declaring no omni-internal
   requirement), the installed ``omnibase_infra`` distribution's own packaged
   requirements. This guard ships inside omnibase_infra, so that anchor is
   present by construction whenever this code runs;
3. failing both, there is nothing to compare -- and that is reported as an
   explicit SKIPPED line naming which fact was missing, never as silence.

Exactly one structured line is written to stderr for EVERY off-registry
verdict, IN_SYNC included::

    drift_guard: mode=off-registry omnimarket=<v> anchor=<d>@<v> pin=<n> \
        expected=<specifier> installed=<v> pins=<n> unsatisfied=<n> \
        verdict=IN_SYNC|DRIFTED|SKIPPED reason=<token>

It is written to ``sys.stderr`` directly rather than through ``logger``
because a logger can be configured to nothing, and a check that logging
config can silence is the silent pass this mode exists to remove.

**Off registry the guard REPORTS and never refuses.** Operator ruling,
2026-09-18 (the OMN-17255 re-scope under the local-path goal, row L2): the
guard must not block dispatch on a machine with no clone. The two positions
that look opposed are not about the same thing -- the goal needs the guard not
to REFUSE off-registry, this ticket needs it not to be SILENTLY ABSENT, and a
comparison against the packaged pins satisfies both. A refusal would also be a
remedy nobody on that machine can apply: there is no clone to reconcile against
and no repair command that is true there, so it would close a customer's only
path with nothing to do about it. Every off-registry verdict therefore exits 0,
and DRIFTED additionally logs the mismatch in prose. ``allow_drift`` is not
consulted here because nothing is refused; it keeps its full meaning on the
canonical-clone path. No ``reconcile`` is attempted either: the bound
reconciler repairs a venv against the clone that does not exist here.

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
from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from omnibase_infra.cli.enum_off_registry_reason import EnumOffRegistryReason
from omnibase_infra.cli.enum_off_registry_verdict import EnumOffRegistryVerdict
from omnibase_infra.cli.model_off_registry_check import ModelOffRegistryCheck
from omnibase_infra.cli.model_omnimarket_lag_stamp import ModelOmnimarketLagStamp
from omnibase_infra.cli.protocol_drift_guard_verdict import (
    ProtocolDriftGuardVerdict,
)
from omnibase_infra.cli.workspace_reconcile import ReconcileFn

__all__ = [
    "CanonicalCloneAttachment",
    "DRIFT_OVERRIDE_ENV",
    "EnumOffRegistryReason",
    "EnumOffRegistryVerdict",
    "ModelOffRegistryCheck",
    "OMNI_INTERNAL_PIN_PREFIXES",
    "OmnimarketDriftError",
    "PACKAGED_PIN_ANCHORS",
    "canonical_clone_attachment",
    "canonical_distribution_name",
    "canonical_local_omnimarket_commit",
    "check_omnimarket_drift",
    "installed_distribution_metadata",
    "installed_omnimarket_commit",
    "resolve_off_registry_check",
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


_DIAGNOSTIC_EXCEPTIONS = (OSError, ValueError)


def _diagnostic_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


class OmnimarketDriftError(RuntimeError):
    """Raised when the installed omnimarket commit diverges from canonical."""


def _path_onex_executable(identity: PathOnexIdentity) -> str:
    """Return the PATH executable without relying on optimisable assertions."""
    if identity.executable is None:
        return "<invalid-path-onex-identity>"
    return identity.executable


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
        # shutil.which() and resolve() cannot be atomic: PATH entries can be
        # replaced between lookup and identity comparison. The result is used
        # only for operator diagnosis and never to authorize execution.
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


# --------------------------------------------------------------------------- #
# Off-registry mode (OMN-17255)
# --------------------------------------------------------------------------- #
# Everything below runs on a machine with no canonical clone. It reads only
# installed-distribution metadata: no git, no network, no OMNI_HOME.

#: The layer this guard can reason about. A distribution outside it (pydantic,
#: click, anthropic) is resolved by the packaging tooling and is not what drifts
#: when an omni artifact is installed piecemeal, so comparing it here would add
#: noise and false refusals without adding evidence.
OMNI_INTERNAL_PIN_PREFIXES: tuple[str, ...] = ("omnibase-", "omninode-", "omnimarket")

#: Resolution order for the packaged pins, most specific first. omnimarket sits
#: ABOVE infra, so its declared requirements are the tightest statement of what
#: it needs beneath it; omnibase_infra's own are the fallback and are present by
#: construction, because this module ships inside that distribution.
PACKAGED_PIN_ANCHORS: tuple[str, ...] = ("omnimarket", "omnibase-infra")


def canonical_distribution_name(name: str) -> str:
    """PEP 503 name for a distribution, so ``omnibase_infra`` and
    ``omnibase-infra`` are one key rather than two."""
    return canonicalize_name(name)


def installed_distribution_metadata(name: str) -> tuple[str, tuple[str, ...]] | None:
    """Return ``(version, Requires-Dist)`` for an installed distribution.

    The single seam through which this module reads the environment: both the
    anchor's packaged pins and every pinned distribution's installed version
    come through here, so a test binds one fake and a reader has one place to
    look. ``None`` means absent or unreadable -- which is never treated as
    satisfied.
    """
    try:
        dist = distribution(name)
    except (PackageNotFoundError, ValueError, OSError):
        return None
    try:
        requires = tuple(dist.metadata.get_all("Requires-Dist") or ())
        return dist.version, requires
    except (ValueError, OSError):
        return None


def _omni_internal_pins(requires: tuple[str, ...]) -> tuple[Requirement, ...]:
    """The omni-internal requirements of a packaged dependency list that APPLY
    to this environment.

    Requirements carrying an ``extra`` marker are excluded: they are optional
    extras, not the base ``[project].dependencies``, and an extra nobody
    installed is not drift. Any other marker is evaluated; one that cannot be
    evaluated is excluded rather than guessed at, and the exclusion is visible
    because the resulting pin COUNT is on the line.
    """
    pins: list[Requirement] = []
    for raw in requires:
        try:
            requirement = Requirement(raw)
        except InvalidRequirement:
            continue
        name = canonical_distribution_name(requirement.name)
        if not name.startswith(OMNI_INTERNAL_PIN_PREFIXES):
            continue
        marker = requirement.marker
        if marker is not None:
            if "extra" in str(marker):
                continue
            try:
                if not marker.evaluate():
                    continue
            except Exception:  # noqa: BLE001 - an unevaluable marker is not a pin
                continue
        pins.append(requirement)
    return tuple(sorted(pins, key=lambda req: canonical_distribution_name(req.name)))


def _pin_is_satisfied(requirement: Requirement, installed: str | None) -> bool:
    """Whether an installed version satisfies a packaged requirement.

    Fails CLOSED: absent, and unparseable, are both "not satisfied". An
    unreadable fact has never been evidence that a fact is fine.
    """
    if installed is None:
        return False
    if not str(requirement.specifier):
        # A requirement with no specifier says "must be present", and it is.
        return True
    try:
        return requirement.specifier.contains(Version(installed), prereleases=True)
    except InvalidVersion:
        return False


def resolve_off_registry_check() -> ModelOffRegistryCheck:
    """Compare the installed omni-internal layer against its packaged pins.

    Pure with respect to the process: reads installed metadata, mutates
    nothing, raises nothing. The caller decides what a verdict means.
    """
    omnimarket_meta = installed_distribution_metadata("omnimarket")
    infra_meta = installed_distribution_metadata("omnibase-infra")
    omnimarket_version = omnimarket_meta[0] if omnimarket_meta else None
    infra_version = infra_meta[0] if infra_meta else None

    anchor_name: str | None = None
    anchor_version: str | None = None
    pins: tuple[Requirement, ...] = ()
    any_anchor_found = False
    for candidate in PACKAGED_PIN_ANCHORS:
        meta = installed_distribution_metadata(candidate)
        if meta is None:
            continue
        any_anchor_found = True
        candidate_pins = _omni_internal_pins(meta[1])
        if candidate_pins:
            anchor_name = canonical_distribution_name(candidate)
            anchor_version = meta[0]
            pins = candidate_pins
            break

    if not pins:
        return ModelOffRegistryCheck(
            verdict=EnumOffRegistryVerdict.SKIPPED,
            reason=(
                EnumOffRegistryReason.NO_APPLICABLE_PACKAGED_PINS
                if any_anchor_found
                else EnumOffRegistryReason.NO_PACKAGED_PIN_ANCHOR
            ),
            omnibase_infra_version=infra_version,
            omnimarket_version=omnimarket_version,
        )

    installed_by_pin: dict[str, str | None] = {}
    unsatisfied: list[str] = []
    for requirement in pins:
        name = canonical_distribution_name(requirement.name)
        meta = installed_distribution_metadata(name)
        installed_by_pin[name] = meta[0] if meta else None
        if not _pin_is_satisfied(requirement, installed_by_pin[name]):
            unsatisfied.append(name)

    if unsatisfied:
        deciding = next(
            req
            for req in pins
            if canonical_distribution_name(req.name) == unsatisfied[0]
        )
        verdict = EnumOffRegistryVerdict.DRIFTED
        reason = EnumOffRegistryReason.PACKAGED_PIN_UNSATISFIED
    else:
        infra_pin = next(
            (
                req
                for req in pins
                if canonical_distribution_name(req.name) == "omnibase-infra"
            ),
            None,
        )
        deciding = infra_pin if infra_pin is not None else pins[0]
        verdict = EnumOffRegistryVerdict.IN_SYNC
        reason = EnumOffRegistryReason.PACKAGED_PINS_SATISFIED

    deciding_name = canonical_distribution_name(deciding.name)
    return ModelOffRegistryCheck(
        verdict=verdict,
        reason=reason,
        omnibase_infra_version=infra_version,
        omnimarket_version=omnimarket_version,
        anchor=anchor_name,
        anchor_version=anchor_version,
        pin_name=deciding_name,
        expected=str(deciding.specifier) or "ANY",
        installed=installed_by_pin[deciding_name],
        pins=len(pins),
        unsatisfied=tuple(unsatisfied),
    )


def _emit_off_registry_line(check: ModelOffRegistryCheck) -> None:
    """Write the verdict line to stderr, unconditionally.

    Deliberately NOT ``logger``: a library logger with no handler, or one a host
    application configured away, drops the line -- which reproduces exactly the
    silence this mode exists to end. ``flush`` because the very next thing on a
    DRIFTED verdict is a raised exception.
    """
    sys.stderr.write(f"{check.line}\n")
    sys.stderr.flush()


def _off_registry_detail(check: ModelOffRegistryCheck) -> str:
    """The human half of a DRIFTED off-registry report.

    Names no canonical clone and no clone-based repair command: on this machine
    neither exists, and a report pointing at a path the reader does not have is
    how a guard teaches people to ignore it.
    """
    installed = check.installed or "ABSENT"
    others = [name for name in check.unsatisfied if name != check.pin_name]
    also = f" Also unsatisfied: {', '.join(others)}." if others else ""
    return (
        f"omnimarket OFF-REGISTRY DRIFT (reported, not blocked): the installed "
        f"omni-internal layer does not satisfy the pins packaged with "
        f"{check.anchor}=={check.anchor_version}. {check.pin_name} is "
        f"{installed}, and {check.anchor}=={check.anchor_version} requires "
        f"{check.pin_name}{check.expected}.{also} No canonical clone was "
        f"resolvable here, so this is judged entirely from installed package "
        f"metadata -- there is no workspace to reconcile against and no repair "
        f"command to hand you that would be true on this machine. Reinstall the "
        f"omni packages as one set so the versions agree ({check.pin_name} first)."
    )


def _run_off_registry_check() -> ModelOffRegistryCheck:
    """The off-registry branch of :func:`check_omnimarket_drift`.

    **Reports; never refuses.** Operator ruling, 2026-09-18 (OMN-17255 re-scope
    under the local-path goal, row L2): off registry the guard must not block
    dispatch. The two positions that look opposed are not about the same thing
    -- the goal needs the guard not to REFUSE on a machine with no clone, and
    this ticket needs the guard not to be SILENTLY ABSENT. A comparison against
    the packaged pins satisfies both: it asserts something, and what it asserts
    is resolvable without a clone.

    Blocking here would also be a remedy nobody on that machine can apply. The
    guard has no clone to reconcile against and no repair command that is true
    there, so a refusal would leave a customer with their only path closed and
    nothing to do about it. That is how a gate gets routed around rather than
    fixed. The line is the deliverable; the exit code never was.

    ``allow_drift`` is not consulted: there is nothing to override, because
    nothing is refused. It keeps its full meaning on the canonical-clone path.
    """
    check = resolve_off_registry_check()
    # Emitted for EVERY verdict. A line that only appears when something is
    # wrong cannot distinguish a clean run from an unrun check, which is the
    # defect this whole mode addresses.
    _emit_off_registry_line(check)

    if check.verdict is EnumOffRegistryVerdict.DRIFTED:
        # Prose detail is best-effort (a library logger can have no handler),
        # which is exactly why it is not the only surface: the structured line
        # above already names the pin, the requirement and the installed
        # version, and it is written to stderr unconditionally.
        logger.warning(
            "%s Results from market-provided nodes come from an UNVERIFIED "
            "omnimarket build and must not be treated as evidence.",
            _off_registry_detail(check),
        )
    return check


def resolve_ancestor_lag(
    *,
    installed: str,
    clone_head: str,
    omni_home: str,
) -> ModelOmnimarketLagStamp | None:
    """Return a stamp when ``installed`` is a strict ANCESTOR of ``clone_head``
    in the canonical clone, else ``None``.

    ``None`` means "not a known ancestor" and nothing more. It is returned for
    a divergent commit, for a commit the clone has never heard of, and for any
    probe that could not be completed -- and the caller treats all three the
    same way, by refusing. That collapse is deliberate: ``git merge-base
    --is-ancestor`` exits non-zero both for "no" and for "no such object", and
    a reading that told them apart in order to be lenient about one would be
    inventing provenance it does not have.

    FAILS CLOSED on every error path. A timeout, a missing git, an unreadable
    clone and a malformed count all return ``None``, because an ancestry
    question that could not be ASKED has not been answered yes. This is the
    single most important property in this module: the whole relaxation rests
    on the claim that the installed bytes are reachable from the head, and a
    probe error is exactly the case where nobody knows whether they are.

    No network I/O. Ancestry is resolved against the objects the canonical
    clone already has; keeping that clone current belongs to the reconcile
    tick (OMN-18815), not to a dispatch.
    """
    clone = Path(omni_home) / "omnimarket"
    if not (clone / ".git").exists():
        return None
    try:
        ancestry = subprocess.run(
            [
                "git",
                "-C",
                str(clone),
                "merge-base",
                "--is-ancestor",
                installed,
                clone_head,
            ],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if ancestry.returncode != 0:
        return None

    try:
        counted = subprocess.run(
            [
                "git",
                "-C",
                str(clone),
                "rev-list",
                "--count",
                f"{installed}..{clone_head}",
            ],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    try:
        behind = int(counted.stdout.strip())
    except ValueError:
        return None
    if behind <= 0:
        # A zero count with a non-equal commit pair is incoherent -- the caller
        # only reaches here when the two differ. Refuse rather than stamp a
        # lag of nothing, which would read as "at the tip".
        return None

    return ModelOmnimarketLagStamp(
        installed_commit=installed,
        clone_head=clone_head,
        commits_behind=behind,
        stamped_at=datetime.now(UTC),
    )


def check_omnimarket_drift(
    omni_home: str | None = None,
    *,
    allow_drift: bool = False,
    reconcile: ReconcileFn | None = None,
) -> ProtocolDriftGuardVerdict | None:
    """Fail fast if the current venv's omnimarket is missing or has drifted
    from its reference point.

    Refusal is the DEFAULT and is never silently skipped. Two ways past it,
    both deliberate:

    * Falls back to the OFF-REGISTRY comparison when the canonical local clone
      cannot be determined (OMN-17255) -- against the pins packaged in the
      installed artifacts, with an explicit verdict line for every outcome and
      no refusal. Before that it returned in silence, which is where the guard
      could not be told apart from a guard that had never run. See the module
      docstring for why that branch reports rather than blocks.
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

    Returns:
        The :class:`ModelOffRegistryCheck` when the off-registry branch ran,
        so a caller can record the verdict in a receipt; ``None`` when a
        canonical clone resolved and the commit comparison was used instead.
        Both are "the check passed"; the return value says WHICH check.

    Args:
        omni_home: Canonical workspace root to resolve the reference clone
            from. ``None`` (no ``$OMNI_HOME``) means no canonical clone, which
            selects the off-registry comparison rather than a silent return.
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
        OmnimarketDriftError: never on the off-registry branch, which reports
            and does not refuse. On registry: a canonical clone IS
            present locally, no
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
        # OMN-17255. No canonical clone: this is a customer machine (or a CI
        # runner, or a fresh one). That branch REPORTS and never refuses, per
        # the 2026-09-18 operator ruling -- see :func:`_run_off_registry_check`.
        # ``reconcile`` is deliberately not passed down: it repairs a venv
        # AGAINST the clone that does not exist here, the same reasoning the
        # detached-HEAD branch below records.
        return _run_off_registry_check()

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
            return None
        else:
            raise OmnimarketDriftError(
                f"{attachment_detail} To dispatch anyway despite the drift "
                f"(results are NOT evidence), set {DRIFT_OVERRIDE_ENV}=1."
            )

    installed = installed_omnimarket_commit()
    if installed == canonical:
        return None

    # OMN-18814. A strict ANCESTOR of the clone head is a staleness fact, not a
    # provenance failure: those bytes are merged, reviewed and reachable from
    # the head, they are simply not the tip. Proceed and STAMP the lag, so the
    # receipt says how far behind the run was instead of the run not happening.
    #
    # Placed after the exact-match return and BEFORE every refusal branch
    # below, and reached only when a canonical clone resolved and is attached
    # -- the off-registry branch (OMN-17255) and the detached-clone branch both
    # return or raise above this point, so neither changes behaviour.
    #
    # `allow_drift` is deliberately NOT consulted here. That flag is the
    # operator's manual, unbounded opt-out and it is untouched by this change;
    # an ancestor lag proceeds on its own merits, so consulting the flag would
    # make an automatic, bounded decision look like it needed the manual one.
    #
    # `reconcile` is deliberately NOT invoked either. It installs packages, and
    # spending that latency mid-dispatch to close a lag that is already safe to
    # proceed on would charge a human for a run that was going to succeed. The
    # reconciler that DOES close the lag is the tick (OMN-18815), off the hot
    # path. The refusal branches below still reconcile exactly as before.
    if installed is not None and omni_home:
        ancestor_lag = resolve_ancestor_lag(
            installed=installed,
            clone_head=canonical,
            omni_home=omni_home,
        )
        if ancestor_lag is not None:
            logger.warning("%s", ancestor_lag.line)
            return ancestor_lag

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
                path_onex_executable = _path_onex_executable(path_onex_identity)
                path_diagnosis = (
                    "PATH resolves 'onex' to "
                    f"{path_onex_executable}, but that entry's "
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
            path_onex_executable = _path_onex_executable(path_onex_identity)
            path_diagnosis = (
                "PATH resolves 'onex' to "
                f"{path_onex_executable}, but the canonical wrapper's "
                f"filesystem identity cannot be compared ({canonical_detail})."
            )
        elif path_onex_identity.identity == canonical_wrapper_identity:
            path_onex_executable = _path_onex_executable(path_onex_identity)
            path_diagnosis = (
                "PATH resolves 'onex' through the canonical wrapper: "
                f"{path_onex_executable}."
            )
        else:
            path_onex_executable = _path_onex_executable(path_onex_identity)
            path_diagnosis = (
                "PATH resolves 'onex' to an entry that is not the canonical "
                f"wrapper by filesystem identity: {path_onex_executable}. "
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
            f"is not the dispatch venv's python "
            f"($OMNI_HOME/.onex-dispatch-venv/bin/python by default; it was "
            f"$OMNI_HOME/omnibase_infra/.venv/bin/python before the OMN-17819 "
            f"gate/dispatch split), invoke the "
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
        return None

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
                f"{detail} Reconcile run {outcome.run_id} was attempted and "
                f"FAILED: {outcome.detail}. That makes this a BROKEN venv, not "
                f"merely a stale one, so fix the reconcile rather than working "
                f"around it -- re-run it directly and read the error:\n"
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
            return None

        # Reaching here means the reconcile outcome claimed a readback-proven
        # success and this guard, making the SAME comparison, disagrees
        # (OMN-18663 made ``ok`` derive from that readback precisely so the two
        # cannot differ). So this is no longer the ordinary "the repair did not
        # land" path -- that one now arrives above as a failed reconcile naming
        # both values. It is a claim about the reconciler itself, which is why
        # it names the run rather than describing drift generically.
        raise OmnimarketDriftError(
            f"{detail} Reconcile run {outcome.run_id} reported a "
            f"readback-PROVEN success and this venv is STILL drifted: installed "
            f"{(installed or 'ABSENT')[:12]} != canonical "
            f"$OMNI_HOME/omnimarket HEAD {canonical[:12]}. That is a "
            f"contradiction between two readings of the same fact, not a stale "
            f"venv, and no retry will resolve it. Reproduce run "
            f"{outcome.run_id} with:\n"
            f"  {outcome.command}\n"
            f"To dispatch anyway despite the drift (results are NOT "
            f"evidence), set {DRIFT_OVERRIDE_ENV}=1."
        )

    raise OmnimarketDriftError(
        f"{detail} To dispatch anyway despite the drift (results are NOT "
        f"evidence), set {DRIFT_OVERRIDE_ENV}=1. "
        "See knowledge-base:runbooks/node-skill-package-install.md."
    )
