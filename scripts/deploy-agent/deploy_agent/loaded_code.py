# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The identity of the code THIS process loaded (OMN-18200).

``self_update`` used to decide whether to re-exec by comparing the deploy
clone's ``HEAD`` to ``origin/<tracking ref>``. Neither side of that comparison
names the code the running process actually imported, and on the lab host they
are routinely different things: an external reconciler resets the same clone to
``origin/dev`` every hour at :19 past. Past each tick the clone is already
current, so the method logged ``already at origin/dev, nothing to do`` and
skipped the re-exec -- literally true, operationally wrong. Measured
2026-09-14T08:00Z: clone at ``ead1f59b1`` (carrying the #3520 fix on disk)
while the process had been running since 2026-09-09 14:06 EDT, so every
automated api build on the lab kept failing ``images_pinned`` against a fix
that was already merged, already pulled, and simply not loaded.

This module holds the missing half. The sha of the tree the process imported is
recorded ONCE, at startup, before anything can move the clone underneath it,
and ``self_update`` compares THAT against the clone.

It is recorded rather than derived on demand for the obvious reason: deriving
it at the moment of comparison would read the clone again and reproduce the
defect exactly.

Rule 8 applies to the read. ``loaded_code_sha()`` raises when nothing has been
recorded rather than returning a placeholder or silently falling back to the
clone -- a fallback here would restore the old comparison under a new name and
would read as healthy while doing so. Both ``self_update`` call sites already
wrap the call and journal ``friction_type=self_update_boundary_failed``, so a
refusal is visible and the agent stays on its current image rather than
guessing.
"""

from __future__ import annotations

import subprocess

_LOADED_CODE_SHA: str | None = None

#: Bounded because this runs on the startup path; a hung git here would hold
#: the health port unbound.
_RESOLVE_TIMEOUT_SECONDS = 30


class LoadedCodeShaUnavailableError(RuntimeError):
    """The loaded-code sha could not be resolved from the clone on disk."""


class LoadedCodeShaNotRecordedError(RuntimeError):
    """A loaded-code sha was read before any was recorded."""


def resolve_clone_sha(agent_dir: str) -> str:
    """Return the ``HEAD`` sha of the clone at ``agent_dir``.

    ``agent_dir`` is required and has no default: the directory whose code this
    process loaded is a property of the deployment, and a guessed one would
    record the identity of a tree nobody is running.
    """
    try:
        result = subprocess.run(
            ["git", "-C", agent_dir, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=_RESOLVE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LoadedCodeShaUnavailableError(
            f"LOADED_CODE_SHA_UNAVAILABLE: could not run git in {agent_dir}: {exc}"
        ) from exc
    if result.returncode != 0:
        raise LoadedCodeShaUnavailableError(
            f"LOADED_CODE_SHA_UNAVAILABLE: git rev-parse HEAD in {agent_dir} exited "
            f"{result.returncode}: {result.stderr.strip()[:200]}"
        )
    sha = result.stdout.strip()
    if not sha:
        raise LoadedCodeShaUnavailableError(
            f"LOADED_CODE_SHA_UNAVAILABLE: git rev-parse HEAD in {agent_dir} "
            "returned nothing"
        )
    return sha


def record_loaded_code_sha(agent_dir: str) -> str:
    """Record, and return, the sha of the tree this process loaded.

    Called once from ``DeployAgent.run`` before the consumer exists. A re-exec
    replaces the process image, so the replacement records its own sha at its
    own startup and the two never share a value by accident.
    """
    global _LOADED_CODE_SHA  # noqa: PLW0603 - one per process, by construction
    _LOADED_CODE_SHA = resolve_clone_sha(agent_dir)
    return _LOADED_CODE_SHA


def loaded_code_sha() -> str:
    """Return the recorded loaded-code sha, or raise.

    Raises:
        LoadedCodeShaNotRecordedError: nothing was recorded. This is a process
            that never reached its own startup step, not a process whose code
            happens to match the clone -- the two are different facts and a
            default would report them as the same one.
    """
    if _LOADED_CODE_SHA is None:
        raise LoadedCodeShaNotRecordedError(
            "LOADED_CODE_SHA_NOT_RECORDED: no loaded-code identity was recorded "
            "at startup, so self-update cannot tell whether this process is "
            "running the clone's code. Refusing rather than comparing the clone "
            "to the remote, which is the OMN-18200 defect."
        )
    return _LOADED_CODE_SHA


def loaded_code_sha_if_recorded() -> str | None:
    """Return the recorded sha, or ``None``. For reporting surfaces only."""
    return _LOADED_CODE_SHA


def reset_loaded_code_sha() -> None:
    """Clear the recorded identity. Test seam."""
    global _LOADED_CODE_SHA  # noqa: PLW0603
    _LOADED_CODE_SHA = None
