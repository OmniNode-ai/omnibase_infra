# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev-lane Infisical service is passed the SMTP variables its image reads.

WHY THIS EXISTS (OMN-20999). The container was passed no SMTP variable at all, so
the instance reported ``emailConfigured: false`` while ``inviteOnlySignup: true``.
Those two together mean no account can self-register and no account can be
invited, which is why the instance held exactly one bootstrap user and every
operator reached the store through a shared machine identity.

Two things these tests pin that a reviewer cannot see by reading the diff.

**The spelling is the image's, not the Keycloak reconciler's.** The running
v0.146 image reads ``SMTP_USERNAME`` and ``SMTP_FROM_ADDRESS``
(``grep -rhoE "SMTP_[A-Z_]+" /backend/dist`` inside the container). The Keycloak
reconciler uses ``SMTP_USER`` and ``SMTP_FROM`` for the *same* SES sender. Copying
those spellings across yields an instance that starts clean, reports no error and
silently never sends mail -- the exact failure this ticket exists to end.

**Every variable must stay override-backed.** A hardcoded value here would turn
mail on for every machine running ``onex up core``, not just the lane that wants
it. ``test_every_smtp_line_is_override_backed`` fails on a single literal.

The compose file carries ``!!merge`` tags and anchors that ``yaml.safe_load``
refuses, so the service block is read as text rather than parsed. That is
deliberate: the assertion is about the bytes the compose engine substitutes into,
and a parser that normalises ``${VAR:-}`` away would hide the thing being checked.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPOSE = _REPO_ROOT / "docker" / "docker-compose.infra.yml"

# The names the v0.146 image reads, verified against the running container.
_REQUIRED = (
    "SMTP_HOST",
    "SMTP_PORT",
    "SMTP_USERNAME",
    "SMTP_PASSWORD",
    "SMTP_FROM_ADDRESS",
    "SMTP_FROM_NAME",
    "SMTP_REQUIRE_TLS",
)

# The spellings the Keycloak reconciler uses for the same sender. Present here,
# they would be ignored by Infisical.
_WRONG_SPELLINGS = ("SMTP_USER:", "SMTP_FROM:")

_SMTP_LINE = re.compile(r"^\s+(SMTP_[A-Z_]+):\s*(.+?)\s*$")


def _infisical_environment() -> list[str]:
    """The lines of the infisical service's ``environment:`` block."""
    text = _COMPOSE.read_text(encoding="utf-8")
    start = text.find("\n  infisical:\n")
    if start == -1:
        pytest.fail("docker-compose.infra.yml declares no `infisical:` service")
    # the next service at the same indent ends the block
    nxt = re.search(r"\n  [a-z0-9][a-z0-9_-]*:\n", text[start + 1 :])
    block = text[start : start + 1 + (nxt.start() if nxt else len(text))]
    env = block.find("\n    environment:\n")
    if env == -1:
        pytest.fail("the infisical service declares no `environment:` block")
    after = block[env + 1 :]
    stop = re.search(r"\n    [a-z][a-z_]*:", after[len("    environment:\n") :])
    end = len("    environment:\n") + (stop.start() if stop else len(after))
    return after[:end].splitlines()


def test_image_read_smtp_variables_are_all_passed() -> None:
    """Every SMTP name the v0.146 image reads reaches the container."""
    names = {
        m.group(1) for line in _infisical_environment() if (m := _SMTP_LINE.match(line))
    }
    missing = [name for name in _REQUIRED if name not in names]
    assert not missing, (
        f"the infisical service does not pass {missing}; without them the instance "
        "reports emailConfigured false and invite-only signup can never issue an "
        "invite (OMN-20999)"
    )


def test_every_smtp_line_is_override_backed() -> None:
    """No SMTP value is hardcoded, so the change is inert where nothing is set."""
    hardcoded = [
        line.strip()
        for line in _infisical_environment()
        if (m := _SMTP_LINE.match(line)) and "${INFISICAL_SMTP_" not in m.group(2)
    ]
    assert not hardcoded, (
        "every SMTP value must resolve from an INFISICAL_SMTP_* override so a lane "
        f"that sets nothing is unaffected; these are hardcoded: {hardcoded}"
    )


def test_the_keycloak_reconciler_spellings_are_not_used() -> None:
    """`SMTP_USER` / `SMTP_FROM` would be silently ignored by this image."""
    block = "\n".join(_infisical_environment())
    present = [name for name in _WRONG_SPELLINGS if name in block]
    assert not present, (
        f"{present} are the Keycloak reconciler's names for the same SES sender; "
        "the v0.146 Infisical image reads SMTP_USERNAME and SMTP_FROM_ADDRESS, so "
        "these would be accepted, ignored, and mail would silently never send"
    )
