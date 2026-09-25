# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Injected violations proving check_no_credential_in_log is non-vacuous (OMN-17423).

NOT PRODUCTION CODE. Every logger call in :func:`violations` is deliberately
wrong. The gate's self-test mode scans this file and asserts an EXACT count, so
a refactor that silently stops detecting one shape turns the count red rather
than passing quietly. Nothing here is imported or executed, and the file is
excluded from the normal scan by its ``fixtures`` path component.

Each violating call is written to yield EXACTLY ONE finding, so the count maps
one-to-one onto the shapes: a call that tripped two rules would hide the loss
of one of them behind the other.

EXPECTED VIOLATIONS: 9. The negative cases in :func:`negatives` carry the same
vocabulary and must contribute ZERO -- they are what stops this gate from
degenerating into a keyword grep.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class _Session:
    session_token: str = ""


def violations(value: str, access_token: str, session: _Session) -> None:
    """Nine violations, one per shape the gate claims to detect."""

    # (1) literal format string, compound fragment, assigned
    logger.info("api_key=%s", value)

    # (2) literal format string, colon form
    logger.warning("minted access_token: %s", value)

    # (3) f-string interpolating a credential-named variable
    logger.error(f"exchange failed for {access_token}")

    # (4) f-string whose literal half names the field
    logger.debug(f"client_secret={value!r}")

    # (5) positional argument whose name is a credential
    logger.info("provisioned for tenant %s", access_token)

    # (6) positional attribute reference
    logger.info("resumed %s", session.session_token)

    # (7) dict literal argument with a credential key
    logger.info("created %s", {"plaintext_key": "onxk_notreal"})

    # (8) extra= dict keyed on the gateway token this ticket's probe traced
    logger.info("attached", extra={"gateway_token": "eyJ.a.b"})

    # (9) extra= dict keyed on a prose-ambiguous bare word. Unambiguous as a
    # FIELD name, which is the only place bare words are consulted.
    logger.info("resumed", extra={"token": value})


def negatives(count: int, rotation_ref: str, token_savings_pct: float) -> None:
    """Same vocabulary, zero violations. The gate's precision tests."""

    # An environment variable NAME in a status message leaks nothing.
    logger.warning("LINEAR_API_KEY is not set - cannot call Linear API.")
    logger.info("Infisical needs INFISICAL_CLIENT_ID and INFISICAL_CLIENT_SECRET")

    # Reference and metric fields carry forensic value and no secret.
    logger.info("issued keys", extra={"api_key_count": count})
    logger.info("rotation done %s", {"api_key_id": rotation_ref})
    logger.info("savings %s", token_savings_pct)
    logger.info("hashed %s", {"key_hash": "sha256:abc"})

    # A bare word in prose is prose. Bare words are never consulted in a
    # format string -- see _format_string_hit.
    logger.warning("refresh token expired, re-minting")

    # Three LIVE lines from onex-api on dev, each of which an earlier draft of
    # this gate rejected. The interpolated value is a reason or a tenant id,
    # never the credential the sentence names.
    logger.warning("OIDC for api-keys router disabled: %s", rotation_ref)
    logger.warning("Invalid tenant_id format in token: %s", rotation_ref)
    logger.warning("Invalid tenant_id format in OIDC token: %s", rotation_ref)

    # A reviewed false positive, suppressed with a stated reason.
    logger.info(
        "rotated secret: %s", "logical-name"
    )  # credential-log-allow: logs the logical name, not the value


def not_a_logger(api_key: str) -> dict[str, Any]:
    """Not a logger call -- the gate must not flag ordinary code."""
    return {"api_key": api_key}
