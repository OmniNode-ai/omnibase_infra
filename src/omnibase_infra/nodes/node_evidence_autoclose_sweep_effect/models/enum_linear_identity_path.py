# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Which credential a sweep run reached Linear with (OMN-17664)."""

from __future__ import annotations

from enum import StrEnum


class EnumLinearIdentityPath(StrEnum):
    """Whose name a write this run makes will carry on the ticket.

    This is not a configuration detail. It is the only thing that decides
    whether a flip this closer writes is distinguishable, on the ticket's own
    history, from a flip a person made. Measured 2026-09-05T20:33Z against
    OMN-17957: every history entry including the sweep's own 19:36:02.430Z flip
    carried ``actorId 7a850ce1-f95e-431f-b4e3-62f7449f04c0``, because
    ``LINEAR_API_KEY`` is a PERSONAL key and Linear attributes its writes to the
    person who minted it. A whole fence (``_prior_revert_reason``'s
    ``actorId``-null branch) was structurally dead for that reason.

    So the path is recorded and named in the run log, rather than inferred from
    which secrets happen to be set: a run that fell back to a person's key and a
    run that wrote as the application look identical afterwards unless the run
    itself said which it was.
    """

    #: Both application secrets were present and were exchanged for an app
    #: actor token. Writes attribute to the OAuth application.
    OAUTH_APPLICATION = "oauth_application"

    #: The documented fallback — no application secrets, a personal key present.
    #: Every write this run makes carries the name of the person who minted that
    #: key. Taken loudly, never silently.
    PERSONAL_API_KEY = "personal_api_key"

    #: Exactly one of the two application secrets was set. This is a REFUSAL,
    #: not a path: the run makes no Linear call at all. It is a distinct value
    #: rather than "unresolved" because the operator who set one of the two
    #: believes the application path is live, and a run that quietly wrote as a
    #: person under that belief is worse than a run that did nothing.
    MISCONFIGURED = "misconfigured"


__all__ = ["EnumLinearIdentityPath"]
