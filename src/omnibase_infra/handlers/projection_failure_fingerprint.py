# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Identify a projection failure so "the same refusal again" is decidable.

OMN-17379 bounds the offset withhold: one record refused the SAME way more than
``projection_withhold_max_redeliveries`` times is dead-lettered and the partition
advances. "The same way" is this fingerprint.

The count must reset when the failure CHANGES, because a write path being
repaired in stages reports a different error as it goes and such a record is
progressing toward a row rather than stuck. Fingerprinting on the type alone
would miss that -- every refusal on this path is a
``ProjectionNotMaterializedError`` -- so the underlying cause's type and the
rendered message are folded in. The message is hashed rather than stored: it
carries the payload's own identifiers on some paths, and a cache of error strings
keyed by offset is not a thing the event bus should hold.

Only the STABLE parts of the message are folded in (OMN-23024). A refusal that
renders the event's age (``the event is 4035s old``) is a different string on
every redelivery of one record, so a hash of the raw message reset the count each
time and the bound never fired for it. The age figure is masked; a different
tenant, table or cause is not.
"""

from __future__ import annotations

import hashlib
import re

#: The event's own age as a refusal message renders it. It grows by the
#: redelivery interval on every delivery of one record, so it is the one part of
#: the message that makes "the same refusal again" look like a new refusal.
_EVENT_AGE_IN_MESSAGE = re.compile(r"\b\d+(?:\.\d+)?s old\b")


def projection_failure_fingerprint(exc: BaseException) -> str:
    """A 32-hex digest of the failure's type, cause type and stable message."""
    cause = exc.__cause__
    stable_message = _EVENT_AGE_IN_MESSAGE.sub("<age>s old", str(exc))
    parts = (
        f"{type(exc).__name__}|{type(cause).__name__ if cause else ''}|{stable_message}"
    )
    return hashlib.sha256(parts.encode("utf-8", "replace")).hexdigest()[:32]
