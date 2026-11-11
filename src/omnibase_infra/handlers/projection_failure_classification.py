# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Is a projection failure the EVENT's defect rather than the write path's?

OMN-17379 / OMN-23024. The projection dispatch callback treats the two classes
oppositely: an event defect is dead-lettered and the offset advances, a write-path
defect withholds the offset so the record is redelivered once the runtime is
repaired. This is a closed allowlist, not a denylist of "infrastructure" errors:
a failure that cannot be positively identified as the event's own defect is the
runtime's, the direction that stalls loudly rather than discards a fact silently.

Event defects:

* the inbound payload cannot be coerced into what the handler declares it needs
  (a pydantic or envelope validation failure);
* the event's own tenant identity resolves to nothing: omnimarket's
  ``TenantRegistryResolutionError``. Redelivering identical bytes against an
  unchanged registry reproduces the identical refusal, so the standalone runner
  classified it POISON (OMN-17985). Folding the delegation writer into the
  runtime (omnibase_infra#5743) dropped that classification and one
  unknown-tenant event (produced 2026-11-07T17:45:07Z) was redelivered for hours.

This package does not depend on omnimarket, so the tenant refusal cannot be
imported for an ``isinstance``. It is matched by defining module and class name
along the MRO; the class name alone would accept any look-alike.
"""

from __future__ import annotations

from pydantic import ValidationError as PydanticValidationError

from omnibase_infra.errors import EnvelopeValidationError

#: ``(defining module, class name)`` of each event-identity refusal. A type joins
#: this set only when omnimarket's own ``projection/error_classification.py``
#: classifies it POISON for the reason stated there.
POISON_EVENT_IDENTITY_REFUSALS: frozenset[tuple[str, str]] = frozenset(
    {
        (
            "omnimarket.projection.tenant_registry_resolution",
            "TenantRegistryResolutionError",
        ),
    }
)


def is_event_identity_refusal(exc: BaseException) -> bool:
    """Is ``exc`` (or a base class of it) a declared event-identity refusal?"""
    return any(
        (cls.__module__, cls.__qualname__) in POISON_EVENT_IDENTITY_REFUSALS
        for cls in type(exc).__mro__
    )


def is_content_failure(exc: BaseException) -> bool:
    """True when the failure is the event's defect, so the record is dead-lettered."""
    return isinstance(
        exc, PydanticValidationError | EnvelopeValidationError
    ) or is_event_identity_refusal(exc)
