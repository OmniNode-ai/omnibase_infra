# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Credential redaction for the shared runtime logging path (OMN-17423).

WHY THIS EXISTS. OMN-17423 asks whether a customer API key or a session token
can reach a service log. The at-rest half (OMN-17352) can be perfectly opaque
while the same secret sits in cleartext in a pod log that a far wider set of
identities can read, and under the BYOK rulings the credentials in flight are
the customer's own provider keys -- a leak there is a customer-data incident,
not an internal one.

WHY IT LIVES HERE AND NOT IN A SERVICE. The first implementation shipped as
``docker/onex-api/log_redaction.py`` in ``omninode_infra``. The operator ruled
on 2026-09-15 (OMN-17423 comment ``5b191b1b``) that a service-local filter does
not close AC2: one of the four services in the credential path being covered is
not evidence about the other three, and the 2026-09-15 probe showed the effects
pod introspecting a gateway token. This module is that filter re-homed into the
shared path. ``service_kernel.configure_logging()`` installs it, and every
runtime process -- ``omninode-runtime``, ``-effects``, ``-worker`` -- boots
through that one function via the ``onex-runtime`` entrypoint, so coverage is
by construction rather than by four separate call sites staying in sync.

ONE VOCABULARY, NOT TWO. Field-name decisions defer to
:func:`omnibase_infra.utils.util_dlq_credential_redaction.is_credential_field_name`,
the same authority the dead-letter path uses (OMN-18385). That is deliberate:
two independently maintained denylists drift, and the drift is invisible until
something leaks. It also inherits that module's reference exemptions, so
``api_key_count``, ``token_count``, ``secret_ref`` and ``api_key_id`` stay
readable -- redacting those destroys forensic value while protecting nothing.

THREE PASSES, AND WHY EACH IS NEEDED.

1. Structural -- walk ``LogRecord.args`` and replace values under
   credential-shaped keys before the record is formatted. Covers
   ``logger.info("event %s", {"api_key": v})``.
2. Pattern -- scan the rendered message for credential-SHAPED values. Covers a
   secret that arrived already serialised (an f-string, ``json.dumps``), where
   no key survives for pass 1 to match.
3. Exception -- pre-format and redact ``exc_info``. A Python formatter appends
   exception text AFTER the message, outside both passes above, so a token
   inside a traceback escapes 1 and 2 entirely.

WHAT IT DOES NOT DO. A credential in a field named ``note``, or concatenated
into prose, is reachable by neither a name rule nor a shape rule. That is the
honest limit: this is defense in depth behind not logging the value at all,
which is what the ``check_no_credential_in_log`` CI gate enforces at the source.
"""

from __future__ import annotations

import logging
import re
import traceback
from collections.abc import Iterable, Mapping
from typing import Final

from omnibase_infra.utils.util_dlq_credential_redaction import (
    is_credential_field_name,
)

__all__ = [
    "CREDENTIAL_VALUE_PATTERNS",
    "LOG_REDACTION_MARKER",
    "CredentialRedactionFilter",
    "install_credential_redaction_filter",
]

# Distinct from the dead-letter marker ("<redacted:credential>"): a reader of a
# pod log should be able to tell which layer removed the value.
LOG_REDACTION_MARKER: Final[str] = "[REDACTED]"

# Bounded walk. A log record can carry attacker-influenced structure; an
# unbounded recursion over a deeply nested payload is a denial-of-service
# shape. Below the limit the subtree is replaced wholesale -- fail closed.
_MAX_DEPTH: Final[int] = 10

# Value SHAPES, applied to already-rendered text where no field name survives.
#   - ``onxk_``: the dashboard API key prefix minted by onex-api.
#   - ``eyJ``: a JWT header's base64 opening, i.e. every Keycloak access token,
#     gateway token and session token on this platform.
CREDENTIAL_VALUE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"onxk_[A-Za-z0-9_-]{20,}"),
    re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
)


def redact_credential_patterns(text: str) -> str:
    """Replace every credential-shaped value in ``text`` with the marker."""
    for pattern in CREDENTIAL_VALUE_PATTERNS:
        text = pattern.sub(LOG_REDACTION_MARKER, text)
    return text


def _redact_value(value: object, depth: int = 0) -> object:
    """Redact credential-keyed entries in ``value``, recursing into containers.

    Typed ``object`` rather than ``Any``, matching
    :mod:`util_dlq_credential_redaction`: a log record's args are arbitrary, but
    ``Any`` would silence the isinstance narrowing below that decides what is
    walked and what passes through.
    """
    if depth >= _MAX_DEPTH:
        return LOG_REDACTION_MARKER
    if isinstance(value, dict):
        return {
            key: (
                LOG_REDACTION_MARKER
                if isinstance(key, str) and is_credential_field_name(key)
                else _redact_value(item, depth + 1)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item, depth + 1) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item, depth + 1) for item in value)
    if isinstance(value, str):
        return redact_credential_patterns(value)
    return value


def _redact_mapping(mapping: Mapping[str, object]) -> dict[str, object]:
    """Redact a log record's mapping args, keyed by credential field name."""
    return {
        key: (
            LOG_REDACTION_MARKER
            if is_credential_field_name(key)
            else _redact_value(value)
        )
        for key, value in mapping.items()
    }


class CredentialRedactionFilter(logging.Filter):
    """Strip credential values from log records before they reach a handler.

    Attached to HANDLERS rather than to loggers. A filter on a logger only sees
    records logged directly to that logger -- records that propagate up from a
    named child bypass it entirely, which is precisely the shape that leaks.

    The filter never drops a record: it always returns ``True``. A missing log
    line is its own incident, and a scrubbed line still carries the correlation
    id an investigation needs.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Pass 1 -- structural, before getMessage() renders anything.
        # ``LogRecord.args`` is a tuple or a Mapping by construction: a lone
        # mapping argument is unwrapped by LogRecord itself, everything else
        # stays a tuple. There is no third shape to handle.
        if isinstance(record.args, Mapping):
            record.args = _redact_mapping(record.args)
        elif isinstance(record.args, tuple):
            record.args = tuple(_redact_value(arg) for arg in record.args)

        # Pass 2 -- shape, over the rendered message. Rendering can raise on a
        # malformed format string; a redaction filter must never be the reason
        # a service fails to log, so a failure here leaves the record as-is
        # rather than propagating.
        try:
            rendered = record.getMessage()
            scrubbed = redact_credential_patterns(rendered)
            if scrubbed != rendered:
                record.msg = scrubbed
                record.args = None
        except Exception:  # noqa: BLE001 - a redaction filter must never be
            # the reason a service stops logging; leave the record as-is.
            pass

        # Pass 3 -- exception text. Caching the redacted text in ``exc_text``
        # and clearing ``exc_info`` is what makes the formatter use our scrubbed
        # copy instead of re-deriving the raw traceback from the live tuple.
        if record.exc_info:
            try:
                raw = "".join(traceback.format_exception(*record.exc_info))
                record.exc_text = redact_credential_patterns(raw)
                record.exc_info = None
            except Exception:  # noqa: BLE001 - see above; a __repr__ raising
                # inside format_exception must not break the log path.
                pass
        elif record.exc_text:
            record.exc_text = redact_credential_patterns(record.exc_text)

        return True


def _loggers_to_cover(extra_logger_names: Iterable[str]) -> list[logging.Logger]:
    """Root, plus any logger whose records cannot reach root's handlers.

    A logger with ``propagate = False`` and handlers of its own is an island:
    its records never reach the root handler the filter is attached to. uvicorn
    configures exactly that shape (``uvicorn``, ``uvicorn.error``,
    ``uvicorn.access``), which is why onex-api needed named coverage. Rather
    than hardcode one framework's names, discover the shape.
    """
    loggers: list[logging.Logger] = [logging.getLogger()]
    seen: set[str] = set()

    for name in extra_logger_names:
        if name not in seen:
            seen.add(name)
            loggers.append(logging.getLogger(name))

    manager_dict = dict(logging.getLogger().manager.loggerDict)
    for name, candidate in manager_dict.items():
        if name in seen or not isinstance(candidate, logging.Logger):
            continue
        if candidate.handlers and not candidate.propagate:
            seen.add(name)
            loggers.append(candidate)

    return loggers


def install_credential_redaction_filter(
    *, extra_logger_names: Iterable[str] = ()
) -> int:
    """Attach :class:`CredentialRedactionFilter` to every reachable handler.

    Idempotent -- a handler that already carries the filter is left alone, so
    a second call (a re-entrant bootstrap, a test) adds nothing.

    Call AFTER handlers are configured: ``basicConfig`` in the runtime
    bootstrap, or the FastAPI lifespan startup block once uvicorn has installed
    its own. Handlers added later are NOT retroactively covered; that is the
    documented limit, and the runtime configures logging exactly once.

    Returns the number of handlers newly filtered, so a caller can log a
    non-vacuous readback ("installed on N handlers") rather than assert
    coverage it never measured.
    """
    installed = 0
    for logger in _loggers_to_cover(extra_logger_names):
        for handler in logger.handlers:
            if any(
                isinstance(existing, CredentialRedactionFilter)
                for existing in handler.filters
            ):
                continue
            handler.addFilter(CredentialRedactionFilter())
            installed += 1
    return installed
