# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Field-name credential redaction for dead-letter envelopes (OMN-18385).

WHY THIS EXISTS. ``sanitize_error_message`` already scrubs the *error string*
a dead-letter envelope carries. Nothing scrubbed the envelope's
``original_message.value`` -- the raw inbound record body, copied verbatim onto
a durable topic. On ``onex.dlq.omnibase-infra.commands.v1`` (onex-dev,
partition 0, read 2026-09-14) offsets 137 and 138 were dead-lettered
``gateway-attach-request`` records whose ``payload.access_token`` sat in
cleartext, 1318 characters each, two distinct customer bearer tokens. Every
attach on that cluster had been dead-lettering since 2026-09-11 (OMN-16504),
so every customer attach in that window put its bearer token on a durable
topic for the topic's full retention.

Typing the field as ``SecretStr`` (OMN-18385 AC1) closes the model-shaped
paths by construction. It does NOT close this one: the raw path never
constructs the model -- it copies the bytes that failed to become one. This
module is the second layer, and it is deliberately NAME-based rather than
type-based so a field added later as a plain ``str`` still cannot leak.

WHAT IT DOES NOT DO. It redacts by field NAME. A credential placed in a field
named ``note``, or concatenated into a free-text string, is not reachable by
any name-based rule and is not covered here. That is the honest limit of
defense in depth; AC1's typing is what covers the fields we know about.

REDACTION IS RECORDED, NOT SILENT. Every call reports the dotted paths of the
fields it replaced so the envelope can carry ``redacted_fields`` -- names
only, never values. A reader of the dead-letter topic can then tell "this
record never had a token" apart from "this record's token was removed", and
the replay engine can refuse to republish a body it knows is incomplete
(``engine_dlq_replay``), instead of pushing a command with a placeholder
credential back onto the original topic.
"""

from __future__ import annotations

import json
import re
from typing import Final

__all__ = [
    "CREDENTIAL_FIELD_NAME_FRAGMENTS",
    "DLQ_REDACTION_MARKER",
    "NON_CREDENTIAL_FIELD_NAMES",
    "NON_CREDENTIAL_FIELD_SUFFIXES",
    "is_credential_field_name",
    "redact_credential_fields",
    "redact_credential_fields_in_text",
]

# The value written in place of a credential. Distinct from
# ``DLQ_UNREADABLE_VALUE_MARKER`` ("<unreadable_value>") and from
# "<decode_failed>"/"<non-serializable>": those say the body could not be
# read, this one says the body was read and one field was deliberately
# removed. Conflating them would make a redacted record look corrupt.
DLQ_REDACTION_MARKER: Final[str] = "<redacted:credential>"

# Normalised name fragments that mark a field as credential-bearing. Matched
# as a SUBSTRING of the normalised key (lowercased, ``-``/``_``/``.``/spaces
# stripped), so ``accessToken``, ``access-token``, ``ACCESS_TOKEN`` and
# ``x_access_token`` all match the same fragment.
CREDENTIAL_FIELD_NAME_FRAGMENTS: Final[tuple[str, ...]] = (
    "accesstoken",
    "refreshtoken",
    "idtoken",
    "sessiontoken",
    # OMN-17423: the credential the 2026-09-15 effects-pod probe traced, and
    # the one the OMN-18385 dead-letters carried. It reached the bare-word
    # "token" fragment before, which is enough for a field NAME but not for a
    # gate that only consults compound fragments in a format string.
    "gatewaytoken",
    "bearertoken",
    "authtoken",
    "apitoken",
    "apikey",
    # OMN-17423: the dashboard key-creation path names the one-time cleartext
    # value ``plaintext_key`` -- it matches none of the other fragments, so a
    # dead-lettered key-create command carried it unredacted. Added here rather
    # than in a second denylist so the log filter and this path agree.
    "plaintextkey",
    "accesskey",
    "secretkey",
    "privatekey",
    "clientsecret",
    "clientkey",
    "password",
    "passwd",
    "passphrase",
    "authorization",
    "credential",
    "credentials",
    # Deliberately last and deliberately broad: the bare words. Anything
    # ending in one of the non-credential suffixes below is exempted before
    # these are consulted, which is what keeps ``token_count``,
    # ``secret_ref`` and ``api_key_id`` readable.
    "secret",
    "token",
)

# Whole normalised names that are NEVER credentials even though they contain
# a fragment above. Checked on the FULL normalised key, so an exemption here
# can never swallow ``access_token``.
NON_CREDENTIAL_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    {
        "maxtokens",
        "mintokens",
        "prompttokens",
        "completiontokens",
        "totaltokens",
        "inputtokens",
        "outputtokens",
        "cachedtokens",
        "reasoningtokens",
        "tokens",
        "tokentype",
        "tokenusage",
        "hastoken",
        "secrets",  # a list of secret NAMES a handler requires, not values
        "requiredsecrets",
        "permittedsecretscopes",
        "secretscopes",
        "credentialscaptured",  # a boolean flag
    }
)

# Normalised suffixes that mark a *reference to* a credential rather than the
# credential itself -- the store-reference pattern this platform already uses
# (``client_secret_api_key_ref``, ``secret_ref``, ``api_key_id``,
# ``credentials_output_path``). Redacting these would destroy the forensic
# value of a dead-letter record while protecting nothing.
NON_CREDENTIAL_FIELD_SUFFIXES: Final[tuple[str, ...]] = (
    "ref",
    "refs",
    "reference",
    "id",
    "ids",
    "name",
    "names",
    "path",
    "paths",
    "url",
    "uri",
    "type",
    "kind",
    "count",
    "length",
    "len",
    "scope",
    "scopes",
    "source",
    "expiry",
    "expiresat",
    "createdat",
    "updatedat",
    "at",
    "enabled",
    "disabled",
    "required",
    "present",
    "hash",
    "fingerprint",
    "digest",
    "sha256",
)

_NORMALISE_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")

# Bounded recursion. A dead-letter body is attacker-influenced input; an
# unbounded walk over a deeply nested payload is a denial-of-service shape.
# Below the limit nothing is inspected, so the subtree is dropped wholesale
# rather than passed through unredacted -- fail closed.
_MAX_DEPTH: Final[int] = 32
_DEPTH_EXCEEDED_MARKER: Final[str] = "<redacted:depth_limit>"

# Fallback for a body that is not JSON: scrub ``key=value``/``key: value``
# pairs whose key is credential-shaped. Deliberately conservative -- it is a
# best-effort second chance, not the primary mechanism.
_TEXT_PAIR_RE: Final[re.Pattern[str]] = re.compile(
    r"""(?P<key>[A-Za-z0-9_.\-]{1,64})     # a key-ish token
        (?P<sep>\s*[=:]\s*)                # = or :
        (?P<quote>["']?)                   # optional quote
        (?P<value>[^\s"',;&}\]]{1,4096})   # the value up to a delimiter
        (?P=quote)""",
    re.VERBOSE,
)


def _normalise(name: str) -> str:
    """Lowercase ``name`` and strip every non-alphanumeric character."""
    return _NORMALISE_RE.sub("", name.lower())


def is_credential_field_name(name: str) -> bool:
    """True when a field called ``name`` should have its value redacted.

    Exemptions are evaluated on the FULL normalised name and are consulted
    BEFORE the fragment list, so ``token_count`` and ``secret_ref`` stay
    readable while ``access_token`` and ``client_secret`` do not.
    """
    normalised = _normalise(name)
    if not normalised:
        return False
    if normalised in NON_CREDENTIAL_FIELD_NAMES:
        return False
    for suffix in NON_CREDENTIAL_FIELD_SUFFIXES:
        if normalised.endswith(suffix) and normalised != suffix:
            return False
    return any(fragment in normalised for fragment in CREDENTIAL_FIELD_NAME_FRAGMENTS)


def _redact(
    value: object,
    *,
    path: str,
    depth: int,
    redacted: list[str],
) -> object:
    if depth > _MAX_DEPTH:
        redacted.append(path or "<root>")
        return _DEPTH_EXCEEDED_MARKER
    if isinstance(value, dict):
        out: dict[str, object] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}" if path else key
            if is_credential_field_name(key):
                out[key] = DLQ_REDACTION_MARKER
                redacted.append(child_path)
            else:
                out[key] = _redact(
                    item, path=child_path, depth=depth + 1, redacted=redacted
                )
        return out
    if isinstance(value, (list, tuple)):
        return [
            _redact(
                item,
                path=f"{path}[{index}]",
                depth=depth + 1,
                redacted=redacted,
            )
            for index, item in enumerate(value)
        ]
    return value


def redact_credential_fields(value: object) -> tuple[object, tuple[str, ...]]:
    """Recursively replace credential-named fields in a decoded structure.

    Returns the redacted structure and the dotted paths that were replaced.
    Scalars and unrecognised types pass through untouched -- this function
    only ever acts on a value that is REACHED THROUGH a credential-named key.
    """
    redacted: list[str] = []
    result = _redact(value, path="", depth=0, redacted=redacted)
    return result, tuple(redacted)


def _redact_text_pairs(text: str) -> tuple[str, tuple[str, ...]]:
    redacted: list[str] = []

    def _sub(match: re.Match[str]) -> str:
        key = match.group("key")
        if not is_credential_field_name(key):
            return match.group(0)
        redacted.append(key)
        quote = match.group("quote")
        return f"{key}{match.group('sep')}{quote}{DLQ_REDACTION_MARKER}{quote}"

    return _TEXT_PAIR_RE.sub(_sub, text), tuple(redacted)


def redact_credential_fields_in_text(text: str) -> tuple[str, tuple[str, ...]]:
    """Redact credential-named fields inside a serialised record body.

    The body of a dead-lettered record is a ``str`` by the time the
    dead-letter publisher sees it (the bytes are decoded with
    ``errors="replace"`` before the envelope is built). The common case is
    JSON, and a JSON body is parsed, walked and re-serialised so the
    structure survives and the redaction is exact.

    A body that is not JSON -- or that is JSON but decodes to a bare scalar,
    where there are no field names to match -- falls back to a
    ``key=value``/``key: value`` text scrub. That fallback is best effort: it
    catches form-encoded and log-shaped bodies and misses anything else,
    which is why AC1's typing exists rather than this being the only layer.

    Returns the redacted text and the names/paths redacted. The text is
    returned unchanged (with an empty tuple) when nothing matched, so a
    record with no credential field is byte-identical to what it was before
    this function existed.
    """
    if not text:
        return text, ()
    try:
        decoded = json.loads(text)
    except (ValueError, TypeError):
        return _redact_text_pairs(text)
    if not isinstance(decoded, (dict, list)):
        # A JSON scalar carries no field names; the text fallback is the
        # only thing that could match, and on a bare scalar it will not.
        return _redact_text_pairs(text)
    redacted_value, paths = redact_credential_fields(decoded)
    if not paths:
        return text, ()
    try:
        return json.dumps(redacted_value, default=str), paths
    except (TypeError, ValueError):
        # Re-serialisation failed after a successful parse. The parsed body
        # provably CONTAINED a credential field, so returning the original
        # text would publish the very value we just found. Fail closed.
        return DLQ_REDACTION_MARKER, paths
