# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shared strict field definitions for the inert RSD delegation overlay."""

from __future__ import annotations

import base64
import binascii
import re
from typing import Annotated
from uuid import UUID

from pydantic import UUID4, BeforeValidator

_CAPABILITY_REF = r"^capability://[a-z][a-z0-9-]*(?:/[a-z][a-z0-9-]*)*$"
_SHA256 = r"^[0-9a-f]{64}$"
_RFC3339_UTC = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
_BASE64URL = r"^[A-Za-z0-9_-]+={0,2}$"
_UUID4 = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"


def _require_canonical_capability_ref(value: object) -> str:
    if type(value) is not str or re.fullmatch(_CAPABILITY_REF, value) is None:
        raise ValueError("capability reference must be a canonical segment-safe ID")
    return value


def _require_canonical_uuid4(value: object) -> object:
    if type(value) is not str or re.fullmatch(_UUID4, value) is None:
        raise ValueError("key ID must be a lowercase hyphenated UUID4")
    return UUID(value)


def _require_canonical_sha256(value: object) -> str:
    if type(value) is not str or re.fullmatch(_SHA256, value) is None:
        raise ValueError("digest must be a lowercase SHA-256 hex value")
    return value


def _require_canonical_ed25519_signature(value: object) -> str:
    if type(value) is not str or re.fullmatch(_BASE64URL, value) is None:
        raise ValueError("signature must be canonical URL-safe base64")
    try:
        decoded = base64.urlsafe_b64decode(value.encode("ascii"))
    except (UnicodeEncodeError, ValueError, binascii.Error) as error:
        raise ValueError("signature must be canonical URL-safe base64") from error
    if len(decoded) != 64 or base64.urlsafe_b64encode(decoded).decode("ascii") != value:
        raise ValueError("signature must be a canonical 64-byte Ed25519 value")
    return value


CanonicalCapabilityRef = Annotated[
    str, BeforeValidator(_require_canonical_capability_ref)
]
CanonicalUuid4 = Annotated[UUID4, BeforeValidator(_require_canonical_uuid4)]
CanonicalSha256 = Annotated[str, BeforeValidator(_require_canonical_sha256)]
CanonicalEd25519Signature = Annotated[
    str, BeforeValidator(_require_canonical_ed25519_signature)
]

__all__ = [
    "CanonicalCapabilityRef",
    "CanonicalEd25519Signature",
    "CanonicalSha256",
    "CanonicalUuid4",
    "_BASE64URL",
    "_CAPABILITY_REF",
    "_RFC3339_UTC",
    "_SHA256",
]
