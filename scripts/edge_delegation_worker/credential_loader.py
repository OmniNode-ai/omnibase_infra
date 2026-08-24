# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Load the worker's gateway credential from an operator-supplied file path.

Hard rules, non-negotiable:

- The credential is read from a file path flag (``--credential-file``) --
  never inline on the command line, never from an environment variable.
- The file must be mode ``0600`` (owner read/write only). Group- or
  other-readable permissions fail closed: a credential file another local
  user or process can read is not a credential this worker will use.
- The raw content is never logged, never included in an exception message,
  and never returned in any string representation. Only the resolved
  ``auth_mode`` and the file path are safe to log.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraAuthenticationError, ModelInfraErrorContext
from scripts.edge_delegation_worker.models import ModelWorkerCredential

_DISALLOWED_MODE_BITS = stat.S_IRWXG | stat.S_IRWXO


class CredentialFilePermissionError(InfraAuthenticationError):
    """The credential file is readable/writable/executable by group or other."""


class CredentialFileFormatError(InfraAuthenticationError):
    """The credential file content did not resolve to a usable credential."""


def _context(*, operation: str, target_name: str) -> ModelInfraErrorContext:
    return ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.HTTP,
        operation=operation,
        target_name=target_name,
    )


def _check_permissions(path: Path) -> None:
    mode = path.stat().st_mode
    if mode & _DISALLOWED_MODE_BITS:
        raise CredentialFilePermissionError(
            "credential file must be mode 0600 (owner-only); "
            f"{path} is group/other accessible (refusing to read it)",
            context=_context(
                operation="check_credential_file_permissions",
                target_name=str(path),
            ),
        )


def load_worker_credential(path: Path) -> ModelWorkerCredential:
    """Load and validate the worker credential file at *path*.

    Resolution order:

    1. Reject a file that is not exactly owner-readable/writable (0600 or
       tighter). This is checked before the file is opened for content.
    2. Attempt to parse the content as a JSON object. If it parses to a
       ``dict``, it is validated as the ``client_credentials`` shape
       (``client_id`` / ``client_secret`` / ``token_endpoint`` [/ ``scope``]).
    3. Otherwise, the entire stripped file content is treated as one opaque
       pre-issued bearer token (the ``bearer_token`` mode).

    Every failure path raises a typed, non-secret-leaking error -- there is
    no ambiguous case that falls through to a usable credential.
    """
    if not path.is_file():
        raise CredentialFileFormatError(
            f"credential file does not exist or is not a regular file: {path}",
            context=_context(operation="load_worker_credential", target_name=str(path)),
        )

    _check_permissions(path)

    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CredentialFileFormatError(
            f"credential file could not be read as UTF-8 text: {path}",
            context=_context(operation="load_worker_credential", target_name=str(path)),
        ) from exc

    stripped = raw.strip()
    if not stripped:
        raise CredentialFileFormatError(
            f"credential file is empty: {path}",
            context=_context(operation="load_worker_credential", target_name=str(path)),
        )

    parsed_json: object | None = None
    try:
        parsed_json = json.loads(stripped)
    except json.JSONDecodeError:
        parsed_json = None

    if isinstance(parsed_json, dict):
        try:
            return ModelWorkerCredential(
                auth_mode="client_credentials",
                client_id=parsed_json.get("client_id"),
                client_secret=parsed_json.get("client_secret"),
                token_endpoint=parsed_json.get("token_endpoint"),
                scope=parsed_json.get("scope"),
            )
        except ValueError as exc:
            raise CredentialFileFormatError(
                "credential file parsed as JSON but is missing required "
                f"client_credentials fields: {path}",
                context=_context(
                    operation="load_worker_credential", target_name=str(path)
                ),
            ) from exc

    if "\n" in stripped or "\r" in stripped:
        raise CredentialFileFormatError(
            "credential file did not parse as JSON and contains embedded "
            f"newlines, so it cannot be treated as a single opaque token: {path}",
            context=_context(operation="load_worker_credential", target_name=str(path)),
        )

    return ModelWorkerCredential(auth_mode="bearer_token", bearer_token=stripped)
