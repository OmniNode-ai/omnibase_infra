# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Credentials writer for onboarding-generated secret material (OMN-16035).

The JSON sibling of :mod:`omnibase_infra.onboarding.config_writer`. Where
``ConfigWriter`` emits a ``KEY=value`` env file, this writer emits a nested JSON
document and enforces a stricter permission invariant: the artifact is always
left at mode ``0600``, and that is verified by an explicit ``stat`` postcondition
rather than inherited from ``mkstemp``'s default or an ``open()`` mode flag.

Explicit invocation only, mirroring ``ConfigWriter``: nothing here runs as a side
effect of another code path. Dry-run callers must use
:meth:`CredentialsWriter.render`, which never touches the filesystem; only
:meth:`CredentialsWriter.write` (or :func:`write_credentials_file`) creates a
file, and the onboarding handler calls it only when the caller supplied an
explicit credentials output path with ``dry_run=False``.
"""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path

from omnibase_core.types import StrictJsonType

#: Mode enforced on the credentials artifact itself.
CREDENTIALS_FILE_MODE = 0o600

#: Mode used when this writer has to create the containing directory.
CREDENTIALS_DIR_MODE = 0o700

#: Env keys whose names mark them as secret material. Same marker vocabulary as
#: ``omnibase_infra.runtime.overlay.overlay_writer``, but anchored to
#: underscore-delimited tokens: that writer only logs a warning, so a substring
#: hit is harmless there, whereas here the match decides what content is copied
#: into the artifact and ``VALKEY_HOST`` must not be mistaken for a key.
CREDENTIAL_KEY_PATTERN = re.compile(
    r"(?:^|_)(PASSWORD|PASSWD|SECRET|TOKEN|KEY|APIKEY|CREDENTIAL|CREDENTIALS)(?:_|$)",
    re.IGNORECASE,
)


class CredentialsWriterError(ValueError):
    """Raised when credentials cannot be rendered or written safely."""


def _validate_keys(node: object, path: str = "") -> None:
    """Reject non-string keys, which ``json.dumps`` would silently coerce."""
    if not isinstance(node, Mapping):
        return
    for key, value in node.items():
        if not isinstance(key, str):
            msg = f"credentials contain a non-string key at {path}{key!r}"
            raise CredentialsWriterError(msg)
        _validate_keys(value, path=f"{path}{key}.")


def _assert_mode_0600(path: Path) -> None:
    """Fail closed unless ``path`` is exactly mode 0600."""
    actual_mode = stat.S_IMODE(path.stat().st_mode)
    if actual_mode != CREDENTIALS_FILE_MODE:
        msg = f"credentials file {path} is mode {actual_mode:04o}, expected 0600"
        raise CredentialsWriterError(msg)


def _deep_merge(
    base: dict[str, StrictJsonType],
    overlay: Mapping[str, StrictJsonType],
) -> dict[str, StrictJsonType]:
    """Merge ``overlay`` into ``base`` recursively, preserving unrelated keys.

    A mapping merges into a mapping; anything else replaces the existing value
    outright (so a rotated credential can collapse a subtree to a scalar).
    """
    merged: dict[str, StrictJsonType] = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


class CredentialsWriter:
    """Writes nested JSON credentials with merge-and-preserve semantics.

    Explicit invocation only. Never write to the real ``~/.onex/`` from tests.
    """

    def render(
        self,
        credentials: Mapping[str, StrictJsonType],
        existing_content: str | None = None,
    ) -> str:
        """Return the merged JSON document without writing to disk.

        Raises:
            CredentialsWriterError: If ``existing_content`` is not a JSON object,
                or ``credentials`` contains non-string keys or values that JSON
                cannot represent.
        """
        existing: dict[str, StrictJsonType] = {}
        if existing_content is not None and existing_content.strip():
            try:
                parsed: object = json.loads(existing_content)
            except json.JSONDecodeError as exc:
                msg = "existing credentials file is not valid JSON"
                raise CredentialsWriterError(msg) from exc
            if not isinstance(parsed, dict):
                msg = (
                    "existing credentials file must contain a JSON object, "
                    f"found {type(parsed).__name__}"
                )
                raise CredentialsWriterError(msg)
            _validate_keys(parsed)
            existing = parsed

        _validate_keys(credentials)
        merged = _deep_merge(existing, credentials)

        try:
            document = json.dumps(
                merged, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False
            )
        except (TypeError, ValueError) as exc:
            msg = f"credentials value is not JSON-serializable: {exc}"
            raise CredentialsWriterError(msg) from exc

        return document + "\n"

    def write(
        self,
        credentials: Mapping[str, StrictJsonType],
        target_path: Path,
    ) -> str:
        """Merge ``credentials`` into any existing file and atomically write it.

        The write is ``mkstemp`` + :func:`os.replace`, so an interrupted run
        leaves either the previous file or the new one — never a torn file. The
        ``0600`` mode is applied explicitly to the temp file descriptor before
        any secret bytes are written, re-applied to the final path, and then
        verified by ``stat``.

        Returns:
            The exact content written to ``target_path``.

        Raises:
            CredentialsWriterError: If rendering fails, or if the final artifact
                is not left at mode ``0600``.
        """
        existing_content: str | None = None
        if target_path.exists():
            existing_content = target_path.read_text(encoding="utf-8")

        content = self.render(credentials, existing_content)

        target_path.parent.mkdir(parents=True, exist_ok=True, mode=CREDENTIALS_DIR_MODE)

        fd, tmp_path_str = tempfile.mkstemp(
            dir=target_path.parent,
            prefix=f".{target_path.name}.tmp.",
        )
        tmp_path = Path(tmp_path_str)
        try:
            # Explicit chmod on the descriptor, before the secret bytes land:
            # not inherited from mkstemp's default and not an open() mode flag.
            os.fchmod(fd, CREDENTIALS_FILE_MODE)
            with os.fdopen(fd, "w", encoding="utf-8") as file_handle:
                file_handle.write(content)
            _assert_mode_0600(tmp_path)
            tmp_path.replace(target_path)
        except Exception:
            with suppress(OSError):
                tmp_path.unlink(missing_ok=True)
            raise

        target_path.chmod(CREDENTIALS_FILE_MODE)
        _assert_mode_0600(target_path)

        return content


def select_credential_entries(env_dict: Mapping[str, str]) -> dict[str, str]:
    """Return only the entries whose key names mark them as secret material.

    Callers use this to keep non-secret configuration out of the ``0600``
    artifact; the non-secret remainder belongs in the overlay/env output.
    """
    return {
        key: value
        for key, value in env_dict.items()
        if CREDENTIAL_KEY_PATTERN.search(key)
    }


def write_credentials_file(
    credentials: Mapping[str, StrictJsonType],
    target_path: Path,
) -> str:
    """Explicit convenience wrapper for callers that do not need an instance."""
    return CredentialsWriter().write(credentials, target_path)


__all__ = [
    "CREDENTIALS_DIR_MODE",
    "CREDENTIALS_FILE_MODE",
    "CREDENTIAL_KEY_PATTERN",
    "CredentialsWriter",
    "CredentialsWriterError",
    "select_credential_entries",
    "write_credentials_file",
]
