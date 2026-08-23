# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for scripts.edge_delegation_worker.credential_loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.edge_delegation_worker.credential_loader import (
    CredentialFileFormatError,
    CredentialFilePermissionError,
    load_worker_credential,
)

pytestmark = pytest.mark.unit


def _write(path: Path, content: str, *, mode: int = 0o600) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(mode)
    return path


def test_bearer_token_mode_loads_opaque_token(tmp_path: Path) -> None:
    path = _write(tmp_path / "cred", "opaque-token-value-123\n")
    credential = load_worker_credential(path)
    assert credential.auth_mode == "bearer_token"
    assert credential.bearer_token == "opaque-token-value-123"


def test_client_credentials_mode_loads_json(tmp_path: Path) -> None:
    payload = {
        "client_id": "ga-tenant-1",
        "client_secret": "s3cr3t",
        "token_endpoint": "https://keycloak.example/realms/omninode/token",
        "scope": "gateway-attach",
    }
    path = _write(tmp_path / "cred.json", json.dumps(payload))
    credential = load_worker_credential(path)
    assert credential.auth_mode == "client_credentials"
    assert credential.client_id == "ga-tenant-1"
    assert credential.client_secret == "s3cr3t"
    assert credential.token_endpoint == payload["token_endpoint"]
    assert credential.scope == "gateway-attach"


def test_client_credentials_mode_rejects_missing_fields(tmp_path: Path) -> None:
    path = _write(tmp_path / "cred.json", json.dumps({"client_id": "ga-tenant-1"}))
    with pytest.raises(CredentialFileFormatError):
        load_worker_credential(path)


def test_rejects_group_readable_file(tmp_path: Path) -> None:
    path = _write(tmp_path / "cred", "opaque-token", mode=0o640)
    with pytest.raises(CredentialFilePermissionError):
        load_worker_credential(path)


def test_rejects_world_readable_file(tmp_path: Path) -> None:
    path = _write(tmp_path / "cred", "opaque-token", mode=0o604)
    with pytest.raises(CredentialFilePermissionError):
        load_worker_credential(path)


def test_rejects_empty_file(tmp_path: Path) -> None:
    path = _write(tmp_path / "cred", "   \n")
    with pytest.raises(CredentialFileFormatError):
        load_worker_credential(path)


def test_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CredentialFileFormatError):
        load_worker_credential(tmp_path / "does-not-exist")


def test_rejects_multiline_non_json_content(tmp_path: Path) -> None:
    path = _write(tmp_path / "cred", "line-one\nline-two")
    with pytest.raises(CredentialFileFormatError):
        load_worker_credential(path)


def test_credential_repr_never_leaks_secret(tmp_path: Path) -> None:
    path = _write(tmp_path / "cred", "super-secret-token-value")
    credential = load_worker_credential(path)
    rendered = repr(credential)
    assert "super-secret-token-value" not in rendered
    assert "super-secret-token-value" not in str(credential)
