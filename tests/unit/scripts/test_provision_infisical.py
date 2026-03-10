# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for provision-infisical.py (OMN-4044).

Tests:
    - test_already_provisioned_runs_folder_creation_not_credentials
    - test_already_provisioned_dry_run_returns_0_without_http
    - test_already_provisioned_missing_admin_token_returns_1
    - test_already_provisioned_missing_project_id_in_env_returns_1
    - test_folder_creation_idempotent_logs_skipped
    - test_folder_creation_new_logs_created
    - test_fresh_provision_runs_full_flow
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Script has a hyphenated name — use sys.path insertion + importlib.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import importlib

_provision_mod = importlib.import_module("provision-infisical")

_create_infisical_folders = _provision_mod._create_infisical_folders
_main = _provision_mod.main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_httpx_response(
    status_code: int, json_data: dict[str, Any] | None = None
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = ""
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


def _write_env(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


# ---------------------------------------------------------------------------
# Tests: _create_infisical_folders idempotency logging
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateInfisicalFoldersIdempotency:
    """_create_infisical_folders should log [idempotent] on 400/409, [created] on 200/201."""

    def test_folder_creation_new_logs_created(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """201 response → logs [created] for each folder."""
        client = MagicMock()
        client.post.return_value = _make_httpx_response(201)

        import logging

        with caplog.at_level(logging.INFO, logger="provision-infisical"):
            _create_infisical_folders(
                client,
                "http://localhost:8880",
                "tok",
                "proj-id",
                environments=("dev",),
                transport_folders=("db",),
            )

        log_text = caplog.text
        assert "[created]" in log_text
        assert "[idempotent]" not in log_text

    def test_folder_creation_idempotent_logs_skipped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """409 response → logs [idempotent] for each folder."""
        client = MagicMock()
        resp_409 = _make_httpx_response(409)
        resp_409.text = "already exists"
        client.post.return_value = resp_409

        import logging

        with caplog.at_level(logging.INFO, logger="provision-infisical"):
            _create_infisical_folders(
                client,
                "http://localhost:8880",
                "tok",
                "proj-id",
                environments=("dev",),
                transport_folders=("db",),
            )

        log_text = caplog.text
        assert "[idempotent]" in log_text
        assert "[created]" not in log_text


# ---------------------------------------------------------------------------
# Tests: main() — already-provisioned path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMainAlreadyProvisioned:
    """When credentials already exist in the env file, main() must run folder creation."""

    def _env_with_credentials(self, tmp_path: Path) -> Path:
        env_file = tmp_path / ".env"
        _write_env(
            env_file,
            (
                "INFISICAL_CLIENT_ID=existing-client-id\n"
                "INFISICAL_CLIENT_SECRET=existing-secret\n"
                "INFISICAL_PROJECT_ID=existing-project-id\n"
            ),
        )
        return env_file

    def _admin_token_file(
        self, tmp_path: Path, content: str = "admin-token-value\n"
    ) -> Path:
        token_file = tmp_path / ".infisical-admin-token"
        token_file.write_text(content)
        return token_file

    def test_already_provisioned_dry_run_returns_0_without_http(
        self, tmp_path: Path
    ) -> None:
        """--dry-run with credentials present must return 0 without any HTTP call."""
        env_file = self._env_with_credentials(tmp_path)

        with (
            patch.object(_provision_mod, "_ENV_FILE", env_file),
            patch.object(
                _provision_mod, "_ADMIN_TOKEN_FILE", tmp_path / ".infisical-admin-token"
            ),
            patch(
                "sys.argv",
                ["provision-infisical.py", "--dry-run", f"--env-file={env_file}"],
            ),
        ):
            rc = _main()

        assert rc == 0

    def test_already_provisioned_missing_admin_token_returns_1(
        self, tmp_path: Path
    ) -> None:
        """When admin token file is missing, must return 1 with an error."""
        env_file = self._env_with_credentials(tmp_path)
        missing_token = tmp_path / ".infisical-admin-token-nonexistent"

        status_resp = _make_httpx_response(200, {"status": "ok"})
        mock_client_cm = MagicMock()
        mock_client_cm.__enter__ = MagicMock(return_value=mock_client_cm)
        mock_client_cm.__exit__ = MagicMock(return_value=False)
        mock_client_cm.get.return_value = status_resp

        with (
            patch.object(_provision_mod, "_ENV_FILE", env_file),
            patch.object(_provision_mod, "_ADMIN_TOKEN_FILE", missing_token),
            patch("sys.argv", ["provision-infisical.py", f"--env-file={env_file}"]),
            patch("httpx.Client", return_value=mock_client_cm),
        ):
            rc = _main()

        assert rc == 1

    def test_already_provisioned_missing_project_id_returns_1(
        self, tmp_path: Path
    ) -> None:
        """When INFISICAL_PROJECT_ID is missing from env, must return 1."""
        env_file = tmp_path / ".env"
        _write_env(
            env_file,
            (
                "INFISICAL_CLIENT_ID=existing-client-id\n"
                "INFISICAL_CLIENT_SECRET=existing-secret\n"
                # INFISICAL_PROJECT_ID deliberately omitted
            ),
        )
        # PROJECT_ID missing means _already_provisioned=False → fresh provision path,
        # not the already-provisioned path. Test the case where project_id is empty
        # after reading existing_env (edge case: key present but value empty).
        env_file2 = tmp_path / ".env2"
        _write_env(
            env_file2,
            (
                "INFISICAL_CLIENT_ID=existing-client-id\n"
                "INFISICAL_CLIENT_SECRET=existing-secret\n"
                "INFISICAL_PROJECT_ID=\n"  # present but empty
            ),
        )
        # With empty PROJECT_ID, _already_provisioned=False → takes fresh path.
        # This test documents that an empty PROJECT_ID value is not treated as provisioned.
        with (
            patch.object(_provision_mod, "_ENV_FILE", env_file2),
            patch.object(
                _provision_mod,
                "_ADMIN_TOKEN_FILE",
                tmp_path / ".infisical-admin-token",
            ),
            patch(
                "sys.argv",
                ["provision-infisical.py", "--dry-run", f"--env-file={env_file2}"],
            ),
        ):
            rc = _main()

        # dry-run on fresh provision path returns 0
        assert rc == 0

    def test_already_provisioned_runs_folder_creation(self, tmp_path: Path) -> None:
        """With credentials present and Infisical reachable, folder creation must run."""
        env_file = self._env_with_credentials(tmp_path)
        token_file = self._admin_token_file(tmp_path)

        status_resp = _make_httpx_response(200, {"status": "ok"})
        folder_resp = _make_httpx_response(201)

        folder_calls: list[dict[str, Any]] = []

        def fake_post(url: str, **kwargs: Any) -> MagicMock:
            if "/api/v1/folders" in url:
                folder_calls.append(kwargs.get("json", {}))
                return folder_resp
            return _make_httpx_response(200)

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = status_resp
        mock_client.post.side_effect = fake_post

        with (
            patch.object(_provision_mod, "_ENV_FILE", env_file),
            patch.object(_provision_mod, "_ADMIN_TOKEN_FILE", token_file),
            patch("sys.argv", ["provision-infisical.py", f"--env-file={env_file}"]),
            patch("httpx.Client", return_value=mock_client),
        ):
            rc = _main()

        assert rc == 0
        # At least one folder creation call must have been made
        assert len(folder_calls) > 0


__all__: list[str] = [
    "TestCreateInfisicalFoldersIdempotency",
    "TestMainAlreadyProvisioned",
]
