# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression tests for required compose environment validation (OMN-15009)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import check_required_env_vars


@pytest.mark.unit
def test_nonempty_process_environment_satisfies_required_compose_var(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compose = tmp_path / "compose.yml"
    compose.write_text(
        "services:\n  effects:\n    environment:\n"
        "      DEPLOY_AGENT_HMAC_SECRET: "
        "${DEPLOY_AGENT_HMAC_SECRET:?required}\n",
        encoding="utf-8",
    )
    empty_env_file = tmp_path / "runtime-policy.env"
    empty_env_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("DEPLOY_AGENT_HMAC_SECRET", "render-only-not-a-secret")

    result = check_required_env_vars.main(
        [
            "--compose-file",
            str(compose),
            "--env-file",
            str(empty_env_file),
        ]
    )

    assert result == 0


@pytest.mark.unit
def test_empty_process_environment_does_not_satisfy_required_compose_var(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compose = tmp_path / "compose.yml"
    compose.write_text(
        "services:\n  effects:\n    environment:\n"
        "      DEPLOY_AGENT_HMAC_SECRET: "
        "${DEPLOY_AGENT_HMAC_SECRET:?required}\n",
        encoding="utf-8",
    )
    empty_env_file = tmp_path / "runtime-policy.env"
    empty_env_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("DEPLOY_AGENT_HMAC_SECRET", "")

    result = check_required_env_vars.main(
        [
            "--compose-file",
            str(compose),
            "--env-file",
            str(empty_env_file),
        ]
    )

    assert result == 1
