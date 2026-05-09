# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for config writer — atomic env file merge + write."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_write_new_file(tmp_path: Path) -> None:
    from omnibase_infra.onboarding.config_writer import write_env_file

    target = tmp_path / "test.env"
    env_dict = {
        "ONEX_DEPLOYMENT_MODE": "local",
        "KAFKA_BOOTSTRAP_SERVERS": "localhost:19092",
    }
    content = write_env_file(env_dict, target)
    assert target.exists()
    assert "ONEX_DEPLOYMENT_MODE=local" in content
    assert "KAFKA_BOOTSTRAP_SERVERS=localhost:19092" in content


def test_merge_preserves_existing_keys(tmp_path: Path) -> None:
    from omnibase_infra.onboarding.config_writer import write_env_file

    target = tmp_path / "test.env"
    target.write_text("EXISTING_KEY=existing_value\nONEX_DEPLOYMENT_MODE=old_value\n")
    env_dict = {"ONEX_DEPLOYMENT_MODE": "local"}
    content = write_env_file(env_dict, target)
    assert "EXISTING_KEY=existing_value" in content
    assert "ONEX_DEPLOYMENT_MODE=local" in content
    assert "old_value" not in content


def test_overwrite_existing_key(tmp_path: Path) -> None:
    from omnibase_infra.onboarding.config_writer import write_env_file

    target = tmp_path / "test.env"
    target.write_text("ONEX_DEPLOYMENT_MODE=cloud\n")
    env_dict = {"ONEX_DEPLOYMENT_MODE": "local"}
    write_env_file(env_dict, target)
    assert target.read_text().count("ONEX_DEPLOYMENT_MODE") == 1
    assert "ONEX_DEPLOYMENT_MODE=local" in target.read_text()


def test_write_is_atomic(tmp_path: Path) -> None:
    from omnibase_infra.onboarding.config_writer import write_env_file

    target = tmp_path / "test.env"
    target.write_text("EXISTING=value\n")
    env_dict = {"NEW_KEY": "new_value"}
    write_env_file(env_dict, target)
    # Atomic means no tmp file left behind
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert len(tmp_files) == 0


def test_returns_merged_content_as_string(tmp_path: Path) -> None:
    from omnibase_infra.onboarding.config_writer import write_env_file

    target = tmp_path / "test.env"
    env_dict = {"KEY_A": "val_a", "KEY_B": "val_b"}
    content = write_env_file(env_dict, target)
    assert isinstance(content, str)
    assert "KEY_A=val_a" in content
    assert "KEY_B=val_b" in content


def test_empty_env_dict_preserves_existing(tmp_path: Path) -> None:
    from omnibase_infra.onboarding.config_writer import write_env_file

    target = tmp_path / "test.env"
    target.write_text("EXISTING=value\n")
    content = write_env_file({}, target)
    assert "EXISTING=value" in content


def test_no_writes_under_real_home(tmp_path: Path) -> None:
    from omnibase_infra.onboarding.config_writer import write_env_file

    real_home_env = Path.home() / ".omnibase" / ".env"
    target = tmp_path / "safe.env"
    write_env_file({"KEY": "value"}, target)
    # Verify the real file was NOT touched (if it exists, its mtime is old enough)
    if real_home_env.exists():
        import time

        mtime = real_home_env.stat().st_mtime
        assert time.time() - mtime > 5, "Real ~/.omnibase/.env was touched during test"


def test_roundtrip_read_back(tmp_path: Path) -> None:
    from omnibase_infra.onboarding.config_writer import write_env_file

    target = tmp_path / "test.env"
    env_dict = {
        "ONEX_DEPLOYMENT_MODE": "hybrid",
        "CLOUD_PROVIDER": "aws",
        "AWS_REGION": "us-east-1",
    }
    write_env_file(env_dict, target)
    lines = target.read_text().splitlines()
    parsed = {}
    for line in lines:
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            parsed[k] = v
    for key, val in env_dict.items():
        assert parsed[key] == val
