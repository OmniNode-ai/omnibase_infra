# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev deploy agent signs with its own dev-only key, not the shared one (OMN-24565).

The operator env store carries one ``DEPLOY_AGENT_HMAC_SECRET``, and the live deploy
agent reads the same name from the same file. The dev agent
(``deploy-agent-dev.service``) must never hold that value: its unit reads the
dev-only key from a second ``EnvironmentFile=`` that holds that one key, and lists
the name in ``DEPLOY_AGENT_ENV_PROTECTED`` so neither the launcher's store source
nor ``deploy_agent.executor.operator_store_overlay`` puts the shared value back.
``auth.py``, the launcher and the compose files do not change.

Every value below is a synthetic throwaway, and assertion messages carry variable
NAMES only, never values.
"""

import json
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from deploy_agent.executor import operator_store_overlay, reset_operator_store_cache

pytestmark = pytest.mark.unit

_DEPLOY_DIR = Path(__file__).resolve().parents[2] / "deploy"
_DEV_UNIT = _DEPLOY_DIR / "deploy-agent-dev.service"
_PROD_DROP_IN_DIR = _DEPLOY_DIR / "deploy-agent.service.d"
_LAUNCHER = _DEPLOY_DIR / "deploy-agent-launch.sh"

_CODE_NAME = "DEPLOY_AGENT_HMAC_SECRET"
_DEV_STORE_NAME = "DEPLOY_AGENT_DEV_HMAC_SECRET"
_PROTECTED_NAME = "DEPLOY_AGENT_ENV_PROTECTED"
#: The dev agent's own key file: the one key and nothing else, in the unit user's home.
_KEY_FILE = "%h/.omnibase/deploy-agent-dev-signing-key.env"

_SHARED_SYNTHETIC = "shared-synthetic"
_DEV_SYNTHETIC = "dev-synthetic"


def _directive_lines(unit: Path) -> list[str]:
    return [
        line
        for line in unit.read_text().splitlines()
        if line and not line.lstrip().startswith("#")
    ]


def _unit_value(unit: Path, name: str) -> str:
    for line in _directive_lines(unit):
        for prefix in (f"Environment={name}=", f'Environment="{name}='):
            if line.startswith(prefix):
                return line[len(prefix) :].rstrip('"')
    raise AssertionError(f"{unit.name} does not declare Environment={name}=")


def _dev_protected() -> list[str]:
    return _unit_value(_DEV_UNIT, _PROTECTED_NAME).split()


# --------------------------------------------------------------------------
# Unit-level invariants. The first is the failing control: on origin/dev the unit
# declares no EnvironmentFile= and the name is not protected.
# --------------------------------------------------------------------------


def test_dev_unit_reads_its_own_signing_key_file() -> None:
    files = [
        line
        for line in _directive_lines(_DEV_UNIT)
        if line.startswith("EnvironmentFile=")
    ]
    # Exactly one, required (no leading `-`: an absent key fails the start loudly),
    # and never the operator store (OMN-18073: systemd mangles its quoting).
    assert files == [f"EnvironmentFile={_KEY_FILE}"], "EnvironmentFile="
    assert ".omnibase/.env" not in files[0], "EnvironmentFile="


def test_dev_unit_protects_the_signing_key_name_from_the_store() -> None:
    """Without the entry the launcher's store source puts the shared value back."""
    assert _CODE_NAME in _dev_protected(), _PROTECTED_NAME


def test_no_other_unit_reads_the_dev_signing_key_file() -> None:
    """The live agent's unit and its drop-in keep reading the shared name alone."""
    texts = [
        path.read_text()
        for path in sorted(_DEPLOY_DIR.glob("*.service"))
        if path != _DEV_UNIT
    ]
    texts += [
        path.read_text()
        for path in sorted(_PROD_DROP_IN_DIR.glob("*"))
        if path.is_file()
    ]
    assert len(texts) >= 2, "no other unit file found -- the glob is wrong"
    for text in texts:
        assert "deploy-agent-dev-signing-key" not in text, "EnvironmentFile="
        assert _DEV_STORE_NAME not in text, _DEV_STORE_NAME


# --------------------------------------------------------------------------
# Behaviour: systemd sets the dev value, the launcher sources a store holding the
# shared one, and the agent's environment carries the dev value.
# --------------------------------------------------------------------------


def _run_launcher(tmp_path: Path, protected: str) -> dict[str, str]:
    """Run the launcher as systemd would start it, with a stub that dumps its env."""
    dump = tmp_path / "env.json"
    stub = tmp_path / "python"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os\n"
        f"open({str(dump)!r}, 'w').write(json.dumps(dict(os.environ)))\n"
    )
    stub.chmod(0o755)
    store = tmp_path / "store.env"
    store.write_text(f"{_CODE_NAME}={_SHARED_SYNTHETIC}\nSTORE_ONLY_NAME=from-store\n")
    result = subprocess.run(
        [str(_LAUNCHER)],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": os.environ.get("PATH", ""),
            "DEPLOY_AGENT_ENV_FILE": str(store),
            "DEPLOY_AGENT_PYTHON": str(stub),
            _PROTECTED_NAME: protected,
            # What systemd's EnvironmentFile= puts in the process before ExecStart.
            _CODE_NAME: _DEV_SYNTHETIC,
        },
    )
    assert result.returncode == 0, (result.returncode, result.stderr)
    return dict(json.loads(dump.read_text()))


def test_launcher_keeps_the_dev_key_over_the_stores_shared_value(
    tmp_path: Path,
) -> None:
    env = _run_launcher(tmp_path, " ".join(_dev_protected()))

    assert env[_CODE_NAME] == _DEV_SYNTHETIC, _CODE_NAME
    # Positive control: the store was sourced, so the line above is precedence and
    # not a source that did nothing.
    assert env["STORE_ONLY_NAME"] == "from-store", "STORE_ONLY_NAME"
    for name, value in env.items():
        assert value != _SHARED_SYNTHETIC, name


def test_control_without_the_protected_entry_the_shared_value_wins(
    tmp_path: Path,
) -> None:
    """The error chain: drop the entry and the dev agent holds the shared key."""
    protected = [name for name in _dev_protected() if name != _CODE_NAME]
    env = _run_launcher(tmp_path, " ".join(protected))

    assert env[_CODE_NAME] == _SHARED_SYNTHETIC, _CODE_NAME


# --------------------------------------------------------------------------
# The seam with the executor: compose commands re-read the store for every name the
# unit does not own, and must not carry the shared key to the dev runtime.
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_operator_store_cache() -> Iterator[None]:
    reset_operator_store_cache()
    yield
    reset_operator_store_cache()


def _overlay(tmp_path: Path, protected: list[str]) -> dict[str, str]:
    store = tmp_path / "store.env"
    store.write_text(f"{_CODE_NAME}={_SHARED_SYNTHETIC}\n")
    return operator_store_overlay(
        {
            "PATH": os.environ.get("PATH", ""),
            "DEPLOY_AGENT_ENV_FILE": str(store),
            _PROTECTED_NAME: " ".join(protected),
        }
    )


def test_the_overlay_never_reapplies_the_shared_value(tmp_path: Path) -> None:
    assert _CODE_NAME not in _overlay(tmp_path, _dev_protected()), _CODE_NAME


def test_control_the_overlay_reapplies_it_when_the_name_is_not_protected(
    tmp_path: Path,
) -> None:
    protected = [name for name in _dev_protected() if name != _CODE_NAME]
    assert _overlay(tmp_path, protected)[_CODE_NAME] == _SHARED_SYNTHETIC, _CODE_NAME
