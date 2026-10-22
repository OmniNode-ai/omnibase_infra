# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read back a CLI pin update through the running executor (OMN-18475).

The CLI runs in another process and replaces a real operator store. The
executor sources that store with bash and passes its environment to a child
process before and after the write. Docker residency and ledger append use
local executables; no shared daemon or rolling work ledger is touched.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from deploy_agent.executor import _compose_env, _run, reset_operator_store_cache

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[2]
_KEY = "ONEX_CLOUD_MIGRATE_IMAGE"
_OLD = "registry.example/cloud-migrate:before"
_NEW = "registry.example/cloud-migrate:after"
_IMAGE_ID = "sha256:" + "7" * 64


def _readback() -> dict[str, str]:
    result = _run(
        [
            sys.executable,
            "-c",
            "import json, os; print(json.dumps({k: os.environ[k] for k in "
            "['ONEX_CLOUD_MIGRATE_IMAGE', 'KAFKA_ENVIRONMENT', 'STORE_NOTE']}))",
        ],
        timeout=10,
        env=_compose_env(),
    )
    assert result.returncode == 0, result.stderr
    return dict(json.loads(result.stdout))


@pytest.mark.parametrize("ledger_exit", [0, 65], ids=["written", "rolled-back"])
def test_cli_pin_write_reaches_next_executor_child_without_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger_exit: int
) -> None:
    store = tmp_path / "operator.env"
    original = (
        f"export {_KEY}='{_OLD}'\n"
        "KAFKA_ENVIRONMENT=store-must-not-win\n"
        "STORE_NOTE=$'first line\\nsecond line'\n"
    )
    store.write_text(original, encoding="utf-8")
    store.chmod(0o600)
    docker = tmp_path / "resident-image"
    docker.write_text(
        "#!/bin/sh\n"
        '[ "$1" = image ] && [ "$2" = inspect ] && '
        '[ "$3" = --format ] && [ "$4" = "{{.Id}}" ] || exit 64\n'
        f'[ "$5" = "{_NEW}" ] || exit 1\n'
        f"printf '%s\\n' '{_IMAGE_ID}'\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    ledger = tmp_path / "local-ledger.txt"
    ledger.write_text("", encoding="utf-8")
    writer = tmp_path / "append-ledger"
    writer.write_text(
        "#!/bin/sh\n"
        '[ "$2" = --append ] || exit 64\n'
        f"if [ {ledger_exit} -ne 0 ]; then echo 'row refused' >&2; exit {ledger_exit}; fi\n"
        'printf "%s\\n" "$3" >> "$1"\n',
        encoding="utf-8",
    )
    writer.chmod(0o755)
    monkeypatch.setenv("DEPLOY_AGENT_ENV_FILE", str(store))
    monkeypatch.setenv(
        "DEPLOY_AGENT_ENV_PROTECTED",
        "PATH DEPLOY_AGENT_ENV_FILE DEPLOY_AGENT_ENV_PROTECTED KAFKA_ENVIRONMENT",
    )
    monkeypatch.setenv(_KEY, _OLD)
    monkeypatch.setenv("KAFKA_ENVIRONMENT", "dev")
    monkeypatch.setenv("PYTHONPATH", str(_ROOT / "src"))
    reset_operator_store_cache()
    try:
        before = _readback()
        assert before == {
            _KEY: _OLD,
            "KAFKA_ENVIRONMENT": "dev",
            "STORE_NOTE": "first line\nsecond line",
        }
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "omnibase_infra.handlers.handler_dev_lane_pin",
                "set",
                "--env-file",
                str(store),
                "--key",
                _KEY,
                "--value",
                _NEW,
                "--lane",
                "dev",
                "--ledger",
                str(ledger),
                "--ledger-command",
                str(writer),
                "--ledger-lane",
                "integration-omn18475",
                "--ticket",
                "OMN-18475",
                "--docker",
                str(docker),
                "--execute",
            ],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == (0 if ledger_exit == 0 else 3), result.stderr
        payload = json.loads(result.stdout)
        after = _readback()
        assert os.environ[_KEY] == _OLD
        assert after == {**before, _KEY: _NEW if ledger_exit == 0 else _OLD}
        backups = list(tmp_path.glob("operator.env.bak.omn18475-*"))
        assert len(backups) == 1
        assert backups[0].read_text(encoding="utf-8") == original
        assert store.stat().st_mode & 0o777 == 0o600
        if ledger_exit == 0:
            assert payload["outcome"] == "WRITTEN"
            assert payload["image_id"] == _IMAGE_ID
            assert store.read_text(encoding="utf-8") == original.replace(
                f"'{_OLD}'", _NEW
            )
            rows = ledger.read_text(encoding="utf-8").splitlines()
            assert len(rows) == 1
            assert f"pin-before={_OLD} | pin-after={_NEW}" in rows[0]
        else:
            assert payload["outcome"] == "REFUSED"
            assert "row refused" in payload["reason"]
            assert "restored" in payload["reason"]
            assert store.read_text(encoding="utf-8") == original
            assert ledger.read_text(encoding="utf-8") == ""
    finally:
        reset_operator_store_cache()
