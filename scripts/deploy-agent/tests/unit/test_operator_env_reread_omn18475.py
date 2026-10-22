# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The operator env store reaches every compose command, not only agent start (OMN-18475).

Measured on the compose dev lane, 2026-10-21: ``ESTIMATOR_FERNET_KEY`` was added
to the store at about 20:25Z, and the agent's ``onex-api`` recreates at 21:52Z
and 23:33Z brought the container up with the key empty, because the agent's
environment was the store as the launcher sourced it at unit start. A change of
``ONEX_CLOUD_MIGRATE_IMAGE`` at 22:48Z needed a hand restart for the same reason.

Every value below is synthetic. No real key or credential appears in this file.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.executor import (
    _compose_env,
    decode_operator_env_store,
    operator_store_overlay,
    reset_operator_store_cache,
)

pytestmark = pytest.mark.unit

_LAUNCHER = Path(__file__).resolve().parents[2] / "deploy" / "deploy-agent-launch.sh"
_PROTECTED = (
    "PATH DEPLOY_AGENT_ENV_FILE DEPLOY_AGENT_PYTHON DEPLOY_AGENT_ENV_PROTECTED "
    "KAFKA_ENVIRONMENT"
)
#: Shaped like a Fernet key (44 url-safe base64 characters), and not one.
_SYNTHETIC_KEY = "c3ludGhldGljLW5vdC1hLWtleS1vbW4xODQ3NV9fX18="
_OLD_MIGRATE = "registry.example/omninode-cloud-migrate:327620c2"
_NEW_MIGRATE = "local-build/omninode-cloud-migrate:e7d62566-20261021T192052Z"


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    reset_operator_store_cache()
    yield
    reset_operator_store_cache()


def _launched_against(
    monkeypatch: pytest.MonkeyPatch, store: Path, protected: str = _PROTECTED
) -> None:
    monkeypatch.setenv("DEPLOY_AGENT_ENV_FILE", str(store))
    monkeypatch.setenv("DEPLOY_AGENT_ENV_PROTECTED", protected)


def _edit(store: Path, text: str) -> None:
    """Rewrite the store and make sure its identity moves, as an edit's does."""
    before = store.stat().st_mtime_ns if store.exists() else 0
    store.write_text(text, encoding="utf-8")
    if store.stat().st_mtime_ns == before:
        os.utime(store, ns=(before + 1_000_000, before + 1_000_000))


def test_replay_estimator_key_added_after_start_reaches_the_next_recreate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED on the tree that re-read only ONEX_API_IMAGE: the key stays empty."""
    store = tmp_path / "operator.env"
    _edit(store, f"ONEX_CLOUD_MIGRATE_IMAGE={_OLD_MIGRATE}\n")
    _launched_against(monkeypatch, store)
    # The process as the launcher left it at unit start: no key yet.
    monkeypatch.delenv("ESTIMATOR_FERNET_KEY", raising=False)
    monkeypatch.setenv("ONEX_CLOUD_MIGRATE_IMAGE", _OLD_MIGRATE)
    assert "ESTIMATOR_FERNET_KEY" not in _compose_env()

    _edit(
        store,
        f"ONEX_CLOUD_MIGRATE_IMAGE={_NEW_MIGRATE}\n"
        f"ESTIMATOR_FERNET_KEY={_SYNTHETIC_KEY}\n",
    )

    env = _compose_env()
    assert env["ESTIMATOR_FERNET_KEY"] == _SYNTHETIC_KEY
    assert env["ONEX_CLOUD_MIGRATE_IMAGE"] == _NEW_MIGRATE


def test_unit_owned_names_are_never_taken_from_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The launcher restores these after its source; the re-read must too."""
    store = tmp_path / "operator.env"
    _edit(store, "KAFKA_ENVIRONMENT=local\nPATH=/store/bin\nPOSTGRES_HOST=db\n")
    _launched_against(monkeypatch, store)
    monkeypatch.setenv("KAFKA_ENVIRONMENT", "dev")

    overlay = operator_store_overlay()

    assert "KAFKA_ENVIRONMENT" not in overlay
    assert "PATH" not in overlay
    assert overlay["POSTGRES_HOST"] == "db"
    assert _compose_env()["KAFKA_ENVIRONMENT"] == "dev"


def test_the_overlay_decodes_exactly_what_the_launcher_decodes(
    tmp_path: Path,
) -> None:
    """Parity with ``deploy-agent-launch.sh``, run for real against one store.

    The narrow re-read existed because a second parser could decode a value
    differently from the launcher (OMN-18073). The overlay must equal what the
    launcher hands a freshly started agent for every name the store assigns,
    including an ANSI-C quoted multi-line value and a value that refers to
    another variable.
    """
    store = tmp_path / "operator.env"
    _edit(
        store,
        "# a comment\n"
        "PLAIN=plain\n"
        "QUOTED='single quoted # not a comment'\n"
        'DOUBLE="double $PLAIN"\n'
        "MULTI=$'line one\\nline two\\n'\n"
        "export EXPORTED=yes\n"
        "KAFKA_ENVIRONMENT=local\n",
    )
    dump = tmp_path / "env.dump"
    stub = tmp_path / "python"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os\n"
        f"open({str(dump)!r}, 'w').write(json.dumps(dict(os.environ)))\n"
    )
    stub.chmod(0o755)
    unit_env = {
        "PATH": os.environ.get("PATH", ""),
        "DEPLOY_AGENT_ENV_FILE": str(store),
        "DEPLOY_AGENT_PYTHON": str(stub),
        "DEPLOY_AGENT_ENV_PROTECTED": _PROTECTED,
        "KAFKA_ENVIRONMENT": "dev",
    }
    launched = subprocess.run(
        [str(_LAUNCHER)], capture_output=True, text=True, check=False, env=unit_env
    )
    assert launched.returncode == 0, launched.stderr
    started = json.loads(dump.read_text())

    overlay = operator_store_overlay(unit_env)

    assert overlay["MULTI"] == "line one\nline two\n"
    for name, value in overlay.items():
        assert started[name] == value, name
    assert set(overlay) == {"PLAIN", "QUOTED", "DOUBLE", "MULTI", "EXPORTED"}
    assert started["KAFKA_ENVIRONMENT"] == "dev"


def test_an_unchanged_store_is_not_decoded_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One stat per compose command between edits, one decode per edit."""
    store = tmp_path / "operator.env"
    _edit(store, "A=1\n")
    _launched_against(monkeypatch, store)
    real = subprocess.run
    decodes: list[object] = []

    def counting(*args: Any, **kwargs: Any) -> Any:
        decodes.append(args)
        return real(*args, **kwargs)

    with patch.object(executor_mod.subprocess, "run", side_effect=counting):
        assert operator_store_overlay()["A"] == "1"
        assert operator_store_overlay()["A"] == "1"
        assert len(decodes) == 1
        _edit(store, "A=2\n")
        assert operator_store_overlay()["A"] == "2"
        assert len(decodes) == 2


def test_a_store_that_fails_to_source_keeps_the_last_values_and_names_no_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Halfway through an edit, the store can be unparseable for a moment."""
    store = tmp_path / "operator.env"
    _edit(store, f"ESTIMATOR_FERNET_KEY={_SYNTHETIC_KEY}\n")
    _launched_against(monkeypatch, store)
    assert operator_store_overlay()["ESTIMATOR_FERNET_KEY"] == _SYNTHETIC_KEY

    _edit(store, f"ESTIMATOR_FERNET_KEY={_SYNTHETIC_KEY}\nif then fi (\n")
    with caplog.at_level(logging.WARNING, logger=executor_mod.__name__):
        overlay = operator_store_overlay()

    assert overlay["ESTIMATOR_FERNET_KEY"] == _SYNTHETIC_KEY
    assert "re-read refused" in caplog.text
    assert _SYNTHETIC_KEY not in caplog.text


def test_the_store_log_line_names_changes_and_never_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = tmp_path / "operator.env"
    _edit(store, f"ESTIMATOR_FERNET_KEY={_SYNTHETIC_KEY}\nSAME=same\n")
    _launched_against(monkeypatch, store)
    monkeypatch.setenv("SAME", "same")

    with caplog.at_level(logging.INFO, logger=executor_mod.__name__):
        operator_store_overlay()

    expected = (
        "1 of them not owned by the unit and applied to compose commands: "
        "ESTIMATOR_FERNET_KEY"
    )
    assert expected in caplog.text
    assert _SYNTHETIC_KEY not in caplog.text


def test_output_the_store_prints_is_not_mistaken_for_the_environment(
    tmp_path: Path,
) -> None:
    store = tmp_path / "operator.env"
    _edit(store, 'echo \'{"A": "forged"}\'\nA=real\n')

    assigned = decode_operator_env_store(store, base_env={"PATH": os.environ["PATH"]})

    assert assigned["A"] == "real"


def test_a_store_without_a_protected_list_is_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a process the launcher started has the list that fences the unit."""
    store = tmp_path / "operator.env"
    _edit(store, "KAFKA_ENVIRONMENT=local\n")
    monkeypatch.setenv("DEPLOY_AGENT_ENV_FILE", str(store))
    monkeypatch.delenv("DEPLOY_AGENT_ENV_PROTECTED", raising=False)

    assert operator_store_overlay() == {}


def test_a_name_deleted_from_the_store_keeps_its_start_time_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stated limit: the overlay sees assignments, not deletions."""
    store = tmp_path / "operator.env"
    _edit(store, "OTHER=1\n")
    _launched_against(monkeypatch, store)
    monkeypatch.setenv("DROPPED", "from-start")

    assert _compose_env()["DROPPED"] == "from-start"


def test_a_slow_store_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "operator.env"
    _edit(store, "sleep 5\nA=1\n")
    _launched_against(monkeypatch, store)
    monkeypatch.setattr(executor_mod, "OPERATOR_STORE_DECODE_TIMEOUT_SECONDS", 1)

    started = time.monotonic()
    assert operator_store_overlay() == {}
    assert time.monotonic() - started < 4
