# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21529 K1: topic provisioning authenticates to a SASL broker from the lane.

Falsifier: provisioning against a SASL broker with only the lane declaration
and the identity succeeds; with the identity missing it refuses naming the
overlay key. Before the fix the script had no ``--lane`` and built an
``AdminClient`` with ``bootstrap.servers`` alone, which a SCRAM listener
refuses.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "create_kafka_topics.py"

_CRED = "k1-test-value"
_OVERLAY = """\
lanes:
  dev:
    broker: "broker.invalid:19092"
    security_protocol: SASL_PLAINTEXT
    sasl_mechanism: SCRAM-SHA-256
  plain:
    broker: "plain.invalid:19092"
    security_protocol: PLAINTEXT
"""


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("create_kafka_topics_k1", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["create_kafka_topics_k1"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config_root = tmp_path / "cfg"
    declaration = config_root / "config" / "lab" / "ci_bus_lanes.yaml"
    declaration.parent.mkdir(parents=True)
    declaration.write_text(_OVERLAY, encoding="utf-8")
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(config_root))
    monkeypatch.setenv("OMNIBASE_PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("KAFKA_SASL_USERNAME", raising=False)
    monkeypatch.delenv("KAFKA_SASL_PASSWORD", raising=False)
    return tmp_path


def _run_live_main(argv: list[str]) -> tuple[int, MagicMock]:
    mod = _load()
    admin_cls = MagicMock()
    admin_cls.return_value.list_topics.side_effect = RuntimeError("stop after config")
    fake_admin = ModuleType("confluent_kafka.admin")
    fake_admin.AdminClient = admin_cls
    fake_admin.NewTopic = MagicMock()
    entry = MagicMock()
    entry.topic = "onex.evt.k1.v1"
    with (
        patch.object(sys, "argv", ["create_kafka_topics.py", *argv]),
        patch.dict(sys.modules, {"confluent_kafka.admin": fake_admin}),
        patch(
            "omnibase_infra.models.model_node_package_entry.ModelNodePackageEntry.discover_pairs",
            return_value=[("p", Path())],
        ),
        patch(
            "omnibase_infra.tools.contract_topic_extractor.ContractTopicExtractor.extract",
            return_value=[entry],
        ),
        patch.object(mod, "_build_specs", return_value=[]),
    ):
        code = mod.main()
    return code, admin_cls


def test_lane_with_env_identity_builds_a_sasl_admin_client(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAFKA_SASL_USERNAME", "svc")
    monkeypatch.setenv("KAFKA_SASL_PASSWORD", _CRED)
    _code, admin_cls = _run_live_main(["--lane", "dev"])
    admin_cls.assert_called_once_with(
        {
            "bootstrap.servers": "broker.invalid:19092",
            "security.protocol": "SASL_PLAINTEXT",
            "sasl.mechanism": "SCRAM-SHA-256",
            "sasl.username": "svc",
            "sasl.password": _CRED,
        }
    )


def test_lane_with_stored_identity_builds_a_sasl_admin_client(
    workspace: Path,
) -> None:
    from omnibase_infra.cli.store_lane_credential import StoreLaneCredential

    StoreLaneCredential(onex_home=workspace / "home" / ".onex").save(
        lane="dev",
        sasl_username="stored",
        sasl_password=_CRED,
    )
    _code, admin_cls = _run_live_main(["--lane", "dev"])
    (config,) = admin_cls.call_args.args
    assert config["sasl.username"] == "stored"
    assert config["sasl.mechanism"] == "SCRAM-SHA-256"
    assert config["bootstrap.servers"] == "broker.invalid:19092"


def test_missing_identity_refuses_naming_the_overlay_key(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, admin_cls = _run_live_main(["--lane", "dev"])
    assert code == 1
    admin_cls.assert_not_called()
    err = capsys.readouterr().err
    assert "lanes.dev" in err
    assert "lane-login" in err


def test_plaintext_lane_needs_no_identity(workspace: Path) -> None:
    _code, admin_cls = _run_live_main(["--lane", "plain"])
    admin_cls.assert_called_once_with(
        {
            "bootstrap.servers": "plain.invalid:19092",
            "security.protocol": "PLAINTEXT",
        }
    )


def test_lane_and_bootstrap_servers_are_mutually_exclusive(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, admin_cls = _run_live_main(
        ["--lane", "dev", "--bootstrap-servers", "x.invalid:1"]
    )
    assert code == 2
    admin_cls.assert_not_called()
    assert "mutually exclusive" in capsys.readouterr().err


def test_bootstrap_servers_alone_is_unchanged(workspace: Path) -> None:
    _code, admin_cls = _run_live_main(["--bootstrap-servers", "x.invalid:1"])
    admin_cls.assert_called_once_with({"bootstrap.servers": "x.invalid:1"})
