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

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import MagicMock, patch
from uuid import uuid4

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


_TOPIC = "onex.evt.k1.topic-provision.v1"


def _run_live_main(
    argv: list[str], *, existing: tuple[str, ...] = ()
) -> tuple[int, MagicMock, MagicMock]:
    """Run ``main()`` through the real spec builder against a fake broker.

    Install a temporary package with real entry-point metadata and a topic
    contract. Only the broker boundary is substituted.
    """

    mod = _load()
    admin_cls = MagicMock()
    brokers = {1: MagicMock()}
    before = MagicMock()
    before.topics = dict.fromkeys(existing)
    before.brokers = brokers
    after = MagicMock()
    after.topics = dict.fromkeys((*existing, _TOPIC))
    after.brokers = brokers
    admin = admin_cls.return_value
    admin.list_topics.side_effect = [before, after]
    done = MagicMock()
    done.exception.return_value = None
    admin.create_topics.return_value = {_TOPIC: done}
    fake_admin = ModuleType("confluent_kafka.admin")
    fake_admin.AdminClient = admin_cls
    fake_admin.NewTopic = MagicMock()
    package_name = f"landing_topic_{uuid4().hex}"
    with TemporaryDirectory() as directory:
        root = Path(directory)
        package = root / package_name
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "contract.yaml").write_text(
            f"event_bus:\n  publish_topics:\n    - topic: {_TOPIC}\n"
            "      topic_config:\n        partitions: 1\n"
            "        replication_factor: 1\n",
            encoding="utf-8",
        )
        metadata = root / f"{package_name}-1.0.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(
            f"Name: {package_name}\nVersion: 1.0\n", encoding="utf-8"
        )
        (metadata / "entry_points.txt").write_text(
            f"[onex.node_package]\n{package_name} = {package_name}\n",
            encoding="utf-8",
        )
        try:
            with (
                patch.object(sys, "path", [str(root), *sys.path]),
                patch.object(
                    sys,
                    "argv",
                    ["create_kafka_topics.py", *argv, "--packages", package_name],
                ),
                patch.dict(sys.modules, {"confluent_kafka.admin": fake_admin}),
            ):
                importlib.invalidate_caches()
                code = mod.main()
        finally:
            sys.modules.pop(package_name, None)
    return code, admin_cls, fake_admin.NewTopic


def test_lane_with_env_identity_builds_a_sasl_admin_client(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAFKA_SASL_USERNAME", "svc")
    monkeypatch.setenv("KAFKA_SASL_PASSWORD", _CRED)
    code, admin_cls, _ = _run_live_main(["--lane", "dev"])
    assert code == 0
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
    code, admin_cls, _ = _run_live_main(["--lane", "dev"])
    assert code == 0
    (config,) = admin_cls.call_args.args
    assert config["sasl.username"] == "stored"
    assert config["sasl.password"] == _CRED
    assert config["sasl.mechanism"] == "SCRAM-SHA-256"
    assert config["bootstrap.servers"] == "broker.invalid:19092"


def test_missing_identity_refuses_naming_the_overlay_key(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, admin_cls, _ = _run_live_main(["--lane", "dev"])
    assert code == 1
    admin_cls.assert_not_called()
    err = capsys.readouterr().err
    assert "lanes.dev" in err
    assert "lane-login" in err


def test_plaintext_lane_needs_no_identity(workspace: Path) -> None:
    code, admin_cls, _ = _run_live_main(["--lane", "plain"])
    assert code == 0
    admin_cls.assert_called_once_with(
        {
            "bootstrap.servers": "plain.invalid:19092",
            "security.protocol": "PLAINTEXT",
        }
    )


def test_lane_and_bootstrap_servers_are_mutually_exclusive(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as raised:
        _run_live_main(["--lane", "dev", "--bootstrap-servers", "x.invalid:1"])
    assert raised.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_bootstrap_servers_alone_is_unchanged(workspace: Path) -> None:
    code, admin_cls, _ = _run_live_main(["--bootstrap-servers", "x.invalid:1"])
    assert code == 0
    admin_cls.assert_called_once_with({"bootstrap.servers": "x.invalid:1"})


def test_undeclared_lane_refuses_naming_the_overlay_key(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, admin_cls, _ = _run_live_main(["--lane", "nope"])
    assert code == 1
    admin_cls.assert_not_called()
    assert "lanes.nope" in capsys.readouterr().err


def test_half_a_credential_pair_refuses(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("KAFKA_SASL_USERNAME", "svc")
    code, admin_cls, _ = _run_live_main(["--lane", "dev"])
    assert code == 1
    admin_cls.assert_not_called()
    assert "lanes.dev" in capsys.readouterr().err


def test_resolved_config_never_exposes_the_password(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from omnibase_infra.models.model_lane_admin_client_config import (
        ModelLaneAdminClientConfig,
    )

    monkeypatch.setenv("KAFKA_SASL_USERNAME", "svc")
    monkeypatch.setenv("KAFKA_SASL_PASSWORD", _CRED)
    config = ModelLaneAdminClientConfig.resolve("dev", dict(os.environ))
    for rendered in (
        repr(config),
        str(config),
        str(config.model_dump()),
        config.model_dump_json(),
    ):
        assert _CRED not in rendered
    assert config.librdkafka_config()["sasl.password"] == _CRED


def test_sasl_lane_creates_the_missing_topic_through_the_authenticated_client(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAFKA_SASL_USERNAME", "svc")
    monkeypatch.setenv("KAFKA_SASL_PASSWORD", _CRED)
    code, admin_cls, new_topic = _run_live_main(["--lane", "dev"])
    assert code == 0
    admin_cls.return_value.create_topics.assert_called_once()
    new_topic.assert_called_once()
    assert new_topic.call_args.args == (_TOPIC,)
    assert new_topic.call_args.kwargs["num_partitions"] == 1
    assert new_topic.call_args.kwargs["replication_factor"] == 1


def test_sasl_lane_rerun_is_idempotent(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAFKA_SASL_USERNAME", "svc")
    monkeypatch.setenv("KAFKA_SASL_PASSWORD", _CRED)
    code, admin_cls, new_topic = _run_live_main(["--lane", "dev"], existing=(_TOPIC,))
    assert code == 0
    admin_cls.assert_called_once()
    admin_cls.return_value.create_topics.assert_not_called()
    new_topic.assert_not_called()


@pytest.mark.parametrize("protocol", ["SASL_SSL", "SSL"])
def test_tls_lane_declares_the_admin_protocol(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, protocol: str
) -> None:
    declaration = workspace / "cfg" / "config" / "lab" / "ci_bus_lanes.yaml"
    declaration.write_text(
        "lanes:\n  dev:\n    broker: broker.invalid:19092\n"
        f"    security_protocol: {protocol}\n"
        + ("    sasl_mechanism: SCRAM-SHA-256\n" if protocol == "SASL_SSL" else ""),
        encoding="utf-8",
    )
    monkeypatch.setenv("KAFKA_SASL_USERNAME", "svc")
    monkeypatch.setenv("KAFKA_SASL_PASSWORD", _CRED)
    code, admin_cls, _ = _run_live_main(["--lane", "dev"])
    assert code == 0
    (config,) = admin_cls.call_args.args
    assert config["security.protocol"] == protocol
    if protocol == "SASL_SSL":
        assert config["sasl.password"] == _CRED
        assert config["sasl.mechanism"] == "SCRAM-SHA-256"
    else:
        assert not any(key.startswith("sasl.") for key in config)


@pytest.mark.parametrize("identity", ["stored", "environment"])
def test_sasl_without_mechanism_refuses_before_connecting(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from omnibase_infra.cli.store_lane_credential import StoreLaneCredential

    declaration = workspace / "cfg" / "config" / "lab" / "ci_bus_lanes.yaml"
    declaration.write_text(_OVERLAY.replace("    sasl_mechanism: SCRAM-SHA-256\n", ""))
    if identity == "stored":
        StoreLaneCredential(onex_home=workspace / "home" / ".onex").save(
            lane="dev", sasl_username="stored", sasl_password=_CRED
        )
    else:
        monkeypatch.setenv("KAFKA_SASL_USERNAME", "svc")
        monkeypatch.setenv("KAFKA_SASL_PASSWORD", _CRED)
    code, admin_cls, _ = _run_live_main(["--lane", "dev"])
    assert code == 1
    admin_cls.assert_not_called()
    assert "sasl_mechanism" in capsys.readouterr().err


def test_lane_dry_run_needs_no_identity_or_broker(workspace: Path) -> None:
    code, admin_cls, _ = _run_live_main(["--lane", "dev", "--dry-run"])
    assert code == 0
    admin_cls.assert_not_called()
