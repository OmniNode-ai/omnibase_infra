# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Offline synthetic harness tests: no source or target connection."""

import base64
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.crypto import generate_keypair
from omnibase_core.models.envelope.model_message_envelope import ModelMessageEnvelope
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from scripts.runtime_build import rehydrate_sim_preflight_selected_chain as harness
from tests.unit.runtime.test_sim_archive_source_receipt_omn19728 import (
    _fixture,
    _topology,
)

pytestmark = pytest.mark.unit


def _inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> harness.ModelSelectedChainHarnessConfig:
    monkeypatch.setenv("ONEX_RUNTIME_LANE", "sim-202")
    authority, _, records, _, _, _ = _fixture()
    archive = tmp_path / "archive.tar"
    with tarfile.open(archive, "w") as output:
        for coordinate, record in zip(
            harness._SELECTED_COORDINATES, records, strict=True
        ):
            topic, partition, offset = coordinate
            raw = {
                "topic": topic,
                "partition": partition,
                "offset": offset,
                "timestamp_ms": record.timestamp_ms,
                "key_b64": base64.b64encode(record.key).decode()
                if record.key
                else None,
                "value_b64": base64.b64encode(record.value).decode(),
                "headers": [
                    [
                        key,
                        base64.b64encode(value).decode() if value is not None else None,
                    ]
                    for key, value in (
                        *record.headers,
                        ("duplicate", b"1"),
                        ("duplicate", b"2"),
                    )
                ],
            }
            body = gzip.compress(json.dumps(raw).encode() + b"\n")
            member = tarfile.TarInfo(f"{topic}/partition=0/selected.jsonl.gz")
            member.size = len(body)
            output.addfile(member, io.BytesIO(body))
    monkeypatch.setattr(harness, "_APPROVED_ARCHIVE_SHA256", harness._sha256(archive))
    monkeypatch.setattr(
        harness,
        "_APPROVED_CORRELATION_SHA256",
        hashlib.sha256(str(authority.correlation_id).encode()).hexdigest(),
    )
    key = generate_keypair()
    keymap = tmp_path / "gateway-keys.json"
    keymap.write_text(
        json.dumps(
            {
                "keys": {
                    "gateway-test": base64.urlsafe_b64encode(
                        key.public_key_bytes
                    ).decode()
                }
            }
        )
    )
    keymap.chmod(0o600)
    inner = ModelEventEnvelope[dict[str, object]](
        tenant_id=str(authority.tenant_id),
        correlation_id=authority.correlation_id,
        metadata={
            "tags": {
                "workflow_id": str(uuid4()),
                "workflow_type": "delegation-execution-graph-read",
                "contract_id": "node_execution_graph_read_effect:1.0.0",
            }
        },
        event_type="omnibase-infra.delegation-execution-graph-requested",
        payload={
            "correlation_id": str(authority.correlation_id),
            "cursor_mode": "latest",
            "source_cursors": None,
        },
    )
    signed = ModelMessageEnvelope[dict[str, object]].create_signed(
        realm="test",
        runtime_id="gateway-test",
        bus_id="test-bus",
        trace_id=authority.correlation_id,
        tenant_id=str(authority.tenant_id),
        payload=inner.model_dump(mode="json"),
        private_key=key.private_key_bytes,
    )
    command = tmp_path / "actual-signed-command.json"
    command.write_text(signed.model_dump_json())
    row = tmp_path / "owner.copy"
    row.write_bytes(b"test-only-owner-binary")
    metadata = tmp_path / "owner.metadata"
    metadata.write_text(
        f"correlation_id={authority.correlation_id}\ntenant_id={authority.tenant_id}\nsource_db=omnidash_analytics\nsource_table=public.delegation_events\nrow_sha256={harness._sha256(row)}\nschema_sha256={'a' * 64}\n"
    )
    return harness.ModelSelectedChainHarnessConfig.model_validate(
        {
            "archive_path": archive,
            "signed_command_path": command,
            "owner_metadata_path": metadata,
            "owner_row_path": row,
            "gateway": {
                "command_topic": harness._COMMAND_TOPIC,
                "runtime_id": "gateway-test",
                "realm": "test",
                "bus_id": "test-bus",
                "public_key_path": keymap,
            },
            "rehydration": {
                "source_databases": {
                    "analytics_dsn": "postgresql://readonly@source.example/omnidash_analytics",
                    "ledger_dsn": "postgresql://readonly@source.example/omnibase_infra",
                },
                "source_kafka": {
                    "bootstrap_servers": "source.example:9092",
                    "environment": "source",
                },
                "source_topic_namespace": "",
                "target_ledger_dsn": "postgresql://target@127.0.0.1:65036/omnibase_infra",
                "topology_version": _topology().version,
            },
        }
    )


@pytest.mark.asyncio
async def test_default_verifies_signed_authority_without_enqueue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _inputs(tmp_path, monkeypatch)
    verify, execute = AsyncMock(return_value=5), AsyncMock(return_value=5)
    monkeypatch.setattr(harness, "verify_source_only", verify)
    monkeypatch.setattr(harness, "rehydrate_verified_sim_preflight_chain", execute)
    result = await harness.run(config)
    assert result["mode"] == "verify-only" and result["records"] == 5
    execute.assert_not_called()
    verify.assert_awaited_once()
    records = verify.await_args.args[2]
    assert tuple(r.source_key for r in records) == harness._SELECTED_COORDINATES
    assert records[0].headers[-2:] == (("duplicate", b"1"), ("duplicate", b"2"))


@pytest.mark.asyncio
async def test_execute_is_explicit_and_passes_only_five(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _inputs(tmp_path, monkeypatch)
    execute = AsyncMock(return_value=5)
    monkeypatch.setattr(harness, "rehydrate_verified_sim_preflight_chain", execute)
    assert (await harness.run(config, execute=True))["records"] == 5
    assert len(execute.await_args.kwargs["archive_records"]) == 5


@pytest.mark.asyncio
async def test_tampered_command_is_refused_before_any_source_or_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _inputs(tmp_path, monkeypatch)
    raw = json.loads(config.signed_command_path.read_text())
    raw["tenant_id"] = str(uuid4())
    config.signed_command_path.write_text(json.dumps(raw))
    verify = AsyncMock()
    monkeypatch.setattr(harness, "verify_source_only", verify)
    with pytest.raises(PermissionError):
        await harness.run(config)
    verify.assert_not_called()


def test_wrong_capture_hash_and_owner_binary_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _inputs(tmp_path, monkeypatch)
    config.owner_row_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="owner capture"):
        harness.corroborate_owner_capture(config, harness.load_signed_authority(config))
    config.archive_path.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="approved capture"):
        harness.load_selected_archive(config.archive_path)


def test_main_never_prints_private_configuration_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "bad-private-config.json"
    config.write_text('{"password":"private-test-value"}')
    monkeypatch.setattr("sys.argv", ["harness", "--config", str(config)])
    assert harness.main() == 1
    assert "private-test-value" not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_verification_phase_opens_only_readonly_source_pools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _inputs(tmp_path, monkeypatch)
    pools = [MagicMock(), MagicMock()]
    for pool in pools:
        pool.close = AsyncMock()
    create_pool = AsyncMock(side_effect=pools)
    verifier = MagicMock()
    verifier.verify = AsyncMock(return_value=(object(),) * 5)
    monkeypatch.setattr(harness.asyncpg, "create_pool", create_pool)
    monkeypatch.setattr(
        harness, "SimArchiveSourceReceiptVerifier", MagicMock(return_value=verifier)
    )
    authority = harness.load_signed_authority(config)
    records = harness.load_selected_archive(config.archive_path)
    assert await harness.verify_source_only(config, authority, records) == 5
    assert create_pool.await_count == 2
    for call in create_pool.await_args_list:
        assert "source.example" in call.kwargs["dsn"]
        assert call.kwargs["server_settings"] == {"default_transaction_read_only": "on"}
    for pool in pools:
        pool.close.assert_awaited_once()
