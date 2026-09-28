# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Offline tests for private selected-chain config preparation."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.runtime_build.prepare_sim_preflight_rehydration_config import (
    OUTPUT_NAME,
    SOURCE_BROKER,
    TARGET_DSN_TEMPLATE,
    prepare,
)
from scripts.runtime_build.rehydrate_sim_preflight_selected_chain import (
    ModelSelectedChainHarnessConfig,
)


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Path]:
    bundle = tmp_path / "bundle"
    bundle.mkdir(mode=0o700)
    bundle.chmod(0o700)
    owner = tmp_path / "owner"
    owner.mkdir(mode=0o700)
    owner.chmod(0o700)
    archive = tmp_path / "archive.tar"
    archive.write_bytes(b"private archive fixture")
    receipt = tmp_path / "relay.json"
    receipt.write_text("{}")
    for path in (archive, receipt):
        path.chmod(0o600)
    keymap = bundle / "gateway-keys.json"
    keymap.write_text('{"keys":{}}')
    keymap.chmod(0o600)
    signed = bundle / "signed-command.json"
    signed.write_text('{"captured":"signed-wire-fixture"}')
    signed.chmod(0o600)
    files = {
        "credentials.env": "POSTGRES_PASSWORD=unit-target-password\n",
        "source-database.env": (
            "SOURCE_ANALYTICS_DSN=postgresql://readonly:sourcepass@source.example:5432/omnidash_analytics\n"
            "SOURCE_LEDGER_DSN=postgresql://readonly:sourcepass@source.example:5432/omnibase_infra\n"
        ),
        "source-readback.env": (
            "READBACK_SASL_USERNAME=unit-readback-user\n"
            "READBACK_SASL_PASSWORD=unit-readback-password\n"
        ),
    }
    for name, contents in files.items():
        path = bundle / name
        path.write_text(contents)
        path.chmod(0o600)
    for name, contents in (
        ("delegation-events-owner-row.metadata", "unit metadata\n"),
        ("delegation-events-owner-row.copy", "unit row"),
    ):
        path = owner / name
        path.write_text(contents)
        path.chmod(0o600)
    return {
        "bundle": bundle,
        "archive": archive,
        "owner": owner,
        "signed": signed,
        "receipt": receipt,
    }


def _prepare(inputs: dict[str, Path]) -> Path:
    return prepare(
        bundle=inputs["bundle"],
        archive=inputs["archive"],
        owner_directory=inputs["owner"],
        signed_command=inputs["signed"],
        relay_receipt=inputs["receipt"],
    )


def test_prepares_mode_0600_typed_private_config_without_logging_secrets(
    inputs: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    output = _prepare(inputs)

    assert output == inputs["bundle"] / OUTPUT_NAME
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    document = json.loads(output.read_text())
    typed = ModelSelectedChainHarnessConfig.model_validate(document)
    assert typed.gateway.command_topic == (
        "onex.cmd.omnibase-infra.delegation-execution-graph-requested.v1"
    )
    assert typed.rehydration.source_kafka.bootstrap_servers == SOURCE_BROKER
    assert typed.rehydration.source_topic_namespace == ""
    assert typed.rehydration.target_ledger_dsn.get_secret_value() == (
        TARGET_DSN_TEMPLATE.format(password="unit-target-password")
    )
    assert document["rehydration"]["source_kafka"]["sasl_plain_password"] == (
        "unit-readback-password"
    )
    assert "unit-target-password" not in capsys.readouterr().out


def test_refuses_overwrite_without_changing_existing_config(
    inputs: dict[str, Path],
) -> None:
    existing = inputs["bundle"] / OUTPUT_NAME
    existing.write_text("do not replace")
    existing.chmod(0o600)

    with pytest.raises(ValueError, match="must not already exist"):
        _prepare(inputs)

    assert existing.read_text() == "do not replace"


def test_typed_model_rejects_non_disposable_target(
    inputs: dict[str, Path],
) -> None:
    document = json.loads(_prepare(inputs).read_text())
    document["rehydration"]["target_ledger_dsn"] = (
        "postgresql://postgres:unit-target-password@127.0.0.1:65085/omnibase_infra"
    )

    with pytest.raises(ValidationError):
        ModelSelectedChainHarnessConfig.model_validate(document)


def test_refuses_input_not_mode_0600(inputs: dict[str, Path]) -> None:
    source = inputs["bundle"] / "source-readback.env"
    source.chmod(0o640)

    with pytest.raises(ValueError, match="mode-0600"):
        _prepare(inputs)
