# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Prepare a private, typed config for selected-chain source verification."""

from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import ValidationError

from omnibase_core.models.execution_graph_replay.model_execution_graph_topology_version import (
    ModelExecutionGraphTopologyVersion,
)
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.runtime.models.model_execution_graph_trusted_gateway_config import (
    ModelExecutionGraphTrustedGatewayConfig,
)
from omnibase_infra.runtime.sim_preflight_rehydration import (
    ModelSimPreflightRehydrationConfig,
)
from scripts.runtime_build.rehydrate_sim_preflight_selected_chain import (
    ModelSelectedChainHarnessConfig,
)

OUTPUT_NAME = "selected-chain-rehydration.json"
GATEWAY_COMMAND_TOPIC = (
    "onex.cmd.omnibase-infra.delegation-execution-graph-requested.v1"
)
SOURCE_BROKER = "192.168.86.201:19092"  # kafka-fallback-ok: operator-approved read-only source capture; fixed target, no fallback.
TARGET_DSN_TEMPLATE = "postgresql://postgres:{password}@127.0.0.1:65036/omnibase_infra"


def _private_file(path: Path) -> Path:
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != 0o600
    ):
        raise ValueError("private input must be an absolute mode-0600 regular file")
    return path


def _private_directory(path: Path) -> Path:
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_dir()
        or stat.S_IMODE(path.stat().st_mode) != 0o700
    ):
        raise ValueError("private input directory must be absolute mode 0700")
    return path


def _read_env(path: Path, required: frozenset[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in _private_file(path).read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in fields:
            raise ValueError("private env input is malformed")
        fields[key] = value
    if not required <= fields.keys():
        raise ValueError("private env input lacks required fields")
    if any(not fields[key] for key in required):
        raise ValueError("required private env values must be nonempty")
    return fields


def _topology_version() -> ModelExecutionGraphTopologyVersion:
    root = Path(__file__).resolve().parents[2]
    snapshots = sorted(
        (root / "src/omnibase_infra/runtime/execution_graph_topologies").glob("*.json")
    )
    if len(snapshots) != 1:
        raise ValueError("packaged graph topology must be uniquely declared")
    snapshot: Any = json.loads(snapshots[0].read_text(encoding="utf-8"))
    return ModelExecutionGraphTopologyVersion.model_validate(
        {
            "contract_version": snapshot["contract_version"],
            "topology_sha256": snapshot["topology_sha256"],
        }
    )


def prepare(
    *,
    bundle: Path,
    archive: Path,
    owner_directory: Path,
    signed_command: Path,
    relay_receipt: Path,
) -> Path:
    """Validate existing private inputs and exclusively write typed JSON config."""
    bundle = _private_directory(bundle)
    owner_directory = _private_directory(owner_directory)
    archive = _private_file(archive)
    relay_receipt = _private_file(relay_receipt)
    signed_command = _private_file(signed_command)

    credentials = _read_env(
        bundle / "credentials.env", frozenset({"POSTGRES_PASSWORD"})
    )
    databases = _read_env(
        bundle / "source-database.env",
        frozenset({"SOURCE_ANALYTICS_DSN", "SOURCE_LEDGER_DSN"}),
    )
    readback = _read_env(
        bundle / "source-readback.env",
        frozenset({"READBACK_SASL_USERNAME", "READBACK_SASL_PASSWORD"}),
    )
    keymap = _private_file(bundle / "gateway-keys.json")
    owner_metadata = _private_file(
        owner_directory / "delegation-events-owner-row.metadata"
    )
    owner_row = _private_file(owner_directory / "delegation-events-owner-row.copy")

    kafka = ModelKafkaEventBusConfig.model_validate(
        {
            "bootstrap_servers": SOURCE_BROKER,
            "environment": "source",
            "security_protocol": "SASL_PLAINTEXT",
            "sasl_mechanism": "SCRAM-SHA-256",
            "sasl_plain_username": readback["READBACK_SASL_USERNAME"],
            "sasl_plain_password": readback["READBACK_SASL_PASSWORD"],
            "enable_auto_commit": False,
            "auto_offset_reset": "earliest",
        }
    )
    rehydration = ModelSimPreflightRehydrationConfig.model_validate(
        {
            "source_databases": {
                "analytics_dsn": databases["SOURCE_ANALYTICS_DSN"],
                "ledger_dsn": databases["SOURCE_LEDGER_DSN"],
            },
            "source_kafka": kafka,
            "source_topic_namespace": "",
            "target_ledger_dsn": TARGET_DSN_TEMPLATE.format(
                password=quote(credentials["POSTGRES_PASSWORD"], safe="")
            ),
            "topology_version": _topology_version(),
            "target_relay_receipt": relay_receipt,
        }
    )
    config = ModelSelectedChainHarnessConfig.model_validate(
        {
            "archive_path": archive,
            "signed_command_path": signed_command,
            "owner_metadata_path": owner_metadata,
            "owner_row_path": owner_row,
            "gateway": ModelExecutionGraphTrustedGatewayConfig(
                command_topic=GATEWAY_COMMAND_TOPIC,
                runtime_id="onex-api-sim-preflight",
                realm="sim-preflight",
                bus_id="sim-preflight",
                public_key_path=keymap,
            ),
            "rehydration": rehydration,
        }
    )
    target = bundle / OUTPUT_NAME
    if target.exists() or target.is_symlink():
        raise ValueError("rehydration config output must not already exist")
    document = _serialize(config, kafka, rehydration)
    descriptor = os.open(
        target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(document, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return target


def _serialize(
    config: ModelSelectedChainHarnessConfig,
    kafka: ModelKafkaEventBusConfig,
    rehydration: ModelSimPreflightRehydrationConfig,
) -> dict[str, Any]:
    """Serialize secret fields for the private file; never print this document."""
    return {
        "archive_path": str(config.archive_path),
        "signed_command_path": str(config.signed_command_path),
        "owner_metadata_path": str(config.owner_metadata_path),
        "owner_row_path": str(config.owner_row_path),
        "gateway": config.gateway.model_dump(mode="json"),
        "rehydration": {
            "source_databases": {
                "analytics_dsn": rehydration.source_databases.analytics_dsn.get_secret_value(),
                "ledger_dsn": rehydration.source_databases.ledger_dsn.get_secret_value(),
            },
            "source_kafka": {
                "bootstrap_servers": kafka.bootstrap_servers,
                "environment": kafka.environment,
                "security_protocol": kafka.security_protocol,
                "sasl_mechanism": kafka.sasl_mechanism,
                "sasl_plain_username": kafka.sasl_plain_username,
                "sasl_plain_password": kafka.sasl_plain_password,
                "enable_auto_commit": kafka.enable_auto_commit,
                "auto_offset_reset": kafka.auto_offset_reset,
            },
            "source_topic_namespace": rehydration.source_topic_namespace,
            "target_ledger_dsn": rehydration.target_ledger_dsn.get_secret_value(),
            "topology_version": rehydration.topology_version.model_dump(mode="json"),
            "target_relay_receipt": str(rehydration.target_relay_receipt),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--owner-directory", type=Path, required=True)
    parser.add_argument("--signed-command", type=Path, required=True)
    parser.add_argument("--relay-receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = prepare(
            bundle=args.bundle,
            archive=args.archive,
            owner_directory=args.owner_directory,
            signed_command=args.signed_command,
            relay_receipt=args.relay_receipt,
        )
    except (
        OSError,
        ValueError,
        ValidationError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(json.dumps({"status": "refused", "error_type": type(exc).__name__}))
        return 65
    print(json.dumps({"status": "prepared", "config": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
