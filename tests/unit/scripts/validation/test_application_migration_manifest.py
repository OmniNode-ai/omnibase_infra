# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""RED/green proof for the deterministic application migration manifest."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
VALIDATOR_PATH = (
    REPO_ROOT / "scripts" / "validation" / "validate_application_migration_manifest.py"
)
MIGRATIONS_DIR = REPO_ROOT / "docker" / "migrations" / "forward"
LEDGER_DIR = MIGRATIONS_DIR / "_ledger"


def _load_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "validate_application_migration_manifest", VALIDATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


def _validate(
    migrations_dir: Path = MIGRATIONS_DIR,
    ledger_dir: Path = LEDGER_DIR,
    *,
    require_complete: bool = False,
) -> Any:
    return validator.validate_manifests(
        migrations_dir,
        ledger_dir / "application-migrations.tsv",
        ledger_dir / "application-migration-blocks.tsv",
        ledger_dir / "cloud-migration-aliases.tsv",
        require_complete=require_complete,
    )


def _minimal_fixture(tmp_path: Path) -> tuple[Path, Path]:
    migrations_dir = tmp_path / "forward"
    ledger_dir = migrations_dir / "_ledger"
    artifact = migrations_dir / "nodes" / "node_example" / "0001.sql"
    artifact.parent.mkdir(parents=True)
    ledger_dir.mkdir(parents=True)
    artifact.write_text("SELECT 1;\n", encoding="utf-8")
    checksum = validator._content_sha256(artifact)
    (ledger_dir / "application-migrations.tsv").write_text(
        "\t".join(
            (
                "nodes/node_example/0001.sql",
                "node:node_example",
                "node:node_example",
                "omninode_internal",
                "node:node_example:0001.sql",
                checksum,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    blocked_artifact = migrations_dir / "nodes" / "node_example" / "0002.sql"
    blocked_artifact.write_text("SELECT 2;\n", encoding="utf-8")
    blocked_checksum = validator._content_sha256(blocked_artifact)
    (ledger_dir / "application-migration-blocks.tsv").write_text(
        "\t".join(
            (
                "nodes/node_example/0002.sql",
                "node:node_example:0002.sql",
                blocked_checksum,
                "OMN-99999",
                "classification pending",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (ledger_dir / "cloud-migration-aliases.tsv").write_text(
        "20260101_example\t20260101_example.sql\n", encoding="utf-8"
    )
    return migrations_dir, ledger_dir


def test_checked_in_manifest_is_exact_and_all_blockers_are_explicit() -> None:
    result = _validate()

    # 96 as of the OMN-15337 vendor-parity repair: the manifest mirrors the
    # vendored node migration tree after restoring the nine house-tenant RLS
    # migrations that are still source-owned in omnimarket dev.
    assert len(result.declarations) == 96
    assert result.blocked == ()
    assert len(result.cloud_aliases) == 30


def test_completion_gate_is_green_after_domain_classification_is_complete() -> None:
    result = _validate(require_complete=True)

    assert result.blocked == ()


def test_empty_block_set_is_valid_after_all_artifacts_are_classified(
    tmp_path: Path,
) -> None:
    migrations_dir, ledger_dir = _minimal_fixture(tmp_path)
    (migrations_dir / "nodes" / "node_example" / "0002.sql").unlink()
    (ledger_dir / "application-migration-blocks.tsv").write_text("", encoding="utf-8")

    result = _validate(migrations_dir, ledger_dir, require_complete=True)

    assert result.blocked == ()


@pytest.mark.parametrize(
    ("field_index", "replacement", "signature"),
    [
        (1, "unknown-stream", "unknown migration stream"),
        (3, "unknown_domain", "unknown domain"),
        (5, "0" * 64, "conflicting checksum"),
    ],
)
def test_active_declaration_drift_fails_closed(
    tmp_path: Path,
    field_index: int,
    replacement: str,
    signature: str,
) -> None:
    migrations_dir, ledger_dir = _minimal_fixture(tmp_path)
    declaration_path = ledger_dir / "application-migrations.tsv"
    fields = declaration_path.read_text(encoding="utf-8").strip().split("\t")
    fields[field_index] = replacement
    declaration_path.write_text("\t".join(fields) + "\n", encoding="utf-8")

    with pytest.raises(validator.ManifestError, match=signature):
        _validate(migrations_dir, ledger_dir)


def test_active_and_blocked_double_declaration_fails_closed(tmp_path: Path) -> None:
    migrations_dir, ledger_dir = _minimal_fixture(tmp_path)
    active_path = ledger_dir / "application-migrations.tsv"
    active_fields = active_path.read_text(encoding="utf-8").strip().split("\t")
    blocked_path = ledger_dir / "application-migration-blocks.tsv"
    blocked_path.write_text(
        "\t".join(
            (
                active_fields[0],
                active_fields[4],
                active_fields[5],
                "OMN-99999",
                "duplicate declaration RED",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(validator.ManifestError, match="double migration declaration"):
        _validate(migrations_dir, ledger_dir)


def test_duplicate_cloud_alias_fails_closed(tmp_path: Path) -> None:
    migrations_dir, ledger_dir = _minimal_fixture(tmp_path)
    alias_path = ledger_dir / "cloud-migration-aliases.tsv"
    alias_path.write_text(
        "20260101_example\t20260101_example.sql\n"
        "20260101_example\t20260102_other.sql\n",
        encoding="utf-8",
    )

    with pytest.raises(
        validator.ManifestError, match="duplicate cloud migration alias"
    ):
        _validate(migrations_dir, ledger_dir)
