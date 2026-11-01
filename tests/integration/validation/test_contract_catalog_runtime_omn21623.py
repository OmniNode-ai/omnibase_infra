# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Load the shipped verification and validation contracts in a fresh runtime."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "first_import",
    [
        "omnibase_infra.models.validation.model_check_catalog_entry",
        "omnibase_infra.validation.checks.registry_infra_check",
    ],
)
def test_contract_catalog_runtime(first_import: str) -> None:
    """Either public import order builds executors that can pass and fail."""
    script = """
import asyncio
import importlib
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

importlib.import_module(sys.argv[1])

from omnibase_infra.models.validation.model_check_catalog_entry import ModelCheckCatalogEntry
from omnibase_infra.models.verification.model_verification_probe_settings import load_probe_settings
from omnibase_infra.nodes.node_validation_orchestrator.models.model_pattern_candidate import ModelPatternCandidate
from omnibase_infra.validation.checks.handler_check_executor import ModelCheckExecutorConfig
from omnibase_infra.validation.checks.registry_infra_check import CHECK_CATALOG_ORDER, CHECK_REGISTRY, get_check_executor

settings = load_probe_settings()
assert settings.rpk_command_timeout_seconds == 10.0
assert settings.rpk_discovery_timeout_seconds == 5.0
entries = ModelCheckCatalogEntry.load_catalog()
assert tuple(entry.check_code for entry in entries) == CHECK_CATALOG_ORDER
assert len(CHECK_REGISTRY) == len(entries) == 12
executor = get_check_executor('CHECK-RISK-001')
assert executor is not None
candidate = ModelPatternCandidate(
    candidate_id=uuid4(), pattern_id=uuid4(), source_path='.',
    changed_files=('security/access.py',),
)
config = ModelCheckExecutorConfig()
failed = asyncio.run(executor.execute(candidate, config))
assert not failed.passed and not failed.skipped
passed = asyncio.run(executor.execute(candidate.model_copy(update={'risk_tags': ('security',)}), config))
assert passed.passed and not passed.skipped
with TemporaryDirectory() as directory:
    artifact_dir = Path(directory)
    artifact_executor = get_check_executor('CHECK-VAL-002', artifact_dir=artifact_dir)
    assert artifact_executor is not None
    missing = asyncio.run(artifact_executor.execute(candidate, config))
    assert not missing.passed and not missing.skipped
    for name in ('result.yaml', 'verdict.yaml'):
        (artifact_dir / name).write_text('{}', encoding='utf-8')
    complete = asyncio.run(artifact_executor.execute(candidate, config))
    assert complete.passed and not complete.skipped
    default = asyncio.run(CHECK_REGISTRY['CHECK-VAL-002'].execute(candidate, config))
    assert default.skipped
print('contract-catalog-runtime: 12 executors, risk rejection and acceptance verified')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, first_import],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "risk rejection and acceptance verified" in completed.stdout
