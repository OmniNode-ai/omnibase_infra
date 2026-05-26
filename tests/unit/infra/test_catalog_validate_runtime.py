# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for cmd_validate_runtime — OMN-7603."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

import omnibase_infra.docker.catalog.cli as catalog_cli
from omnibase_infra.docker.catalog.cli import cmd_validate_runtime
from omnibase_infra.docker.catalog.resolver import CatalogResolver

CATALOG_DIR = str(Path(__file__).resolve().parents[3] / "docker" / "catalog")


@pytest.mark.unit
def test_validate_runtime_passes_on_core_bundle(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Core bundle must pass structural validation with no errors."""
    with patch.object(catalog_cli, "_CATALOG_DIR", CATALOG_DIR):
        rc = cmd_validate_runtime(["core"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "OK" in out


@pytest.mark.unit
def test_validate_runtime_passes_on_runtime_bundle(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Full runtime bundle must pass validate-runtime without errors."""
    with patch.object(catalog_cli, "_CATALOG_DIR", CATALOG_DIR):
        rc = cmd_validate_runtime(["runtime"])
    assert rc == 0


@pytest.mark.unit
def test_validate_runtime_consumer_health_emitter_flag_set() -> None:
    """consumer-health-projection must declare ENABLE_CONSUMER_HEALTH_EMITTER=true."""
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime-observability-projections"])
    manifest = resolved.manifests.get("consumer-health-projection")
    assert manifest is not None, "consumer-health-projection not found in bundle"
    assert manifest.operational_defaults.get("ENABLE_CONSUMER_HEALTH_EMITTER") == "true"


@pytest.mark.unit
def test_validate_runtime_detects_empty_hardcoded_env_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Empty hardcoded_env value is flagged as an error."""
    services_dir = tmp_path / "services"
    services_dir.mkdir()
    (services_dir / "bad-svc.yaml").write_text(
        textwrap.dedent("""\
            name: bad-svc
            description: test service with empty hardcoded value
            image: test:latest
            layer: infrastructure
            hardcoded_env:
              SOME_KEY: ''
        """)
    )
    (tmp_path / "bundles.yaml").write_text(
        yaml.dump({"test-bundle": {"description": "test", "services": ["bad-svc"]}})
    )

    with patch.object(catalog_cli, "_CATALOG_DIR", str(tmp_path)):
        rc = cmd_validate_runtime(["test-bundle"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "SOME_KEY" in err
    assert "empty" in err.lower()


@pytest.mark.unit
def test_validate_runtime_detects_key_in_both_hardcoded_and_required(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A key in both hardcoded_env and required_env is a catalog authoring error."""
    services_dir = tmp_path / "services"
    services_dir.mkdir()
    (services_dir / "conflict-svc.yaml").write_text(
        textwrap.dedent("""\
            name: conflict-svc
            description: service with conflicting env declarations
            image: test:latest
            layer: infrastructure
            required_env:
              - SHARED_KEY
            hardcoded_env:
              SHARED_KEY: 'hardcoded-value'
        """)
    )
    (tmp_path / "bundles.yaml").write_text(
        yaml.dump(
            {"conflict-bundle": {"description": "test", "services": ["conflict-svc"]}}
        )
    )

    with patch.object(catalog_cli, "_CATALOG_DIR", str(tmp_path)):
        rc = cmd_validate_runtime(["conflict-bundle"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "SHARED_KEY" in err
    assert "hardcoded_env" in err and "required_env" in err
