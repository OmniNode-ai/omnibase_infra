# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""No resolved catalog bundle may publish one host port from two services (OMN-19497).

docker compose refuses to start the second service that publishes a host port
already published in the same project, after the images are built and half the
stack is up. On omnibase_infra dev at c74a2fb01 the resolved ``runtime`` bundle
published host port 8091 from two services and 8097 from three. These tests
resolve EVERY bundle in docker/catalog/bundles.yaml, so a new bundle or a new
manifest port is covered the day it lands, and they prove the check can fail.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from omnibase_infra.docker.catalog import cli as catalog_cli
from omnibase_infra.docker.catalog.resolver import CatalogResolver
from omnibase_infra.docker.catalog.validator_host_ports import (
    find_duplicate_host_ports,
)

pytestmark = pytest.mark.unit

_CATALOG_DIR = Path(__file__).resolve().parents[3] / "docker" / "catalog"
_BUNDLE_NAMES = sorted(
    yaml.safe_load((_CATALOG_DIR / "bundles.yaml").read_text(encoding="utf-8"))
)


def _manifest(name: str, port: int) -> dict[str, object]:
    return {
        "name": name,
        "description": f"{name} fixture",
        "image": "busybox:1.36",
        "layer": "infrastructure",
        "required_env": [],
        "hardcoded_env": {"FIXTURE": "1"},
        "operational_defaults": {},
        "ports": {"external": port, "internal": 80},
        "healthcheck": None,
        "volumes": [],
        "depends_on": [],
    }


def _write_catalog(root: Path, ports: dict[str, int]) -> Path:
    services = root / "services"
    services.mkdir(parents=True)
    for name, port in ports.items():
        (services / f"{name}.yaml").write_text(
            yaml.safe_dump(_manifest(name, port)), encoding="utf-8"
        )
    (root / "bundles.yaml").write_text(
        yaml.safe_dump(
            {"planted": {"description": "fixture", "services": sorted(ports)}}
        ),
        encoding="utf-8",
    )
    return root


def test_bundle_list_is_not_empty() -> None:
    """Positive control for the parametrised sweep below: it must iterate something."""
    assert "runtime" in _BUNDLE_NAMES
    assert "local" in _BUNDLE_NAMES
    assert len(_BUNDLE_NAMES) > 10


@pytest.mark.parametrize("bundle", _BUNDLE_NAMES)
def test_no_duplicate_host_port_in_resolved_bundle(bundle: str) -> None:
    resolved = CatalogResolver(catalog_dir=str(_CATALOG_DIR)).resolve([bundle])
    result = find_duplicate_host_ports(resolved.manifests)
    assert result.ok, f"bundle {bundle!r}: {result.messages()}"


def test_duplicate_host_port_planted_clash_is_reported(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path / "catalog", {"alpha": 18091, "beta": 18091})
    resolved = CatalogResolver(catalog_dir=str(catalog)).resolve(["planted"])
    result = find_duplicate_host_ports(resolved.manifests)
    assert not result.ok
    assert result.clashes == {18091: ["alpha", "beta"]}
    assert result.messages() == ["host port 18091 is published by alpha, beta"]


def test_duplicate_host_port_distinct_ports_pass(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path / "catalog", {"alpha": 18091, "beta": 18092})
    resolved = CatalogResolver(catalog_dir=str(catalog)).resolve(["planted"])
    assert find_duplicate_host_ports(resolved.manifests).ok


def test_duplicate_host_port_cli_generate_refuses_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog = _write_catalog(tmp_path / "catalog", {"alpha": 18091, "beta": 18091})
    monkeypatch.setattr(catalog_cli, "_CATALOG_DIR", str(catalog))
    # The CLI loads env files into os.environ; keep that inside this test.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    env_file = tmp_path / "stack.env"
    env_file.write_text("FIXTURE_ONLY=1\n", encoding="utf-8")
    output = tmp_path / "compose.yml"

    rc = catalog_cli.cmd_generate(
        ["planted", "--env-file", str(env_file), "--output", str(output)]
    )

    assert rc == 1
    assert not output.exists()
    assert "host port 18091 is published by alpha, beta" in capsys.readouterr().err
