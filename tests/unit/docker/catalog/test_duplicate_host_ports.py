# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Check rendered host ports at the OMN-19497 acceptance-command path.

The existing infra suite checks the validator and CLI. These tests independently
inspect generated compose ports and exercise the compose writer's refusal.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from omnibase_infra.docker.catalog.generator import generate_compose, write_compose
from omnibase_infra.docker.catalog.resolver import CatalogResolver
from omnibase_infra.errors import ProtocolConfigurationError

pytestmark = pytest.mark.unit

_CATALOG_DIR = Path(__file__).resolve().parents[4] / "docker" / "catalog"


def test_duplicate_host_port_absent_from_every_rendered_bundle() -> None:
    bundles = yaml.safe_load((_CATALOG_DIR / "bundles.yaml").read_text())
    assert {"runtime", "local"} <= bundles.keys()
    resolver = CatalogResolver(catalog_dir=str(_CATALOG_DIR))
    for bundle in sorted(bundles):
        resolved = resolver.resolve([bundle])
        services = generate_compose(resolved)["services"]
        assert isinstance(services, dict)
        claimants: dict[int, str] = {}
        for name, service in services.items():
            for binding in service.get("ports", []):
                # The short compose form is [host-address:]host-port:target-port.
                host_port = int(binding.rsplit(":", 2)[-2])
                assert host_port not in claimants, (
                    f"bundle {bundle!r}: host port {host_port} is published by "
                    f"{claimants.get(host_port)} and {name}"
                )
                claimants[host_port] = name


def test_duplicate_host_port_writer_refuses_planted_clash(tmp_path: Path) -> None:
    catalog = Path(shutil.copytree(_CATALOG_DIR, tmp_path / "catalog"))
    resolved = CatalogResolver(catalog_dir=str(catalog)).resolve(["runtime"])
    ported = [manifest for manifest in resolved.manifests.values() if manifest.ports]
    assert len(ported) >= 2
    first, second = ported[:2]
    assert first.ports is not None
    manifest_file = catalog / "services" / f"{second.name}.yaml"
    manifest = yaml.safe_load(manifest_file.read_text())
    manifest["ports"]["external"] = first.ports.external
    manifest_file.write_text(yaml.safe_dump(manifest))
    output = tmp_path / "compose.yml"
    output.write_text("existing compose output\n")

    with pytest.raises(ProtocolConfigurationError, match="duplicate host ports"):
        write_compose(
            ["runtime"], catalog_dir=catalog, output_path=output, environment={}
        )

    assert output.read_text() == "existing compose output\n"
