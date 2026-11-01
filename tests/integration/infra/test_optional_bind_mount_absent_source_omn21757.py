# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The optional bind-mount fallback comes from the shipped catalog manifest (OMN-21757).

Hermetic: loads the real ``docker/catalog`` tree through the resolver and renders
compose with the generator, so the declared ``absent_source`` is proven to travel
from the manifest to the rendered volume, and a malformed declaration is refused.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.docker.catalog.generator import generate_compose
from omnibase_infra.docker.catalog.resolver import CatalogResolver, _load_manifest

_CATALOG_DIR = Path(__file__).resolve().parents[3] / "docker" / "catalog"
_MANIFEST = _CATALOG_DIR / "services" / "runtime-effects.yaml"


@pytest.mark.integration
def test_rendered_volume_uses_the_manifest_declared_absent_source() -> None:
    declared = yaml.safe_load(_MANIFEST.read_text())["optional_directory_bind_mounts"]
    assert declared, "runtime-effects must declare an optional directory bind mount"
    mount = declared[0]
    resolver = CatalogResolver(catalog_dir=str(_CATALOG_DIR))
    compose = generate_compose(
        resolver.resolve(bundles=["runtime-core"]), environment={}
    )
    volumes = compose["services"]["runtime-effects"]["volumes"]
    expected = (
        f"${{{mount['source_env']}:-{mount['absent_source']}}}"
        f":{mount['container_path']}:ro"
    )
    assert expected in volumes


def _manifest_with_mount(tmp_path: Path, mount: dict[str, object]) -> Path:
    manifest = yaml.safe_load(_MANIFEST.read_text())
    manifest["optional_directory_bind_mounts"] = [mount]
    path = tmp_path / "runtime-effects.yaml"
    path.write_text(yaml.safe_dump(manifest))
    return path


@pytest.mark.integration
@pytest.mark.parametrize(
    "mount",
    [
        {"source_env": "X", "container_path": "/c"},
        {"source_env": "X", "container_path": "/c", "absent_source": "rel"},
        {"source_env": "X", "container_path": "/c", "absent_source": "/a", "bogus": 1},
        {
            "source_env": "X",
            "container_path": "/c",
            "absent_source": "/a",
            "read_only": "yes",
        },
        {"source_env": 3, "container_path": "/c", "absent_source": "/a"},
    ],
)
def test_malformed_mount_declaration_is_refused_with_value_error(
    tmp_path: Path, mount: dict[str, object]
) -> None:
    path = _manifest_with_mount(tmp_path, mount)
    with pytest.raises(ValueError):
        _load_manifest(path)


@pytest.mark.integration
def test_the_shipped_manifest_loads_through_the_same_loader(tmp_path: Path) -> None:
    path = _manifest_with_mount(
        tmp_path,
        {"source_env": "X", "container_path": "/c", "absent_source": "/a"},
    )
    (loaded,) = _load_manifest(path).optional_directory_bind_mounts
    assert loaded.absent_source == "/a"
