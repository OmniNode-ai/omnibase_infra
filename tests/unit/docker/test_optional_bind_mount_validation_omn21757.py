# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Optional directory bind mount declarations are validated strictly (OMN-21757)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.docker.catalog.model_optional_directory_bind_mount import (
    ModelOptionalDirectoryBindMount,
)
from omnibase_infra.docker.catalog.resolver import _load_manifest

_GOOD: dict[str, Any] = {
    "source_env": "SRC",
    "container_path": "/data",
    "absent_source": "/var/empty",
}


def _manifest(tmp_path: Path, mount: object) -> Path:
    path = tmp_path / "svc.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "svc",
                "description": "d",
                "image": "i",
                "optional_directory_bind_mounts": [mount],
            }
        )
    )
    return path


@pytest.mark.unit
def test_good_mount_defaults_read_only() -> None:
    assert ModelOptionalDirectoryBindMount(**_GOOD).read_only is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "override",
    [
        {"read_only": "yes"},
        {"source_env": True},
        {"source_env": ""},
        {"container_path": "data"},
        {"absent_source": "relative/dir"},
        {"absent_source": 5},
    ],
)
def test_model_refuses_bad_fields(override: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ModelOptionalDirectoryBindMount(**{**_GOOD, **override})


@pytest.mark.unit
@pytest.mark.parametrize(
    "mount",
    [
        {k: v for k, v in _GOOD.items() if k != "absent_source"},
        {**_GOOD, "extra": 1},
        {**_GOOD, "absent_source": "relative"},
        {**_GOOD, "read_only": "no"},
        "not-a-mapping",
    ],
)
def test_loader_raises_value_error(tmp_path: Path, mount: object) -> None:
    with pytest.raises(ValueError):
        _load_manifest(_manifest(tmp_path, mount))
