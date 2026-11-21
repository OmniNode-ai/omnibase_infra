# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The image-build step over the real installed packages (OMN-26361).

Runs :func:`build_prebuilt_manifest` against the packages installed in this
environment, writes the one stamped file, reads it back, and holds the recorded
boot measurement to what that file serves.
"""

from __future__ import annotations

import json
from pathlib import Path

from omnibase_infra.models.node_announce.model_prebuilt_node_manifest import (
    ModelPrebuiltNodeManifest,
)
from omnibase_infra.nodes.node_boot_catalog_effect.runtime_prebuilt_manifest import (
    build_prebuilt_manifest,
)

_MEASUREMENT = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "omn26361"
    / "boot_measurement.json"
)


def test_prebuilt_manifest_boot_real_packages_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "prebuilt" / "node_manifest.json"
    built = build_prebuilt_manifest(path, workload="main")
    loaded = ModelPrebuiltNodeManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    assert loaded == built
    assert len(loaded.manifest.contracts) > 0


def test_prebuilt_manifest_boot_measurement_is_recorded() -> None:
    d = json.loads(_MEASUREMENT.read_text(encoding="utf-8"))
    assert d["host"] == "h051"
    assert d["after"]["contract_yaml_parses"] == 0
    assert d["before"]["contract_yaml_parses"] > 0
    assert d["read_utc"].endswith("Z")
