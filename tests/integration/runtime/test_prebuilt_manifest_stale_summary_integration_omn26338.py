# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A stale prebuilt manifest file read from disk names its changed switch at boot (OMN-26338).

Unlike the unit tests, which hand ``run_boot_catalog`` a fabricated list of differing
fields, this builds the image's manifest file with the real builder, ages its stamp the
way a dev image ages (the stored node listing is older than the installed packages and the
active package set moved), and boots through the real handler. The differing fields are the
ones the real stamp comparison emits for the packages installed here, so the warning under
test is the one an operator reads in the boot log.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import pytest

from omnibase_infra.handlers import handler_package_announcer as announcer
from omnibase_infra.handlers.handler_prebuilt_manifest import (
    SCHEMA_VERSION,
    run_boot_catalog,
)
from omnibase_infra.models.node_announce.model_package_announce_result import (
    ModelPackageAnnounceResult,
)
from omnibase_infra.nodes.node_boot_catalog_effect.runtime_prebuilt_manifest import (
    build_prebuilt_manifest,
)
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_cache_clear

pytestmark = pytest.mark.integration

_ACTIVE = "ONEX_ACTIVE_RUNTIME_PACKAGES"
_VERBATIM_LIMIT = 20


@pytest.mark.asyncio
async def test_stale_file_with_many_added_nodes_still_names_the_active_package_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv(_ACTIVE, "omnibase_infra")
    discover_contracts_cache_clear()
    path = tmp_path / "prebuilt_node_manifest.json"
    build_prebuilt_manifest(path, workload="main")

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["schema_version"] == SCHEMA_VERSION
    added_by_package: Counter[str] = Counter(
        dist for _entry, dist, _version in record["stamp"]["fingerprint"]
    )
    installed_nodes = sum(added_by_package.values())
    assert installed_nodes > _VERBATIM_LIMIT
    # The image was built before these packages and with another active set.
    record["stamp"]["fingerprint"] = []
    record["stamp"]["active_packages"] = "omnibase_infra,omnimarket"
    path.write_text(json.dumps(record), encoding="utf-8")

    caplog.set_level(logging.WARNING)
    announced: list[str] = []

    async def announce(*, workload: str) -> ModelPackageAnnounceResult:
        announced.append(workload)
        return await announcer.run_boot_step0(workload=workload)

    await run_boot_catalog(workload="main", announce=announce, path=path)

    messages = [
        r.getMessage()
        for r in caplog.records
        if "differing stamp fields" in r.getMessage()
    ]
    assert len(messages) == 1, messages
    message = messages[0]
    assert announced == ["main"]
    assert "active_packages:'omnibase_infra,omnimarket'->'omnibase_infra'" in message
    assert f"{installed_nodes + 1} fields" in message
    for package, count in added_by_package.items():
        assert f"distribution_added:{package} x{count}" in message
