# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The boot warning for a stale prebuilt node manifest names every kind of difference (OMN-26338).

Read on omni-201 on 2026-11-24 (fixture ``tests/fixtures/omn26338/prebuilt_manifest_stale_dev_201.json``):
the dev runtime's prebuilt manifest differed from the installed packages in 1268 stamp
fields, 1267 of them ``distribution_added:omnimarket:*`` and one ``active_packages``. The
boot log named only the first twenty, all ``distribution_added``, so the changed active
package set was invisible and the only difference an operator could read was the one the
image can never fix.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from omnibase_infra.handlers import handler_package_announcer as announcer
from omnibase_infra.handlers.handler_prebuilt_manifest import (
    HandlerPrebuiltManifest,
    run_boot_catalog,
)
from omnibase_infra.models.node_announce.model_prebuilt_manifest_result import (
    ModelPrebuiltManifestResult,
)

_FIXTURE = (
    Path(__file__).parents[2] / "fixtures/omn26338/prebuilt_manifest_stale_dev_201.json"
)
_EP = "omnibase_infra.handlers.handler_package_announcer.entry_points"


def _recorded_differing_fields() -> tuple[str, ...]:
    """The 1268 fields the dev read counted, in the order ``differing_fields`` emits them."""
    record = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    (package, count), *_ = record["distribution_added"].items()
    added = tuple(f"distribution_added:{package}:node_{i:04d}" for i in range(count))
    active = record["active_packages"]
    changed = f"active_packages:{active['stored']!r}->{active['current']!r}"
    fields = (*added, changed)
    assert len(fields) == record["differing_total"]
    return fields


async def _boot_warning(
    caplog: pytest.LogCaptureFixture, differing: tuple[str, ...]
) -> str:
    stale = ModelPrebuiltManifestResult(reason="stale", differing_fields=differing)
    with (
        patch.object(HandlerPrebuiltManifest, "handle", return_value=stale),
        patch(_EP, return_value=[]),
        caplog.at_level(logging.WARNING),
    ):
        await run_boot_catalog(workload="main", announce=announcer.run_boot_step0)
    messages = [
        r.getMessage()
        for r in caplog.records
        if "differing stamp fields" in r.getMessage()
    ]
    assert len(messages) == 1
    return messages[0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stale_warning_names_the_changed_active_package_set_among_many_added_distributions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    message = await _boot_warning(caplog, _recorded_differing_fields())

    assert "active_packages:'omnibase_infra,omnimarket'" in message
    assert "omniclaude,omniintelligence,omnimarket'" in message
    assert "1268" in message
    assert "distribution_added:omnimarket x1267" in message


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stale_warning_with_few_differences_still_lists_each_field(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fields = (
        "distribution_version:pkg-one:node_alpha_effect:1.0.0->1.0.1",
        "distribution_added:pkg-three:node_delta_effect",
    )
    message = await _boot_warning(caplog, fields)

    assert all(field in message for field in fields)
