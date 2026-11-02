# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""In full_load mode with hot load off, the boot attach report did not change (OMN-21590, A2 AC2).

The golden file was recorded by running this same scenario against the code at the
commit before the catalog existed (the entry-point scan inside ``discover_contracts``),
with the fixture directory path replaced by ``<fixtures>``. The scenario discovers three
fixture contracts through the catalog, wires them as the kernel does and serialises the
manifest and the attach report with sorted keys. A diff of one byte fails.
"""

from __future__ import annotations

import difflib
import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.discovery import (
    discover_contracts,
    discover_contracts_cache_clear,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

FIXTURES = Path(__file__).parents[2] / "fixtures"
CONTRACTS = FIXTURES / "node_announce"
GOLDEN = FIXTURES / "golden" / "boot_attach_golden_full_load_omn21590.json"
NAMES = ("node_attach_one_effect", "node_attach_two_effect", "node_attach_three_effect")


class HandlerAttachFixture:
    """A wirable handler class; the report does not depend on what it does."""

    async def handle(self, envelope: object) -> None:
        return None


def _entry_points(root: Path) -> list[MagicMock]:
    eps = []
    for name in NAMES:
        ep = MagicMock()
        ep.name = name
        ep.dist = MagicMock()
        ep.dist.name = "attach-fixture"
        ep.dist.version = "1.0.0"
        ep.load.return_value = type(
            "FixtureNode", (), {"contract_path": root / name / "contract.yaml"}
        )
        eps.append(ep)
    return eps


async def _manifest_and_report_json(root: Path = CONTRACTS) -> str:
    discover_contracts_cache_clear()
    try:
        with patch(
            "omnibase_infra.handlers.handler_package_announcer.entry_points",
            return_value=_entry_points(root),
        ):
            manifest = discover_contracts()
        bus = MagicMock(spec=ProtocolEventBusLike)
        bus.subscribe = AsyncMock(return_value=AsyncMock())
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=HandlerAttachFixture,
        ):
            report = await wire_from_manifest(
                manifest=manifest,
                dispatch_engine=MessageDispatchEngine(),
                event_bus=bus,
                environment="local",
                subscribe_immediately=True,
            )
    finally:
        discover_contracts_cache_clear()
    text = json.dumps(
        {
            "manifest": json.loads(manifest.model_dump_json()),
            "report": json.loads(report.model_dump_json()),
        },
        sort_keys=True,
        indent=2,
    )
    # The manifest may carry the path as built (``root``) or resolved (a symlinked
    # temporary directory); both forms are redacted.
    for form in {str(root), str(root.resolve())}:
        text = text.replace(form, "<fixtures>")
    return text


def _changed_lines(produced: str) -> list[str]:
    """The lines on which the produced report and the golden differ."""
    golden = GOLDEN.read_text(encoding="utf-8")
    return [
        line
        for line in difflib.unified_diff(
            golden.splitlines(), (produced + "\n").splitlines(), lineterm="", n=0
        )
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_boot_attach_report_is_byte_identical_to_the_pre_change_report() -> (
    None
):
    produced = await _manifest_and_report_json()

    assert produced + "\n" == GOLDEN.read_text(encoding="utf-8")


@pytest.mark.unit
def test_the_golden_names_every_fixture_contract() -> None:
    """Positive control: a golden that omitted a contract would make the diff meaningless."""
    golden = GOLDEN.read_text(encoding="utf-8")
    assert all(name in golden for name in NAMES)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_planted_violation_a_changed_contract_changes_only_its_hash(
    tmp_path: Path,
) -> None:
    """Control: a byte change in one contract moves exactly that contract's content hash."""
    root = tmp_path / "node_announce"
    shutil.copytree(CONTRACTS, root)
    changed = root / NAMES[0] / "contract.yaml"
    changed.write_bytes(changed.read_bytes() + b"# a comment moves the content hash\n")

    changed_lines = _changed_lines(await _manifest_and_report_json(root))

    assert len(changed_lines) == 2
    assert all("contract_content_hash" in line for line in changed_lines)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_planted_violation_a_changed_topic_changes_the_report(
    tmp_path: Path,
) -> None:
    """Control: a semantic change (a subscribed topic) shows in the report, not only in a hash."""
    root = tmp_path / "node_announce"
    shutil.copytree(CONTRACTS, root)
    changed = root / NAMES[0] / "contract.yaml"
    text = changed.read_text(encoding="utf-8")
    assert "attach-one.v1" in text
    changed.write_text(
        text.replace("attach-one.v1", "attach-one-changed.v1"), encoding="utf-8"
    )

    changed_lines = _changed_lines(await _manifest_and_report_json(root))

    assert any("attach-one-changed.v1" in line for line in changed_lines)
