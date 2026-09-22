# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18999 — the exposure-reader gate, end to end, over all four of its inputs.

The unit tests exercise each resolver on its own. This one runs the thing CI runs: the
module's command-line entry point, over a contract tree, an omnidash component registry,
a shipped-layout directory and a Market status-page surface at once -- the same four
arguments ``.github/workflows/exposure-reader-coverage.yml`` passes.

That combination is where this gate's real failure mode lives. Every individual piece
can be correct while the assembled verdict is wrong, which is exactly what happened on
the live run of this change: the judgement counted the backend reader, the report did
not, and an exposure carrying no opt-out at all printed as opted out. Only a run over
all four inputs at once produces that line.

Nothing here skips. The inputs are files, not services, so there is no environment in
which this test is allowed to decline to answer (OMN-14172).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnibase_infra.validators.bus_backed_exposure_readers import main

pytestmark = pytest.mark.integration

PROMOTION_GATE_TOPIC = "onex.snapshot.projection.prod-promotion-gate.v1"
RENDERED_TOPIC = "onex.snapshot.projection.registration.v1"

# The Market surface, in the source shape the live one has: the reader kind is a Literal
# on the owning model, and the page names its reader and slot through module constants
# rather than inline strings. A fixture that inlined them would not exercise the
# constant resolution the live surface actually requires.
_MODELS_PY = '''\
from typing import Literal

from pydantic import BaseModel


class ModelProjectionBackendReader(BaseModel):
    """One contract-declared backend surface that reads an exposure."""

    id: str
    kind: Literal["projection_status_page"]
    route: str
    projection_slot: str
'''

_MORNING_PAGE_PY = """\
_PROMOTION_GATE_READER_ID = "onex_status_page"
_PROMOTION_GATE_SLOT = "promotion_gate"


def build_morning_page(topic_map, cache):
    return {
        "promotion_gate": read_backend_projection(
            topic_map,
            cache,
            reader_id=_PROMOTION_GATE_READER_ID,
            projection_slot=_PROMOTION_GATE_SLOT,
            route="/",
            limit=50,
            tenant_id=None,
        ),
    }
"""

_API_SERVER_PY = """\
@app.get("/", response_class=HTMLResponse)
async def status_page() -> HTMLResponse:
    return _render_status_page()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
"""


def _write_surface(root: Path) -> Path:
    surface = root / "omnimarket" / "src" / "omnimarket" / "projection"
    surface.mkdir(parents=True, exist_ok=True)
    (surface / "models.py").write_text(_MODELS_PY, encoding="utf-8")
    (surface / "morning_page.py").write_text(_MORNING_PAGE_PY, encoding="utf-8")
    (surface / "api_server.py").write_text(_API_SERVER_PY, encoding="utf-8")
    return surface


def _write_contracts(root: Path, *, promotion_gate_slot: str) -> Path:
    """A contract tree with both reader classes represented, as the live tree has."""
    contracts = root / "contracts"

    gate_dir = contracts / "node_projection_prod_promotion_gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    (gate_dir / "contract.yaml").write_text(
        "name: node_projection_prod_promotion_gate\n"
        "projection_api:\n"
        "  expose: true\n"
        "  exposures:\n"
        f"    - topic: {PROMOTION_GATE_TOPIC}\n"
        "      bus_backed: true\n"
        "      backend_readers:\n"
        "        - id: onex_status_page\n"
        "          kind: projection_status_page\n"
        "          route: /\n"
        f"          projection_slot: {promotion_gate_slot}\n",
        encoding="utf-8",
    )

    rendered_dir = contracts / "node_projection_registration"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    (rendered_dir / "contract.yaml").write_text(
        "name: node_projection_registration\n"
        "projection_api:\n"
        "  expose: true\n"
        "  exposures:\n"
        f"    - topic: {RENDERED_TOPIC}\n"
        "      bus_backed: true\n",
        encoding="utf-8",
    )
    return contracts


def _write_omnidash(root: Path) -> tuple[Path, Path]:
    omnidash = root / "omnidash"
    registry = omnidash / "src" / "registry" / "component-registry.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        json.dumps(
            {
                "manifestVersion": "1.0",
                "components": {
                    "event-stream": {
                        "name": "event-stream",
                        "dataSources": [
                            {
                                "type": "projection",
                                "topic": RENDERED_TOPIC,
                                "required": True,
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    layouts = omnidash / "src" / "templates"
    layouts.mkdir(parents=True, exist_ok=True)
    (layouts / "platform-health.ts").write_text(
        "import type { DashboardDefinition } from '@shared/types/dashboard';\n\n"
        "export const platformhealthTemplate: DashboardDefinition = {\n"
        "  id: 'template-platform-health',\n"
        "  schemaVersion: '1.0',\n"
        "  name: 'platform-health',\n"
        "  layout: [\n"
        "    { i: 'w0', componentName: 'event-stream', componentVersion: '1.0.0', "
        "x: 0, y: 0, w: 6, h: 6, config: {} },\n"
        "  ],\n"
        "  shared: true,\n"
        "};\n",
        encoding="utf-8",
    )
    return registry, layouts


def _argv(contracts: Path, registry: Path, layouts: Path, surface: Path) -> list[str]:
    """Exactly the argument shape the workflow passes."""
    return [
        str(contracts),
        "--registry",
        str(registry),
        "--layouts-dir",
        str(layouts),
        "--backend-reader-surface",
        str(surface),
    ]


def test_the_whole_gate_passes_and_names_both_reader_classes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both reader classes resolve in one run, and the report says which is which.

    The second assertion is the one worth having. A pass line naming the wrong reason
    is worse than a failure: it tells the next reader that a live exposure was silenced.
    """
    contracts = _write_contracts(tmp_path, promotion_gate_slot="promotion_gate")
    registry, layouts = _write_omnidash(tmp_path)
    surface = _write_surface(tmp_path)

    assert main(_argv(contracts, registry, layouts, surface)) == 0

    report = capsys.readouterr().err
    assert f"{PROMOTION_GATE_TOPIC} :: backend:onex_status_page" in report
    assert f"{RENDERED_TOPIC} :: event-stream" in report
    assert "opted out" not in report


def test_the_whole_gate_refuses_a_reader_the_market_surface_does_not_make(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The drift a gate carrying its own copy of Market's names cannot see.

    Every field is well formed; only the slot has moved. A validator holding
    ``promotion_gate`` as its own constant would keep passing this forever while the
    status page read nothing.
    """
    contracts = _write_contracts(tmp_path, promotion_gate_slot="promotion_gate_v2")
    registry, layouts = _write_omnidash(tmp_path)
    surface = _write_surface(tmp_path)

    assert main(_argv(contracts, registry, layouts, surface)) == 1

    report = capsys.readouterr().err
    assert "invalid_backend_reader" in report
    assert "does not read" in report


def test_the_whole_gate_fails_closed_when_the_market_surface_is_absent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A vanished sibling checkout is refused, not treated as an empty reader set.

    This is the sparse-checkout regression the gate has to survive. Accepting nothing
    would accuse every contract in the tree; skipping the check would accept every
    declaration. Neither is a verdict this gate has the evidence to give.
    """
    contracts = _write_contracts(tmp_path, promotion_gate_slot="promotion_gate")
    registry, layouts = _write_omnidash(tmp_path)

    exit_code = main(
        _argv(contracts, registry, layouts, tmp_path / "surface-not-checked-out")
    )

    assert exit_code == 1
    assert "fail-closed" in capsys.readouterr().err
