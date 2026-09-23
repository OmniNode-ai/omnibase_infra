# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17199 — the reader end of a ``bus_backed`` exposure must be asserted.

Every ratchet before this one guards the write end (OMN-15864 producer,
OMN-16795 publisher, OMN-16783 flow expectation). Nothing asserted that anybody
reads the result, and three incidents came out of that one asymmetry
(OMN-14440, Phoenix, and ``consumer-flow.v1`` inside epic OMN-16776's own
Phase 1 deliverable).

These tests are the falsification set for the ticket's acceptance criteria. The
most important one is :func:`test_ac4_the_live_condition_this_gate_was_built_for`
— a gate that passes on the exact condition it was built to catch is how
OMN-15864's sibling failed, so that condition is pinned here as a permanent
regression test rather than left to a one-off manual run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnibase_infra.validators.bus_backed_exposure_readers import (
    BackendReaderRegistration,
    BackendReaderSurface,
    ReaderSurfaceError,
    collect_backend_reader_surface,
    collect_bus_backed_exposures,
    collect_layout_readers,
    collect_registry_readers,
    evaluate,
    main,
)

pytestmark = pytest.mark.unit

CONSUMER_FLOW_TOPIC = "onex.snapshot.projection.consumer-flow.v1"
TENANT_CREDENTIALS_TOPIC = "onex.snapshot.projection.tenant-credentials.v1"
REGISTRATION_TOPIC = "onex.snapshot.projection.registration.v1"
LIVE_EVENTS_TOPIC = "onex.snapshot.projection.live-events.v1"


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


#: The Market status-page surface as it stands on omnimarket ``dev`` after #2774: one
#: registration, the promotion-gate slot at the root route. Built directly for the
#: judgement tests; :func:`_write_surface` writes the source form the resolver reads, so
#: the two are pinned against each other by
#: :func:`test_the_resolver_reads_the_surface_the_judgement_tests_assume`.
_SURFACE = BackendReaderSurface(
    kinds=frozenset({"projection_status_page"}),
    registrations=frozenset(
        {
            BackendReaderRegistration(
                reader_id="onex_status_page",
                route="/",
                projection_slot="promotion_gate",
            )
        }
    ),
    routes=frozenset({"/", "/morning", "/health"}),
)


def _write_surface(
    root: Path,
    *,
    kind_literal: str = 'Literal["projection_status_page"]',
    registration: str = (
        "    return read_backend_projection(\n"
        "        topic_map,\n"
        "        cache,\n"
        "        reader_id=_PROMOTION_GATE_READER_ID,\n"
        "        projection_slot=_PROMOTION_GATE_SLOT,\n"
        '        route="/",\n'
        "    )\n"
    ),
    routes: str = '@app.get("/", response_class=HTMLResponse)\n',
) -> Path:
    """Write a minimal but source-shaped copy of the Market projection surface.

    The resolver reads real Python, so the fixture is real Python -- including the
    indirection the live page actually uses (module constants rather than inline
    literals), because a fixture that inlined them would not exercise the resolution
    that makes the live surface readable at all.
    """
    surface = root / "projection"
    surface.mkdir(parents=True, exist_ok=True)
    (surface / "models.py").write_text(
        "from typing import Literal\n\n\n"
        "class ModelProjectionBackendReader(BaseModel):\n"
        "    id: str\n"
        f"    kind: {kind_literal}\n"
        "    route: str\n"
        "    projection_slot: str\n",
        encoding="utf-8",
    )
    (surface / "morning_page.py").write_text(
        '_PROMOTION_GATE_READER_ID = "onex_status_page"\n'
        '_PROMOTION_GATE_SLOT = "promotion_gate"\n\n\n'
        "def build_morning_page(topic_map, cache):\n" + registration,
        encoding="utf-8",
    )
    (surface / "api_server.py").write_text(
        routes + "async def status_page() -> HTMLResponse:\n"
        "    return _render_status_page()\n\n\n"
        '@app.get("/health")\n'
        "async def health() -> dict[str, str]:\n"
        '    return {"status": "ok"}\n',
        encoding="utf-8",
    )
    return surface


def _backend_reader(**overrides: object) -> dict[str, object]:
    reader: dict[str, object] = {
        "id": "onex_status_page",
        "kind": "projection_status_page",
        "route": "/",
        "projection_slot": "promotion_gate",
    }
    reader.update(overrides)
    return reader


def _write_contract(
    root: Path,
    node: str,
    exposure: dict[str, object],
) -> Path:
    node_dir = root / node
    node_dir.mkdir(parents=True, exist_ok=True)
    path = node_dir / "contract.yaml"
    body = ["name: " + node, "projection_api:"]
    for key, value in exposure.items():
        body.append(f"  {key}: {json.dumps(value)}")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return path


def _write_registry(path: Path, topics_by_component: dict[str, list[str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    components = {
        name: {
            "name": name,
            "dataSources": [
                {"type": "projection", "topic": topic, "required": True}
                for topic in topics
            ],
        }
        for name, topics in topics_by_component.items()
    }
    path.write_text(
        json.dumps({"manifestVersion": "1.0", "components": components}),
        encoding="utf-8",
    )
    return path


def _write_layout(layouts_dir: Path, name: str, component_names: list[str]) -> Path:
    """Write a shipped layout in the real shape: `omnidash/src/templates/<name>.ts`.

    NOT `dashboard-layouts/*.json`. That directory is gitignored in omnidash
    (`.gitignore:27`) and holds a user's locally-saved dashboards, so nothing in it ever
    reaches a checkout. A fixture in the wrong shape here would have let the gate ship
    resolving zero layout readers on every CI run while its tests stayed green.
    """
    layouts_dir.mkdir(parents=True, exist_ok=True)
    path = layouts_dir / f"{name}.ts"
    entries = "\n".join(
        f"    {{ i: 'w{i}', componentName: '{component}', "
        f"componentVersion: '1.0.0', x: 0, y: {i}, w: 6, h: 6, config: {{}} }},"
        for i, component in enumerate(component_names)
    )
    path.write_text(
        "import type { DashboardDefinition } from '@shared/types/dashboard';\n\n"
        f"export const {name.replace('-', '')}Template: DashboardDefinition = {{\n"
        f"  id: 'template-{name}',\n"
        "  schemaVersion: '1.0',\n"
        f"  name: '{name}',\n"
        "  layout: [\n"
        f"{entries}\n"
        "  ],\n"
        "  shared: true,\n"
        "};\n",
        encoding="utf-8",
    )
    return path


def _readers(registry: Path, layouts_dir: Path) -> dict[str, set[str]]:
    registry_readers = collect_registry_readers(registry)
    layout_readers = collect_layout_readers(layouts_dir, registry_readers)
    merged = {topic: set(names) for topic, names in registry_readers.items()}
    for topic, names in layout_readers.items():
        merged.setdefault(topic, set()).update(names)
    return merged


# ---------------------------------------------------------------------------
# AC1 — a bus_backed exposure with no reader and no consumers key fails
# ---------------------------------------------------------------------------


def test_ac1_bus_backed_with_no_reader_and_no_consumers_key_is_a_violation(
    tmp_path: Path,
) -> None:
    contracts = tmp_path / "contracts"
    _write_contract(
        contracts,
        "node_projection_orphan",
        {
            "expose": True,
            "topic": "onex.snapshot.projection.orphan.v1",
            "bus_backed": True,
        },
    )
    registry = _write_registry(
        tmp_path / "registry.json", {"some-widget": [REGISTRATION_TOPIC]}
    )
    layouts = tmp_path / "layouts"
    _write_layout(layouts, "default", ["some-widget"])

    findings = evaluate(
        collect_bus_backed_exposures([contracts], _SURFACE), _readers(registry, layouts)
    )

    assert [f.topic for f in findings] == ["onex.snapshot.projection.orphan.v1"]
    assert "no Omnidash or typed backend reader" in findings[0].reason


def test_ac1_exit_code_is_nonzero_so_the_gate_can_block(tmp_path: Path) -> None:
    """A finding must reach CI as a failing exit status, not merely as output."""
    contracts = tmp_path / "contracts"
    _write_contract(
        contracts,
        "node_projection_orphan",
        {
            "expose": True,
            "topic": "onex.snapshot.projection.orphan.v1",
            "bus_backed": True,
        },
    )
    registry = _write_registry(tmp_path / "registry.json", {"w": [REGISTRATION_TOPIC]})
    layouts = tmp_path / "layouts"
    _write_layout(layouts, "default", ["w"])

    assert (
        main(
            [
                str(contracts),
                "--registry",
                str(registry),
                "--layouts-dir",
                str(layouts),
                "--backend-reader-surface",
                str(_write_surface(tmp_path)),
            ]
        )
        == 1
    )


def test_a_bus_backed_exposure_with_a_registry_reader_passes(tmp_path: Path) -> None:
    contracts = tmp_path / "contracts"
    _write_contract(
        contracts,
        "node_projection_registration",
        {"expose": True, "topic": REGISTRATION_TOPIC, "bus_backed": True},
    )
    registry = _write_registry(
        tmp_path / "registry.json", {"event-stream": [REGISTRATION_TOPIC]}
    )
    layouts = tmp_path / "layouts"
    _write_layout(layouts, "default", [])

    assert (
        evaluate(
            collect_bus_backed_exposures([contracts], _SURFACE),
            _readers(registry, layouts),
        )
        == []
    )


def test_a_bus_backed_exposure_with_a_typed_backend_reader_passes(
    tmp_path: Path,
) -> None:
    contracts = tmp_path / "contracts"
    _write_contract(
        contracts,
        "node_projection_promotion_gate",
        {
            "expose": True,
            "topic": "onex.snapshot.projection.prod-promotion-gate.v1",
            "bus_backed": True,
            "backend_readers": [_backend_reader()],
        },
    )
    registry = _write_registry(tmp_path / "registry.json", {"w": [REGISTRATION_TOPIC]})
    layouts = tmp_path / "layouts"
    _write_layout(layouts, "default", ["w"])

    assert (
        evaluate(
            collect_bus_backed_exposures([contracts], _SURFACE),
            _readers(registry, layouts),
        )
        == []
    )


@pytest.mark.parametrize(
    "backend_readers",
    [
        pytest.param("onex_status_page", id="not-a-list"),
        pytest.param(["onex_status_page"], id="entry-not-a-mapping"),
        pytest.param([{}], id="missing-required-fields"),
        pytest.param([_backend_reader(kind="unknown_reader")], id="unknown-kind"),
        pytest.param([_backend_reader(route="morning")], id="relative-route"),
        pytest.param([_backend_reader(route="/history")], id="unknown-absolute-route"),
        pytest.param(
            [_backend_reader(projection_slot="promotion-gate")], id="invalid-slot"
        ),
        pytest.param(
            [_backend_reader(projection_slot="other_panel")], id="unknown-slot"
        ),
        pytest.param([_backend_reader(id="other_status_page")], id="unknown-reader-id"),
        pytest.param(
            [_backend_reader(), _backend_reader(projection_slot="secondary_gate")],
            id="duplicate-id",
        ),
    ],
)
def test_malformed_backend_reader_declarations_fail_closed(
    tmp_path: Path, backend_readers: object
) -> None:
    contracts = tmp_path / "contracts"
    _write_contract(
        contracts,
        "node_projection_promotion_gate",
        {
            "expose": True,
            "topic": "onex.snapshot.projection.prod-promotion-gate.v1",
            "bus_backed": True,
            "backend_readers": backend_readers,
        },
    )
    registry = _write_registry(
        tmp_path / "registry.json",
        {"real-dashboard-reader": ["onex.snapshot.projection.prod-promotion-gate.v1"]},
    )
    layouts = tmp_path / "layouts"
    _write_layout(layouts, "default", ["real-dashboard-reader"])

    findings = evaluate(
        collect_bus_backed_exposures([contracts], _SURFACE), _readers(registry, layouts)
    )

    assert len(findings) == 1
    assert findings[0].code == "invalid_backend_reader"


def test_consumers_none_is_stale_when_a_backend_reader_exists(tmp_path: Path) -> None:
    contracts = tmp_path / "contracts"
    topic = "onex.snapshot.projection.prod-promotion-gate.v1"
    _write_contract(
        contracts,
        "node_projection_promotion_gate",
        {
            "expose": True,
            "topic": topic,
            "bus_backed": True,
            "backend_readers": [_backend_reader()],
            "consumers": "none",
            "consumers_reason": "The old reader was intentionally retired.",
        },
    )
    registry = _write_registry(tmp_path / "registry.json", {"w": [REGISTRATION_TOPIC]})
    layouts = tmp_path / "layouts"
    _write_layout(layouts, "default", ["w"])

    findings = evaluate(
        collect_bus_backed_exposures([contracts], _SURFACE), _readers(registry, layouts)
    )

    assert len(findings) == 1
    assert findings[0].code == "stale_opt_out"
    assert "backend:onex_status_page" in findings[0].reason


def test_a_layout_entry_alone_counts_as_a_reader(tmp_path: Path) -> None:
    """The ticket admits a shipped layout entry as a reader in its own right.

    Every fixture in this file declares the section-level ``expose: true`` gate on
    purpose. It is not decoration: ``_projection_api_served_exposures`` -- the same
    predicate omnimarket's ``build_projection_topic_map`` and ``SnapshotCache`` apply --
    treats an unexposed section as not served, and an exposure nothing serves cannot be
    rendered by anyone, so demanding a reader for it would be a false positive. Omitting
    the key makes a test pass because nothing was collected, which is a vacuous green.
    """
    contracts = tmp_path / "contracts"
    _write_contract(
        contracts,
        "node_projection_live_events",
        {"expose": True, "topic": LIVE_EVENTS_TOPIC, "bus_backed": True},
    )
    registry = _write_registry(
        tmp_path / "registry.json", {"live-event-stream": [LIVE_EVENTS_TOPIC]}
    )
    layouts = tmp_path / "layouts"
    _write_layout(layouts, "default", ["live-event-stream"])

    readers = _readers(registry, layouts)
    assert "default.ts" in readers[LIVE_EVENTS_TOPIC]
    assert evaluate(collect_bus_backed_exposures([contracts], _SURFACE), readers) == []


def test_an_exposure_that_is_not_bus_backed_is_out_of_scope(tmp_path: Path) -> None:
    """OMN-15800 reverted savings.v1 to bus_backed: false; it is served by SQL."""
    contracts = tmp_path / "contracts"
    _write_contract(
        contracts,
        "node_projection_savings",
        {
            "expose": True,
            "topic": "onex.snapshot.projection.savings.v1",
            "bus_backed": False,
        },
    )
    registry = _write_registry(tmp_path / "registry.json", {"w": [REGISTRATION_TOPIC]})
    layouts = tmp_path / "layouts"
    _write_layout(layouts, "default", ["w"])

    assert collect_bus_backed_exposures([contracts], _SURFACE) == []


# ---------------------------------------------------------------------------
# AC2 — `consumers: none` without a reason is also a failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exposure_extra",
    [
        pytest.param({"consumers": "none"}, id="no-reason-key"),
        pytest.param({"consumers": "none", "consumers_reason": ""}, id="empty-reason"),
        pytest.param(
            {"consumers": "none", "consumers_reason": "   "}, id="whitespace-reason"
        ),
        pytest.param({"consumers": "none", "consumers_reason": None}, id="null-reason"),
        pytest.param(
            {"consumers": "none", "consumers_reason": 7}, id="non-string-reason"
        ),
    ],
)
def test_ac2_consumers_none_without_a_real_reason_is_a_violation(
    tmp_path: Path, exposure_extra: dict[str, object]
) -> None:
    contracts = tmp_path / "contracts"
    _write_contract(
        contracts,
        "node_projection_orphan",
        {
            "expose": True,
            "topic": "onex.snapshot.projection.orphan.v1",
            "bus_backed": True,
            **exposure_extra,
        },
    )
    registry = _write_registry(tmp_path / "registry.json", {"w": [REGISTRATION_TOPIC]})
    layouts = tmp_path / "layouts"
    _write_layout(layouts, "default", ["w"])

    findings = evaluate(
        collect_bus_backed_exposures([contracts], _SURFACE), _readers(registry, layouts)
    )

    assert len(findings) == 1
    assert "consumers_reason" in findings[0].reason


def test_consumers_none_with_a_real_reason_passes(tmp_path: Path) -> None:
    contracts = tmp_path / "contracts"
    _write_contract(
        contracts,
        "node_projection_tenant_credentials",
        {
            "expose": True,
            "topic": TENANT_CREDENTIALS_TOPIC,
            "bus_backed": True,
            "consumers": "none",
            "consumers_reason": (
                "tenant-scoped BYOK credential references served to the owning "
                "tenant's API client; never rendered on a dashboard surface."
            ),
        },
    )
    registry = _write_registry(tmp_path / "registry.json", {"w": [REGISTRATION_TOPIC]})
    layouts = tmp_path / "layouts"
    _write_layout(layouts, "default", ["w"])

    assert (
        evaluate(
            collect_bus_backed_exposures([contracts], _SURFACE),
            _readers(registry, layouts),
        )
        == []
    )


def test_an_unrecognised_consumers_value_fails_closed(tmp_path: Path) -> None:
    """`consumers: tbd` must not become a silent third escape hatch."""
    contracts = tmp_path / "contracts"
    _write_contract(
        contracts,
        "node_projection_orphan",
        {
            "expose": True,
            "topic": "onex.snapshot.projection.orphan.v1",
            "bus_backed": True,
            "consumers": "tbd",
            "consumers_reason": "we will wire a widget later",
        },
    )
    registry = _write_registry(tmp_path / "registry.json", {"w": [REGISTRATION_TOPIC]})
    layouts = tmp_path / "layouts"
    _write_layout(layouts, "default", ["w"])

    findings = evaluate(
        collect_bus_backed_exposures([contracts], _SURFACE), _readers(registry, layouts)
    )

    assert len(findings) == 1
    assert "not a recognised declaration" in findings[0].reason


# ---------------------------------------------------------------------------
# AC4 — the live condition this gate exists to catch
# ---------------------------------------------------------------------------


def test_ac4_the_live_condition_this_gate_was_built_for(tmp_path: Path) -> None:
    """Pin the 2026-08-30 tree state: consumer-flow.v1 must be NAMED as violating.

    The registry fixture below is the omnidash component registry as it stood on
    2026-08-30 with respect to the four ``bus_backed`` exposures in omnimarket:
    ``registration.v1`` and ``live-events.v1`` had widgets, ``consumer-flow.v1``
    and ``tenant-credentials.v1`` had none. That is the exact shape the ticket
    was filed on — 2.55M rows at 43s freshness, 79 STALLED consumer-windows in
    ten minutes, and zero hits for ``consumer_flow`` anywhere in the registry or
    the layouts.

    This stays pinned after the live tree is fixed. A gate that passes on the
    condition it was built to catch is how OMN-15864's sibling failed.
    """
    contracts = tmp_path / "contracts"
    _write_contract(
        contracts,
        "node_projection_consumer_flow",
        {"expose": True, "topic": CONSUMER_FLOW_TOPIC, "bus_backed": True},
    )
    _write_contract(
        contracts,
        "node_projection_tenant_credentials",
        {"expose": True, "topic": TENANT_CREDENTIALS_TOPIC, "bus_backed": True},
    )
    _write_contract(
        contracts,
        "node_projection_registration",
        {"expose": True, "topic": REGISTRATION_TOPIC, "bus_backed": True},
    )
    _write_contract(
        contracts,
        "node_projection_live_events",
        {"expose": True, "topic": LIVE_EVENTS_TOPIC, "bus_backed": True},
    )
    registry = _write_registry(
        tmp_path / "registry.json",
        {
            "event-stream": [REGISTRATION_TOPIC],
            "live-event-stream": [LIVE_EVENTS_TOPIC],
            "cost-trend-panel": ["onex.snapshot.projection.llm_cost.v1"],
        },
    )
    layouts = tmp_path / "layouts"
    _write_layout(layouts, "default", ["live-event-stream", "cost-trend-panel"])

    findings = evaluate(
        collect_bus_backed_exposures([contracts], _SURFACE), _readers(registry, layouts)
    )
    named = {f.topic for f in findings}

    assert CONSUMER_FLOW_TOPIC in named, (
        "the gate did not name the exposure it was built to catch; "
        f"named={sorted(named)}"
    )
    assert TENANT_CREDENTIALS_TOPIC in named
    assert REGISTRATION_TOPIC not in named
    assert LIVE_EVENTS_TOPIC not in named


# ---------------------------------------------------------------------------
# AC5 — no allowlist, no baseline, no suppression
# ---------------------------------------------------------------------------


def test_ac5_the_validator_ships_no_allowlist_or_baseline_surface() -> None:
    """OMN-17068: the sibling ratchet decayed into a 682-entry grandfather list.

    Guarding this in a test rather than in a comment is the whole point — a
    reviewer's good intentions are not a mechanism. The check is over the
    validator's *identifiers*, not its prose: the module docstring and the
    remediation text both talk about allowlists in order to forbid them, and a
    naive substring scan would fire on the very sentence that bans the shape.
    """
    import ast

    from omnibase_infra.validators import bus_backed_exposure_readers as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    identifiers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, ast.arg):
            identifiers.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            identifiers.add(node.name)

    banned_tokens = (
        "allowlist",
        "allow_list",
        "baseline",
        "grandfather",
        "skiplist",
        "skip_list",
        "suppress",
        "exempt",
        "waiver",
    )
    for identifier in identifiers:
        lowered = identifier.lower()
        for token in banned_tokens:
            assert token not in lowered, (
                f"identifier {identifier!r} in the validator carries {token!r}; "
                "this gate must not grow the grandfathered-ratchet shape "
                "OMN-17068 records."
            )

    # Baselines in this repo live under config/validation/*.yaml. This gate
    # must never read one, and must never gain a companion file to freeze
    # today's violations into.
    assert "config/validation" not in source
    assert not list(Path("config/validation").glob("*exposure_reader*")), (
        "a baseline/allowlist companion file was added for this gate"
    )


# ---------------------------------------------------------------------------
# Fail-closed on an unresolvable reader surface
# ---------------------------------------------------------------------------


def test_a_missing_registry_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ReaderSurfaceError):
        collect_registry_readers(tmp_path / "does-not-exist.json")


def test_a_registry_without_a_components_mapping_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"manifestVersion": "1.0"}), encoding="utf-8")
    with pytest.raises(ReaderSurfaceError):
        collect_registry_readers(path)


def test_a_missing_layouts_dir_fails_closed(tmp_path: Path) -> None:
    registry = _write_registry(tmp_path / "registry.json", {"w": [REGISTRATION_TOPIC]})
    with pytest.raises(ReaderSurfaceError):
        collect_layout_readers(
            tmp_path / "no-such-dir", collect_registry_readers(registry)
        )


def test_an_empty_scan_is_not_reported_as_compliance(tmp_path: Path) -> None:
    registry = _write_registry(tmp_path / "registry.json", {"w": [REGISTRATION_TOPIC]})
    layouts = tmp_path / "layouts"
    _write_layout(layouts, "default", ["w"])
    assert (
        main(
            [
                "--registry",
                str(registry),
                "--layouts-dir",
                str(layouts),
                "--backend-reader-surface",
                str(_write_surface(tmp_path)),
            ]
        )
        == 1
    )


# ---------------------------------------------------------------------------
# Exposure-shape coverage
# ---------------------------------------------------------------------------


def test_exposures_declared_as_a_list_are_all_scanned(tmp_path: Path) -> None:
    contracts = tmp_path / "contracts" / "node_multi"
    contracts.mkdir(parents=True)
    (contracts / "contract.yaml").write_text(
        "name: node_multi\n"
        "projection_api:\n"
        "  expose: true\n"
        "  exposures:\n"
        f"    - topic: {REGISTRATION_TOPIC}\n"
        "      bus_backed: true\n"
        "    - topic: onex.snapshot.projection.orphan.v1\n"
        "      bus_backed: true\n",
        encoding="utf-8",
    )
    registry = _write_registry(tmp_path / "registry.json", {"w": [REGISTRATION_TOPIC]})
    layouts = tmp_path / "layouts"
    _write_layout(layouts, "default", ["w"])

    findings = evaluate(
        collect_bus_backed_exposures([tmp_path / "contracts"], _SURFACE),
        _readers(registry, layouts),
    )
    assert [f.topic for f in findings] == ["onex.snapshot.projection.orphan.v1"]


def test_a_bus_backed_exposure_with_no_topic_is_a_violation(tmp_path: Path) -> None:
    contracts = tmp_path / "contracts"
    _write_contract(contracts, "node_topicless", {"expose": True, "bus_backed": True})
    registry = _write_registry(tmp_path / "registry.json", {"w": [REGISTRATION_TOPIC]})
    layouts = tmp_path / "layouts"
    _write_layout(layouts, "default", ["w"])

    findings = evaluate(
        collect_bus_backed_exposures([contracts], _SURFACE), _readers(registry, layouts)
    )
    assert len(findings) == 1
    assert findings[0].topic == "<missing>"


# ---------------------------------------------------------------------------
# OMN-18999 — a typed backend reader is resolved against the surface that reads
# it, not against a copy of that surface's names kept in this repo
# ---------------------------------------------------------------------------


def test_the_resolver_reads_the_surface_the_judgement_tests_assume(
    tmp_path: Path,
) -> None:
    """The fixture surface and the hand-built one must be the same surface.

    Every judgement test above takes :data:`_SURFACE` on trust. If the resolver read
    the source form differently, those tests would go on passing while the gate this
    module ships judged against something else -- so the two forms are pinned here.
    """
    resolved = collect_backend_reader_surface(_write_surface(tmp_path))

    assert resolved.kinds == _SURFACE.kinds
    assert resolved.registrations == _SURFACE.registrations
    assert "/" in resolved.routes


# The live Market surface is deliberately NOT probed from this suite. A test that
# resolves a sibling omnimarket checkout cannot execute in the job that collects it --
# the unit suite has no sibling checkout -- so it could only ever be a skip, and a test
# that is collected and never executed proves nothing (OMN-18776). The live proof lives
# where the checkout actually exists: the `exposure-reader-coverage` job runs this
# validator against omnimarket `dev` on every pull request, and
# `tests/integration/validators/test_omn18999_backend_reader_gate_integration.py` runs
# the same entry point over all four inputs with no skip. Do not re-add a skipping
# probe here.


def test_the_omn18999_declaration_shipped_by_omnimarket_2774_is_accepted(
    tmp_path: Path,
) -> None:
    """The exact contract shape that landed on omnimarket dev must pass.

    This is the condition nine omnibase_infra pull requests were blocked on. A gate
    that refused the real declaration would be no better than one that had no support
    for the field at all.
    """
    contracts = tmp_path / "contracts"
    _write_contract(
        contracts,
        "node_projection_prod_promotion_gate",
        {
            "expose": True,
            "topic": "onex.snapshot.projection.prod-promotion-gate.v1",
            "bus_backed": True,
            "backend_readers": [
                {
                    "id": "onex_status_page",
                    "kind": "projection_status_page",
                    "route": "/",
                    "projection_slot": "promotion_gate",
                }
            ],
        },
    )

    assert evaluate(collect_bus_backed_exposures([contracts], _SURFACE), {}) == []


def test_a_wellformed_reader_for_a_slot_the_page_does_not_read_is_refused(
    tmp_path: Path,
) -> None:
    """The point of resolving the surface at all.

    ``fleet_health`` is a perfectly shaped declaration: known kind, lower_snake id and
    slot, absolute route. Nothing in its own text is wrong. It is refused because the
    status page makes no such read -- which is the promise-with-no-reader this gate
    exists to catch, wearing the new field as a disguise.
    """
    contracts = tmp_path / "contracts"
    _write_contract(
        contracts,
        "node_projection_invented",
        {
            "expose": True,
            "topic": LIVE_EVENTS_TOPIC,
            "bus_backed": True,
            "backend_readers": [_backend_reader(projection_slot="fleet_health")],
        },
    )

    findings = evaluate(collect_bus_backed_exposures([contracts], _SURFACE), {})

    assert [finding.code for finding in findings] == ["invalid_backend_reader"]
    assert "does not read" in findings[0].reason
    assert str(findings[0].contract).endswith("contract.yaml")


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param({"kind_literal": "str"}, id="kind-is-not-a-literal"),
        pytest.param(
            {"registration": "    return None\n"}, id="page-registers-no-slot"
        ),
        pytest.param({"routes": '@app.get("/morning")\n'}, id="route-is-not-served"),
        pytest.param(
            {
                "registration": (
                    "    return read_backend_projection(\n"
                    "        topic_map,\n"
                    "        cache,\n"
                    "        reader_id=resolve_reader_id(),\n"
                    "        projection_slot=_PROMOTION_GATE_SLOT,\n"
                    '        route="/",\n'
                    "    )\n"
                )
            },
            id="registration-is-not-statically-resolvable",
        ),
    ],
)
def test_an_unresolvable_market_surface_fails_closed(
    tmp_path: Path, mutation: dict[str, str]
) -> None:
    """Every way the surface can fail to resolve refuses, none returns an empty set.

    An empty surface would accept nothing and read as a wall of contract findings; a
    *skipped* surface would accept everything. Both are verdicts this gate has no basis
    to give, so it gives neither.
    """
    with pytest.raises(ReaderSurfaceError):
        collect_backend_reader_surface(_write_surface(tmp_path, **mutation))


def test_a_missing_backend_reader_surface_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ReaderSurfaceError):
        collect_backend_reader_surface(tmp_path / "no-such-surface")


def test_the_surface_argument_is_required_so_it_cannot_be_skipped(
    tmp_path: Path,
) -> None:
    """No default path, on purpose.

    A defaulted surface argument turns a sparse-checkout regression into a silently
    weaker gate, which is the OMN-15606 fail-open shape this epic closes.
    """
    registry = _write_registry(tmp_path / "registry.json", {"w": [REGISTRATION_TOPIC]})
    layouts = tmp_path / "layouts"
    _write_layout(layouts, "default", ["w"])

    with pytest.raises(SystemExit):
        main(["--registry", str(registry), "--layouts-dir", str(layouts)])


def test_an_omnidash_reader_still_counts_with_no_backend_reader_declared(
    tmp_path: Path,
) -> None:
    """The pre-existing reader path is untouched by the new one.

    Regression pin: the backend-reader arm is an addition, not a replacement, and a
    contract that declares no ``backend_readers`` at all must judge exactly as it did
    before OMN-18999.
    """
    contracts = tmp_path / "contracts"
    _write_contract(
        contracts,
        "node_registration",
        {"expose": True, "topic": REGISTRATION_TOPIC, "bus_backed": True},
    )
    registry = _write_registry(
        tmp_path / "registry.json", {"registration-widget": [REGISTRATION_TOPIC]}
    )
    layouts = tmp_path / "layouts"
    _write_layout(layouts, "default", ["registration-widget"])

    exposures = collect_bus_backed_exposures([contracts], _SURFACE)

    assert [exposure.backend_readers for exposure in exposures] == [()]
    assert evaluate(exposures, _readers(registry, layouts)) == []


def test_the_pass_line_names_the_backend_reader_not_an_opt_out(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A pass line must say why it passed.

    Found on the live run of this change: the success listing consulted only the
    omnidash reader map, so an exposure with no ``consumers`` key at all, read by the
    status page, printed as "opted out with a stated reason". The verdict was right and
    its stated reason was wrong, which is worse than a plain failure -- it tells the
    next reader a live exposure was silenced.
    """
    contracts = tmp_path / "contracts"
    _write_contract(
        contracts,
        "node_projection_promotion_gate",
        {
            "expose": True,
            "topic": "onex.snapshot.projection.prod-promotion-gate.v1",
            "bus_backed": True,
            "backend_readers": [_backend_reader()],
        },
    )
    registry = _write_registry(tmp_path / "registry.json", {"w": [REGISTRATION_TOPIC]})
    layouts = tmp_path / "layouts"
    _write_layout(layouts, "default", ["w"])

    assert (
        main(
            [
                str(contracts),
                "--registry",
                str(registry),
                "--layouts-dir",
                str(layouts),
                "--backend-reader-surface",
                str(_write_surface(tmp_path)),
            ]
        )
        == 0
    )

    report = capsys.readouterr().err
    assert "backend:onex_status_page" in report
    assert "opted out" not in report
