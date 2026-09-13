# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Strict wiring must also refuse a process that wires NOTHING [OMN-18324].

The measurement this closes, read off the persistent ``onex-lab`` lane on
2026-09-13 (k3s namespace ``onex-dev``, lab host), from two pods at once::

    omnimarket-projection-delegation-writer
      Auto-wiring runtime profile ownership: profile=projection-writer-delegation
        owned=1 skipped=492
      Auto-wiring complete: wired=0 skipped=0 failed=1 quarantined=0
      Runtime health DEGRADED dimension=projection_attachment
        detail=1/1 declared projection(s) have no attached consumer and persist nothing

    omnimarket-tenant-projection-writer
      Auto-wiring runtime profile ownership: profile=tenant-projection owned=8 skipped=486
      Auto-wiring completed in 5.258s: wired=0 skipped=0 failed=8 quarantined=0

Both Deployments reported ``READY 1``. Neither carries
``ONEX_WIRING_STRICT_MODE``; OMN-17531 bound it on the runtime family only, so
a writer whose whole reason to exist is one contract runs the quarantining
semantics and reports healthy over a total loss.

Why this file exists ON TOP OF binding the flag
-----------------------------------------------
Binding the flag (the ``omninode_infra`` half of OMN-18324) closes the
``failed > 0`` shape above, because ``wire_from_manifest`` already raises on it
and the kernel's health server binds AFTER that point — a strict wiring failure
is a boot crash and therefore a NotReady pod. That is the whole mechanism, and
the reason no second one is introduced here.

It does NOT close the neighbouring shape. Strict mode reacts to FAILURES. A
process whose declared projections all resolve to ``SKIPPED`` reports
``wired=0 skipped=N failed=0``, persists nothing, raises nothing, and reads
identically green at any flag value with no exception available to catch. The
two pods above happened to fail rather than skip; that was luck, not a property
of the gate.

So the condition asserted here is the one the defect class actually names:
**a manifest that declares at least one projection and wires none of it is a
process that cannot write a row, whichever outcome its contracts carry.**

Scope is pinned in both directions on purpose:

* Strict + declared projections + ``wired == 0`` refuses.
* Strict + declared projections + ``wired >= 1`` does not. A partially-wired
  process is a weaker, different finding and ``projection_attachment`` already
  reports it by name; escalating it here would turn every narrow runtime
  profile into an outage.
* Strict + ``wired == 0`` + NO declared projection does not refuse. A process
  legitimately filtered to an empty projection set must not be converted into a
  crash by a gate about writers.
* Non-strict is unchanged in every one of those cases. The OMN-9126 default
  stays fail-open; this ticket widens what STRICT means, never what the default
  does.

Two levels, deliberately
------------------------
The four-way truth table is asserted against the pure predicate, where a report
can be synthesised for each cell — including the partially-wired cell, which
end-to-end would need a real importable handler and would make the fixture,
not the gate, the thing under test. The end-to-end tests then prove the
predicate is actually REACHED from ``wire_from_manifest`` with the real
report, which a pure-function test alone can never show. Neither level is
sufficient; the pair is.

Blast radius, measured live BEFORE the gate was written rather than argued
after (OMN-18324 AC2). Every process currently running at
``ONEX_WIRING_STRICT_MODE=1`` on the onex-lab lane, from its own boot line::

    omninode-runtime          profile=main     owned=299  wired=218 skipped=81 failed=0
    omninode-runtime-effects  profile=effects  owned=174  wired=167 skipped=7  failed=0
    omninode-runtime-worker   profile=workers  owned=5    wired=4   skipped=1  failed=0

None of the three is at ``wired == 0``, so none is newly refused by this
condition. On the ``.201`` compose dev lane the same reading gave
``runtime-effects wired=171`` and ``runtime-worker wired=4``; that lane's main
service and the stability-test and lakshman lanes had their boot line rotated
out of the retained log buffer, and are recorded as UNMEASURED rather than
assumed — a lane is not restarted to produce evidence for a gate.

Related tickets:
    - OMN-18324: this gate.
    - OMN-18307: the live measurement, which put strict mode on the writer
      explicitly out of scope and asked for this measurement first.
    - OMN-17531: bound the flag on the three lab compose lanes.
    - OMN-9126: made the raise conditional on the flag; the default preserved
      here.
    - OMN-16994 / OMN-16843: ``projection_attachment``, the health dimension
      that reported the defect truthfully while readiness stayed green.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_core.models.contracts.subcontracts.model_db_ownership_subcontract import (
    ModelDbOwnershipSubcontract,
)
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_core.models.errors import ModelOnexError
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    assert_strict_projection_coverage,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
)
from omnibase_infra.runtime.auto_wiring.report import (
    EnumWiringOutcome,
    ModelAutoWiringReport,
    ModelContractWiringResult,
)

STRICT_KEY = "ONEX_WIRING_STRICT_MODE"


def _db_io(table: str) -> ModelDbOwnershipSubcontract:
    """A minimal ``db_io.db_tables`` declaration.

    ``db_io.db_tables`` is the SAME discriminator the wiring seam reads to
    choose the projection dispatch arm and the one
    ``projection_liveness._declared_projection_refs`` reads to decide what a
    declared projection is. Using it here rather than a test-local notion of
    "projection" is what keeps this gate and the health dimension from
    disagreeing about the set they are both about.
    """
    return ModelDbOwnershipSubcontract(
        db_tables=[
            ModelDbTableDeclaration(
                name=table,
                database_ref="application",
                schema="public",
                migration="0001_test",
                access="write",
                role="omninode_runtime",
            )
        ]
    )


def _projection_contract(
    name: str, *, table: str = "test_projection_rows"
) -> ModelDiscoveredContract:
    """A declared projection that ``_prepare_contract_wiring`` will SKIP.

    No ``handler_routing``, which the preparation step skips by name ("No
    handler_routing declared in contract"). That is the shape this gate is
    about: the contract is a projection by every discriminator the runtime
    uses, and the process still ends its boot having wired none of it.
    """
    return ModelDiscoveredContract(
        name=name,
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name=name,
        package_name="test-package",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(f"onex.evt.platform.{name}.v1",),
            publish_topics=(),
        ),
        db_io=_db_io(table),
        handler_routing=None,
    )


def _nonprojection_contract(name: str) -> ModelDiscoveredContract:
    """The same skipped shape with NO ``db_io``, so it declares no projection."""
    return ModelDiscoveredContract(
        name=name,
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name=name,
        package_name="test-package",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(f"onex.evt.platform.{name}.v1",),
            publish_topics=(),
        ),
        handler_routing=None,
    )


def _report(*rows: tuple[str, EnumWiringOutcome]) -> ModelAutoWiringReport:
    return ModelAutoWiringReport(
        results=tuple(
            ModelContractWiringResult(
                contract_name=name,
                package_name="test-package",
                outcome=outcome,
                reason="synthesised for the truth table",
            )
            for name, outcome in rows
        )
    )


# ---------------------------------------------------------------------------
# The truth table, against the pure predicate.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_predicate_refuses_a_declared_projection_that_wired_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate itself."""
    monkeypatch.setenv(STRICT_KEY, "1")

    manifest = ModelAutoWiringManifest(
        contracts=[_projection_contract("node_projection_lonely")]
    )
    report = _report(("node_projection_lonely", EnumWiringOutcome.SKIPPED))

    with pytest.raises(ModelOnexError) as exc_info:
        assert_strict_projection_coverage(manifest, report)

    message = str(exc_info.value)
    assert "node_projection_lonely" in message, (
        "the refusal must name the projection that was never wired, or an "
        f"operator cannot act on it: {message!r}"
    )


@pytest.mark.unit
def test_predicate_allows_a_partially_wired_projection_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One wired contract is enough; this gate is about a TOTAL loss only.

    The cell that cannot be reached end to end without a real importable
    handler, and the one that bounds the gate. Without it, "refuse when
    anything is unwired" would pass every other test in this file while
    crashing every narrow runtime profile on the fleet.
    """
    monkeypatch.setenv(STRICT_KEY, "1")

    manifest = ModelAutoWiringManifest(
        contracts=[
            _projection_contract("node_projection_skipped"),
            _projection_contract("node_projection_wired", table="second_table"),
        ]
    )
    report = _report(
        ("node_projection_skipped", EnumWiringOutcome.SKIPPED),
        ("node_projection_wired", EnumWiringOutcome.WIRED),
    )

    assert_strict_projection_coverage(manifest, report)


@pytest.mark.unit
def test_predicate_ignores_a_manifest_with_no_declared_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``wired == 0`` alone is not the finding; a DECLARED projection is.

    The half that keeps the gate from firing on a process legitimately filtered
    to nothing. The delegation writer's defect was that it declared one and
    wrote none; a process that declares none writes none correctly.
    """
    monkeypatch.setenv(STRICT_KEY, "1")

    manifest = ModelAutoWiringManifest(
        contracts=[_nonprojection_contract("node_plain_skipped")]
    )
    report = _report(("node_plain_skipped", EnumWiringOutcome.SKIPPED))

    assert_strict_projection_coverage(manifest, report)


@pytest.mark.unit
def test_predicate_is_inert_without_the_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The OMN-9126 default. Prod and judge must acquire no new crash path.

    Anti-vacuity for the first test in this table: the SAME inputs that refuse
    under the flag must pass without it, or the gate has stopped being
    conditional.
    """
    monkeypatch.delenv(STRICT_KEY, raising=False)

    manifest = ModelAutoWiringManifest(
        contracts=[_projection_contract("node_projection_lonely")]
    )
    report = _report(("node_projection_lonely", EnumWiringOutcome.SKIPPED))

    assert_strict_projection_coverage(manifest, report)


@pytest.mark.unit
def test_predicate_refuses_the_live_failed_shape_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shape both lab pods were actually in: ``wired=0`` with FAILED rows.

    Already refused upstream by the OMN-9126 failure arm, so this is belt over
    braces — but it pins that the two arms agree rather than one of them
    quietly excluding what the other catches.
    """
    monkeypatch.setenv(STRICT_KEY, "1")

    manifest = ModelAutoWiringManifest(
        contracts=[_projection_contract("projection_delegation")]
    )
    report = _report(("projection_delegation", EnumWiringOutcome.FAILED))

    with pytest.raises(ModelOnexError):
        assert_strict_projection_coverage(manifest, report)


# ---------------------------------------------------------------------------
# The predicate is actually REACHED from the real wiring path.
# ---------------------------------------------------------------------------


async def _wire(manifest: ModelAutoWiringManifest) -> ModelAutoWiringReport:
    return await wire_from_manifest(
        manifest=manifest,
        dispatch_engine=MagicMock(),
        event_bus=AsyncMock(spec=ProtocolEventBusLike),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_wire_from_manifest_refuses_the_wired_nothing_boot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end. RED before OMN-18324.

    Before the change this returns a clean report with
    ``wired=0 skipped=1 failed=0``, and the kernel goes on to bind its health
    server and answer ``/ready`` with 200 — the exact live shape on the lab.
    """
    monkeypatch.setenv(STRICT_KEY, "1")

    manifest = ModelAutoWiringManifest(
        contracts=[_projection_contract("node_projection_lonely")]
    )

    with pytest.raises(ModelOnexError, match="node_projection_lonely"):
        await _wire(manifest)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_wire_from_manifest_reaches_the_gate_through_the_skipped_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fixture must reach the gate by SKIPPED, not by the old failure arm.

    Without this, the end-to-end test above would pass against the PRE-change
    code as soon as the fixture happened to fail, and would prove nothing new.
    """
    monkeypatch.delenv(STRICT_KEY, raising=False)

    manifest = ModelAutoWiringManifest(
        contracts=[_projection_contract("node_projection_lonely")]
    )

    report = await _wire(manifest)

    assert report.total_wired == 0
    assert report.total_failed == 0, (
        "this fixture must be SKIPPED, not FAILED — otherwise the end-to-end "
        "test is satisfied by the pre-existing OMN-9126 failure arm"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_wire_from_manifest_still_boots_with_no_declared_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict + wired nothing + no projection declared stays a clean boot."""
    monkeypatch.setenv(STRICT_KEY, "1")

    report = await _wire(
        ModelAutoWiringManifest(
            contracts=[_nonprojection_contract("node_plain_skipped")]
        )
    )

    assert report.total_wired == 0
    assert report.total_failed == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_wire_from_manifest_still_boots_on_an_empty_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty manifest declares nothing and must stay a clean boot."""
    monkeypatch.setenv(STRICT_KEY, "1")

    report = await _wire(ModelAutoWiringManifest(contracts=[]))

    assert report.total_wired == 0
