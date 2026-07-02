# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-12548 — dispatch-selection parity gate (S0 GO signal for epic OMN-12525).

Design: ``docs/plans/2026-07-02-omn-12548-dispatch-parity-gate-design.md``.

This is the single GO gate before any dispatch-chain seam cutover (OMN-12549 →
S1..S5). It pins the CURRENT behavior of the live :class:`MessageDispatchEngine`
selection (design D1: warts included) as a committed fixture and fails any PR that
drifts selection semantics.

Two modes (design D4), authored against the protocol and parameterized by
implementation so Mode B is a fixture addition, not a rewrite:

* **Mode A — regression pin (ACTIVE NOW).** Regenerate the live engine's selection
  in-process and assert every probe's equivalence tuple (design D2) equals the
  committed ``baseline-selection-v2.json`` oracle. Valuable from day one; satisfies
  the S0-gate-before-S0-seam ordering.
* **Mode B — dual-implementation parity (LIVE as of OMN-12549).** The same probe
  corpus driven through both ``MessageDispatchEngine`` and ``MixinNodeDispatch``
  (``omnibase_core.runtime.mixin_node_dispatch``) behind ``ProtocolDispatchEngine``
  (now in ``omnibase_spi.protocols.runtime``), asserting tuple equality against the
  committed oracle. OMN-12549's DoD ("parity pytest green with the seam") is this
  Mode B green; ``test_mode_b_is_live_not_skipped`` fails closed so a stale/pre-seam
  ``omnibase_core`` cannot masquerade as parity by skipping Mode B.

Equivalence tuple (design D2)::

    (status, [dispatcher_id...] IN ORDER, message_category, message_type, dlq_topic)

Corpus (design D6): the fixture covers exactly the ``onex.nodes`` distributions
inside the ``omnibase_infra`` dependency closure (core + infra + onex-change-control).
Sibling packages (omnimarket, omniclaude, omniintelligence, omnimemory) are OUTSIDE
that closure and are EXPECTED-EXCLUDED — recorded, never silent. The corpus-guard
test FAILS if a required package is missing from the CI venv.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.dispatch_parity import harness

pytestmark = pytest.mark.integration

# The committed Mode-A oracle lives beside the harness so CI reads it locally.
_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "dispatch_parity"
    / "baseline-selection-v2.json"
)


@pytest.fixture(scope="module")
def committed_fixture() -> dict[str, Any]:
    """Load the committed Mode-A oracle. Missing fixture is a hard failure."""
    if not _FIXTURE_PATH.exists():
        pytest.fail(
            f"Committed parity fixture missing: {_FIXTURE_PATH}. "
            "Regenerate with: uv run python -m tests.fixtures.dispatch_parity.harness "
            f"--out {_FIXTURE_PATH}"
        )
    with _FIXTURE_PATH.open() as fh:
        loaded: dict[str, Any] = json.load(fh)
    return loaded


@pytest.fixture(scope="module")
def live_snapshot() -> dict[str, Any]:
    """Regenerate the live engine's selection snapshot in-process.

    The engine logs a WARNING per unroutable topic (NO_DISPATCHER -> DLQ). That is
    correct engine behavior — this gate PINS it — but it floods CI logs, so we raise
    the dispatch-engine logger to ERROR for the duration of the build only.
    """
    engine_logger = logging.getLogger("omnibase_infra.runtime.message_dispatch_engine")
    prior_level = engine_logger.level
    engine_logger.setLevel(logging.ERROR)
    try:
        return harness.build_snapshot()
    finally:
        engine_logger.setLevel(prior_level)


# ---------------------------------------------------------------------------
# Corpus guard (design D6): the test asserts EXACTLY which corpus slice it covers
# and FAILS if a required package is missing.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_required_corpus_packages_installed() -> None:
    """Every REQUIRED onex.nodes package must be present in the CI venv.

    A missing required package silently shrinks the corpus and hides regressions,
    so this is a hard failure — not a skip.
    """
    installed = harness._installed_onex_nodes_packages()
    missing = [
        pkg
        for pkg in harness.REQUIRED_ONEX_NODES_PACKAGES
        if installed.get(pkg) is None
    ]
    assert not missing, (
        f"Required onex.nodes packages missing from the venv: {missing}. "
        "The parity corpus is incomplete; the gate cannot certify selection parity. "
        f"Installed onex.nodes package versions: {installed}"
    )


@pytest.mark.integration
def test_corpus_slice_matches_committed_fixture(
    committed_fixture: dict[str, Any], live_snapshot: dict[str, Any]
) -> None:
    """The live corpus slice must match the committed fixture's declared slice.

    Guards against a package being added to / removed from the dependency closure
    without regenerating the fixture (which would make the oracle stale).
    """
    live_pkgs = live_snapshot["header"]["corpus"]["contracts_by_package"]
    committed_pkgs = committed_fixture["header"]["corpus"]["contracts_by_package"]
    assert set(live_pkgs) == set(committed_pkgs), (
        "Corpus package set drifted from the committed fixture. "
        f"live={sorted(live_pkgs)} committed={sorted(committed_pkgs)}. "
        "Regenerate baseline-selection-v2.json if the dependency closure changed."
    )


@pytest.mark.integration
def test_expected_exclusions_documented(
    committed_fixture: dict[str, Any],
) -> None:
    """Every sibling package outside the closure must be a DOCUMENTED exclusion.

    Zero silent exclusions (design D6): each expected-excluded package appears in the
    fixture header with a non-empty reason.
    """
    exclusions = {e["package"]: e for e in committed_fixture["header"]["exclusions"]}
    for pkg in harness.EXPECTED_EXCLUDED_ONEX_NODES_PACKAGES:
        assert pkg in exclusions, f"Expected-excluded package {pkg} not documented"
        assert exclusions[pkg]["reason"].strip(), f"Empty exclusion reason for {pkg}"


# ---------------------------------------------------------------------------
# Mode A — regression pin. The live engine vs the committed oracle, per D2 tuple.
# ---------------------------------------------------------------------------


def _tuple(sel: dict[str, Any]) -> tuple[Any, ...]:
    """The design-D2 equivalence tuple, hashable + order-preserving."""
    return (
        sel["status"],
        tuple(sel["dispatcher_ids"]),
        sel["message_category"],
        sel["message_type"],
        sel["dlq_topic"],
    )


@pytest.mark.integration
def test_mode_a_selection_parity_against_committed_oracle(
    committed_fixture: dict[str, Any], live_snapshot: dict[str, Any]
) -> None:
    """Mode A: every probe's live selection tuple equals the committed oracle.

    This is the S0 GO signal. A single drift = a selection-semantics regression the
    seam work must not silently carry. The diff is emitted probe-by-probe so a
    failure names exactly which selection changed.
    """
    committed = committed_fixture["probes"]
    live = live_snapshot["probes"]

    committed_ids = set(committed)
    live_ids = set(live)
    missing = sorted(committed_ids - live_ids)
    added = sorted(live_ids - committed_ids)
    assert not missing and not added, (
        "Probe set drifted from the committed oracle. "
        f"missing_from_live={missing[:20]} added_in_live={added[:20]}. "
        "Regenerate baseline-selection-v2.json (and review the diff) if this is "
        "an intentional corpus/probe change."
    )

    diffs: list[str] = []
    for pid in sorted(committed_ids & live_ids):
        c = _tuple(committed[pid]["selection"])
        li = _tuple(live[pid]["selection"])
        if c != li:
            diffs.append(f"  {pid}\n    committed={c}\n    live     ={li}")
    assert not diffs, (
        f"{len(diffs)} probe(s) drifted from the committed selection oracle:\n"
        + "\n".join(diffs[:40])
        + ("\n  ... (truncated)" if len(diffs) > 40 else "")
    )


@pytest.mark.integration
def test_p0_1_uniform_no_dispatcher_orchestrators_pinned(
    committed_fixture: dict[str, Any],
) -> None:
    """P8/D5 tripwire: the 4 uniformly-guard-tripped orchestrators stay NO_DISPATCHER.

    The 2026-07-02 live trace CONFIRMED node_rsd_orchestrator routes ZERO dispatchers
    and falls through to DLQ. The static oracle REFINES design D5: only 4 of the 6
    are uniformly NO_DISPATCHER on every topic; 2 (chain / registration) register
    some routes and are tracked separately (see
    ``test_p0_1_mixed_orchestrators_documented``). This tripwire pins the uniform
    set — if a future change wires any of them, THIS assertion flips first. It must
    NOT encode the (false) ticket-comment claim that they "payload-route at runtime".
    """
    p0 = committed_fixture["header"]["p0_outcomes"]["P0-1_guard_tripped_orchestrators"]
    assert p0["verdict"] == "NO_DISPATCHER_DLQ_REFINED"
    uniform = set(p0["uniform_no_dispatcher_orchestrators"])

    offenders: list[str] = []
    for pid, probe in committed_fixture["probes"].items():
        did = probe.get("dispatcher_id", "")
        if any(orch in did for orch in uniform):
            if probe["selection"]["status"] != "no_dispatcher":
                offenders.append(f"{pid} -> {probe['selection']['status']}")
    assert not offenders, (
        "A uniformly-guard-tripped orchestrator now routes (P0-1 tripwire fired — a "
        "wiring path changed; re-run the live trace and update the oracle "
        "deliberately):\n  " + "\n  ".join(offenders[:20])
    )


@pytest.mark.integration
def test_p0_1_mixed_orchestrators_documented(
    committed_fixture: dict[str, Any],
) -> None:
    """P8/D5: the 2 mixed-routing orchestrators are documented AND actually mixed.

    node_chain_orchestrator and node_registration_orchestrator each have BOTH
    routing (success) and non-routing (no_dispatcher) topics in this corpus slice —
    the evidence that corrected design D5's blanket over-generalization. This test
    pins that mixed reality: each must show both statuses, so a regression that
    silently makes them uniform (either all-route or all-drop) is caught.
    """
    p0 = committed_fixture["header"]["p0_outcomes"]["P0-1_guard_tripped_orchestrators"]
    mixed = p0["mixed_routing_orchestrators"]
    for orch in mixed:
        statuses = {
            probe["selection"]["status"]
            for probe in committed_fixture["probes"].values()
            if orch in probe.get("dispatcher_id", "")
        }
        assert "success" in statuses and "no_dispatcher" in statuses, (
            f"{orch} is documented as mixed-routing but the oracle shows only "
            f"{sorted(statuses)}. Re-run the live trace and correct the P0-1 record "
            "if its routing behavior changed."
        )


# ---------------------------------------------------------------------------
# Mode B — dual-implementation parity (LIVE as of OMN-12549).
# ---------------------------------------------------------------------------

# Parameterized by implementation. Mode A drives MessageDispatchEngine (the oracle
# builder); Mode B drives MixinNodeDispatch behind ProtocolDispatchEngine. As of
# OMN-12549 the mixin exists, so Mode B is a LIVE parameter point (no longer
# skipped) and the same tuple-equality assertion applies to both impls.
_MODE_B_UNAVAILABLE = (
    "MixinNodeDispatch seam not importable — OMN-12549 core dependency missing "
    "from this venv (expected only if omnibase_core predates the S0 seam)"
)


def _mixin_node_dispatch_available() -> bool:
    """True once OMN-12549 introduces MixinNodeDispatch behind ProtocolDispatchEngine.

    Guards Mode B against a stale omnibase_core in the venv (e.g. a released core
    older than the S0 seam). When the core dep carries the seam, Mode B is live.
    """
    try:
        import importlib

        importlib.import_module("omnibase_core.runtime.mixin_node_dispatch")
        return True
    except Exception:  # noqa: BLE001 — absence means a pre-seam core dependency
        return False


@pytest.fixture(scope="module")
def mixin_snapshot() -> dict[str, Any]:
    """Regenerate MixinNodeDispatch's selection snapshot in-process (Mode B).

    Drives the same corpus + probe taxonomy as :func:`live_snapshot` through the
    node-owned mixin instead of the engine. Built once per module.
    """
    return harness.build_mixin_snapshot()


@pytest.mark.integration
@pytest.mark.parametrize(
    "implementation",
    [
        pytest.param("message_dispatch_engine", id="mode_a_engine"),
        pytest.param(
            "mixin_node_dispatch",
            marks=pytest.mark.skipif(
                not _mixin_node_dispatch_available(), reason=_MODE_B_UNAVAILABLE
            ),
            id="mode_b_mixin",
        ),
    ],
)
def test_dual_implementation_parity(
    implementation: str,
    committed_fixture: dict[str, Any],
    live_snapshot: dict[str, Any],
    mixin_snapshot: dict[str, Any],
) -> None:
    """Protocol-parameterized parity.

    Mode A (``mode_a_engine``): the engine snapshot matches the committed oracle.

    Mode B (``mode_b_mixin``): the MixinNodeDispatch snapshot — the same probe
    corpus driven through the node-owned mixin behind ``ProtocolDispatchEngine`` —
    matches the committed oracle tuple-for-tuple. This is OMN-12549's DoD: the
    mixin SELECTS exactly what the live engine selects for every probe.
    """
    snapshot = (
        live_snapshot if implementation == "message_dispatch_engine" else mixin_snapshot
    )
    committed = committed_fixture["probes"]
    actual = snapshot["probes"]
    assert set(committed) == set(actual), (
        f"{implementation}: probe set drift vs committed oracle. "
        f"missing={sorted(set(committed) - set(actual))[:20]} "
        f"added={sorted(set(actual) - set(committed))[:20]}"
    )
    diffs = [
        pid
        for pid in committed
        if _tuple(committed[pid]["selection"]) != _tuple(actual[pid]["selection"])
    ]
    assert not diffs, (
        f"{implementation}: {len(diffs)} probe(s) drifted from the committed "
        f"selection oracle: {diffs[:40]}"
    )


@pytest.mark.integration
def test_mode_b_is_live_not_skipped() -> None:
    """Adversarial guard: Mode B must be genuinely exercised, never silently skipped.

    OMN-12549's DoD is "Mode B green" — a green run where Mode B was skipped is a
    false pass. This fails loudly if the mixin seam is not importable in the venv,
    so a stale/pre-seam omnibase_core cannot masquerade as parity.
    """
    assert _mixin_node_dispatch_available(), (
        "MixinNodeDispatch (omnibase_core.runtime.mixin_node_dispatch) is not "
        "importable — Mode B would be SKIPPED, which is not a valid OMN-12549 pass. "
        "Ensure the venv's omnibase_core carries the S0 seam."
    )


@pytest.mark.integration
def test_mode_b_mixin_corpus_is_nonempty(mixin_snapshot: dict[str, Any]) -> None:
    """Mode B must exercise a real corpus — a zero-probe mixin snapshot is not parity."""
    assert mixin_snapshot["probes"], "Mode B mixin snapshot has zero probes"
