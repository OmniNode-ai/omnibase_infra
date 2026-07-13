# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-14489 — dispatch-ENTRYPOINT coverage for every contract-declared handler.

INVARIANT: a handler class named in a contract's ``handler_routing.handlers[]``
MUST expose a callable ``handle`` or ``handle_async``. Auto-wiring binds one of
them at wiring time (``_make_dispatch_callback``); a handler exposing NEITHER is
bound to ``_missing_handle``, which raises ``ModelOnexError`` on EVERY dispatch.

WHY THIS GATE EXISTS (the defect class it makes impossible):
``HandlerA2ATask`` (node_remote_agent_invoke_effect) was contract-declared, wired,
ingress-valid, and CI-green — while exposing only ``submit()`` / ``watch()`` and no
dispatch entrypoint. The A2A remote-agent branch therefore passed ingress validation
and died at the FIRST dispatch. Nothing caught it:

  * contract tests   — assert the contract's declared models, never that the handler
                       can be INVOKED;
  * OMN-14488 route-coverage (`test_dispatcher_route_coverage_omn14488.py`) — proves a
                       payload ROUTES to a registered dispatcher, but a dispatcher bound
                       to ``_missing_handle`` still registers and still routes, and that
                       gate's `_FakeHandler` *has* a ``handle``, so it is blind here;
  * OMN-14488 operation_match ingress coverage — validates the declared ``input_model``
                       against the producer payload, which is upstream of dispatch.

"Registered" and "routable" are NOT "executable". This gate closes the third seam.

RATCHET: ``_KNOWN_ENTRYPOINTLESS`` freezes the 44 handlers that already carried this
defect when the gate landed (a platform-wide census: 44 of 158 contract-declared
handlers). It is a burn-down list, not an exemption:

  * a handler NOT on the list with no entrypoint  -> FAIL (no new instances, ever);
  * a handler ON the list that GAINS an entrypoint -> FAIL until it is removed from the
    list (the allowlist can only shrink — it can never go stale).

The end-state is an empty allowlist, at which point ``_missing_handle`` becomes a
wiring-time hard failure instead of a per-message one. Burn-down: OMN-14510.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

# tests/unit/runtime/auto_wiring/<this file> -> repo root is 4 parents up.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRC_ROOT = _REPO_ROOT / "src" / "omnibase_infra"

# Frozen census taken when this gate landed (OMN-14489). SHRINK ONLY.
_KNOWN_ENTRYPOINTLESS: frozenset[tuple[str, str]] = frozenset(
    {
        ("handler_architecture_validation", "HandlerArchitectureValidation"),
        # HandlerLedgerProjection removed from the shrink-only census (OMN-14516):
        # it now exposes a handle() dispatch entrypoint, so it is no longer
        # entrypointless. Do not re-add.
        ("node_artifact_change_detector_effect", "HandlerContractFileWatcher"),
        ("node_artifact_change_detector_effect", "HandlerManualTrigger"),
        ("node_artifact_change_detector_effect", "HandlerPRWebhookIngestion"),
        ("node_artifact_reconciliation_orchestrator", "HandlerPlanToPRComment"),
        ("node_artifact_reconciliation_orchestrator", "HandlerPlanToYaml"),
        ("node_auth_gate_compute", "HandlerAuthGate"),
        ("node_broker_disk_watermark_compute", "HandlerBrokerDiskWatermark"),
        ("node_chain_verify_reducer", "HandlerChainVerify"),
        ("node_checkpoint_effect", "HandlerCheckpointList"),
        ("node_checkpoint_effect", "HandlerCheckpointRead"),
        ("node_checkpoint_effect", "HandlerCheckpointWrite"),
        ("node_checkpoint_validate_compute", "HandlerCheckpointValidate"),
        ("node_contract_validate_compute", "HandlerContractValidate"),
        ("node_impact_analyzer_compute", "HandlerImpactAnalysis"),
        ("node_invariant_evaluate_compute", "HandlerInvariantEvaluate"),
        # (node_ledger_projection_compute, HandlerLedgerProjection) removed with the
        # entry above — HandlerLedgerProjection gained handle() in OMN-14516.
        ("node_llm_embedding_effect", "HandlerEmbeddingOpenaiCompatible"),
        ("node_llm_inference_effect", "HandlerLlmCliSubprocess"),
        ("node_model_health_effect", "HandlerProbeHealth"),
        ("node_model_router_compute", "HandlerScoreModels"),
        ("node_registration_storage_effect", "HandlerRegistrationStorageMock"),
        ("node_registration_storage_effect", "HandlerRegistrationStoragePostgres"),
        ("node_reward_binder_effect", "HandlerRewardBinder"),
        ("node_routing_orchestrator", "HandlerHealthComplete"),
        ("node_routing_orchestrator", "HandlerRoutingInitiate"),
        ("node_routing_orchestrator", "HandlerScoringComplete"),
        ("node_routing_score_reducer", "HandlerUpdateScores"),
        ("node_runtime_source_attestor_effect", "HandlerSourceAttestation"),
        ("node_setup_infisical_effect", "HandlerInfisicalFullSetup"),
        ("node_setup_local_provision_effect", "HandlerLocalProvision"),
        ("node_setup_local_provision_effect", "HandlerLocalStatus"),
        ("node_setup_local_provision_effect", "HandlerLocalTeardown"),
        ("node_setup_preflight_effect", "HandlerPreflightCheck"),
        ("node_setup_validate_effect", "HandlerServiceValidate"),
        ("node_topic_migration_executor_effect", "HandlerTopicMigrationExecutor"),
        ("node_topic_migration_projection", "HandlerTopicMigrationProjection"),
        ("node_update_plan_reducer", "HandlerCreatePlan"),
        (
            "node_validation_ledger_projection_compute",
            "HandlerValidationLedgerProjection",
        ),
    }
)


def _has_dispatch_entrypoint(cls: type) -> bool:
    """The exact predicate auto-wiring uses to bind a dispatch entrypoint."""
    return callable(getattr(cls, "handle_async", None)) or callable(
        getattr(cls, "handle", None)
    )


def _declared_handlers() -> list[tuple[str, str, str]]:
    """Every (contract_name, handler_name, handler_module) declared in the repo."""
    declared: list[tuple[str, str, str]] = []
    for contract_path in sorted(_SRC_ROOT.rglob("contract.yaml")):
        data = yaml.safe_load(contract_path.read_text())
        if not isinstance(data, dict):
            continue
        contract_name = str(data.get("name") or contract_path.parent.name)
        routing = data.get("handler_routing") or {}
        for entry in routing.get("handlers") or []:
            handler = (entry or {}).get("handler") or {}
            name, module = handler.get("name"), handler.get("module")
            if name and module:
                declared.append((contract_name, str(name), str(module)))
    return declared


def _entrypointless() -> set[tuple[str, str]]:
    """Contract-declared handlers that expose NEITHER handle nor handle_async."""
    missing: set[tuple[str, str]] = set()
    for contract_name, name, module in _declared_handlers():
        try:
            cls = getattr(importlib.import_module(module), name)
        except Exception:  # noqa: BLE001 — import health is a separate gate
            continue
        if not _has_dispatch_entrypoint(cls):
            missing.add((contract_name, name))
    return missing


@pytest.mark.unit
def test_contracts_were_actually_scanned() -> None:
    """Non-vacuity: an empty/failed scan must not make the gate below pass silently."""
    declared = _declared_handlers()
    assert len(declared) > 100, (
        f"Only {len(declared)} contract-declared handlers found — the contract scan is "
        f"broken (expected >100 under {_SRC_ROOT}). A gate over an empty set is vacuous."
    )


@pytest.mark.unit
def test_no_new_entrypointless_handlers() -> None:
    """LOAD-BEARING: a contract-declared handler MUST expose handle/handle_async.

    Goes RED against the EXISTS-but-WRONG state: HandlerA2ATask was declared, wired,
    and green while exposing only submit()/watch(). It is deliberately NOT allowlisted.
    """
    violations = _entrypointless() - _KNOWN_ENTRYPOINTLESS
    assert not violations, (
        "Contract-declared handler(s) expose NEITHER handle() nor handle_async(). "
        "Auto-wiring binds these to _missing_handle, so EVERY dispatch raises "
        "ModelOnexError at runtime while CI stays green:\n"
        + "\n".join(f"  - {c}: {h}" for c, h in sorted(violations))
        + "\n\nAdd a def-B `handle(request) -> response` entrypoint. Do NOT add the "
        "handler to _KNOWN_ENTRYPOINTLESS — that list is frozen and shrink-only."
    )


@pytest.mark.unit
def test_allowlist_is_shrink_only_and_never_stale() -> None:
    """A handler that GAINS an entrypoint must be REMOVED from the allowlist.

    This is what makes the list a ratchet rather than a permanent exemption: it can
    only shrink, and it can never go stale (a fixed handler still listed = FAIL).
    """
    stale = _KNOWN_ENTRYPOINTLESS - _entrypointless()
    assert not stale, (
        "These handlers now HAVE a dispatch entrypoint but are still listed in "
        "_KNOWN_ENTRYPOINTLESS. Remove them — the allowlist is shrink-only:\n"
        + "\n".join(f"  - {c}: {h}" for c, h in sorted(stale))
    )


@pytest.mark.unit
def test_entrypoint_predicate_discriminates() -> None:
    """Discriminator: the predicate must actually distinguish the two states.

    Without this, a predicate that returned True unconditionally would make every
    assertion above vacuously green — the exact failure mode this gate exists to catch.
    """

    class _NoEntrypoint:
        async def submit(self, command: object) -> None: ...  # the HandlerA2ATask shape

    class _DefB:
        async def handle(self, request: object) -> None: ...

    class _AsyncVariant:
        async def handle_async(self, request: object) -> None: ...

    assert not _has_dispatch_entrypoint(_NoEntrypoint)
    assert _has_dispatch_entrypoint(_DefB)
    assert _has_dispatch_entrypoint(_AsyncVariant)
