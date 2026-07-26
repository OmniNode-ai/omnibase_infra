# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""OMN-14817 — HandlerImpactAnalysis exposes the canonical def-B ``handle`` entrypoint.

RED-against-EXISTS-but-WRONG proof for the canonical-shape flip (hand-flip, OMN-14781).

Before this ticket ``HandlerImpactAnalysis`` was contract-declared and CI-green while
exposing ONLY a two-positional op-method ``analyze(trigger, registry)`` and NO
``handle`` — so the ratchet classified it ``op_method`` (non-canonical), it sat in
``canonical_handler_shape_baseline.py`` NON_CANONICAL and in
``handler_dispatch_entrypoint_baseline.yaml`` known_entrypointless, and auto-wiring's
``_make_dispatch_callback`` would bind ``_missing_handle`` (raising ``ModelOnexError``
on every dispatch).

``test_handler_exposes_handle_entrypoint`` FAILS against the pre-flip handler (no
``handle`` / a retained ``analyze``) and passes only once the def-B ``handle`` exists
with ``analyze`` removed — a hand-flip that wraps the two inputs into one
``ModelImpactAnalysisRequest`` and preserves the scoring helpers
(``_score_artifact`` / ``_count_matches`` / ``_assign_action`` / ``_level_to_policy`` /
``_pattern_to_reason_code``) byte-identical base_ref↔HEAD, which the canonical-shape
ratchet re-derives from git (the ``.handflip.json`` proof).

``CORPUS_REQUESTS`` below is the deterministic SELECTED input corpus bound (by
``input_hash``) into both the adequacy receipt
(``omnibase_infra.nodes.node_impact_analyzer_compute.json``) and the hand-flip proof
(``...handflip.json``) under ``scripts/ci/adequacy_receipts/``. Each parity case asserts
the def-B ``handle`` reproduces the exact OMN-3925 scoring the legacy ``analyze``
produced.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_artifact_change_detector_effect.models.model_update_trigger import (
    ModelUpdateTrigger,
)
from omnibase_infra.nodes.node_impact_analyzer_compute.handlers.handler_impact_analysis import (
    HandlerImpactAnalysis,
)
from omnibase_infra.nodes.node_impact_analyzer_compute.models.model_impact_analysis_request import (
    ModelImpactAnalysisRequest,
)
from omnibase_infra.registry.models.model_artifact_registry import ModelArtifactRegistry
from omnibase_infra.registry.models.model_artifact_registry_entry import (
    ModelArtifactRegistryEntry,
)
from omnibase_infra.registry.models.model_source_trigger import ModelSourceTrigger

pytestmark = [pytest.mark.unit]

# Deterministic timestamp so the SELECTED corpus hashes reproducibly — the adequacy
# receipt + hand-flip proof pin these exact payloads via input_hash.
_TS = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def _uid(n: int) -> UUID:
    return UUID(int=n)


def _trig(
    n: int,
    trigger_type: str,
    changed_files: list[str],
    source_repo: str = "omnibase_infra",
) -> ModelUpdateTrigger:
    return ModelUpdateTrigger(
        trigger_id=_uid(n),
        trigger_type=trigger_type,  # type: ignore[arg-type]
        source_repo=source_repo,
        changed_files=changed_files,
        timestamp=_TS,
    )


def _entry(
    n: int,
    update_policy: str,
    patterns: list[str],
    repo: str = "omnibase_infra",
) -> ModelArtifactRegistryEntry:
    return ModelArtifactRegistryEntry(
        artifact_id=_uid(n),
        artifact_type="doc",
        title="Corpus Doc",
        path="docs/corpus.md",
        repo=repo,
        update_policy=update_policy,  # type: ignore[arg-type]
        source_triggers=[
            ModelSourceTrigger(pattern=p, change_scope="structural") for p in patterns
        ],
    )


def _reg(*entries: ModelArtifactRegistryEntry) -> ModelArtifactRegistry:
    return ModelArtifactRegistry(
        version="1.0.0", description="OMN-14817 corpus", artifacts=list(entries)
    )


def _req(
    trigger: ModelUpdateTrigger, registry: ModelArtifactRegistry
) -> ModelImpactAnalysisRequest:
    return ModelImpactAnalysisRequest(trigger=trigger, registry=registry)


# Expected: (impacted_count, highest_merge_policy, first_strength_or_None,
#            first_action_or_None, reason_subset)
_Expected = tuple[int, str, float | None, str | None, frozenset[str]]

_CASES: list[tuple[str, ModelImpactAnalysisRequest, _Expected]] = [
    (
        "C1_contract_regenerate",
        _req(
            _trig(
                1, "contract_changed", ["src/omnibase_infra/nodes/node_x/contract.yaml"]
            ),
            _reg(_entry(10, "warn", ["src/omnibase_infra/nodes/*/contract.yaml"])),
        ),
        (1, "warn", 1.0, "regenerate", frozenset({"contract_yaml_changed"})),
    ),
    (
        "C2_no_match",
        _req(
            _trig(2, "pr_opened", ["tests/unit/test_smoke.py"]),
            _reg(_entry(11, "warn", ["src/omnibase_infra/nodes/*/contract.yaml"])),
        ),
        (0, "none", None, None, frozenset()),
    ),
    (
        "C3_review_half",
        _req(
            _trig(
                3, "contract_changed", ["src/omnibase_infra/nodes/node_x/contract.yaml"]
            ),
            _reg(
                _entry(
                    12,
                    "warn",
                    [
                        "src/omnibase_infra/nodes/*/contract.yaml",
                        "docs/architecture/*.md",
                    ],
                )
            ),
        ),
        (1, "warn", 0.5, "review", frozenset({"contract_yaml_changed"})),
    ),
    (
        "C4_strict_floor",
        _req(
            _trig(4, "pr_opened", ["src/omnibase_infra/nodes/node_x/contract.yaml"]),
            _reg(
                _entry(
                    13,
                    "strict",
                    [
                        "src/omnibase_infra/nodes/*/contract.yaml",
                        "docs/architecture/*.md",
                        "src/omnibase_infra/scripts/*.sh",
                        "src/omnibase_infra/config/*.yaml",
                    ],
                )
            ),
        ),
        (1, "strict", 0.5, "review", frozenset({"contract_yaml_changed"})),
    ),
    (
        "C5_require_regenerate",
        _req(
            _trig(
                5, "contract_changed", ["src/omnibase_infra/nodes/node_x/contract.yaml"]
            ),
            _reg(_entry(14, "require", ["src/omnibase_infra/nodes/*/contract.yaml"])),
        ),
        (1, "require", 1.0, "regenerate", frozenset({"contract_yaml_changed"})),
    ),
    (
        "C6_highest_strict",
        _req(
            _trig(
                6, "contract_changed", ["src/omnibase_infra/nodes/node_x/contract.yaml"]
            ),
            _reg(
                _entry(15, "warn", ["src/omnibase_infra/nodes/*/contract.yaml"]),
                _entry(16, "strict", ["src/omnibase_infra/nodes/*/contract.yaml"]),
            ),
        ),
        (2, "strict", 1.0, "regenerate", frozenset({"contract_yaml_changed"})),
    ),
    (
        "C7_manual_reconciliation",
        _req(
            _trig(7, "manual_plan_request", [], source_repo="omnibase_infra"),
            _reg(
                _entry(
                    17,
                    "warn",
                    ["src/omnibase_infra/nodes/*/contract.yaml"],
                    repo="omnibase_infra",
                ),
                _entry(18, "strict", ["src/other/contract.yaml"], repo="other_repo"),
            ),
        ),
        (1, "warn", 0.7, "review", frozenset({"manual_reconciliation"})),
    ),
    (
        "C8_none_policy_filtered",
        _req(
            _trig(8, "pr_opened", ["src/omnibase_infra/nodes/node_x/contract.yaml"]),
            _reg(
                _entry(
                    19,
                    "none",
                    [
                        "src/omnibase_infra/nodes/*/contract.yaml",
                        "docs/architecture/*.md",
                        "src/scripts/*.sh",
                        "src/config/*.yaml",
                    ],
                )
            ),
        ),
        (0, "none", None, None, frozenset()),
    ),
    (
        "C9_empty_registry",
        _req(
            _trig(
                9, "contract_changed", ["src/omnibase_infra/nodes/node_x/contract.yaml"]
            ),
            _reg(),
        ),
        (0, "none", None, None, frozenset()),
    ),
    (
        "C10_reason_codes_all",
        _req(
            _trig(
                20,
                "contract_changed",
                [
                    "src/omnibase_infra/nodes/node_x/handlers/handler_y.py",
                    "schemas/foo.json",
                    "scripts/deploy.sh",
                    "config/event_bus_main.yaml",
                    "config/topics_core.yaml",
                    "tests/unit/test_z.py",
                    "docs/misc_readme.md",
                ],
            ),
            _reg(
                _entry(
                    21,
                    "warn",
                    [
                        "src/omnibase_infra/nodes/*/handlers/handler_*.py",
                        "schemas/*.json",
                        "scripts/*.sh",
                        "config/event_bus_*.yaml",
                        "config/topics_*.yaml",
                        "tests/unit/*.py",
                        "docs/misc_*.md",
                    ],
                )
            ),
        ),
        (
            1,
            "warn",
            1.0,
            "regenerate",
            frozenset(
                {
                    "handler_routing_changed",
                    "schema_changed",
                    "script_changed",
                    "event_bus_topics_changed",
                    "config_changed",
                }
            ),
        ),
    ),
    (
        "C11_require_low_review",
        _req(
            _trig(22, "pr_opened", ["src/omnibase_infra/nodes/node_x/contract.yaml"]),
            _reg(
                _entry(
                    23,
                    "require",
                    [
                        "src/omnibase_infra/nodes/*/contract.yaml",
                        "docs/a/*.md",
                        "docs/b/*.md",
                        "docs/c/*.md",
                    ],
                )
            ),
        ),
        (1, "require", 0.3, "review", frozenset({"contract_yaml_changed"})),
    ),
    (
        "C12_empty_source_triggers",
        _req(
            _trig(
                24,
                "contract_changed",
                ["src/omnibase_infra/nodes/node_x/contract.yaml"],
            ),
            _reg(
                _entry(25, "warn", []),
                _entry(26, "warn", ["src/omnibase_infra/nodes/*/contract.yaml"]),
            ),
        ),
        (1, "warn", 1.0, "regenerate", frozenset({"contract_yaml_changed"})),
    ),
]

_CASE_IDS = [c[0] for c in _CASES]

# The deterministic candidate pool consumed by the adequacy recorder (scripts/ci
# recorder for OMN-14817) — order-stable so input_hash selection reproduces.
CORPUS_REQUESTS: list[ModelImpactAnalysisRequest] = [c[1] for c in _CASES]


@pytest.mark.unit
def test_handler_exposes_handle_entrypoint() -> None:
    """The bare def-B invariant: auto-wiring can only bind handle/handle_async.

    RED against the pre-OMN-14817 handler, which exposed only ``analyze``.
    """
    assert callable(getattr(HandlerImpactAnalysis, "handle", None)), (
        "HandlerImpactAnalysis exposes no handle(); auto-wiring binds _missing_handle "
        "and every dispatch raises ModelOnexError."
    )
    # The hand-flip removed the legacy two-arg op-method (no retained analyze, no shim).
    assert not hasattr(HandlerImpactAnalysis, "analyze"), (
        "analyze must be replaced by handle (no retained analyze, no delegating shim)."
    )


@pytest.mark.unit
def test_handle_single_typed_request_param() -> None:
    """handle() is a single-positional typed request (definition B adaptable)."""
    import inspect

    sig = inspect.signature(HandlerImpactAnalysis.handle)
    params = [p for p in sig.parameters if p != "self"]
    assert params == ["request"], params
    ann = sig.parameters["request"].annotation
    assert ann is ModelImpactAnalysisRequest or ann == "ModelImpactAnalysisRequest"


@pytest.mark.unit
@pytest.mark.parametrize(("case_id", "request_obj", "expected"), _CASES, ids=_CASE_IDS)
def test_handle_matches_scoring_table(
    case_id: str,
    request_obj: ModelImpactAnalysisRequest,
    expected: _Expected,
) -> None:
    """def-B handle() reproduces the OMN-3925 scoring for the selected corpus."""
    handler = HandlerImpactAnalysis()
    result = handler.handle(request_obj)

    exp_count, exp_policy, exp_strength, exp_action, exp_reasons = expected
    assert len(result.impacted_artifacts) == exp_count, case_id
    assert result.highest_merge_policy == exp_policy, case_id
    assert result.source_trigger_id == request_obj.trigger.trigger_id
    if exp_count:
        artifact = result.impacted_artifacts[0]
        assert exp_strength is not None
        assert abs(artifact.impact_strength - exp_strength) < 1e-9, case_id
        assert artifact.required_action == exp_action, case_id
        assert exp_reasons.issubset(set(artifact.reason_codes)), case_id


@pytest.mark.unit
def test_handler_classification_unchanged() -> None:
    """Handler classification is preserved across the flip."""
    handler = HandlerImpactAnalysis()
    assert handler.handler_type == EnumHandlerType.INFRA_HANDLER
    assert handler.handler_category == EnumHandlerTypeCategory.COMPUTE
    assert handler.handler_id == "handler-impact-analysis"
