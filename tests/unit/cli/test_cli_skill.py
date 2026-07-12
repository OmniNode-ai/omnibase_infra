# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``onex skill <name>`` (OMN-13097).

The skill subcommand resolves the declarative skill→node mapping
(``skill_mapping.yaml``), builds the backing node's input payload from the
skill's CLI args, and dispatches through the proven receipt-mode path. These
tests assert the DECLARATIVE LAYER (registry validity, arg parsing, payload
construction, classifiers) directly, plus the command wiring through a stubbed
``run_receipt_mode`` so we verify the constructed payload without standing up
a live runtime.
"""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from uuid import uuid4

import pytest
from click.testing import CliRunner

from omnibase_infra.cli import cli_skill
from omnibase_infra.cli.cli_skill import (
    _apply_classifiers,
    _parse_skill_args,
    load_skill_registry,
    run_skill_by_name,
)
from omnibase_infra.cli.enum_skill_arg_type import EnumSkillArgType
from omnibase_infra.cli.model_skill_arg_spec import ModelSkillArgSpec
from omnibase_infra.cli.model_skill_classifier import ModelSkillClassifier
from omnibase_infra.cli.model_skill_mapping import ModelSkillMapping
from omnibase_infra.cli.model_skill_mapping_registry import ModelSkillMappingRegistry

pytestmark = pytest.mark.unit

# The dispatch shims migrated to the declarative skill->node mapping.
# OMN-13097 seeded the first 24; `gap` was added once node_gap_compute landed
# in omnimarket but the CLI still rejected `onex skill gap` (this fix).
_EXPECTED_SKILLS = frozenset(
    {
        "aislop_sweep",
        "data_flow_sweep",
        # OMN-14151: auto_merge's legacy node_auto_merge_effect backing node is
        # deregistered; the active path is the merge-queue governor.
        "build_loop",
        "coderabbit_triage",
        "compliance_sweep",
        "coverage_sweep",
        "create_ticket",
        "database_sweep",
        "design_to_plan",
        "doc_freshness_sweep",
        "dod_verify",
        "duplication_sweep",
        # OMN-13995: node_dep_cascade_dedup_orchestrator was in the omnimarket
        # catalog but absent from skill_mapping.yaml, so
        # `onex skill dep_cascade_dedup` returned "Unknown skill".
        "dep_cascade_dedup",
        "gap",
        "hostile_reviewer",
        "linear_housekeeping",
        "merge_sweep",
        "plan_to_tickets",
        "platform_readiness",
        "pr_polish",
        "pr_review",
        "pr_review_bot",
        "session",
        "shim_audit",
        "delegate",
        # OMN-13511: the four dogfood runtime sweeps + skill_functional_audit
        # were node-backed but unregistered — `onex skill <name>` returned
        # "Unknown skill" so the sweeps were unrunnable headless.
        "runtime_sweep",
        "golden_chain_sweep",
        "integration_sweep",
        "contract_sweep",
        "skill_functional_audit",
        "dod_sweep",
        # OMN-13688: pipeline_fill / wave_scheduler node-backed but unregistered
        "pipeline_fill",
        "wave_scheduler",
    }
)


@pytest.fixture(autouse=True)
def _clear_registry_cache() -> None:
    """The registry loader is lru_cached; clear it per test for isolation."""
    load_skill_registry.cache_clear()


def test_registry_loads_and_validates() -> None:
    registry = load_skill_registry()
    assert isinstance(registry, ModelSkillMappingRegistry)
    assert len(registry.skills) >= len(_EXPECTED_SKILLS)


def test_registry_covers_all_24_dispatch_shims() -> None:
    registry = load_skill_registry()
    declared = {s.skill_name for s in registry.skills}
    missing = _EXPECTED_SKILLS - declared
    assert not missing, f"mapping is missing skills: {sorted(missing)}"


def test_registry_result_models_are_fully_qualified() -> None:
    registry = load_skill_registry()
    for skill in registry.skills:
        assert "." in skill.result_model, (
            f"{skill.skill_name}: result_model must be a fully qualified name, "
            f"got {skill.result_model!r}"
        )
        assert skill.node_name.startswith("node_"), skill.node_name


# --------------------------------------------------------------------------- #
# skill_mapping.yaml node_name -> real node catalog resolution (OMN-13531)
#
# Every node_name in skill_mapping.yaml MUST resolve to a node that actually
# exists in the canonical omnimarket entry-point catalog. The live failure that
# motivated this fixture: `onex skill pr_review_bot` dispatched to
# `node_pr_review_orchestrator`, but the resolver raised
# `Unknown node node_pr_review_orchestrator` because the mapping referenced a
# node that the installed/declared catalog did not carry (stale skill->node
# mapping after the OMN-13212 node_pr_review_bot decomposition). The existing
# tests only asserted `node_name.startswith("node_")` and monkeypatched
# `_resolve_packaged_contract`, so no test ever proved the target node exists.
#
# omnimarket is NOT a declared dependency of omnibase_infra and is not installed
# in this repo's CI, so resolving against the live `onex.nodes` entry points
# would be a false gate. Instead we read the canonical source of truth: the
# `[project.entry-points."onex.nodes"]` table in omnimarket's pyproject.toml
# (the same sibling-checkout pattern node-migration-sync.yml already uses for
# the omnimarket node tree). The check skips cleanly when the source is not
# resolvable locally; CI wires the sibling checkout so it runs as a real gate.
# --------------------------------------------------------------------------- #


def _resolve_omnimarket_src() -> Path | None:
    """Resolve the omnimarket repo root if available for the catalog check.

    Mirrors the resolution order used by node-migration-sync
    (tests/unit/migrations/test_node_migration_discovery.py): an explicit
    ``OMNIMARKET_SRC`` env (CI sibling checkout) first, then ``OMNI_HOME``.
    """
    explicit = os.environ.get("OMNIMARKET_SRC")
    if explicit and (Path(explicit) / "pyproject.toml").is_file():
        return Path(explicit)
    omni_home = os.environ.get("OMNI_HOME")
    if omni_home and (Path(omni_home) / "omnimarket" / "pyproject.toml").is_file():
        return Path(omni_home) / "omnimarket"
    return None


def _omnimarket_declared_nodes(omnimarket_root: Path) -> frozenset[str]:
    """Parse omnimarket's canonical ``onex.nodes`` entry-point declarations.

    This is the source-of-truth node catalog: ``onex node <name>`` /
    ``onex skill <name>`` resolve ``<name>`` via this exact entry-point group
    (see ``omnibase_infra.cli.cli_node._resolve_packaged_contract``).
    """
    data = tomllib.loads((omnimarket_root / "pyproject.toml").read_text())
    entry_points = data.get("project", {}).get("entry-points", {}).get("onex.nodes", {})
    return frozenset(entry_points.keys())


def test_every_mapped_node_resolves_in_omnimarket_catalog() -> None:
    """Every skill_mapping.yaml node_name exists in the canonical node catalog.

    Guards against the stale skill->node mapping class of bug (OMN-13531): a
    dispatch entry that points at a node which no longer exists in the catalog
    surfaces only at dispatch time as `Unknown node <name>`. This pins the
    whole table — pr_review, pr_review_bot, hostile_reviewer, and the rest.
    """
    omnimarket_root = _resolve_omnimarket_src()
    if omnimarket_root is None:
        pytest.skip(
            "omnimarket source tree not resolvable "
            "(set OMNIMARKET_SRC or OMNI_HOME); CI wires the sibling checkout"
        )

    declared_nodes = _omnimarket_declared_nodes(omnimarket_root)
    assert declared_nodes, (
        "parsed zero onex.nodes entry points from omnimarket pyproject.toml at "
        f"{omnimarket_root}; the catalog source-of-truth could not be read"
    )

    registry = load_skill_registry()
    unresolved = {
        skill.skill_name: skill.node_name
        for skill in registry.skills
        if skill.node_name not in declared_nodes
    }
    assert not unresolved, (
        "skill_mapping.yaml references node_name(s) absent from the canonical "
        "omnimarket onex.nodes catalog — `onex skill <name>` will fail with "
        f"`Unknown node ...` at dispatch time: {unresolved}"
    )


def test_pr_review_and_hostile_skills_target_existing_nodes() -> None:
    """Targeted regression for the OMN-13531 stale-mapping triplet.

    Explicitly proves pr_review_bot / pr_review / hostile_reviewer resolve to
    nodes the catalog actually carries, independent of the table-wide sweep so
    a regression on these three names is unambiguous in the failure output.
    """
    omnimarket_root = _resolve_omnimarket_src()
    if omnimarket_root is None:
        pytest.skip(
            "omnimarket source tree not resolvable "
            "(set OMNIMARKET_SRC or OMNI_HOME); CI wires the sibling checkout"
        )

    declared_nodes = _omnimarket_declared_nodes(omnimarket_root)
    registry = load_skill_registry()
    by_name = {s.skill_name: s for s in registry.skills}

    for skill_name in ("pr_review_bot", "pr_review", "hostile_reviewer"):
        mapping = by_name.get(skill_name)
        assert mapping is not None, f"{skill_name} missing from skill_mapping.yaml"
        # The decomposed-but-deleted shells must never reappear in the mapping.
        assert mapping.node_name not in {
            "node_pr_review_bot",
            "node_hostile_reviewer",
        }, (
            f"{skill_name} maps to the deleted shell node {mapping.node_name!r}; "
            "the OMN-13212 decomposition removed it from the catalog"
        )
        assert mapping.node_name in declared_nodes, (
            f"{skill_name} -> {mapping.node_name!r} is not a real node in the "
            "omnimarket onex.nodes catalog"
        )


# --------------------------------------------------------------------------- #
# Every node-backed skill MUST be registered in skill_mapping.yaml (OMN-13511)
#
# The inverse of the OMN-13531 gate above (mapping -> catalog). OMN-13531 proves
# every *mapped* node_name exists; this proves every *node-backed sweep skill*
# is mapped. The live failure that motivated this fixture: runtime_sweep,
# golden_chain_sweep, integration_sweep, and contract_sweep each had a real
# omnimarket backing node but were absent from skill_mapping.yaml, so
# `onex skill <name>` returned "Unknown skill" and the dogfood runtime sweeps
# were unrunnable headless.
#
# Source of truth is the canonical omnimarket onex.nodes catalog (same parse as
# OMN-13531), NOT a fragile SKILL.md grep. The map below pins each dogfood
# sweep + audit skill to its backing node; the test asserts that for every pair
# whose node is in the catalog, the skill is registered AND resolves. A new
# node-backed sweep that lands in omnimarket but is never wired into the
# mapping fails here at CI time instead of at dispatch time.
# --------------------------------------------------------------------------- #

# skill_name -> canonical omnimarket node_name for the dogfood verification
# family. These are the skills a session is told to dogfood (CLAUDE.md Rule 1:
# /onex:runtime_sweep, /onex:contract_sweep, etc.). Each entry must be a real
# node in the omnimarket catalog and a registered `onex skill`.
_NODE_BACKED_DOGFOOD_SKILLS: dict[str, str] = {
    "runtime_sweep": "node_runtime_sweep",
    "golden_chain_sweep": "node_golden_chain_sweep",
    "integration_sweep": "node_integration_sweep_orchestrator",
    "contract_sweep": "node_contract_sweep",
    "skill_functional_audit": "node_skill_functional_audit_compute",
    "compliance_sweep": "node_compliance_sweep",
    "data_flow_sweep": "node_data_flow_sweep",
    "database_sweep": "node_database_sweep",
    "dod_sweep": "node_dod_sweep_orchestrator",
    "coverage_sweep": "node_coverage_sweep",
    # OMN-13995: post-release dep-bump dedup sweep — node existed, unregistered.
    "dep_cascade_dedup": "node_dep_cascade_dedup_orchestrator",
}


def test_every_node_backed_sweep_skill_is_registered() -> None:
    """Every dogfood sweep with a real backing node is registered + resolves.

    Guards the OMN-13511 regression class (node exists, skill unregistered) for
    the dogfood verification family. Without this, a sweep can ship a backing
    node in omnimarket yet stay unrunnable via `onex skill <name>` because the
    declarative mapping was never updated — exactly how runtime_sweep,
    golden_chain_sweep, integration_sweep, and contract_sweep regressed.
    """
    omnimarket_root = _resolve_omnimarket_src()
    if omnimarket_root is None:
        pytest.skip(
            "omnimarket source tree not resolvable "
            "(set OMNIMARKET_SRC or OMNI_HOME); CI wires the sibling checkout"
        )

    declared_nodes = _omnimarket_declared_nodes(omnimarket_root)
    registry = load_skill_registry()
    by_name = {s.skill_name: s for s in registry.skills}

    unregistered: dict[str, str] = {}
    mismatched: dict[str, str] = {}
    for skill_name, node_name in _NODE_BACKED_DOGFOOD_SKILLS.items():
        # Only enforce registration for nodes the catalog actually carries; a
        # skill whose backing node was deleted/renamed is a different bug class
        # (covered by the OMN-13531 mapping->catalog gate).
        if node_name not in declared_nodes:
            continue
        mapping = by_name.get(skill_name)
        if mapping is None:
            unregistered[skill_name] = node_name
            continue
        if mapping.node_name != node_name:
            mismatched[skill_name] = (
                f"mapped to {mapping.node_name!r}, expected {node_name!r}"
            )

    assert not unregistered, (
        "node-backed dogfood sweep skill(s) absent from skill_mapping.yaml — "
        "`onex skill <name>` returns 'Unknown skill' so the sweep is unrunnable "
        f"headless (OMN-13511): {unregistered}"
    )
    assert not mismatched, (
        f"dogfood sweep skill(s) wired to the wrong backing node: {mismatched}"
    )


# --------------------------------------------------------------------------- #
# OMN-13712: node-proven WIRE-MAP skills converge onto `onex skill <name>`.
#
# Each skill below already had a working omnimarket backing node proven over
# the local in-memory bus (`onex node <name>`, migration=WIRE-MAP +
# verdict=WORKS-E2E in docs/evidence/2026-06-28-skill-e2e-foreground/matrix.md)
# but no skill_mapping entry, so the only documented invocation was the remote
# `onex run-node` path. Before this change `onex skill bus_audit` (etc.)
# returned "Unknown skill". This pins each skill to its backing node and proves
# the node resolves in the canonical omnimarket catalog.
# --------------------------------------------------------------------------- #
_OMN_13712_WIRE_MAP_SKILLS: dict[str, str] = {
    "agent_healthcheck": "node_worker_stall_recovery",
    # OMN-13925: env_parity re-wired from the pure compute primitive (which
    # only ever saw a static sample payload) to the live collection EFFECT.
    "env_parity": "node_env_parity_collect_effect",
    "bus_audit": "node_bus_audit_compute",
    "plan_audit": "node_plan_audit_compute",
    "recall": "node_recall_compute",
    "dispatch_watchdog": "node_dispatch_watchdog_orchestrator",
    "env_sync_alert": "node_env_sync_alert_effect",
    "resume_session": "node_resume_session_compute",
    "verification_receipt_generator": "node_verification_receipt_generator",
    "rrh": "node_rrh_compute",
    "rewind": "node_rewind_compute",
    "checkpoint": "node_checkpoint_compute",
    "insights_to_plan": "node_insights_to_plan_compute",
    "local_review": "node_local_review",
    "autopilot": "node_autopilot_orchestrator",
    "two_strike_arbiter": "node_two_strike_arbiter",
    "feature_dashboard": "node_feature_dashboard_compute",
}


def test_omn_13712_wire_map_skills_registered() -> None:
    """Every node-proven WIRE-MAP skill is registered + wired to its node.

    Reproducing assertion for OMN-13712: before the mapping entries existed,
    `onex skill bus_audit` (and the other 16) returned "Unknown skill" because
    the backing node was only reachable via the remote `onex run-node` path.
    """
    registry = load_skill_registry()
    by_name = {s.skill_name: s for s in registry.skills}

    missing = sorted(s for s in _OMN_13712_WIRE_MAP_SKILLS if s not in by_name)
    assert not missing, (
        "node-proven WIRE-MAP skill(s) absent from skill_mapping.yaml — "
        f"`onex skill <name>` returns 'Unknown skill': {missing}"
    )

    mismatched = {
        skill: f"mapped to {by_name[skill].node_name!r}, expected {node!r}"
        for skill, node in _OMN_13712_WIRE_MAP_SKILLS.items()
        if by_name[skill].node_name != node
    }
    assert not mismatched, f"WIRE-MAP skill(s) wired to the wrong node: {mismatched}"


def test_omn_13712_wire_map_nodes_resolve_in_catalog() -> None:
    """Every OMN-13712 backing node exists in the omnimarket onex.nodes catalog.

    Without this, a converged skill could ship pointing at a node the catalog
    does not carry and fail only at dispatch with `Unknown node <name>`.
    """
    omnimarket_root = _resolve_omnimarket_src()
    if omnimarket_root is None:
        pytest.skip(
            "omnimarket source tree not resolvable "
            "(set OMNIMARKET_SRC or OMNI_HOME); CI wires the sibling checkout"
        )

    declared_nodes = _omnimarket_declared_nodes(omnimarket_root)
    unresolved = {
        skill: node
        for skill, node in _OMN_13712_WIRE_MAP_SKILLS.items()
        if node not in declared_nodes
    }
    assert not unresolved, (
        "OMN-13712 skill(s) reference node_name(s) absent from the canonical "
        f"omnimarket onex.nodes catalog: {unresolved}"
    )


def test_env_parity_mapping_has_no_static_lane_snapshot() -> None:
    """OMN-13925 recurrence guard: env_parity must execute LIVE collection.

    The defect class: the mapping carried a static two-lane ``env_by_lane``
    sample in ``static_payload``, so ``onex skill env_parity`` emitted a
    sample-data verdict (status=success over zero live entities) masquerading
    as live lane parity. The skill must dispatch the live collection EFFECT
    and the mapping must never bake an env snapshot into the payload again —
    snapshots are collected at run time or the node fails fast stating that
    no live collection input was provided.
    """
    registry = load_skill_registry()
    mapping = next((s for s in registry.skills if s.skill_name == "env_parity"), None)
    assert mapping is not None, "env_parity missing from skill_mapping.yaml"
    assert mapping.node_name == "node_env_parity_collect_effect", (
        f"env_parity must dispatch the live collection effect, got "
        f"{mapping.node_name!r}"
    )
    assert "env_by_lane" not in mapping.static_payload, (
        "env_parity static_payload smuggles a static env_by_lane snapshot — "
        "the OMN-13925 sample-data defect has recurred"
    )
    assert not any(
        isinstance(value, dict) for value in mapping.static_payload.values()
    ), (
        "env_parity static_payload carries a nested mapping; env snapshots "
        "must be collected live, never baked into the dispatch mapping"
    )
    assert "ModelEnvParityCollectResult" in mapping.result_model, (
        "env_parity result_model must be the collect receipt (provenance + "
        f"parity), got {mapping.result_model!r}"
    )


def test_registry_rejects_duplicate_skill_names() -> None:
    with pytest.raises(ValueError, match="duplicate skill_name"):
        ModelSkillMappingRegistry(
            skills=(
                ModelSkillMapping(
                    skill_name="dup",
                    node_name="node_x",
                    result_model="a.B",
                ),
                ModelSkillMapping(
                    skill_name="dup",
                    node_name="node_y",
                    result_model="a.C",
                ),
            )
        )


def test_mapping_rejects_multiple_positionals() -> None:
    with pytest.raises(ValueError, match="at most one positional"):
        ModelSkillMapping(
            skill_name="s",
            node_name="node_x",
            result_model="a.B",
            args=(
                ModelSkillArgSpec(
                    name="a",
                    payload_field="a",
                    arg_type=EnumSkillArgType.STRING,
                    positional=True,
                ),
                ModelSkillArgSpec(
                    name="b",
                    payload_field="b",
                    arg_type=EnumSkillArgType.STRING,
                    positional=True,
                ),
            ),
        )


def test_arg_spec_required_forbids_default() -> None:
    with pytest.raises(ValueError, match="required args must not declare a default"):
        ModelSkillArgSpec(
            name="x",
            payload_field="x",
            arg_type=EnumSkillArgType.STRING,
            required=True,
            default="oops",
        )


def _mapping_with(*args: ModelSkillArgSpec, **kw: object) -> ModelSkillMapping:
    return ModelSkillMapping(
        skill_name="t",
        node_name="node_t",
        result_model="a.B",
        args=tuple(args),
        **kw,
    )


def test_parse_string_list_and_boolean() -> None:
    mapping = _mapping_with(
        ModelSkillArgSpec(
            name="repos",
            payload_field="repos",
            arg_type=EnumSkillArgType.STRING_LIST,
        ),
        ModelSkillArgSpec(
            name="dry-run",
            payload_field="dry_run",
            arg_type=EnumSkillArgType.BOOLEAN,
            default=False,
        ),
    )
    payload = _parse_skill_args(mapping, ("--repos", "a, b ,c", "--dry-run"))
    assert payload == {"repos": ["a", "b", "c"], "dry_run": True}


def test_parse_applies_boolean_default_when_omitted() -> None:
    mapping = _mapping_with(
        ModelSkillArgSpec(
            name="dry-run",
            payload_field="dry_run",
            arg_type=EnumSkillArgType.BOOLEAN,
            default=False,
        ),
    )
    assert _parse_skill_args(mapping, ()) == {"dry_run": False}


def test_parse_integer_coercion_and_failure() -> None:
    mapping = _mapping_with(
        ModelSkillArgSpec(
            name="pr-number",
            payload_field="pr_number",
            arg_type=EnumSkillArgType.INTEGER,
        ),
    )
    assert _parse_skill_args(mapping, ("--pr-number", "42")) == {"pr_number": 42}
    from click import ClickException

    with pytest.raises(ClickException, match="expects an integer"):
        _parse_skill_args(mapping, ("--pr-number", "notanint"))


def test_pr_review_declarative_path_supplies_reviewer_and_judge_defaults() -> None:
    """OMN-13719 regression: ``onex skill pr_review --repo X --pr-number N`` builds a
    valid ReviewRequest payload from the real registry.

    Before the fix, ``reviewer-models`` / ``judge-model`` had no declarative default,
    so the convenience path produced a payload missing ``reviewer_models`` (a
    ``list[str]`` the runtime run-identity injection does NOT backfill), and the
    backing ``ReviewRequest`` failed validation with::

        1 validation error for ReviewRequest
        reviewer_models  Field required [type=missing, ...]

    Both ``pr_review`` and ``pr_review_bot`` share node_pr_review_orchestrator and the
    same input gap, so both are asserted here.
    """
    registry = load_skill_registry()
    by_name = {s.skill_name: s for s in registry.skills}

    for skill_name in ("pr_review", "pr_review_bot"):
        mapping = by_name.get(skill_name)
        assert mapping is not None, f"{skill_name} missing from skill_mapping.yaml"
        payload = _parse_skill_args(
            mapping,
            ("--repo", "OmniNode-ai/omnimarket", "--pr-number", "1505", "--dry-run"),
        )
        # The required ReviewRequest fields the declarative path must now supply.
        assert payload["repo"] == "OmniNode-ai/omnimarket"
        assert payload["pr_number"] == 1505
        assert payload["dry_run"] is True
        # The previously-missing fields are now defaulted (local-first).
        assert payload["reviewer_models"] == ["local"], (
            f"{skill_name}: reviewer_models default missing — the declarative path "
            "cannot build a valid ReviewRequest (OMN-13719 regression)"
        )
        assert payload["judge_model"] == "local"


def test_parse_positional_joins_tokens() -> None:
    mapping = _mapping_with(
        ModelSkillArgSpec(
            name="prompt",
            payload_field="prompt",
            arg_type=EnumSkillArgType.STRING,
            positional=True,
            required=True,
        ),
    )
    assert _parse_skill_args(mapping, ("summarize", "this", "text")) == {
        "prompt": "summarize this text"
    }


def test_parse_missing_required_fails() -> None:
    from click import ClickException

    mapping = _mapping_with(
        ModelSkillArgSpec(
            name="repo",
            payload_field="repo",
            arg_type=EnumSkillArgType.STRING,
            required=True,
        ),
    )
    with pytest.raises(ClickException, match="requires --repo"):
        _parse_skill_args(mapping, ())


def test_parse_unknown_flag_fails() -> None:
    from click import ClickException

    mapping = _mapping_with()
    with pytest.raises(ClickException, match="Unknown argument '--nope'"):
        _parse_skill_args(mapping, ("--nope", "x"))


def test_static_payload_merged() -> None:
    mapping = _mapping_with(
        ModelSkillArgSpec(
            name="prompt",
            payload_field="prompt",
            arg_type=EnumSkillArgType.STRING,
            positional=True,
            required=True,
        ),
        static_payload={"source": "claude-code"},
    )
    payload = _parse_skill_args(mapping, ("hello",))
    assert payload == {"source": "claude-code", "prompt": "hello"}


def test_classifier_assigns_first_match() -> None:
    classifier = ModelSkillClassifier(
        target_field="task_type",
        source_field="prompt",
        rules=(
            (("test", "pytest"), "test"),
            (("write", "implement"), "code_generation"),
        ),
        fallback="research",
    )
    mapping = _mapping_with(classifiers=(classifier,))
    payload: dict[str, object] = {"prompt": "please write a pytest"}
    _apply_classifiers(mapping, payload)  # type: ignore[arg-type]
    # "test"/"pytest" group comes first → wins over "write".
    assert payload["task_type"] == "test"


def test_classifier_fallback_when_no_match() -> None:
    classifier = ModelSkillClassifier(
        target_field="task_type",
        source_field="prompt",
        rules=((("write",), "code_generation"),),
        fallback="research",
    )
    mapping = _mapping_with(classifiers=(classifier,))
    payload: dict[str, object] = {"prompt": "tell me about the weather"}
    _apply_classifiers(mapping, payload)  # type: ignore[arg-type]
    assert payload["task_type"] == "research"


def test_classifier_does_not_override_explicit_value() -> None:
    classifier = ModelSkillClassifier(
        target_field="task_type",
        source_field="prompt",
        rules=((("write",), "code_generation"),),
        fallback="research",
    )
    mapping = _mapping_with(classifiers=(classifier,))
    payload: dict[str, object] = {"prompt": "write code", "task_type": "document"}
    _apply_classifiers(mapping, payload)  # type: ignore[arg-type]
    assert payload["task_type"] == "document"


def test_delegate_mapping_classifies_and_builds_payload() -> None:
    """End-to-end declarative-layer check for the worst-case skill."""
    registry = load_skill_registry()
    delegate = registry.get("delegate")
    assert delegate is not None
    payload = _parse_skill_args(
        delegate, ("write", "a", "unit", "test", "for", "the", "parser")
    )
    _apply_classifiers(delegate, payload)
    assert payload["prompt"] == "write a unit test for the parser"
    assert payload["source"] == "claude-code"
    # No --max-tokens override supplied: the field is omitted from the payload so
    # the delegate node resolves it per-backend from its routing contract
    # (OMN-13161 — no hardcoded CLI-side default).
    assert "max_tokens" not in payload
    # "unit test" / "test" keyword group wins.
    assert payload["task_type"] == "test"


def test_command_unknown_skill_fails() -> None:
    runner = CliRunner()
    result = runner.invoke(run_skill_by_name, ["does_not_exist"])
    assert result.exit_code != 0
    assert "Unknown skill" in result.output


def test_command_dispatches_via_receipt_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The command builds the payload and hands it to run_receipt_mode."""
    captured: dict[str, object] = {}

    def _fake_receipt_mode(**kwargs: object) -> int:
        captured.update(kwargs)
        # Read back the payload the command wrote.
        input_path = kwargs["input_path"]
        assert isinstance(input_path, Path)
        captured["payload"] = json.loads(input_path.read_text(encoding="utf-8"))
        return 0

    def _fake_resolve(node_name: str) -> Path:
        return tmp_path / f"{node_name}-contract.yaml"

    monkeypatch.setattr(cli_skill, "run_receipt_mode", _fake_receipt_mode)
    monkeypatch.setattr(cli_skill, "_resolve_packaged_contract", _fake_resolve)

    runner = CliRunner()
    result = runner.invoke(
        run_skill_by_name,
        [
            "compliance_sweep",
            "--state-root",
            str(tmp_path / "state"),
            "--repos",
            "omnibase_core,omnibase_infra",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["node_name"] == "node_compliance_sweep"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["repos"] == ["omnibase_core", "omnibase_infra"]
    assert payload["dry_run"] is True
    assert captured["backend_overrides"] == {"event_bus": "inmemory"}


def test_gap_mapping_routes_positional_subcommand_to_node(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`onex skill gap detect --scope local` dispatches node_gap_compute.

    Guards the regression this fix addresses: the gap SKILL.md and the
    omnimarket node_gap_compute both existed, but the CLI rejected
    ``onex skill gap`` because it was absent from skill_mapping.yaml.
    """
    captured: dict[str, object] = {}

    def _fake_receipt_mode(**kwargs: object) -> int:
        captured.update(kwargs)
        input_path = kwargs["input_path"]
        assert isinstance(input_path, Path)
        captured["payload"] = json.loads(input_path.read_text(encoding="utf-8"))
        return 0

    monkeypatch.setattr(cli_skill, "run_receipt_mode", _fake_receipt_mode)
    monkeypatch.setattr(
        cli_skill,
        "_resolve_packaged_contract",
        lambda n: tmp_path / f"{n}-contract.yaml",
    )

    runner = CliRunner()
    result = runner.invoke(
        run_skill_by_name,
        [
            "gap",
            "detect",
            "--state-root",
            str(tmp_path / "state"),
            "--scope",
            "local",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["node_name"] == "node_gap_compute"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["subcommand"] == "detect"
    assert payload["scope"] == "local"


def test_gap_payload_validates_against_request_model() -> None:
    """The gap mapping builds a payload the node's request model accepts.

    omnimarket is the backing-node package and is not a hard test dependency
    of omnibase_infra; skip when it is not installed (CI parity with the other
    omnimarket-optional tests in this suite).
    """
    request_module = pytest.importorskip(
        "omnimarket.nodes.node_gap_compute.models.model_gap_compute_request"
    )
    ModelGapComputeRequest = request_module.ModelGapComputeRequest

    registry = load_skill_registry()
    gap = registry.get("gap")
    assert gap is not None
    assert gap.node_name == "node_gap_compute"
    payload = _parse_skill_args(
        gap, ("detect", "--scope", "local", "--severity-threshold", "CRITICAL")
    )
    request = ModelGapComputeRequest.model_validate(payload)
    assert request.subcommand.value == "detect"
    assert request.scope == "local"
    assert request.severity_threshold.value == "CRITICAL"


def test_contract_sweep_payload_validates_against_request_model() -> None:
    """The OMN-13511 contract_sweep mapping builds an acceptable payload.

    The node request model is ``extra="forbid"``, so every CLI arg the mapping
    surfaces must be a real request-model field — this proves it for the
    newly-registered sweep rather than only that it dispatches.
    """
    handler_module = pytest.importorskip(
        "omnimarket.nodes.node_contract_sweep.handlers.handler_contract_sweep"
    )
    ContractSweepRequest = handler_module.ContractSweepRequest

    registry = load_skill_registry()
    contract_sweep = registry.get("contract_sweep")
    assert contract_sweep is not None
    assert contract_sweep.node_name == "node_contract_sweep"
    payload = _parse_skill_args(
        contract_sweep, ("--repos", "omnibase_core,omnibase_infra", "--dry-run")
    )
    request = ContractSweepRequest.model_validate(payload)
    assert request.repos == ["omnibase_core", "omnibase_infra"]
    assert request.dry_run is True


def test_runtime_sweep_payload_validates_against_request_model() -> None:
    """The OMN-13511 runtime_sweep mapping builds an acceptable payload."""
    handler_module = pytest.importorskip(
        "omnimarket.nodes.node_runtime_sweep.handlers.handler_runtime_sweep"
    )
    RuntimeSweepRequest = handler_module.RuntimeSweepRequest

    registry = load_skill_registry()
    runtime_sweep = registry.get("runtime_sweep")
    assert runtime_sweep is not None
    assert runtime_sweep.node_name == "node_runtime_sweep"
    payload = _parse_skill_args(runtime_sweep, ("--dry-run",))
    request = RuntimeSweepRequest.model_validate(payload)
    assert request.dry_run is True


def test_runtime_sweep_scope_arg_wired_and_forwarded() -> None:
    """OMN-13715: --scope is accepted by the CLI mapping and forwarded to the payload.

    Before this fix, runtime_sweep was missing the scope arg in skill_mapping.yaml.
    Passing --scope caused `Unknown argument '--scope'` (ClickException). After the
    fix, --scope maps to the ``scope`` payload field and the RuntimeSweepRequest
    model validates it cleanly (extra="forbid" — if the field is absent from the
    model the validate call raises).
    """
    handler_module = pytest.importorskip(
        "omnimarket.nodes.node_runtime_sweep.handlers.handler_runtime_sweep"
    )
    RuntimeSweepRequest = handler_module.RuntimeSweepRequest

    registry = load_skill_registry()
    runtime_sweep = registry.get("runtime_sweep")
    assert runtime_sweep is not None

    # Before fix: _parse_skill_args raised ClickException("Unknown argument '--scope'")
    payload = _parse_skill_args(runtime_sweep, ("--scope", "omnidash-only"))
    assert payload.get("scope") == "omnidash-only"

    # Payload must validate against the request model (extra="forbid").
    request = RuntimeSweepRequest.model_validate(payload)
    assert request.scope == "omnidash-only"  # type: ignore[attr-defined]


def test_runtime_sweep_scope_arg_omitted_gives_none() -> None:
    """OMN-13715: omitting --scope leaves scope absent from payload (no default injected)."""
    handler_module = pytest.importorskip(
        "omnimarket.nodes.node_runtime_sweep.handlers.handler_runtime_sweep"
    )
    RuntimeSweepRequest = handler_module.RuntimeSweepRequest

    registry = load_skill_registry()
    runtime_sweep = registry.get("runtime_sweep")
    assert runtime_sweep is not None

    payload = _parse_skill_args(runtime_sweep, ())
    # Scope omitted → not in payload → model defaults to None.
    assert "scope" not in payload or payload.get("scope") is None
    request = RuntimeSweepRequest.model_validate(payload)
    assert request.scope is None  # type: ignore[attr-defined]


def test_integration_sweep_payload_validates_against_request_model() -> None:
    """The OMN-13511 integration_sweep mapping builds an acceptable payload."""
    request_module = pytest.importorskip(
        "omnimarket.nodes.node_integration_sweep_orchestrator.models."
        "model_integration_sweep_orchestrator_request"
    )
    ModelIntegrationSweepOrchestratorRequest = (
        request_module.ModelIntegrationSweepOrchestratorRequest
    )

    registry = load_skill_registry()
    integration_sweep = registry.get("integration_sweep")
    assert integration_sweep is not None
    assert integration_sweep.node_name == "node_integration_sweep_orchestrator"
    payload = _parse_skill_args(
        integration_sweep, ("--scope", "post-merge", "--tickets", "OMN-1,OMN-2")
    )
    request = ModelIntegrationSweepOrchestratorRequest.model_validate(payload)
    assert request.scope == "post-merge"
    assert request.tickets == ["OMN-1", "OMN-2"]
    # run_surface_probes defaults to True via the mapping default.
    assert request.run_surface_probes is True


def test_skill_functional_audit_payload_validates_against_request_model() -> None:
    """The OMN-13511 skill_functional_audit mapping builds an acceptable payload."""
    request_module = pytest.importorskip(
        "omnimarket.nodes.node_skill_functional_audit_compute.models."
        "model_skill_functional_audit_compute_request"
    )
    ModelSkillFunctionalAuditComputeRequest = (
        request_module.ModelSkillFunctionalAuditComputeRequest
    )

    registry = load_skill_registry()
    audit = registry.get("skill_functional_audit")
    assert audit is not None
    assert audit.node_name == "node_skill_functional_audit_compute"
    payload = _parse_skill_args(audit, ("--skills-filter", "merge_sweep,gap"))
    request = ModelSkillFunctionalAuditComputeRequest.model_validate(payload)
    assert request.skills_filter == ["merge_sweep", "gap"]


# --------------------------------------------------------------------------- #
# OMN-13918: `redeploy` was documented (docs/) but had no skill_mapping.yaml
# entry — `onex skill redeploy` returned "Unknown skill" and the .201 runtime
# redeploy path had no executable CLI surface or explicit lane targeting.
# --------------------------------------------------------------------------- #


def test_redeploy_registered_with_no_default_lane() -> None:
    """`redeploy` is mapped to the real orchestrator node.

    The `lane` arg spec must declare NO default (`default is None` means the
    arg is omitted entirely when unset, letting the node-input model's own
    "dev" default apply) — the mapping itself must never be able to resolve
    the lane to "prod" when the caller omits `--lane`.
    """
    registry = load_skill_registry()
    redeploy = registry.get("redeploy")
    assert redeploy is not None
    assert redeploy.node_name == "node_redeploy_orchestrator"

    lane_spec = next(
        spec for spec in redeploy.args if spec.payload_field == "runtime_lane"
    )
    assert lane_spec.default is None
    assert lane_spec.required is False


def test_redeploy_payload_validates_against_request_model() -> None:
    """The OMN-13918 redeploy mapping builds a payload the command model accepts.

    ``ModelRedeployStartCommand`` is ``extra="forbid"`` — every CLI arg the
    mapping surfaces must be a real field on that model. ``correlation_id`` is
    a required field the mapping intentionally does NOT expose as a CLI arg
    (RuntimeLocal auto-injects it, OMN-13591); the test supplies one directly
    to validate the rest of the CLI-constructed payload.
    """
    command_module = pytest.importorskip(
        "omnimarket.nodes.node_redeploy_orchestrator.models."
        "model_redeploy_start_command"
    )
    ModelRedeployStartCommand = command_module.ModelRedeployStartCommand

    registry = load_skill_registry()
    redeploy = registry.get("redeploy")
    assert redeploy is not None
    payload = _parse_skill_args(
        redeploy,
        ("--lane", "dev", "--scope", "full", "--git-ref", "origin/main", "--dry-run"),
    )
    assert "correlation_id" not in payload
    payload["correlation_id"] = str(uuid4())

    request = ModelRedeployStartCommand.model_validate(payload)
    assert request.runtime_lane.value == "dev"
    assert request.scope.value == "full"
    assert request.git_ref == "origin/main"
    assert request.dry_run is True


def test_redeploy_lane_omitted_defaults_to_dev_never_prod() -> None:
    """Omitting `--lane` must never silently resolve to prod (OMN-13918 DoD).

    The mapping sets no default for `lane`, so the payload field is absent
    entirely; ``ModelRedeployStartCommand.runtime_lane`` then applies its own
    "dev" default. Prod is reachable ONLY via an explicit ``--lane prod``.
    """
    command_module = pytest.importorskip(
        "omnimarket.nodes.node_redeploy_orchestrator.models."
        "model_redeploy_start_command"
    )
    ModelRedeployStartCommand = command_module.ModelRedeployStartCommand

    registry = load_skill_registry()
    redeploy = registry.get("redeploy")
    assert redeploy is not None
    payload = _parse_skill_args(redeploy, ())
    assert "runtime_lane" not in payload
    payload["correlation_id"] = str(uuid4())

    request = ModelRedeployStartCommand.model_validate(payload)
    assert request.runtime_lane.value == "dev"


def test_redeploy_prod_lane_is_explicit_and_still_requires_the_gate() -> None:
    """`--lane prod` is honored explicitly but never bypasses the promotion gate.

    This proves only the mapping/payload layer: the CLI faithfully forwards an
    explicit `--lane prod` request unmodified. The orchestrator itself (tested
    in omnimarket) always routes a prod ``ModelRedeployStartCommand`` through
    the out-of-band grant-resolve -> gate-evaluate chain and never trusts a
    caller-attached grant — dry-run included, which reports BLOCKED rather
    than fabricating a passing gate decision.
    """
    command_module = pytest.importorskip(
        "omnimarket.nodes.node_redeploy_orchestrator.models."
        "model_redeploy_start_command"
    )
    ModelRedeployStartCommand = command_module.ModelRedeployStartCommand

    registry = load_skill_registry()
    redeploy = registry.get("redeploy")
    assert redeploy is not None
    payload = _parse_skill_args(redeploy, ("--lane", "prod", "--dry-run"))
    payload["correlation_id"] = str(uuid4())

    request = ModelRedeployStartCommand.model_validate(payload)
    assert request.runtime_lane.value == "prod"
    # promotion_grant is not a CLI-exposed field — a caller cannot self-grant.
    assert "promotion_grant" not in payload
    assert request.promotion_grant is None


def test_command_writes_payload_under_state_root_not_tmp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen_paths: list[Path] = []

    def _fake_receipt_mode(**kwargs: object) -> int:
        input_path = kwargs["input_path"]
        assert isinstance(input_path, Path)
        seen_paths.append(input_path)
        return 0

    monkeypatch.setattr(cli_skill, "run_receipt_mode", _fake_receipt_mode)
    monkeypatch.setattr(
        cli_skill, "_resolve_packaged_contract", lambda n: tmp_path / "c.yaml"
    )

    state_root = tmp_path / "state"
    runner = CliRunner()
    result = runner.invoke(
        run_skill_by_name,
        ["platform_readiness", "--state-root", str(state_root)],
    )
    assert result.exit_code == 0, result.output
    assert len(seen_paths) == 1
    # Payload scratch lives under the state root, never /tmp.
    assert state_root in seen_paths[0].parents
    assert "tmp" in seen_paths[0].parts


def test_plan_to_tickets_plan_path_is_named_not_positional() -> None:
    """plan_to_tickets plan-path must be a named --plan-path flag (OMN-13718).

    Regression guard: the original mapping declared plan-path with
    positional=True, which registered it as a positional argument instead of
    a named option. This caused `--plan-path` to be rejected as an unknown flag.
    The fix removes positional=True so --plan-path is the accepted form.
    """
    registry = load_skill_registry()
    skill = registry.get("plan_to_tickets")
    assert skill is not None, "plan_to_tickets is absent from skill_mapping.yaml"

    plan_path_args = [a for a in skill.args if a.name == "plan-path"]
    assert plan_path_args, "plan-path arg is absent from plan_to_tickets mapping"
    arg = plan_path_args[0]

    # The arg must be required (named option, not positional).
    assert arg.required is True, "plan-path must be required=True"
    # Must NOT be positional — positional=True causes --plan-path to be rejected.
    assert not arg.positional, (
        "plan-path must NOT be positional; positional=True registers it as a "
        "positional CLI arg and causes `--plan-path <file>` to be rejected as "
        "'No such option: --plan-path' (OMN-13718 regression)"
    )


# --------------------------------------------------------------------------- #
# OMN-13995: `dep_cascade_dedup` was documented as a skill and had a real
# omnimarket backing node (node_dep_cascade_dedup_orchestrator) with request +
# result models, but skill_mapping.yaml carried no entry, so
# `onex skill dep_cascade_dedup` returned "Unknown skill" and the post-release
# dep-bump dedup sweep was unrunnable headless from the canonical infra venv.
# Same regression class as OMN-13511 / OMN-13712 (node exists, skill unmapped).
# --------------------------------------------------------------------------- #


def test_dep_cascade_dedup_registered_and_wired() -> None:
    """`dep_cascade_dedup` resolves to the real orchestrator node + result model.

    Reproducing assertion: before this mapping entry existed,
    ``onex skill dep_cascade_dedup`` returned "Unknown skill". This pins the
    skill to ``node_dep_cascade_dedup_orchestrator`` and its typed result model
    so a future rename/removal of either surfaces here at CI time instead of at
    dispatch time.
    """
    registry = load_skill_registry()
    mapping = registry.get("dep_cascade_dedup")
    assert mapping is not None, "dep_cascade_dedup absent from skill_mapping.yaml"
    assert mapping.node_name == "node_dep_cascade_dedup_orchestrator"
    assert mapping.result_model == (
        "omnimarket.nodes.node_dep_cascade_dedup_orchestrator.models."
        "model_dep_cascade_dedup_result.ModelDepCascadeDedupResult"
    )


def test_dep_cascade_dedup_payload_validates_against_request_model() -> None:
    """The OMN-13995 mapping builds a payload ModelDepCascadeDedupRequest accepts.

    ``ModelDepCascadeDedupRequest`` is ``extra="forbid"``, so every CLI arg the
    mapping surfaces must be a real request-model field. This proves the sweep
    resolves its roots/repos (``--repos`` → the ``repos`` field it scans), not
    merely that the skill name is registered. omnimarket is the backing-node
    package and is not a hard test dependency of omnibase_infra; skip when it is
    not installed (CI parity with the other omnimarket-optional tests here).
    """
    request_module = pytest.importorskip(
        "omnimarket.nodes.node_dep_cascade_dedup_orchestrator.models."
        "model_dep_cascade_dedup_request"
    )
    ModelDepCascadeDedupRequest = request_module.ModelDepCascadeDedupRequest

    registry = load_skill_registry()
    dedup = registry.get("dep_cascade_dedup")
    assert dedup is not None
    assert dedup.node_name == "node_dep_cascade_dedup_orchestrator"
    payload = _parse_skill_args(
        dedup,
        (
            "--repos",
            "omnibase_core,omnibase_infra",
            "--dependency-type",
            "python",
            "--label",
            "dependencies",
            "--dry-run",
        ),
    )
    request = ModelDepCascadeDedupRequest.model_validate(payload)
    # repos = the sweep's roots/repos it scans for superseded dep-bump PRs.
    assert request.repos == ("omnibase_core", "omnibase_infra")
    assert request.dependency_type == "python"
    assert request.label == "dependencies"
    assert request.dry_run is True


def test_dep_cascade_dedup_dry_run_omitted_defaults_false() -> None:
    """Omitting ``--dry-run`` yields a wet-run payload the model accepts.

    Guards the boolean-default wiring: the mapping declares ``dry-run`` with
    ``default: false`` so an omitted flag produces ``dry_run=False`` rather than
    dropping the field.
    """
    request_module = pytest.importorskip(
        "omnimarket.nodes.node_dep_cascade_dedup_orchestrator.models."
        "model_dep_cascade_dedup_request"
    )
    ModelDepCascadeDedupRequest = request_module.ModelDepCascadeDedupRequest

    registry = load_skill_registry()
    dedup = registry.get("dep_cascade_dedup")
    assert dedup is not None
    payload = _parse_skill_args(dedup, ())
    assert payload["dry_run"] is False
    request = ModelDepCascadeDedupRequest.model_validate(payload)
    assert request.dry_run is False
    # repos defaults to the empty tuple → node discovers all OmniNode-ai repos.
    assert request.repos == ()


# --------------------------------------------------------------------------- #
# OMN-13995 / CLAUDE.md rule #8: the sweep-repo-fallback must FAIL FAST.
#
# The dogfood sweeps that resolve a repo-registry root from the environment
# (integration_sweep, dod_sweep) share a ``_resolve_root`` fallback: when no
# explicit root is passed AND the repo-registry env var is unset, they must
# RAISE — never silently default to a wrong path (rule #8: "Fail-fast on
# missing env, not silent fallback"). This is the exact anti-pattern the rule
# exists to prevent (a silent default produces cross-machine breakage). We pin
# BOTH sweeps so a future refactor that swaps the raise for an
# ``os.environ.get(..., <default>)`` silent fallback is caught here.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("handler_import", "handler_attr"),
    [
        (
            "omnimarket.nodes.node_integration_sweep_orchestrator.handlers."
            "handler_integration_sweep_orchestrator",
            "HandlerIntegrationSweepOrchestrator",
        ),
        (
            "omnimarket.nodes.node_dod_sweep_orchestrator.handlers."
            "handler_dod_sweep_orchestrator",
            "HandlerDodSweepOrchestrator",
        ),
    ],
)
def test_sweep_repo_fallback_fails_fast_without_env(
    monkeypatch: pytest.MonkeyPatch, handler_import: str, handler_attr: str
) -> None:
    """`_resolve_root('')` raises when the repo-registry env is unset (rule #8).

    Both integration_sweep and dod_sweep fall back to the ``ONEX_CC_REPO_PATH``
    repo-registry env when no explicit root is supplied. With it unset and no
    explicit path, resolution MUST raise rather than silently default — this is
    the CLAUDE.md rule #8 fail-fast discipline the dep_cascade_dedup work must
    preserve, not weaken.
    """
    module = pytest.importorskip(handler_import)
    handler_cls = getattr(module, handler_attr)

    # Ensure the fallback env is absent so the no-explicit-root path is exercised.
    monkeypatch.delenv("ONEX_CC_REPO_PATH", raising=False)

    with pytest.raises(RuntimeError, match="ONEX_CC_REPO_PATH is not set"):
        handler_cls._resolve_root("")


def test_pr_state_payload_validates_against_request_model() -> None:
    """OMN-14374: the pr_state mapping builds a payload node_github_repo_gateway_effect accepts.

    Wires the EXISTING read-only status reader (OMN-14307) so `onex skill
    pr_state` returns one small typed row per operation instead of a raw
    `gh pr view/checks --json ...` dump. --pr is required only for the 5
    PR-scoped operations; the request model's own validator enforces that
    (not the CLI mapping), so this proves both the scoped and unscoped shape.
    """
    request_module = pytest.importorskip(
        "omnimarket.nodes.node_github_repo_gateway_effect.models.model_gateway_io"
    )
    ModelGithubGatewayRequest = request_module.ModelGithubGatewayRequest

    registry = load_skill_registry()
    pr_state = registry.get("pr_state")
    assert pr_state is not None
    assert pr_state.node_name == "node_github_repo_gateway_effect"

    # PR-scoped operation.
    payload = _parse_skill_args(
        pr_state,
        (
            "--operation",
            "pr_status",
            "--repo",
            "OmniNode-ai/omnimarket",
            "--pr",
            "1704",
        ),
    )
    request = ModelGithubGatewayRequest.model_validate(payload)
    assert request.operation.value == "pr_status"
    assert request.repo == "OmniNode-ai/omnimarket"
    assert request.pr_number == 1704

    # Repo-scoped operation — --pr omitted, still validates.
    payload = _parse_skill_args(
        pr_state, ("--operation", "open_prs_list", "--repo", "OmniNode-ai/omnimarket")
    )
    request = ModelGithubGatewayRequest.model_validate(payload)
    assert request.operation.value == "open_prs_list"
    assert request.pr_number is None
