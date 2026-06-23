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
        "auto_merge",
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
