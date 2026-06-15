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

# The 24 dispatch shims this ticket migrates (ticket deliverable 1).
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
        **kw,  # type: ignore[arg-type]
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
