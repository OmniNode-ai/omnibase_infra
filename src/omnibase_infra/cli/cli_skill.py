# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""``onex skill <name> [args]`` — single-command dispatch surface (OMN-13097).

Phase 4a of the skill-output-suppression slice
(``docs/plans/2026-06-12-skill-output-suppression-plan.md`` Phase 4 item 1).

A dispatch skill IS one CLI call (user directive 2026-06-12). This command
replaces the 24 hand-written dispatch shims: it resolves the skill via the
declarative ``skill_mapping.yaml`` registry, builds the backing node's input
payload from the skill's CLI arguments, and dispatches through the proven
receipt-mode path (``run_receipt_mode``, OMN-13094). stdout carries exactly
one typed :class:`~omnibase_core.models.dispatch.model_skill_result.ModelSkillResult`
JSON with the FULL handler result; runtime logs go to the capture file.

The skill→node mapping is DECLARATIVE DATA in ``skill_mapping.yaml`` — never
hardcoded Python branching here (ticket deliverable 2). Adding a skill is a
YAML edit plus a fixture, not a code change.

.. versionadded:: OMN-13097
"""

from __future__ import annotations

import json
import logging
import uuid
from functools import lru_cache
from pathlib import Path

import click
import yaml
from pydantic import JsonValue, ValidationError

from omnibase_infra.cli.cli_node import _resolve_packaged_contract
from omnibase_infra.cli.enum_skill_arg_type import EnumSkillArgType
from omnibase_infra.cli.model_skill_arg_spec import ModelSkillArgSpec
from omnibase_infra.cli.model_skill_mapping import ModelSkillMapping
from omnibase_infra.cli.model_skill_mapping_registry import ModelSkillMappingRegistry
from omnibase_infra.cli.receipt_mode import (
    default_emit_socket_path,
    run_receipt_mode,
)

__all__ = ["MAPPING_FILENAME", "load_skill_registry", "run_skill_by_name"]

logger = logging.getLogger(__name__)

# Declarative skill→node mapping ships beside this module.
MAPPING_FILENAME = "skill_mapping.yaml"

# Internal scratch for the constructed input payload lives under the state
# root, NEVER /tmp (feedback_no_tmp_use_workspace.md). run_id-suffixed names
# avoid collisions between parallel dispatches.
_TMP_DIR_NAME = "tmp"


@lru_cache(maxsize=1)
def load_skill_registry() -> ModelSkillMappingRegistry:
    """Load and validate the declarative skill→node mapping registry.

    Raises:
        click.ClickException: when the mapping file is missing or invalid.
    """
    mapping_path = Path(__file__).parent / MAPPING_FILENAME
    if not mapping_path.is_file():
        raise click.ClickException(
            f"Skill mapping registry not found at {mapping_path}. "
            "This file is required for 'onex skill'."
        )
    raw = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    try:
        return ModelSkillMappingRegistry.model_validate(raw)
    except ValidationError as exc:
        raise click.ClickException(
            f"Invalid skill mapping registry at {mapping_path}: {exc}"
        ) from exc


def _coerce_value(spec: ModelSkillArgSpec, raw: str) -> JsonValue:
    """Coerce a raw CLI string to the spec's declared type. Fail fast."""
    if spec.arg_type is EnumSkillArgType.STRING:
        return raw
    if spec.arg_type is EnumSkillArgType.INTEGER:
        try:
            return int(raw)
        except ValueError as exc:
            raise click.ClickException(
                f"--{spec.name} expects an integer, got {raw!r}"
            ) from exc
    if spec.arg_type is EnumSkillArgType.STRING_LIST:
        return [item.strip() for item in raw.split(",") if item.strip()]
    # BOOLEAN is a presence flag and never reaches value coercion.
    raise click.ClickException(
        f"--{spec.name}: value coercion not supported for {spec.arg_type}"
    )


def _parse_skill_args(
    mapping: ModelSkillMapping, extra: tuple[str, ...]
) -> dict[str, JsonValue]:
    """Parse trailing ``onex skill`` tokens into a node-input payload.

    Boolean specs are presence flags (``--dry-run``); typed specs take the
    next token (``--repos a,b``). A single positional spec collects all
    non-flag trailing tokens, joined with spaces.

    Raises:
        click.ClickException: on unknown flags, missing values, missing
            required args, or coercion failure.
    """
    by_flag = {spec.name: spec for spec in mapping.args if not spec.positional}
    positional_spec = next((s for s in mapping.args if s.positional), None)

    payload: dict[str, JsonValue] = dict(mapping.static_payload)
    positional_tokens: list[str] = []
    seen: set[str] = set()

    tokens = list(extra)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            flag = token[2:]
            spec = by_flag.get(flag)
            if spec is None:
                raise click.ClickException(
                    f"Unknown argument '--{flag}' for skill '{mapping.skill_name}'."
                )
            if spec.arg_type is EnumSkillArgType.BOOLEAN:
                payload[spec.payload_field] = True
                seen.add(spec.name)
                i += 1
                continue
            if i + 1 >= len(tokens):
                raise click.ClickException(f"--{flag} requires a value.")
            payload[spec.payload_field] = _coerce_value(spec, tokens[i + 1])
            seen.add(spec.name)
            i += 2
            continue
        positional_tokens.append(token)
        i += 1

    if positional_tokens:
        if positional_spec is None:
            raise click.ClickException(
                f"Skill '{mapping.skill_name}' takes no positional arguments; "
                f"got {positional_tokens!r}."
            )
        payload[positional_spec.payload_field] = " ".join(positional_tokens)
        seen.add(positional_spec.name)

    # Apply defaults for unset, non-required args; enforce required args.
    for spec in mapping.args:
        if spec.name in seen:
            continue
        if spec.required:
            label = "positional argument" if spec.positional else f"--{spec.name}"
            raise click.ClickException(
                f"Skill '{mapping.skill_name}' requires {label}."
            )
        if spec.arg_type is EnumSkillArgType.BOOLEAN:
            payload.setdefault(spec.payload_field, bool(spec.default))
        elif spec.default is not None:
            payload[spec.payload_field] = spec.default

    return payload


def _apply_classifiers(
    mapping: ModelSkillMapping, payload: dict[str, JsonValue]
) -> None:
    """Assign classifier target fields that remain unset after arg parsing."""
    for classifier in mapping.classifiers:
        if payload.get(classifier.target_field) is not None:
            continue
        source = payload.get(classifier.source_field)
        source_text = str(source).lower() if source is not None else ""
        assigned = classifier.fallback
        for keywords, value in classifier.rules:
            if any(keyword.lower() in source_text for keyword in keywords):
                assigned = value
                break
        payload[classifier.target_field] = assigned


def _write_payload(
    state_root: Path, skill_name: str, payload: dict[str, JsonValue]
) -> Path:
    """Write the input payload to a run_id-suffixed file under the state root."""
    tmp_dir = state_root / _TMP_DIR_NAME
    tmp_dir.mkdir(parents=True, exist_ok=True)
    payload_path = tmp_dir / f"{skill_name}-{uuid.uuid4()}.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload_path


@click.command(
    "skill",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("skill_name")
@click.option(
    "--state-root",
    type=click.Path(path_type=Path),
    default=".onex_state",
    show_default=True,
    help="Root directory for disk state and capture files.",
)
@click.option(
    "--timeout",
    type=int,
    default=None,
    help="Override the skill's default dispatch timeout (seconds).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Capture DEBUG-level runtime logging (still never on stdout).",
)
@click.option(
    "--emit-socket",
    "emit_socket",
    type=click.Path(path_type=Path),
    default=None,
    help="Unix socket of the emit daemon (default: ~/.claude/emit.sock).",
)
@click.argument("skill_args", nargs=-1, type=click.UNPROCESSED)
def run_skill_by_name(
    skill_name: str,
    state_root: Path,
    timeout: int | None,
    verbose: bool,
    emit_socket: Path | None,
    skill_args: tuple[str, ...],
) -> None:
    """Dispatch a skill to its backing node and print one typed result.

    \b
    A dispatch skill is exactly one CLI call: this resolves the declarative
    skill→node mapping, builds the node-input payload from SKILL_ARGS, and
    runs the dispatch in receipt mode — stdout is exactly one
    ModelSkillResult JSON carrying the FULL handler result; runtime logs go
    to the capture file under the state root.

    \b
    Examples:
        onex skill compliance_sweep --repos omnibase_core,omnibase_infra
        onex skill dod_verify OMN-1234
        onex skill delegate "summarize this paragraph" --task-type document
    """
    registry = load_skill_registry()
    mapping = registry.get(skill_name)
    if mapping is None:
        known = ", ".join(sorted(s.skill_name for s in registry.skills)) or "(none)"
        raise click.ClickException(
            f"Unknown skill '{skill_name}'. Known skills: {known}"
        )

    payload = _parse_skill_args(mapping, skill_args)
    _apply_classifiers(mapping, payload)

    contract_path = _resolve_packaged_contract(mapping.node_name)
    payload_path = _write_payload(state_root, skill_name, payload)

    exit_code = run_receipt_mode(
        node_name=mapping.node_name,
        contract_path=contract_path,
        input_path=payload_path,
        state_root=state_root,
        backend_overrides={"event_bus": mapping.event_bus},
        timeout=timeout if timeout is not None else mapping.timeout,
        verbose=verbose,
        emit_socket=emit_socket or default_emit_socket_path(),
    )
    raise SystemExit(exit_code)
