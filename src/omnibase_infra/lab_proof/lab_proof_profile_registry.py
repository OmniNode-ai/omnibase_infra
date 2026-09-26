# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Load the proof-profile registry, check it against the repository, classify exemptions.

Three functions, all deterministic:

- ``load_lab_proof_profile_registry`` parses the YAML into the typed registry;
  every row-shape rule lives in the models, so a bad row fails here with the
  row named.
- ``validate_steps_against_repo`` checks what a row cannot check about itself:
  that each step names a node directory that exists and an operation that
  node's contract routes.
- ``classify_exemption`` decides the two ruled exemptions from the diff and the
  author only (RULING 2026-09-25T13:07:40Z): never from a PR body token.

Ticket: OMN-19565
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import yaml

from omnibase_infra.lab_proof.enum_lab_proof_exempt_class import (
    EnumLabProofExemptClass,
)
from omnibase_infra.lab_proof.model_lab_proof_exemption_decision import (
    ModelLabProofExemptionDecision,
)
from omnibase_infra.lab_proof.model_lab_proof_profile_registry import (
    ModelLabProofProfileRegistry,
)

NODES_RELATIVE_DIR = Path("src/omnibase_infra/nodes")


def load_lab_proof_profile_registry(path: Path) -> ModelLabProofProfileRegistry:
    """Parse and validate the registry file. Raises on any invalid row."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: the registry must be a mapping")
    return ModelLabProofProfileRegistry.model_validate(loaded)


def _contract_operations(contract_path: Path) -> set[str]:
    loaded = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    operations: set[str] = set()
    if not isinstance(loaded, dict):
        return operations
    routing = loaded.get("handler_routing")
    if not isinstance(routing, dict):
        return operations
    handlers = routing.get("handlers")
    if not isinstance(handlers, list):
        return operations
    for entry in handlers:
        if not isinstance(entry, dict):
            continue
        operation = entry.get("operation")
        if isinstance(operation, str):
            operations.add(operation)
        supported = entry.get("supported_operations")
        if isinstance(supported, list):
            operations.update(item for item in supported if isinstance(item, str))
    return operations


def validate_steps_against_repo(
    registry: ModelLabProofProfileRegistry, repo_root: Path
) -> list[str]:
    """Return one error per step that names a missing node or operation."""
    errors: list[str] = []
    nodes_dir = repo_root / NODES_RELATIVE_DIR
    for profile in registry.profiles:
        for variant in profile.variants:
            for step in variant.steps:
                node_name, operation = step.split(":", 1)
                contract = nodes_dir / node_name / "contract.yaml"
                where = f"{profile.profile_key}/{variant.variant_key} step {step!r}"
                if not contract.is_file():
                    errors.append(
                        f"{where}: no node {node_name} "
                        f"(no {NODES_RELATIVE_DIR / node_name / 'contract.yaml'})"
                    )
                    continue
                if operation not in _contract_operations(contract):
                    errors.append(
                        f"{where}: node {node_name} routes no operation {operation!r}"
                    )
    return errors


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a path glob to a regex: ``**`` spans directories, ``*`` does not."""
    out: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def paths_all_match(paths: Iterable[str], globs: Iterable[str]) -> bool:
    """True when every path matches at least one glob (and there is a path)."""
    compiled = [glob_to_regex(glob) for glob in globs]
    items = list(paths)
    return bool(items) and all(
        any(regex.match(path) for regex in compiled) for path in items
    )


def classify_exemption(
    registry: ModelLabProofProfileRegistry,
    repo: str,
    changed_paths: Iterable[str],
    author: str,
) -> ModelLabProofExemptionDecision:
    """Decide whether one PR is exempt, from its diff and author only."""
    profile = registry.profile_for(repo)
    paths = sorted(set(changed_paths))
    if not paths:
        return ModelLabProofExemptionDecision(
            repo=repo,
            exempt=False,
            reason="empty diff: nothing to classify, so no exemption is derived",
        )
    if EnumLabProofExemptClass.DOCS_ONLY in profile.exempt_classes and paths_all_match(
        paths, profile.docs_globs
    ):
        return ModelLabProofExemptionDecision(
            repo=repo,
            exempt=True,
            exempt_class=EnumLabProofExemptClass.DOCS_ONLY,
            reason=f"all {len(paths)} changed paths match the docs globs",
        )
    rule = registry.change_control_companion
    if (
        EnumLabProofExemptClass.BOT_CHANGE_CONTROL_COMPANION in profile.exempt_classes
        and repo == rule.repo
        and author in rule.authors
        and paths_all_match(paths, rule.path_globs)
    ):
        return ModelLabProofExemptionDecision(
            repo=repo,
            exempt=True,
            exempt_class=EnumLabProofExemptClass.BOT_CHANGE_CONTROL_COMPANION,
            reason=(
                f"author {author} is a change-control bot and all {len(paths)} "
                "changed paths are companion paths"
            ),
        )
    return ModelLabProofExemptionDecision(
        repo=repo,
        exempt=False,
        reason="not documentation-only and not a bot change-control companion",
    )


__all__ = [
    "classify_exemption",
    "glob_to_regex",
    "load_lab_proof_profile_registry",
    "paths_all_match",
    "validate_steps_against_repo",
]
