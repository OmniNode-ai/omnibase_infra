# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Load and validate the static module adjacency map."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelAdjacencyEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    reverse_deps: list[str] = Field(default_factory=list)


class ModelThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    modules_changed_for_full_suite: int = Field(..., ge=1)


class ModelAdjacencyMap(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(..., ge=1)
    shared_modules: list[str]
    thresholds: ModelThresholds
    test_infrastructure_paths: list[str]
    adjacency: dict[str, ModelAdjacencyEntry]

    # OMN-18012 -- BOUNDARY source -> the integration suite that proves it.
    #
    # `adjacency` can only ever emit `tests/unit/<module>/`, so a change to a
    # boundary module selected its MOCKS and nothing else. That is the second
    # half of escape 6 of 2026-09-06 and it is independent of the pre-push
    # hook: even with the hook's integration drop removed, an edit to
    # `event_bus/kafka_auth.py` still selected no integration test, because the
    # selector never emitted one. Measured on dev before this change:
    # `--changed-files src/omnibase_infra/event_bus/kafka_auth.py` emitted
    # seven `tests/unit/*` directories and zero integration paths, while the
    # only suite in the repo that exercises that module against a real
    # auth-required broker is tests/integration/customer_path/.
    #
    # Keys are exact source paths or path PREFIXES (a trailing "/" makes it a
    # prefix); values are the integration directories that change implicates.
    # Deliberately narrow and hand-curated: this is a fail-CLOSED gate whose
    # cost is a lab slot, so entries are added per proven boundary, never by a
    # wildcard over `src/`.
    boundary_integration_tests: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_shared_modules_in_adjacency(self) -> ModelAdjacencyMap:
        for shared in self.shared_modules:
            if shared not in self.adjacency:
                raise ValueError(f"shared_module '{shared}' has no adjacency entry")
        for module, entry in self.adjacency.items():
            for dep in entry.reverse_deps:
                if dep not in self.adjacency:
                    raise ValueError(
                        f"adjacency['{module}'].reverse_deps references unknown module '{dep}'"
                    )
        for source, targets in self.boundary_integration_tests.items():
            if not targets:
                raise ValueError(
                    f"boundary_integration_tests['{source}'] selects nothing; "
                    "an empty mapping is a silent no-op, not a narrowing"
                )
            for target in targets:
                if not target.startswith("tests/integration/") or not target.endswith(
                    "/"
                ):
                    raise ValueError(
                        f"boundary_integration_tests['{source}'] -> '{target}' must be "
                        "a directory under tests/integration/ (trailing slash required)"
                    )
        return self


def load_adjacency_map(path: Path) -> ModelAdjacencyMap:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ModelAdjacencyMap.model_validate(raw)
