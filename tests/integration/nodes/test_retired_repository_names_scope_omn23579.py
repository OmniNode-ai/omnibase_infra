# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The scope and learning-extraction repository lists no longer name retired repositories."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_scope_extract_compute.handlers.handler_scope_extract import (
    HandlerScopeExtract,
)
from omnibase_infra.nodes.node_scope_extract_compute.models.model_scope_extract_input import (
    ModelScopeExtractInput,
)
from omnibase_infra.services.agent_learning_extraction.consumer import (
    extract_repo_from_working_dir,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_RETIRED = ("onex_change_control", "omnibase_compat")


async def test_scope_extract_reads_only_active_repositories(tmp_path: Path) -> None:
    """A plan naming an active and two retired repositories scopes only the active one."""
    plan = tmp_path / "plan.md"
    content = "Update omnibase_infra, omnibase_compat and onex_change_control.\n"
    plan.write_text(content, encoding="utf-8")

    extracted = await HandlerScopeExtract().handle(
        ModelScopeExtractInput(
            correlation_id=uuid4(),
            plan_file_path=str(plan),
            output_path=str(tmp_path / "scope.json"),
            content=content,
        )
    )

    assert "omnibase_infra" in extracted.repos
    assert not set(_RETIRED) & set(extracted.repos)


@pytest.mark.parametrize("retired", _RETIRED)
async def test_learning_extraction_does_not_recognise_retired_working_dir(
    retired: str,
) -> None:
    """A working directory named for a retired repository reads as unknown."""
    assert extract_repo_from_working_dir(f"/work/{retired}") == "unknown"
    assert extract_repo_from_working_dir("/work/omnibase_infra") == "omnibase_infra"
