# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``load_env_file``: the host settings store reader moved out of the endpoint guard (OMN-24017)."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.models.contracts.model_llm_endpoint_env_file import load_env_file


@pytest.mark.unit
def test_load_env_file_reads_names_and_strips_quotes(tmp_path: Path) -> None:
    store = tmp_path / ".env"
    store.write_text(
        "LLM_A_URL=\"http://h:1\"\nLLM_B_URL='http://h:2'\nLLM_C_URL=http://h:3\n"
    )

    assert load_env_file(store) == {
        "LLM_A_URL": "http://h:1",
        "LLM_B_URL": "http://h:2",
        "LLM_C_URL": "http://h:3",
    }


@pytest.mark.unit
def test_load_env_file_skips_comments_blank_lines_and_lines_without_equals(
    tmp_path: Path,
) -> None:
    store = tmp_path / ".env"
    store.write_text("# comment\n\nNOT_A_PAIR\nLLM_A_URL=x=y\n")

    assert load_env_file(store) == {"LLM_A_URL": "x=y"}
