# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Replay the subject that failed the release PR's public hygiene check."""

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.cli_node import run_node_by_name


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/omnimarket_release_changelog_omn25794.json"
)
def test_release_writer_replays_the_recorded_hygiene_failure(
    tmp_path: Path, recorded_response: dict[str, Any]
) -> None:
    response = recorded_response["response"]
    subject = response["subject_line"]
    alias = response["private_repo_alias"]
    # A repository that keeps a changelog: without one the node writes no file
    # (OMN-25958).
    (tmp_path / "CHANGELOG.md").write_text("", encoding="utf-8")
    entry = tmp_path / "entry.md"
    entry.write_text(
        f"{response['release_heading']}\n\n{response['included_since_heading']}\n{subject}\n",
        encoding="utf-8",
    )
    vocabulary = tmp_path / "vocabulary.yaml"
    vocabulary.write_text(json.dumps({"private_repo_names": [alias]}), encoding="utf-8")
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "clone_path": str(tmp_path),
                "entry_file": str(entry),
                "repository_visibility": "public",
                "hygiene_vocabulary_file": str(vocabulary),
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        run_node_by_name,
        [
            "node_release_notes_apply_effect",
            "--input",
            str(request),
            "--state-root",
            str(tmp_path / "state"),
            "--output",
            "receipt",
        ],
    )

    assert result.exit_code == 0, result.output
    changelog = (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")
    assert alias in subject
    assert alias not in changelog
    assert subject.replace(alias, "…") in changelog.splitlines()
