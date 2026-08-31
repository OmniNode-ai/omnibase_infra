# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "resolve_node_migration_source_ref.py"
)


def _run(tmp_path: Path, body: str | None) -> subprocess.CompletedProcess[str]:
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps({"pull_request": {"body": body}}), encoding="utf-8"
    )
    output_path = tmp_path / "github_output.txt"
    env = {
        **os.environ,
        "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_OUTPUT": str(output_path),
    }
    return subprocess.run(
        ["python3", str(SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_defaults_to_dev_without_metadata(tmp_path: Path) -> None:
    result = _run(tmp_path, "Refs OMN-15038")

    assert result.returncode == 0
    assert result.stdout.strip() == "dev"
    assert (tmp_path / "github_output.txt").read_text(encoding="utf-8") == "ref=dev\n"


def test_reads_explicit_omnimarket_source_ref(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        "Refs OMN-15038\nOmnimarket-Source-Ref: jonah/omn-15038-drop-unwired-routing-columns",
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "jonah/omn-15038-drop-unwired-routing-columns"


def test_rejects_unsafe_ref(tmp_path: Path) -> None:
    result = _run(tmp_path, "Omnimarket-Source-Ref: ../dev")

    assert result.returncode == 1
    assert "invalid omnimarket source ref" in result.stderr


# ---------------------------------------------------------------------------
# OMN-17294 defect B: the trailer was matched anywhere in the body, including
# inside fenced code blocks, and the FIRST match anywhere won.
#
# A PR body that merely *quotes* a trailer -- a runbook excerpt, a pasted CI
# log, a diff of another PR's body, a "the trailer looks like this" example --
# therefore chose the omnimarket tree the required Application Database Domain
# Enforcement job derives its TABLE grants from. Same matcher class as
# OMN-15345 (table names matched inside SQL comments).
# ---------------------------------------------------------------------------


def test_fenced_decoy_trailer_is_ignored(tmp_path: Path) -> None:
    """A trailer quoted inside a ``` fence is documentation, not a trailer."""
    result = _run(
        tmp_path,
        "Refs OMN-17294\n"
        "\n"
        "The vendoring runbook says to write:\n"
        "\n"
        "```\n"
        "Omnimarket-Source-Ref: attacker/branch\n"
        "```\n"
        "\n"
        "This PR declares no ref of its own.\n",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "dev"


def test_tilde_fenced_decoy_trailer_is_ignored(tmp_path: Path) -> None:
    """``~~~`` opens a code fence too (CommonMark), not only backticks."""
    result = _run(
        tmp_path,
        "Refs OMN-17294\n~~~text\nOmnimarket-Source-Ref: attacker/branch\n~~~\n",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "dev"


def test_fenced_decoy_does_not_outrank_the_real_trailer(tmp_path: Path) -> None:
    """First-match-anywhere let quoted text above the real trailer win."""
    result = _run(
        tmp_path,
        "Refs OMN-17294\n"
        "\n"
        "Prior art (quoted from omnibase_infra#3046):\n"
        "\n"
        "```markdown\n"
        "Omnimarket-Source-Ref: attacker/branch\n"
        "```\n"
        "\n"
        "Omnimarket-Source-Ref: jonah/omn-17294-real-branch\n",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "jonah/omn-17294-real-branch"


def test_indented_code_block_trailer_is_ignored(tmp_path: Path) -> None:
    """Four-space indentation is a markdown code block, and a git trailer
    lives at column 0 -- an indented line is neither a trailer nor prose."""
    result = _run(
        tmp_path,
        "Refs OMN-17294\n\n    Omnimarket-Source-Ref: attacker/branch\n",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "dev"


def test_inline_code_span_trailer_is_ignored(tmp_path: Path) -> None:
    """A whole-line inline code span is quoted text, not a trailer."""
    result = _run(
        tmp_path,
        "Refs OMN-17294\n`Omnimarket-Source-Ref: attacker/branch`\n",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "dev"


def test_conflicting_trailer_values_are_an_error(tmp_path: Path) -> None:
    """Two different declared refs is ambiguous; silent first-wins picked one."""
    result = _run(
        tmp_path,
        "Omnimarket-Source-Ref: jonah/first\nOmnimarket-Source-Ref: jonah/second\n",
    )

    assert result.returncode == 1
    assert "conflicting" in result.stderr.lower()
    assert "jonah/first" in result.stderr
    assert "jonah/second" in result.stderr


def test_conflicting_field_aliases_are_an_error(tmp_path: Path) -> None:
    """The two accepted field names must not disagree either."""
    result = _run(
        tmp_path,
        "Omnimarket-Source-Ref: jonah/first\nNode-Migration-Source-Ref: jonah/second\n",
    )

    assert result.returncode == 1
    assert "conflicting" in result.stderr.lower()


def test_repeated_identical_trailer_is_not_a_conflict(tmp_path: Path) -> None:
    """Idempotent re-stamping of the same value stays legal."""
    result = _run(
        tmp_path,
        "Omnimarket-Source-Ref: jonah/same\nOmnimarket-Source-Ref: jonah/same\n",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "jonah/same"


def test_unterminated_fence_swallows_the_rest_of_the_body(tmp_path: Path) -> None:
    """CommonMark: an unclosed fence runs to end of document. Failing to the
    default ref is the safe direction -- the trailer is not honoured."""
    result = _run(
        tmp_path,
        "Refs OMN-17294\n```\nOmnimarket-Source-Ref: attacker/branch\n",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "dev"
