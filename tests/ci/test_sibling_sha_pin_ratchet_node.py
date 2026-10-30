# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shrink-only ratchet on internal-sibling commit-sha pins (OMN-18594)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.nodes.node_sibling_sha_pin_ratchet_compute.handler import (
    NodeSiblingShaPinRatchetCompute,
    count_pins,
    in_scope,
    is_pin_line,
)
from omnibase_infra.nodes.node_sibling_sha_pin_ratchet_compute.models import (
    ModelSiblingShaPinRatchetInput,
)
from omnibase_infra.nodes.node_sibling_sha_pin_ratchet_compute.runtime_sibling_sha_pin_ratchet import (
    main,
)

pytestmark = pytest.mark.unit

SHA = "0123456789abcdef0123456789abcdef01234567"
OTHER = "f" * 40
REPO_ROOT = Path(__file__).resolve().parents[2]
WF = ".github/workflows/a.yml"
PIN = f"    uses: OmniNode-ai/omniclaude/.github/workflows/x.yml@{SHA}"


@pytest.mark.parametrize(
    ("path", "line"),
    [
        (WF, PIN),
        (WF, f"          ref: {SHA} # .github/sibling-pins.yaml"),
        (WF, f"      vocabulary_ref: {SHA}"),
        (".github/sibling-pins.yaml", f"  omnimarket: {SHA}"),
        (".github/omnimarket-contract-pin.yaml", f"omnimarket_contract_ref: {SHA}"),
        (".pre-commit-config.yaml", f"    rev: {SHA}"),
        (
            "pyproject.toml",
            f'omnibase-core = {{ git = "https://github.com/OmniNode-ai/omnibase_core.git", rev = "{SHA}" }}',
        ),
        (
            ".pre-commit-config.yaml",
            f'entry: uv run --with "omnibase-core @ git+https://github.com/OmniNode-ai/omnibase_core@{SHA}" x',
        ),
        (
            "uv.lock",
            f'source = {{ git = "https://github.com/OmniNode-ai/omnimarket.git?rev=dev#{SHA}" }}',
        ),
    ],
)
def test_each_pin_shape_counts(path: str, line: str) -> None:
    assert is_pin_line(path, line)


@pytest.mark.parametrize(
    ("path", "line"),
    [
        (WF, f"      - uses: actions/checkout@{SHA} # v7.0.0"),
        (WF, "          ref: dev"),
        ("pyproject.toml", '"omnibase-core>=0.47.31,<0.48",'),
        ("scripts/x.py", f'SHA256 = "{SHA}{SHA[:24]}"'),
        ("src/omnibase_infra/x.py", f"ref: {SHA}"),
    ],
)
def test_non_pins_do_not_count(path: str, line: str) -> None:
    assert not is_pin_line(path, line)


TAG = "dev-20261029143607-342000ec5702"


def test_uv_lock_line_resolved_from_a_tag_is_not_a_sha_pin() -> None:
    """The lock records the commit a tag resolved to; the dependency is the tag."""
    line = f'source = {{ git = "https://github.com/OmniNode-ai/omnibase_core.git?tag={TAG}#{SHA}" }}'
    assert not is_pin_line("uv.lock", line)
    assert count_pins([("uv.lock", line)]) == {}


@pytest.mark.parametrize(
    "query",
    [f"?rev={SHA}", "?rev=dev", "?branch=dev", ""],
)
def test_uv_lock_line_resolved_from_a_rev_branch_or_bare_sha_still_counts(
    query: str,
) -> None:
    """Only a tag source is exempt: every other lock source is a commit pin."""
    line = f'source = {{ git = "https://github.com/OmniNode-ai/omnibase_core.git{query}#{SHA}" }}'
    assert is_pin_line("uv.lock", line)


def test_a_tag_source_does_not_hide_another_sibling_pinned_to_a_rev() -> None:
    tag_line = f'source = {{ git = "https://github.com/OmniNode-ai/omnibase_core.git?tag={TAG}#{SHA}" }}'
    rev_line = f'source = {{ git = "https://github.com/OmniNode-ai/omnimarket.git?rev={OTHER}#{OTHER}" }}'
    assert count_pins([("uv.lock", tag_line), ("uv.lock", rev_line)]) == {"uv.lock": 1}


def test_tests_docs_and_ticket_contracts_are_out_of_scope() -> None:
    lines = [("tests/ci/x.yml", PIN), ("docs/a.md", PIN), ("contracts/OMN-1.yaml", PIN)]
    assert count_pins(lines) == {}
    assert count_pins([(WF, PIN), (WF, PIN)]) == {WF: 2}


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (".github/actions/tests/action.yml", True),
        (".github/fixtures/x.yml", True),
        (".github/README.md", False),
        ("tests/ci/x.yml", False),
        ("src/test/x.yml", False),
        ("docs/x.yml", False),
        ("src/doc/x.yml", False),
        ("src/fixtures/x.yml", False),
        ("contracts/OMN-1.yaml", False),
    ],
)
def test_in_scope(path: str, expected: bool) -> None:
    assert in_scope(path) is expected


def _status(base: list[tuple[str, str]], head: list[tuple[str, str]]) -> str:
    report = NodeSiblingShaPinRatchetCompute().handle(
        ModelSiblingShaPinRatchetInput(
            base_ref="base", base_lines=tuple(base), lines=tuple(head)
        )
    )
    return report.overall_status


def test_unchanged_and_moved_pins_pass() -> None:
    assert _status([(WF, PIN)], [(WF, PIN)]) != "FAIL"
    assert _status([(WF, PIN)], [(WF, PIN.replace(SHA, OTHER))]) != "FAIL"


def test_removing_a_pin_passes() -> None:
    assert _status([(WF, PIN), (WF, PIN)], [(WF, PIN)]) != "FAIL"


def test_a_new_pin_fails() -> None:
    assert _status([(WF, PIN)], [(WF, PIN), (WF, PIN)]) == "FAIL"
    assert _status([(WF, PIN)], [(WF, PIN), (".github/workflows/b.yml", PIN)]) == "FAIL"


def test_a_pin_moved_between_files_fails_in_the_file_that_gained_it() -> None:
    assert _status([(WF, PIN)], [(".github/workflows/b.yml", PIN)]) == "FAIL"


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        env=scrub_git_location_env(os.environ),
        check=True,
        capture_output=True,
    )


def test_runtime_end_to_end(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q", "-b", "dev")
    _git(tmp_path, "config", "user.email", "ci@example.invalid")
    _git(tmp_path, "config", "user.name", "ci")
    (tmp_path / ".github/workflows").mkdir(parents=True)
    (tmp_path / WF).write_text(PIN + "\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    assert main(["--repo-root", str(tmp_path), "--base", "HEAD"]) == 0
    (tmp_path / WF).write_text(PIN.replace(SHA, OTHER) + "\n")
    assert main(["--repo-root", str(tmp_path), "--base", "HEAD"]) == 0
    (tmp_path / WF).write_text(PIN + "\n" + PIN + "\n")
    assert main(["--repo-root", str(tmp_path), "--base", "HEAD"]) == 1
    assert main(["--repo-root", str(tmp_path), "--base", "no-such-ref"]) == 2


def test_this_repository_holds_no_more_pins_than_its_head() -> None:
    assert main(["--repo-root", str(REPO_ROOT), "--base", "HEAD"]) == 0
