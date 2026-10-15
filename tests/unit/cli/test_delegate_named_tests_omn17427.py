# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex delegate`` names the tests a code delegation is graded by (OMN-17427).

The dev lane's gate refused all 125 Qwen code runs as INSUFFICIENT because no
named test run reached it, 84 of them correct answers. omnimarket's delegation
port reads the tests from the request ``metadata`` map (``named_tests``,
``named_tests_answer_path`` and ``named_tests_files``, a JSON object of
workspace-relative path to source), and this CLI had no flag that wrote them.

What these tests pin:

* the three options write exactly those keys, with the test files' source
  inline so the runner needs no workspace;
* a partial or unsafe naming is a usage error naming the flag, never dropped;
* the tests ride in the request payload the bus command is built from;
* a run that would reach the deployed lane is refused before anything is
  published, because that lane's dispatch port cannot carry the tests yet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import click
import pytest
from click.testing import CliRunner

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import (
    _request_payload,
    _write_payload,
    delegate_command,
)
from omnibase_infra.cli.model_delegate_locus_decision import ModelDelegateLocusDecision
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus
from omnibase_infra.models.delegation.model_delegate_named_tests import (
    MAX_NAMED_TEST_FILE_CHARS,
    MAX_NAMED_TEST_FILES,
    MAX_NAMED_TESTS,
    ModelDelegateNamedTests,
)

pytestmark = pytest.mark.unit

_SOURCE = "from solution import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_add.py").write_text(_SOURCE, encoding="utf-8")
    (tests / "test_more.py").write_text("def test_more():\n    pass\n", "utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_the_command_declares_the_three_options() -> None:
    declared = {
        opt for param in delegate_command.params for opt in getattr(param, "opts", ())
    }
    assert {"--named-test", "--test-file", "--answer-path"} <= declared


def test_no_flags_name_no_tests(workspace: Path) -> None:
    assert ModelDelegateNamedTests.from_flags((), (), None) is None


def test_a_test_file_alone_names_every_test_in_it(workspace: Path) -> None:
    named = ModelDelegateNamedTests.from_flags(
        (), ("tests/test_add.py",), "solution.py"
    )
    assert named is not None
    metadata = named.as_metadata()
    assert metadata["named_tests"] == "tests/test_add.py"
    assert metadata["named_tests_answer_path"] == "solution.py"
    assert json.loads(metadata["named_tests_files"]) == {"tests/test_add.py": _SOURCE}
    assert set(metadata) == {
        "named_tests",
        "named_tests_answer_path",
        "named_tests_files",
    }


def test_named_node_ids_are_written_in_the_order_given(workspace: Path) -> None:
    named = ModelDelegateNamedTests.from_flags(
        ("tests/test_more.py::test_more", "tests/test_add.py::test_add"),
        ("tests/test_add.py", "tests/test_more.py"),
        "solution.py",
    )
    assert named is not None
    metadata = named.as_metadata()
    assert metadata["named_tests"] == (
        "tests/test_more.py::test_more tests/test_add.py::test_add"
    )
    assert sorted(json.loads(metadata["named_tests_files"])) == [
        "tests/test_add.py",
        "tests/test_more.py",
    ]


def test_an_absolute_test_file_lands_at_its_own_name(workspace: Path) -> None:
    named = ModelDelegateNamedTests.from_flags(
        (), (str(workspace / "tests" / "test_add.py"),), "solution.py"
    )
    assert named is not None
    assert named.as_metadata()["named_tests"] == "test_add.py"


@pytest.mark.parametrize(
    ("ids", "files", "answer", "fragment"),
    [
        ((), (), "solution.py", "--answer-path names where an answer"),
        (("tests/test_add.py",), (), "solution.py", "--test-file"),
        ((), ("tests/test_add.py",), None, "--answer-path"),
        (("tests/test_add.py",), ("tests/test_add.py",), None, "--answer-path"),
        (
            ("tests/test_other.py::test_x",),
            ("tests/test_add.py",),
            "solution.py",
            "is not in any --test-file",
        ),
        (("-k",), ("tests/test_add.py",), "solution.py", "is not a pytest node id"),
        (
            ("tests/test_add.py::test a",),
            ("tests/test_add.py",),
            "solution.py",
            "whitespace",
        ),
        ((), ("tests/missing.py",), "solution.py", "cannot be read"),
        ((), ("tests/test_add.py",), "../solution.py", "must stay inside"),
        ((), ("tests/test_add.py",), "/etc/solution.py", "must stay inside"),
        ((), ("tests/test_add.py",), "tests/test_add.py", "over a test file"),
        ((), ("../test_add.py",), "solution.py", "must stay inside"),
    ],
)
def test_a_partial_or_unsafe_naming_is_refused_by_name(
    workspace: Path,
    ids: tuple[str, ...],
    files: tuple[str, ...],
    answer: str | None,
    fragment: str,
) -> None:
    with pytest.raises(ValueError, match=fragment):
        ModelDelegateNamedTests.from_flags(ids, files, answer)


def test_the_consumer_limits_are_enforced_here(workspace: Path) -> None:
    many = tuple(f"tests/test_{n}.py" for n in range(MAX_NAMED_TEST_FILES + 1))
    for name in many:
        (workspace / name).write_text("def test_x():\n    pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at most"):
        ModelDelegateNamedTests.from_flags((), many, "solution.py")
    with pytest.raises(ValueError, match="at most"):
        ModelDelegateNamedTests.from_flags(
            tuple(f"tests/test_add.py::t{n}" for n in range(MAX_NAMED_TESTS + 1)),
            ("tests/test_add.py",),
            "solution.py",
        )
    big = workspace / "tests" / "test_big.py"
    big.write_text("#" * (MAX_NAMED_TEST_FILE_CHARS + 1), encoding="utf-8")
    with pytest.raises(ValueError, match="characters"):
        ModelDelegateNamedTests.from_flags((), ("tests/test_big.py",), "solution.py")


def test_the_tests_ride_in_the_request_payload(workspace: Path) -> None:
    named = ModelDelegateNamedTests.from_flags(
        ("tests/test_add.py::test_add",), ("tests/test_add.py",), "solution.py"
    )
    assert named is not None
    payload = _request_payload(
        prompt="Write add(a, b).",
        task_type="code_generation",
        source="claude-code",
        max_tokens=None,
        correlation_id=uuid4(),
        ticket_id="OMN-17427",
        named_tests=named.as_metadata(),
    )
    assert payload["metadata"] == {
        "ticket_id": "OMN-17427",
        "named_tests": "tests/test_add.py::test_add",
        "named_tests_answer_path": "solution.py",
        "named_tests_files": json.dumps({"tests/test_add.py": _SOURCE}, sort_keys=True),
    }


def test_the_payload_file_the_bus_command_is_built_from_carries_them(
    workspace: Path, tmp_path: Path
) -> None:
    named = ModelDelegateNamedTests.from_flags(
        (), ("tests/test_add.py",), "solution.py"
    )
    assert named is not None
    path = _write_payload(
        prompt="Write add(a, b).",
        task_type="code_generation",
        source="claude-code",
        max_tokens=None,
        state_root=tmp_path / "state",
        run_id=uuid4(),
        correlation_id=uuid4(),
        named_tests=named.as_metadata(),
    )
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["metadata"]["named_tests"] == "tests/test_add.py"
    assert loaded["metadata"]["named_tests_answer_path"] == "solution.py"


def test_absent_tests_write_no_key(workspace: Path) -> None:
    payload = _request_payload(
        prompt="Write add(a, b).",
        task_type="code_generation",
        source="claude-code",
        max_tokens=None,
        correlation_id=uuid4(),
        named_tests={},
    )
    assert "metadata" not in payload


def test_the_options_refuse_before_anything_runs(workspace: Path) -> None:
    result = CliRunner().invoke(
        delegate_command,
        ["Write add.", "--task-type", "code_generation", "--named-test", "t.py::a"],
    )
    assert result.exit_code == 2
    assert "--test-file" in result.output


def test_the_answer_path_alone_is_a_usage_error(workspace: Path) -> None:
    result = CliRunner().invoke(
        delegate_command,
        ["Write add.", "--task-type", "code_generation", "--answer-path", "sol.py"],
    )
    assert result.exit_code == 2
    assert "--answer-path names where an answer" in result.output


_COMMAND = "onex.cmd.synthetic.delegate-skill.v1"
_BROKER = "broker.invalid:19092"


def _chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    locus: EnumDelegateLocus,
) -> list[dict[str, Any]]:
    """Run the real command with a stand-in for the runtime; return what it was handed."""
    calls: list[dict[str, Any]] = []
    contract = tmp_path / "contract.yaml"
    contract.write_text("name: node_delegate_skill_orchestrator\n", encoding="utf-8")

    def receipt(**kwargs: Any) -> int:
        calls.append(
            {**kwargs, "payload": json.loads(kwargs["input_path"].read_text())}
        )
        return 0

    def resolve(**_: Any) -> ModelDelegateLocusDecision:
        return ModelDelegateLocusDecision(
            locus=locus,
            resolved_from="test",
            orchestrator_contract=str(contract),
            orchestrator_distribution="synthetic",
            command_topic=_COMMAND,
            broker=_BROKER if locus is EnumDelegateLocus.DEPLOYED_LANE else "",
            lane_consumer_groups=(),
        )

    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
    monkeypatch.setattr(cli_delegate, "_resolve_packaged_contract", lambda _: contract)
    monkeypatch.setattr(cli_delegate, "run_receipt_mode", receipt)
    monkeypatch.setattr(cli_delegate, "resolve_delegate_locus", resolve)
    return calls


_NAMING = [
    "--task-type",
    "code_generation",
    "--test-file",
    "tests/test_add.py",
    "--named-test",
    "tests/test_add.py::test_add",
    "--answer-path",
    "solution.py",
]


def test_golden_chain_in_process_the_run_is_handed_the_tests(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _chain(monkeypatch, tmp_path, EnumDelegateLocus.IN_PROCESS)
    result = CliRunner().invoke(
        delegate_command,
        [
            "Write add(a, b).",
            *_NAMING,
            "--bus",
            "inmemory",
            "--locus",
            "in-process",
            "--state-root",
            str(tmp_path / "state"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "named tests: tests/test_add.py::test_add" in result.stderr
    (call,) = calls
    metadata = call["payload"]["metadata"]
    assert metadata["named_tests"] == "tests/test_add.py::test_add"
    assert metadata["named_tests_answer_path"] == "solution.py"
    assert json.loads(metadata["named_tests_files"]) == {"tests/test_add.py": _SOURCE}
    assert call["payload"]["task_type"] == "code_generation"


def test_golden_chain_deployed_lane_is_handed_the_tests_too(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OMN-17427: the deployed lane carries named tests, so the CLI no longer refuses them."""
    calls = _chain(monkeypatch, tmp_path, EnumDelegateLocus.DEPLOYED_LANE)
    result = CliRunner().invoke(
        delegate_command,
        [
            "Write add(a, b).",
            *_NAMING,
            "--bus",
            "kafka",
            "--locus",
            "deployed-lane",
            "--kafka-bootstrap",
            _BROKER,
            "--state-root",
            str(tmp_path / "state"),
        ],
    )
    assert result.exit_code == 0, result.output
    (call,) = calls
    metadata = call["payload"]["metadata"]
    assert metadata["named_tests"] == "tests/test_add.py::test_add"
    assert metadata["named_tests_answer_path"] == "solution.py"
    assert json.loads(metadata["named_tests_files"]) == {"tests/test_add.py": _SOURCE}
