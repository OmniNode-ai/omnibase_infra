# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Every ``onex delegate`` invocation that parses arguments leaves a receipt (OMN-19006).

The measured defect: ``delegate-fanout`` builds each item's command with an
``--omni-home`` flag the command does not have. Click refuses an unknown option
at argument parsing, exit 2, before a line of ``cli_delegate`` runs, so the run
had no directory and no ``receipt.json``: the one surface every lane is told to
read did not exist, and the fan-out reader's only account was "no run directory
was created". The reproduction is :func:`test_the_unknown_option_the_fanout_passes`.

The invariant is asserted over the command's exit branches STRUCTURALLY, not one
branch at a time:

* :class:`TestEveryExitBranchOfTheCommandHasACase` reads ``delegate_command`` and
  ``run_delegate`` with :mod:`ast`, inventories every handler that ends the
  command, every ``raise`` outside a handler and every ``sys.exit``, and fails
  when one has no case in :data:`BRANCH_CASES`, so an early return added later
  fails this suite until it is shown to leave a receipt;
* :class:`TestEveryCaseLeavesARunDirectoryAndAReceipt` drives every case through
  the real command and asserts one run directory, a ``receipt.json`` whose
  terminal names the cause in actionable words, and that no case leaves a
  second directory behind;
* :class:`TestEveryWayAnyCallbackCanEndIsAnswered` asserts the funnel itself
  against a synthetic callback, for every kind of exit rather than every
  current call site.

The positive control is :class:`TestASuccessfulRunIsUnchanged`.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import click
import pytest
from click.testing import CliRunner, Result

from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_infra.backends.auto_configure import EventBusResolutionAmbiguousError
from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import (
    DelegateCommand,
    DelegateTimeoutExceededError,
    delegate_command,
)
from omnibase_infra.cli.contract_registry import RegistryUnresolvedError
from omnibase_infra.cli.delegate_lane import DelegateLaneSelectionError
from omnibase_infra.cli.delegate_lane_credentials import DelegateLaneCredentialError
from omnibase_infra.cli.delegate_locus import (
    DelegateLocusRefusedError,
    DelegateLocusSaslRefusedError,
)
from omnibase_infra.cli.omnimarket_drift_guard import OmnimarketDriftError
from omnibase_infra.cli.task_class_registry import TaskClassContractError
from omnibase_infra.errors import ProtocolConfigurationError
from tests.fixtures.handler_correlated_noop import (
    HandlerCorrelatedNoop,
    ModelCorrelatedNoopRequest,
)

pytestmark = pytest.mark.integration

_PROMPT = "Reply with exactly the word READY"
_HANDLER_MODULE = HandlerCorrelatedNoop.__module__
_MODEL_IMPORT_PATH = (
    f"{ModelCorrelatedNoopRequest.__module__}.ModelCorrelatedNoopRequest"
)

_NOOP_CONTRACT = (
    "---\n"
    "name: correlated_noop\n"
    "node_type: compute\n"
    "terminal_event: onex.evt.proof.correlated-noop-completed.v1\n"
    f"input_model: {_MODEL_IMPORT_PATH}\n"
    "handler:\n"
    f"  module: {_HANDLER_MODULE}\n"
    "  class: HandlerCorrelatedNoop\n"
    f"  input_model: {_MODEL_IMPORT_PATH}\n"
    "handler_routing:\n"
    f"  default_handler: {_HANDLER_MODULE}:HandlerCorrelatedNoop\n"
)


@pytest.fixture(autouse=True)
def stand_in_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Route ``run_delegate`` at a fixture contract, offline and co-install-free."""
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    monkeypatch.delenv("ONEX_CONTRACTS_DIR", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
    contract_path = tmp_path / cli_delegate.DELEGATE_NODE_NAME / "contract.yaml"
    contract_path.parent.mkdir()
    contract_path.write_text(_NOOP_CONTRACT, encoding="utf-8")
    monkeypatch.setattr(
        cli_delegate, "_resolve_packaged_contract", lambda _name: contract_path
    )
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
    plain_cwd = tmp_path / "cwd"
    plain_cwd.mkdir()
    monkeypatch.chdir(plain_cwd)


def _run_directories(state_root: Path) -> list[Path]:
    runs = state_root / "runs"
    return sorted(runs.iterdir()) if runs.is_dir() else []


def _sole_receipt(state_root: Path) -> tuple[Path, dict[str, object]]:
    directories = _run_directories(state_root)
    assert len(directories) == 1, (
        f"an invocation leaves exactly one run directory, got {directories}"
    )
    receipt_path = directories[0] / "receipt.json"
    assert receipt_path.is_file(), f"{directories[0]} holds no receipt.json"
    return directories[0], json.loads(receipt_path.read_text(encoding="utf-8"))


def _common(tmp_path: Path, *, with_bus: bool = True) -> list[str]:
    """A command that parses and would dispatch in-process, offline."""
    args = [
        _PROMPT,
        "--task-type",
        "summarization",
        "--state-root",
        str(tmp_path / "state"),
        "--emit-socket",
        str(tmp_path / "no-daemon.sock"),
    ]
    if with_bus:
        args += ["--bus", "inmemory", "--locus", "in-process"]
    return args


def _raises(exc: BaseException) -> Callable[..., object]:
    def _raise(*_args: object, **_kwargs: object) -> object:
        raise exc

    return _raise


class _ProfileWithBrokenLaneBinding:
    def __init__(self, **_kwargs: object) -> None: ...

    def lane_binding(self) -> object:
        raise ModelOnexError(message="the lane binding file is unreadable")


class _ProfileWithNoLaneBinding:
    def __init__(self, **_kwargs: object) -> None: ...

    def lane_binding(self) -> None:
        return None


@dataclass(frozen=True)
class BranchCase:
    """One way the command can end before a dispatched run produces a receipt."""

    covers: str
    """The inventory key this case exercises (see :func:`exit_inventory`)."""
    args: Callable[[Path], list[str]]
    reason: str
    """A substring the receipt's ``failure_reason`` must carry."""
    patches: dict[str, Callable[..., object]] = field(default_factory=dict)
    expected_stage: str | None = None
    """``refusal.stage`` the funnel records; ``None`` when a dispatch path
    writes the receipt itself (the transport refusal, the settle of a run that
    returned non-zero) and the funnel only has to leave it alone."""


_KEY_VALUE_ERROR = "except ValueError"
_KEY_JSON_HUMAN = "raise click.UsageError '--json and --human are mutually exclusiv'"
_KEY_SYS_EXIT = "call sys.exit"
_KEY_KAFKA_WITHOUT_BUS = "raise ValueError '--kafka-bootstrap is only valid with --b'"

BRANCH_CASES: dict[str, BranchCase] = {
    "json-and-human": BranchCase(
        covers=_KEY_JSON_HUMAN,
        args=lambda tmp: [*_common(tmp), "--json", "--human"],
        reason="--json and --human are mutually exclusive",
        expected_stage="refused_before_dispatch",
    ),
    "malformed-ticket": BranchCase(
        covers=_KEY_VALUE_ERROR,
        args=lambda tmp: [*_common(tmp), "--ticket", "not a ticket"],
        reason="is not a ticket identifier",
        expected_stage="refused_before_dispatch",
    ),
    "malformed-caller-lane": BranchCase(
        covers=_KEY_VALUE_ERROR,
        args=lambda tmp: [*_common(tmp), "--caller-lane", "two words"],
        reason="--caller-lane",
        expected_stage="refused_before_dispatch",
    ),
    "response-contract-not-json": BranchCase(
        covers=_KEY_VALUE_ERROR,
        args=lambda tmp: [*_common(tmp), "--response-contract", "not json"],
        reason="--response-contract is neither a readable .json file nor valid",
        expected_stage="refused_before_dispatch",
    ),
    "response-contract-not-an-object": BranchCase(
        covers=_KEY_VALUE_ERROR,
        args=lambda tmp: [*_common(tmp), "--response-contract", "[1, 2]"],
        reason="--response-contract must be a JSON object",
        expected_stage="refused_before_dispatch",
    ),
    "response-contract-missing-file": BranchCase(
        covers=_KEY_VALUE_ERROR,
        args=lambda tmp: [
            *_common(tmp),
            "--response-contract",
            str(tmp / "absent.json"),
        ],
        reason="not a readable file",
        expected_stage="refused_before_dispatch",
    ),
    "both-task-class-spellings": BranchCase(
        covers=_KEY_VALUE_ERROR,
        args=lambda tmp: [*_common(tmp), "--task-class", "summarization"],
        reason="two spellings of the same flag",
        expected_stage="refused_before_dispatch",
    ),
    "empty-backend-id": BranchCase(
        covers=_KEY_VALUE_ERROR,
        args=lambda tmp: [*_common(tmp), "--backend-id", "  "],
        reason="--backend-id was given an empty value",
        expected_stage="refused_before_dispatch",
    ),
    "replace-mode-without-authority": BranchCase(
        covers=_KEY_VALUE_ERROR,
        args=lambda tmp: [
            *_common(tmp),
            "--criteria-mode",
            "replace-task-class",
            "--criteria",
            "a-reject-only-slug",
        ],
        reason="can never be accepted",
        patches={
            "load_acceptance_capable_criteria": lambda: frozenset({"accepts"}),
        },
        expected_stage="refused_before_dispatch",
    ),
    "kafka-bootstrap-without-bus": BranchCase(
        covers=_KEY_KAFKA_WITHOUT_BUS,
        args=lambda tmp: [*_common(tmp, with_bus=False), "--kafka-bootstrap", "h:1"],
        reason="--kafka-bootstrap is only valid with --bus kafka",
        expected_stage="refused_before_dispatch",
    ),
    "state-root-unresolvable": BranchCase(
        covers="except ProtocolConfigurationError",
        args=lambda tmp: _common(tmp),
        reason="ONEX_STATE_DIR must be an absolute path",
        patches={
            "resolve_state_root": _raises(
                ProtocolConfigurationError("ONEX_STATE_DIR must be an absolute path")
            )
        },
        expected_stage="refused_before_dispatch",
    ),
    "install-drift": BranchCase(
        covers="except OmnimarketDriftError",
        args=lambda tmp: _common(tmp),
        reason="omnimarket is stale",
        patches={
            "check_omnimarket_drift": _raises(
                OmnimarketDriftError("omnimarket is stale: run the reconcile command")
            )
        },
        expected_stage="refused_before_dispatch",
    ),
    "task-class-contract-unreadable": BranchCase(
        covers="except (TaskClassContractError, ValueError)",
        args=lambda tmp: _common(tmp),
        reason="the task-class contract cannot be read",
        patches={
            "resolve_task_class": _raises(
                TaskClassContractError("the task-class contract cannot be read")
            )
        },
        expected_stage="refused_before_dispatch",
    ),
    "request-refused-by-the-contract": BranchCase(
        covers="except RegistryUnresolvedError",
        args=lambda tmp: _common(tmp),
        reason="the delegate request model is not advertised",
        patches={
            "validate_request_against_contract": _raises(
                RegistryUnresolvedError("the delegate request model is not advertised")
            )
        },
        expected_stage="refused_before_dispatch",
    ),
    "lane-binding-unreadable": BranchCase(
        covers="except ModelOnexError",
        args=lambda tmp: _common(tmp, with_bus=False),
        reason="the lane binding file is unreadable",
        patches={"StoreDeveloperProfile": _ProfileWithBrokenLaneBinding},
        expected_stage="refused_before_dispatch",
    ),
    "default-transport-ambiguous": BranchCase(
        covers="except (EventBusResolutionAmbiguousError, ProtocolConfigurationError)",
        args=lambda tmp: _common(tmp, with_bus=False),
        reason="no transport could be decided",
        patches={
            "StoreDeveloperProfile": _ProfileWithNoLaneBinding,
            "resolve_default_bus": _raises(
                EventBusResolutionAmbiguousError("no transport could be decided")
            ),
        },
        expected_stage="refused_before_dispatch",
    ),
    "kafka-without-a-lane": BranchCase(
        covers="except DelegateLaneSelectionError",
        args=lambda tmp: [
            *_common(tmp, with_bus=False),
            "--bus",
            "kafka",
            "--locus",
            "deployed-lane",
        ],
        reason="lane",
        patches={
            "resolve_lane_target": _raises(
                DelegateLaneSelectionError("a kafka run names a lane or a broker")
            )
        },
        expected_stage="refused_before_dispatch",
    ),
    "lane-identity-missing": BranchCase(
        covers="except DelegateLaneCredentialError",
        args=lambda tmp: _common(tmp),
        reason="this machine holds no identity for the lane",
        patches={
            "resolve_lane_client_transport_for": _raises(
                DelegateLaneCredentialError(
                    "this machine holds no identity for the lane"
                )
            )
        },
        expected_stage="refused_before_dispatch",
    ),
    "locus-probe-refused": BranchCase(
        covers="except DelegateLocusRefusedError",
        args=lambda tmp: _common(tmp),
        reason="no live consumer is bound to the command topic",
        patches={
            "resolve_delegate_locus": _raises(
                DelegateLocusRefusedError(
                    "no live consumer is bound to the command topic"
                )
            )
        },
    ),
    "locus-sasl-refused": BranchCase(
        covers="except DelegateLocusRefusedError",
        args=lambda tmp: _common(tmp),
        reason="the broker rejected the SASL login",
        patches={
            "resolve_delegate_locus": _raises(
                DelegateLocusSaslRefusedError(
                    "the broker rejected the SASL login", broker="b:1", principal="p"
                )
            )
        },
    ),
    "hard-timeout": BranchCase(
        covers="except DelegateTimeoutExceededError",
        args=lambda tmp: _common(tmp),
        reason="",
        patches={
            "run_receipt_mode": _raises(
                DelegateTimeoutExceededError("delegation exceeded its hard bound")
            ),
        },
    ),
    "exit-without-a-receipt": BranchCase(
        covers=_KEY_SYS_EXIT,
        args=lambda tmp: _common(tmp),
        reason="exit",
        patches={"run_receipt_mode": lambda **_: 1},
    ),
}


def exit_inventory() -> set[str]:
    """Every way ``delegate_command`` and ``run_delegate`` can end the command.

    A handler that raises or returns, a ``raise`` outside any handler, and a
    call that exits the process. Read from the source so that a branch added
    later is in the set before anyone remembers to name it.
    """
    source = Path(cli_delegate.__file__).read_text(encoding="utf-8")
    inventory: set[str] = set()
    for node in ast.parse(source).body:
        if not (
            isinstance(node, ast.FunctionDef)
            and node.name in {"delegate_command", "run_delegate"}
        ):
            continue
        in_handler: set[int] = set()
        for handler in (n for n in ast.walk(node) if isinstance(n, ast.ExceptHandler)):
            ends_command = any(
                isinstance(inner, ast.Raise | ast.Return) for inner in ast.walk(handler)
            )
            if handler.type is not None and ends_command:
                inventory.add(f"except {ast.unparse(handler.type)}")
            in_handler.update(id(inner) for inner in ast.walk(handler))
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Raise)
                and isinstance(inner.exc, ast.Call)
                and id(inner) not in in_handler
            ):
                first = inner.exc.args[0] if inner.exc.args else None
                literal = (
                    first.value[:40]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str)
                    else ""
                )
                inventory.add(f"raise {ast.unparse(inner.exc.func)} {literal!r}")
            if isinstance(inner, ast.Call) and ast.unparse(inner.func) == "sys.exit":
                inventory.add("call sys.exit")
    return inventory


class TestEveryExitBranchOfTheCommandHasACase:
    """The inventory of exit branches is the inventory of cases, by construction."""

    def test_the_command_is_the_funnel_class(self) -> None:
        assert isinstance(delegate_command, DelegateCommand)

    def test_every_exit_branch_in_the_source_has_a_case(self) -> None:
        covered = {case.covers for case in BRANCH_CASES.values()}
        uncovered = exit_inventory() - covered
        assert not uncovered, (
            "these exit branches of the delegate command have no case that "
            f"proves they leave a receipt: {sorted(uncovered)}"
        )

    def test_no_case_names_a_branch_that_is_gone(self) -> None:
        stale = {case.covers for case in BRANCH_CASES.values()} - exit_inventory()
        assert not stale, f"cases cover branches that no longer exist: {sorted(stale)}"

    def test_no_other_function_ends_the_process_behind_the_funnel(self) -> None:
        source = Path(cli_delegate.__file__).read_text(encoding="utf-8")
        enders = {
            node.name
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef)
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and ast.unparse(call.func)
            in {"sys.exit", "os._exit", "ctx.exit", "ctx.abort", "ctx.fail"}
        }
        assert enders == {"delegate_command"}


class TestEveryCaseLeavesARunDirectoryAndAReceipt:
    """AC2 and AC3 over every exit branch, through the real command."""

    @pytest.mark.parametrize("case_id", sorted(BRANCH_CASES))
    def test_the_case_leaves_one_run_with_an_actionable_receipt(
        self, case_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        case = BRANCH_CASES[case_id]
        for name, replacement in case.patches.items():
            monkeypatch.setattr(cli_delegate, name, replacement)
        result = CliRunner().invoke(delegate_command, case.args(tmp_path))
        assert result.exit_code != 0, result.output

        run_dir, receipt = _sole_receipt(tmp_path / "state")
        assert (run_dir / "result.txt").is_file()
        assert (run_dir / "run.json").is_file()
        assert receipt["status"] == "failed"
        assert receipt["run_id"] == run_dir.name
        assert run_dir.name in result.output, (
            "the caller is told which run directory to read"
        )

        reason = str(receipt["failure_reason"])
        assert case.reason in reason
        if case.expected_stage is not None:
            refusal = receipt["refusal"]
            assert isinstance(refusal, dict)
            assert refusal["stage"] == case.expected_stage
            assert refusal["exit_code"] == result.exit_code
            assert case.reason in str(refusal["message"])
            # Never a bare classification: the cause is followed by what to do.
            assert refusal["remedy"]
            assert str(refusal["remedy"]) in reason
            assert receipt["terminal_recorded"] is False


class TestArgumentParsingFailuresLeaveAReceipt:
    """The refusals click makes before the command body runs."""

    def test_the_unknown_option_the_fanout_passes(self, tmp_path: Path) -> None:
        """The reproduction: ``delegate-fanout`` passes ``--omni-home``.

        Before this change: exit 2, no run directory, and the fan-out row said
        "no run directory was created".
        """
        state_root = tmp_path / "state"
        result = CliRunner().invoke(
            delegate_command,
            [
                _PROMPT,
                "--bus",
                "kafka",
                "--locus",
                "deployed-lane",
                "--lane",
                "dev",
                "--omni-home",
                str(tmp_path),
                "--state-root",
                str(state_root),
                "--task-type",
                "document",
            ],
        )
        assert result.exit_code == 2
        _, receipt = _sole_receipt(state_root)
        assert "No such option '--omni-home'" in str(receipt["failure_reason"])
        refusal = receipt["refusal"]
        assert isinstance(refusal, dict)
        assert refusal["stage"] == "argument_parsing"
        assert refusal["error_type"] == "NoSuchOption"

    @pytest.mark.parametrize(
        ("arguments", "reason"),
        [
            pytest.param(
                ["--bus", "carrier-pigeon"], "carrier-pigeon", id="bad-choice"
            ),
            pytest.param(["--max-tokens", "many"], "many", id="not-an-integer"),
            pytest.param(["--timeout", "0"], "0", id="below-the-range"),
            pytest.param(["--no-such-flag"], "--no-such-flag", id="unknown-flag"),
        ],
    )
    def test_a_value_click_refuses_names_itself_in_the_receipt(
        self, arguments: list[str], reason: str, tmp_path: Path
    ) -> None:
        state_root = tmp_path / "state"
        result = CliRunner().invoke(
            delegate_command,
            [_PROMPT, "--state-root", str(state_root), *arguments],
        )
        assert result.exit_code == 2
        _, receipt = _sole_receipt(state_root)
        assert reason in str(receipt["failure_reason"])

    def test_a_missing_prompt_leaves_a_receipt_where_state_root_says(
        self, tmp_path: Path
    ) -> None:
        state_root = tmp_path / "state"
        result = CliRunner().invoke(
            delegate_command, [f"--state-root={state_root}", "--task-type", "document"]
        )
        assert result.exit_code == 2
        _, receipt = _sole_receipt(state_root)
        assert "PROMPT" in str(receipt["failure_reason"])

    def test_with_no_state_root_given_onex_state_dir_is_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OMN-19232: the command's own resolution order, never the cwd."""
        env_root = tmp_path / "env-state"
        monkeypatch.setenv("ONEX_STATE_DIR", str(env_root))
        result = CliRunner().invoke(delegate_command, [_PROMPT, "--no-such-flag"])
        assert result.exit_code == 2
        _, receipt = _sole_receipt(env_root)
        assert "--no-such-flag" in str(receipt["failure_reason"])
        assert not (tmp_path / "cwd" / ".onex_state").exists()

    def test_with_nothing_set_the_home_default_is_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.delenv("ONEX_STATE_DIR", raising=False)
        monkeypatch.setenv("HOME", str(home))
        result = CliRunner().invoke(delegate_command, [_PROMPT, "--no-such-flag"])
        assert result.exit_code == 2
        _, receipt = _sole_receipt(home / ".onex_state")
        assert "--no-such-flag" in str(receipt["failure_reason"])
        assert not (tmp_path / "cwd" / ".onex_state").exists()

    def test_an_unresolvable_root_says_so_and_writes_nothing_under_the_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ONEX_STATE_DIR", "relative/state")
        result = CliRunner().invoke(delegate_command, [_PROMPT, "--no-such-flag"])
        assert result.exit_code == 2
        assert "no refusal receipt was written" in result.output
        assert "ONEX_STATE_DIR must be an absolute path" in result.output
        assert list((tmp_path / "cwd").iterdir()) == []

    def test_help_is_not_a_refusal_and_writes_nothing(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            delegate_command, ["--help", "--state-root", str(tmp_path / "state")]
        )
        assert result.exit_code == 0
        assert not (tmp_path / "state").exists()


class TestAnErrorNoCheckNamesStillLeavesAReceipt:
    def test_an_unhandled_error_is_a_receipt_and_not_a_bare_traceback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            cli_delegate, "_write_payload", _raises(RuntimeError("scratch is full"))
        )
        result = CliRunner().invoke(delegate_command, _common(tmp_path))
        assert isinstance(result.exception, RuntimeError)
        _, receipt = _sole_receipt(tmp_path / "state")
        assert "scratch is full" in str(receipt["failure_reason"])
        refusal = receipt["refusal"]
        assert isinstance(refusal, dict)
        assert refusal["stage"] == "unhandled_error"
        assert refusal["error_type"] == "RuntimeError"


class TestEveryWayAnyCallbackCanEndIsAnswered:
    """The funnel itself, against a synthetic body: every kind of exit, not every call site."""

    @staticmethod
    def _command(callback: Callable[..., object]) -> DelegateCommand:
        return DelegateCommand(
            "delegate",
            params=[
                click.Argument(["prompt"]),
                click.Option(["--state-root"], type=click.Path(path_type=Path)),
            ],
            callback=callback,
        )

    @pytest.mark.parametrize(
        "ending",
        [
            pytest.param(
                lambda: (_ for _ in ()).throw(click.UsageError("u")), id="usage"
            ),
            pytest.param(
                lambda: (_ for _ in ()).throw(click.ClickException("c")), id="click"
            ),
            pytest.param(
                lambda: (_ for _ in ()).throw(click.BadParameter("b")),
                id="bad-parameter",
            ),
            pytest.param(
                lambda: (_ for _ in ()).throw(ValueError("v")), id="value-error"
            ),
            pytest.param(lambda: (_ for _ in ()).throw(OSError("o")), id="os-error"),
            pytest.param(lambda: (_ for _ in ()).throw(SystemExit(3)), id="exit-3"),
            pytest.param(
                lambda: (_ for _ in ()).throw(SystemExit("text")), id="exit-text"
            ),
            pytest.param(
                lambda: (_ for _ in ()).throw(KeyboardInterrupt()), id="interrupt"
            ),
        ],
    )
    def test_each_ending_without_a_receipt_leaves_one(
        self, ending: Callable[[], object], tmp_path: Path
    ) -> None:
        state_root = tmp_path / "state"
        result = CliRunner().invoke(
            self._command(lambda **_: ending()),
            ["a prompt", "--state-root", str(state_root)],
        )
        assert result.exit_code != 0
        _, receipt = _sole_receipt(state_root)
        assert receipt["status"] == "failed"
        assert receipt["failure_reason"]
        refusal = receipt["refusal"]
        assert isinstance(refusal, dict)
        assert refusal["stage"] in {
            "refused_before_dispatch",
            "exit_without_receipt",
            "unhandled_error",
        }

    def test_a_receipt_the_body_wrote_under_the_run_id_is_never_replaced(
        self, tmp_path: Path
    ) -> None:
        state_root = tmp_path / "state"

        def body(**_: object) -> None:
            ctx = click.get_current_context()
            run_id = ctx.meta["omnibase_infra.delegate.run_id"]
            run_dir = state_root / "runs" / str(run_id)
            run_dir.mkdir(parents=True)
            (run_dir / "receipt.json").write_text('{"owned": "by the body"}')
            raise SystemExit(1)

        result = CliRunner().invoke(
            self._command(body), ["a prompt", "--state-root", str(state_root)]
        )
        assert result.exit_code == 1
        _, receipt = _sole_receipt(state_root)
        assert receipt == {"owned": "by the body"}

    def test_an_unwritable_state_root_does_not_hide_the_original_error(
        self, tmp_path: Path
    ) -> None:
        blocker = tmp_path / "state"
        blocker.write_text("a file where the state root should be")
        result = CliRunner().invoke(
            self._command(
                lambda **_: (_ for _ in ()).throw(click.UsageError("the cause"))
            ),
            ["a prompt", "--state-root", str(blocker)],
        )
        assert result.exit_code == 2
        assert "the cause" in result.output
        assert "could not write the refusal receipt" in result.output


class TestASuccessfulRunIsUnchanged:
    """AC5: the positive control. A run that completes is not touched by the funnel."""

    def test_a_completed_run_has_one_run_directory_and_no_refusal(
        self, tmp_path: Path
    ) -> None:
        result: Result = CliRunner().invoke(
            delegate_command, _common(tmp_path), catch_exceptions=False
        )
        assert result.exit_code == 0, result.output
        run_dir, receipt = _sole_receipt(tmp_path / "state")
        assert "refusal" not in receipt
        assert receipt.get("terminal_class") != "refused"
        assert "REFUSED" not in result.output
        assert (run_dir / "result.txt").is_file()
        assert (run_dir / "run.json").is_file()
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert run["run_id"] == run_dir.name
        assert run["task_type_resolution"] != "unresolved"
