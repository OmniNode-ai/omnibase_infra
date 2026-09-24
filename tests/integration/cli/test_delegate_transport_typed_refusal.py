# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""``onex delegate`` leaves a receipt when the broker is unreachable (OMN-19043).

Sibling of ``test_delegate_timeout_typed_refusal.py``, one layer further down.
That module covers "the command was published and no terminal came back". This
one covers "the command was never published at all", which until OMN-19043 was
the one delegate outcome that produced no artifact of any kind.

End-to-end on purpose, and under ``tests/integration`` for the same reason the
timeout module is: the defect lives in the seam between the transport, the
refusal it raises and the CLI writer that was never reached. A test that stubbed
any one of those three would have passed on 2026-09-21 while the real command
wrote nothing, because each layer was individually behaving correctly -- the
transport raised, the resolver refused a terminal it genuinely could not
resolve, and the caller caught it. The bug was that the sequence left no file.

**The broker here is genuinely unreachable, not mocked.** The fixture binds a
TCP port, reads its number and releases it, so the operating system refuses
every subsequent connection to it for the lifetime of the test. A mock told to
raise proves the branch; it does not prove the behaviour against a real refused
connection, and AC-4's falsifier is a hang, which only a real socket and a
measured wall clock can rule out.
"""

from __future__ import annotations

import json
import socket
import time
import uuid
from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import delegate_command

pytestmark = pytest.mark.integration


#: A delegate-shaped contract standing in for the omnimarket-provided
#: orchestrator, which this repo does not depend on by layering and therefore
#: cannot resolve. Its handler is never reached: the locus probe refuses
#: before anything is dispatched, which is the whole point of the path under
#: test. It round-trips ``correlation_id`` because the receipt shape these
#: tests read is the real delegate receipt, not a stand-in.
_CORRELATED_NOOP_CONTRACT = (
    "---\n"
    "name: correlated_noop\n"
    "node_type: compute\n"
    "terminal_event: onex.evt.proof.correlated-noop-completed.v1\n"
    "handler:\n"
    "  module: tests.fixtures.handler_correlated_noop\n"
    "  class: HandlerCorrelatedNoop\n"
    "  input_model: tests.fixtures.handler_correlated_noop"
    ".ModelCorrelatedNoopRequest\n"
    "handler_routing:\n"
    "  default_handler: tests.fixtures.handler_correlated_noop"
    ":HandlerCorrelatedNoop\n"
)

#: The pre-dispatch locus probe's own bound. The refusal must arrive well
#: inside it; the assertion below allows generous slack because the number
#: under test is "bounded at all", not "fast".
_PROBE_BOUND_SECONDS = 60.0


@pytest.fixture(autouse=True)
def _no_ambient_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the run off the launching host's workspace and drift guard.

    ``OMNI_HOME`` binds the omnimarket co-install drift check, which is a real
    refusal about the machine rather than about the code under test. Left
    bound, these tests would report the host's venv state instead of the
    transport's.
    """
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)


@pytest.fixture(autouse=True)
def _stand_in_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve the orchestrator contract locally, since omnimarket is absent."""
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(_CORRELATED_NOOP_CONTRACT, encoding="utf-8")
    monkeypatch.setattr(
        cli_delegate, "_resolve_packaged_contract", lambda _name: contract_path
    )
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))


@pytest.fixture
def dead_broker() -> str:
    """An address nothing is listening on, and nothing will start listening on.

    Bound only long enough to learn a port the kernel considers free, then
    released. Every connect to it is refused immediately, which is the
    "permanently unreachable broker" AC-4 names -- distinct from the stalled
    broker of AC-2, which answers eventually.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    return f"127.0.0.1:{port}"


def _invoke(*, tmp_path: Path, broker: str) -> tuple[object, float, Path]:
    """Run the real command against the dead broker, returning the state root."""
    state_root = tmp_path / "state"
    started = time.monotonic()
    result = CliRunner().invoke(
        delegate_command,
        [
            "Reply with exactly the word READY",
            "--bus",
            "kafka",
            "--kafka-bootstrap",
            broker,
            "--locus",
            "deployed-lane",
            "--state-root",
            str(state_root),
            "--emit-socket",
            str(tmp_path / "no-daemon.sock"),
        ],
        catch_exceptions=False,
    )
    return result, time.monotonic() - started, state_root


def _sole_receipt(state_root: Path) -> dict[str, object]:
    written = sorted(state_root.glob("runs/*/receipt.json"))
    assert written, (
        "the broker was unreachable and the run left NO receipt.json -- this is "
        "the OMN-19043 defect exactly: every lane is told to read the terminal "
        "from the receipt, and on this path the file did not exist"
    )
    assert len(written) == 1, f"expected exactly one run directory, got {written}"
    parsed = json.loads(written[0].read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


class TestUnreachableBrokerLeavesATypedReceipt:
    """AC-1 and AC-4 through the real command line."""

    def test_all_three_run_files_exist(self, tmp_path: Path, dead_broker: str) -> None:
        """The artifact contract is the same one every other outcome honours.

        Not a bespoke shape for a third kind of failure: same directory, same
        three filenames. A caller that already knows how to read a failed
        delegation can read this one.
        """
        _result, _elapsed, state_root = _invoke(tmp_path=tmp_path, broker=dead_broker)

        run_dirs = sorted((state_root / "runs").iterdir())
        assert len(run_dirs) == 1, f"expected one run directory, got {run_dirs}"
        for name in ("result.txt", "receipt.json", "run.json"):
            assert (run_dirs[0] / name).is_file(), f"{name} was not written"

    def test_the_refusal_is_bounded_not_a_hang(
        self, tmp_path: Path, dead_broker: str
    ) -> None:
        """AC-4's falsifier is a hang, so the assertion is a measured clock."""
        _result, elapsed, _state_root = _invoke(tmp_path=tmp_path, broker=dead_broker)

        assert elapsed < _PROBE_BOUND_SECONDS, (
            f"the unreachable-broker path took {elapsed:.1f}s, which is not "
            "inside the pre-dispatch probe's bound"
        )

    def test_the_cause_names_the_transport_and_not_a_provider(
        self, tmp_path: Path, dead_broker: str
    ) -> None:
        """A broker that never accepted the command says nothing about a provider.

        The falsifier this pins is the tempting one: reaching for a member of
        the delegation failure enum because a failure needs a cause. All three
        members are provider-side, and assigning one would send the next reader
        to the inference provider, the credential or the endpoint -- every one
        of which was fine, because none of them ran.
        """
        _result, _elapsed, state_root = _invoke(tmp_path=tmp_path, broker=dead_broker)
        receipt = _sole_receipt(state_root)

        assert receipt["status"] == "failed"
        assert receipt["terminal_class"] == "transport"
        assert receipt["terminal_failure_cause"] is None

        refusal = receipt["transport_refusal"]
        assert isinstance(refusal, dict)
        assert refusal["awaited"] == "broker_connection"
        # The pre-dispatch probe, not the mid-dispatch connect: a probe that
        # finds no live consumer group may be reporting a lane that is simply
        # not running rather than a sick broker, and those send a reader to
        # different places.
        assert refusal["reason"] == "locus_probe_refused"
        assert refusal["bus"] == "kafka"
        assert refusal["locus"] == "deployed-lane"
        assert refusal["broker"] == dead_broker
        assert refusal["transport_error_type"]
        assert refusal["transport_error"]

    def test_elapsed_and_bound_are_both_recorded_and_consistent(
        self, tmp_path: Path, dead_broker: str
    ) -> None:
        """The two numbers that tell "lane is down" from "lane is overloaded".

        A connect refused instantly and one that burned its whole budget are
        different findings, and only the pair distinguishes them. Asserted
        together because a bound with no elapsed beside it is a figure nobody
        can act on.
        """
        _result, _elapsed, state_root = _invoke(tmp_path=tmp_path, broker=dead_broker)
        refusal = _sole_receipt(state_root)["transport_refusal"]
        assert isinstance(refusal, dict)

        assert float(refusal["bound_seconds"]) > 0.0
        assert float(refusal["elapsed_seconds"]) >= 0.0
        assert float(refusal["elapsed_seconds"]) < float(refusal["bound_seconds"]), (
            "a refused connection should exit far under its bound; an elapsed "
            "at or past the bound means it stalled instead, which is a "
            "different finding and must not be reported as this one"
        )
        assert int(refusal["attempts_permitted"]) >= 1

    def test_no_route_identity_is_synthesised(
        self, tmp_path: Path, dead_broker: str
    ) -> None:
        """No rung ran, so there is nothing to attribute -- even wrongly.

        Fail-closed attribution matters more here than on any other failure
        path: a receipt naming a backend or a model for a run that never left
        the machine would be read as evidence about an inference provider that
        was never contacted.
        """
        _result, _elapsed, state_root = _invoke(tmp_path=tmp_path, broker=dead_broker)
        receipt = _sole_receipt(state_root)

        assert receipt["route_attributed"] is False
        assert receipt["route_unattributed"]
        assert receipt["attempts"] == []
        for forbidden in ("backend_id", "model", "endpoint", "routing_tier"):
            assert forbidden not in receipt, (
                f"{forbidden} was written for a run that never reached a broker"
            )

    def test_the_receipt_carries_this_runs_own_correlation_id(
        self, tmp_path: Path, dead_broker: str
    ) -> None:
        """The refusal is attributable to the invocation that produced it.

        Two concurrent runs against a sick lane must stay tellable apart, and
        the refusal has to keep its identity when it is read apart from its
        envelope -- which is why the id appears both on the receipt and inside
        the refusal rather than only once.
        """
        _result, _elapsed, state_root = _invoke(tmp_path=tmp_path, broker=dead_broker)
        receipt = _sole_receipt(state_root)

        minted = uuid.UUID(str(receipt["correlation_id"]))
        refusal = receipt["transport_refusal"]
        assert isinstance(refusal, dict)
        assert uuid.UUID(str(refusal["correlation_id"])) == minted
        assert uuid.UUID(str(receipt["receipt_id"])) == minted

    def test_the_human_facing_line_is_kept_too(
        self, tmp_path: Path, dead_broker: str
    ) -> None:
        """The receipt is for the caller that parses; stderr is for the watcher.

        Adding the artifact must not silence the message a person staring at a
        terminal actually reads.
        """
        result, _elapsed, _state_root = _invoke(tmp_path=tmp_path, broker=dead_broker)

        assert "TRANSPORT FAILURE" in result.stderr
