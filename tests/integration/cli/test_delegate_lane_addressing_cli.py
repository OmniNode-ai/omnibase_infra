# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""End-to-end CLI coverage for lane-resolved broker addressing (OMN-16871).

The unit module beside this one drives the resolver and the policy directly.
This one goes through ``click`` -- the real command, the real option parsing,
the real refusal path, and a real declaration file on disk -- because the
defect was only ever reachable from a command line. `onex delegate --bus
kafka` typed with no broker resolved its address from the ambient
``KAFKA_BOOTSTRAP_SERVERS``, which on the launching Mac names the `.201`
STABILITY-TEST lane, so ad hoc delegations from a developer shell published
onto a governed proof lane.

Every test here exports that ambient variable to the stability address before
invoking, so a silent return of the env path is visible as a test that
resolves an address where it should have refused.

Dispatch itself needs a co-installed omnimarket and a live model endpoint,
neither of which belongs in this gate. Everything under test happens in front
of dispatch: option parsing, the declaration read, and the refusal.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.cli_delegate import delegate_command
from omnibase_infra.cli.delegate_lane import LANE_DECLARATION_RELATIVE_PATH

pytestmark = pytest.mark.integration

#: The ambient value the ticket is about: the governed stability-test lane.
AMBIENT_STABILITY_BROKER = "192.168.86.201:39092"  # onex-allow-internal-ip OMN-16871 reason="test fixture quoting the ambient env value the CLI must no longer resolve; not a configurable endpoint"

#: Declared dev address, deliberately unlike the ambient one.
DECLARED_DEV_BROKER = "declared-dev.example:19092"

_DECLARATION = f"""
lanes:
  dev:
    broker: "{DECLARED_DEV_BROKER}"
    security_protocol: SASL_PLAINTEXT
    sasl_mechanism: SCRAM-SHA-256
  stability-test:
    broker: "stability.example:39092"
    security_protocol: PLAINTEXT
"""


@pytest.fixture(autouse=True)
def _ambient_env_names_the_governed_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", AMBIENT_STABILITY_BROKER)
    monkeypatch.delenv("ONEX_CONTRACTS_DIR", raising=False)


def _workspace(root: Path) -> Path:
    declaration = root / LANE_DECLARATION_RELATIVE_PATH
    declaration.parent.mkdir(parents=True, exist_ok=True)
    declaration.write_text(_DECLARATION, encoding="utf-8")
    return root


def _invoke(args: list[str]) -> object:
    return CliRunner().invoke(delegate_command, args, catch_exceptions=False)


class TestASharedBusRunMustNameItsLane:
    def test_no_lane_is_refused_and_the_refusal_is_actionable(
        self, tmp_path: Path
    ) -> None:
        """RED before the fix: this published to whatever the shell exported."""
        result = _invoke(
            [
                "document the router",
                "--task-type",
                "document",
                "--bus",
                "kafka",
                "--locus",
                "in-process",
                "--omnibase-path",
                str(_workspace(tmp_path / "workspace")),
                "--state-root",
                str(tmp_path / "state"),
            ]
        )
        assert result.exit_code != 0
        output = str(result.output)
        assert "--lane" in output, "the refusal must name the missing selection"
        assert "dev" in output, "the refusal must list the declared lanes"
        assert AMBIENT_STABILITY_BROKER not in output, (
            "the ambient value must play no part in the resolution, not even "
            "as a suggested remedy"
        )

    def test_both_broker_flags_together_are_refused(self, tmp_path: Path) -> None:
        """Two addresses is not a selection."""
        result = _invoke(
            [
                "document the router",
                "--task-type",
                "document",
                "--bus",
                "kafka",
                "--locus",
                "in-process",
                "--lane",
                "dev",
                "--kafka-bootstrap",
                "redpanda:9092",
                "--omnibase-path",
                str(_workspace(tmp_path / "workspace")),
                "--state-root",
                str(tmp_path / "state"),
            ]
        )
        assert result.exit_code != 0
        assert "both" in str(result.output)

    def test_an_undeclared_lane_is_refused_by_name(self, tmp_path: Path) -> None:
        result = _invoke(
            [
                "document the router",
                "--task-type",
                "document",
                "--bus",
                "kafka",
                "--locus",
                "in-process",
                "--lane",
                "judge",
                "--omnibase-path",
                str(_workspace(tmp_path / "workspace")),
                "--state-root",
                str(tmp_path / "state"),
            ]
        )
        assert result.exit_code != 0
        output = str(result.output)
        assert "judge" in output
        assert "dev" in output


class TestTheFlagIsDiscoverable:
    def test_help_documents_the_lane_flag_and_not_the_env_var(self) -> None:
        """A flag nobody can discover is a flag nobody uses."""
        help_text = " ".join(str(_invoke(["--help"]).output).split())
        assert "--lane" in help_text
        assert "Omit to resolve from KAFKA_BOOTSTRAP_SERVERS" not in help_text, (
            "the help text must not keep advertising the path this ticket removed"
        )
