# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""End-to-end CLI coverage: a default delegation on a registry workspace (OMN-19193).

The unit module drives the resolver with a synthetic tier-1 file. This one goes
through ``click`` with the REAL checked-in
``config/workspace/runtime/runtime_config.yaml`` of this repository, found the
only way the product finds it: a workspace root bound through ``$OMNIBASE_PATH``
(the variable ``--omni-home`` reads, which the sanctioned wrapper exports),
with no ``--bus``, no ``--lane`` and no ``--omni-home`` on the command line.

The operator's shell is reproduced around it: ``ONEX_EVENT_BUS_TYPE=kafka``,
which holds no tier since OMN-17304, and ``KAFKA_BOOTSTRAP_SERVERS`` naming
the governed stability-test lane. Before the fix, that exact invocation
resolved the shipped tier-0 in-memory bus, so the delegation's evidence never
reached the shared projection.

Dispatch itself needs a co-installed omnimarket and a live model endpoint,
neither of which belongs in this gate, so the receipt-mode dispatch is captured
rather than run -- the same stand-in the lane-identity integration test uses.
Everything under test happens in front of dispatch: the env binding, the
workspace tier, the lane taken from that config, and the declaration read.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import delegate_command
from omnibase_infra.cli.delegate_lane import LANE_DECLARATION_RELATIVE_PATH
from omnibase_infra.cli.store_lane_credential import StoreLaneCredential
from omnibase_infra.runtime.service_kernel import (
    WORKSPACE_RUNTIME_CONTRACTS_RELATIVE_PATH,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: The ambient value on the operator host: the governed stability-test lane.
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

STAND_IN_TASK_CLASS_CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "delegation"
    / "omn18305"
    / "task_class_contracts_vocabulary.yaml"
)


def _workspace(root: Path, *, with_repo_config: bool) -> Path:
    """A workspace root: the lane declaration, and THIS repo as omnibase_infra.

    ``omnibase_infra`` is a symlink to the repository under test, so the tier-1
    file the CLI reads is the committed one, byte for byte.
    """
    declaration = root / LANE_DECLARATION_RELATIVE_PATH
    declaration.parent.mkdir(parents=True, exist_ok=True)
    declaration.write_text(_DECLARATION, encoding="utf-8")
    if with_repo_config:
        (root / WORKSPACE_RUNTIME_CONTRACTS_RELATIVE_PATH.parts[0]).symlink_to(
            _REPO_ROOT, target_is_directory=True
        )
    return root


@pytest.fixture
def captured_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    captured: dict[str, object] = {}

    def _fake_run_receipt_mode(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setenv("ONEX_EVENT_BUS_TYPE", "kafka")
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", AMBIENT_STABILITY_BROKER)
    monkeypatch.delenv("ONEX_CONTRACTS_DIR", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
    monkeypatch.setattr(
        cli_delegate,
        "resolve_task_class_contract_path",
        lambda: STAND_IN_TASK_CLASS_CONTRACT,
    )
    monkeypatch.setattr(
        cli_delegate,
        "_resolve_packaged_contract",
        lambda _name: tmp_path / "contract.yaml",
    )
    monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_run_receipt_mode)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    StoreLaneCredential(onex_home=home / ".onex").save(
        lane="dev",
        sasl_username="dev-cli-under-test",
        sasl_password="not-a-real-secret",
    )
    return captured


def _delegate(tmp_path: Path, root: Path) -> object:
    # No --bus, no --lane, no --omni-home: the default invocation.
    return CliRunner(env={"OMNIBASE_PATH": str(root)}).invoke(
        delegate_command,
        [
            "document the router",
            "--task-type",
            "document",
            # Pinned in-process: the subject is the resolved transport and its
            # address, not the live-consumer gate, which needs a broker.
            "--locus",
            "in-process",
            "--state-root",
            str(tmp_path / "state"),
        ],
        catch_exceptions=False,
    )


class TestTheDefaultInvocationOnAWorkspace:
    def test_the_committed_tier1_config_puts_it_on_the_declared_dev_lane(
        self, tmp_path: Path, captured_dispatch: dict[str, object]
    ) -> None:
        """RED before the fix: this resolved the in-memory bus."""
        root = _workspace(tmp_path / "workspace", with_repo_config=True)
        result = _delegate(tmp_path, root)
        assert result.exit_code == 0, result.output
        assert captured_dispatch["backend_overrides"] == {
            "event_bus": "kafka",
            "kafka_bootstrap": DECLARED_DEV_BROKER,
        }

    def test_positive_control_without_the_config_it_is_still_tier0(
        self, tmp_path: Path, captured_dispatch: dict[str, object]
    ) -> None:
        """The same invocation on a root with no tier-1 file is unchanged."""
        root = _workspace(tmp_path / "workspace", with_repo_config=False)
        result = _delegate(tmp_path, root)
        assert result.exit_code == 0, result.output
        assert captured_dispatch["backend_overrides"] == {"event_bus": "inmemory"}
