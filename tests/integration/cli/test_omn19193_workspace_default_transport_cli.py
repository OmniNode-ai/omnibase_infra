# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""End-to-end CLI coverage: a default delegation on a registry workspace (OMN-19193).

Through ``click``, with the workspace root bound the only way the product binds
it: ``$OMNIBASE_PATH`` (the variable ``--omnibase-path`` reads, which the sanctioned
wrapper exports), and no ``--bus``, ``--lane`` or ``--omnibase-path`` on the
command line. The workspace carries its OWN tier-1 runtime config at
``config/onex/runtime/runtime_config.yaml``; this package ships no lab values
(OMN-19184), so the file here is written by the test exactly as a workspace
declares it.

The operator's shell is reproduced around it: ``ONEX_EVENT_BUS_TYPE=kafka``,
which holds no tier since OMN-17304, and ``KAFKA_BOOTSTRAP_SERVERS`` naming
the governed stability-test lane. Before the fix that invocation resolved the
shipped tier-0 in-memory bus, so the delegation's evidence never reached the
shared projection. A bound root that declares no config is now refused.

Dispatch itself needs a co-installed omnimarket and a live model endpoint,
neither of which belongs in this gate, so the receipt-mode dispatch is captured
rather than run -- the same stand-in the lane-identity integration test uses.
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


_WORKSPACE_RUNTIME_CONFIG = """
event_bus:
  type: "kafka"
  profile: "local"
  lane: "dev"
"""


def _workspace(root: Path, *, with_config: bool) -> Path:
    """A workspace root: the lane declaration and, optionally, its tier-1 config."""
    declaration = root / LANE_DECLARATION_RELATIVE_PATH
    declaration.parent.mkdir(parents=True, exist_ok=True)
    declaration.write_text(_DECLARATION, encoding="utf-8")
    if with_config:
        config = (
            root
            / WORKSPACE_RUNTIME_CONTRACTS_RELATIVE_PATH
            / "runtime"
            / "runtime_config.yaml"
        )
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(_WORKSPACE_RUNTIME_CONFIG, encoding="utf-8")
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
    # No --bus, no --lane, no --omnibase-path: the default invocation.
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
    def test_the_workspace_tier1_config_puts_it_on_the_declared_dev_lane(
        self, tmp_path: Path, captured_dispatch: dict[str, object]
    ) -> None:
        """RED before the fix: this resolved the in-memory bus."""
        root = _workspace(tmp_path / "workspace", with_config=True)
        result = _delegate(tmp_path, root)
        assert result.exit_code == 0, result.output
        assert captured_dispatch["backend_overrides"] == {
            "event_bus": "kafka",
            "kafka_bootstrap": DECLARED_DEV_BROKER,
        }

    def test_a_bound_root_with_no_config_is_refused_loudly(
        self, tmp_path: Path, captured_dispatch: dict[str, object]
    ) -> None:
        """Never the silent tier-0 fall that stranded the evidence."""
        root = _workspace(tmp_path / "workspace", with_config=False)
        result = _delegate(tmp_path, root)
        assert result.exit_code != 0
        output = " ".join(str(result.output).split())
        assert "config/onex/runtime/runtime_config.yaml" in output
        assert "--bus inmemory" in output
        assert captured_dispatch == {}, "nothing may be dispatched on a refusal"
