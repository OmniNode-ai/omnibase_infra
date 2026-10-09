# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""End-to-end CLI coverage: a fresh developer workspace delegates (OMN-20700).

Through ``click``, with public clones only, no private sibling repository, and
``ONEX_WORKSPACE_CONFIG_ROOT`` unset. The default invocation binds the workspace
through ``$OMNIBASE_PATH`` and delegates on the tier-1 runtime config that phase
2 of the public ``omniclaude`` onboarding script, ``lab-onboarding.sh``, wrote
at ``.onex/workspace-config/config/onex/runtime/runtime_config.yaml``. Before
OMN-20700 the default invocation was refused naming ``../omnibase_internal``.

Dispatch itself needs a co-installed omnimarket and a live model endpoint,
neither of which belongs in this gate, so the receipt-mode dispatch is captured
rather than run -- the same stand-in the workspace transport integration test
uses. A workspace with no config anywhere is refused naming the public fix.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import delegate_command

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def isolated_config_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ONEX_WORKSPACE_CONFIG_ROOT", raising=False)
    monkeypatch.delenv("ONEX_CONTRACTS_DIR", raising=False)


_DEVELOPER_RUNTIME_CONFIG = """# Written by omninode-dev-setup.
description: "Developer workspace tier-1 runtime configuration: in-memory bus, local profile"
input_topic: "requests"
output_topic: "responses"
group_id: "onex-runtime"
event_bus:
  type: "inmemory"
  profile: "local"
  environment: "local"
  max_history: 1000
  circuit_breaker_threshold: 5
"""


def _workspace(root: Path, *, with_config: bool) -> Path:
    """A public developer workspace with, optionally, its onboarding config."""
    root.mkdir(parents=True)
    if with_config:
        config = (
            root
            / ".onex"
            / "workspace-config"
            / "config"
            / "onex"
            / "runtime"
            / "runtime_config.yaml"
        )
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(_DEVELOPER_RUNTIME_CONFIG, encoding="utf-8")
    return root


@pytest.fixture
def captured_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    captured: dict[str, object] = {}

    def _fake_run_receipt_mode(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

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
    return captured


def _delegate(tmp_path: Path, root: Path) -> object:
    # No --bus, --lane, --locus or --omnibase-path: the default invocation.
    return CliRunner(env={"OMNIBASE_PATH": str(root)}).invoke(
        delegate_command,
        [
            "Reply with exactly one word: hello",
            "--task-type",
            "document",
            "--state-root",
            str(tmp_path / "state"),
        ],
        catch_exceptions=False,
    )


def test_a_fresh_developer_workspace_delegates_on_the_config_onboarding_wrote(
    tmp_path: Path, captured_dispatch: dict[str, object]
) -> None:
    """RED before OMN-20700: the default invocation required a private repo."""
    root = _workspace(tmp_path / "code" / "omni", with_config=True)
    result = _delegate(tmp_path, root)
    assert result.exit_code == 0, result.output
    assert captured_dispatch
    assert captured_dispatch["backend_overrides"] == {"event_bus": "inmemory"}
    assert "omnibase_internal" not in result.output


def test_no_config_anywhere_is_refused_naming_the_public_fix(
    tmp_path: Path, captured_dispatch: dict[str, object]
) -> None:
    """A fresh workspace without onboarding config gets an actionable refusal."""
    root = _workspace(tmp_path / "code" / "omni", with_config=False)
    result = _delegate(tmp_path, root)
    assert result.exit_code != 0
    output = " ".join(str(result.output).split())
    assert "ONEX_WORKSPACE_CONFIG_ROOT" in output
    assert "lab-onboarding.sh" in output
    assert ".onex/workspace-config" in output
    assert "--bus inmemory" in output
    assert captured_dispatch == {}, "nothing may be dispatched on a refusal"
