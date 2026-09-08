# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the LLM endpoint env contract guard.

OMN-16442 (2026-08-28) retired the ``coder-fast-4090`` slot after its RTX 4090 was
physically removed from .201, and RELEASED ``LLM_CODER_FAST_URL`` from it
(``url_env_var: null`` in ``contracts/llm_endpoints.yaml``) on the rule that a
disabled slot must not own a runtime env var.

The variable itself did NOT go away, and was never meant to: it is still a
fail-closed requirement of ``docker/docker-compose.infra.yml`` (the PluginLlm
activation block) and ``config/shared_key_registry.yaml`` records the deliberate
decision to keep the KEY seeded, aliased onto the surviving coder slot, until
retiring it across compose + the adapters + transport_config_map + service_kernel
is done as its own change.

So a correct host DOES carry ``LLM_CODER_FAST_URL`` set, and the guard must not
fail there. The follow-up to the OMN-16442 contract change edited THESE FIXTURES
to drop the variable instead of the code that asserted it, which made the suite
green against an input that no longer resembled any real host and left the guard
failing every dev deploy for eleven days. The header comment removed here called
that guard behaviour correct; it was not.

The fix is that the checked-variable set is now DERIVED from the contract's own
``url_env_var`` values, so a released slot cannot leave an orphaned assertion
behind. ``test_accepts_host_shaped_env_with_released_variable_set`` is the
regression: it reinstates a host-shaped fixture carrying the released variable.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "check_llm_endpoint_env_contract.py"
_CONTRACT = _REPO_ROOT / "contracts" / "llm_endpoints.yaml"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("_llm_endpoint_env_guard", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_check(env_file: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python", str(_SCRIPT), "--env-file", str(env_file), *extra],
        cwd=_REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )


@pytest.mark.unit
def test_rejects_disabled_embedding_endpoint(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_CODER_URL=http://192.168.86.201:8000",
                "LLM_EMBEDDING_URL=http://192.168.86.200:8100",
                "LLM_DEEPSEEK_R1_URL=http://192.168.86.200:8101",
            ]
        )
    )

    result = _run_check(env_file)

    assert result.returncode == 1
    assert "LLM_EMBEDDING_URL=http://192.168.86.200:8100" in result.stderr
    assert "status='disabled'" in result.stderr


@pytest.mark.unit
def test_accepts_running_embedding_endpoint(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_CODER_URL=http://192.168.86.201:8000",
                "LLM_EMBEDDING_URL=http://192.168.86.201:8002",
                "LLM_DEEPSEEK_R1_URL=http://192.168.86.200:8101",
            ]
        )
    )

    result = _run_check(env_file)

    assert result.returncode == 0
    assert "passed" in result.stdout


@pytest.mark.unit
def test_accepts_host_shaped_env_with_released_variable_set(tmp_path: Path) -> None:
    """A host that still seeds a RELEASED variable must pass.

    This fixture is shaped like the real ``.201`` operator env file: the three
    variables the contract still assigns, at their canonical running URLs, PLUS
    the two variables OMN-16442 released from now-disabled slots but which the
    host must keep seeded because compose declares them fail-closed. Values are
    dummies -- no real endpoint value is asserted for the released names, and
    none is read.

    Before the derivation fix this returned exit 1 with
    "LLM_CODER_FAST_URL is not assigned to a canonical endpoint slot", which is
    what failed every deploy-agent rebuild on the dev lane.
    """
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_CODER_URL=http://192.168.86.201:8000",
                "LLM_EMBEDDING_URL=http://192.168.86.201:8002",
                "LLM_DEEPSEEK_R1_URL=http://192.168.86.200:8101",
                # Released by OMN-16442 (slot coder-fast-4090, hardware removed);
                # kept seeded per config/shared_key_registry.yaml because
                # docker/docker-compose.infra.yml requires it fail-closed.
                "LLM_CODER_FAST_URL=http://127.0.0.1:9/dummy-not-a-slot",
                # Released by OMN-16442 (slot reasoning-moe-35b, no listener).
                "LLM_QWEN3_NEXT_URL=http://127.0.0.1:9/dummy-not-a-slot",
            ]
        )
    )

    result = _run_check(env_file)

    assert result.returncode == 0, result.stderr
    assert "LLM_CODER_FAST_URL" not in result.stderr
    assert "passed" in result.stdout


@pytest.mark.unit
def test_derived_env_var_set_excludes_released_variables() -> None:
    """The checked set is exactly the contract's assigned ``url_env_var`` values."""
    module = _load_module()
    endpoints = module._load_endpoints(_CONTRACT)

    derived = module.contract_env_vars(endpoints)

    assert derived == (
        "LLM_CODER_URL",
        "LLM_EMBEDDING_URL",
        "LLM_DEEPSEEK_R1_URL",
    )
    assert "LLM_CODER_FAST_URL" not in derived
    assert "LLM_QWEN3_NEXT_URL" not in derived


@pytest.mark.unit
def test_derived_set_drops_a_variable_released_from_its_slot() -> None:
    """Releasing a slot's variable removes it from the checked set, mechanically.

    Positive control first (the variable IS derived while the slot declares it),
    then the release. This is the property that makes an orphaned assertion
    impossible, independent of the live contract's current contents.
    """
    module = _load_module()
    assigned: list[dict[str, Any]] = [
        {
            "slot_id": "s1",
            "url_env_var": "LLM_EXAMPLE_URL",
            "endpoint_url": "http://127.0.0.1:9",
            "status": "running",
        }
    ]
    assert module.contract_env_vars(assigned) == ("LLM_EXAMPLE_URL",)

    released = [{**assigned[0], "url_env_var": None, "status": "disabled"}]
    assert module.contract_env_vars(released) == ()


@pytest.mark.unit
def test_explicit_env_var_still_reports_an_unassigned_variable(tmp_path: Path) -> None:
    """``--env-var`` overrides the derived set; its diagnostics are unchanged."""
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_CODER_FAST_URL=http://192.168.86.201:8001")

    result = _run_check(env_file, "--env-var", "LLM_CODER_FAST_URL")

    assert result.returncode == 1
    assert "LLM_CODER_FAST_URL is not assigned to a canonical endpoint slot" in (
        result.stderr
    )
