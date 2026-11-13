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

OMN-24017: the canonical LLM_EMBEDDING_URL slot (embeddings-201) no longer
carries an address in the public contract; its address is read from the
``llm_endpoints`` overlay (``ONEX_SKILL_OVERLAY_ROOTS``) at run time. The guard
resolves flagged slots before validating, so every check here builds a tmp
overlay root with a synthetic documentation address (192.0.2.0/24) and passes
it through ``ONEX_SKILL_OVERLAY_ROOTS``; a store naming the overlay's address
passes, and a store naming the removed address is refused. Added fixture lines
use only the synthetic address.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "check_llm_endpoint_env_contract.py"
_CONTRACT = _REPO_ROOT / "contracts" / "llm_endpoints.yaml"

# Synthetic documentation address (TEST-NET-1 range): never a real lab host.
_SYNTHETIC_EMBEDDING_URL = "http://192.0.2.10:8100"
_REMOVED_EMBEDDING_URL = "http://192.0.2.10:8002"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("_llm_endpoint_env_guard", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_overlay(root: Path, endpoint_urls: dict[str, str]) -> Path:
    """Write a synthetic ``llm_endpoints/overlay.yaml`` under a tmp root."""
    path = root / "llm_endpoints" / "overlay.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump({"schema_version": 1, "endpoint_urls": endpoint_urls}),
        encoding="utf-8",
    )
    return root


def _run_check(
    env_file: Path,
    *extra: str,
    overlay_root: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    for var in (
        "ONEX_SKILL_OVERLAY_ROOTS",
        "DEPLOY_AGENT_OVERLAY_HOME_CLONE",
        "ONEX_WORKSPACE_CONFIG_ROOT",
    ):
        env.pop(var, None)
    if overlay_root is not None:
        env["ONEX_SKILL_OVERLAY_ROOTS"] = str(overlay_root)
    return subprocess.run(
        ["python", str(_SCRIPT), "--env-file", str(env_file), *extra],
        cwd=_REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )


@pytest.mark.unit
def test_rejects_disabled_embedding_endpoint(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_CODER_URL=http://192.168.86.201:8000",
                "LLM_EMBEDDING_URL=http://192.168.86.200:8100",
            ]
        )
    )
    overlay_root = _write_overlay(
        tmp_path / "overlay-root", {"embeddings-201": _SYNTHETIC_EMBEDDING_URL}
    )

    result = _run_check(env_file, overlay_root=overlay_root)

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
                f"LLM_EMBEDDING_URL={_SYNTHETIC_EMBEDDING_URL}",
            ]
        )
    )
    overlay_root = _write_overlay(
        tmp_path / "overlay-root", {"embeddings-201": _SYNTHETIC_EMBEDDING_URL}
    )

    result = _run_check(env_file, overlay_root=overlay_root)

    assert result.returncode == 0, result.stderr
    assert "passed" in result.stdout


@pytest.mark.unit
def test_accepts_host_shaped_env_with_released_variable_set(tmp_path: Path) -> None:
    """A host that still seeds a RELEASED variable must pass.

    This fixture is shaped like the real lab operator env file: the two
    variables the contract still assigns, at their canonical running URLs, PLUS
    the variables released from now-disabled slots but which the
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
                f"LLM_EMBEDDING_URL={_SYNTHETIC_EMBEDDING_URL}",
                "LLM_DEEPSEEK_R1_URL=http://llm.test:8000",
                # Released by OMN-16442 (slot coder-fast-4090, hardware removed);
                # kept seeded per config/shared_key_registry.yaml because
                # docker/docker-compose.infra.yml requires it fail-closed.
                "LLM_CODER_FAST_URL=http://127.0.0.1:9/dummy-not-a-slot",
                # Released by OMN-16442 (slot reasoning-moe-35b, no listener).
                "LLM_QWEN3_NEXT_URL=http://127.0.0.1:9/dummy-not-a-slot",
            ]
        )
    )
    overlay_root = _write_overlay(
        tmp_path / "overlay-root", {"embeddings-201": _SYNTHETIC_EMBEDDING_URL}
    )

    result = _run_check(env_file, overlay_root=overlay_root)

    assert result.returncode == 0, result.stderr
    assert "LLM_CODER_FAST_URL" not in result.stderr
    assert "passed" in result.stdout


@pytest.mark.unit
def test_store_naming_the_embedding_overlay_address_passes(tmp_path: Path) -> None:
    """OMN-24017: a settings store naming the overlay's address passes."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                f"LLM_EMBEDDING_URL={_SYNTHETIC_EMBEDDING_URL}",
            ]
        )
    )
    overlay_root = _write_overlay(
        tmp_path / "overlay-root", {"embeddings-201": _SYNTHETIC_EMBEDDING_URL}
    )

    result = _run_check(env_file, overlay_root=overlay_root)

    assert result.returncode == 0, result.stderr
    assert "passed" in result.stdout


@pytest.mark.unit
def test_store_naming_the_removed_embedding_address_is_refused(tmp_path: Path) -> None:
    """OMN-24017: the removed address is refused on the canonical slot."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                f"LLM_EMBEDDING_URL={_REMOVED_EMBEDDING_URL}",
            ]
        )
    )
    overlay_root = _write_overlay(
        tmp_path / "overlay-root", {"embeddings-201": _SYNTHETIC_EMBEDDING_URL}
    )

    result = _run_check(env_file, overlay_root=overlay_root)

    assert result.returncode == 1
    assert "does not match canonical running slot" in result.stderr
    assert "'embeddings-201'" in result.stderr


@pytest.mark.unit
def test_guard_refuses_embedding_slot_when_no_overlay_root(tmp_path: Path) -> None:
    """OMN-24017: with no overlay root, the flagged slot cannot resolve."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                f"LLM_EMBEDDING_URL={_SYNTHETIC_EMBEDDING_URL}",
            ]
        )
    )

    result = _run_check(env_file)

    assert result.returncode == 1
    assert "'embeddings-201'" in result.stderr
    assert "no llm_endpoints overlay file" in result.stderr


@pytest.mark.unit
def test_guard_refuses_when_embedding_slot_missing_from_overlay(
    tmp_path: Path,
) -> None:
    """OMN-24017: an overlay without the slot is a refusal, not a fallback."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                f"LLM_EMBEDDING_URL={_SYNTHETIC_EMBEDDING_URL}",
            ]
        )
    )
    overlay_root = _write_overlay(
        tmp_path / "overlay-root", {"other-slot": _SYNTHETIC_EMBEDDING_URL}
    )

    result = _run_check(env_file, overlay_root=overlay_root)

    assert result.returncode == 1
    assert "'embeddings-201'" in result.stderr
    assert "not in the llm_endpoints overlay" in result.stderr


@pytest.mark.unit
def test_derived_env_var_set_excludes_released_variables() -> None:
    """The checked set is exactly the contract's assigned ``url_env_var`` values."""
    module = _load_module()
    endpoints = module.load_endpoints(_CONTRACT)

    derived = module.contract_env_vars(endpoints)

    assert derived == (
        "LLM_CODER_URL",
        "LLM_EMBEDDING_URL",
    )
    assert "LLM_CODER_FAST_URL" not in derived
    assert "LLM_QWEN3_NEXT_URL" not in derived
    assert "LLM_DEEPSEEK_R1_URL" not in derived


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
def test_explicit_env_var_still_reports_an_unassigned_variable(
    tmp_path: Path,
) -> None:
    """``--env-var`` overrides the derived set; its diagnostics are unchanged."""
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_CODER_FAST_URL=http://192.168.86.201:8001")
    overlay_root = _write_overlay(
        tmp_path / "overlay-root", {"embeddings-201": _SYNTHETIC_EMBEDDING_URL}
    )

    result = _run_check(
        env_file, "--env-var", "LLM_CODER_FAST_URL", overlay_root=overlay_root
    )

    assert result.returncode == 1
    assert "LLM_CODER_FAST_URL is not assigned to a canonical endpoint slot" in (
        result.stderr
    )


@pytest.mark.unit
def test_skip_env_var_leaves_the_stores_superseded_value_unchecked(
    tmp_path: Path,
) -> None:
    """OMN-24025: a name the caller renders itself is not checked in the store.

    The dev lane's deploy agent renders ``LLM_EMBEDDING_URL`` from the overlay over
    the store, so the store's stale value never reaches compose. The control is the
    same file without the option, which the guard refuses.
    """
    env_file = tmp_path / ".env"
    env_file.write_text(f"LLM_EMBEDDING_URL={_REMOVED_EMBEDDING_URL}\n")
    overlay_root = _write_overlay(
        tmp_path / "overlay-root", {"embeddings-201": _SYNTHETIC_EMBEDDING_URL}
    )

    control = _run_check(env_file, overlay_root=overlay_root)
    skipped = _run_check(
        env_file, "--skip-env-var", "LLM_EMBEDDING_URL", overlay_root=overlay_root
    )

    assert control.returncode == 1
    assert "LLM_EMBEDDING_URL" in control.stderr
    assert skipped.returncode == 0, skipped.stderr


@pytest.mark.unit
def test_skip_env_var_wins_over_an_explicit_env_var(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"LLM_EMBEDDING_URL={_REMOVED_EMBEDDING_URL}\n")
    overlay_root = _write_overlay(
        tmp_path / "overlay-root", {"embeddings-201": _SYNTHETIC_EMBEDDING_URL}
    )

    result = _run_check(
        env_file,
        "--env-var",
        "LLM_EMBEDDING_URL",
        "--skip-env-var",
        "LLM_EMBEDDING_URL",
        overlay_root=overlay_root,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.unit
def test_skip_env_var_still_checks_every_other_name(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                f"LLM_EMBEDDING_URL={_REMOVED_EMBEDDING_URL}",
                "LLM_CODER_URL=http://192.0.2.99:8000",
            ]
        )
    )
    overlay_root = _write_overlay(
        tmp_path / "overlay-root", {"embeddings-201": _SYNTHETIC_EMBEDDING_URL}
    )

    result = _run_check(
        env_file, "--skip-env-var", "LLM_EMBEDDING_URL", overlay_root=overlay_root
    )

    assert result.returncode == 1
    assert "LLM_CODER_URL" in result.stderr
    assert "LLM_EMBEDDING_URL" not in result.stderr
