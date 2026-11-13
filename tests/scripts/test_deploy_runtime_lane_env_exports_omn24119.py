# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression guard: onex-runtime-deploy must let the lane overlay exports win.

OMN-24119: the refresh scripts render the lane's overlay exports (the embedding
address and the lane overlay directory) into LANE_ENV_EXPORTS and hand them to
onex-runtime-deploy via ONEX_DEPLOY_LANE_ENV_EXPORTS. The handler sources the
operator env file with `set -a` (which would overwrite any value a caller
exported) and then, right after that, sources the lane exports document, so the
values a refresh script rendered from the overlays win over a person's older
operator env values.

These tests pin that ordering and its failure modes: the rendered address must
win over the operator file, the operator file value must stand when no exports
are passed, a broken exports document must stop the handler before any build,
and both refresh scripts must hand their exports to the handler after they
source them and before they invoke the deploy runtime.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

DEPLOY_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "omnibase_infra"
    / "handlers"
    / "handler_runtime_deploy.sh"
)

REFRESH_SCRIPTS = [
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "runtime_build"
    / "refresh_dev_lane.sh",
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "runtime_build"
    / "refresh_stability_lane.sh",
]

STALE_ADDRESS = "http://192.0.2.50:8002"
OVERLAY_ADDRESS = "http://192.0.2.50:8100"


def _run_prelude(
    operator_env: Path, exports: str | None
) -> subprocess.CompletedProcess[str]:
    """Run the handler prelude (through the `unset` of the operator copies) in a subprocess.

    The prelude is the handler text from the start through the line that unsets
    the operator copies, plus one extra final line that prints the
    effective LLM_EMBEDDING_URL so the test can read which value won.
    """
    text = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    marker = "unset OPERATOR_OMNI_HOME OPERATOR_HEALTH_CHECK_URL"
    idx = text.index(marker)
    end = text.index("\n", idx)
    prelude = (
        text[: end + 1] + "printf 'LLM_EMBEDDING_URL=%s\\n' \"$LLM_EMBEDDING_URL\"\n"
    )
    prelude_path = operator_env.parent / "prelude.sh"
    prelude_path.write_text(prelude, encoding="utf-8")

    env = {
        **os.environ,
        "ONEX_DEPLOY_REPOSITORY_ROOT": str(Path(__file__).resolve().parents[2]),
    }
    env["OMNIBASE_OPERATOR_ENV_FILE"] = str(operator_env)
    if exports is not None:
        env["ONEX_DEPLOY_LANE_ENV_EXPORTS"] = exports
    else:
        env.pop("ONEX_DEPLOY_LANE_ENV_EXPORTS", None)

    return subprocess.run(
        ["bash", str(prelude_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env=env,
    )


@pytest.mark.unit
def test_seam_the_handler_prelude_keeps_the_rendered_address_over_the_operator_file(
    tmp_path: Path,
) -> None:
    """The lane overlay exports win over the operator env file's value."""
    operator_env = tmp_path / "operator.env"
    operator_env.write_text(f"LLM_EMBEDDING_URL={STALE_ADDRESS}\n", encoding="utf-8")

    exports = f"export LLM_EMBEDDING_URL='{OVERLAY_ADDRESS}'\n"
    result = _run_prelude(operator_env, exports)

    assert result.returncode == 0, f"prelude failed: {result.stderr}"
    last_line = result.stdout.strip().splitlines()[-1]
    assert last_line == f"LLM_EMBEDDING_URL={OVERLAY_ADDRESS}", (
        f"expected the overlay address to win, got: {last_line!r}"
    )


@pytest.mark.unit
def test_control_without_exports_the_operator_file_value_stands(tmp_path: Path) -> None:
    """Without ONEX_DEPLOY_LANE_ENV_EXPORTS the operator env file value stands."""
    operator_env = tmp_path / "operator.env"
    operator_env.write_text(f"LLM_EMBEDDING_URL={STALE_ADDRESS}\n", encoding="utf-8")

    result = _run_prelude(operator_env, None)

    assert result.returncode == 0, f"prelude failed: {result.stderr}"
    last_line = result.stdout.strip().splitlines()[-1]
    assert last_line == f"LLM_EMBEDDING_URL={STALE_ADDRESS}", (
        f"expected the operator file value to stand, got: {last_line!r}"
    )


@pytest.mark.unit
def test_error_an_exports_document_that_cannot_run_stops_the_handler(
    tmp_path: Path,
) -> None:
    """A broken exports document must stop the handler before any build."""
    operator_env = tmp_path / "operator.env"
    operator_env.write_text(f"LLM_EMBEDDING_URL={STALE_ADDRESS}\n", encoding="utf-8")

    result = _run_prelude(operator_env, "export 'not a name'=1\n")

    assert result.returncode != 0, (
        "a broken ONEX_DEPLOY_LANE_ENV_EXPORTS document must stop the handler "
        "(non-zero returncode) so it never reaches a build"
    )


@pytest.mark.unit
def test_the_dev_refresh_script_hands_its_exports_to_the_handler() -> None:
    """The dev script exports its lane exports to the handler, after sourcing them."""
    text = REFRESH_SCRIPTS[0].read_text(encoding="utf-8")
    export_line = 'export ONEX_DEPLOY_LANE_ENV_EXPORTS="${LANE_ENV_EXPORTS}"'
    source_line = "source <(printf '%s' \"${LANE_ENV_EXPORTS}\")"
    deploy_line = '"${DEPLOY_RUNTIME}" --repository-root'
    assert text.index(source_line) < text.index(export_line) < text.index(deploy_line)


@pytest.mark.unit
def test_the_stability_refresh_script_sources_the_exports_its_program_hands_over() -> (
    None
):
    """The stability script sources the program's output, which carries the handler's line."""
    text = REFRESH_SCRIPTS[1].read_text(encoding="utf-8")
    source_line = "source <(printf '%s' \"${LANE_ENV_EXPORTS}\")"
    deploy_line = '"${DEPLOY_RUNTIME}" --repository-root'
    assert text.index(source_line) < text.index(deploy_line)
    module = (
        DEPLOY_SCRIPT.parents[1]
        / "nodes"
        / "node_setup_local_provision_effect"
        / "runtime_llm_endpoint_env.py"
    )
    assert "export ONEX_DEPLOY_LANE_ENV_EXPORTS=" in module.read_text(encoding="utf-8")
