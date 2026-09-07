# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The one-interpreter boot preflight produces the same artifacts (OMN-17372).

Collapsing four cold interpreters into one is only safe if every artifact and
every exported environment value the four steps produced is unchanged. These
tests prove that where it is provable in a unit test:

* the secret-resolver config rendered IN-PROCESS is **byte-identical** to the
  one the old out-of-process ``python -m
  omnibase_infra.runtime.render_secret_resolver_config`` produces, compared as
  raw bytes with a positive control that the comparison can actually fail;
* the render steps are gated on exactly the environment variables the shell
  gated them on, and a non-zero render still aborts boot (the shell had
  ``set -e``);
* the handoff calls the packaged kernel entry point **in this process** for the
  image's own ``CMD`` and ``execvp``s anything else, which is what ``exec "$@"``
  did.

The Bifrost renderer is not byte-diffed here: it resolves its base contract from
the packaged ``omnimarket`` distribution, which is not installed in the unit
environment. Its identity is structural instead — ``render_bifrost_contract``
calls the module's own ``main()`` with no arguments and no environment
rewriting, asserted below — and it is exercised end to end by the compose boot
smoke gate.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_infra.runtime import entrypoint_preflight

pytestmark = [pytest.mark.unit]

#: A minimal but real secret-resolver config, supplied inline so the render needs
#: no file fixture and no secret store.
_RESOLVER_CONFIG: dict[str, object] = {
    "mappings": [
        {
            "logical_name": "omn17372_probe_secret",
            "source": {
                "source_type": "env",
                "source_path": "OMN17372_PROBE_SECRET",
            },
        }
    ]
}


def _render_out_of_process(target: Path, config_json: str) -> int:
    """Render the way the entrypoint used to: a fresh cold interpreter."""
    result = subprocess.run(
        [sys.executable, "-m", "omnibase_infra.runtime.render_secret_resolver_config"],
        env={
            "PATH": "/usr/bin:/bin",
            "ONEX_SECRET_RESOLVER_CONFIG_PATH": str(target),
            "ONEX_SECRET_RESOLVER_CONFIG_JSON": config_json,
        },
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.returncode


def test_resolver_artifact_is_byte_identical_in_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The rendered secret-resolver config must not change by a single byte."""
    config_json = json.dumps(_RESOLVER_CONFIG)

    out_of_process_target = tmp_path / "subprocess" / "secret_resolver.yaml"
    out_of_process_target.parent.mkdir()
    _render_out_of_process(out_of_process_target, config_json)

    in_process_target = tmp_path / "inprocess" / "secret_resolver.yaml"
    in_process_target.parent.mkdir()
    monkeypatch.setenv("ONEX_SECRET_RESOLVER_CONFIG_PATH", str(in_process_target))
    monkeypatch.setenv("ONEX_SECRET_RESOLVER_CONFIG_JSON", config_json)

    rc = entrypoint_preflight.render_resolver_config(
        {
            "ONEX_SECRET_RESOLVER_CONFIG_PATH": str(in_process_target),
            "ONEX_SECRET_RESOLVER_CONFIG_JSON": config_json,
        }
    )

    assert rc == 0
    before = out_of_process_target.read_bytes()
    after = in_process_target.read_bytes()
    assert before, "positive control: the out-of-process render produced no bytes"
    assert after == before, (
        "the in-process render produced different bytes than the cold "
        f"interpreter did:\n---subprocess---\n{before!r}\n---in-process---\n{after!r}"
    )


def test_control_byte_comparison_can_fail(tmp_path: Path) -> None:
    """POSITIVE CONTROL: the comparison above is not vacuously true.

    A different input must produce different bytes; otherwise a renderer that
    wrote a constant would satisfy the identity test.
    """
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    _render_out_of_process(a, json.dumps(_RESOLVER_CONFIG))
    other = {
        "mappings": [
            {
                "logical_name": "omn17372_probe_secret_other",
                "source": {
                    "source_type": "env",
                    "source_path": "OMN17372_PROBE_SECRET_OTHER",
                },
            }
        ]
    }
    _render_out_of_process(b, json.dumps(other))

    assert a.read_bytes() != b.read_bytes()


# ---------------------------------------------------------------------------
# The render steps keep the shell's guards and its `set -e` failure semantics
# ---------------------------------------------------------------------------


def test_resolver_render_is_skipped_without_its_path() -> None:
    """The shell gated this on ONEX_SECRET_RESOLVER_CONFIG_PATH being non-empty."""
    assert entrypoint_preflight.render_resolver_config({}) == 0
    assert (
        entrypoint_preflight.render_resolver_config(
            {"ONEX_SECRET_RESOLVER_CONFIG_PATH": ""}
        )
        == 0
    )


def test_bifrost_render_is_skipped_without_its_path() -> None:
    """The shell gated this on BIFROST_CONTRACT_PATH being non-empty."""
    assert entrypoint_preflight.render_bifrost_contract({}) == 0
    assert (
        entrypoint_preflight.render_bifrost_contract({"BIFROST_CONTRACT_PATH": ""}) == 0
    )


def test_bifrost_render_delegates_to_the_module_main_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structural identity: same function, no args, no environment rewriting."""
    from omnibase_infra.runtime import render_bifrost_delegation_contract

    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        render_bifrost_delegation_contract,
        "main",
        lambda *args: (calls.append(args), 0)[1],
    )

    rc = entrypoint_preflight.render_bifrost_contract({"BIFROST_CONTRACT_PATH": "/x"})

    assert rc == 0
    assert calls == [()]


def test_a_failed_render_aborts_boot(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shell ran under `set -e`: a non-zero render never reached the kernel."""
    from omnibase_infra.runtime import render_secret_resolver_config

    monkeypatch.setattr(render_secret_resolver_config, "main", lambda: 3)
    monkeypatch.setattr(entrypoint_preflight, "stamp_all_fingerprints", lambda _env: 0)

    rc = entrypoint_preflight.run_preflight({"ONEX_SECRET_RESOLVER_CONFIG_PATH": "/x"})

    assert rc == 3


# ---------------------------------------------------------------------------
# Handoff: in-process for the image's CMD, execvp for anything else
# ---------------------------------------------------------------------------


def test_handoff_calls_the_kernel_in_this_process(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``CMD ["onex-runtime"]`` must NOT re-exec — that is the warm-import win."""
    from omnibase_infra.runtime import kernel

    started: list[str] = []
    monkeypatch.setattr(kernel, "main", lambda: started.append("kernel"))
    monkeypatch.setattr(
        entrypoint_preflight.os,
        "execvp",
        lambda *_a: pytest.fail("execvp must not run for the packaged kernel CMD"),
    )

    rc = entrypoint_preflight.handoff(["onex-runtime"])

    assert rc == 0
    assert started == ["kernel"]
    assert "[entrypoint] Starting runtime kernel..." in capsys.readouterr().out


def test_handoff_execs_any_other_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every other CMD keeps the old ``exec "$@"`` semantics exactly."""
    execs: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        entrypoint_preflight.os,
        "execvp",
        lambda file, args: execs.append((file, list(args))),
    )

    with pytest.raises(AssertionError, match="unreachable"):
        entrypoint_preflight.handoff(["onex-gateway-forwarder", "--flag"])

    assert execs == [("onex-gateway-forwarder", ["onex-gateway-forwarder", "--flag"])]


def test_handoff_with_arguments_does_not_take_the_in_process_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``onex-runtime --something`` is not the packaged no-arg CMD: exec it."""
    execs: list[list[str]] = []
    monkeypatch.setattr(
        entrypoint_preflight.os,
        "execvp",
        lambda _file, args: execs.append(list(args)),
    )

    with pytest.raises(AssertionError, match="unreachable"):
        entrypoint_preflight.handoff(["onex-runtime", "--debug"])

    assert execs == [["onex-runtime", "--debug"]]


def test_empty_cmd_aborts_rather_than_starting_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert entrypoint_preflight.handoff([]) == 1
    assert "no CMD to start" in capsys.readouterr().err


__all__: list[str] = []
