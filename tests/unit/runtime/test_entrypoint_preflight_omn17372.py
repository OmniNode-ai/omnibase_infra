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
    repo_src = Path(__file__).resolve().parents[3] / "src"
    result = subprocess.run(
        [sys.executable, "-m", "omnibase_infra.runtime.render_secret_resolver_config"],
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(repo_src),
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


# ---------------------------------------------------------------------------
# OMN-24483: the boot wait that replaces the migration-gate container
# ---------------------------------------------------------------------------

_WAIT_ENV = {
    "ONEX_BOOT_WAIT_FOR_MIGRATIONS": "true",
    "OMNIBASE_INFRA_DB_URL": "postgresql://u:p@postgres:5432/omnibase_infra",
    "OMNIDASH_ANALYTICS_DB_URL": "postgresql://u:p@postgres:5432/omnidash_analytics",
}


class _Clock:
    """A fake clock whose ``sleep`` advances it, so the wait runs with no wall time."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_wait_is_off_unless_opted_in() -> None:
    calls: list[object] = []
    rc = entrypoint_preflight.wait_for_migrations(
        {"OMNIBASE_INFRA_DB_URL": "postgresql://u:p@h/db"},
        probe=lambda *a: calls.append(a) or False,
    )
    assert rc == 0
    assert calls == []


def test_wait_skips_without_a_database_url() -> None:
    calls: list[object] = []
    rc = entrypoint_preflight.wait_for_migrations(
        {"ONEX_BOOT_WAIT_FOR_MIGRATIONS": "true"},
        probe=lambda *a: calls.append(a) or False,
    )
    assert rc == 0
    assert calls == []


def test_wait_returns_when_the_migrations_complete_after_a_few_polls() -> None:
    clock = _Clock()
    answers = iter([False, False, True])
    seen: list[tuple[str, str, tuple[str, ...]]] = []

    def probe(infra: str, node: str, tables: tuple[str, ...]) -> bool:
        seen.append((infra, node, tables))
        return next(answers)

    rc = entrypoint_preflight.wait_for_migrations(
        _WAIT_ENV, probe=probe, sleep=clock.sleep, monotonic=clock.monotonic
    )
    assert rc == 0
    assert len(seen) == 3
    assert len(clock.sleeps) == 2
    assert seen[0][1] == _WAIT_ENV["OMNIDASH_ANALYTICS_DB_URL"]
    assert seen[0][2] == ("delegation_events", "node_service_registry")


def test_wait_aborts_boot_when_the_window_ends_incomplete(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Falsifier control: a probe that never turns true must fail, not hang."""
    clock = _Clock()
    rc = entrypoint_preflight.wait_for_migrations(
        {**_WAIT_ENV, "ONEX_BOOT_MIGRATION_WAIT_SECONDS": "20"},
        probe=lambda *a: False,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert rc == 1
    assert "migrations not complete after 20s" in capsys.readouterr().err


def test_wait_reads_the_table_list_from_the_environment() -> None:
    clock = _Clock()
    seen: list[tuple[str, ...]] = []

    def probe(infra: str, node: str, tables: tuple[str, ...]) -> bool:
        seen.append(tables)
        return True

    entrypoint_preflight.wait_for_migrations(
        {**_WAIT_ENV, "ONEX_BOOT_REQUIRED_PROJECTION_TABLES": "a b c"},
        probe=probe,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert seen == [("a", "b", "c")]


def test_the_wait_runs_before_the_fingerprint_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stamp taken before the tables exist is wrong, so the wait comes first."""
    order: list[str] = []
    monkeypatch.setattr(
        entrypoint_preflight,
        "wait_for_migrations",
        lambda env: order.append("wait") or 0,
    )
    monkeypatch.setattr(
        entrypoint_preflight,
        "stamp_all_fingerprints",
        lambda env: order.append("stamp") or 0,
    )
    monkeypatch.setattr(entrypoint_preflight, "render_bifrost_contract", lambda e: 0)
    monkeypatch.setattr(entrypoint_preflight, "render_resolver_config", lambda e: 0)
    assert entrypoint_preflight.run_preflight({}) == 0
    assert order == ["wait", "stamp"]


def test_a_failed_wait_aborts_before_the_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stamped: list[str] = []
    monkeypatch.setattr(entrypoint_preflight, "wait_for_migrations", lambda env: 1)
    monkeypatch.setattr(
        entrypoint_preflight,
        "stamp_all_fingerprints",
        lambda env: stamped.append("stamp") or 0,
    )
    assert entrypoint_preflight.run_preflight({}) == 1
    assert stamped == []
