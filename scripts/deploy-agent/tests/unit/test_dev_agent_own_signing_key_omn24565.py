# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev deploy agent signs with its own dev-only key, not the shared one (OMN-24565).

The operator env store carries one ``DEPLOY_AGENT_HMAC_SECRET``, and the live deploy
agent reads the same name from the same file. The dev agent
(``deploy-agent-dev.service``) must never hold that value: its unit reads the
dev-only key from a second ``EnvironmentFile=`` that holds that one key, and lists
the name in ``DEPLOY_AGENT_ENV_PROTECTED`` so neither the launcher's store source
nor ``deploy_agent.executor.operator_store_overlay`` puts the shared value back.
``auth.py`` and the compose files do not change.

The key file is required-to-exist (no leading ``-``), but that is not
required-to-parse (OMN-24611): an empty-valued line sets the name to the EMPTY
string, a wrong-named line sets only the wrong name, and an unparseable line
sets nothing -- ``DEPLOY_AGENT_HMAC_SECRET`` stays unset or empty, and the
launcher would then source the store and silently run the dev agent on the
SHARED key. The launcher is not changed; the unit itself refuses with an
``ExecStartPre`` -- a ``/bin/sh -c`` test on the name that exits 2 and prints
only the NAME -- ordered before the port preflight, so it runs before the
launcher sources the store. These tests derive that argv from the unit file,
execute it against the three broken key files, and run the failing control
through the unchanged launcher to prove the silent fallback.

Every value below is a synthetic throwaway, and assertion messages carry variable
NAMES only, never values.
"""

import json
import os
import re
import shlex
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from deploy_agent.executor import operator_store_overlay, reset_operator_store_cache

pytestmark = pytest.mark.unit

_DEPLOY_DIR = Path(__file__).resolve().parents[2] / "deploy"
_DEV_UNIT = _DEPLOY_DIR / "deploy-agent-dev.service"
_PROD_DROP_IN_DIR = _DEPLOY_DIR / "deploy-agent.service.d"
_LAUNCHER = _DEPLOY_DIR / "deploy-agent-launch.sh"

_CODE_NAME = "DEPLOY_AGENT_HMAC_SECRET"
_DEV_STORE_NAME = "DEPLOY_AGENT_DEV_HMAC_SECRET"
_PROTECTED_NAME = "DEPLOY_AGENT_ENV_PROTECTED"
#: The dev agent's own key file: the one key and nothing else, in the unit user's home.
_KEY_FILE = "%h/.omnibase/deploy-agent-dev-signing-key.env"

_SHARED_SYNTHETIC = "shared-synthetic"
_DEV_SYNTHETIC = "dev-synthetic"


def _directive_lines(unit: Path) -> list[str]:
    return [
        line
        for line in unit.read_text().splitlines()
        if line and not line.lstrip().startswith("#")
    ]


def _unit_value(unit: Path, name: str) -> str:
    for line in _directive_lines(unit):
        for prefix in (f"Environment={name}=", f'Environment="{name}='):
            if line.startswith(prefix):
                return line[len(prefix) :].rstrip('"')
    raise AssertionError(f"{unit.name} does not declare Environment={name}=")


def _dev_protected() -> list[str]:
    return _unit_value(_DEV_UNIT, _PROTECTED_NAME).split()


# --------------------------------------------------------------------------
# Unit-level invariants. The first is the failing control: on origin/dev the unit
# declares no EnvironmentFile= and the name is not protected.
# --------------------------------------------------------------------------


def test_dev_unit_reads_its_own_signing_key_file() -> None:
    files = [
        line
        for line in _directive_lines(_DEV_UNIT)
        if line.startswith("EnvironmentFile=")
    ]
    # Exactly one, required (no leading `-`: an absent key fails the start loudly),
    # and never the operator store (OMN-18073: systemd mangles its quoting).
    assert files == [f"EnvironmentFile={_KEY_FILE}"], "EnvironmentFile="
    assert ".omnibase/.env" not in files[0], "EnvironmentFile="


def test_dev_unit_protects_the_signing_key_name_from_the_store() -> None:
    """Without the entry the launcher's store source puts the shared value back."""
    assert _CODE_NAME in _dev_protected(), _PROTECTED_NAME


def test_dev_unit_declares_the_signing_key_start_check() -> None:
    """OMN-24611: a broken key file must refuse the start, not run on the shared key.

    The launcher is not changed, so the guard lives in the dev unit itself: an
    ``ExecStartPre`` that runs ``/bin/sh -c`` on the name -- unset or empty
    refuses with exit 2 and prints only the NAME -- ordered before the port
    preflight, therefore before the launcher sources the store.
    """
    argv = _start_check_argv()
    script = argv[argv.index("-c") + 1]

    assert argv[:2] == ["/bin/sh", "-c"], argv
    assert f'test -n "${{{_CODE_NAME}:-}}"' in script, _CODE_NAME
    assert "exit 2" in script, _CODE_NAME
    assert f"deploy-agent-dev: {_CODE_NAME} is unset or empty" in script, _CODE_NAME
    # No value of any kind is baked into the unit -- the NAME only.
    assert _DEV_SYNTHETIC not in script, _CODE_NAME
    assert _SHARED_SYNTHETIC not in script, _CODE_NAME


def test_no_other_unit_reads_the_dev_signing_key_file() -> None:
    """The live agent's unit and its drop-in keep reading the shared name alone."""
    texts = [
        path.read_text()
        for path in sorted(_DEPLOY_DIR.glob("*.service"))
        if path != _DEV_UNIT
    ]
    texts += [
        path.read_text()
        for path in sorted(_PROD_DROP_IN_DIR.glob("*"))
        if path.is_file()
    ]
    assert len(texts) >= 2, "no other unit file found -- the glob is wrong"
    for text in texts:
        assert "deploy-agent-dev-signing-key" not in text, "EnvironmentFile="
        assert _DEV_STORE_NAME not in text, _DEV_STORE_NAME


# --------------------------------------------------------------------------
# Behaviour: systemd sets the dev value, the launcher sources a store holding the
# shared one, and the agent's environment carries the dev value.
# --------------------------------------------------------------------------


def _systemd_environment(key_file_text: str) -> dict[str, str]:
    """What systemd's ``EnvironmentFile=`` hands the process for this key file.

    Only ``NAME=value`` lines whose NAME is a valid identifier become variables.
    Blank lines, lines starting with ``#`` or ``;``, and any other line are
    ignored -- silently, which is exactly the failure mode: a wrong-named or
    garbage line drops the key without a sound and the variable it was meant to
    set stays unset. Values lose their surrounding whitespace and one pair of
    matching quotes.
    """
    variables: dict[str, str] = {}
    for line in key_file_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        name, sep, value = stripped.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        variables[name] = value
    return variables


def _run_launcher(
    tmp_path: Path,
    protected: str,
    systemd_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run the launcher as systemd would start it, with a stub that dumps its env.

    ``systemd_env`` is what the unit's ``EnvironmentFile=`` put in the process
    before ExecStart (default: the good dev key). Returns the raw
    CompletedProcess and the dump path; `_launch` is the success-shaped wrapper.
    """
    dump = tmp_path / "env.json"
    stub = tmp_path / "python"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os\n"
        f"open({str(dump)!r}, 'w').write(json.dumps(dict(os.environ)))\n"
    )
    stub.chmod(0o755)
    store = tmp_path / "store.env"
    store.write_text(f"{_CODE_NAME}={_SHARED_SYNTHETIC}\nSTORE_ONLY_NAME=from-store\n")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "DEPLOY_AGENT_ENV_FILE": str(store),
        "DEPLOY_AGENT_PYTHON": str(stub),
        _PROTECTED_NAME: protected,
    }
    # What systemd's EnvironmentFile= puts in the process before ExecStart.
    env.update(systemd_env if systemd_env is not None else {_CODE_NAME: _DEV_SYNTHETIC})
    result = subprocess.run(
        [str(_LAUNCHER)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    return result, dump


def _launch(
    tmp_path: Path,
    protected: str,
    systemd_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """A start that is expected to reach the interpreter: exit 0, env dumped."""
    result, dump = _run_launcher(tmp_path, protected, systemd_env=systemd_env)
    assert result.returncode == 0, (result.returncode, result.stderr)
    return dict(json.loads(dump.read_text()))


def test_launcher_keeps_the_dev_key_over_the_stores_shared_value(
    tmp_path: Path,
) -> None:
    env = _launch(tmp_path, " ".join(_dev_protected()))

    assert env[_CODE_NAME] == _DEV_SYNTHETIC, _CODE_NAME
    # Positive control: the store was sourced, so the line above is precedence and
    # not a source that did nothing.
    assert env["STORE_ONLY_NAME"] == "from-store", "STORE_ONLY_NAME"
    for name, value in env.items():
        assert value != _SHARED_SYNTHETIC, name


def test_control_without_the_protected_entry_the_shared_value_wins(
    tmp_path: Path,
) -> None:
    """The error chain: drop the entry and the dev agent holds the shared key."""
    protected = [name for name in _dev_protected() if name != _CODE_NAME]
    env = _launch(tmp_path, " ".join(protected))

    assert env[_CODE_NAME] == _SHARED_SYNTHETIC, _CODE_NAME


# --------------------------------------------------------------------------
# OMN-24611: the key file is required-to-exist, not required-to-parse. An empty
# file, a wrong-named line, or a line systemd cannot read sets nothing, and the
# unit's ExecStartPre must refuse the start rather than let the launcher fall
# back to the store's shared key.
# --------------------------------------------------------------------------


def _start_check_argv() -> list[str]:
    """The dev unit's signing-key ExecStartPre, as the argv systemd runs.

    systemd turns ``$$`` into a literal ``$`` before the shell sees the line,
    so the ``-c`` script word is un-escaped the same way here. Exactly one such
    ExecStartPre may exist, and it must be ordered BEFORE the port preflight's,
    so a broken key refuses the start before anything else is probed.
    """
    check: tuple[int, str] | None = None
    preflight: int | None = None
    for index, line in enumerate(_directive_lines(_DEV_UNIT)):
        if not line.startswith("ExecStartPre="):
            continue
        if _CODE_NAME in line:
            assert check is None, _CODE_NAME
            check = (index, line)
        elif "preflight_port_free.sh" in line:
            assert preflight is None, "preflight_port_free.sh"
            preflight = index
    assert check is not None, _CODE_NAME
    assert preflight is not None, "preflight_port_free.sh"
    assert check[0] < preflight, _CODE_NAME

    argv = shlex.split(check[1][len("ExecStartPre=") :])
    script_index = argv.index("-c") + 1
    argv[script_index] = argv[script_index].replace("$$", "$")
    return argv


_BAD_KEY_FILES = [
    # systemd accepts the line and sets the variable to the EMPTY string.
    pytest.param("DEPLOY_AGENT_HMAC_SECRET=\n", id="empty-valued-line"),
    # A typo'd NAME is silently dropped; the real name is never set at all.
    pytest.param("DEPLOY_AGENT_HMAC=dev-synthetic\n", id="wrong-named-line"),
    # Not an assignment at all: silently ignored by systemd's parser.
    pytest.param("this is not an assignment\n", id="garbage-line"),
]


@pytest.mark.parametrize("key_file_text", _BAD_KEY_FILES)
def test_the_units_start_check_refuses_a_broken_key_file(
    key_file_text: str,
) -> None:
    """Exit 2, before the launcher -- and the store it sources -- ever run."""
    result = subprocess.run(
        _start_check_argv(),
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": os.environ.get("PATH", ""),
            **_systemd_environment(key_file_text),
        },
    )

    assert result.returncode == 2, (result.returncode, result.stdout)
    assert _CODE_NAME in result.stderr, _CODE_NAME
    assert _DEV_SYNTHETIC not in result.stderr, _CODE_NAME
    assert _SHARED_SYNTHETIC not in result.stderr, _CODE_NAME


def test_the_units_start_check_passes_a_good_key_file() -> None:
    """The guard must not disturb the good path: a real dev key exits 0."""
    result = subprocess.run(
        _start_check_argv(),
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": os.environ.get("PATH", ""),
            **_systemd_environment(f"{_CODE_NAME}={_DEV_SYNTHETIC}\n"),
        },
    )

    assert result.returncode == 0, (result.returncode, result.stderr)
    assert _CODE_NAME not in result.stderr, _CODE_NAME


@pytest.mark.parametrize(
    ("key_file_text", "expected_env"),
    [
        # The name is set EMPTY: the protected snapshot restores the empty value
        # over the store, so the agent starts signing with an empty key.
        pytest.param("DEPLOY_AGENT_HMAC_SECRET=\n", "", id="empty-valued-line"),
        # The name is absent from the process environment, so the store's shared
        # value survives the source.
        pytest.param(
            "DEPLOY_AGENT_HMAC=dev-synthetic\n",
            _SHARED_SYNTHETIC,
            id="wrong-named-line",
        ),
        pytest.param(
            "this is not an assignment\n",
            _SHARED_SYNTHETIC,
            id="garbage-line",
        ),
    ],
)
def test_control_without_the_start_check_a_broken_key_file_starts_silently(
    tmp_path: Path, key_file_text: str, expected_env: str
) -> None:
    """The error chain: no ExecStartPre and the agent starts anyway.

    With the same broken key files, the same launcher, and the same store, the
    dev agent runs WITHOUT its own key -- on the shared value when the name is
    dropped entirely, on the empty value when systemd set it empty. Neither is
    the dev key; this silent fallback is what the start check refuses.
    """
    env = _launch(
        tmp_path,
        " ".join(_dev_protected()),
        systemd_env=_systemd_environment(key_file_text),
    )

    assert env[_CODE_NAME] == expected_env, _CODE_NAME
    # The store really was sourced, so the run above is a real start on the
    # shared environment and not a vacuous no-op.
    assert env["STORE_ONLY_NAME"] == "from-store", "STORE_ONLY_NAME"


# --------------------------------------------------------------------------
# The seam with the executor: compose commands re-read the store for every name the
# unit does not own, and must not carry the shared key to the dev runtime.
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_operator_store_cache() -> Iterator[None]:
    reset_operator_store_cache()
    yield
    reset_operator_store_cache()


def _overlay(tmp_path: Path, protected: list[str]) -> dict[str, str]:
    store = tmp_path / "store.env"
    store.write_text(f"{_CODE_NAME}={_SHARED_SYNTHETIC}\n")
    return operator_store_overlay(
        {
            "PATH": os.environ.get("PATH", ""),
            "DEPLOY_AGENT_ENV_FILE": str(store),
            _PROTECTED_NAME: " ".join(protected),
        }
    )


def test_the_overlay_never_reapplies_the_shared_value(tmp_path: Path) -> None:
    assert _CODE_NAME not in _overlay(tmp_path, _dev_protected()), _CODE_NAME


def test_control_the_overlay_reapplies_it_when_the_name_is_not_protected(
    tmp_path: Path,
) -> None:
    protected = [name for name in _dev_protected() if name != _CODE_NAME]
    assert _overlay(tmp_path, protected)[_CODE_NAME] == _SHARED_SYNTHETIC, _CODE_NAME
