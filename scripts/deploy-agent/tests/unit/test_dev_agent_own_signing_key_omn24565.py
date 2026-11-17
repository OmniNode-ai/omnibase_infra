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
launcher would then keep the empty value or source the store and silently run the
dev agent on the SHARED key. The launcher is not changed; the unit itself refuses
with an ``ExecStartPre`` -- a ``/bin/sh -c`` check that exits 2 and prints only the
NAME -- ordered before the port preflight, so it runs before the launcher sources
the store. The check reads the KEY FILE, not the assembled environment: a user
manager that already carries the (protected) name would otherwise let a broken file
through. These tests derive that argv from the unit file, execute it against
broken key files placed in a temporary home, and run the failing controls through
the unchanged launcher to prove the silent fallback.

Every value below is a synthetic throwaway, and assertion messages carry variable
NAMES only, never values.
"""

import ast
import json
import os
import re
import shlex
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
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
_INHERITED_SYNTHETIC = "inherited-synthetic"
#: The template placeholder form (``docker/catalog`` refuses the same prefix).
_PLACEHOLDER = "__REPLACE_WITH_DEV_KEY__"
_REPO_ROOT = Path(__file__).resolve().parents[4]


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
    ``ExecStartPre`` that runs ``/bin/sh -c`` on the key FILE -- anything but one
    valid ``DEPLOY_AGENT_HMAC_SECRET=<key>`` line refuses with exit 2 and prints
    only the NAME -- ordered before the port preflight, therefore before the
    launcher sources the store.
    """
    argv = _start_check_argv()
    script = argv[argv.index("-c") + 1]

    assert argv[:2] == ["/bin/sh", "-c"], argv
    assert "exit 2" in script, _CODE_NAME
    # It reads the file the unit declares, by the same ``%h`` path.
    assert _KEY_FILE in script, _KEY_FILE
    # The accepted line is exactly the name and letters, digits, `-` and `_`.
    assert f'"{_CODE_NAME}=[A-Za-z0-9_-]+"' in script, _CODE_NAME
    # The test is on the file, never on the assembled environment.
    assert f'test -n "${{{_CODE_NAME}' not in script, _CODE_NAME
    assert f"${_CODE_NAME}" not in script, _CODE_NAME
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


#: The check this unit carried at the reviewed head: a test on the environment.
_OLD_ENVIRONMENT_CHECK = (
    '/bin/sh -c \'test -n "$${DEPLOY_AGENT_HMAC_SECRET:-}" || '
    '{ echo "unset or empty" >&2; exit 2; }\''
)


def _start_check_argv(home: Path | None = None) -> list[str]:
    """The dev unit's signing-key ExecStartPre, as the argv systemd runs.

    systemd turns ``$$`` into a literal ``$`` and ``%h`` into the unit user's home
    before the shell sees the line, so the ``-c`` script word is expanded the same
    way here (``home`` stands in for ``%h``). Exactly one such ExecStartPre may
    exist, and it must be ordered BEFORE the port preflight's, so a broken key
    refuses the start before anything else is probed.
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

    return _argv_of(check[1][len("ExecStartPre=") :], home)


def _argv_of(exec_line: str, home: Path | None = None) -> list[str]:
    argv = shlex.split(exec_line)
    script_index = argv.index("-c") + 1
    script = argv[script_index].replace("$$", "$")
    if home is not None:
        script = script.replace("%h", str(home))
    argv[script_index] = script
    return argv


def _run_start_check(
    tmp_path: Path,
    key_file_text: str | None,
    *,
    environment: dict[str, str] | None = None,
    argv: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the start check with ``tmp_path`` as the unit user's home.

    ``key_file_text`` is the key file's bytes (``None`` leaves no file at all).
    ``environment`` is what the user manager already carries -- the check must not
    take it for the file's content. systemd applies the file's assignments over it
    (``_systemd_environment``), so the process sees both, exactly as ExecStartPre does.
    """
    key_dir = tmp_path / ".omnibase"
    key_dir.mkdir(exist_ok=True)
    key_file = key_dir / Path(_KEY_FILE).name
    process_env = dict(environment or {})
    if key_file_text is not None:
        key_file.write_text(key_file_text)
        process_env.update(_systemd_environment(key_file_text))
    return subprocess.run(
        argv if argv is not None else _start_check_argv(tmp_path),
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ.get("PATH", ""), **process_env},
    )


_BAD_KEY_FILES = [
    # systemd accepts the line and sets the variable to the EMPTY string.
    pytest.param("DEPLOY_AGENT_HMAC_SECRET=\n", id="empty-valued-line"),
    # A typo'd NAME is silently dropped; the real name is never set at all.
    pytest.param("DEPLOY_AGENT_HMAC=dev-synthetic\n", id="wrong-named-line"),
    # Not an assignment at all: silently ignored by systemd's parser.
    pytest.param("this is not an assignment\n", id="garbage-line"),
    # Zero bytes: the file exists (so the required EnvironmentFile= is satisfied).
    pytest.param("", id="zero-byte-file"),
    # Nothing but blanks: no assignment at all.
    pytest.param("   \n\t\n", id="whitespace-only-file"),
    # systemd strips the blanks and sets the name to the empty string.
    pytest.param("DEPLOY_AGENT_HMAC_SECRET=   \n", id="whitespace-only-value"),
    # The template placeholder has only letters, digits and `_`, so the character
    # set alone would let it through.
    pytest.param(f"DEPLOY_AGENT_HMAC_SECRET={_PLACEHOLDER}\n", id="placeholder"),
    # systemd takes the LAST assignment: a good line then an empty one is empty.
    pytest.param(
        f"DEPLOY_AGENT_HMAC_SECRET={_DEV_SYNTHETIC}\nDEPLOY_AGENT_HMAC_SECRET=\n",
        id="good-line-then-empty-line",
    ),
    # A value outside letters, digits, `-` and `_` (the generator never writes one).
    pytest.param(f'DEPLOY_AGENT_HMAC_SECRET="{_DEV_SYNTHETIC}"\n', id="quoted-value"),
]


@pytest.mark.parametrize("key_file_text", _BAD_KEY_FILES)
def test_the_units_start_check_refuses_a_broken_key_file(
    tmp_path: Path, key_file_text: str
) -> None:
    """Exit 2, before the launcher -- and the store it sources -- ever run."""
    result = _run_start_check(tmp_path, key_file_text)

    assert result.returncode == 2, (result.returncode, result.stdout)
    assert _CODE_NAME in result.stderr, _CODE_NAME
    assert _DEV_SYNTHETIC not in result.stderr, _CODE_NAME
    assert _SHARED_SYNTHETIC not in result.stderr, _CODE_NAME
    assert _PLACEHOLDER not in result.stderr, _CODE_NAME


@pytest.mark.parametrize("key_file_text", _BAD_KEY_FILES)
def test_the_units_start_check_refuses_a_broken_file_when_the_name_is_inherited(
    tmp_path: Path, key_file_text: str
) -> None:
    """OMN-24611 review: the user manager may already carry the (protected) name.

    ``environment.d`` or ``systemctl --user import-environment`` puts
    ``DEPLOY_AGENT_HMAC_SECRET`` in the environment before the unit's own file is
    read; the launcher keeps a protected name, so a check on the environment would
    pass a broken file. The check is on the file: still exit 2.
    """
    result = _run_start_check(
        tmp_path,
        key_file_text,
        environment={_CODE_NAME: _INHERITED_SYNTHETIC},
    )

    assert result.returncode == 2, (result.returncode, result.stdout)
    assert _CODE_NAME in result.stderr, _CODE_NAME
    assert _INHERITED_SYNTHETIC not in result.stderr, _CODE_NAME


def test_control_an_environment_check_passes_a_broken_file_with_an_inherited_name(
    tmp_path: Path,
) -> None:
    """The error chain: the old check (on the environment) lets the broken file by."""
    old_argv = _argv_of(_OLD_ENVIRONMENT_CHECK)
    result = _run_start_check(
        tmp_path,
        "this is not an assignment\n",
        environment={_CODE_NAME: _INHERITED_SYNTHETIC},
        argv=old_argv,
    )

    assert result.returncode == 0, (result.returncode, result.stderr)


def test_the_units_start_check_passes_a_good_key_file(tmp_path: Path) -> None:
    """The guard must not disturb the good path: a real dev key exits 0."""
    result = _run_start_check(tmp_path, f"{_CODE_NAME}={_DEV_SYNTHETIC}\n")

    assert result.returncode == 0, (result.returncode, result.stderr)
    assert _CODE_NAME not in result.stderr, _CODE_NAME


def test_the_units_start_check_passes_a_good_key_file_with_comments(
    tmp_path: Path,
) -> None:
    """Positive control for the line check: comments and a key with `-` and `_` pass."""
    result = _run_start_check(
        tmp_path, f"# delivered by the host installer\n{_CODE_NAME}=Ab-9_zY\n"
    )

    assert result.returncode == 0, (result.returncode, result.stderr)


@pytest.mark.parametrize(
    ("key_file_text", "expected_env"),
    [
        # The name is set EMPTY: the protected snapshot restores the empty value
        # over the store, so the agent starts signing with an empty key.
        pytest.param("DEPLOY_AGENT_HMAC_SECRET=\n", "", id="empty-valued-line"),
        pytest.param("DEPLOY_AGENT_HMAC_SECRET=   \n", "", id="whitespace-only-value"),
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
        pytest.param("", _SHARED_SYNTHETIC, id="zero-byte-file"),
        pytest.param("   \n\t\n", _SHARED_SYNTHETIC, id="whitespace-only-file"),
        # The placeholder is a well-formed assignment: the agent would sign with it.
        pytest.param(
            f"DEPLOY_AGENT_HMAC_SECRET={_PLACEHOLDER}\n",
            _PLACEHOLDER,
            id="placeholder",
        ),
    ],
)
def test_control_without_the_start_check_a_broken_key_file_starts_silently(
    tmp_path: Path, key_file_text: str, expected_env: str
) -> None:
    """The error chain: no ExecStartPre and the agent starts anyway.

    With the same broken key files, the same launcher, and the same store, the
    dev agent runs WITHOUT its own key -- on the shared value when the name is
    dropped entirely, on the empty or placeholder value when systemd set that.
    None of them is the dev key; this silent fallback is what the start check
    refuses.
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
# The refusal must not loop (OMN-24611). `Restart=on-failure` restarts an exit 2
# from ExecStartPre; systemd's RestartPreventExitStatus= judges the MAIN process
# only, so the bound is the start limit: the sixth start inside the interval is
# refused and the unit lands in `failed`.
# --------------------------------------------------------------------------


def _seconds(value: str) -> int:
    assert re.fullmatch(r"\d+", value), value
    return int(value)


def _unit_setting(name: str) -> str:
    values = [
        line.split("=", 1)[1]
        for line in _directive_lines(_DEV_UNIT)
        if line.startswith(f"{name}=")
    ]
    assert len(values) == 1, name
    return values[0]


def test_the_exit_2_refusal_is_bounded_by_the_start_limit() -> None:
    assert _unit_setting("Restart") == "on-failure"
    interval = _seconds(_unit_setting("StartLimitIntervalSec"))
    burst = _seconds(_unit_setting("StartLimitBurst"))
    restart_sec = _seconds(_unit_setting("RestartSec"))

    # `burst` starts are allowed; the next one inside the interval is refused. The
    # burst must therefore fit inside the interval, or the limit never trips.
    assert burst >= 1, "StartLimitBurst"
    assert burst * restart_sec < interval, "StartLimitIntervalSec"


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


# --------------------------------------------------------------------------
# The dev runtime's containers sign with the same name, and get the dev key only
# because compose interpolates it from the agent's environment. Anything that
# loads the operator store AROUND that environment hands them the shared key (an
# ``env_file:`` entry lands in the container directly; the dev agent then rejects
# what they sign). Pin that nothing does.
# --------------------------------------------------------------------------

#: What names the operator store: its home directory, or the unit's own variable.
_OPERATOR_STORE_MARKERS = (".omnibase", "DEPLOY_AGENT_ENV_FILE", "OMNIBASE_ENV_FILE")


class _ComposeLoader(yaml.SafeLoader):
    """Safe loader that reads compose's merge tags (``!override``, ``!reset``) as data."""


def _read_tagged(loader: _ComposeLoader, tag_suffix: str, node: yaml.Node) -> object:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    return None


_ComposeLoader.add_multi_constructor("!", _read_tagged)


def _env_file_entries(compose_text: str) -> list[str]:
    """Every ``env_file`` entry of every service in one compose document, as text."""
    loader = _ComposeLoader(compose_text)
    try:
        document = loader.get_single_data()
    finally:
        loader.dispose()
    services = document.get("services") if isinstance(document, dict) else None
    entries: list[str] = []
    for service in (services or {}).values():
        declared = service.get("env_file") if isinstance(service, dict) else None
        if declared is None:
            continue
        for item in declared if isinstance(declared, list) else [declared]:
            entries.append(
                str(item.get("path", "") if isinstance(item, dict) else item)
            )
    return entries


def _compose_files() -> list[Path]:
    docker_dir = _REPO_ROOT / "docker"
    files = sorted({*docker_dir.rglob("*.yml"), *docker_dir.rglob("*.yaml")})
    return [path for path in files if "services:" in path.read_text()]


def _operator_store_entries(compose_text: str) -> list[str]:
    return [
        entry
        for entry in _env_file_entries(compose_text)
        if any(marker in entry for marker in _OPERATOR_STORE_MARKERS)
    ]


def test_no_compose_file_loads_the_operator_store_through_env_file() -> None:
    files = _compose_files()
    # A glob that finds nothing would pass vacuously.
    assert len(files) >= 10, "compose files under docker/"
    assert any(path.name == "docker-compose.infra.yml" for path in files)
    for path in files:
        assert _operator_store_entries(path.read_text()) == [], path.name


def test_control_a_compose_env_file_naming_the_operator_store_is_found() -> None:
    """The failing control: the scan above is not blind to what it forbids."""
    for declared in (
        "      - ${HOME}/.omnibase/.env",
        "      - path: ${DEPLOY_AGENT_ENV_FILE}\n        required: true",
    ):
        text = f"services:\n  runtime:\n    image: x\n    env_file:\n{declared}\n"
        assert _operator_store_entries(text) != [], declared
    # ...and a file that is not the operator store is not a finding.
    other = "services:\n  runtime:\n    image: x\n    env_file:\n      - ./local.env\n"
    assert _operator_store_entries(other) == []


def _argv_lists_naming(source: Path, literal: str) -> list[list[object]]:
    return [
        [
            element.value if isinstance(element, ast.Constant) else None
            for element in node.elts
        ]
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.List)
        and any(
            isinstance(element, ast.Constant) and element.value == literal
            for element in node.elts
        )
    ]


def test_no_deploy_agent_compose_argv_passes_env_file() -> None:
    """``--env-file`` on a compose command loads the store under the agent's environment."""
    package = _REPO_ROOT / "scripts" / "deploy-agent" / "deploy_agent"
    seen = 0
    for source in sorted(package.glob("*.py")):
        for argv in _argv_lists_naming(source, "--env-file"):
            seen += 1
            # The one argv that passes it is the dev-lane pin repoint script's, and
            # it names no compose command.
            assert "compose" not in argv and "docker" not in argv, (source.name, argv)
    assert seen >= 1, "the --env-file scan found nothing, so it proves nothing"


def test_control_a_compose_argv_with_env_file_is_found(tmp_path: Path) -> None:
    source = tmp_path / "executor.py"
    source.write_text('ARGV = ["docker", "compose", "--env-file", "x", "up"]\n')
    (argv,) = _argv_lists_naming(source, "--env-file")
    assert "compose" in argv
