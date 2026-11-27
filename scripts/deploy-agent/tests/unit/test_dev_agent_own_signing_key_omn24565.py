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
import sys
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


#: The contract of this pull request's own ticket: its summary carries the rollout order.
_CONTRACT = _REPO_ROOT / "contracts" / "OMN-24611.yaml"
_HEADER_ORDER = "put the key file on the host FIRST, then install this unit"
_SUMMARY_ORDER = "Rollout order is the key file on h201 first and then the unit"


def _flat(comment_lines: list[str]) -> str:
    """Comment text with the ``#`` markers and line breaks collapsed to single spaces."""
    return " ".join(line.lstrip("#").strip() for line in comment_lines if line.strip())


def _unit_header_states_the_rollout_order(unit_text: str) -> bool:
    """The comment block above ``[Unit]`` names the order: key file first, unit second."""
    header = unit_text.split("\n[Unit]", 1)[0].splitlines()
    return "ROLLOUT ORDER" in _flat(header) and _HEADER_ORDER in _flat(header)


def _contract_summary_states_the_rollout_order(summary: str) -> bool:
    return _SUMMARY_ORDER in " ".join(summary.split())


_COUNT_CLAIM = re.compile(r"fails (\d+) of its (\d+) tests")


def _collected_test_count() -> int:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-n",
            "0",
            "-p",
            "no:cacheprovider",
            str(Path(__file__)),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-500:]
    return sum(1 for line in result.stdout.splitlines() if "::" in line)


def test_the_contract_counts_the_tests_of_this_file() -> None:
    """The count in the contract is the measured count: both places state the same total.

    The failing count (how many of them fail against the dev unit on origin/dev) is
    measured on the lab and named in the pull request comment; the total is checked
    here so a stale ``of its N tests`` cannot survive a new test.
    """
    total = _collected_test_count()
    document = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    statements = [
        criterion["statement"]
        for requirement in document["requirements"]
        for criterion in requirement["acceptance"]
        if criterion["id"] == "AC1"
    ] + [
        item["description"]
        for item in document["dod_evidence"]
        if item["id"] == "dod-omn24611-ac1-unit"
    ]
    assert len(statements) == 2, statements
    for statement in statements:
        # The newest claim: the one about the dev unit on origin/dev at this head.
        claims = _COUNT_CLAIM.findall(statement)
        failing, of_total = (int(n) for n in claims[0])
        assert of_total == total, (of_total, total)
        assert 0 < failing < total, (failing, total)


def test_control_a_stale_count_is_found() -> None:
    assert _COUNT_CLAIM.findall("fails 28 of its 41 tests")[0] == ("28", "41")


def test_rollout_order_is_stated_in_the_unit_header() -> None:
    """The unit installed before its key file fails every start: the header says so."""
    assert _unit_header_states_the_rollout_order(_DEV_UNIT.read_text())


def test_rollout_order_is_stated_in_the_contract_summary() -> None:
    document = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    assert _contract_summary_states_the_rollout_order(document["summary"])


def test_rollout_order_control_removing_the_statement_is_found() -> None:
    """The failing control: the checks above are red when the order is not stated."""
    text = _DEV_UNIT.read_text()
    assert not _unit_header_states_the_rollout_order(
        text.replace("ROLLOUT ORDER", "ORDER")
    )
    assert not _unit_header_states_the_rollout_order(
        text.replace("put the key file on the host FIRST", "install this unit FIRST")
    )
    summary = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))["summary"]
    assert not _contract_summary_states_the_rollout_order(
        summary.replace("key file on h201 first", "unit first")
    )


# --------------------------------------------------------------------------
# Behaviour: systemd sets the dev value, the launcher sources a store holding the
# shared one, and the agent's environment carries the dev value.
# --------------------------------------------------------------------------


def _systemd_environment(key_file_text: str) -> dict[str, str]:
    r"""What systemd's ``EnvironmentFile=`` hands the process for this key file.

    Only ``NAME=value`` lines whose NAME is a valid identifier become variables.
    Leading and trailing blanks are stripped, so an INDENTED assignment counts, and
    the LAST assignment of a name wins. A line ending in a backslash continues on the
    next physical line -- a comment line too, so ``# note \`` swallows the line after
    it. Blank lines, lines starting with ``#`` or ``;``, and any other line are
    ignored -- silently, which is exactly the failure mode: a wrong-named or
    garbage line drops the key without a sound and the variable it was meant to
    set stays unset. Values lose their surrounding whitespace and one pair of
    matching quotes. Settings made here override the unit's own ``Environment=``
    (systemd.exec: "Settings from these files override settings made with
    Environment="), so an extra name such as ``DEPLOY_AGENT_ENV_PROTECTED=`` also
    replaces the unit's value.
    """
    logical: list[str] = []
    pending = ""
    for physical in key_file_text.splitlines():
        joined = pending + physical
        if joined.endswith("\\"):
            pending = joined[:-1]
            continue
        logical.append(joined)
        pending = ""
    if pending:
        logical.append(pending)

    variables: dict[str, str] = {}
    for line in logical:
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


#: The check this unit carried at the first reviewed head 08f64f57f7ca: a test on the
#: environment, copied from its ExecStartPre line.
_OLD_ENVIRONMENT_CHECK = (
    '/bin/sh -c \'test -n "$${DEPLOY_AGENT_HMAC_SECRET:-}" || '
    '{ echo "deploy-agent-dev: DEPLOY_AGENT_HMAC_SECRET is unset or empty in the key '
    "file, refusing to start (OMN-24611)\" >&2; exit 2; }'"
)

#: The check this unit carried at the second reviewed head 835b8e9671e5: a test on the
#: key FILE, one line check plus a count of lines starting with the name (the line
#: check the review of OMN-25201 names as AC1's falsifier). Copied from that head's
#: ExecStartPre line.
_OLD_LINE_CHECK = (
    '/bin/sh -c \'f="%h/.omnibase/deploy-agent-dev-signing-key.env"; grep -Eqx "DEPLOY_AGENT_HMAC_SECRET=[A-Za-z0-9_-]+" "$$f"'
    ' && ! grep -Eqx "DEPLOY_AGENT_HMAC_SECRET=__REPLACE_WITH_.*" "$$f" && test "$$(grep -c "^DEPLOY_AGENT_HMAC_SECRET" "$$f")" = 1'
    ' || { echo "deploy-agent-dev: the key file does not hold exactly one valid DEPLOY_AGENT_HMAC_SECRET line, refusing to start (OMN-24611)" >&2; exit 2; }\''
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
    # OMN-24611 review: systemd skips leading blanks and takes the LAST assignment, so
    # a good line followed by an INDENTED one (here the placeholder) sets the name to
    # the placeholder; a line-anchored count of the name sees one line.
    pytest.param(
        f"DEPLOY_AGENT_HMAC_SECRET={_DEV_SYNTHETIC}\n"
        f"  DEPLOY_AGENT_HMAC_SECRET={_PLACEHOLDER}\n",
        id="good-line-then-indented-placeholder",
    ),
    # A comment line ending in a backslash continues onto the next line, so it
    # swallows the key line: the name is never set and the shared key wins.
    pytest.param(
        f"# delivered by the host installer \\\nDEPLOY_AGENT_HMAC_SECRET={_DEV_SYNTHETIC}\n",
        id="backslash-comment-before-the-key-line",
    ),
    # Any other name overrides the unit's own Environment= (an EnvironmentFile=
    # setting beats it): here the protected list is cut down to `PATH`, so the
    # launcher's store source puts the shared key back.
    pytest.param(
        f"DEPLOY_AGENT_HMAC_SECRET={_DEV_SYNTHETIC}\nDEPLOY_AGENT_ENV_PROTECTED=PATH\n",
        id="extra-name-overriding-the-units-environment",
    ),
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


#: Files the OLD environment check lets by with the name inherited from the user
#: manager (or set by the file itself): garbage, plus the three shapes the line-based
#: check of the reviewed head let by.
_OLD_CHECK_PASSES = [
    pytest.param("this is not an assignment\n", id="garbage-line"),
    *[
        pytest.param(case.values[0], id=str(case.id))
        for case in _BAD_KEY_FILES
        if case.id
        in {
            "good-line-then-indented-placeholder",
            "backslash-comment-before-the-key-line",
            "extra-name-overriding-the-units-environment",
        }
    ],
]


@pytest.mark.parametrize("key_file_text", _OLD_CHECK_PASSES)
def test_control_an_environment_check_passes_a_broken_file_with_an_inherited_name(
    tmp_path: Path, key_file_text: str
) -> None:
    """The error chain: the old check (on the environment) lets the broken file by."""
    old_argv = _argv_of(_OLD_ENVIRONMENT_CHECK)
    result = _run_start_check(
        tmp_path,
        key_file_text,
        environment={_CODE_NAME: _INHERITED_SYNTHETIC},
        argv=old_argv,
    )

    assert result.returncode == 0, (result.returncode, result.stderr)


_LINE_CHECK_PASSES = [
    case
    for case in _OLD_CHECK_PASSES
    if case.id != "garbage-line"  # the line check refuses it: that is its job
]


@pytest.mark.parametrize("key_file_text", _LINE_CHECK_PASSES)
def test_control_the_line_check_of_the_reviewed_head_passes_a_broken_file(
    tmp_path: Path, key_file_text: str
) -> None:
    """The error chain of OMN-25201: the line check lets the three shapes by (exit 0)."""
    old_argv = _argv_of(_OLD_LINE_CHECK, tmp_path)
    result = _run_start_check(tmp_path, key_file_text, argv=old_argv)

    assert result.returncode == 0, (result.returncode, result.stderr)


def test_control_the_line_check_of_the_reviewed_head_refuses_garbage(
    tmp_path: Path,
) -> None:
    """Positive control for the line-check control: it is a real check, not `true`."""
    old_argv = _argv_of(_OLD_LINE_CHECK, tmp_path)
    result = _run_start_check(tmp_path, "this is not an assignment\n", argv=old_argv)

    assert result.returncode == 2, (result.returncode, result.stderr)


def test_the_units_start_check_refuses_a_missing_key_file(tmp_path: Path) -> None:
    """No file at all (``key_file_text=None``): the check refuses, naming only the name."""
    result = _run_start_check(tmp_path, None)

    assert result.returncode == 2, (result.returncode, result.stdout)
    assert _CODE_NAME in result.stderr, _CODE_NAME


def test_the_units_start_check_passes_a_good_key_file(tmp_path: Path) -> None:
    """The guard must not disturb the good path: a real dev key exits 0."""
    result = _run_start_check(tmp_path, f"{_CODE_NAME}={_DEV_SYNTHETIC}\n")

    assert result.returncode == 0, (result.returncode, result.stderr)
    assert _CODE_NAME not in result.stderr, _CODE_NAME


def test_the_units_start_check_passes_a_key_with_dash_and_underscore(
    tmp_path: Path,
) -> None:
    """Positive control for the line check: the one line, a key with `-` and `_`."""
    result = _run_start_check(tmp_path, f"{_CODE_NAME}=Ab-9_zY\n")

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
        # The last (indented) assignment wins: the agent signs with the placeholder.
        pytest.param(
            f"DEPLOY_AGENT_HMAC_SECRET={_DEV_SYNTHETIC}\n"
            f"  DEPLOY_AGENT_HMAC_SECRET={_PLACEHOLDER}\n",
            _PLACEHOLDER,
            id="good-line-then-indented-placeholder",
        ),
        # The comment swallows the key line: the name is never set, the store wins.
        pytest.param(
            f"# delivered by the host installer \\\n{_CODE_NAME}={_DEV_SYNTHETIC}\n",
            _SHARED_SYNTHETIC,
            id="backslash-comment-before-the-key-line",
        ),
        # The protected list is cut down by the file, so the store's value wins.
        pytest.param(
            f"{_CODE_NAME}={_DEV_SYNTHETIC}\n{_PROTECTED_NAME}=PATH\n",
            _SHARED_SYNTHETIC,
            id="extra-name-overriding-the-units-environment",
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

#: Directories no scan walks: VCS metadata, virtualenvs, caches, build output.
_SKIPPED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".onex_state",
        ".tox",
        "site-packages",
    }
)


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


def _tracked_files(root: Path, suffixes: tuple[str, ...] | None) -> list[Path]:
    """Every file under ``root`` with one of ``suffixes`` (any file when ``None``)."""
    found: list[Path] = []
    for directory, subdirs, names in os.walk(root):
        subdirs[:] = sorted(name for name in subdirs if name not in _SKIPPED_DIRS)
        found.extend(
            Path(directory) / name
            for name in sorted(names)
            if suffixes is None or name.endswith(suffixes)
        )
    return found


def _env_file_keys(node: object) -> list[str]:
    """Every ``env_file`` key anywhere in a parsed document, as text.

    A recursive walk, not a read of ``services.<name>.env_file``: an ``env_file``
    under an ``x-`` extension field, an anchor or a nested block loads the store just
    the same once compose merges it.
    """
    entries: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "env_file":
                for item in value if isinstance(value, list) else [value]:
                    entries.append(
                        str(item.get("path", "") if isinstance(item, dict) else item)
                    )
            entries.extend(_env_file_keys(value))
    elif isinstance(node, list):
        for value in node:
            entries.extend(_env_file_keys(value))
    return entries


def _env_file_entries(text: str) -> list[str]:
    """Every ``env_file`` entry of one YAML file; an unparseable file is scanned as text."""
    try:
        documents = list(yaml.load_all(text, Loader=_ComposeLoader))
    except yaml.YAMLError:
        return [
            match.group(0).strip()
            for match in re.finditer(r"^\s*env_file\s*:.*$", text, re.MULTILINE)
        ]
    return [entry for document in documents for entry in _env_file_keys(document)]


def _env_file_findings(root: Path) -> list[tuple[str, str]]:
    """``(relative path, entry)`` for every ``env_file`` entry in every YAML file."""
    return [
        (str(path.relative_to(root)), entry)
        for path in _tracked_files(root, (".yml", ".yaml"))
        for entry in _env_file_entries(path.read_text(encoding="utf-8"))
    ]


#: Every ``env_file`` entry the repository may carry, as ``(path, entry)``. None
#: today: an entry here would load a file around the agent's environment and so
#: could carry the shared key to the dev runtime's containers.
_ALLOWED_ENV_FILES: frozenset[tuple[str, str]] = frozenset()


def test_no_yaml_file_declares_an_env_file() -> None:
    """Every YAML file in the repository, not only ``docker/``, and every entry."""
    yaml_files = _tracked_files(_REPO_ROOT, (".yml", ".yaml"))
    # A glob that finds nothing would pass vacuously.
    assert len(yaml_files) >= 100, "YAML files in the repository"
    names = {str(path.relative_to(_REPO_ROOT)) for path in yaml_files}
    assert "docker/docker-compose.infra.yml" in names
    # ...and files outside docker/ are in the scan.
    assert any(not name.startswith("docker/") and "/" in name for name in names)

    findings = [
        finding
        for finding in _env_file_findings(_REPO_ROOT)
        if finding not in _ALLOWED_ENV_FILES
    ]
    assert findings == [], findings


@pytest.mark.parametrize(
    ("relative", "declared"),
    [
        pytest.param(
            "ops/stack.yml", "      - ${HOME}/.env", id="outside-docker-home-env"
        ),
        pytest.param(
            "docker/docker-compose.x.yml",
            "      - ${HOME}/.omnibase/.env",
            id="operator-store",
        ),
        pytest.param(
            "docker/docker-compose.x.yml",
            "      - path: ${DEPLOY_AGENT_ENV_FILE}\n        required: true",
            id="long-form-variable",
        ),
        pytest.param("deploy/stack.yml", "      - ./local.env", id="any-other-file"),
    ],
)
def test_control_an_env_file_entry_anywhere_is_found(
    tmp_path: Path, relative: str, declared: str
) -> None:
    """The failing control: the scan is not blind to what it forbids."""
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        f"services:\n  runtime:\n    image: x\n    env_file:\n{declared}\n"
    )
    findings = _env_file_findings(tmp_path)
    assert [path for path, _ in findings] == [relative], findings


def test_control_an_env_file_in_an_extension_field_is_found(tmp_path: Path) -> None:
    (tmp_path / "stack.yaml").write_text(
        "x-common: &common\n  env_file:\n    - ${HOME}/.env\n"
        "services:\n  runtime:\n    <<: *common\n    image: x\n"
    )
    assert _env_file_findings(tmp_path) != []


def test_control_a_service_without_an_env_file_is_not_a_finding(tmp_path: Path) -> None:
    (tmp_path / "stack.yml").write_text(
        "services:\n  runtime:\n    image: x\n    environment:\n      A: b\n"
    )
    assert _env_file_findings(tmp_path) == []


# ``--env-file`` on a compose command loads a file under the agent's environment, so
# the same pin covers it. The scan reads every source line of every file -- whatever
# its suffix, extensionless scripts included -- so it does not depend on how the argv
# is written: a split list, a tuple, ``extend``, ``--env-file=<path>``, a
# ``self._run("compose", ...)`` call, a shell line. Its scope is the directories the
# deploy agent runs code from (its own package and scripts, the runtime build scripts
# it invokes, the handlers it routes to) and the single scripts elsewhere in
# ``scripts/`` that its source names (checked below, so a script the agent starts to
# run cannot sit outside the scan). Each hit is pinned by its LINE TEXT with the
# command it belongs to: swapping a listed line for another ``--env-file`` fails.
#
# What this scan does NOT read (the unit header says the same): the overlays
# repository, which the agent reads as its overlay home and which holds
# ``env_file`` keys of host installer units and tools, and ``onex-runtime-deploy``,
# a console script of omnimarket (an extensionless file in the lab virtualenv, not a
# file of this repository).
_ARGV_ROOTS = (
    "scripts/deploy-agent",
    "scripts/runtime_build",
    "src/omnibase_infra/handlers",
)
_ARGV_FILES = (
    "scripts/check_prod_promotion_lineage.py",
    "scripts/ci/check_lab_probe_windows.py",
    "scripts/ci/lab_pass_receipt.py",
    "scripts/deploy-gateway.sh",
    "scripts/preflight_required_compose_env.py",
    "scripts/render_runtime_policy_env.py",
    "scripts/run-forward-migrations.sh",
    "scripts/seed-infisical.py",
    "scripts/trigger_rebuild_on_merge.py",
)

#: ``(path, stripped line text) -> the command the line belongs to``. Adding a line to a
#: listed file, editing a listed line, or an ``--env-file`` in any other file in scope,
#: fails the scan until the command is traced and listed here.
_ENV_FILE_ARGV_ALLOWED: dict[tuple[str, str], str] = {
    ("scripts/deploy-agent/deploy_agent/executor.py", '"--env-file",'): (
        "`python3 <REPOINT_ONEX_API_SCRIPT> --env-file <store>`: the onex-api pin repoint script, not a compose command (traced by the AST test below)"
    ),
    (
        "scripts/preflight_required_compose_env.py",
        'command = ["docker", "compose", "--env-file", str(policy_env)]',
    ): (
        "`docker compose --env-file <runtime policy env> -f <files> --profile runtime config --format json`: a render of the dev broker stack with the tracked docker/runtime-policy.env, which the agent passes as --runtime-policy-env (no signing key in it, pinned below); it starts nothing"
    ),
    (
        "scripts/runtime_build/prepr_pool_prove.sh",
        'uv run -q python -m omnibase_infra.docker.catalog.cli generate "$BUNDLE" --env-file "$W/local.env" 2>&1 | tail -2',
    ): (
        "pre-PR slot proof: the slot's own generated local.env, run by a lane, not by the deploy agent"
    ),
    (
        "scripts/runtime_build/prepr_pool_prove.sh",
        'docker compose -f docker/docker-compose.generated.yml --env-file docker/runtime-policy.env --env-file "$W/local.env" build --build-arg BUILD_SOURCE=workspace --build-arg FIRST_PARTY_BUILD_CONTEXT="$(python3 -c "import importlib.util as u, sys; s = u.spec_from_file_location(\'c\', sys.argv[1]); m = u.module_from_spec(s); sys.modules[\'c\'] = m; s.loader.exec_module(m); print(m.load_contract().pre_pr_slot_context)" src/omnibase_infra/validation/checker_first_party_sources.py)" --build-arg EXPECTED_BUILD_SOURCE=workspace --build-arg OMNI_HOME="$R" --build-arg PROMOTION_CLASS=stability-candidate --build-arg NON_MAIN_LINEAGE=true --build-arg GIT_SHA="$(git rev-parse HEAD)" --build-arg VCS_REF="$(git rev-parse HEAD)" --build-arg RUNTIME_VERSION="$RV" --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$W/build.log" 2>&1',
    ): (
        "pre-PR slot proof: the slot's own generated local.env, run by a lane, not by the deploy agent"
    ),
    (
        "scripts/runtime_build/prepr_pool_prove.sh",
        'uv run -q python -m omnibase_infra.docker.catalog.cli up "$BUNDLE" --env-file "$W/local.env" $COMPOSED_FLAG > "$W/up.log" 2>&1',
    ): (
        "pre-PR slot proof: the slot's own generated local.env, run by a lane, not by the deploy agent"
    ),
    (
        "scripts/runtime_build/prepr_verify_lane.sh",
        'ONEX_DB_SLOT="${DB_SLOT}" bash "${PROVISION_DB}" --apply --env-file "${PROVISIONER_ENV}" \\',
    ): ("the slot's DB provisioner flag, run by a lane, not by the deploy agent"),
    (
        "scripts/runtime_build/prepr_verify_rendered_slot.py",
        "an interpolation from the OS environment first and from ``--env-file`` only",
    ): ("a docstring about compose's `--env-file` precedence"),
    (
        "scripts/runtime_build/refresh_dev_lane.sh",
        '--env-file "${OMNIBASE_OPERATOR_ENV_FILE}" \\',
    ): ("the same repoint script, not a compose command"),
    (
        "scripts/runtime_build/repoint_dev_lane_onex_api.py",
        "NO DEFAULTS (rule 8). ``--env-file`` and ``--omninode-clone`` are both required.",
    ): ("the repoint script's own required `--env-file` flag (docstring)"),
    (
        "scripts/runtime_build/repoint_dev_lane_onex_api.py",
        'msg = f"--env-file {env_file} does not exist; nothing was written."',
    ): ("the repoint script's own required `--env-file` flag (error message)"),
    ("scripts/runtime_build/repoint_dev_lane_onex_api.py", '"--env-file",'): (
        "the repoint script's own required `--env-file` flag (argparse)"
    ),
    (
        "src/omnibase_infra/handlers/handler_dev_lane_pin.py",
        "--env-file <store> --key ONEX_CLOUD_MIGRATE_IMAGE",
    ): (
        "the dev lane pin tool's own `--env-file` flag, not a compose command (docstring)"
    ),
    (
        "src/omnibase_infra/handlers/handler_dev_lane_pin.py",
        "--env-file <store> --key ONEX_CLOUD_MIGRATE_IMAGE --value <image:tag> \\\\",
    ): (
        "the dev lane pin tool's own `--env-file` flag, not a compose command (docstring)"
    ),
    (
        "src/omnibase_infra/handlers/handler_dev_lane_pin.py",
        'sub.add_argument("--env-file", required=True, help="operator env store")',
    ): (
        "the dev lane pin tool's own `--env-file` flag, not a compose command (argparse)"
    ),
    (
        "src/omnibase_infra/handlers/handler_standalone_compose_recreate.py",
        '"--env-file",',
    ): (
        "`docker compose --file <declared file> --env-file <env_file> up --detach --no-deps --force-recreate -- <service>`: the standalone recreate, whose node contract declares one target, factory-floor-web (docker/docker-compose.factory-floor-web.yml), with the env file named by DEPLOY_AGENT_FACTORY_FLOOR_WEB_ENV_FILE; its own test is below"
    ),
}


def _scanned_files(root: Path) -> list[Path]:
    """Every text file in scope: the roots (any suffix) and the named single scripts."""
    paths = [
        path for scope in _ARGV_ROOTS for path in _tracked_files(root / scope, None)
    ]
    paths.extend(root / name for name in _ARGV_FILES if (root / name).is_file())
    return paths


def _env_file_argv_lines(root: Path) -> list[tuple[str, str]]:
    """``(path, line)`` of every non-comment line naming ``--env-file`` in the scan scope."""
    hits: list[tuple[str, str]] = []
    for path in _scanned_files(root):
        relative = path.relative_to(root)
        # Tests quote the literal on purpose.
        if "tests" in relative.parts:
            continue
        raw = path.read_bytes()
        if b"\0" in raw[:1024]:
            continue
        hits.extend(
            (str(relative), line.strip())
            for line in raw.decode("utf-8", errors="replace").splitlines()
            if "--env-file" in line and not line.lstrip().startswith("#")
        )
    return sorted(hits)


def _unlisted(hits: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [hit for hit in hits if hit not in _ENV_FILE_ARGV_ALLOWED]


def test_no_deploy_agent_compose_argv_passes_env_file() -> None:
    hits = _env_file_argv_lines(_REPO_ROOT)
    # The scan must see the one existing hit in the agent package, or it proves nothing.
    assert any(path.endswith("deploy_agent/executor.py") for path, _ in hits), hits
    assert _unlisted(hits) == []
    # A listed line that no longer exists is a stale entry, not a pass.
    assert sorted(_ENV_FILE_ARGV_ALLOWED) == hits, sorted(
        set(hits) ^ set(_ENV_FILE_ARGV_ALLOWED)
    )


def _scripts_named_by_the_agent() -> set[str]:
    """Repository scripts the agent's own source names, as ``scripts/...`` paths."""
    named: set[str] = set()
    sources = [
        *sorted((_REPO_ROOT / "scripts/deploy-agent/deploy_agent").glob("*.py")),
        *sorted((_REPO_ROOT / "scripts/deploy-agent/deploy").glob("*.sh")),
    ]
    assert len(sources) >= 10, "the agent's source files"
    for source in sources:
        text = source.read_text(encoding="utf-8", errors="replace")
        named.update(re.findall(r"scripts/[A-Za-z0-9_./-]+\.(?:sh|py)\b", text))
        for chain in re.findall(r'"scripts"(?:\s*/\s*"[^"]+")+', text):
            named.add("/".join(re.findall(r'"([^"]+)"', chain)))
    return {name for name in named if (_REPO_ROOT / name).is_file()}


def _outside_the_scan(names: set[str]) -> list[str]:
    return sorted(
        name
        for name in names
        if name not in _ARGV_FILES
        and not any(name.startswith(f"{scope}/") for scope in _ARGV_ROOTS)
    )


def test_every_script_the_agent_names_is_inside_the_scan() -> None:
    """The roots are what the agent runs: a script it names that is not scanned fails."""
    names = _scripts_named_by_the_agent()
    assert "scripts/runtime_build/refresh_dev_lane.sh" in names, sorted(names)
    assert "scripts/preflight_required_compose_env.py" in names, sorted(names)
    assert _outside_the_scan(names) == []
    for name in _ARGV_FILES:
        assert (_REPO_ROOT / name).is_file(), name


def test_control_a_script_the_agent_names_outside_the_scan_is_found() -> None:
    assert _outside_the_scan({"scripts/brand_new_tool.sh"}) == [
        "scripts/brand_new_tool.sh"
    ]


def test_the_executor_env_file_argv_is_the_repoint_script_and_no_compose_command() -> (
    None
):
    """Trace the one hit in the deploy-agent package to its command."""
    source = (_REPO_ROOT / "scripts/deploy-agent/deploy_agent/executor.py").read_text()
    lists = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.List)
        and any(
            isinstance(element, ast.Constant) and element.value == "--env-file"
            for element in node.elts
        )
    ]
    assert len(lists) == 1, "list literals naming --env-file"
    (argv,) = lists
    words = [
        element.value
        for element in argv.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    ]
    assert words[0] == "python3", words
    assert "compose" not in words and "docker" not in words, words
    # The script it runs is the repoint script, named in the list by its constant.
    assert any(
        isinstance(element, ast.Call)
        and ast.unparse(element) == "str(REPOINT_ONEX_API_SCRIPT)"
        for element in argv.elts
    )


@pytest.mark.parametrize(
    ("relative", "text"),
    [
        pytest.param(
            "scripts/deploy-agent/deploy_agent/new.py",
            'ARGV = ["docker", "compose", "--env-file", "x", "up"]\n',
            id="top-level-compose-list",
        ),
        pytest.param(
            "scripts/deploy-agent/deploy_agent/new.py",
            'ARGV = ["docker", "compose"]\nARGV += ["--env-file",\n  "x"]\n',
            id="split-argv-extend",
        ),
        pytest.param(
            "scripts/deploy-agent/deploy_agent/new.py",
            'ARGV = ("run", "--env-file=x")\n',
            id="tuple-with-equals",
        ),
        pytest.param(
            "scripts/deploy-agent/deploy_agent/new.py",
            'cmd = "docker compose --env-file x up"\n',
            id="shell-string",
        ),
        pytest.param(
            "scripts/deploy-agent/deploy/new.sh",
            "docker compose --env-file x up\n",
            id="shell-script",
        ),
        pytest.param(
            "scripts/runtime_build/onex-new-tool",
            "#!/bin/sh\ndocker compose --env-file x up\n",
            id="extensionless-script",
        ),
        pytest.param(
            "scripts/deploy-gateway.sh",
            "docker compose --env-file x up\n",
            id="named-single-script-outside-the-roots",
        ),
    ],
)
def test_control_an_env_file_argv_in_any_form_is_found(
    tmp_path: Path, relative: str, text: str
) -> None:
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    hits = _env_file_argv_lines(tmp_path)
    assert [path for path, _ in hits] == [relative], hits
    assert _unlisted(hits) == hits, hits


def test_control_swapping_an_allowed_line_for_a_compose_env_file_is_found(
    tmp_path: Path,
) -> None:
    """The allow-list pins line text: a listed file with a swapped line is a finding.

    ``prepr_verify_lane.sh`` is listed with one line; here the same file carries a
    compose ``--env-file`` on the operator store instead. A count per file would pass
    it (one hit, one listed).
    """
    listed = "scripts/runtime_build/prepr_verify_lane.sh"
    assert sum(1 for path, _ in _ENV_FILE_ARGV_ALLOWED if path == listed) == 1
    target = tmp_path / listed
    target.parent.mkdir(parents=True)
    target.write_text("docker compose --env-file ${OPERATOR_STORE} up\n")
    hits = _env_file_argv_lines(tmp_path)
    assert [path for path, _ in hits] == [listed], hits
    assert _unlisted(hits) == hits, hits


def test_control_a_comment_naming_env_file_is_not_a_hit(tmp_path: Path) -> None:
    """The comment skip: a comment line inside the scan scope is not a hit.

    The file sits under ``scripts/deploy-agent/deploy/`` -- inside the scan roots --
    so the empty result comes from the comment skip alone, and the same text without
    its ``#`` is a hit (the positive control right below).
    """
    target = tmp_path / "scripts/deploy-agent/deploy/run.sh"
    target.parent.mkdir(parents=True)
    target.write_text("# docker compose --env-file x up\n")
    assert target.relative_to(tmp_path).as_posix().startswith(_ARGV_ROOTS[0] + "/")
    assert _env_file_argv_lines(tmp_path) == []

    target.write_text("docker compose --env-file x up\n")
    assert _env_file_argv_lines(tmp_path) != []


# --------------------------------------------------------------------------
# The two compose ``--env-file`` argv in scope that are real compose commands: each
# is traced to the file it passes, and the files themselves are pinned not to carry
# the signing key (OMN-25201 review).
# --------------------------------------------------------------------------

_STANDALONE_CONTRACT = (
    _REPO_ROOT
    / "src/omnibase_infra/nodes/node_standalone_compose_recreate_effect/contract.yaml"
)
_STANDALONE_HANDLER = (
    _REPO_ROOT / "src/omnibase_infra/handlers/handler_standalone_compose_recreate.py"
)
_FACTORY_FLOOR_COMPOSE = "docker/docker-compose.factory-floor-web.yml"
_FACTORY_FLOOR_ENV_VARIABLE = "DEPLOY_AGENT_FACTORY_FLOOR_WEB_ENV_FILE"


def _standalone_targets_are_only_the_factory_floor(
    targets: list[dict[str, str]],
) -> bool:
    """One target, the factory floor page's own compose file, its own env-file variable."""
    return (
        len(targets) == 1
        and targets[0]["compose_file"] == _FACTORY_FLOOR_COMPOSE
        and targets[0]["service"] == "factory-floor-web"
        and targets[0]["env_file_variable"] == _FACTORY_FLOOR_ENV_VARIABLE
    )


def test_the_standalone_recreate_env_file_is_the_one_targets_own_file() -> None:
    """handler_standalone_compose_recreate.py: ``--env-file`` is not the operator store.

    The argv takes the env file from the declared target's variable, and the node's
    contract declares exactly one target, the factory floor page; the dev runtime's
    compose file is not a target, and the factory floor compose file never names the
    signing key, so no env file can carry it there.
    """
    contract = yaml.safe_load(_STANDALONE_CONTRACT.read_text(encoding="utf-8"))
    assert _standalone_targets_are_only_the_factory_floor(
        contract["standalone_targets"]
    )
    handler = _STANDALONE_HANDLER.read_text(encoding="utf-8")
    # The env file is resolved from the target's variable and refused when undeclared...
    assert "env_file = docker.resolve_env_file(target.env_file_variable)" in handler
    # ...the request's compose file must be one the contract declares...
    assert (
        "request.compose_file not in {t.compose_file for t in policy.targets}"
        in handler
    )
    # ...and the one compose argv passes that resolved value, nothing else.
    assert (
        "docker.recreate_service(request.compose_file, env_file, request.service)"
        in handler
    )
    # The unit does not point the variable at the operator store.
    assert not any(
        _FACTORY_FLOOR_ENV_VARIABLE in line for line in _directive_lines(_DEV_UNIT)
    )
    compose = (_REPO_ROOT / _FACTORY_FLOOR_COMPOSE).read_text(encoding="utf-8")
    assert _CODE_NAME not in compose, _CODE_NAME
    assert _DEV_STORE_NAME not in compose, _DEV_STORE_NAME


def test_control_a_second_standalone_target_on_the_dev_runtime_is_found() -> None:
    target = {
        "compose_file": _FACTORY_FLOOR_COMPOSE,
        "service": "factory-floor-web",
        "env_file_variable": _FACTORY_FLOOR_ENV_VARIABLE,
    }
    assert _standalone_targets_are_only_the_factory_floor([target])
    dev_runtime = {**target, "compose_file": "docker/docker-compose.infra.yml"}
    assert not _standalone_targets_are_only_the_factory_floor([target, dev_runtime])
    assert not _standalone_targets_are_only_the_factory_floor([dev_runtime])
    store = {**target, "env_file_variable": "DEPLOY_AGENT_ENV_FILE"}
    assert not _standalone_targets_are_only_the_factory_floor([store])


def test_the_runtime_policy_env_file_carries_no_signing_key() -> None:
    """preflight_required_compose_env.py renders with ``--env-file`` on this file only."""
    policy = (_REPO_ROOT / "docker/runtime-policy.env").read_text(encoding="utf-8")
    assert "HMAC" not in policy, _CODE_NAME
    assert _CODE_NAME not in policy, _CODE_NAME
    executor = (_REPO_ROOT / "scripts/deploy-agent/deploy_agent/executor.py").read_text(
        encoding="utf-8"
    )
    assert (
        '"--runtime-policy-env",\n            f"{deploy_repo_dir()}/docker/runtime-policy.env",'
        in executor
    )


#: ``DEPLOY_AGENT_HMAC_SECRET: ${DEPLOY_AGENT_HMAC_SECRET...}`` -- the runtime's
#: containers take the signing key from the agent's environment by interpolation.
_COMPOSE_INTERPOLATES_THE_NAME = re.compile(
    r"^\s*DEPLOY_AGENT_HMAC_SECRET:\s*\$\{DEPLOY_AGENT_HMAC_SECRET[:?}-]", re.MULTILINE
)


def test_the_dev_runtime_compose_file_interpolates_the_signing_key_name() -> None:
    """The dev key reaches the runtime's containers only through this interpolation."""
    compose = (_REPO_ROOT / "docker/docker-compose.infra.yml").read_text(
        encoding="utf-8"
    )
    assert len(_COMPOSE_INTERPOLATES_THE_NAME.findall(compose)) == 1


@pytest.mark.parametrize(
    "line",
    [
        pytest.param("  DEPLOY_AGENT_HMAC_SECRET: literal-value\n", id="literal"),
        pytest.param("  DEPLOY_AGENT_HMAC_SECRET: ${OTHER_NAME:-}\n", id="other-name"),
        pytest.param(
            "  DEPLOY_AGENT_HMAC_SECRET: ${DEPLOY_AGENT_DEV_HMAC_SECRET}\n",
            id="store-name",
        ),
        pytest.param(
            "  # DEPLOY_AGENT_HMAC_SECRET: ${DEPLOY_AGENT_HMAC_SECRET:?x}\n",
            id="comment",
        ),
    ],
)
def test_control_the_interpolation_pin_rejects_anything_else(line: str) -> None:
    assert _COMPOSE_INTERPOLATES_THE_NAME.findall(line) == []


def test_the_unit_header_narrows_the_env_file_claim_to_what_is_scanned() -> None:
    """The claim names its scope: this repository only; overlays and onex-runtime-deploy apart."""
    header = _flat(_DEV_UNIT.read_text().split("\n[Unit]", 1)[0].splitlines())
    assert "scans read this repository only" in header
    assert "overlays repository" in header
    assert "onex-runtime-deploy" in header
