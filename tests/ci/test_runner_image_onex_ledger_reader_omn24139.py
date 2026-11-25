# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runner image carries the ``onex`` ledger query client (OMN-24139).

The operator repository's freeze step reads the live work ledger through
``onex ledger query --stdin --bus-lane dev`` and died with FileNotFoundError on
the build runners because the image held no ``onex``. This pins three things:

* the image installs the published ``omnibase-infra`` package at an exact version
  into its own virtual environment, and keeps a shim named ``onex`` in a directory that is not on the default PATH
  (OMN-24562: the customer checks fail a job that finds ``onex`` on its PATH);
* the shim hands the mounted read-only credential (``KAFKA_SASL_USERNAME`` and
  ``KAFKA_SASL_PASSWORD`` from ``ci-ledger-read.env``) to the reader command and
  to nothing else, parses the file as data rather than running it, never echoes a
  value, and refuses with a nonzero exit when the file is absent;
* the lock's image version moved, so the pool roll sees a new image.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNERS = REPO_ROOT / "docker" / "runners"
DOCKERFILE = RUNNERS / "Dockerfile"
HEREDOC = re.compile(
    r"^COPY --chmod=0755 <<'ONEX_LEDGER_SHIM' /opt/onex-ledger-client/shim/onex\n(?P<body>.*?)^ONEX_LEDGER_SHIM$",
    re.DOTALL | re.MULTILINE,
)


def _shim_text() -> str:
    """The shim exactly as the image's Dockerfile installs it as /opt/onex-ledger-client/shim/onex."""
    found = HEREDOC.search(DOCKERFILE.read_text(encoding="utf-8"))
    assert found, "the Dockerfile must install the shim as a quoted COPY heredoc"
    return found.group("body")


LOCK = RUNNERS / "runner-image.lock.json"

LEDGER_READER_IMAGE_VERSION = 15

USER_VALUE = "-".join(["ledger", "reader", "user", "0001"])
PASS_VALUE = "-".join(["value", "that", "must", "never", "print", "9f2c"])


def _fake_real_onex(tmp_path: Path) -> tuple[Path, Path]:
    """A stand-in for the installed client: records what it was handed, prints no value."""
    record = tmp_path / "record.txt"
    fake = tmp_path / "real-onex"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        f'record="{record}"\n'
        'printf "argv=%s\\n" "$*" > "$record"\n'
        'printf "user=%s\\n" "${KAFKA_SASL_USERNAME-<unset>}" >> "$record"\n'
        'printf "pass=%s\\n" "${KAFKA_SASL_PASSWORD-<unset>}" >> "$record"\n'
        'printf "config_root=%s\\n" "${ONEX_WORKSPACE_CONFIG_ROOT-<unset>}" >> "$record"\n'
        'printf "workspace=%s\\n" "${OMNIBASE_PATH-<unset>}" >> "$record"\n'
        'printf "{\\"ok\\": true}\\n"\n',
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    return fake, record


def _run(
    tmp_path: Path,
    args: list[str],
    credential: str | None,
    *,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    fake, record = _fake_real_onex(tmp_path)
    cred_path = tmp_path / "ci-ledger-read.env"
    if credential is not None:
        cred_path.write_text(credential, encoding="utf-8")
    env = {
        "PATH": os.environ["PATH"],
        "ONEX_LEDGER_CLIENT_BIN": str(fake),
        "ONEX_LEDGER_READ_CREDENTIAL_FILE": str(cred_path),
        "GITHUB_WORKSPACE": str(tmp_path),
    }
    env.update(extra_env or {})
    shim = tmp_path / "onex"
    shim.write_text(_shim_text(), encoding="utf-8")
    done = subprocess.run(
        ["bash", str(shim), *args],
        capture_output=True,
        text=True,
        env=env,
        input="{}",
        timeout=30,
        check=False,
    )
    return done, record


AMBIENT_VALUE = "-".join(["ambient", "other", "identity"])
GOOD_FILE = f"KAFKA_SASL_USERNAME={USER_VALUE}\nKAFKA_SASL_PASSWORD={PASS_VALUE}\n"
READER = ["ledger", "query", "--stdin", "--bus-lane", "dev"]


def test_shim_is_installed_from_the_dockerfile_and_is_safe_source() -> None:
    text = _shim_text()
    assert text.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in text
    assert "set -x" not in text, "xtrace would print the credential"


def test_reader_command_refuses_nonzero_when_credential_file_is_absent(
    tmp_path: Path,
) -> None:
    done, record = _run(tmp_path, READER, credential=None)
    assert done.returncode != 0, "the reader must refuse without the mounted credential"
    assert not record.exists(), (
        "the real client must not run when the credential is absent"
    )
    assert "ci-ledger-read.env" in done.stderr, "the refusal names the missing file"


def test_reader_command_gets_the_credential_and_never_prints_it(tmp_path: Path) -> None:
    done, record = _run(tmp_path, READER, credential=GOOD_FILE)
    assert done.returncode == 0, done.stderr
    seen = record.read_text(encoding="utf-8")
    assert f"user={USER_VALUE}" in seen
    assert f"pass={PASS_VALUE}" in seen
    assert "argv=ledger query --stdin --bus-lane dev" in seen
    assert PASS_VALUE not in done.stdout + done.stderr
    assert USER_VALUE not in done.stdout + done.stderr


def test_reader_command_finds_the_lane_declaration_in_the_jobs_checkout(
    tmp_path: Path,
) -> None:
    """The client resolves config/lab/ci_bus_lanes.yaml under ONEX_WORKSPACE_CONFIG_ROOT."""
    done, record = _run(tmp_path, READER, credential=GOOD_FILE)
    assert done.returncode == 0, done.stderr
    seen = record.read_text(encoding="utf-8")
    assert f"config_root={tmp_path}" in seen
    assert f"workspace={tmp_path}" in seen, (
        "the client refuses a lane read with no workspace root"
    )


def test_an_explicit_config_root_is_kept(tmp_path: Path) -> None:
    done, record = _run(
        tmp_path,
        READER,
        GOOD_FILE,
        extra_env={"ONEX_WORKSPACE_CONFIG_ROOT": "/declared/root"},
    )
    assert done.returncode == 0, done.stderr
    assert "config_root=/declared/root" in record.read_text(encoding="utf-8")


def test_other_commands_pass_through_without_the_credential(tmp_path: Path) -> None:
    done, record = _run(tmp_path, ["--version"], credential=None)
    assert done.returncode == 0, done.stderr
    seen = record.read_text(encoding="utf-8")
    assert "argv=--version" in seen
    assert "user=<unset>" in seen
    assert "pass=<unset>" in seen


def test_cloud_ledger_read_does_not_receive_the_credential(tmp_path: Path) -> None:
    done, record = _run(
        tmp_path, ["ledger", "read", "--correlation-id", "x"], GOOD_FILE
    )
    assert done.returncode == 0, done.stderr
    seen = record.read_text(encoding="utf-8")
    assert "user=<unset>" in seen
    assert "pass=<unset>" in seen


def test_other_commands_keep_the_callers_own_environment(tmp_path: Path) -> None:
    done, record = _run(
        tmp_path,
        ["--version"],
        credential=GOOD_FILE,
        extra_env={"KAFKA_SASL_PASSWORD": AMBIENT_VALUE},
    )
    assert done.returncode == 0, done.stderr
    assert AMBIENT_VALUE in record.read_text(encoding="utf-8"), (
        "a non-reader command keeps the caller's own environment untouched"
    )


def test_credential_file_is_read_as_data_never_executed(tmp_path: Path) -> None:
    marker = tmp_path / "executed.marker"
    hostile = GOOD_FILE + f"touch {marker}\nKAFKA_SASL_X=$(touch {marker})\n"
    done, record = _run(tmp_path, READER, credential=hostile)
    assert done.returncode == 0, done.stderr
    assert not marker.exists(), "the credential file must never be sourced or evaluated"
    assert f"pass={PASS_VALUE}" in record.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "content",
    [
        f"KAFKA_SASL_USERNAME={USER_VALUE}\n",
        f"KAFKA_SASL_PASSWORD={PASS_VALUE}\n",
        "KAFKA_SASL_USERNAME=\nKAFKA_SASL_PASSWORD=\n",
        "",
    ],
)
def test_half_or_empty_credential_refuses_without_echoing_values(
    tmp_path: Path, content: str
) -> None:
    done, record = _run(tmp_path, READER, credential=content)
    assert done.returncode != 0
    assert not record.exists()
    assert PASS_VALUE not in done.stdout + done.stderr
    assert USER_VALUE not in done.stdout + done.stderr


def test_dockerfile_installs_published_client_pinned_and_ships_the_shim() -> None:
    source = DOCKERFILE.read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    ).replace("\\\n", " ")
    pin = re.search(r"ARG ONEX_LEDGER_CLIENT_VERSION=(\d+\.\d+\.\d+)\b", code)
    assert pin, "the client version is a pinned ARG, never a floating latest"
    assert re.search(r"omnibase-infra==\$\{ONEX_LEDGER_CLIENT_VERSION\}", code), (
        "the image installs the published omnibase-infra at the pinned version"
    )
    assert "/opt/onex-ledger-client" in code
    assert "/opt/onex-ledger-client/shim/onex" in code
    assert "onex --version" in code, (
        "the build fails if the installed client does not run"
    )
    assert code.index("FROM toolchain AS runner") < code.index("omnibase-infra==")


def test_lock_image_version_carries_the_ledger_client() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    assert lock["image_version"] >= LEDGER_READER_IMAGE_VERSION, (
        "the onex ledger client is baked from image v15 (OMN-24139)"
    )


def test_dockerfile_puts_no_onex_in_a_default_path_directory() -> None:
    """OMN-24562: the shim file is off every default PATH dir; only a link to it is on one."""
    code = "\n".join(
        line
        for line in DOCKERFILE.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )
    assert (
        "COPY --chmod=0755 <<'ONEX_LEDGER_SHIM' /opt/onex-ledger-client/shim/onex"
        in code
    )
    for default_dir in ("/usr/local/bin", "/usr/bin", "/bin", "/usr/local/sbin"):
        for hit in re.finditer(rf"(?<![\w/.-]){re.escape(default_dir)}/onex\b", code):
            line = code[
                code.rfind("\n", 0, hit.start()) + 1 : code.find("\n", hit.end())
            ]
            assert default_dir == "/usr/local/bin" and line.strip().startswith(
                "RUN ln -s /opt/onex-ledger-client/shim/onex "
            ), line
    assert not re.search(r"ENV PATH=[^\n]*onex-ledger-client", code), (
        "the image-wide PATH must not carry the shim directory"
    )


# A bare ``onex`` word at the start of a command inside a workflow or composite action.
# Off the default PATH it resolves to nothing on a runner, so every such caller must name
# the shim path (/opt/onex-ledger-client/shim/onex), add its directory to its own PATH in
# the job, or run the checkout's own entry point (``uv run onex``, ``.venv/bin/...``).
_BARE_ONEX = re.compile(
    r"(?:^|[|;&(]|\$\(|`)\s*(?:-\s+)?(?:run:\s*)?(?:sudo\s+)?onex(?:\s|$)", re.MULTILINE
)


def _bare_onex_lines(text: str) -> list[str]:
    """Non-comment lines of a workflow or action that run ``onex`` by bare name."""
    hits: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        if _BARE_ONEX.search(line):
            hits.append(line.strip())
    return hits


def test_bare_onex_scan_finds_a_caller_that_runs_onex_by_name() -> None:
    """Positive control: the scan flags the shapes it must refuse and passes the sanctioned ones."""
    assert _bare_onex_lines("      - run: onex ledger query --stdin --bus-lane dev")
    assert _bare_onex_lines("        echo x | onex ledger query --stdin")
    assert _bare_onex_lines("        out=$(onex --version)")
    assert not _bare_onex_lines("        uv run onex ledger query")
    assert not _bare_onex_lines(
        "        /opt/onex-ledger-client/shim/onex ledger query"
    )
    assert not _bare_onex_lines(
        '        PATH="/opt/onex-ledger-client/shim:$PATH" onex ledger query'
    )
    assert not _bare_onex_lines("      # onex ledger query is read here")


def test_no_workflow_or_action_runs_onex_by_bare_name() -> None:
    """With the shim off the default PATH, no caller in this repository may rely on it by name."""
    offenders: list[str] = []
    for root in (
        REPO_ROOT / ".github" / "workflows",
        REPO_ROOT / ".github" / "actions",
    ):
        for path in sorted(root.rglob("*.y*ml")):
            for line in _bare_onex_lines(path.read_text(encoding="utf-8")):
                if "/opt/onex-ledger-client/shim" in line:
                    continue
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {line}")
    assert not offenders, offenders
