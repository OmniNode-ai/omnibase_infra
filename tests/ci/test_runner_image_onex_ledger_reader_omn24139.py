# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runner image carries the ``onex`` ledger query client (OMN-24139).

The operator repository's freeze step reads the live work ledger through
``onex ledger query --stdin --bus-lane dev`` and died with FileNotFoundError on
the build runners because the image held no ``onex``. This pins three things:

* the image installs the published ``omnibase-infra`` package at an exact version
  into its own virtual environment, and puts a shim named ``onex`` on the PATH;
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
    r"^COPY --chmod=0755 <<'ONEX_LEDGER_SHIM' /usr/local/bin/onex\n(?P<body>.*?)^ONEX_LEDGER_SHIM$",
    re.DOTALL | re.MULTILINE,
)


def _shim_text() -> str:
    """The shim exactly as the image's Dockerfile installs it as /usr/local/bin/onex."""
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
    assert "/usr/local/bin/onex" in code, "the shim is the onex on the PATH"
    assert "onex --version" in code, (
        "the build fails if the installed client does not run"
    )
    assert code.index("FROM toolchain AS runner") < code.index("omnibase-infra==")


def test_lock_image_version_carries_the_ledger_client() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    assert lock["image_version"] >= LEDGER_READER_IMAGE_VERSION, (
        "the onex ledger client is baked from image v15 (OMN-24139)"
    )
