# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``handler_dev_lane_pin``: one typed writer for dev-lane image pins (OMN-18475).

On 2026-10-21 at 22:48Z ``ONEX_CLOUD_MIGRATE_IMAGE`` was changed in the dev
lane's operator env store by a one-off script, because no tool set it. These tests pin
the command that replaces that: it sets only registered dev pins, refuses prod,
writes one line with a backup, and never changes a pin without a ledger row.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.executor import (
    lane_config_for,
    operator_store_overlay,
    reset_operator_store_cache,
)

from omnibase_infra.enums.enum_dev_lane_pin_key import EnumDevLanePinKey
from omnibase_infra.handlers import handler_dev_lane_pin as dev_pin
from omnibase_infra.handlers.handler_dev_lane_pin import main

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
OLD = "registry.example/omninode-cloud-migrate:327620c2"
NEW = "local-build/omninode-cloud-migrate:e7d62566-20261021T192052Z"


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    reset_operator_store_cache()
    yield
    reset_operator_store_cache()


class _Host:
    """A store, a docker that knows which images are resident, and a ledger."""

    def __init__(
        self, tmp_path: Path, *, resident: set[str], ledger_exit: int = 0
    ) -> None:
        self.store = tmp_path / ".env"
        self.store.write_text(
            "# operator env store\n"
            "POSTGRES_HOST=db\n"
            f"ONEX_CLOUD_MIGRATE_IMAGE={OLD}\n"
            "ONEX_API_IMAGE=local-build/omnicloud-core:e7d62566-20261020T201624Z\n",
            encoding="utf-8",
        )
        self.store.chmod(0o600)
        self.ledger = tmp_path / "ledger.md"
        self.ledger.write_text("", encoding="utf-8")
        self.docker = tmp_path / "docker"
        allowed = " ".join(resident)
        self.docker.write_text(
            "#!/bin/sh\n"
            'ref="$5"\n'
            f"for r in {allowed}; do\n"
            '  if [ "$r" = "$ref" ]; then echo "sha256:$(printf %064d 7)"; exit 0; fi\n'
            "done\n"
            'echo "Error: No such image: $ref" >&2; exit 1\n',
            encoding="utf-8",
        )
        self.docker.chmod(0o755)
        self.ledger_cli = tmp_path / "onex-ledger"
        self.ledger_cli.write_text(
            "#!/bin/sh\n"
            f"if [ {ledger_exit} -ne 0 ]; then echo 'row grammar refused' >&2; exit {ledger_exit}; fi\n"
            'printf "%s\\n" "$3" >> "$1"\n',
            encoding="utf-8",
        )
        self.ledger_cli.chmod(0o755)

    def set(
        self,
        *extra: str,
        key: str = "ONEX_CLOUD_MIGRATE_IMAGE",
        value: str = NEW,
        lane: str = "dev",
    ) -> int:
        return main(
            [
                "set",
                "--env-file",
                str(self.store),
                "--key",
                key,
                "--value",
                value,
                "--lane",
                lane,
                "--ledger",
                str(self.ledger),
                "--ledger-command",
                str(self.ledger_cli),
                "--ledger-lane",
                "dev-deploy-agent-env-0140",
                "--ticket",
                "OMN-18475",
                "--docker",
                str(self.docker),
                *extra,
            ]
        )


def _output(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    return dict(json.loads(capsys.readouterr().out))


# --- refuses prod ---------------------------------------------------------------


def _rendered_keys(compose_file: Path) -> set[str]:
    text = compose_file.read_text(encoding="utf-8")
    return {key.value for key in EnumDevLanePinKey if "${" + key.value in text}


def test_every_dev_pin_is_rendered_by_the_dev_lane_and_by_no_governed_lane() -> None:
    """The dev lane's store is shared with the prod agent unit.

    So a dev pin stays dev only if no compose file of a governed lane reads
    it. This is the check that makes the shared file safe; a pin added to the
    registry that a governed lane reads fails here.
    """
    dev = [
        REPO_ROOT / "docker" / Path(f).name
        for f in lane_config_for(EnumRuntimeLane.DEV).compose_files
    ]
    assert set().union(*(_rendered_keys(f) for f in dev)) == {
        k.value for k in EnumDevLanePinKey
    }
    for lane in (lane for lane in EnumRuntimeLane if lane is not EnumRuntimeLane.DEV):
        for name in lane_config_for(lane).compose_files:
            governed = REPO_ROOT / "docker" / Path(name).name
            assert governed.is_file(), governed
            assert _rendered_keys(governed) == set(), (lane, governed)


@pytest.mark.parametrize("lane", ["prod", "PROD", "judge", ""])
def test_a_lane_other_than_dev_is_refused_and_nothing_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], lane: str
) -> None:
    host = _Host(tmp_path, resident={NEW})
    before = host.store.read_bytes()

    assert host.set("--execute", lane=lane) == dev_pin.EXIT_REFUSED

    out = _output(capsys)
    assert out["outcome"] == "REFUSED"
    assert "OMN-13418" in str(out["reason"])
    assert host.store.read_bytes() == before
    assert host.ledger.read_text() == ""


@pytest.mark.parametrize(
    ("key", "names"),
    [
        ("PROD_RUNTIME_IMAGE", "OMN-13418"),
        ("ESTIMATOR_FERNET_KEY", "not a dev-lane pin"),
        ("POSTGRES_HOST", "not a dev-lane pin"),
    ],
)
def test_only_registered_dev_pins_are_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], key: str, names: str
) -> None:
    host = _Host(tmp_path, resident={NEW})
    before = host.store.read_bytes()

    assert host.set("--execute", key=key) == dev_pin.EXIT_REFUSED

    assert names in str(_output(capsys)["reason"])
    assert host.store.read_bytes() == before


@pytest.mark.parametrize(
    "value",
    [
        "local-build/omninode-cloud-migrate",
        "local-build/omninode-cloud-migrate:latest",
        "a b",
        "x|y:1",
        "",
    ],
)
def test_a_value_that_is_not_a_fixed_image_reference_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    host = _Host(tmp_path, resident={value})
    before = host.store.read_bytes()

    assert host.set("--execute", value=value) == dev_pin.EXIT_REFUSED

    assert host.store.read_bytes() == before


def test_an_image_that_is_not_resident_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    host = _Host(tmp_path, resident=set())
    before = host.store.read_bytes()

    assert host.set("--execute") == dev_pin.EXIT_REFUSED

    assert "not resident" in str(_output(capsys)["reason"])
    assert host.store.read_bytes() == before


# --- the write ------------------------------------------------------------------


def test_set_changes_one_line_keeps_a_backup_and_writes_the_ledger_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    host = _Host(tmp_path, resident={NEW})
    before = host.store.read_text().splitlines()

    assert host.set("--execute") == dev_pin.EXIT_OK

    out = _output(capsys)
    assert out["outcome"] == "WRITTEN"
    assert (out["pin_before"], out["pin_after"]) == (OLD, NEW)
    after = host.store.read_text().splitlines()
    assert [
        i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b
    ] == [2]
    assert after[2] == f"ONEX_CLOUD_MIGRATE_IMAGE={NEW}"
    assert stat.S_IMODE(host.store.stat().st_mode) == 0o600
    assert Path(str(out["backup_path"])).read_text().splitlines() == before
    rows = host.ledger.read_text().splitlines()
    assert len(rows) == 1
    row = rows[0]
    assert " | STATUS | lane=dev-deploy-agent-env-0140 | ticket=OMN-18475 | " in row
    assert f"pin-before={OLD}" in row and f"pin-after={NEW}" in row
    assert str(tmp_path) not in row


def test_a_refused_ledger_row_restores_the_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No pin changes without the row that names it."""
    host = _Host(tmp_path, resident={NEW}, ledger_exit=65)
    before = host.store.read_bytes()

    assert host.set("--execute") == dev_pin.EXIT_REFUSED

    out = _output(capsys)
    assert "row grammar refused" in str(out["reason"])
    assert "restored" in str(out["reason"])
    assert host.store.read_bytes() == before


def test_without_execute_nothing_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    host = _Host(tmp_path, resident={NEW})
    before = host.store.read_bytes()

    assert host.set() == dev_pin.EXIT_OK

    assert _output(capsys)["outcome"] == "PLANNED"
    assert host.store.read_bytes() == before
    assert host.ledger.read_text() == ""


def test_an_unchanged_pin_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    host = _Host(tmp_path, resident={OLD})

    assert host.set("--execute", value=OLD) == dev_pin.EXIT_OK

    assert _output(capsys)["outcome"] == "UNCHANGED"
    assert host.ledger.read_text() == ""


def test_a_duplicated_pin_line_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    host = _Host(tmp_path, resident={NEW})
    host.store.write_text(host.store.read_text() + f"ONEX_CLOUD_MIGRATE_IMAGE={OLD}\n")

    assert host.set("--execute") == dev_pin.EXIT_REFUSED

    assert "2 ONEX_CLOUD_MIGRATE_IMAGE= lines" in str(_output(capsys)["reason"])


def test_read_prints_the_pin_and_whether_it_is_resident(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    host = _Host(tmp_path, resident={OLD})

    code = main(
        [
            "read",
            "--env-file",
            str(host.store),
            "--key",
            "ONEX_CLOUD_MIGRATE_IMAGE",
            "--docker",
            str(host.docker),
        ]
    )

    out = _output(capsys)
    assert code == dev_pin.EXIT_OK
    assert (out["outcome"], out["value"], out["resident"]) == ("READ", OLD, True)


def test_the_next_compose_command_sees_the_new_pin_with_no_restart(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two halves together: set the pin, and the running agent reads it."""
    host = _Host(tmp_path, resident={NEW})
    monkeypatch.setenv("DEPLOY_AGENT_ENV_FILE", str(host.store))
    monkeypatch.setenv(
        "DEPLOY_AGENT_ENV_PROTECTED", "DEPLOY_AGENT_ENV_FILE DEPLOY_AGENT_ENV_PROTECTED"
    )
    monkeypatch.setenv("ONEX_CLOUD_MIGRATE_IMAGE", OLD)
    assert (
        operator_store_overlay().get(
            "ONEX_CLOUD_MIGRATE_IMAGE", os.environ["ONEX_CLOUD_MIGRATE_IMAGE"]
        )
        == OLD
    )

    assert host.set("--execute") == dev_pin.EXIT_OK

    assert operator_store_overlay()["ONEX_CLOUD_MIGRATE_IMAGE"] == NEW
