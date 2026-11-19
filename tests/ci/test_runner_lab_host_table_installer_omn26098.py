# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26098: every runner compose that mounts the lab host table is served by the installer's own key write.

The compose files refuse to create a container without ``RUNNER_LAB_HOST_TABLE`` (``create_host_path: false``), and the
host's ``docker/.env`` held a hand-set path to a file a later change deleted. The key is now written by the deploy
effect, so the compose side has to stay what the effect writes for: the same variable, a read-only mount at the
container path the runner reads, and no instruction left that tells an operator to set it by hand.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from omnibase_infra.models.model_runner_lab_host_table_policy import (
    ModelRunnerLabHostTablePolicy,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
POLICY = ModelRunnerLabHostTablePolicy()
COMPOSES = sorted(
    path
    for path in (ROOT / "docker").glob("docker-compose.runners-*.yml")
    if POLICY.env_key in path.read_text(encoding="utf-8")
)
CONTAINER_PATH = "/etc/onex/lab_run_hosts.yaml"


def test_the_five_hp_pools_are_the_composes_that_mount_the_table() -> None:
    assert [path.name for path in COMPOSES] == [
        "docker-compose.runners-hp051-ci-mq-runner.yml",
        "docker-compose.runners-hp051-ci-runner.yml",
        "docker-compose.runners-hp051-omnibase-ci-runner.yml",
        "docker-compose.runners-hp053-ci-mq-runner.yml",
        "docker-compose.runners-hp053-ci-runner.yml",
    ]


@pytest.mark.parametrize("compose", COMPOSES, ids=lambda path: path.name)
def test_compose_mounts_the_key_the_installer_writes_read_only_without_creating_it(
    compose: Path,
) -> None:
    document = yaml.safe_load(compose.read_text(encoding="utf-8"))
    mount = document["x-lab-host-table"]
    assert mount["source"].startswith(f"${{{POLICY.env_key}:?")
    assert mount["target"] == CONTAINER_PATH
    assert mount["read_only"] is True
    assert mount["bind"]["create_host_path"] is False


@pytest.mark.parametrize("compose", COMPOSES, ids=lambda path: path.name)
def test_no_compose_tells_an_operator_to_put_the_key_in_the_env_file_by_hand(
    compose: Path,
) -> None:
    for line in compose.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#") and re.search(r"\bput\b", line):
            assert POLICY.env_key not in line, line


def test_the_files_this_change_touches_spell_no_machine_home_path() -> None:
    """The local-path hook's rule, held where the installer's own sources and tests are."""
    touched = [
        "src/omnibase_infra/handlers/handler_deploy_runners.py",
        "src/omnibase_infra/handlers/handler_runner_image_roll.py",
        "src/omnibase_infra/models/model_runner_lab_host_table_result.py",
        "src/omnibase_infra/models/model_runner_lab_host_table_policy.py",
        "tests/unit/handlers/test_handler_deploy_runners_lab_host_table_omn26098.py",
        "tests/ci/test_runner_lab_host_table_installer_omn26098.py",
    ]
    pattern = re.compile("/" + "Users/|/" + "home/")
    for name in touched:
        text = (ROOT / name).read_text(encoding="utf-8")
        assert pattern.search(text) is None, name


@pytest.mark.parametrize(
    "installer",
    ["handlers/handler_deploy_runners.py", "handlers/handler_runner_image_roll.py"],
)
def test_both_installers_that_recreate_a_pool_converge_the_table_first(
    installer: str,
) -> None:
    """The deploy stand-up and ``--rolling --host`` both end in ``docker compose up``; each must write the key."""
    source = (ROOT / "src" / "omnibase_infra" / installer).read_text(encoding="utf-8")
    assert "ModelRunnerLabHostTablePolicy" in source
    assert ".converge(" in source
