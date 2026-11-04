# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Estimator image delivery against a real Git source and the real node contract.

The dev lane's cost estimator image is built from a web source repository that
this repository does not own. This module drives ``build_dev_lane_services``
exactly as the deploy agent does: the build recipe comes from the shipped
``node_cost_estimator_build_effect/contract.yaml`` on disk, the source identity
comes from the deploy environment, and the handler clones a real Git repository
and invokes a real ``docker`` executable found on ``PATH``.

Only the container engine is replaced: a small executable named ``docker`` that
keeps image labels in a state file stands in for the daemon, so a hosted runner
needs no daemon, credential or lab host. Git is not replaced.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.handlers.handler_cost_estimator_build import (
    REVISION_LABEL,
    SOURCE_LABEL,
    build_dev_lane_services,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONTRACT = (
    REPOSITORY_ROOT
    / "src/omnibase_infra/nodes/node_cost_estimator_build_effect/contract.yaml"
)
SOURCE_REPOSITORY = "web-front-end"

# A stand-in container engine: `image inspect` answers from a label file and
# `build` records the labels it was given, so label readback is a real round trip.
FAKE_DOCKER = """\
#!/bin/sh
state="$FAKE_DOCKER_STATE"
printf '%s\\n' "$*" >> "$state/calls.log"
case "$1 $2" in
  "image inspect")
    [ -f "$state/labels.json" ] || exit 1
    cat "$state/labels.json"
    ;;
  "build "*)
    shift
    python3 - "$state/labels.json" "$@" <<'PY'
import json, sys
target, argv = sys.argv[1], sys.argv[2:]
labels = {}
for index, token in enumerate(argv):
    if token == "--label":
        key, _, value = argv[index + 1].partition("=")
        labels[key] = value
open(target, "w").write(json.dumps(labels))
PY
    ;;
  *) exit 2 ;;
esac
"""


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        capture_output=True,
        text=True,
        check=True,
        env={
            **scrub_git_location_env(os.environ),
            "GIT_AUTHOR_NAME": "fixture",
            "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
            "GIT_COMMITTER_NAME": "fixture",
            "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
        },
    )
    return result.stdout.strip()


def _commit(repository: Path, content: str) -> str:
    (repository / "Dockerfile").write_text(content, encoding="utf-8")
    _git(repository, "add", "Dockerfile")
    _git(repository, "commit", "-q", "-m", content.strip())
    return _git(repository, "rev-parse", "HEAD")


@pytest.fixture
def web_source(tmp_path: Path) -> Path:
    repository = tmp_path / "web-source"
    repository.mkdir()
    _git(repository, "init", "-q", "-b", "dev")
    return repository


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    state = tmp_path / "engine-state"
    state.mkdir()
    binaries = tmp_path / "bin"
    binaries.mkdir()
    docker = binaries / "docker"
    docker.write_text(FAKE_DOCKER, encoding="utf-8")
    docker.chmod(docker.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{binaries}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DOCKER_STATE", str(state))
    return state


def _deliver(
    web_source: Path,
    *,
    lane: str = "dev",
    disabled: tuple[str, ...] = (),
    settings: dict[str, str] | None = None,
) -> Mock:
    """Run the dev-lane delivery the way the deploy agent calls it."""
    environment = (
        {
            "COST_ESTIMATOR_WEB_SOURCE_REPOSITORY": SOURCE_REPOSITORY,
            "COST_ESTIMATOR_WEB_SOURCE_URL": f"file://{web_source}",
        }
        if settings is None
        else settings
    )
    execution = SimpleNamespace(
        # The handler locates the contract relative to the deploy agent module.
        __file__=str(REPOSITORY_ROOT / "scripts/deploy-agent/deploy_agent/executor.py"),
        EnumRuntimeLane=SimpleNamespace(DEV="dev"),
        DEV_LANE_BUILD_SERVICES=("runtime",),
        Scope=SimpleNamespace(RUNTIME="runtime"),
        lane_config_for=lambda _lane: SimpleNamespace(
            compose_files=("base", "dev"), disabled_services=disabled
        ),
        _compose_env=lambda: dict(environment),
    )
    build_images = Mock()
    build_dev_lane_services(
        execution,
        build_images,
        "c" * 40,
        lambda *args, **kwargs: None,
        build_source="local",
        lane=lane,
        git_ref="dev",
    )
    return build_images


def _labels(engine: Path) -> dict[str, str]:
    return json.loads((engine / "labels.json").read_text(encoding="utf-8"))


def _builds(engine: Path) -> list[str]:
    calls = (engine / "calls.log").read_text(encoding="utf-8").splitlines()
    return [call for call in calls if call.startswith("build ")]


def test_shipped_contract_declares_the_recipe_the_delivery_reads() -> None:
    build = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))["build"]
    assert build["branch"] == "dev"
    assert build["repository_env"] == "COST_ESTIMATOR_WEB_SOURCE_REPOSITORY"
    assert build["repository_url_env"] == "COST_ESTIMATOR_WEB_SOURCE_URL"


def test_absent_image_is_built_and_stamped_with_the_web_head(
    web_source: Path, engine: Path
) -> None:
    head = _commit(web_source, "FROM scratch\n")

    _deliver(web_source)

    assert _labels(engine) == {SOURCE_LABEL: SOURCE_REPOSITORY, REVISION_LABEL: head}
    assert len(_builds(engine)) == 1


def test_unchanged_web_head_reuses_the_image(web_source: Path, engine: Path) -> None:
    _commit(web_source, "FROM scratch\n")
    _deliver(web_source)

    _deliver(web_source)

    assert len(_builds(engine)) == 1


def test_advanced_web_head_rebuilds_the_image(web_source: Path, engine: Path) -> None:
    _commit(web_source, "FROM scratch\n")
    _deliver(web_source)
    advanced = _commit(web_source, "FROM scratch\nLABEL page=advanced\n")

    _deliver(web_source)

    assert _labels(engine)[REVISION_LABEL] == advanced
    assert len(_builds(engine)) == 2


def test_unreachable_source_refuses_before_any_build(
    web_source: Path, engine: Path, tmp_path: Path
) -> None:
    missing = tmp_path / "no-such-repository"

    with pytest.raises(RuntimeError, match="estimator git clone exited"):
        _deliver(
            web_source,
            settings={
                "COST_ESTIMATOR_WEB_SOURCE_REPOSITORY": SOURCE_REPOSITORY,
                "COST_ESTIMATOR_WEB_SOURCE_URL": f"file://{missing}",
            },
        )

    assert not (engine / "calls.log").exists()


def test_missing_source_setting_refuses_before_any_engine_call(
    web_source: Path, engine: Path
) -> None:
    with pytest.raises(RuntimeError, match="COST_ESTIMATOR_WEB_SOURCE_URL"):
        _deliver(
            web_source,
            settings={
                "COST_ESTIMATOR_WEB_SOURCE_REPOSITORY": SOURCE_REPOSITORY,
            },
        )

    assert not (engine / "calls.log").exists()


def test_disabled_instance_does_no_estimator_work(
    web_source: Path, engine: Path
) -> None:
    _commit(web_source, "FROM scratch\n")

    _deliver(web_source, disabled=("cost-estimator-web",))

    assert not (engine / "calls.log").exists()
