# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise pool clone selection and staging arguments without a Docker daemon."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
PROVE = (
    Path(__file__).resolve().parents[3] / "scripts/runtime_build/prepr_pool_prove.sh"
)
SHA = "1" * 40


def _params(tmp_path: Path, **overrides: str) -> Path:
    values = dict.fromkeys(
        (
            "INFRA_PR",
            "INFRA_HEAD",
            "INFRA_GROUP",
            "INFRA_REF_SHA",
            "INFRA_DEV_TAG",
            "MARKET_PR",
            "MARKET_HEAD",
            "MARKET_GROUP",
            "MARKET_REF_SHA",
            "CORE_REF",
            "SPI_REF",
            "COMPAT_REF",
            "STAGE_DEV_TAGS",
        ),
        "",
    )
    values.update(TAG="stage0-test", W=str(tmp_path / "work"), SLOT_KIND="isolated")
    values.update(overrides)
    params = tmp_path / "params.env"
    params.write_text(
        "".join(f"{key}={shlex.quote(value)}\n" for key, value in values.items())
        + f"""
git() {{
  printf '%s\\n' "$*" >> {shlex.quote(str(tmp_path / "git.log"))}
  if [ "$1" = clone ]; then
    mkdir -p "${{@: -1}}"
  else
    printf '%s\\n' {SHA}
  fi
}}
""",
        encoding="utf-8",
    )
    return params


@pytest.mark.parametrize("tag", ["", "dev-20261030223038-9de6fcbd36e7"])
def test_pool_clone_selects_infra_tag_and_only_current_siblings(
    tmp_path: Path, tag: str
) -> None:
    params = _params(tmp_path, INFRA_DEV_TAG=tag)
    done = subprocess.run(
        ["bash", str(PROVE), str(params), "clone"],
        env={**os.environ, "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    calls = (tmp_path / "git.log").read_text().splitlines()
    clones = [line for line in calls if line.startswith("clone ")]
    assert len(clones) == 4
    assert {line.split("/")[-1] for line in clones} == {
        "omnibase_infra",
        "omnimarket",
        "omnibase_core",
        "omnibase_spi",
    }
    infra = next(line for line in clones if "omnibase_infra.git" in line)
    assert f"--branch {tag or 'dev'} " in infra


@pytest.mark.parametrize("stage_tags", ["", "1"])
def test_pool_staging_passes_spi_pins_or_selects_dev_tags(
    tmp_path: Path, stage_tags: str
) -> None:
    params = _params(tmp_path, STAGE_DEV_TAGS=stage_tags)
    infra = tmp_path / "work/root/omnibase_infra"
    stage = infra / "scripts/runtime_build/stage_workspace.sh"
    stage.parent.mkdir(parents=True)
    stage.write_text(
        'printf "%s\\n" "$DEPLOY_REF" > "$OMNI_HOME/deploy-ref"\n'
        'printf "%s\\n" "$@" > "$OMNI_HOME/stage-args"\n',
        encoding="utf-8",
    )
    # Run the script through staging, stopping before local-env and all container
    # operations. Keep the actual build case and staging pipeline unchanged.
    source = PROVE.read_text()
    before_make = source.split("  make local-env ", 1)[0]
    assert before_make != source
    staging_only = tmp_path / "staging-only.sh"
    staging_only.write_text(before_make + "  ;;\nesac\n", encoding="utf-8")
    done = subprocess.run(
        ["bash", str(staging_only), str(params), "build"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "stage rc=0" in done.stdout
    root = infra.parent
    assert (root / "deploy-ref").read_text().strip() == (
        "origin/dev" if stage_tags else ""
    )
    assert (root / "stage-args").read_text().split() == (
        []
        if stage_tags
        else [
            "--repo-ref",
            f"omnibase_core={SHA}",
            "--repo-ref",
            f"omnibase_spi={SHA}",
            "--repo-ref",
            f"omnimarket={SHA}",
        ]
    )
