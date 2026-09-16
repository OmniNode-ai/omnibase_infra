# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""`deploy-runners.sh --host` targets one declared host (OMN-17477).

The script read the fleet config's scalar fields, which describe ONE host. That
made a second host unrepresentable rather than merely unconfigured. `--host`
selects a row from the `hosts:` inventory instead.

Two properties matter and are asserted separately because they fail in opposite
directions:

* an unset `--host` must behave exactly as before, or every existing call site
  and cron silently changes what it deploys;
* an unknown `--host` must FAIL, not fall through to the primary host. A typo
  that deployed to `.201` while its operator believed they were deploying
  elsewhere is the one outcome a host flag must never produce.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "deploy-runners.sh"
FLEET_CONFIG = REPO_ROOT / "config" / "runner_fleet.yaml"

# Sourcing the script would run a deploy. Extract the host-row reader and drive
# it directly, so the test exercises the real parser rather than a copy of it.
HARNESS = r"""
set -euo pipefail
RUNNER_FLEET_CONFIG="$FLEET_CONFIG"
eval "$(sed -n '/^runner_host_field()/,/^}/p' "$SCRIPT_PATH")"
"""


def _field(host: str, field: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", HARNESS + f'runner_host_field "{host}" "{field}"'],
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "FLEET_CONFIG": str(FLEET_CONFIG),
            "SCRIPT_PATH": str(SCRIPT),
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_reads_the_named_hosts_own_row_not_the_first_one() -> None:
    """The reader must anchor on the host, not match the first key in the file.

    The script's pre-existing ``runner_config_field`` matches a key ANYWHERE in
    the file, which is correct for a top-level scalar and actively wrong inside
    a list of mappings -- it would hand back the FIRST host's value for every
    host asked about, so a second host would deploy under the first host's name
    prefix and collide on every container name.
    """
    primary = _field("omninode-pc.tail75df5e.ts.net", "runner_name_prefix")
    assert primary.returncode == 0, primary.stderr
    assert primary.stdout.strip() == "omninode-runner"

    second = _field("stickybeatz-2.tail75df5e.ts.net", "runner_name_prefix")
    assert second.returncode == 0, second.stderr
    assert second.stdout.strip() == "omninode-mini-runner", (
        "the reader returned the wrong host's prefix; it is matching a key "
        "rather than anchoring on the host row"
    )


def test_reads_the_architecture_of_each_host() -> None:
    assert _field("omninode-pc.tail75df5e.ts.net", "arch").stdout.strip() == "amd64"
    assert _field("stickybeatz-2.tail75df5e.ts.net", "arch").stdout.strip() == "arm64"


def test_an_undeclared_host_fails_instead_of_falling_through() -> None:
    result = _field("not-a-declared-host.example", "arch")
    assert result.returncode != 0, (
        "an undeclared host must fail closed; falling through would deploy to "
        f"the primary host under another host's name. stdout={result.stdout!r}"
    )
    assert "declared hosts" in result.stderr, (
        "the refusal must name what IS declared, so the operator can see the "
        f"typo rather than guess at it. stderr={result.stderr!r}"
    )


def test_the_help_text_documents_the_flag() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"], capture_output=True, text=True, check=False
    )
    assert "--host=NAME" in result.stdout, result.stdout


def test_every_declared_non_primary_host_has_a_compose_file() -> None:
    """A declared host with no compose file is a deploy that fails at the host.

    The primary host's compose file hand-writes its 60 literal service blocks,
    so it cannot be reused; each additional host brings its own, named for that
    host's prefix. The naming convention is what `--host` resolves, so a missing
    file is a broken inventory row, not a missing feature.
    """
    import yaml

    config = yaml.safe_load(FLEET_CONFIG.read_text(encoding="utf-8"))
    primary = config["runner_host"]
    for row in config.get("hosts") or []:
        if row["host"] == primary:
            continue
        compose = (
            REPO_ROOT
            / "docker"
            / f"docker-compose.runners-{row['runner_name_prefix']}.yml"
        )
        assert compose.is_file(), (
            f"host {row['host']} declares prefix {row['runner_name_prefix']} but "
            f"{compose.relative_to(REPO_ROOT)} does not exist"
        )
