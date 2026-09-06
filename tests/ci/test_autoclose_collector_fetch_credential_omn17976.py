# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-17976: the collector's OWN git fetch must carry a credential.

`EvidenceCollector._compute_product_clone_resolution` establishes a product
clone's freshness in three steps, and step 2 is `git fetch <remote> <branch>`,
spawned by the collector itself from inside the sweep step. Before this change
the credential helper was written and exported ONLY inside the materialise
step, so the clone authenticated and the collector's own later fetch did not.

Public repos hid it — an anonymous https fetch needs no credential — so
OMN-17907 and OMN-17975 executed and flipped while every `omninode_infra`-homed
check was refused UNEXECUTED. Measured on run 34004893099, 2026-09-06T01:53:04Z,
three candidates in a row:

    PRODUCT_CLONE_NOT_FRESH: ... upstream origin/dev, behind None; git fetch
    origin dev failed ...: fatal: could not read Username for
    'https://github.com': No such device or address

`upstream origin/dev` in that message is OMN-17975's fix already working: the
refusal had moved off step 1 onto step 2. This pins step 2 closed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "evidence-autoclose-sweep.yml"

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def steps(workflow: dict) -> list[dict]:
    return workflow["jobs"]["evidence-autoclose-sweep"]["steps"]


def _named(steps: list[dict], fragment: str) -> dict:
    for step in steps:
        if fragment in (step.get("name") or ""):
            return step
    raise AssertionError(f"no step whose name contains {fragment!r}")


def test_every_step_that_spawns_a_collector_fetch_carries_the_credential(
    steps: list[dict],
) -> None:
    """AC3. Removing this wiring must fail, not silently stop executing checks."""
    for fragment in ("Run evidence autoclose sweep", "Diagnose verdict divergence"):
        step = _named(steps, fragment)
        env = step.get("env") or {}
        assert "GIT_ASKPASS" in env, (
            f"{fragment!r} spawns dod_verify, which runs `git fetch` for itself as "
            "step 2 of its freshness predicate; without GIT_ASKPASS every "
            "private-repo-homed behaviour check is refused unexecuted"
        )
        assert env["GIT_ASKPASS"].endswith("git-askpass.sh")
        # A credential-less fetch must fail fast and by name, never hang on a
        # prompt inside a scheduled job nobody is watching.
        assert str(env.get("GIT_TERMINAL_PROMPT")) == "0"
        assert "GH_TOKEN" in env, "the helper answers with GH_TOKEN from its own env"


def test_the_helper_is_provisioned_before_anything_consumes_it(
    steps: list[dict],
) -> None:
    """Ordering is the whole mechanism: a later step cannot use a file not yet written."""
    names = [s.get("name") or "" for s in steps]
    provision = next(
        i for i, n in enumerate(names) if "Provision the git credential" in n
    )
    for fragment in (
        "Derive and materialise the cwd repo set",
        "Diagnose verdict divergence",
        "Run evidence autoclose sweep",
    ):
        consumer = next(i for i, n in enumerate(names) if fragment in n)
        assert provision < consumer, f"{fragment!r} runs before the helper exists"


def test_the_helper_answers_git_the_way_git_asks(tmp_path: Path) -> None:
    """Execute the helper's real body rather than asserting its text.

    git invokes an askpass program with the prompt as argv[1] and reads one line
    from stdout. A helper that answers the username prompt with the token (or
    the reverse) authenticates nothing, and the failure would look identical to
    having no helper at all.
    """
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    opener = next(
        i for i, line in enumerate(lines) if "git-askpass.sh\" <<'EOS'" in line
    )
    indent = len(lines[opener]) - len(lines[opener].lstrip())
    body_lines: list[str] = []
    for line in lines[opener + 1 :]:
        if line.strip() == "EOS":
            break
        body_lines.append(line[indent:] if line.startswith(" " * indent) else line)
    else:  # pragma: no cover - only reachable if the heredoc loses its terminator
        raise AssertionError("the git-askpass heredoc has no EOS terminator")
    assert body_lines, "the git-askpass heredoc is empty"
    script = "\n".join(body_lines) + "\n"

    helper = tmp_path / "git-askpass.sh"
    helper.write_text(script, encoding="utf-8")
    helper.chmod(0o755)

    env = {"GH_TOKEN": "sentinel-token-value", "PATH": "/usr/bin:/bin"}
    user = subprocess.run(
        [str(helper), "Username for 'https://github.com': "],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    password = subprocess.run(
        [str(helper), "Password for 'https://x-access-token@github.com': "],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert user.stdout.strip() == "x-access-token"
    assert password.stdout.strip() == "sentinel-token-value"
    # The token is answered to git and nowhere else: it must not be echoed onto
    # the username channel, where it would land in a remote URL.
    assert "sentinel-token-value" not in user.stdout


def test_the_token_is_never_written_into_a_clone_or_a_command_line() -> None:
    """The discipline this extends, asserted so extending it cannot erode it."""
    body = WORKFLOW.read_text(encoding="utf-8")
    for laundering in (
        "https://x-access-token:",
        "credential.helper store",
        "extraheader",
    ):
        assert laundering not in body, (
            f"{laundering!r} would place the token in a clone's config or a URL; "
            "the credential reaches git through GIT_ASKPASS only"
        )
