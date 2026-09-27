# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Merges that do not change what a rebuilt lane runs stay out of the rebuild train (OMN-19597).

Operator ruling 2026-09-25T13:17:25Z (roadmap decision 1, "Two is yes"): a PR
that does not touch the deployed runtime never enters the rebuild train. It
still gets its pre-merge lab proof (RULING 2026-09-25T13:07:40Z), in a parallel
slot, not in the train.

MEASURED 2026-09-25 by replaying this classifier, with omniclaude main's real
deploy-gate validator, over the last 160 merges to dev in omnibase_infra and in
omnimarket. The 100 most recent merges it classified runtime-affecting
(2026-09-23T22:38Z to 2026-09-25T13:09Z, 51 omnibase_infra and 49 omnimarket)
included 17 that changed nothing the lane runs:

* 12 changed only the deploy agent (``scripts/deploy-agent/**``). The agent is
  not in the runtime image, and since OMN-18200's own second half
  (omnibase_infra#3524) an idle agent fetches its tracking branch and re-execs
  every ``DEPLOY_AGENT_SELF_UPDATE_IDLE_INTERVAL`` seconds (default 300) at the
  ``IDLE_HEARTBEAT`` boundary, so its fixes reach it without a rebuild. The
  lane-state entry predates that boundary and its stated reason ("self_update
  has only PRE_ACCEPT and POST_TERMINAL boundaries") no longer holds.
* 5 changed only compose files of lanes the rebuild never deploys (the runner
  pool, sim-202, dogfood) or the gate-runner and delegated-test-loop images,
  which no deployed compose file builds.

Four of those were read back from the trigger's own run logs, each "Runtime
change detected" followed by "Published redeploy-start": runs 36130599047
(#4115, the dev-202 agent unit), 36114145961 (#4110, the .202 runner compose),
36077861772 (#4040, the runner pool compose) and 36002578834 (#4033, the two
test-image Dockerfiles). Negative control: run 36134517716 (#4119, a workflow
and its test) logged "No rebuild trigger", which the replay also said.

Two latent classes had no member in the window and are closed here too: a
``tests/`` directory inside ``src/`` (252 files in omnimarket, 7 in this repo)
matched both the canonical ``src/*/nodes/**/*.py`` pattern and the supplement's
``src/<package>/**``, and a Markdown file inside ``src/`` matched the latter.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER_PATH = REPO_ROOT / "scripts" / "runtime_change_classifier.py"
DEPLOY_AGENT_PACKAGE = REPO_ROOT / "scripts" / "deploy-agent" / "deploy_agent"


def _load() -> Any:
    name = "_runtime_change_classifier_omn19597"
    spec = importlib.util.spec_from_file_location(name, CLASSIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def classifier() -> Any:
    return _load()


_CANONICAL_PATTERNS = (
    re.compile(r"^docker/Dockerfile[^/]*$"),
    re.compile(r"^docker/docker-compose[^/]*\.ya?ml$"),
    re.compile(r"^src/[^/]+/nodes/.+\.py$"),
    re.compile(r"^src/[^/]+/runtime/.+\.py$"),
)


def _canonical_like(changed_files: list[str]) -> list[str]:
    """The canonical deploy-gate patterns these merges actually hit.

    omniclaude main's ``RUNTIME_PATH_PATTERNS`` carries ``docker/Dockerfile*``,
    ``docker/docker-compose*.yml``, ``src/*/nodes/**/*.py`` and
    ``src/*/runtime/**/*.py``; those four are what the replayed false positives
    matched, so the double reproduces them rather than the whole list.
    """
    return [f for f in changed_files if any(p.match(f) for p in _CANONICAL_PATTERNS)]


def _triggers(module: Any, files: list[str], source_repo: str | None) -> bool:
    paths = module.classify_runtime_paths(
        files, _canonical_like, source_repo=source_repo
    )
    return bool(module.is_runtime_affecting(paths, []))


# ---------------------------------------------------------------------------
# The replayed merges, file lists exact.
# ---------------------------------------------------------------------------

PR_4110_FILES = ["docker/docker-compose.runners-omnipc2-verify-runner.yml"]
PR_4040_FILES = ["docker/docker-compose.runners.yml"]
PR_4033_FILES = [
    "docker/Dockerfile.dtl-env",
    "docker/Dockerfile.gate-runner",
    "tests/ci/test_dtl_env_dockerfile.py",
]
PR_4041_FILES = ["docker/docker-compose.sim-202.yml"]
PR_4025_FILES = [
    "docker/docker-compose.dogfood.yml",
    "docker/docker-compose.runners-omninode-air-runner.yml",
    "docker/docker-compose.runners-omninode-mini-runner.yml",
    "docker/docker-compose.runners.yml",
]
PR_4115_FILES = ["scripts/deploy-agent/deploy/deploy-agent-dev-202.service"]
PR_4096_FILES = ["scripts/deploy-agent/deploy_agent/coalesce.py"]


@pytest.mark.unit
class TestReplayedFalsePositivesNoLongerTrigger:
    @pytest.mark.parametrize(
        "files",
        [PR_4110_FILES, PR_4040_FILES, PR_4033_FILES, PR_4041_FILES, PR_4025_FILES],
        ids=["4110", "4040", "4033", "4041", "4025"],
    )
    def test_undeployed_compose_and_image_files(
        self, classifier: Any, files: list[str]
    ) -> None:
        assert not _triggers(classifier, files, "omnibase_infra")

    @pytest.mark.parametrize(
        "files", [PR_4115_FILES, PR_4096_FILES], ids=["4115", "4096"]
    )
    def test_deploy_agent_only(self, classifier: Any, files: list[str]) -> None:
        assert not _triggers(classifier, files, "omnibase_infra")
        assert classifier.find_lane_state_paths(files) == []


@pytest.mark.unit
class TestRuntimePathsStillTrigger:
    """Positive controls: the narrowing must never reach a path the lane runs."""

    @pytest.mark.parametrize(
        ("path", "repo"),
        [
            (
                "src/omnimarket/nodes/node_delegation_orchestrator/handlers/"
                "handler_delegation_orchestrator.py",
                "omnimarket",
            ),
            (
                "src/omnibase_infra/nodes/node_dlq_replay_effect/handlers/"
                "handler_dlq_replay.py",
                "omnibase_infra",
            ),
            ("src/omnibase_infra/runtime/service_kernel.py", "omnibase_infra"),
            ("src/omnimarket/configs/delegation_routing.yaml", "omnimarket"),
            ("docker/docker-compose.infra.yml", "omnibase_infra"),
            ("docker/docker-compose.dev-lane.yml", "omnibase_infra"),
            ("docker/docker-compose.dev-202.yml", "omnibase_infra"),
            ("docker/docker-compose.stability-test.yml", "omnibase_infra"),
            ("docker/docker-compose.gateway.yml", "omnibase_infra"),
            ("docker/Dockerfile.runtime", "omnibase_infra"),
            ("docker/migrations/forward/001_x.sql", "omnibase_infra"),
        ],
    )
    def test_triggers(self, classifier: Any, path: str, repo: str) -> None:
        assert _triggers(classifier, [path], repo), path

    def test_a_mixed_merge_still_triggers_on_its_runtime_half(
        self, classifier: Any
    ) -> None:
        files = [
            *PR_4040_FILES,
            "README.md",
            "src/omnibase_infra/runtime/service_kernel.py",
        ]
        paths = classifier.classify_runtime_paths(
            files, _canonical_like, source_repo="omnibase_infra"
        )
        assert paths == ["src/omnibase_infra/runtime/service_kernel.py"]

    def test_an_unknown_new_compose_file_stays_runtime(self, classifier: Any) -> None:
        """Only NAMED compose files are excluded: a new one fails toward a rebuild."""
        assert _triggers(
            classifier, ["docker/docker-compose.brand-new-lane.yml"], "omnibase_infra"
        )

    def test_the_label_still_forces_a_trigger(self, classifier: Any) -> None:
        paths = classifier.classify_runtime_paths(
            PR_4040_FILES, _canonical_like, source_repo="omnibase_infra"
        )
        assert paths == []
        assert classifier.is_runtime_affecting(paths, ["runtime_change"])


@pytest.mark.unit
class TestDocsTestsAndCiNeverTrigger:
    @pytest.mark.parametrize(
        "path",
        [
            "README.md",
            "docs/runbooks/cold-lane-full-bringup.md",
            "src/omnimarket/nodes/node_delegation_orchestrator/README.md",
            "src/omnimarket/adapters/codex/skills/merge-sweep/SKILL.md",
            "src/omnibase_infra/nodes/node_dlq_replay_effect/README.md",
            "tests/unit/test_anything.py",
            "src/omnimarket/nodes/node_swarm_dispatch_orchestrator/tests/test_x.py",
            "src/omnibase_infra/services/observability/agent_actions/tests/test_consumer.py",
            ".github/workflows/ci.yml",
            ".github/omnimarket-contract-pin.yaml",
            "scripts/deploy-agent/tests/unit/test_executor.py",
        ],
    )
    def test_does_not_trigger(self, classifier: Any, path: str) -> None:
        for repo in ("omnibase_infra", "omnimarket", None):
            assert not _triggers(classifier, [path], repo), (path, repo)


@pytest.mark.unit
class TestPerRepositoryData:
    def test_repository_entries_apply_only_to_their_repository(
        self, classifier: Any
    ) -> None:
        """omninode_infra has no runner pool; its compose files are its own business."""
        assert _triggers(classifier, PR_4040_FILES, "omninode_infra")
        assert not _triggers(classifier, PR_4040_FILES, "omnibase_infra")

    def test_no_repository_named_applies_every_entry(self, classifier: Any) -> None:
        """The release train walks one clone and names no repository.

        It must agree with the trigger on every real path, and every
        repository-scoped entry names a path only that repository has.
        """
        assert not _triggers(classifier, PR_4040_FILES, None)

    def test_every_excluded_repository_path_exists_here(self, classifier: Any) -> None:
        """A dead entry is how a list stops describing the tree."""
        for pattern in classifier.NOT_RUNTIME_PATH_PATTERNS["omnibase_infra"]:
            assert list(REPO_ROOT.glob(pattern)), pattern

    def test_every_entry_carries_a_reason(self, classifier: Any) -> None:
        for repo, entries in classifier.NOT_RUNTIME_PATH_REASONS.items():
            assert set(entries) == set(classifier.NOT_RUNTIME_PATH_PATTERNS[repo])
            for pattern, reason in entries.items():
                assert reason.strip(), (repo, pattern)


def _compose_files_the_deploy_agent_runs() -> set[str]:
    """Every compose file the deploy agent's code names, docstrings excluded."""
    named: set[str] = set()
    compose = re.compile(r"docker/docker-compose[A-Za-z0-9._-]*\.ya?ml")
    for source in DEPLOY_AGENT_PACKAGE.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(
                node,
                ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
            )
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
            ):
                named.update(compose.findall(node.value))
    return named


def _dockerfiles_built_by(compose_files: set[str]) -> set[str]:
    built: set[str] = set()
    for compose in compose_files:
        path = REPO_ROOT / compose
        if not path.is_file():
            continue
        for match in re.finditer(
            r"^\s*dockerfile:\s*(\S+)", path.read_text(encoding="utf-8"), re.M
        ):
            built.add(match.group(1).strip("'\""))
    return built


@pytest.mark.unit
class TestNothingTheDeployAgentRunsIsExcluded:
    def test_the_agent_names_compose_files(self) -> None:
        """Positive control: the scan finds the files the dev lane is built from."""
        named = _compose_files_the_deploy_agent_runs()
        assert {
            "docker/docker-compose.infra.yml",
            "docker/docker-compose.dev-lane.yml",
            "docker/docker-compose.stability-test.yml",
        } <= named

    def test_no_compose_file_the_agent_runs_is_excluded(self, classifier: Any) -> None:
        for compose in sorted(_compose_files_the_deploy_agent_runs()):
            assert classifier.classify_runtime_paths(
                [compose], _canonical_like, source_repo="omnibase_infra"
            ) == [compose], compose

    def test_no_dockerfile_those_compose_files_build_is_excluded(
        self, classifier: Any
    ) -> None:
        built = _dockerfiles_built_by(_compose_files_the_deploy_agent_runs())
        assert "docker/Dockerfile.runtime" in built
        for dockerfile in sorted(built):
            assert classifier.classify_runtime_paths(
                [dockerfile], _canonical_like, source_repo="omnibase_infra"
            ) == [dockerfile], dockerfile
