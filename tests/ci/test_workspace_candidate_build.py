# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Invariants for the workspace stability-candidate ECR build path (OMN-13656).

The workspace-from-dev build (``.github/workflows/build-workspace-candidate-
runtime.yml`` + ``docker/Dockerfile.runtime`` workspace mode) must produce an
image that is UNMISTAKABLY a non-prod stability-candidate so the omnimarket
prod-promotion gate (``node_prod_promotion_gate_compute``) and the lineage guard
(OMN-12626) refuse it for prod. These deterministic tests pin the stamping
invariants so they cannot silently regress into a clean-main-looking image.

They assert static file contents only — no Docker build, no network.
"""

from __future__ import annotations

import ast
import re
import shlex
import sys
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKERFILE = _REPO_ROOT / "docker" / "Dockerfile.runtime"
_WORKFLOW = (
    _REPO_ROOT / ".github" / "workflows" / "build-workspace-candidate-runtime.yml"
)


def _candidate_steps(workflow: dict[object, object]) -> list[dict]:
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    build_job = jobs["build-workspace-candidate"]
    assert isinstance(build_job, dict)
    steps = build_job["steps"]
    assert isinstance(steps, list)
    assert all(isinstance(step, dict) for step in steps)
    return steps


def _step_index(steps: list[dict], name: str) -> int:
    indices = [index for index, step in enumerate(steps) if step.get("name") == name]
    assert len(indices) == 1, f"expected exactly one step named {name!r}, got {indices}"
    return indices[0]


def _stage_host_distributions(steps: list[dict]) -> set[str]:
    """Read direct host pip installs before the executable staging command."""
    stage_indices = [
        index
        for index, step in enumerate(steps)
        if re.search(
            r"^\s*bash\s+scripts/runtime_build/stage_workspace\.sh(?:\s|$)",
            step.get("run", ""),
            re.MULTILINE,
        )
    ]
    assert len(stage_indices) == 1, "expected exactly one staging command"
    installed: set[str] = set()
    for step in steps[: stage_indices[0]]:
        script = step.get("run", "").replace("\\\n", " ")
        for line in script.splitlines():
            if not re.match(r"^\s*python3\s+-m\s+pip\s+install\s", line):
                continue
            tokens = shlex.split(line, comments=True)
            for token in tokens[4:]:
                if token in {"--upgrade", "-U"}:
                    continue
                installed.add(canonicalize_name(Requirement(token).name))
    return installed


def _stage_python_sources(stage_script: Path) -> dict[str, str]:
    """Inspect SCRIPT_DIR-relative helpers and Python heredocs in staging."""
    shell = stage_script.read_text(encoding="utf-8")
    sources = {}
    for relative_path in sorted(
        set(re.findall(r"\$\{SCRIPT_DIR\}/([\w./-]+\.py)\b", shell))
    ):
        path = (stage_script.parent / relative_path).resolve()
        sources[str(path)] = path.read_text(encoding="utf-8")
    assert sources, "expected staging to reference Python helpers"
    for index, match in enumerate(
        re.finditer(
            r"\bpython3\b[^\n]*<<'(?P<delimiter>\w+)'\n"
            r"(?P<source>.*?)\n(?P=delimiter)(?:\n|$)",
            shell,
            re.DOTALL,
        )
    ):
        sources[f"{stage_script}:heredoc-{index}"] = match["source"]
    return sources


@pytest.mark.unit
def test_stage_third_party_sources_follow_script_references(tmp_path: Path) -> None:
    helper = tmp_path / "new_host_helper.py"
    helper.write_text("import newly_required_package\n", encoding="utf-8")
    stage = tmp_path / "stage_workspace.sh"
    stage.write_text(
        'python3 "${SCRIPT_DIR}/new_host_helper.py"\n'
        "python3 - <<'PY'\nimport inline_dependency\nPY\n",
        encoding="utf-8",
    )
    sources = _stage_python_sources(stage)
    assert sources[str(helper)] == "import newly_required_package\n"
    assert sources[f"{stage}:heredoc-0"] == "import inline_dependency"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("install_run", "expected"),
    [
        ("python3 -m pip install --upgrade 'PyYAML>=6'", True),
        (
            "python3 -m pip install -U \\\n            'pyyaml>=6' # host dependencies",
            True,
        ),
        ("python3 -m pip install 'pydantic>=2' # pyyaml>=6", False),
        ("python3 -m pip install 'unrelated-pyyaml>=6'", False),
        ("# python3 -m pip install 'pyyaml>=6'", False),
        ("echo python3 -m pip install 'pyyaml>=6'", False),
        ("python -m pip install 'pyyaml>=6'", False),
    ],
)
def test_stage_third_party_distributions_use_requirement_names(
    install_run: str, expected: bool
) -> None:
    steps = [
        {"name": "Renamed dependency setup", "run": install_run},
        {
            "name": "Renamed staging step",
            "run": "set -euo pipefail\nbash scripts/runtime_build/stage_workspace.sh",
        },
    ]
    assert ("pyyaml" in _stage_host_distributions(steps)) is expected


@pytest.mark.unit
def test_stage_third_party_distributions_exclude_late_installs() -> None:
    steps = [
        {"run": "bash scripts/runtime_build/stage_workspace.sh"},
        {"run": "python3 -m pip install 'pyyaml>=6'"},
    ]
    assert _stage_host_distributions(steps) == set()


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    assert _DOCKERFILE.is_file(), f"missing {_DOCKERFILE}"
    raw = _DOCKERFILE.read_text(encoding="utf-8")
    # Collapse shell line continuations so a single statement is one line.
    return re.sub(r"\\\n\s*", " ", raw)


@pytest.fixture(scope="module")
def workflow() -> dict[object, object]:
    assert _WORKFLOW.is_file(), f"missing {_WORKFLOW}"
    loaded = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


@pytest.fixture(scope="module")
def workflow_text() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


@pytest.mark.unit
class TestDockerfileCandidateStamping:
    def test_promotion_class_and_lineage_args_default_clean_main(
        self, dockerfile_text: str
    ) -> None:
        # Release builds must default to clean-main / non-main-lineage=false.
        assert "ARG PROMOTION_CLASS=clean-main" in dockerfile_text
        assert "ARG NON_MAIN_LINEAGE=false" in dockerfile_text

    def test_oci_labels_carry_promotion_class_and_lineage(
        self, dockerfile_text: str
    ) -> None:
        assert 'com.omninode.promotion_class="${PROMOTION_CLASS}"' in dockerfile_text
        assert 'com.omninode.non_main_lineage="${NON_MAIN_LINEAGE}"' in dockerfile_text

    def test_workspace_build_must_stamp_candidate(self, dockerfile_text: str) -> None:
        # The Dockerfile fails closed if a workspace build does not stamp the
        # candidate provenance — it can never be built as a clean-main image.
        guard = 'if [ "${BUILD_SOURCE}" = "workspace" ]'
        assert guard in dockerfile_text
        assert '[ "${PROMOTION_CLASS}" != "stability-candidate" ]' in dockerfile_text
        assert '[ "${NON_MAIN_LINEAGE}" != "true" ]' in dockerfile_text


@pytest.mark.unit
class TestWorkspaceCandidateWorkflow:
    def test_workflow_never_auto_triggers_on_a_push(
        self, workflow: dict[object, object]
    ) -> None:
        # Never a bare `push:` trigger on THIS file — a candidate is built only
        # when something explicitly asks for one, so it cannot masquerade as a
        # clean-main build. PyYAML parses the bare `on:` key as the boolean True.
        #
        # OMN-15796 widened this from "workflow_dispatch and nothing else" to
        # "workflow_dispatch and workflow_call", because dispatch-only meant a
        # candidate existed only when a human remembered to ask, and onex-dev
        # routinely ran days-old code behind a green deploy as a result.
        # deliver-dev-candidate-to-staging.yml now CALLS this workflow on a dev
        # merge. That is still a deliberate request — made by a workflow rather
        # than a person — and it changes nothing about the stamping asserted
        # throughout this class: workspace mode, stability-candidate promotion
        # class, non_main_lineage=true, and the prod-promotion gate's refusal
        # all hold identically on the called path. The invariant that actually
        # protects the lineage guard is the absence of `push:`, which is what
        # this asserts.
        on = workflow.get(True)
        if on is None:
            on = workflow.get("on")
        assert isinstance(on, dict)
        assert "push" not in on, (
            "a push-triggered candidate build would produce images on this "
            "repo's own branch pushes with no explicit request; the delivery "
            "chain calls this workflow instead (OMN-15796)"
        )
        assert set(on.keys()) <= {"workflow_dispatch", "workflow_call"}, on.keys()
        assert "workflow_dispatch" in on, "the manual escape hatch must survive"

    def test_workflow_uses_clean_cloud_egress_for_ecr_push(
        self, workflow: dict[object, object]
    ) -> None:
        jobs = workflow.get("jobs")
        assert isinstance(jobs, dict)
        build_job = jobs.get("build-workspace-candidate")
        assert isinstance(build_job, dict)
        assert build_job.get("runs-on") == "ubuntu-latest", (
            "workspace candidate builds must use hosted clean-cloud egress; "
            "the .201 self-hosted Docker daemon is a known unreliable ECR push lane"
        )

    def test_workflow_builds_workspace_mode_with_candidate_stamp(
        self, workflow_text: str
    ) -> None:
        assert "BUILD_SOURCE=workspace" in workflow_text
        assert "EXPECTED_BUILD_SOURCE=workspace" in workflow_text
        assert "PROMOTION_CLASS=stability-candidate" in workflow_text
        assert "NON_MAIN_LINEAGE=true" in workflow_text

    def test_workflow_stages_siblings(self, workflow_text: str) -> None:
        assert "scripts/runtime_build/stage_workspace.sh" in workflow_text

    def test_stage_step_host_python_has_every_third_party_import(
        self, workflow: dict[object, object]
    ) -> None:
        """Install every host-side staging dependency before staging (OMN-19998)."""
        steps = _candidate_steps(workflow)
        installed = _stage_host_distributions(steps)
        import_to_distribution = {"yaml": "pyyaml"}
        sources = _stage_python_sources(
            _REPO_ROOT / "scripts/runtime_build/stage_workspace.sh"
        )
        for source_path, source in sources.items():
            tree = ast.parse(source, filename=source_path)
            imports: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split(".")[0] for alias in node.names)
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module
                    and node.module != "__future__"
                ):
                    imports.add(node.module.split(".")[0])
            for name in sorted(imports - sys.stdlib_module_names):
                distribution = import_to_distribution.get(name, name)
                assert canonicalize_name(distribution) in installed, (
                    f"missing host dependency {distribution!r} imported by {source_path}"
                )

    def test_workflow_runs_unweakened_trivy_scan(self, workflow_text: str) -> None:
        # SAME scan as the prod path: CRITICAL,HIGH, --ignore-unfixed, exit 1.
        assert "aquasecurity/trivy-action" in workflow_text
        assert "severity: 'CRITICAL,HIGH'" in workflow_text
        assert "ignore-unfixed: true" in workflow_text
        assert "exit-code: '1'" in workflow_text
        # No advisory / continue-on-error softener on the scan.
        assert "continue-on-error" not in workflow_text

    def test_workflow_attests_source_hash(self, workflow_text: str) -> None:
        assert "OMN-9139" in workflow_text
        assert "RUNTIME_SOURCE_HASH=${{ github.sha }}" in workflow_text
        assert "sed -n 's/^RUNTIME_SOURCE_HASH=//p'" in workflow_text
        assert 'org.opencontainers.image.revision" }}' not in workflow_text

    def test_workflow_pins_third_party_actions(self, workflow_text: str) -> None:
        action_refs = re.findall(r"^\s*uses:\s*([^\s#]+)", workflow_text, re.MULTILINE)
        assert action_refs
        for action_ref in action_refs:
            assert "@" in action_ref, f"{action_ref} is missing an immutable ref"
            _, ref = action_ref.rsplit("@", 1)
            assert re.fullmatch(r"[0-9a-f]{40}", ref), (
                f"{action_ref} is not pinned to a 40-character commit SHA"
            )

    def test_workflow_does_not_interpolate_sibling_ref_in_shell(
        self, workflow_text: str
    ) -> None:
        assert '--arg sibling_ref "${SIBLING_REF}"' in workflow_text
        assert 'echo "| sibling ref | \\`${SIBLING_REF}\\` |"' in workflow_text
        assert '--arg sibling_ref "${{ inputs.sibling_ref }}"' not in workflow_text
        assert "`${{ inputs.sibling_ref }}`" not in workflow_text

    def test_workflow_emits_provenance_manifest(self, workflow_text: str) -> None:
        assert 'build_source: "workspace"' in workflow_text
        assert 'promotion_class: "stability-candidate"' in workflow_text
        assert "non_main_lineage: true" in workflow_text
        assert "per_sibling_vcs_provenance" in workflow_text
        # Candidate is pinnable to dev/stability only — never prod.
        assert "prod_pinnable: false" in workflow_text

    def test_workflow_asserts_lineage_labels_before_push(
        self, workflow_text: str
    ) -> None:
        # The lineage-label assertion must precede the push step in the file so a
        # mis-stamped image is never pushed.
        assert "Assert stability-candidate lineage labels" in workflow_text
        assert_idx = workflow_text.index("Assert stability-candidate lineage labels")
        push_idx = workflow_text.index("Push candidate image to Amazon ECR")
        assert assert_idx < push_idx

    def test_workflow_composes_omnimarket_layer_before_scan(
        self, workflow: dict[object, object]
    ) -> None:
        steps = _candidate_steps(workflow)
        compose_idx = _step_index(
            steps, "Compose the omnimarket layer onto the candidate (OMN-21279)"
        )
        assert (
            _step_index(steps, "Build runtime image (workspace stability-candidate)")
            < compose_idx
        )
        for step_name in (
            "Generate unignored Trivy audit report (OMN-16229)",
            "Run Trivy vulnerability scanner",
            "Assert required config paths exist in the built image (OMN-15676)",
            "Push candidate image to Amazon ECR",
        ):
            assert compose_idx < _step_index(steps, step_name)

    def test_workflow_composes_from_the_staged_omnimarket_tree(
        self, workflow: dict[object, object]
    ) -> None:
        steps = _candidate_steps(workflow)
        step = steps[
            _step_index(
                steps, "Compose the omnimarket layer onto the candidate (OMN-21279)"
            )
        ]
        assert (
            step["env"]["OMNIMARKET_TREE"]
            == "${{ steps.stage-siblings.outputs.workspace_root }}/omnimarket"
        )
        assert step["env"]["CANDIDATE_IMAGE"] == "${{ steps.meta.outputs.tags }}"
        assert '--file "${OMNIMARKET_TREE}/Dockerfile.runtime"' in step["run"]
        assert '--build-arg "INFRA_RUNTIME_IMAGE=${INFRA_IMAGE}"' in step["run"]
        assert '--build-arg "OMNIMARKET_VCS_REF=${OMNIMARKET_VCS_REF}"' in step["run"]
        assert '--tag "${CANDIDATE_IMAGE}"' in step["run"]
        assert 'docker tag "${CANDIDATE_IMAGE}" "${INFRA_IMAGE}"' in step["run"]
        assert "set -euo pipefail" in step["run"]
        assert "import omnimarket" in step["run"]

    def test_workflow_has_no_skip_or_bypass_tokens(self, workflow_text: str) -> None:
        assert "[skip-" not in workflow_text
        assert "--no-verify" not in workflow_text


@pytest.mark.unit
@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/candidate_topic_provision_no_omnimarket_omn19998.json"
)
def test_recorded_infra_only_candidate_is_composed_before_push(
    recorded_response: dict[str, object], workflow: dict[object, object]
) -> None:
    """The recorded run 40272282023 predates the compose step.

    Pin the failure that step answers: an infra-only candidate built, then
    could not import omnimarket in the lab topic-provision Job. Check that the
    job now carries the import probe. Step order is the ordering test's concern.
    """
    response = recorded_response["response"]
    assert isinstance(response, dict)
    jobs = {job["name"]: job for job in response["jobs"]}
    build = jobs[
        "Build runtime stability-candidate / Build + scan + push workspace stability-candidate"
    ]
    smoke = jobs["Candidate render/wiring smoke on an ephemeral kind cluster"]
    assert build["conclusion"] == "success"
    assert smoke["conclusion"] == "failure"
    assert (
        "Provision the contract-declared topics on the lane (OMN-18264)"
        in smoke["failed_steps"]
    )
    log_lines = response["job_log"]["lines"]
    assert any(
        "FATAL omnimarket contracts unresolvable: No module named 'omnimarket'" in line
        for line in log_lines
    )

    steps = _candidate_steps(workflow)
    compose = _step_index(
        steps, "Compose the omnimarket layer onto the candidate (OMN-21279)"
    )
    assert "import omnimarket" in steps[compose]["run"]
