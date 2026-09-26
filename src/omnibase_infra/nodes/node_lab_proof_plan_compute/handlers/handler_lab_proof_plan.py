# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""HandlerLabProofPlan — renders one lab proof run as ordered argv steps.

Canonical definition-B handler: ``handle(ModelLabProofPlanRequest) ->
ModelLabProofPlan``. Pure: it reads nothing from disk or network, and the same
request always renders the same plan, so a recorded plan replays exactly.

WHAT A foundation_override PROOF IS (OMN-19572). A foundation repository
(omnibase_core, omnibase_spi, omnibase_compat) reaches the runtime only as a
package pinned by omnibase_infra's lock. CI proves the package against its own
tests; it cannot see a change that breaks the runtime that installs it. So the
plan builds the laptop-bundle runtime image from omnibase_infra at a pinned
commit, installs the PR head's package tree OVER the image's pinned copy in a
derived image, boots the bundle on that image, and reads back:

  identity     the installed package tree hashes equal to the PR head's tree,
               in both runtime containers (not a label, not a URL: the bytes)
  health       migration gate, runtime main and runtime effects Docker-healthy
  imports      every consumer module imports inside both runtime containers
  wiring       no "Auto-wiring failed" or duplicate-dispatcher line in either log
  golden chain one delegation round trip through the effects kernel on the bus
  tests        the PR's changed test files, in the PR clone

Teardown and the zero-residue readback are always planned, and always run.

SAFETY BOUNDS RENDERED HERE, not trusted to the caller:
  * every path is under ``<lane_root>/<run_key>`` or ``<lane_root>/logs/<run_key>``;
  * only public https GitHub clones of the profile's own repository and of
    omnibase_infra, at exact 40-character commits;
  * the stack is the bundle's compose project only; nothing else is touched.

Ticket: OMN-19572 (the profile); OMN-19565 (the registry it reads)
"""

from __future__ import annotations

from urllib.parse import urlsplit

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.lab_proof.enum_lab_proof_attribution import (
    EnumLabProofAttribution,
)
from omnibase_infra.lab_proof.enum_lab_proof_execution import EnumLabProofExecution
from omnibase_infra.lab_proof.enum_lab_proof_kind import (
    EnumLabProofKind,
)
from omnibase_infra.lab_proof.enum_lab_proof_step_id import EnumLabProofStepId
from omnibase_infra.lab_proof.enum_lab_proof_step_phase import EnumLabProofStepPhase
from omnibase_infra.lab_proof.model_lab_proof_plan import ModelLabProofPlan
from omnibase_infra.lab_proof.model_lab_proof_plan_request import (
    ModelLabProofPlanRequest,
)
from omnibase_infra.lab_proof.model_lab_proof_retry import ModelLabProofRetry
from omnibase_infra.lab_proof.model_lab_proof_step import ModelLabProofStep

PLANNED_PROOF_KINDS: frozenset[EnumLabProofKind] = frozenset(
    {EnumLabProofKind.FOUNDATION_OVERRIDE}
)
COMPOSE_PROJECT_LABEL = "com.docker.compose.project"
SABOTAGE_LINE = 'raise ImportError("lab-proof negative control: deliberate sabotage")\n'

_H = EnumLabProofAttribution.HARNESS
_S = EnumLabProofAttribution.SUBJECT
_SETUP = EnumLabProofStepPhase.SETUP
_PROVE = EnumLabProofStepPhase.PROVE
_TEARDOWN = EnumLabProofStepPhase.TEARDOWN
_RESIDUE = EnumLabProofStepPhase.RESIDUE
_ID = EnumLabProofStepId

# --- small host-side programs, passed to ``python3 -c`` as argv ---------------
# Each takes its inputs as argv and prints one line; none reads the environment.

HOST_LOAD_PY = (
    "import os, sys\n"
    "load1 = os.getloadavg()[0]\n"
    "cpus = os.cpu_count() or 1\n"
    "ratio = load1 / cpus\n"
    "print(f'load1={load1:.2f} cpus={cpus} ratio={ratio:.3f} max={sys.argv[1]}')\n"
    "sys.exit(0 if ratio <= float(sys.argv[1]) else 3)\n"
)

PORTS_OPEN_PY = (
    "import socket, sys\n"
    "open_ports = []\n"
    "for port in sys.argv[1:]:\n"
    "    sock = socket.socket()\n"
    "    sock.settimeout(0.5)\n"
    "    try:\n"
    "        if sock.connect_ex(('127.0.0.1', int(port))) == 0:\n"
    "            open_ports.append(port)\n"
    "    finally:\n"
    "        sock.close()\n"
    "print(' '.join(open_ports))\n"
)

EXTRACT_PY = (
    "import os, sys, tarfile\n"
    "os.makedirs(sys.argv[2], exist_ok=True)\n"
    "with tarfile.open(sys.argv[1]) as archive:\n"
    "    archive.extractall(sys.argv[2], filter='data')\n"
    "print('extracted')\n"
)

SABOTAGE_PY = (
    "import pathlib, sys\n"
    "target = pathlib.Path(sys.argv[1])\n"
    "if not target.is_file():\n"
    "    sys.exit(f'sabotage target missing: {target}')\n"
    "with target.open('a', encoding='utf-8') as handle:\n"
    "    handle.write('\\n' + sys.argv[2])\n"
    "print(f'sabotaged {target.name}')\n"
)

# Hash a package tree: every .py file, sorted by relative path. With ``path`` it
# hashes a directory; with ``module`` it finds the installed package WITHOUT
# importing it (find_spec does not execute __init__), so a sabotaged package can
# still be identified, and it adds the distribution's version and direct_url.
TREE_HASH_PY = (
    "import hashlib, json, pathlib, sys\n"
    "mode = sys.argv[1]\n"
    "extra = {}\n"
    "if mode == 'path':\n"
    "    root = pathlib.Path(sys.argv[2])\n"
    "else:\n"
    "    import importlib.metadata as metadata, importlib.util as util\n"
    "    spec = util.find_spec(sys.argv[2])\n"
    "    root = pathlib.Path(list(spec.submodule_search_locations)[0])\n"
    "    dist = metadata.distribution(sys.argv[3])\n"
    "    extra = {'version': dist.version, "
    "'direct_url': dist.read_text('direct_url.json') or '', 'root': str(root)}\n"
    "digest = hashlib.sha256()\n"
    "count = 0\n"
    "for path in sorted(root.rglob('*.py')):\n"
    "    if '__pycache__' in path.parts:\n"
    "        continue\n"
    "    digest.update(path.relative_to(root).as_posix().encode())\n"
    "    digest.update(b'\\0')\n"
    "    digest.update(path.read_bytes())\n"
    "    digest.update(b'\\0')\n"
    "    count += 1\n"
    "print(json.dumps({'sha256': digest.hexdigest(), 'files': count, **extra}, "
    "sort_keys=True))\n"
)

IMPORT_SMOKE_PY = (
    "import importlib, sys\n"
    "for name in sys.argv[1:]:\n"
    "    importlib.import_module(name)\n"
    "print('imported ' + ' '.join(sys.argv[1:]))\n"
)

# Points the bundle's model overlay at the lab model server and sets the three
# legacy URLs the laptop bundle defaults to host.docker.internal, which does not
# resolve on a Linux lab host (FRICTION ledger:3503).
MODEL_ENDPOINT_PY = (
    "import re, sys\n"
    "overlay, env_file, endpoint, base = sys.argv[1:5]\n"
    "text = open(overlay, encoding='utf-8').read()\n"
    'new, count = re.subn(r\'(endpoint_url: &model_endpoint )"[^"]*"\', '
    "lambda m: m.group(1) + '\"' + endpoint + '\"', text)\n"
    "if count != 1:\n"
    "    sys.exit(f'expected one model_endpoint line in the overlay, found {count}')\n"
    "open(overlay, 'w', encoding='utf-8').write(new)\n"
    "with open(env_file, 'a', encoding='utf-8') as handle:\n"
    "    for key in ('LLM_CODER_URL', 'LLM_CODER_FAST_URL', 'LLM_DEEPSEEK_R1_URL'):\n"
    "        handle.write(f'{key}={base}\\n')\n"
    "print(f'model endpoint set to {endpoint}')\n"
)

REMOVE_TREE_PY = (
    "import os, shutil, sys\n"
    "target, lane_root, run_key = sys.argv[1:4]\n"
    "if not target.startswith(lane_root + '/') or os.path.basename(target) != run_key:\n"
    "    sys.exit(f'refusing to remove {target}: not {lane_root}/{run_key}')\n"
    "shutil.rmtree(target, ignore_errors=False) if os.path.exists(target) else None\n"
    "print('removed' if not os.path.exists(target) else 'still present')\n"
)

PATH_ABSENT_PY = (
    "import os, sys\nprint('present' if os.path.exists(sys.argv[1]) else '')\n"
)


class LabProofPlanError(ValueError):
    """The request cannot be rendered into a safe plan."""


def _is_test_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return (
        path.startswith("tests/")
        and path.endswith(".py")
        and (name.startswith("test_") or name.endswith("_test.py"))
    )


class HandlerLabProofPlan:
    """Pure rendering of one proof run."""

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: compute handler."""
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: pure compute, no external I/O."""
        return EnumHandlerTypeCategory.COMPUTE

    def handle(self, request: ModelLabProofPlanRequest) -> ModelLabProofPlan:
        """Render the plan, or raise LabProofPlanError naming why it cannot be."""
        variant = request.variant
        if variant.execution is not EnumLabProofExecution.NODE:
            raise LabProofPlanError(
                f"variant {variant.variant_key} runs by {variant.execution}, not by nodes"
            )
        if variant.proof_kind not in PLANNED_PROOF_KINDS:
            raise LabProofPlanError(
                f"no node plan exists for proof kind {variant.proof_kind}"
            )
        if request.subject.repo != request.profile_repo:
            raise LabProofPlanError(
                f"subject {request.subject.repo} is not this profile's repository "
                f"{request.profile_repo}"
            )
        if request.negative_control and variant.negative_control is None:
            raise LabProofPlanError("negative control asked for, none declared")
        return ModelLabProofPlan(
            run_key=request.run_key,
            host=request.host,
            profile_key=request.profile_key,
            profile_version=request.profile_version,
            variant_key=variant.variant_key,
            proof_kind=variant.proof_kind,
            subject=request.subject,
            infra_sha=request.infra_sha,
            workdir=f"{request.lane_root}/{request.run_key}",
            log_dir=f"{request.lane_root}/logs/{request.run_key}",
            negative_control=request.negative_control,
            steps=tuple(self._foundation_override_steps(request)),
        )

    def _foundation_override_steps(
        self, request: ModelLabProofPlanRequest
    ) -> list[ModelLabProofStep]:
        variant = request.variant
        foundation = variant.foundation
        host_selector = variant.host_selector
        negative = variant.negative_control
        if foundation is None or host_selector is None or negative is None:
            raise LabProofPlanError("foundation_override variant is incomplete")
        bundle = request.bundle
        subject = request.subject
        lane = request.lane_root
        work = f"{lane}/{request.run_key}"
        infra = f"{work}/omnibase_infra"
        subject_dir = f"{work}/subject"
        context = f"{work}/override"
        env_file = f"{work}/local.env"
        overlay = f"{work}/local.bifrost.yaml"
        image_latest = f"{bundle.runtime_image}:latest"
        image_base = f"{bundle.runtime_image}:lab-proof-base-{request.run_key}"
        project_filter = f"label={COMPOSE_PROJECT_LABEL}={bundle.compose_project}"
        ports = [str(port) for port in bundle.host_ports]
        endpoint = request.model_endpoint_url
        parts = urlsplit(endpoint)
        endpoint_base = f"{parts.scheme}://{parts.netloc}"
        package_tree = (
            f"{context}/subject/{foundation.source_root}/{foundation.import_package}"
        )
        cli = [
            "uv",
            "run",
            "--frozen",
            "python",
            "-m",
            "omnibase_infra.docker.catalog.cli",
        ]
        health = ModelLabProofRetry(
            interval_seconds=bundle.health_interval_seconds,
            deadline_seconds=bundle.health_deadline_seconds,
        )

        def step(
            step_id: EnumLabProofStepId,
            phase: EnumLabProofStepPhase,
            attribution: EnumLabProofAttribution,
            purpose: str,
            argv: list[str],
            *,
            cwd: str = lane,
            timeout: int = 120,
            must_succeed: bool = True,
            **extra: object,
        ) -> ModelLabProofStep:
            return ModelLabProofStep.model_validate(
                {
                    "step_id": step_id,
                    "phase": phase,
                    "attribution": attribution,
                    "purpose": purpose,
                    "argv": tuple(argv),
                    "cwd": cwd,
                    "timeout_seconds": timeout,
                    "must_succeed": must_succeed,
                    **extra,
                }
            )

        steps: list[ModelLabProofStep] = [
            step(
                _ID.HOST_LOAD,
                _SETUP,
                _H,
                "refuse a host above its load ceiling",
                ["python3", "-c", HOST_LOAD_PY, str(host_selector.max_load_ratio)],
            ),
            step(
                _ID.PREFLIGHT_PROJECT_ABSENT,
                _SETUP,
                _H,
                "no container of the bundle's project exists before the run",
                ["docker", "ps", "-a", "--filter", project_filter, "-q"],
                expect_stdout_empty=True,
            ),
            step(
                _ID.PREFLIGHT_IMAGE_ABSENT,
                _SETUP,
                _H,
                "no runtime image of the bundle exists before the run",
                [
                    "docker",
                    "images",
                    "--filter",
                    f"reference={bundle.runtime_image}",
                    "-q",
                ],
                expect_stdout_empty=True,
            ),
            step(
                _ID.PREFLIGHT_PORTS_FREE,
                _SETUP,
                _H,
                "the bundle's host ports are free before the run",
                ["python3", "-c", PORTS_OPEN_PY, *ports],
                expect_stdout_empty=True,
            ),
            step(
                _ID.INFRA_INIT,
                _SETUP,
                _H,
                "an empty omnibase_infra clone",
                ["git", "init", "-q", infra],
            ),
            step(
                _ID.INFRA_FETCH,
                _SETUP,
                _H,
                "fetch omnibase_infra at the pinned commit",
                [
                    "git",
                    "-C",
                    infra,
                    "fetch",
                    "-q",
                    "--depth",
                    "1",
                    f"{bundle.clone_base_url}/{bundle.infra_repo}.git",
                    request.infra_sha,
                ],
                timeout=600,
            ),
            step(
                _ID.INFRA_CHECKOUT,
                _SETUP,
                _H,
                "check out the pinned commit",
                [
                    "git",
                    "-C",
                    infra,
                    "-c",
                    "advice.detachedHead=false",
                    "checkout",
                    "-q",
                    "FETCH_HEAD",
                ],
            ),
            step(
                _ID.INFRA_REV,
                _SETUP,
                _H,
                "the infra clone is at the pinned commit",
                ["git", "-C", infra, "rev-parse", "HEAD"],
                expect_stdout_equals=request.infra_sha,
            ),
            step(
                _ID.SUBJECT_INIT,
                _SETUP,
                _H,
                "an empty clone of the subject repository",
                ["git", "init", "-q", subject_dir],
            ),
            step(
                _ID.SUBJECT_FETCH,
                _SETUP,
                _H,
                "fetch the commit under test",
                [
                    "git",
                    "-C",
                    subject_dir,
                    "fetch",
                    "-q",
                    "--depth",
                    "1",
                    f"{bundle.clone_base_url}/{subject.repo}.git",
                    subject.fetch_ref,
                ],
                timeout=600,
            ),
            step(
                _ID.SUBJECT_CHECKOUT,
                _SETUP,
                _H,
                "check out the commit under test",
                [
                    "git",
                    "-C",
                    subject_dir,
                    "-c",
                    "advice.detachedHead=false",
                    "checkout",
                    "-q",
                    "FETCH_HEAD",
                ],
            ),
            step(
                _ID.SUBJECT_REV,
                _SETUP,
                _H,
                "the fetched commit is exactly the one under test (a moved head stops here)",
                ["git", "-C", subject_dir, "rev-parse", "HEAD"],
                expect_stdout_equals=subject.proved_sha,
            ),
        ]
        tests = sorted(path for path in subject.changed_files if _is_test_file(path))
        if tests:
            steps.append(
                step(
                    _ID.FOCUSED_TESTS,
                    _SETUP,
                    _S,
                    "the PR's changed test files",
                    [
                        "uv",
                        "run",
                        "--frozen",
                        "pytest",
                        "-q",
                        "-p",
                        "no:cacheprovider",
                        *tests,
                    ],
                    cwd=subject_dir,
                    timeout=1800,
                    must_succeed=False,
                )
            )
        steps += [
            step(
                _ID.SUBJECT_ARCHIVE,
                _SETUP,
                _H,
                "the committed tree at the commit under test, nothing uncommitted",
                [
                    "git",
                    "-C",
                    subject_dir,
                    "archive",
                    "--format=tar",
                    "--prefix=subject/",
                    "-o",
                    f"{work}/subject.tar",
                    "HEAD",
                ],
            ),
            step(
                _ID.SUBJECT_EXTRACT,
                _SETUP,
                _H,
                "unpack it as the override build context",
                ["python3", "-c", EXTRACT_PY, f"{work}/subject.tar", context],
            ),
        ]
        if request.negative_control:
            sabotage_path = negative.sabotage_path
            if sabotage_path is None:
                raise LabProofPlanError("negative control needs a sabotage_path")
            steps.append(
                step(
                    _ID.NEGATIVE_CONTROL_SABOTAGE,
                    _SETUP,
                    _H,
                    "negative control: the package raises on import; this run must FAIL",
                    [
                        "python3",
                        "-c",
                        SABOTAGE_PY,
                        f"{context}/subject/{sabotage_path}",
                        SABOTAGE_LINE,
                    ],
                )
            )
        steps += [
            step(
                _ID.SUBJECT_HASH,
                _SETUP,
                _H,
                "hash of the package tree being installed",
                ["python3", "-c", TREE_HASH_PY, "path", package_tree],
            ),
            step(
                _ID.OVERRIDE_DOCKERFILE,
                _SETUP,
                _H,
                "the derived-image Dockerfile",
                [
                    "cp",
                    f"{request.harness_root}/{bundle.override_dockerfile}",
                    f"{context}/Dockerfile",
                ],
            ),
            step(
                _ID.LOCAL_ENV,
                _SETUP,
                _H,
                "the bundle's env file and model overlay, inside the run directory",
                [
                    "make",
                    "local-env",
                    f"LOCAL_ENV_FILE={env_file}",
                    f"LOCAL_OVERLAY_FILE={overlay}",
                ],
                cwd=infra,
            ),
            step(
                _ID.MODEL_ENDPOINT,
                _SETUP,
                _H,
                "point delegation at the lab model server",
                [
                    "python3",
                    "-c",
                    MODEL_ENDPOINT_PY,
                    overlay,
                    env_file,
                    endpoint,
                    endpoint_base,
                ],
            ),
            step(
                _ID.GENERATE,
                _SETUP,
                _H,
                "render the bundle's compose file",
                [*cli, "generate", bundle.bundle, "--env-file", env_file],
                cwd=infra,
                timeout=900,
            ),
            step(
                _ID.BUILD_BASE,
                _SETUP,
                _H,
                "build the runtime image at the pinned infra commit",
                [
                    "docker",
                    "compose",
                    "-f",
                    "docker/docker-compose.generated.yml",
                    "--env-file",
                    bundle.runtime_policy_env,
                    "--env-file",
                    env_file,
                    "build",
                    "--build-arg",
                    f"GIT_SHA={request.infra_sha}",
                    "--build-arg",
                    f"VCS_REF={request.infra_sha}",
                ],
                cwd=infra,
                timeout=3600,
            ),
            step(
                _ID.TAG_BASE,
                _SETUP,
                _H,
                "keep the base image under a run-scoped tag",
                ["docker", "tag", image_latest, image_base],
            ),
            step(
                _ID.BUILD_OVERRIDE,
                _SETUP,
                _S,
                "install the commit under test over the image's pinned package",
                [
                    "docker",
                    "build",
                    "-f",
                    f"{context}/Dockerfile",
                    "--build-arg",
                    f"BASE_IMAGE={image_base}",
                    "--build-arg",
                    f"UV_IMAGE={bundle.uv_image}",
                    "--build-arg",
                    f"PACKAGE_NAME={foundation.distribution}",
                    "--build-arg",
                    f"PROVED_SHA={subject.proved_sha}",
                    "--build-arg",
                    f"LAB_PROOF_RUN={request.run_key}",
                    "-t",
                    image_latest,
                    context,
                ],
                timeout=1800,
            ),
            step(
                _ID.UP,
                _SETUP,
                _S,
                "boot the bundle on the derived image (no rebuild)",
                [*cli, "up", bundle.bundle, "--env-file", env_file],
                cwd=infra,
                timeout=1800,
                must_succeed=False,
            ),
        ]
        containers = (
            (
                _ID.HEALTH_RUNTIME_MAIN,
                _ID.IDENTITY_RUNTIME_MAIN,
                _ID.IMPORT_SMOKE_RUNTIME_MAIN,
                _ID.WIRING_LOGS_RUNTIME_MAIN,
                bundle.container_runtime_main,
            ),
            (
                _ID.HEALTH_RUNTIME_EFFECTS,
                _ID.IDENTITY_RUNTIME_EFFECTS,
                _ID.IMPORT_SMOKE_RUNTIME_EFFECTS,
                _ID.WIRING_LOGS_RUNTIME_EFFECTS,
                bundle.container_runtime_effects,
            ),
        )
        steps.append(
            step(
                _ID.HEALTH_MIGRATION_GATE,
                _PROVE,
                _S,
                "migration gate Docker-healthy",
                [
                    "docker",
                    "inspect",
                    "-f",
                    "{{.State.Health.Status}}",
                    bundle.container_migration_gate,
                ],
                timeout=30,
                must_succeed=False,
                retry=health,
                expect_stdout_equals="healthy",
            )
        )
        for health_id, _identity, _smoke, _logs, container in containers:
            steps.append(
                step(
                    health_id,
                    _PROVE,
                    _S,
                    f"{container} Docker-healthy",
                    ["docker", "inspect", "-f", "{{.State.Health.Status}}", container],
                    timeout=30,
                    must_succeed=False,
                    retry=health,
                    expect_stdout_equals="healthy",
                )
            )
        for _health, identity_id, smoke_id, logs_id, container in containers:
            python = [
                "docker",
                "exec",
                container,
                "/app/.venv/bin/python",
                "-c",
            ]
            steps += [
                step(
                    identity_id,
                    _PROVE,
                    _S,
                    f"{container} carries the commit under test, byte for byte",
                    [
                        *python,
                        TREE_HASH_PY,
                        "module",
                        foundation.import_package,
                        foundation.distribution,
                    ],
                    timeout=60,
                    must_succeed=False,
                ),
                step(
                    smoke_id,
                    _PROVE,
                    _S,
                    f"every consumer imports in {container}",
                    [*python, IMPORT_SMOKE_PY, *bundle.consumer_modules],
                    timeout=180,
                    must_succeed=False,
                ),
                step(
                    logs_id,
                    _PROVE,
                    _S,
                    f"{container} wired every handler",
                    ["docker", "logs", container],
                    timeout=60,
                    must_succeed=False,
                    grep_patterns=bundle.wiring_failure_patterns,
                    extract_pattern=bundle.wiring_failure_extract,
                    record_output=False,
                ),
            ]
        steps.append(
            step(
                _ID.GOLDEN_CHAIN_DELEGATION,
                _PROVE,
                _S,
                "one delegation round trip through the effects kernel on the bus",
                [
                    "docker",
                    "exec",
                    bundle.container_runtime_effects,
                    "onex",
                    "delegate",
                    bundle.delegation_prompt,
                    "--bus",
                    "kafka",
                    "--kafka-bootstrap",
                    "redpanda:9092",
                    "--locus",
                    "deployed-lane",
                ],
                timeout=bundle.delegation_timeout_seconds,
                must_succeed=False,
            )
        )
        steps += [
            step(
                _ID.DOWN,
                _TEARDOWN,
                _H,
                "stop the bundle and delete its volumes",
                [*cli, "down", "--volumes"],
                cwd=infra,
                timeout=600,
                must_succeed=False,
            ),
            step(
                _ID.REMOVE_IMAGES,
                _TEARDOWN,
                _H,
                "delete both run images",
                ["docker", "image", "rm", image_latest, image_base],
                timeout=300,
                must_succeed=False,
            ),
            step(
                _ID.REMOVE_WORKDIR,
                _TEARDOWN,
                _H,
                "delete the run directory",
                ["python3", "-c", REMOVE_TREE_PY, work, lane, request.run_key],
                timeout=300,
                must_succeed=False,
            ),
            step(
                _ID.RESIDUE_CONTAINERS,
                _RESIDUE,
                _H,
                "zero containers of the project",
                ["docker", "ps", "-a", "--filter", project_filter, "-q"],
                must_succeed=False,
                expect_stdout_empty=True,
            ),
            step(
                _ID.RESIDUE_VOLUMES,
                _RESIDUE,
                _H,
                "zero volumes of the project",
                ["docker", "volume", "ls", "--filter", project_filter, "-q"],
                must_succeed=False,
                expect_stdout_empty=True,
            ),
            step(
                _ID.RESIDUE_NETWORKS,
                _RESIDUE,
                _H,
                "zero networks of the project",
                ["docker", "network", "ls", "--filter", project_filter, "-q"],
                must_succeed=False,
                expect_stdout_empty=True,
            ),
            step(
                _ID.RESIDUE_IMAGES,
                _RESIDUE,
                _H,
                "zero runtime images of the bundle",
                [
                    "docker",
                    "images",
                    "--filter",
                    f"reference={bundle.runtime_image}",
                    "-q",
                ],
                must_succeed=False,
                expect_stdout_empty=True,
            ),
            step(
                _ID.RESIDUE_PORTS,
                _RESIDUE,
                _H,
                "the bundle's host ports are free again",
                ["python3", "-c", PORTS_OPEN_PY, *ports],
                must_succeed=False,
                expect_stdout_empty=True,
            ),
            step(
                _ID.RESIDUE_WORKDIR,
                _RESIDUE,
                _H,
                "the run directory is gone",
                ["python3", "-c", PATH_ABSENT_PY, work],
                must_succeed=False,
                expect_stdout_empty=True,
            ),
            step(
                _ID.RESIDUE_POSITIVE_CONTROL,
                _RESIDUE,
                _H,
                "the same label filter reads a project known to be running (a zero "
                "above is only evidence when this is non-zero)",
                [
                    "docker",
                    "ps",
                    "--filter",
                    f"label={COMPOSE_PROJECT_LABEL}={request.positive_control_project}",
                    "-q",
                ],
                must_succeed=False,
                expect_stdout_nonempty=True,
            ),
        ]
        return steps


__all__ = [
    "PLANNED_PROOF_KINDS",
    "SABOTAGE_LINE",
    "HandlerLabProofPlan",
    "LabProofPlanError",
]
