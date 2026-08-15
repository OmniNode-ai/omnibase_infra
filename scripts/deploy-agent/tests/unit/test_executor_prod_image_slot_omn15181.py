# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RED-first reproduction + fix coverage for the OMN-15181 round-3 Finding 9 defect.

Live-reproduced 2026-07-26 on omninode-pc (ledger 13:35Z/13:40Z PREFLIGHT-STOP
entries, "Finding 9"): ``docker/docker-compose.prod.yml`` declares
``omninode-runtime`` and ``runtime-effects`` with only a ``build:`` stanza
(inherited from the shared ``x-runtime-base`` anchor in
``docker-compose.infra.yml``) and no ``image:`` key anywhere. Docker Compose
therefore auto-derives the image reference these services resolve to at
``up``/``config`` time as ``<compose_project>-<service>`` (confirmed live via
``docker compose ... --profile runtime config --images``:
``omnibase-infra-prod-omninode-runtime`` / ``omnibase-infra-prod-runtime-effects``)
-- a name that has zero relationship to any digest a deploy-agent prod
dispatch pins. ``_pull_pinned_image()`` (OMN-15181 round 2) only proves the
pinned digest exists *somewhere* in the local docker store (normally under
the stability-test tag); nothing repointed the prod service's resolved image
at it, so ``docker compose -p omnibase-infra-prod up -d --force-recreate
--pull never <service>`` always recreated from the stale auto-derived tag
regardless of what was pinned. ``verify_running_image_digest`` then correctly
fails closed post-recreate (``DigestMismatchError``) -- the defect never
produces a false "success", but the deploy is mechanically incapable of ever
landing a promoted digest.

Fix (declarative, no ``docker tag``/retag command):

1. ``docker-compose.prod.yml`` gains a parameterized ``image:`` field on both
   runtime services: ``${PROD_OMNINODE_RUNTIME_IMAGE:-omnibase-infra-prod-omninode-runtime:latest}``
   / ``${PROD_RUNTIME_EFFECTS_IMAGE:-omnibase-infra-prod-runtime-effects:latest}``.
   The default resolves to the EXACT pre-existing auto-derived name -- a
   no-op for every caller that does not set the override (dev/stability-test
   builds, and any prod dispatch that does not pin a digest, are unaffected).
2. ``DeployExecutor._resolve_prod_image_env`` resolves the pinned digest to an
   already-locally-present ``repository:tag`` reference
   (``docker image inspect --format '{{.RepoTags}}' <digest>`` --
   ``_resolve_local_image_reference``) and exports it as the relevant
   ``PROD_*_IMAGE`` env var for the compose invocation via
   ``_compose_env(extra_env=...)`` / ``_compose_up(..., extra_env=...)``. No
   ``docker tag`` command is ever run -- the already-existing tag (e.g. the
   stability-proven build's own tag) is used as-is.

``TestProdComposeImageSlotRealDocker`` runs the actual ``docker compose
config`` CLI against the real, committed compose files (skipped when no real
docker daemon is reachable) to prove both the no-op default and the override
mechanism against genuine Docker Compose interpolation/merge semantics -- not
a mocked stand-in. ``TestResolveProdImageEnvUnit`` covers the executor-level
resolution and wiring with mocked subprocess calls.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from deploy_agent.events import EnumRuntimeLane, Phase, PhaseStatus, Scope
from deploy_agent.executor import (
    PROD_IMAGE_ENV_VAR_FOR_SERVICE,
    DeployExecutor,
    _load_runtime_policy_env,
)

pytestmark = pytest.mark.unit

_DIGEST = "sha256:" + "e" * 64

_REPO_ROOT = Path(__file__).resolve().parents[4]
_INFRA_COMPOSE = _REPO_ROOT / "docker" / "docker-compose.infra.yml"
_PROD_OVERLAY = _REPO_ROOT / "docker" / "docker-compose.prod.yml"
_RUNTIME_POLICY_ENV = _REPO_ROOT / "docker" / "runtime-policy.env"

# Placeholder-only values (no real secrets) for every ``:?required``/``?required``
# env var referenced anywhere in docker-compose.infra.yml + docker-compose.prod.yml
# that is NOT already supplied by the committed docker/runtime-policy.env file.
# Enumerated empirically 2026-07-26 by iteratively resolving
# `docker compose ... config` failures on omninode-pc against a clean shell (no
# ~/.omnibase/.env sourced) until the merged file interpolated cleanly.
_DUMMY_BOOTSTRAP_ENV: dict[str, str] = dict.fromkeys(
    (
        "GITHUB_TOKEN",
        "DEPLOY_AGENT_HMAC_SECRET",
        "LINEAR_API_KEY",
        "LLM_CODER_FAST_URL",
        "POSTGRES_PASSWORD",
        "LLM_CODER_URL",
        "LLM_GLM_URL",
        "ONEX_SERVICE_CLIENT_SECRET",
        "LOCAL_LLM_SHARED_SECRET",
        "LLM_GLM_MODEL_NAME",
        "LLM_GLM_API_KEY",
        "LLM_ENDPOINT_CIDR_ALLOWLIST",
        "LLM_EMBEDDING_URL",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "ONEX_REGISTRATION_AUTO_ACK",
        "LLM_DEEPSEEK_R1_URL",
        "INFISICAL_DB_CONNECTION_URI",
        "INFISICAL_REDIS_URL",
        "INFISICAL_ENCRYPTION_KEY",
        "INFISICAL_AUTH_SECRET",
        "PROD_REDPANDA_ADVERTISE_HOST",
        # OMN-15378: docker-compose.infra.yml (merged as the base file for
        # EVERY lane, including prod) hard-requires DEV_REDPANDA_ADVERTISE_HOST
        # with no default (OMN-15173's deliberate no-silent-default design).
        # The original 2026-07-26 enumeration missed it because the author's
        # shell already had it set via a sourced ~/.omnibase/.env -- these
        # tests never ran anywhere else (this uncollected-tests-root ticket)
        # until now, so a CI runner's clean shell was the first environment to
        # expose the gap.
        "DEV_REDPANDA_ADVERTISE_HOST",
    ),
    "dummy-placeholder-value",
)


def _docker_daemon_reachable() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


requires_docker = pytest.mark.skipif(
    not _docker_daemon_reachable(),
    reason="requires a real, reachable docker daemon (self-hosted CI / omninode-pc)",
)


def _prod_compose_config(extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    """Run the real `docker compose config` against the committed prod overlay.

    Merges docker/runtime-policy.env (real, committed) with the placeholder
    bootstrap dict above, plus any test-supplied override, then resolves the
    full merged compose config as JSON -- genuine Docker Compose
    interpolation/anchor-merge semantics, no PyYAML approximation.
    """
    env = {**os.environ, **_load_runtime_policy_env(_RUNTIME_POLICY_ENV)}
    env.update(_DUMMY_BOOTSTRAP_ENV)
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(_INFRA_COMPOSE),
            "-f",
            str(_PROD_OVERLAY),
            "-p",
            "omnibase-infra-prod",
            "--profile",
            "runtime",
            "config",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=env,
    )
    assert result.returncode == 0, (
        f"docker compose config failed (stderr): {result.stderr}"
    )
    config: dict[str, Any] = json.loads(result.stdout)
    return config


@requires_docker
class TestProdComposeImageSlotRealDocker:
    """Executes the real docker CLI against the committed compose files."""

    def test_prod_runtime_services_have_no_bare_build_only_shape(self) -> None:
        """Static, docker-independent guard: both services must declare an
        `image:` key in the committed source -- the round-3 fix. Runs
        regardless of docker availability via the shared fixture below, but
        kept here since it is the direct source-level assertion the rest of
        this test class's live-docker behavior depends on."""
        prod_text = _PROD_OVERLAY.read_text(encoding="utf-8")
        assert "PROD_OMNINODE_RUNTIME_IMAGE" in prod_text, (
            "docker-compose.prod.yml must parameterize omninode-runtime's "
            "image: field (OMN-15181 round-3 Finding 9 fix)"
        )
        assert "PROD_RUNTIME_EFFECTS_IMAGE" in prod_text, (
            "docker-compose.prod.yml must parameterize runtime-effects' "
            "image: field (OMN-15181 round-3 Finding 9 fix)"
        )

    def test_default_resolves_to_the_exact_preexisting_autoderived_name(
        self,
    ) -> None:
        """No-op guarantee: with the override env vars unset, the resolved
        image must be byte-identical to the tag Docker Compose auto-derived
        BEFORE this fix (`<project>-<service>`, live-confirmed via
        `docker compose ... config --images` on omninode-pc as
        `omnibase-infra-prod-omninode-runtime` / `omnibase-infra-prod-runtime-effects`,
        equivalent to the same reference with an explicit `:latest`)."""
        config = _prod_compose_config()
        assert (
            config["services"]["omninode-runtime"]["image"]
            == "omnibase-infra-prod-omninode-runtime:latest"
        )
        assert (
            config["services"]["runtime-effects"]["image"]
            == "omnibase-infra-prod-runtime-effects:latest"
        )

    def test_override_env_var_repoints_the_resolved_image(self) -> None:
        """This is the round-3 fix's actual mechanism: setting
        PROD_OMNINODE_RUNTIME_IMAGE / PROD_RUNTIME_EFFECTS_IMAGE repoints the
        compose-resolved image reference declaratively -- no docker-tag/retag
        command involved, just compose env interpolation."""
        config = _prod_compose_config(
            {
                "PROD_OMNINODE_RUNTIME_IMAGE": "omnibase-infra-stability-test-omninode-runtime:latest",
                "PROD_RUNTIME_EFFECTS_IMAGE": "omnibase-infra-stability-test-runtime-effects:latest",
            }
        )
        assert (
            config["services"]["omninode-runtime"]["image"]
            == "omnibase-infra-stability-test-omninode-runtime:latest"
        )
        assert (
            config["services"]["runtime-effects"]["image"]
            == "omnibase-infra-stability-test-runtime-effects:latest"
        )

    def test_red_reproduction_without_the_image_key_resolves_to_none(
        self, tmp_path: Path
    ) -> None:
        """RED-reproduction: reconstructs the exact pre-fix shape (build: with
        no image: key) as a temp overlay copied from the real committed file
        with the fix lines stripped, proving `docker compose config` resolves
        `image: null` for a build-only service regardless of any digest
        "pinned" via an unrelated env var -- Docker Compose has no mechanism
        to consult a digest that isn't wired into the file at all. This is
        the live, pre-fix defect this PR closes."""
        prod_text = _PROD_OVERLAY.read_text(encoding="utf-8")
        pre_fix_text = "\n".join(
            line
            for line in prod_text.splitlines()
            if "PROD_OMNINODE_RUNTIME_IMAGE" not in line
            and "PROD_RUNTIME_EFFECTS_IMAGE" not in line
        )
        assert pre_fix_text != prod_text, "fix lines were not found to strip"
        pre_fix_overlay = tmp_path / "docker-compose.prod.yml"
        pre_fix_overlay.write_text(pre_fix_text, encoding="utf-8")

        env = {**os.environ, **_load_runtime_policy_env(_RUNTIME_POLICY_ENV)}
        env.update(_DUMMY_BOOTSTRAP_ENV)
        # Simulate a caller trying to pin a digest via some unrelated env var
        # -- proves the pre-fix file has literally no wiring that could ever
        # consume it, regardless of naming.
        env["SOME_UNRELATED_PINNED_DIGEST"] = "sha256:" + "f" * 64
        result = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(_INFRA_COMPOSE),
                "-f",
                str(pre_fix_overlay),
                "-p",
                "omnibase-infra-prod",
                "--profile",
                "runtime",
                "config",
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        config = json.loads(result.stdout)
        assert config["services"]["omninode-runtime"].get("image") is None
        assert config["services"]["runtime-effects"].get("image") is None
        assert "build" in config["services"]["omninode-runtime"]


class TestResolveProdImageEnvUnit:
    """Mock-based: executor-level resolution + wiring, no real docker."""

    def test_resolve_local_image_reference_returns_first_repo_tag(self) -> None:
        executor = DeployExecutor()

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            assert cmd[:3] == ["docker", "image", "inspect"]
            assert cmd[-1] == _DIGEST
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="omnibase-infra-stability-test-omninode-runtime:latest\n",
                stderr="",
            )

        with patch("deploy_agent.executor._run", side_effect=fake_run):
            ref = executor._resolve_local_image_reference(_DIGEST)
        assert ref == "omnibase-infra-stability-test-omninode-runtime:latest"

    def test_resolve_local_image_reference_fails_loud_on_dangling_image(
        self,
    ) -> None:
        executor = DeployExecutor()

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="<none>:<none>\n", stderr=""
            )

        with (
            patch("deploy_agent.executor._run", side_effect=fake_run),
            pytest.raises(RuntimeError, match="no usable repository:tag"),
        ):
            executor._resolve_local_image_reference(_DIGEST)

    def test_resolve_local_image_reference_fails_loud_on_inspect_failure(
        self,
    ) -> None:
        executor = DeployExecutor()

        def fake_run(
            cmd: list[str], timeout: int, **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr="no such image"
            )

        with (
            patch("deploy_agent.executor._run", side_effect=fake_run),
            pytest.raises(RuntimeError, match="could not resolve"),
        ):
            executor._resolve_local_image_reference(_DIGEST)

    def test_resolve_prod_image_env_maps_only_known_services(self) -> None:
        executor = DeployExecutor()
        with patch.object(
            DeployExecutor,
            "_resolve_local_image_reference",
            return_value="omnibase-infra-stability-test-omninode-runtime:latest",
        ):
            result = executor._resolve_prod_image_env(
                _DIGEST, ["omninode-runtime", "forward-migration"]
            )
        assert result == {
            "PROD_OMNINODE_RUNTIME_IMAGE": "omnibase-infra-stability-test-omninode-runtime:latest"
        }

    def test_resolve_prod_image_env_empty_when_no_target_service(self) -> None:
        executor = DeployExecutor()
        with patch.object(
            DeployExecutor,
            "_resolve_local_image_reference",
            side_effect=AssertionError("must not resolve when no target service"),
        ):
            result = executor._resolve_prod_image_env(_DIGEST, ["forward-migration"])
        assert result == {}

    def test_service_env_var_mapping_covers_both_prod_runtime_services(
        self,
    ) -> None:
        assert PROD_IMAGE_ENV_VAR_FOR_SERVICE == {
            "omninode-runtime": "PROD_OMNINODE_RUNTIME_IMAGE",
            "runtime-effects": "PROD_RUNTIME_EFFECTS_IMAGE",
        }

    def test_rebuild_scope_prod_runtime_threads_extra_env_into_compose_up(
        self,
    ) -> None:
        """Cross-boundary wiring: rebuild_scope's prod branch must actually
        pass the resolved PROD_*_IMAGE overrides into `_compose_up`'s
        `extra_env`, not just compute them and drop them."""
        executor = DeployExecutor()
        captured_extra_env: list[dict[str, str] | None] = []

        def fake_compose_up(
            self: DeployExecutor,
            phase: Phase,
            scope: Scope,
            services: list[str],
            on_phase_update: object,
            *,
            lane: EnumRuntimeLane = EnumRuntimeLane.DEV,
            extra_env: dict[str, str] | None = None,
        ) -> None:
            captured_extra_env.append(dict(extra_env) if extra_env else extra_env)

        with (
            patch.object(DeployExecutor, "_pull_pinned_image", return_value=None),
            patch.object(
                DeployExecutor,
                "_resolve_local_image_reference",
                return_value="omnibase-infra-stability-test-omninode-runtime:latest",
            ),
            patch.object(DeployExecutor, "_compose_up", new=fake_compose_up),
        ):
            executor.rebuild_scope(
                Scope.RUNTIME,
                ["omninode-runtime"],
                lambda *_args: None,
                lane=EnumRuntimeLane.PROD,
                image_digest=_DIGEST,
            )

        assert captured_extra_env == [
            {
                "PROD_OMNINODE_RUNTIME_IMAGE": "omnibase-infra-stability-test-omninode-runtime:latest"
            }
        ]

    def test_compose_env_extra_env_overrides_ambient_environ(self) -> None:
        from deploy_agent.executor import _compose_env

        with patch.dict(
            os.environ, {"PROD_OMNINODE_RUNTIME_IMAGE": "ambient-should-lose"}
        ):
            env = _compose_env({"PROD_OMNINODE_RUNTIME_IMAGE": "explicit-wins"})
        assert env["PROD_OMNINODE_RUNTIME_IMAGE"] == "explicit-wins"

    def test_compose_env_without_extra_env_is_unaffected(self) -> None:
        """Regression guard: the extra_env parameter must be additive-only —
        omitting it must not change `_compose_env()`'s existing behavior for
        every other call site."""
        from deploy_agent.executor import _compose_env

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PROD_OMNINODE_RUNTIME_IMAGE", None)
            env = _compose_env()
        assert "PROD_OMNINODE_RUNTIME_IMAGE" not in env
