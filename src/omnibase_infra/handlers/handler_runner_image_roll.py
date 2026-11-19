# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Installer effects for rolling the image of one existing secondary runner pool."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
import stat
from pathlib import Path
from typing import cast

import yaml

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_runner_image_roll_request import (
    ModelRunnerImageRollRequest,
)
from omnibase_infra.models.model_runner_image_roll_result import (
    ModelRunnerImageRollResult,
)
from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    load_runner_fleet_config,
)

_LOG = logging.getLogger(__name__)
_COMMAND_LIMIT_SECONDS = 30
_STAGE_LIMIT_SECONDS = 180
_BUILD_LIMIT_SECONDS = 2400
_POLL_INTERVAL_SECONDS = 15
_BUILD_INPUTS = (
    ".github/actions/setup-python-uv/action.yml",
    "pyproject.toml",
    "uv.lock",
    "docker/runners/Dockerfile",
    "docker/runners/runner-image.lock.json",
    "docker/runners/entrypoint.sh",
    "docker/runners/cgroup-cpu-env.sh",
    "docker/runners/runner-job-started.sh",
    "docker/runners/healthcheck.sh",
    "docker/runners/omni-curl",
    "scripts/ci/build_runner_image.sh",
    "scripts/ci/ci_env_digest.py",
    "scripts/ci/ensure_ci_env.sh",
    "scripts/ci/runner_image_identity.py",
)
_DOCKER_RELATIVE = ".omnibase/runners/docker"
_DOCKER_DIR = f'"$HOME/{_DOCKER_RELATIVE}"'
_DOCKER_ENV = (
    'export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"; '
    'if test -f "$HOME/.omnibase/runners/docker-cli/config.json"; then '
    'export DOCKER_CONFIG="$HOME/.omnibase/runners/docker-cli"; fi; '
)
_CONTAINER_FORMAT = (
    "{{.Id}} {{.Image}} {{.State.Status}} "
    '{{index .Config.Labels "com.docker.compose.project"}} '
    '{{index .Config.Labels "com.docker.compose.service"}}'
)
_RETRY_PASSES = 3
_ONLINE_POLLS = 20
_BUSY = 42


class RunnerRollError(Exception):
    """An installer refusal safe to show without subprocess output."""


class RunnerUncertainRecreateError(RunnerRollError):
    """The remote recreate may still be running."""


class HandlerRunnerImageRoll:
    """Plan the pool locally, then recreate only verified idle pool members."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelRunnerImageRollRequest
    ) -> ModelRunnerImageRollResult:
        try:
            token = self._read_token(request.token_file)
            org, targets, token_vars = self._pool_plan(request)
            version, identity = self._image_lock(request.repo_root)
            image_tag = f"omninode-runner-{request.prefix}:v{version}-{identity}"
            if request.dry_run:
                plan = targets[: request.limit or None]
                _LOG.info(
                    "[DRY RUN] %s on %s: would roll %s",
                    image_tag,
                    request.host,
                    ", ".join(plan),
                )
                return ModelRunnerImageRollResult(
                    success=True, image=image_tag, planned=plan
                )
            return await self._run_with_lock(
                request, org, targets, token_vars, image_tag, token
            )
        except RunnerRollError as exc:
            return ModelRunnerImageRollResult(success=False, error_message=str(exc))
        except TimeoutError:
            return ModelRunnerImageRollResult(
                success=False,
                error_message=f"Installer command timed out on {request.host}.",
            )
        except (OSError, ValueError, yaml.YAMLError):
            return ModelRunnerImageRollResult(
                success=False,
                error_message="Could not read the token, fleet, Compose or image lock.",
            )

    @staticmethod
    def _read_token(path: Path) -> bytes:
        if (
            path.is_symlink()
            or not path.is_file()
            or stat.S_IMODE(path.stat().st_mode) != 0o600
        ):
            raise RunnerRollError(
                "Secondary --rolling requires a regular 0600 --token-file."
            )
        token = path.read_bytes().strip()
        if len(token.split()) != 1 or b"\0" in token or token == b"null":
            raise RunnerRollError(
                "--token-file must contain one nonempty registration token."
            )
        return token

    @staticmethod
    def _pool_plan(
        request: ModelRunnerImageRollRequest,
    ) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        fleet = load_runner_fleet_config(request.fleet_config)
        host = next((host for host in fleet.hosts if host.host == request.host), None)
        if host is None or request.host == fleet.runner_host:
            raise RunnerRollError("Image roll requires a declared secondary host.")
        pools = {
            host.runner_name_prefix: (host.expected_count, host.classes),
            **{
                pool.runner_name_prefix: (pool.expected_count, pool.classes)
                for pool in host.pools
            },
        }
        count, classes = pools.get(request.prefix, (0, ()))
        if not count or "deploy" in classes:
            raise RunnerRollError("Image roll requires a declared secondary CI pool.")
        all_targets = tuple(f"{request.prefix}-{i}" for i in range(1, count + 1))
        if request.only and request.only not in all_targets:
            raise RunnerRollError(
                "--only must name a service in the selected host's declared pool."
            )
        compose = (
            request.repo_root
            / "docker"
            / f"docker-compose.runners-{request.prefix}.yml"
        )
        source = compose.read_text()
        declaration: object = yaml.safe_load(source)
        services: object = (
            declaration.get("services") if isinstance(declaration, dict) else None
        )
        if not isinstance(services, dict) or set(services) != set(all_targets):
            raise RunnerRollError("Compose services disagree with the declared pool.")
        token_vars = tuple(
            sorted(set(re.findall(r"RUNNER_TOKEN:\s*\$\{([A-Z0-9_]+)", source)))
        )
        if not token_vars:
            raise RunnerRollError(
                "The declared Compose file has no runner token variable."
            )
        return (
            fleet.github_org,
            (request.only,) if request.only else all_targets,
            token_vars,
        )

    @staticmethod
    def _image_lock(repo_root: Path) -> tuple[str, str]:
        lock: object = json.loads(
            (repo_root / "docker/runners/runner-image.lock.json").read_bytes()
        )
        if not isinstance(lock, dict):
            raise RunnerRollError("Invalid runner image lock.")
        version, identity = (
            str(lock.get("image_version")),
            str(lock.get("identity_digest")),
        )
        if not re.fullmatch(r"[1-9][0-9]*", version) or not re.fullmatch(
            r"[a-f0-9]{32}", identity
        ):
            raise RunnerRollError("Invalid runner image lock version or identity.")
        return version, identity

    @staticmethod
    async def _command(
        *argv: str, stdin: bytes | None = None, timeout: int = _COMMAND_LIMIT_SECONDS
    ) -> tuple[int, str]:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE
            if stdin is not None
            else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(stdin), timeout=timeout
            )
        except (TimeoutError, asyncio.CancelledError):
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        return cast("int", process.returncode), stdout.decode(errors="replace").strip()

    async def _remote(
        self,
        host: str,
        script: str,
        *,
        stdin: bytes | None = None,
        timeout: int = _COMMAND_LIMIT_SECONDS,
    ) -> tuple[int, str]:
        return await self._command(
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "--",
            host,
            _DOCKER_ENV + "exec bash -c " + shlex.quote(script),
            stdin=stdin,
            timeout=timeout,
        )

    async def _registry(self, org: str, name: str) -> tuple[str, bool | None]:
        rc, response = await self._command(
            "gh",
            "api",
            "--paginate",
            "--slurp",
            "-H",
            "Cache-Control: no-cache",
            f"/orgs/{org}/actions/runners?per_page=100",
        )
        if rc:
            raise RunnerRollError(
                f"Could not read the organization runner registry (gh exit {rc})."
            )
        try:
            rows = [
                row
                for page in json.loads(response)
                for row in page["runners"]
                if row["name"] == name
            ]
            if len(rows) == 1:
                busy = rows[0].get("busy")
                return str(rows[0].get("status")), busy if isinstance(
                    busy, bool
                ) else None
        except (KeyError, TypeError, ValueError) as exc:
            raise RunnerRollError(
                "Could not decode the organization runner registry."
            ) from exc
        return "unknown", None

    @staticmethod
    def _compose(prefix: str) -> str:
        return f"docker compose -p {prefix} -f docker-compose.runners-{prefix}.yml"

    async def _run_with_lock(
        self,
        request: ModelRunnerImageRollRequest,
        org: str,
        targets: tuple[str, ...],
        token_vars: tuple[str, ...],
        image_tag: str,
        token: bytes,
    ) -> ModelRunnerImageRollResult:
        # Prove registry access before creating lock directories or building an image.
        await self._registry(org, targets[0])
        lock = f"{_DOCKER_DIR}/.image-roll-{request.prefix}.lock"
        rc, _ = await self._remote(request.host, f"mkdir -- {lock}")
        if rc:
            raise RunnerRollError(
                f"Could not acquire the pool roll lock on {request.host} (exit {rc})."
            )
        release = True
        try:
            await self._verify_compose(request, targets, token_vars)
            return await self._roll(
                request,
                targets=targets,
                image_tag=image_tag,
                token=token,
                org=org,
                token_vars=token_vars,
            )
        except (RunnerUncertainRecreateError, asyncio.CancelledError):
            release = False
            _LOG.error(
                "Pool roll lock kept on %s pending recreate inspection: %s",
                request.host,
                lock,
            )
            raise
        finally:
            if release:
                rc, _ = await self._remote(request.host, f"rmdir -- {lock}")
                if rc:
                    raise RunnerRollError(
                        f"Could not release the pool roll lock on {request.host} (exit {rc})."
                    )

    async def _verify_compose(
        self,
        request: ModelRunnerImageRollRequest,
        targets: tuple[str, ...],
        token_vars: tuple[str, ...],
    ) -> None:
        exports = " ".join(f"{var}=onex-roll-preflight" for var in token_vars)
        # Keep the rendered environment on the host. Return only checked runner names.
        guard = (
            ".services | to_entries[] | select(.key == .value.container_name and "
            ".key == .value.environment.RUNNER_NAME and "
            '.value.environment.RUNNER_TOKEN == "onex-roll-preflight") | .key'
        )
        rc, output = await self._remote(
            request.host,
            f"set -eo pipefail; cd {_DOCKER_DIR}; "
            f"{exports} {self._compose(request.prefix)} config --format json | jq -r {shlex.quote(guard)}",
        )
        if rc or not set(targets).issubset(output.splitlines()):
            raise RunnerRollError(
                "Installed Compose could not confirm target names and token wiring."
            )

    async def _prepare_image(
        self, request: ModelRunnerImageRollRequest, image_tag: str
    ) -> str:
        installed = await self._verified_image_id(request, image_tag)
        if installed is not None:
            return installed
        if not request.skip_build:
            stage_root = f"{_DOCKER_RELATIVE}/.image-roll-{request.prefix}.lock"
            rc, name = await self._remote(
                request.host, f'cd "$HOME/{stage_root}" && mktemp -d build.XXXXXXXX'
            )
            if rc or not re.fullmatch(r"build\.[A-Za-z0-9]{8}", name):
                raise RunnerRollError(
                    "Could not allocate the runner image build directory."
                )
            stage = f"{stage_root}/{name}"
            try:
                rc, _ = await self._command(
                    "rsync",
                    "-aR",
                    "-e",
                    "ssh -o BatchMode=yes -o ConnectTimeout=10",
                    *(f"{request.repo_root}/./{name}" for name in _BUILD_INPUTS),
                    f"{request.host}:{stage}/",
                    timeout=_STAGE_LIMIT_SECONDS,
                )
                if rc:
                    raise RunnerRollError(
                        f"Could not stage the runner build inputs (rsync exit {rc})."
                    )
                rc, _ = await self._remote(
                    request.host,
                    f'cd "$HOME/{stage}" && bash scripts/ci/build_runner_image.sh --tag {image_tag}',
                    timeout=_BUILD_LIMIT_SECONDS,
                )
                if rc:
                    raise RunnerRollError(
                        f"Declared runner image build failed (exit {rc})."
                    )
            finally:
                rc, _ = await self._remote(request.host, f'rm -rf -- "$HOME/{stage}"')
                if rc:
                    _LOG.warning(
                        "Could not remove image build directory %s on %s",
                        stage,
                        request.host,
                    )
        image = await self._verified_image_id(request, image_tag)
        if image is None:
            raise RunnerRollError(
                "Host image could not be verified against the declared image lock."
            )
        return image

    async def _verified_image_id(
        self, request: ModelRunnerImageRollRequest, image_tag: str
    ) -> str | None:
        fmt = '{{.Id}} {{index .Config.Labels "org.omninode.runner.image.version"}} {{index .Config.Labels "org.omninode.runner.image.identity"}}'
        rc, output = await self._remote(
            request.host,
            f"docker image inspect --format {shlex.quote(fmt)} {image_tag}",
        )
        version, identity = self._image_lock(request.repo_root)
        fields = output.split()
        if rc or len(fields) != 3 or fields[1:] != [version, identity]:
            return None
        return fields[0]

    async def _container(
        self, request: ModelRunnerImageRollRequest, name: str
    ) -> tuple[str, str] | None:
        rc, output = await self._remote(
            request.host,
            f"docker inspect --type container --format {shlex.quote(_CONTAINER_FORMAT)} {name}",
        )
        fields = output.split()
        if rc or len(fields) != 5 or fields[2:] != ["running", request.prefix, name]:
            return None
        return fields[0], fields[1]

    async def _idle_container(
        self, request: ModelRunnerImageRollRequest, name: str, org: str
    ) -> tuple[str, str] | None:
        container = await self._container(request, name)
        if container is None or await self._registry(org, name) != ("online", False):
            return None
        return container

    async def _recreate(
        self,
        request: ModelRunnerImageRollRequest,
        name: str,
        previous_id: str,
        token: bytes,
        image_tag: str,
        token_vars: tuple[str, ...],
    ) -> bool:
        # Do the process check and stop back-to-back on the host. Use the inspected
        # ID so a concurrent replacement cannot cause us to stop a different runner.
        exports = " ".join(f'export {var}="$handle";' for var in token_vars)
        override = json.dumps({"services": {name: {"image": image_tag}}})
        script = f"""
set +x
set -euo pipefail
cd {_DOCKER_DIR}
handle=$(cat)
test -n "$handle"
dead=$(docker ps -a --filter 'name=^/{request.prefix}-[0-9]+$' --filter status=dead -q) || exit {_BUSY}
test -z "$dead" || exit {_BUSY}
current=$(docker inspect --type container --format '{{{{.Id}}}}' {name}) || exit {_BUSY}
test "$current" = {shlex.quote(previous_id)} || exit {_BUSY}
processes=$(docker top "$current") || exit {_BUSY}
[[ "$processes" == *Runner.Listener* && "$processes" != *Runner.Worker* ]] || exit {_BUSY}
docker stop -t 30 "$current"
{exports}
printf '%s\n' {shlex.quote(override)} |
    {self._compose(request.prefix)} -f - up -d --force-recreate --no-deps --no-build --pull never {name}
"""
        try:
            rc, _ = await self._remote(
                request.host, script, stdin=token, timeout=_STAGE_LIMIT_SECONDS
            )
        except (OSError, TimeoutError) as exc:
            raise RunnerUncertainRecreateError(
                f"HALT: lost recreate readback for {name}."
            ) from exc
        if rc == _BUSY:
            return False
        if rc == 255:
            raise RunnerUncertainRecreateError(
                f"HALT: SSH lost recreate readback for {name}."
            )
        if rc:
            raise RunnerRollError(f"HALT: recreate failed for {name} (exit {rc}).")
        return True

    async def _wait_online(
        self,
        request: ModelRunnerImageRollRequest,
        name: str,
        previous_id: str,
        image: str,
        org: str,
    ) -> bool:
        for _ in range(_ONLINE_POLLS):
            container = await self._container(request, name)
            if (
                container is not None
                and container[0] != previous_id
                and container[1] == image
            ):
                rc, _ = await self._remote(
                    request.host,
                    f"set -o pipefail; docker logs --tail 80 {shlex.quote(container[0])} 2>&1 | grep -F 'Listening for Jobs'",
                )
                if rc == 0 and (await self._registry(org, name))[0] == "online":
                    return True
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        return False

    @staticmethod
    async def _retry_pause() -> None:
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    async def _roll(
        self,
        request: ModelRunnerImageRollRequest,
        *,
        targets: tuple[str, ...],
        image_tag: str,
        token: bytes,
        org: str,
        token_vars: tuple[str, ...],
    ) -> ModelRunnerImageRollResult:
        image = await self._prepare_image(request, image_tag)
        recreated: list[str] = []
        pending = list(targets)
        for attempt in range(_RETRY_PASSES + 1):
            if attempt:
                await self._retry_pause()
            retry: list[str] = []
            for name in pending:
                try:
                    container = await self._idle_container(request, name, org)
                    if container is None:
                        retry.append(name)
                        continue
                    previous_id, current_image = container
                    if current_image == image:
                        _LOG.info("%s: online on the declared image", name)
                        continue
                    if not await self._recreate(
                        request, name, previous_id, token, image_tag, token_vars
                    ):
                        retry.append(name)
                        continue
                    if not await self._wait_online(
                        request, name, previous_id, image, org
                    ):
                        raise RunnerRollError(
                            f"HALT: {name} did not return online on {image_tag}."
                        )
                except RunnerUncertainRecreateError:
                    raise
                except (RunnerRollError, OSError, TimeoutError) as exc:
                    reason = (
                        str(exc)
                        if isinstance(exc, RunnerRollError)
                        else type(exc).__name__
                    )
                    return ModelRunnerImageRollResult(
                        success=False,
                        image=image_tag,
                        recreated=tuple(recreated),
                        error_message=f"{name}: {reason}",
                    )
                recreated.append(name)
                _LOG.info("%s: recreated and read back online on %s", name, image_tag)
                if request.limit and len(recreated) == request.limit:
                    return ModelRunnerImageRollResult(
                        success=True, image=image_tag, recreated=tuple(recreated)
                    )
            pending = retry
            if not pending:
                return ModelRunnerImageRollResult(
                    success=True, image=image_tag, recreated=tuple(recreated)
                )
            _LOG.info("Busy or unreadable, left untouched: %s", ", ".join(pending))
        return ModelRunnerImageRollResult(
            success=False,
            image=image_tag,
            recreated=tuple(recreated),
            error_message=f"Still busy or unreadable after retries: {', '.join(pending)}",
        )
