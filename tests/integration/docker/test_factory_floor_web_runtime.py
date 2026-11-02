# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise the candidate server in an isolated Compose project (OMN-21733).

Run with pytest on a host with Docker Engine and Compose, including a hosted
Ubuntu runner. Missing Docker is an error. Only the container name, restart
policy and deployment values differ from the candidate deployment. No existing
server or published content is used or changed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

LOGGER = logging.getLogger(__name__)

COMPOSE = (
    Path(__file__).resolve().parents[3]
    / "docker"
    / "docker-compose.factory-floor-web.yml"
)
SERVICE = "factory-floor-web"
DEFAULT_POLICY = (
    "default-src 'self'; connect-src 'self'; form-action 'none'; base-uri 'none'; "
    "frame-ancestors 'none'; object-src 'none'; style-src 'self' 'unsafe-inline'; "
    "script-src 'self' 'unsafe-inline'; img-src 'self' data:"
)
SCOPED_POLICY = DEFAULT_POLICY.replace(
    "script-src 'self' 'unsafe-inline';",
    "script-src 'self' 'unsafe-inline' "
    "https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js "
    "https://cdn.jsdelivr.net/npm/elkjs@0.9.2/lib/elk.bundled.js;",
)
REQUIRED = {
    "FACTORY_FLOOR_BIND_ADDR": "set FACTORY_FLOOR_BIND_ADDR to the host tailnet address",
    "FACTORY_FLOOR_PORT": "set FACTORY_FLOOR_PORT",
    "FACTORY_FLOOR_SERVE_DIR": "set FACTORY_FLOOR_SERVE_DIR",
}


def run(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, env=env, text=True, capture_output=True, timeout=120, check=False
    )


def require(args: list[str], env: dict[str, str]) -> str:
    result = run(args, env)
    if result.returncode:
        raise RuntimeError(f"{args!r}: {result.stdout}\n{result.stderr}")
    return result.stdout


def test_factory_floor_web_runtime() -> None:
    project = (
        os.environ.get("COMPOSE_PROJECT_NAME", "factory-floor-probe")
        + "-"
        + uuid.uuid4().hex[:12]
    )
    base_env = {k: v for k, v in os.environ.items() if k not in REQUIRED}
    base_env["COMPOSE_PROJECT_NAME"] = project
    command = [
        "docker",
        "compose",
        "--project-name",
        project,
        "--env-file",
        os.devnull,
        "-f",
        str(COMPOSE),
    ]
    require(["docker", "compose", "version"], base_env)
    with (
        tempfile.TemporaryDirectory(prefix="factory-floor-probe-") as temporary,
        tempfile.TemporaryDirectory(
            prefix="factory-floor-override-"
        ) as temporary_override,
    ):
        site = Path(temporary)
        site.chmod(0o755)
        content = {
            "index.html": "factory floor probe",
            "factory-floor.json": '{"probe":true}',
            "reports/index.html": "reports probe",
            "reports/decision-map/index.html": "decision map probe",
            ".publish.tmp": "private root temporary file",
            "reports/decision-map/.publish.tmp": "private report temporary file",
        }
        for name, body in content.items():
            file_path = site / name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(body, encoding="utf-8")
            file_path.chmod(0o644)
        values = {
            "FACTORY_FLOOR_BIND_ADDR": "127.0.0.1",
            "FACTORY_FLOOR_PORT": "0",
            "FACTORY_FLOOR_SERVE_DIR": str(site),
        }
        for missing, message in REQUIRED.items():
            for empty in (False, True):
                env = {**base_env, **values}
                env.pop(missing)
                if empty:
                    env[missing] = ""
                result = run([*command, "config"], env)
                assert result.returncode != 0 and message in result.stderr, result
        LOGGER.info("PASS: Compose refuses all six unset/empty deployment values")
        env = {**base_env, **values}
        document = json.loads(require([*command, "config", "--format", "json"], env))
        service = document["services"][SERVICE]
        assert re.fullmatch(r"nginx:1\.29-alpine@sha256:[0-9a-f]{64}", service["image"])
        # Compose versions differ in the type of `published` and in whether they emit `mode`.
        assert [
            (port["host_ip"], port["target"], str(port["published"]))
            for port in service["ports"]
        ] == [("127.0.0.1", 8080, "0")], service["ports"]
        override = Path(temporary_override) / "probe-compose.json"
        override.write_text(
            json.dumps(
                {
                    "services": {
                        SERVICE: {
                            "container_name": project,
                            "restart": "no",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        deployed = [*command, "-f", str(override)]
        try:
            require([*deployed, "up", "-d", "--wait", "--wait-timeout", "45"], env)
            require([*deployed, "exec", "-T", SERVICE, "nginx", "-t"], env)
            LOGGER.info("PASS: nginx -t")
            container = require([*deployed, "ps", "-q", SERVICE], env).strip()
            inspected = json.loads(require(["docker", "inspect", container], env))[0]
            host = inspected["HostConfig"]
            assert host["ReadonlyRootfs"] and host["CapDrop"] == ["ALL"], host
            assert inspected["Config"]["User"] == "101:101"
            assert "no-new-privileges:true" in host["SecurityOpt"]
            mounts = {mount["Destination"]: mount for mount in inspected["Mounts"]}
            for destination in ("/srv/site", "/etc/nginx/nginx.conf"):
                assert not mounts[destination]["RW"], mounts[destination]
            bindings = {
                port: bound
                for port, bound in inspected["NetworkSettings"]["Ports"].items()
                if bound is not None
            }
            assert set(bindings) == {"8080/tcp"}, bindings
            assert len(bindings["8080/tcp"]) == 1, bindings
            binding = bindings["8080/tcp"][0]
            assert binding["HostIp"] == "127.0.0.1", bindings
            LOGGER.info("PASS: isolated container %s ports=%s", container, bindings)
            opener = build_opener(ProxyHandler({}))
            # (path, GET status, POST status, policy, expected body). The dot-file rule answers 404 to every
            # method; elsewhere limit_except refuses POST with 403.
            cases = [
                ("/", 200, 403, DEFAULT_POLICY, content["index.html"]),
                (
                    "/factory-floor.json",
                    200,
                    403,
                    DEFAULT_POLICY,
                    content["factory-floor.json"],
                ),
                ("/reports/", 200, 403, DEFAULT_POLICY, content["reports/index.html"]),
                (
                    "/reports/decision-map/",
                    200,
                    403,
                    SCOPED_POLICY,
                    content["reports/decision-map/index.html"],
                ),
                (
                    "/reports/decision-map/index.html",
                    200,
                    403,
                    SCOPED_POLICY,
                    content["reports/decision-map/index.html"],
                ),
                ("/.publish.tmp", 404, 404, DEFAULT_POLICY, None),
                ("/reports/decision-map/.publish.tmp", 404, 404, DEFAULT_POLICY, None),
                ("/missing", 404, 403, DEFAULT_POLICY, None),
                ("/reports/decision-map/missing", 404, 403, SCOPED_POLICY, None),
            ]
            for path, status, post_status, policy, expected_body in cases:
                for method in ("GET", "HEAD", "POST"):
                    request = Request(
                        f"http://127.0.0.1:{binding['HostPort']}{path}", method=method
                    )
                    try:
                        response = opener.open(request, timeout=5)
                    except HTTPError as error:
                        response = error
                    with response:
                        expected_status = post_status if method == "POST" else status
                        assert response.status == expected_status, (
                            method,
                            path,
                            response.status,
                        )
                        assert response.headers.get_all("Cache-Control") == ["no-store"]
                        assert response.headers.get_all("X-Content-Type-Options") == [
                            "nosniff"
                        ]
                        assert response.headers.get_all("Content-Security-Policy") == [
                            policy
                        ]
                        body = response.read().decode()
                        if method == "HEAD":
                            assert not body
                        elif method == "GET" and expected_body is not None:
                            assert body == expected_body, (path, body)
                    LOGGER.info("PASS: %s %s", method, path)
        finally:
            # Teardown must not mask the assertion that failed or leave the project behind.
            down = run([*deployed, "down", "--volumes", "--timeout", "5"], env)
            if down.returncode:
                LOGGER.error("teardown of %s failed: %s", project, down.stderr)
