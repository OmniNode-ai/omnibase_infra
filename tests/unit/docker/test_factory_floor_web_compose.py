# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The factory floor web service binds one address, serves read-only and needs its deployment values (OMN-21733)."""

from __future__ import annotations

import gzip
import http.client
import json
import os
import re
import subprocess
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

DOCKER_DIR = Path(__file__).resolve().parents[3] / "docker"
COMPOSE = DOCKER_DIR / "docker-compose.factory-floor-web.yml"
NGINX = DOCKER_DIR / "factory-floor-web" / "nginx.conf"
ALL_INTERFACES = ".".join(("0",) * 4)
REQUIRED = (
    "FACTORY_FLOOR_BIND_ADDR",
    "FACTORY_FLOOR_PORT",
    "FACTORY_FLOOR_SERVE_DIR",
)
RENDER_ENV = {
    "FACTORY_FLOOR_BIND_ADDR": "192.0.2.1",
    "FACTORY_FLOOR_PORT": "8088",
    "FACTORY_FLOOR_SERVE_DIR": "/srv/example-site",
}


def _service() -> dict[str, Any]:
    document = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    service = document["services"]["factory-floor-web"]
    assert isinstance(service, dict)
    return service


def _compose_config(
    env_values: dict[str, str], *extra: str
) -> subprocess.CompletedProcess[str]:
    # The ambient environment stays (the container tooling needs its own wiring to run at all); only the
    # three deployment values are replaced, so one set in the caller's shell can never satisfy a refusal test.
    env = {k: v for k, v in os.environ.items() if k not in REQUIRED}
    env.update(env_values)
    try:
        return subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE), "config", *extra],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.fail(
            "docker compose is required to render the factory floor deployment; "
            "its absence is a failure, never a skip (OMN-21733 AC4)"
        )


# One nginx statement: its words, and its child statements when it opens a block.
NginxStatement = tuple[list[str], "list[NginxStatement] | None"]
NGINX_TOKEN = re.compile(r'"[^"]*"|\'[^\']*\'|#[^\n]*|[{};]|[^\s{};"\'#]+')


def _parse_nginx(text: str) -> list[NginxStatement]:
    """Parse nginx.conf into nested statements: comments dropped, quoted strings kept whole."""
    tokens = [t for t in NGINX_TOKEN.findall(text) if not t.startswith("#")]
    position = 0

    def block() -> list[NginxStatement]:
        nonlocal position
        statements: list[NginxStatement] = []
        words: list[str] = []
        while position < len(tokens):
            token = tokens[position]
            position += 1
            if token == ";":
                statements.append((words, None))
                words = []
            elif token == "{":
                statements.append((words, block()))
                words = []
            elif token == "}":
                return statements
            else:
                words.append(token)
        assert not words, f"unterminated nginx statement: {words}"
        return statements

    return block()


def _children(statements: list[NginxStatement], name: str) -> list[NginxStatement]:
    return [s for s in statements if s[0][0] == name]


def _body(statement: NginxStatement) -> list[NginxStatement]:
    assert statement[1] is not None
    return statement[1]


def _server() -> list[NginxStatement]:
    top = _parse_nginx(NGINX.read_text(encoding="utf-8"))
    (http,) = _children(top, "http")
    (server,) = _children(_body(http), "server")
    return _body(server)


def _walk(statements: list[NginxStatement]) -> list[NginxStatement]:
    found: list[NginxStatement] = []
    for statement in statements:
        found.append(statement)
        if statement[1] is not None:
            found.extend(_walk(statement[1]))
    return found


def test_published_port_names_the_bind_address_and_has_no_default() -> None:
    ports = _service()["ports"]
    assert isinstance(ports, list)
    assert len(ports) == 1
    port = str(ports[0])
    assert port.startswith("${FACTORY_FLOOR_BIND_ADDR:?")
    assert port.endswith(":8080")
    assert ALL_INTERFACES not in COMPOSE.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", REQUIRED)
def test_every_deployment_value_is_required_without_a_default(name: str) -> None:
    text = COMPOSE.read_text(encoding="utf-8")
    assert f"${{{name}:?" in text
    assert f"${{{name}:-" not in text
    assert f"${{{name}-" not in text


def test_serve_directory_is_mounted_read_only_and_the_container_is_locked_down() -> (
    None
):
    service = _service()
    volumes = service["volumes"]
    serve = [v for v in volumes if "FACTORY_FLOOR_SERVE_DIR" in v]
    assert len(serve) == 1
    assert serve[0].endswith(":/srv/site:ro")
    assert all(
        v.endswith(":ro") if isinstance(v, str) else v["read_only"] is True
        for v in volumes
    )
    config = [v for v in volumes if isinstance(v, dict)]
    assert config == [
        {
            "type": "bind",
            "source": "./factory-floor-web/nginx.conf",
            "target": "/etc/nginx/nginx.conf",
            "read_only": True,
            "bind": {"create_host_path": False},
        }
    ]
    assert (COMPOSE.parent / config[0]["source"]).resolve() == NGINX
    assert NGINX.is_file()
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in service["security_opt"]
    assert service["user"] == "101:101"
    assert re.fullmatch(r"[\w./-]+:[\w.-]+@sha256:[0-9a-f]{64}", str(service["image"]))
    assert service["healthcheck"]["test"][1].endswith(
        "http://127.0.0.1:8080/healthz || exit 1"
    )


def test_nginx_serves_only_get_with_no_store_and_a_same_origin_policy() -> None:
    server = _server()
    assert [s[0] for s in _children(server, "listen")] == [["listen", "8080"]]
    assert [s[0] for s in _children(server, "root")] == [["root", "/srv/site"]]
    headers = {s[0][1]: s[0][2:] for s in _children(server, "add_header")}
    assert headers["Cache-Control"] == ['"no-store"', "always"]
    assert headers["X-Content-Type-Options"] == ['"nosniff"', "always"]
    policy = headers["Content-Security-Policy"]
    assert policy[1] == "always"
    directives = {
        parts[0]: parts[1:]
        for parts in (d.split() for d in policy[0].strip('"').split(";"))
    }
    assert directives["default-src"] == ["'self'"]
    assert directives["connect-src"] == ["'self'"]
    assert directives["form-action"] == ["'none'"]
    assert directives["base-uri"] == ["'none'"]
    assert directives["frame-ancestors"] == ["'none'"]
    assert directives["object-src"] == ["'none'"]
    top = _parse_nginx(NGINX.read_text(encoding="utf-8"))
    (http,) = _children(top, "http")
    assert ["gzip", "on"] in [s[0] for s in _body(http)]
    (location,) = [s for s in _children(server, "location") if s[0][1:] == ["/"]]
    (limit,) = _children(_body(location), "limit_except")
    assert limit[0][1:] == ["GET", "HEAD"]
    assert _body(limit) == [(["deny", "all"], None)]
    assert [s[0] for s in _children(_body(location), "try_files")] == [
        ["try_files", "$uri", "$uri/index.html", "=404"]
    ]


def test_nginx_locations_declare_no_add_header_except_the_decision_map() -> None:
    server = _server()
    locations = _children(server, "location")
    assert [loc[0][1:] for loc in locations] == [
        ["~", "/\\."],
        ["=", "/healthz"],
        ["/"],
        ["/reports/decision-map/"],
    ]
    for location in locations[:3]:
        names = [s[0][0] for s in _walk(_body(location))]
        assert "add_header" not in names, location[0]
    dotfiles, health, _, decision_map = locations
    assert [s[0] for s in _body(dotfiles)] == [["return", "404"]]
    health_names = [s[0][0] for s in _body(health)]
    assert ["return", "200", '"ok\\n"'] in [s[0] for s in _body(health)]
    assert "try_files" not in health_names
    # A location that declares add_header drops every server-level header, so the decision map repeats all three.
    repeated = [s[0][1] for s in _children(_body(decision_map), "add_header")]
    assert repeated == [
        "Cache-Control",
        "X-Content-Type-Options",
        "Content-Security-Policy",
    ]


def _location_block(conf: str, header: str) -> str:
    start = conf.index(header)
    depth = 0
    for index in range(conf.index("{", start), len(conf)):
        depth += {"{": 1, "}": -1}.get(conf[index], 0)
        if depth == 0:
            return conf[start : index + 1]
    raise AssertionError(f"unterminated block {header}")


def test_only_the_decision_map_directory_may_load_two_pinned_cdn_scripts() -> None:
    conf = NGINX.read_text(encoding="utf-8")
    cdn = "https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"
    elk = "https://cdn.jsdelivr.net/npm/elkjs@0.9.2/lib/elk.bundled.js"
    policies = re.findall(r'Content-Security-Policy "([^"]+)"', conf)
    assert len(policies) == 2
    default, scoped = policies
    assert "http" not in default
    assert f"script-src 'self' 'unsafe-inline' {cdn} {elk};" in scoped
    assert scoped.count("http") == 2
    block = _location_block(conf, "location /reports/decision-map/ {")
    assert scoped in block
    assert "limit_except GET HEAD" in block
    assert 'Cache-Control "no-store"' in block
    assert 'X-Content-Type-Options "nosniff"' in block
    # A regex location wins over a plain prefix only unless the prefix is `^~`, so a plain prefix keeps the
    # dot-file rule in force for the directory.
    assert "location ^~" not in conf
    assert re.search(r"location\s+~\s+/\\\.\s*\{\s*return 404;", conf)


def test_nginx_parser_sees_a_header_hidden_after_a_nested_block() -> None:
    conf = (
        "server { location / { limit_except GET { deny all; } "
        'add_header X-Late "1"; } }'
    )
    (server,) = _children(_parse_nginx(conf), "server")
    (location,) = _children(_body(server), "location")
    assert "add_header" in [s[0][0] for s in _walk(_body(location))]


@pytest.mark.parametrize("missing", REQUIRED)
def test_compose_refuses_to_render_when_a_required_value_is_missing(
    missing: str,
) -> None:
    values = {k: v for k, v in RENDER_ENV.items() if k != missing}
    done = _compose_config(values)
    assert done.returncode != 0
    assert (
        f"required variable {missing} is missing a value: set {missing}" in done.stderr
    )


def test_compose_renders_one_published_port_on_the_given_address() -> None:
    done = _compose_config(RENDER_ENV, "--format", "json")
    assert done.returncode == 0, done.stderr
    published = json.loads(done.stdout)["services"]["factory-floor-web"]["ports"]
    assert [(p["host_ip"], str(p["published"]), p["target"]) for p in published] == [
        ("192.0.2.1", "8088", 8080)
    ]
    volumes = json.loads(done.stdout)["services"]["factory-floor-web"]["volumes"]
    config = [v for v in volumes if v["target"] == "/etc/nginx/nginx.conf"]
    assert len(config) == 1
    assert Path(config[0]["source"]) == NGINX
    assert config[0]["read_only"] is True
    assert config[0]["bind"].get("create_host_path", False) is False
    serve = [v for v in volumes if v["target"] == "/srv/site"]
    assert len(serve) == 1
    assert serve[0]["read_only"] is True
    assert serve[0]["source"] == RENDER_ENV["FACTORY_FLOOR_SERVE_DIR"]


def _fetch(
    host: str,
    port: int,
    method: str,
    path: str,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection(host, port, timeout=5)
    try:
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        return (
            response.status,
            {k.lower(): v for k, v in response.getheaders()},
            response.read(),
        )
    finally:
        connection.close()


def _container_logs(name: str) -> str:
    return subprocess.run(
        ["docker", "logs", name], capture_output=True, text=True, check=False
    ).stderr


@pytest.fixture(scope="module")
def nginx_address(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[str, int]]:
    """The pinned image running the committed nginx.conf with the compose file's lock-down."""
    site = tmp_path_factory.mktemp("site")
    site.chmod(0o755)
    files = {
        "index.html": "<!doctype html><title>floor</title>" + "<p>x</p>" * 400,
        "factory-floor.json": json.dumps({"generated_at": "t", "pad": "y" * 3000}),
        ".hidden": "secret",
    }
    for file_name, text in files.items():
        (site / file_name).write_text(text, encoding="utf-8")
        (site / file_name).chmod(0o644)
    service = _service()
    name = f"factory-floor-web-test-{uuid.uuid4().hex[:12]}"
    started = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--read-only",
            "--user",
            str(service["user"]),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--tmpfs",
            str(service["tmpfs"][0]),
            "-p",
            "127.0.0.1::8080",
            "-v",
            f"{site}:/srv/site:ro",
            "-v",
            f"{NGINX}:/etc/nginx/nginx.conf:ro",
            str(service["image"]),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert started.returncode == 0, started.stderr
    try:
        mapped = subprocess.run(
            ["docker", "port", name, "8080/tcp"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()[0]
        host, _, port = mapped.rpartition(":")
        deadline = time.monotonic() + 30
        while True:
            try:
                if _fetch(host, int(port), "GET", "/healthz")[0] == 200:
                    break
            except OSError:
                pass
            assert time.monotonic() < deadline, _container_logs(name)
            time.sleep(0.3)
        yield host, int(port)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


@pytest.mark.parametrize(
    ("path", "status"),
    [
        ("/", 200),
        ("/factory-floor.json", 200),
        ("/healthz", 200),
        ("/missing", 404),
        ("/.hidden", 404),
        ("/nested/.hidden", 404),
    ],
)
def test_running_nginx_sends_the_security_headers_on_every_response(
    nginx_address: tuple[str, int], path: str, status: int
) -> None:
    code, headers, _body = _fetch(*nginx_address, "GET", path)
    assert code == status
    assert headers["cache-control"] == "no-store"
    assert headers["x-content-type-options"] == "nosniff"
    assert "default-src 'self'" in headers["content-security-policy"]
    assert "frame-ancestors 'none'" in headers["content-security-policy"]


def test_running_nginx_serves_the_record_and_answers_health_without_a_page(
    nginx_address: tuple[str, int],
) -> None:
    code, _headers, body = _fetch(*nginx_address, "GET", "/factory-floor.json")
    assert code == 200
    assert json.loads(body)["generated_at"] == "t"
    code, _headers, body = _fetch(*nginx_address, "GET", "/healthz")
    assert (code, body) == (200, b"ok\n")


def test_running_nginx_compresses_the_record(nginx_address: tuple[str, int]) -> None:
    _code, headers, body = _fetch(
        *nginx_address, "GET", "/factory-floor.json", {"Accept-Encoding": "gzip"}
    )
    assert headers["content-encoding"] == "gzip"
    assert json.loads(gzip.decompress(body))["generated_at"] == "t"


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE"])
def test_running_nginx_refuses_every_method_but_get_and_head(
    nginx_address: tuple[str, int], method: str
) -> None:
    code, headers, _body = _fetch(*nginx_address, method, "/")
    assert code == 403
    assert headers["cache-control"] == "no-store"
    assert _fetch(*nginx_address, "HEAD", "/")[0] == 200
