# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Offline tests for the provider-rung liveness canary (OMN-16987).

``tests/fixtures/omn16987/lane_2026_09_24.json`` is the observer's output as
recorded on the lab dev lane's ``omninode-runtime-effects`` container on
2026-09-24 (omnimarket 0.4.215, delegation contract 2.7.0): four credentialed
backends LIVE over three distinct rungs, the OpenRouter rung's key unresolved,
and three wrong-key controls refused. Every other case is that recording with
ONE observation changed, or the observer driven through fakes, so each test
names the property its verdict turns on. Nothing here touches docker, the
network, a key or a lane.
"""

from __future__ import annotations

import asyncio
import copy
import json
import stat
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci import provider_rung_canary_observe as observer
from scripts.ci import provider_rung_canary_probe as probe

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "omn16987"
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "provider-rung-canary.yml"
OPENROUTER = "openrouter-qwen3-coder-480b"
SENTINEL = "sk-SENTINEL-omn16987-must-never-appear"  # nosec B105


def _recorded() -> dict[str, Any]:
    payload = json.loads(
        (FIXTURES / "lane_2026_09_24.json").read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    return payload


def _without_openrouter() -> dict[str, Any]:
    obs = _recorded()
    obs["backends"] = [b for b in obs["backends"] if b["backend_id"] != OPENROUTER]
    obs["probes"] = [p for p in obs["probes"] if OPENROUTER not in p["backend_ids"]]
    obs["controls"] = [
        c for c in obs["controls"] if c["endpoint_host"] != "openrouter.ai"
    ]
    return obs


def _failed(record: probe.Record) -> set[str]:
    return {c.name for c in record.checks if not c.ok}


def _verdicts(record: probe.Record) -> dict[str, str]:
    return {r["backend_id"]: r["verdict"] for r in record.rungs}


# ---------------------------------------------------------------- the lab reading


@pytest.mark.unit
def test_the_recorded_lab_reading_grades_each_rung_as_observed() -> None:
    record = probe.grade(_recorded())
    assert _verdicts(record) == {
        "local-coder": probe.SKIPPED_NO_SECRET_REF,
        "local-heavy-reasoning": probe.SKIPPED_NO_SECRET_REF,
        "local-embedding": probe.SKIPPED_NO_ENDPOINT,
        "cloud-gemini-pro": probe.LIVE,
        "cloud-glm": probe.LIVE,
        "cloud-glm-judge": probe.LIVE,
        "cloud-gemini-flash": probe.LIVE,
        "cloud-vertex-gemini": probe.SKIPPED_NO_ENDPOINT,
        "local-ds-v4-flash": probe.SKIPPED_NO_ENDPOINT,
        OPENROUTER: probe.UNRESOLVED,
        "local-omnipc2-chat": probe.SKIPPED_NO_SECRET_REF,
    }
    assert [c["verdict"] for c in record.controls] == [probe.AUTH_DEAD] * 3


@pytest.mark.unit
def test_an_unresolved_key_is_never_a_pass() -> None:
    """The recorded lane has one declared key that does not resolve, so the run
    is red on exactly that rung and nothing else."""
    record = probe.grade(_recorded())
    assert record.verdict == "fail"
    assert record.exit_code == probe.EXIT_FINDINGS
    assert _failed(record) == {f"rung/{OPENROUTER}"}


@pytest.mark.unit
def test_every_rung_live_and_every_control_refused_is_a_pass() -> None:
    record = probe.grade(_without_openrouter())
    assert record.verdict == "pass", record.failures
    assert record.exit_code == probe.EXIT_OK


@pytest.mark.unit
def test_an_empty_observation_cannot_pass() -> None:
    assert probe.grade({}).verdict == "fail"
    empty: dict[str, Any] = {"backends": [], "probes": [], "controls": []}
    record = probe.grade(empty)
    assert record.verdict == "fail"
    assert "every_credentialed_backend_probed" in _failed(record)


# ---------------------------------------------------------------- AC1 coverage


@pytest.mark.unit
def test_a_credentialed_backend_silently_skipped_fails_coverage() -> None:
    obs = _without_openrouter()
    obs["probes"] = [p for p in obs["probes"] if "cloud-glm" not in p["backend_ids"]]
    record = probe.grade(obs)
    assert "every_credentialed_backend_probed" in _failed(record)
    assert _verdicts(record)["cloud-glm"] == probe.PROBE_ERROR


@pytest.mark.unit
def test_coverage_is_rederived_not_read_from_the_observer() -> None:
    """An observer that marks a credentialed backend SKIPPED does not shrink
    what the grader expects."""
    obs = _without_openrouter()
    for row in obs["backends"]:
        if row["backend_id"] == "cloud-glm":
            row["disposition"] = "SKIPPED_NO_SECRET_REF"
    obs["probes"] = [p for p in obs["probes"] if "cloud-glm" not in p["backend_ids"]]
    assert "every_credentialed_backend_probed" in _failed(probe.grade(obs))


@pytest.mark.unit
def test_the_observer_plans_one_probe_per_distinct_rung() -> None:
    rows, probes = observer.plan(
        [
            {
                "backend_id": "a",
                "endpoint_url": "https://x/c",
                "model_name": "m",
                "secret_ref": "llm.x.api_key",
            },
            {
                "backend_id": "b",
                "endpoint_url": "https://x/c",
                "model_name": "m",
                "secret_ref": "llm.x.api_key",
            },
            {
                "backend_id": "c",
                "endpoint_url": "https://x/c",
                "model_name": "n",
                "secret_ref": "llm.x.api_key",
            },
            {"backend_id": "d", "endpoint_url": None, "secret_ref": "llm.v.token"},
            {"backend_id": "e", "endpoint_url": "http://lab/c", "secret_ref": None},
            {
                "backend_id": "f",
                "endpoint_url": "https://y/c",
                "model_name": "m",
                "api_key_env": "Y_API_KEY",
            },
        ]
    )
    assert [r["disposition"] for r in rows] == [
        observer.PROBE,
        observer.PROBE,
        observer.PROBE,
        observer.SKIPPED_NO_ENDPOINT,
        observer.SKIPPED_NO_SECRET_REF,
        observer.PROBE,
    ]
    assert [p["backend_ids"] for p in probes] == [["a", "b"], ["c"], ["f"]]
    assert probes[2]["secret_ref"] == "Y_API_KEY"


# ---------------------------------------------------------------- AC2 verdicts


@pytest.mark.unit
@pytest.mark.parametrize(
    ("fact", "verdict"),
    [
        (
            {"http_status": 200, "body_is_chat_completion": True, "model_echo": "m"},
            probe.LIVE,
        ),
        ({"http_status": 401}, probe.AUTH_DEAD),
        ({"http_status": 403}, probe.AUTH_DEAD),
        (
            {
                "http_status": 400,
                "provider_error_status": "INVALID_ARGUMENT",
                "provider_error_message": "Please pass a valid API key",
            },
            probe.AUTH_DEAD,
        ),
        (
            {
                "http_status": 400,
                "provider_error_status": "INVALID_ARGUMENT",
                "provider_error_message": "model not found",
            },
            probe.HTTP_ERROR,
        ),
        ({"http_status": 429, "provider_error_code": "1310"}, probe.QUOTA_DEAD),
        (
            {"exception": "ConnectError", "exception_family": "transport"},
            probe.UNREACHABLE,
        ),
        (
            {"exception": "ReadTimeout", "exception_family": "transport"},
            probe.UNREACHABLE,
        ),
        (
            {"http_status": 200, "body_is_chat_completion": False, "model_echo": "m"},
            probe.PROTOCOL_ERROR,
        ),
        (
            {"http_status": 200, "body_is_chat_completion": True, "model_echo": None},
            probe.PROTOCOL_ERROR,
        ),
        (
            {"exception": "JSONDecodeError", "exception_family": "decode"},
            probe.PROTOCOL_ERROR,
        ),
        ({"http_status": 503}, probe.HTTP_ERROR),
        ({"exception": "RuntimeError", "exception_family": "other"}, probe.PROBE_ERROR),
        ({"secret_resolved": False, "request_sent": False}, probe.UNRESOLVED),
        ({}, probe.PROBE_ERROR),
    ],
)
def test_classify(fact: dict[str, Any], verdict: str) -> None:
    assert probe.classify({"secret_resolved": True, **fact}) == verdict


class _FakeTransport:
    """Stands in for the deployed resolver and transport. Records every call."""

    def __init__(self, keys: dict[str, str | None], answer: Any) -> None:
        self.keys = keys
        self.answer = answer
        self.posts: list[tuple[str, dict[str, str]]] = []

    async def resolve(self, ref: str, env_fallback: str | None) -> str | None:
        return self.keys.get(ref)

    def post(
        self, url: str, payload: dict[str, Any], headers: dict[str, str], t: float
    ) -> dict[str, Any]:
        self.posts.append((url, dict(headers)))
        return dict(self.answer(url, headers) if callable(self.answer) else self.answer)


def _run(
    backends: list[dict[str, Any]], fake: _FakeTransport
) -> tuple[dict[str, Any], str]:
    _rows, probes = observer.plan(backends)
    secrets: set[str] = {observer.INVALID_KEY}
    results, controls = asyncio.run(
        observer.observe_probes(
            probes, resolve=fake.resolve, post=fake.post, secrets=secrets
        )
    )
    obs = {
        "backends": _rows,
        "probes": results,
        "controls": controls,
        "runtime_binds_contract": True,
        "quota_policy": [],
    }
    return obs, observer.render(obs, secrets)


_ONE = [
    {
        "backend_id": "cloud-x",
        "endpoint_url": "https://x.example/v1/chat",
        "model_name": "m",
        "secret_ref": "llm.x.api_key",
        "timeout_ms": 5000,
    },
]


@pytest.mark.unit
def test_observer_and_grader_turn_each_answer_into_its_verdict() -> None:
    """Driven end to end through the observer with a fake transport."""
    cases = [
        (observer.response_fields(200, {"choices": [{}], "model": "m"}, 3), probe.LIVE),
        (observer.response_fields(401, {"error": {"code": 401}}, 3), probe.AUTH_DEAD),
        (
            observer.response_fields(429, {"error": {"code": "1310"}}, 3),
            probe.QUOTA_DEAD,
        ),
        (
            {"exception": "ConnectError", "exception_family": "transport"},
            probe.UNREACHABLE,
        ),
        (observer.response_fields(200, {"object": "error"}, 3), probe.PROTOCOL_ERROR),
    ]
    for answer, verdict in cases:
        fake = _FakeTransport({"llm.x.api_key": "k"}, answer)
        obs, _line = _run(_ONE, fake)
        record = probe.grade(obs)
        assert _verdicts(record)["cloud-x"] == verdict, answer


# ---------------------------------------------------------------- AC3 unresolved


@pytest.mark.unit
def test_an_unresolved_key_sends_no_authenticated_request() -> None:
    fake = _FakeTransport({"llm.x.api_key": None}, {"http_status": 200})
    obs, _line = _run(_ONE, fake)
    # The only request that left is the wrong-key control; none carried a real key.
    assert [h["Authorization"] for _u, h in fake.posts] == [
        f"Bearer {observer.INVALID_KEY}"
    ]
    assert obs["probes"][0]["secret_resolved"] is False
    assert obs["probes"][0]["request_sent"] is False
    record = probe.grade(obs)
    assert _verdicts(record)["cloud-x"] == probe.UNRESOLVED
    assert record.verdict == "fail"
    names = {c.name: c.ok for c in record.checks}
    assert names["no_request_without_key/cloud-x"] is True


@pytest.mark.unit
def test_a_resolver_that_raises_reads_unresolved_by_class_name() -> None:
    class _Raising(_FakeTransport):
        async def resolve(self, ref: str, env_fallback: str | None) -> str | None:
            raise LookupError(f"store refused {SENTINEL}")

    fake = _Raising({}, {"http_status": 200})
    obs, line = _run(_ONE, fake)
    assert obs["probes"][0]["resolver_error"] == "LookupError"
    assert SENTINEL not in line
    assert _verdicts(probe.grade(obs))["cloud-x"] == probe.UNRESOLVED


@pytest.mark.unit
def test_an_unresolved_rung_that_sent_a_request_fails() -> None:
    obs = _recorded()
    for p in obs["probes"]:
        if OPENROUTER in p["backend_ids"]:
            p["request_sent"] = True
    assert f"no_request_without_key/{OPENROUTER}" in _failed(probe.grade(obs))


# ---------------------------------------------------------------- AC4 redaction


@pytest.mark.unit
def test_no_key_material_in_the_observation_or_the_record(tmp_path: Path) -> None:
    """A provider that echoes the Authorization header back cannot leak the key:
    the observer scrubs it, counts it, and the grader fails the run on it."""

    def echo(url: str, headers: dict[str, str]) -> dict[str, Any]:
        return observer.response_fields(
            401, {"error": {"code": 401, "message": headers["Authorization"]}}, 3
        )

    fake = _FakeTransport({"llm.x.api_key": SENTINEL}, echo)
    _obs, line = _run(_ONE, fake)
    assert SENTINEL not in line
    assert observer.INVALID_KEY not in line
    parsed = json.loads(line)
    assert parsed["probes"][0]["key_echoes"] == 1
    replay = tmp_path / "obs.json"
    replay.write_text(line, encoding="utf-8")
    record_path = tmp_path / "rec.json"
    summary_path = tmp_path / "summary.md"
    rc = probe.main(
        [
            "--container",
            "c",
            "--user",
            "u",
            "--contract-path",
            "p",
            "--replay",
            str(replay),
            "--record",
            str(record_path),
            "--summary",
            str(summary_path),
        ]
    )
    assert rc == probe.EXIT_FINDINGS
    for path in (record_path, summary_path):
        assert SENTINEL not in path.read_text(encoding="utf-8")
    assert "no_key_material_echoed" in _failed(probe.grade(parsed))


@pytest.mark.unit
def test_a_key_quoted_across_the_message_cap_is_not_left_half_scrubbed() -> None:
    """The cap is applied after the scrub: a key straddling it cannot survive
    as a prefix."""
    pad = "x" * (observer.MESSAGE_CAP - 10)

    def echo(url: str, headers: dict[str, str]) -> dict[str, Any]:
        quoted = headers["Authorization"].removeprefix("Bearer ")
        return observer.response_fields(
            401, {"error": {"code": 401, "message": pad + quoted + " rejected"}}, 3
        )

    fake = _FakeTransport({"llm.x.api_key": SENTINEL}, echo)
    obs, line = _run(_ONE, fake)
    assert SENTINEL[:10] not in line
    message = obs["probes"][0]["provider_error_message"]
    assert len(message) <= observer.MESSAGE_CAP
    assert obs["probes"][0]["key_echoes"] == 1
    assert "no_key_material_echoed" in _failed(probe.grade(obs))


@pytest.mark.unit
def test_a_live_rung_leaves_no_key_anywhere(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = _FakeTransport(
        {"llm.x.api_key": SENTINEL},
        lambda url, h: observer.response_fields(
            200 if SENTINEL in h["Authorization"] else 401,
            {"choices": [{}], "model": "m"},
            3,
        ),
    )
    obs, line = _run(_ONE, fake)
    assert SENTINEL not in line
    assert "redactions" not in json.loads(line)
    assert json.loads(line)["probes"][0]["secret_resolved"] is True
    record = probe.grade(obs)
    assert SENTINEL not in json.dumps(record.to_dict(target={}, as_of="t", observed={}))
    out = capsys.readouterr()
    assert SENTINEL not in out.out + out.err


# ---------------------------------------------------------------- AC5 quota codes


@pytest.mark.unit
@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("api.z.ai", "disable_until_reset"),
        ("API.Z.AI", "disable_until_reset"),
        ("eu.api.z.ai", "disable_until_reset"),
        ("notapi.z.ai", None),
        ("z.ai", None),
    ],
)
def test_quota_host_matching_follows_the_runtime(
    host: str, expected: str | None
) -> None:
    assert probe.quota_class(_recorded(), host, "1310") == expected


@pytest.mark.unit
def test_zai_1310_and_1113_are_distinguished() -> None:
    by_code: dict[str, str | None] = {}
    for code in ("1310", "1113"):
        obs = _without_openrouter()
        for p in obs["probes"]:
            if "cloud-glm" in p["backend_ids"]:
                for k in ("body_is_chat_completion", "model_echo", "usage"):
                    p.pop(k, None)
                p.update(
                    {
                        "http_status": 429,
                        "provider_error_code": code,
                        "provider_error_message": "quota",
                    }
                )
        record = probe.grade(obs)
        row = next(r for r in record.rungs if r["backend_id"] == "cloud-glm")
        assert row["verdict"] == probe.QUOTA_DEAD
        assert row["provider_error_code"] == code
        by_code[code] = row["quota_class"]
    assert by_code == {"1310": "disable_until_reset", "1113": "disable_until_billing"}


# ---------------------------------------------------------------- controls


@pytest.mark.unit
def test_a_control_that_reads_live_fails_the_run() -> None:
    obs = _without_openrouter()
    obs["controls"][0].update(
        {"http_status": 200, "body_is_chat_completion": True, "model_echo": "m"}
    )
    host = obs["controls"][0]["endpoint_host"]
    assert f"control/{host}" in _failed(probe.grade(obs))


@pytest.mark.unit
def test_an_endpoint_without_a_control_fails_the_run() -> None:
    obs = _without_openrouter()
    obs["controls"] = obs["controls"][1:]
    assert "wrong_key_control_per_endpoint" in _failed(probe.grade(obs))


@pytest.mark.unit
def test_a_contract_the_runtime_does_not_bind_fails() -> None:
    obs = _without_openrouter()
    obs["runtime_binds_contract"] = False
    assert "contract_is_runtime_binding" in _failed(probe.grade(obs))


# ---------------------------------------------------------------- could not look


@pytest.mark.unit
def test_an_observer_error_is_could_not_run_not_a_verdict() -> None:
    with pytest.raises(probe.ProbeInputError):
        probe.parse_observer_output('{"error": {"type": "ImportError"}}\n')
    with pytest.raises(probe.ProbeInputError):
        probe.parse_observer_output("")
    with pytest.raises(probe.ProbeInputError):
        probe.parse_observer_output("Traceback (most recent call last):\n")


def _fake_docker(tmp_path: Path, body: str) -> str:
    script = tmp_path / "docker"
    script.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


_ARGS = [
    "--container",
    "omninode-runtime-effects",
    "--user",
    "omniinfra",
    "--contract-path",
    "/app/data/delegation/bifrost_delegation.yaml",
]


@pytest.mark.unit
def test_an_unreachable_container_cannot_produce_a_pass(tmp_path: Path) -> None:
    docker = _fake_docker(tmp_path, 'echo "error: no such object" >&2; exit 1')
    record_path = tmp_path / "rec.json"
    rc = probe.main([*_ARGS, "--docker-bin", docker, "--record", str(record_path)])
    assert rc == probe.EXIT_INPUT
    assert not record_path.exists()


@pytest.mark.unit
def test_a_stopped_container_cannot_produce_a_pass(tmp_path: Path) -> None:
    docker = _fake_docker(tmp_path, 'echo "img|exited|2026-09-24T00:00:00Z"')
    assert probe.main([*_ARGS, "--docker-bin", docker]) == probe.EXIT_INPUT


@pytest.mark.unit
def test_the_live_path_grades_what_the_container_printed(tmp_path: Path) -> None:
    """The docker argv is pinned, the inspect never asks for the environment,
    and the exec's stdout is what gets graded."""
    obs = json.dumps(_without_openrouter())
    argv_log = tmp_path / "argv.log"
    docker = _fake_docker(
        tmp_path,
        f'echo "$@" >> "{argv_log}"\n'
        'if [ "$1" = inspect ]; then echo "omnibase-infra-runtime-effects|running|t"; exit 0; fi\n'
        "cat > /dev/null\n"
        f"cat <<'JSON'\n{obs}\nJSON",
    )
    record_path = tmp_path / "rec.json"
    rc = probe.main([*_ARGS, "--docker-bin", docker, "--record", str(record_path)])
    assert rc == probe.EXIT_OK
    calls = argv_log.read_text(encoding="utf-8").splitlines()
    assert calls[0] == (
        "inspect --format {{.Config.Image}}|{{.State.Status}}|{{.State.StartedAt}} "
        "omninode-runtime-effects"
    )
    assert calls[1] == (
        "exec -i -u omniinfra omninode-runtime-effects python - "
        "/app/data/delegation/bifrost_delegation.yaml"
    )
    written = json.loads(record_path.read_text(encoding="utf-8"))
    assert written["verdict"] == "pass"
    assert written["ticket"] == "OMN-16987"


@pytest.mark.unit
def test_the_contract_path_has_no_default() -> None:
    with pytest.raises(SystemExit):
        probe.main(["--container", "c", "--user", "u"])


# ---------------------------------------------------------------- the workflow


@pytest.mark.unit
def test_the_workflow_cannot_soften_its_verdict() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "continue-on-error" not in text
    assert "|| true" not in text
    assert "runs-on: [self-hosted, omnibase-verify, host-201]" in text
    triggers = yaml.safe_load(text)[True]  # PyYAML reads the bare `on` key as True
    assert set(triggers) == {"schedule", "workflow_dispatch"}
    assert 'exit "${status}"' in text
    step = next(
        s
        for s in yaml.safe_load(text)["jobs"]["provider-rung-canary"]["steps"]
        if "provider_rung_canary_probe.py" in str(s.get("run", ""))
    )
    run = step["run"]
    for flag in ("--container", "--user", "--contract-path", "--record", "--summary"):
        assert flag in run
    assert "omninode-runtime-effects" in run
    assert "/app/data/delegation/bifrost_delegation.yaml" in run


@pytest.mark.unit
def test_the_workflow_deploys_nothing() -> None:
    text = WORKFLOW.read_text(encoding="utf-8").lower()
    body = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    for verb in ("compose", "restart", "deploy-runtime", "docker run", "docker stop"):
        assert verb not in body, verb


# ---------------------------------------------------------------- import surface


@pytest.mark.unit
def test_the_observer_is_standard_library_plus_the_subject() -> None:
    import ast

    tree = ast.parse(Path(observer.__file__).read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    stdlib = set(getattr(__import__("sys"), "stdlib_module_names", ()))
    assert roots - stdlib - {"omnimarket", "httpx"} == set(), roots


@pytest.mark.unit
def test_the_probe_is_standard_library_only() -> None:
    import ast

    tree = ast.parse(Path(probe.__file__).read_text(encoding="utf-8"))
    stdlib = set(getattr(__import__("sys"), "stdlib_module_names", ()))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name.split(".")[0] in stdlib, a.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            assert node.module.split(".")[0] in stdlib | {"__future__"}, node.module


@pytest.mark.unit
def test_the_recorded_fixture_carries_no_key_material() -> None:
    text = (FIXTURES / "lane_2026_09_24.json").read_text(encoding="utf-8")
    assert "Bearer" not in text
    assert "redactions" not in json.loads(text)
    assert copy.deepcopy(json.loads(text))["runtime_binds_contract"] is True
