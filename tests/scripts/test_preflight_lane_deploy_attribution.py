# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for scripts/preflight_lane_deploy_attribution.py (OMN-15218).

The behavior these tests pin is the behavior that did NOT exist during the two
incidents:

  * 2026-07-26T21:45:15Z — stability lane rebuilt while ``grant-6dbeae94`` was
    live and pinned to the previous digests. No actor, no reason, no block.
  * 2026-07-27T10:05:43-10:09:07Z — stability containers restarted by an unknown
    actor while the three ``batch-b551aa00`` grants were live.

Old behavior (RED): a stability deploy proceeds silently with live grants and
with nothing recorded about who or why. New behavior (GREEN): refused by
default, named grants in the refusal, override only by naming every live grant
id, and a durable attribution record either way.

Everything here is hermetic — the grant registry is a real file passed via
``--grants-file`` (or a real local git repo for the ``@main`` resolution test),
evaluation time is pinned with ``--now``, records are written to ``tmp_path``.
No lane is contacted, no network is used, and no real ``onex_change_control@main``
is required. Faithful dependency substitution, not mocks: the code path under
test is the one deploy-runtime.sh actually invokes.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "preflight_lane_deploy_attribution.py"


def _load_module() -> Any:
    mod_name = "preflight_lane_deploy_attribution"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPT)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module()

NOW = "2026-07-27T12:00:00Z"

# A grant shaped exactly like the live batch-b551aa00 entries that were pinned
# when the lane was restarted on 2026-07-27.
LIVE_GRANT = {
    "grant_id": "grant-6dbeae94-1111-4111-8111-111111111111",
    "runtime_lane": "prod",
    "image_digest": "sha256:" + "bb3bab37" * 8,
    "promotion_batch_id": "batch-b551aa00-2222-4222-8222-222222222222",
    "approved_by": "jonahgabriel",
    "expires_at": "2026-07-28T23:59:00Z",
    "created_at": "2026-07-26T21:00:00Z",
    "reason": "prod bootstrap promotion window",
}
SECOND_LIVE_GRANT = {
    **LIVE_GRANT,
    "grant_id": "grant-b94b0386-3333-4333-8333-333333333333",
    "image_digest": "sha256:" + "b94b0386" * 8,
}


def _write_grants(path: Path, entries: list[dict[str, Any]]) -> Path:
    import yaml

    path.write_text(
        yaml.safe_dump({"entries": entries}, sort_keys=False), encoding="utf-8"
    )
    return path


def _run(
    args: list[str],
    *,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke the preflight CLI with a scrubbed environment."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("ONEX_DEPLOY_") and key != "OMNI_HOME"
    }
    env.update(env_overrides or {})
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=120,
    )


def _record(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(result.stdout.strip().splitlines()[-1])
    return payload


def _stability_args(grants_file: Path, *, record_dir: Path | None = None) -> list[str]:
    args = [
        "--lane",
        "stability-test",
        "--compose-project",
        "omnibase-infra-stability-test",
        "--source",
        "pytest",
        "--invoking-command",
        "refresh_stability_lane.sh --execute",
        "--grants-file",
        str(grants_file),
        "--now",
        NOW,
        "--json",
    ]
    if record_dir is None:
        args.append("--check-only")
    else:
        args.extend(["--record-dir", str(record_dir)])
    return args


# --- ATTRIBUTION -------------------------------------------------------------


@pytest.mark.unit
def test_stability_deploy_refused_without_reason(tmp_path: Path) -> None:
    """RED on the old behavior: an unattributed stability deploy just ran."""
    grants = _write_grants(tmp_path / "grants.yaml", [])
    result = _run(_stability_args(grants))
    assert result.returncode == 1, result.stdout + result.stderr
    record = _record(result)
    assert record["result"] == "REFUSE"
    assert any("ONEX_DEPLOY_REASON" in reason for reason in record["refusal_reasons"])


@pytest.mark.unit
@pytest.mark.parametrize("placeholder", ["x", "test", "redeploy", "n/a", "   ", "..."])
def test_placeholder_reason_refused(tmp_path: Path, placeholder: str) -> None:
    grants = _write_grants(tmp_path / "grants.yaml", [])
    result = _run(
        _stability_args(grants), env_overrides={"ONEX_DEPLOY_REASON": placeholder}
    )
    assert result.returncode == 1
    assert _record(result)["result"] == "REFUSE"


@pytest.mark.unit
def test_stability_deploy_allowed_with_real_reason(tmp_path: Path) -> None:
    grants = _write_grants(tmp_path / "grants.yaml", [])
    result = _run(
        _stability_args(grants),
        env_overrides={
            "ONEX_DEPLOY_REASON": "OMN-15181 prod bootstrap rehearsal on stability"
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = _record(result)
    assert record["result"] == "ALLOW"
    assert record["ticket"] == "OMN-15181"
    assert record["lane"] == "stability-test"
    assert record["grant_guard"]["verdict"] == "CLEAR"
    # Attribution must actually identify somebody, not just exist as a key.
    assert record["actor"]["identity"]
    assert record["actor"]["host"]
    assert record["invoking_command"] == "refresh_stability_lane.sh --execute"


@pytest.mark.unit
def test_explicit_ticket_env_wins_over_reason_text(tmp_path: Path) -> None:
    grants = _write_grants(tmp_path / "grants.yaml", [])
    result = _run(
        _stability_args(grants),
        env_overrides={
            "ONEX_DEPLOY_REASON": "refresh to pick up merged dev work",
            "ONEX_DEPLOY_TICKET": "OMN-15218",
        },
    )
    assert result.returncode == 0
    assert _record(result)["ticket"] == "OMN-15218"


@pytest.mark.unit
def test_dev_lane_is_not_governed(tmp_path: Path) -> None:
    """dev is the fully-mutable test platform: recorded, never blocked."""
    result = _run(
        [
            "--compose-project",
            "omnibase-infra",
            "--source",
            "pytest",
            "--now",
            NOW,
            "--check-only",
            "--json",
        ]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = _record(result)
    assert record["lane"] == "dev"
    assert record["attribution_required"] is False
    assert record["grant_guard"]["verdict"] == "NOT_APPLICABLE"
    assert record["result"] == "ALLOW"


# --- GRANT INTERLOCK ---------------------------------------------------------


@pytest.mark.unit
def test_live_grant_refuses_stability_deploy(tmp_path: Path) -> None:
    """The 2026-07-27 scenario: live grants, deploy must NOT proceed silently."""
    grants = _write_grants(tmp_path / "grants.yaml", [LIVE_GRANT])
    result = _run(
        _stability_args(grants),
        env_overrides={
            "ONEX_DEPLOY_REASON": "OMN-15218 refresh stability to latest dev"
        },
    )
    assert result.returncode == 1, result.stdout + result.stderr
    record = _record(result)
    assert record["result"] == "REFUSE"
    assert record["grant_guard"]["verdict"] == "LIVE_GRANTS"
    # The refusal must NAME the grants — a nameless "blocked" is not actionable.
    assert LIVE_GRANT["grant_id"] in " ".join(record["refusal_reasons"])
    assert record["grant_guard"]["live_grants"][0]["grant_id"] == LIVE_GRANT["grant_id"]


@pytest.mark.unit
def test_acknowledging_every_live_grant_allows_and_is_recorded(tmp_path: Path) -> None:
    grants = _write_grants(tmp_path / "grants.yaml", [LIVE_GRANT, SECOND_LIVE_GRANT])
    ack = f"{LIVE_GRANT['grant_id']},{SECOND_LIVE_GRANT['grant_id']}"
    result = _run(
        _stability_args(grants),
        env_overrides={
            "ONEX_DEPLOY_REASON": "OMN-15218 stability refresh, grants reissued after",
            "ONEX_DEPLOY_GRANT_ACK": ack,
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = _record(result)
    assert record["result"] == "ALLOW"
    assert record["grant_guard"]["acknowledged"] is True
    # The override itself is attribution: it must be in the record.
    assert LIVE_GRANT["grant_id"] in record["grant_guard"]["acknowledgement_tokens"]
    assert (
        SECOND_LIVE_GRANT["grant_id"] in record["grant_guard"]["acknowledgement_tokens"]
    )


@pytest.mark.unit
def test_partial_acknowledgement_still_refuses(tmp_path: Path) -> None:
    grants = _write_grants(tmp_path / "grants.yaml", [LIVE_GRANT, SECOND_LIVE_GRANT])
    result = _run(
        _stability_args(grants),
        env_overrides={
            "ONEX_DEPLOY_REASON": "OMN-15218 stability refresh",
            "ONEX_DEPLOY_GRANT_ACK": LIVE_GRANT["grant_id"],
        },
    )
    assert result.returncode == 1
    record = _record(result)
    assert record["result"] == "REFUSE"
    assert record["grant_guard"]["unacknowledged_grant_ids"] == [
        SECOND_LIVE_GRANT["grant_id"]
    ]


@pytest.mark.unit
@pytest.mark.parametrize("blanket", ["true", "1", "yes", "ack", "all"])
def test_blanket_acknowledgement_is_not_an_override(
    tmp_path: Path, blanket: str
) -> None:
    """A stale env flag must not pre-authorize a grant that did not exist yet."""
    grants = _write_grants(tmp_path / "grants.yaml", [LIVE_GRANT])
    result = _run(
        _stability_args(grants),
        env_overrides={
            "ONEX_DEPLOY_REASON": "OMN-15218 stability refresh",
            "ONEX_DEPLOY_GRANT_ACK": blanket,
        },
    )
    assert result.returncode == 1
    assert _record(result)["grant_guard"]["acknowledged"] is False


@pytest.mark.unit
def test_consumed_and_expired_grants_do_not_block(tmp_path: Path) -> None:
    consumed = {
        **LIVE_GRANT,
        "grant_id": "grant-consumed-4444-4444-8444-444444444444",
        "consumed": True,
    }
    expired = {
        **LIVE_GRANT,
        "grant_id": "grant-expired-5555-4555-8555-555555555555",
        "expires_at": "2026-07-26T00:00:00Z",
    }
    grants = _write_grants(tmp_path / "grants.yaml", [consumed, expired])
    result = _run(
        _stability_args(grants),
        env_overrides={
            "ONEX_DEPLOY_REASON": "OMN-15218 stability refresh to latest dev"
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = _record(result)
    assert record["grant_guard"]["verdict"] == "CLEAR"
    assert record["grant_guard"]["live_grants"] == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "content",
    [
        "entries: [[[",  # unparseable YAML
        "not_entries: []",  # wrong top-level key
        "entries: 3",  # entries not a list
        "entries:\n  - grant_id: grant-x\n",  # entry missing required fields
        "entries:\n  - "
        + "\n    ".join(
            f"{k}: {v}" for k, v in {**LIVE_GRANT, "expires_at": "not-a-date"}.items()
        ),
    ],
)
def test_unreadable_grant_state_fails_closed(tmp_path: Path, content: str) -> None:
    """Indeterminate grant state is not a pass."""
    grants = tmp_path / "grants.yaml"
    grants.write_text(content, encoding="utf-8")
    result = _run(
        _stability_args(grants),
        env_overrides={
            "ONEX_DEPLOY_REASON": "OMN-15218 stability refresh to latest dev"
        },
    )
    assert result.returncode == 1, result.stdout + result.stderr
    record = _record(result)
    assert record["grant_guard"]["verdict"] == "UNREADABLE"
    assert record["grant_guard"]["errors"]


@pytest.mark.unit
def test_missing_grant_source_fails_closed(tmp_path: Path) -> None:
    """No resolvable onex_change_control clone == UNREADABLE, not CLEAR."""
    result = _run(
        [
            "--lane",
            "stability-test",
            "--source",
            "pytest",
            "--grants-repo",
            str(tmp_path / "no-such-clone"),
            "--now",
            NOW,
            "--check-only",
            "--json",
        ],
        env_overrides={
            "ONEX_DEPLOY_REASON": "OMN-15218 stability refresh to latest dev"
        },
    )
    assert result.returncode == 1
    assert _record(result)["grant_guard"]["verdict"] == "UNREADABLE"


@pytest.mark.unit
def test_unreadable_state_override_requires_the_sentinel_token(tmp_path: Path) -> None:
    grants = tmp_path / "grants.yaml"
    grants.write_text("entries: [[[", encoding="utf-8")
    base_env = {
        "ONEX_DEPLOY_REASON": "OMN-15218 emergency stability refresh, grants unreadable"
    }

    wrong = _run(
        _stability_args(grants),
        env_overrides={**base_env, "ONEX_DEPLOY_GRANT_ACK": "grant-6dbeae94"},
    )
    assert wrong.returncode == 1

    right = _run(
        _stability_args(grants),
        env_overrides={**base_env, "ONEX_DEPLOY_GRANT_ACK": _mod.UNREADABLE_ACK_TOKEN},
    )
    assert right.returncode == 0, right.stdout + right.stderr
    assert _record(right)["grant_guard"]["acknowledged"] is True


# --- DURABLE RECORD ----------------------------------------------------------


@pytest.mark.unit
def test_allowed_deploy_writes_durable_record(tmp_path: Path) -> None:
    grants = _write_grants(tmp_path / "grants.yaml", [])
    record_dir = tmp_path / "state"
    result = _run(
        _stability_args(grants, record_dir=record_dir),
        env_overrides={
            "ONEX_DEPLOY_REASON": "OMN-15218 stability refresh to latest dev"
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr

    log_path = record_dir / "deploy-log.jsonl"
    assert log_path.exists(), "the append-only deploy log must exist"
    logged = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert logged["lane"] == "stability-test"
    assert logged["reason"].startswith("OMN-15218")
    assert logged["actor"]["identity"]

    records = list((record_dir / "deploy-attribution").glob("*.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text(encoding="utf-8"))["result"] == "ALLOW"


@pytest.mark.unit
def test_refused_attempt_is_also_recorded(tmp_path: Path) -> None:
    """A refused rebuild attempt is itself attribution-worthy."""
    grants = _write_grants(tmp_path / "grants.yaml", [LIVE_GRANT])
    record_dir = tmp_path / "state"
    result = _run(
        _stability_args(grants, record_dir=record_dir),
        env_overrides={
            "ONEX_DEPLOY_REASON": "OMN-15218 stability refresh to latest dev"
        },
    )
    assert result.returncode == 1
    logged = json.loads(
        (record_dir / "deploy-log.jsonl").read_text(encoding="utf-8").strip()
    )
    assert logged["result"] == "REFUSE"
    assert logged["grant_guard"]["live_grants"][0]["grant_id"] == LIVE_GRANT["grant_id"]


@pytest.mark.unit
def test_check_only_writes_nothing(tmp_path: Path) -> None:
    grants = _write_grants(tmp_path / "grants.yaml", [])
    result = _run(
        _stability_args(grants),
        env_overrides={
            "ONEX_DEPLOY_REASON": "OMN-15218 stability refresh to latest dev"
        },
    )
    assert result.returncode == 0
    assert _record(result)["mode"] == "check-only"
    assert not (tmp_path / "deploy-log.jsonl").exists()


# --- @main resolution (real git, no network) ---------------------------------


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("git") is None, reason="git binary not available")
def test_grant_state_resolves_from_origin_main(tmp_path: Path) -> None:
    """The interlock reads ``origin/main``, not the working tree or a branch.

    Built against a real local git remote so the ``@main`` anchor (the
    anti-self-issue property of the whole grant scheme) is exercised, without any
    network or a real onex_change_control.
    """
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig"),
        "GIT_CONFIG_SYSTEM": "/dev/null",
    }

    def git(cwd: Path, *args: str) -> None:
        subprocess.run(
            ["git", "-C", str(cwd), *args], check=True, capture_output=True, env=env
        )

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git(upstream, "init", "--initial-branch=main")
    git(upstream, "config", "user.email", "test@omninode.ai")
    git(upstream, "config", "user.name", "test")
    grants_dir = upstream / "grants"
    grants_dir.mkdir()
    _write_grants(grants_dir / "prod_promotion_grants.yaml", [LIVE_GRANT])
    git(upstream, "add", "-A")
    git(upstream, "commit", "-m", "grant")

    clone = tmp_path / "onex_change_control"
    subprocess.run(
        ["git", "clone", str(upstream), str(clone)],
        check=True,
        capture_output=True,
        env=env,
    )

    # The working tree says "no grants"; origin/main says otherwise. The
    # interlock must follow origin/main.
    _write_grants(clone / "grants" / "prod_promotion_grants.yaml", [])

    raw, commit = _mod.fetch_grant_bytes_from_main(clone)
    assert commit
    block = _mod.evaluate_grant_state(raw, now=datetime(2026, 7, 27, 12, 0, tzinfo=UTC))
    assert block["verdict"] == "LIVE_GRANTS"
    assert block["live_grants"][0]["grant_id"] == LIVE_GRANT["grant_id"]


# --- pure helpers ------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("compose_project", "expected"),
    [
        ("omnibase-infra", "dev"),
        ("omnibase-infra-stability-test", "stability-test"),
        ("omnibase-infra-prod", "prod"),
        ("omnibase-infra-judge", "judge"),
    ],
)
def test_lane_from_compose_project(compose_project: str, expected: str) -> None:
    assert _mod.lane_from_compose_project(compose_project) == expected


@pytest.mark.unit
def test_governed_lane_policy_matches_the_incident_surface() -> None:
    assert "stability-test" in _mod.GOVERNED_LANES
    assert "prod" in _mod.GOVERNED_LANES
    assert frozenset({"stability-test"}) == _mod.GRANT_INTERLOCK_LANES


@pytest.mark.unit
def test_apply_acknowledgement_is_case_insensitive_and_id_scoped() -> None:
    block = _mod.evaluate_grant_state(
        json.dumps({"entries": [LIVE_GRANT]}).encode(),  # JSON is valid YAML
        now=datetime(2026, 7, 27, 12, 0, tzinfo=UTC),
    )
    assert (
        _mod.apply_acknowledgement(block, [str(LIVE_GRANT["grant_id"]).upper()])[
            "acknowledged"
        ]
        is True
    )
    assert _mod.apply_acknowledgement(block, ["grant-other"])["acknowledged"] is False


@pytest.mark.unit
def test_evaluate_grant_state_hashes_the_exact_bytes() -> None:
    raw = b"entries: []\n"
    block = _mod.evaluate_grant_state(raw, now=datetime(2026, 7, 27, 12, 0, tzinfo=UTC))
    import hashlib

    assert block["grants_sha256"] == hashlib.sha256(raw).hexdigest()
