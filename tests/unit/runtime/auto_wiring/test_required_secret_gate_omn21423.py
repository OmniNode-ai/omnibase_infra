# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Required secret presence gates consumer group membership (OMN-21423 AC3)."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from omnibase_infra.models.model_secret_gated_membership import (
    ModelSecretGatedMembership,
)
from omnibase_infra.runtime.auto_wiring import (
    discover_contracts_from_paths,
    drop_blank_secret_names,
    env_secret_store_is_authoritative,
    gate_manifest_on_required_secrets,
    required_secret_names,
    resolve_present_secret_names,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.discovery import _extract_required_secrets
from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
from omnibase_infra.secret_stores.adapter_env_secret_store import AdapterEnvSecretStore
from omnibase_infra.utils.util_runtime_packages import ENV_ACTIVE_RUNTIME_PACKAGES

pytestmark = pytest.mark.unit

_WRITE_CONTRACT = "node_github_repo_write_effect"
_PLAIN_CONTRACT = "node_plain_effect"


class _NamesOnlyStore:
    """Fail if a caller attempts to read a secret value."""

    def __init__(self, keys: list[str]) -> None:
        self.keys = keys
        self.list_calls = 0

    async def list_keys(self, prefix: str | None = None) -> list[str]:
        self.list_calls += 1
        return self.keys

    async def get_secret(self, key: str) -> str | None:
        raise AssertionError("The gate must only list names")


@pytest.fixture
def manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ModelAutoWiringManifest:
    """Discover GitHub-shaped and ungated contracts without a package filter."""
    monkeypatch.delenv(ENV_ACTIVE_RUNTIME_PACKAGES, raising=False)
    paths = []
    for name in (_WRITE_CONTRACT, _PLAIN_CONTRACT):
        raw = {
            "name": name,
            "node_type": "EFFECT_GENERIC",
            "contract_version": {"major": 1, "minor": 0, "patch": 0},
            "event_bus": {
                "subscribe_topics": [f"onex.cmd.omnimarket.{name}-requested.v1"],
                "publish_topics": [f"onex.evt.omnimarket.{name}-completed.v1"],
            },
        }
        if name == _WRITE_CONTRACT:
            raw["secrets"] = {
                "GITHUB_TOKEN": {"description": "x", "required": True},
                "OPTIONAL_KEY": {"required": False},
            }
        path = tmp_path / name / "contract.yaml"
        path.parent.mkdir()
        path.write_text(yaml.safe_dump(raw))
        paths.append(path)
    result = discover_contracts_from_paths(paths)
    assert not result.errors
    assert not result.skips
    assert len(result.contracts) == 2
    return result


@pytest.mark.parametrize(
    ("policy", "addr", "expected"),
    [
        ("disabled", "", True),
        ("disabled", "https://infisical.example", True),
        ("best_effort", "", True),
        ("best_effort", "   ", True),
        ("best_effort", "https://infisical.example", False),
        ("required", "https://infisical.example", False),
    ],
)
def test_env_secret_store_is_authoritative(
    policy: str, addr: str, expected: bool
) -> None:
    assert env_secret_store_is_authoritative(policy, addr) is expected


def test_discovery_parses_required_secret_names(
    manifest: ModelAutoWiringManifest,
) -> None:
    contracts = {c.name: c for c in manifest.contracts}
    assert contracts[_WRITE_CONTRACT].required_secrets == ("GITHUB_TOKEN",)
    assert contracts[_PLAIN_CONTRACT].required_secrets == ()
    assert required_secret_names(manifest) == frozenset({"GITHUB_TOKEN"})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({}, ()),
        ({"secrets": None}, ()),
        ({"secrets": ["TOKEN"]}, ()),
        (
            {
                "secrets": {
                    "Z_TOKEN": {"required": True},
                    "A_TOKEN": {"required": True},
                    "OPTIONAL": {"required": False},
                    "UNSPECIFIED": {},
                    "SCALAR": True,
                    "STRING": {"required": "true"},
                    "NUMBER": {"required": 1},
                    123: {"required": True},
                }
            },
            ("A_TOKEN", "Z_TOKEN"),
        ),
    ],
)
def test_extract_required_secrets(raw: dict, expected: tuple[str, ...]) -> None:
    assert _extract_required_secrets(raw) == expected


@pytest.mark.asyncio
async def test_absent_secret_excludes_contract_with_membership(
    manifest: ModelAutoWiringManifest, caplog: pytest.LogCaptureFixture
) -> None:
    store = _NamesOnlyStore(["OTHER"])
    present = await resolve_present_secret_names(
        store.list_keys, required_secret_names(manifest)
    )
    assert present == frozenset()
    gated = gate_manifest_on_required_secrets(manifest, present)

    assert store.list_calls == 1
    assert [c.name for c in gated.contracts] == [_PLAIN_CONTRACT]
    assert len(gated.skips) == 1
    skip = gated.skips[0]
    assert skip.reason == "required_secret_absent"
    assert skip.contract_path == manifest.contracts[0].contract_path
    assert len(skip.membership) == 1
    membership = skip.membership[0]
    assert membership.contract == _WRITE_CONTRACT
    assert membership.consumer_group == ""
    assert membership.token == "absent"
    assert membership.joined is False
    assert membership.secret_ref == "GITHUB_TOKEN"
    assert "token=absent" in membership.render()
    assert membership.render() in caplog.text


@pytest.mark.asyncio
async def test_store_error_fails_closed(
    manifest: ModelAutoWiringManifest, caplog: pytest.LogCaptureFixture
) -> None:
    sentinel = "ghp_SENTINEL_never_print"

    class _FailingStore:
        async def list_keys(self, prefix: str | None = None) -> list[str]:
            raise RuntimeError(sentinel)

    present = await resolve_present_secret_names(
        _FailingStore().list_keys, required_secret_names(manifest)
    )
    assert present == frozenset()
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    assert "RuntimeError" in record.getMessage()
    assert record.exc_info is None

    gated = gate_manifest_on_required_secrets(manifest, present)
    assert [c.name for c in gated.contracts] == [_PLAIN_CONTRACT]
    assert len(gated.skips) == 1
    assert gated.skips[0].reason == "required_secret_absent"
    assert gated.skips[0].membership[0].joined is False
    assert sentinel not in caplog.text


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_secret_value_counts_as_absent(value: str) -> None:
    names = frozenset({"GITHUB_TOKEN", "OTHER"})
    values = {"GITHUB_TOKEN": value, "OTHER": "x"}
    assert drop_blank_secret_names(names, values) == frozenset({"OTHER"})


@pytest.mark.asyncio
async def test_blank_env_secret_gates_contract(
    manifest: ModelAutoWiringManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "")
    present = drop_blank_secret_names(
        await resolve_present_secret_names(
            AdapterEnvSecretStore().list_keys, required_secret_names(manifest)
        ),
        {"GITHUB_TOKEN": ""},
    )
    assert present == frozenset()
    gated = gate_manifest_on_required_secrets(manifest, present)
    assert [c.name for c in gated.contracts] == [_PLAIN_CONTRACT]
    assert gated.skips[0].reason == "required_secret_absent"


@pytest.mark.asyncio
async def test_present_secret_keeps_contract(manifest: ModelAutoWiringManifest) -> None:
    store = _NamesOnlyStore(["GITHUB_TOKEN", "OTHER"])
    present = await resolve_present_secret_names(
        store.list_keys, required_secret_names(manifest)
    )
    assert present == frozenset({"GITHUB_TOKEN"})
    gated = gate_manifest_on_required_secrets(manifest, present)
    assert store.list_calls == 1
    assert gated is manifest
    assert _WRITE_CONTRACT in {c.name for c in gated.contracts}
    assert not any(s.reason == "required_secret_absent" for s in gated.skips)


@pytest.mark.asyncio
async def test_empty_names_do_not_list_store() -> None:
    store = _NamesOnlyStore(["OTHER"])
    assert (
        await resolve_present_secret_names(store.list_keys, frozenset()) == frozenset()
    )
    assert store.list_calls == 0


@pytest.mark.asyncio
async def test_env_store_never_leaks_secret_value(
    manifest: ModelAutoWiringManifest,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "ghp_SENTINEL_never_print"
    monkeypatch.setenv("GITHUB_TOKEN", sentinel)
    caplog.set_level(logging.DEBUG)

    async def forbidden_get_secret(self: AdapterEnvSecretStore, key: str) -> str | None:
        raise AssertionError("The gate must only list names")

    monkeypatch.setattr(AdapterEnvSecretStore, "get_secret", forbidden_get_secret)
    for present in (True, False):
        if not present:
            monkeypatch.delenv("GITHUB_TOKEN")
        names = await resolve_present_secret_names(
            AdapterEnvSecretStore().list_keys, required_secret_names(manifest)
        )
        gated = gate_manifest_on_required_secrets(manifest, names)
        assert (_WRITE_CONTRACT in {c.name for c in gated.contracts}) is present
        assert sentinel not in caplog.text
        assert sentinel not in repr(gated)
        assert sentinel not in gated.model_dump_json()


@pytest.mark.asyncio
async def test_gated_contract_never_reaches_wiring(
    manifest: ModelAutoWiringManifest,
) -> None:
    gated = gate_manifest_on_required_secrets(manifest, frozenset())
    assert _WRITE_CONTRACT not in {c.name for c in gated.contracts}
    engine = MagicMock()
    report = await wire_from_manifest(gated, dispatch_engine=engine, event_bus=None)
    assert {r.contract_name for r in report.results} == {_PLAIN_CONTRACT}
    engine.register_dispatcher.assert_not_called()


@pytest.mark.parametrize(("joined", "rendered"), [(False, "false"), (True, "true")])
def test_membership_render_format(joined: bool, rendered: str) -> None:
    membership = ModelSecretGatedMembership(
        contract=_WRITE_CONTRACT,
        secret_ref="GITHUB_TOKEN",
        token="present" if joined else "absent",
        joined=joined,
    )
    assert membership.render() == (
        f"contract={_WRITE_CONTRACT} secret=GITHUB_TOKEN "
        f"token={'present' if joined else 'absent'} joined={rendered}"
    )
