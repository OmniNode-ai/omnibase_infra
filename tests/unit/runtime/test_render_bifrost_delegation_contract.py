# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for rendering Bifrost delegation solely from a typed overlay."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.models.enum_bifrost_endpoint_probe_failure_kind import (
    EnumBifrostEndpointProbeFailureKind,
)
from omnibase_infra.runtime.models.model_bifrost_endpoint_probe_failure import (
    ModelBifrostEndpointProbeFailure,
)
from omnibase_infra.runtime.render_bifrost_delegation_contract import (
    render_bifrost_delegation_contract,
)

_ROOT = Path(__file__).resolve().parents[3]
_OVERLAY = _ROOT / "docker" / "lane-overlays" / "dev.bifrost.yaml"
# OMN-17502: the cloud lane's committed overlay — zero local backends, stated.
_CLOUD_OVERLAY = _ROOT / "docker" / "lane-overlays" / "onex-dev.bifrost.yaml"
_ENDPOINT = "http://192.168.86.201:8000/v1/chat/completions"
# OMN-16833: the second live local rung — DS-V4-Flash on .200:8101.
_DS_V4_ENDPOINT = "http://192.168.86.200:8101/v1/chat/completions"
# OMN-17099: the embedding rung the dev overlay now binds (live probe
# 2026-09-27: .201:8002/v1/embeddings serves text-embedding-qwen3).
_EMBEDDING_ENDPOINT = "http://192.168.86.201:8002/v1/embeddings"


@pytest.mark.unit
def test_renderer_has_only_the_overlay_contract_arguments() -> None:
    assert tuple(inspect.signature(render_bifrost_delegation_contract).parameters) == (
        "source_path",
        "overlay_path",
        "target_path",
        "environ",
        "verify_endpoints",
        "endpoint_probe",
    )


def _write_base_contract(
    path: Path, *, coder_model: str | None = "Qwen3.8-27B"
) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "backends": [
                    {
                        "backend_id": "local-coder",
                        "model_name": coder_model,
                        "endpoint_url_env": "LLM_CODER_URL",
                        "required": True,
                    },
                    {
                        "backend_id": "local-heavy-reasoning",
                        "model_name": "Qwen3.8-27B",
                        "endpoint_url_env": "BIFROST_LOCAL_REASONER_ENDPOINT_URL",
                        "required": True,
                    },
                    {
                        "backend_id": "local-ds-v4-flash",
                        "model_name": "deepseek-v4-flash",
                        "endpoint_url_env": "BIFROST_LOCAL_DS_V4_FLASH_ENDPOINT_URL",
                        "required": True,
                    },
                    {
                        "backend_id": "local-reasoner",
                        "model_name": "retired",
                        "endpoint_url_env": "BIFROST_LOCAL_REASONER_ENDPOINT_URL",
                    },
                    {
                        "backend_id": "local-embedding",
                        "model_name": "text-embedding-qwen3",
                        "endpoint_url_env": "BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL",
                        "tier": "local",
                    },
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


@pytest.mark.unit
def test_typed_overlay_wins_over_poisoned_model_and_endpoint_environment(
    tmp_path: Path,
) -> None:
    source = tmp_path / "base.yaml"
    target = tmp_path / "rendered.yaml"
    _write_base_contract(source)

    rendered = render_bifrost_delegation_contract(
        source_path=source,
        overlay_path=_OVERLAY,
        target_path=target,
        environ={
            "LLM_CODER_URL": "http://cloud.invalid/v1/chat/completions",
            "LLM_CODER_MODEL_NAME": "poisoned",
            "BIFROST_LOCAL_CODER_ENDPOINT_URL": "http://192.168.86.201:8001/v1/chat/completions",
            "BIFROST_LOCAL_REASONER_ENDPOINT_URL": "http://cloud.invalid/v1/chat/completions",
            "BIFROST_LOCAL_DS_V4_FLASH_ENDPOINT_URL": "http://cloud.invalid/v1/chat/completions",
        },
    )

    assert rendered == target
    contract = yaml.safe_load(target.read_text(encoding="utf-8"))
    by_id = {backend["backend_id"]: backend for backend in contract["backends"]}
    for backend_id in ("local-coder", "local-heavy-reasoning"):
        assert by_id[backend_id]["endpoint_url"] == _ENDPOINT
        assert by_id[backend_id]["model_name"] == "Qwen3.8-27B"
        assert by_id[backend_id]["max_tokens"] == 65_536
        assert by_id[backend_id]["timeout_ms"] == 300_000
    # OMN-19251: the .200:8101 ds4-flash host is gone and the lane overlays no
    # longer declare this backend, so it now renders unbound exactly like
    # local-reasoner below — declared by the base contract, no live rung.
    assert by_id["local-ds-v4-flash"].get("endpoint_url") is None
    assert by_id["local-ds-v4-flash"]["model_name"] == "deepseek-v4-flash"
    # local-reasoner (.201:8001, GPU1 removed for RMA) stays unbound by design —
    # the renderer strips its stale env hint so it fails closed rather than
    # resolving a dead endpoint.
    assert by_id["local-reasoner"].get("endpoint_url") is None
    # OMN-17099: the dev overlay now binds local-embedding to its real
    # /v1/embeddings endpoint — the gap this ticket closes. A poisoned env
    # var must not resurrect it either.
    assert by_id["local-embedding"]["endpoint_url"] == _EMBEDDING_ENDPOINT
    assert by_id["local-embedding"]["model_name"] == "text-embedding-qwen3"
    assert all("endpoint_url_env" not in backend for backend in by_id.values())
    assert all("required" not in backend for backend in by_id.values())


@pytest.mark.unit
def test_render_accepts_and_writes_a_complete_embeddings_endpoint(
    tmp_path: Path,
) -> None:
    """OMN-17099: the render's own endpoint-shape gate must accept /v1/embeddings.

    ``_validate_rendered_contract`` used to require every non-null
    ``endpoint_url`` to end in ``/chat/completions``, so even a binding the
    schema now accepts (test_model_bifrost_lane_overlay's embeddings-path
    tests) would still fail here. A minimal base + overlay isolates this gate
    from the shared ``dev.bifrost.yaml`` fixture above.
    """
    source = tmp_path / "base.yaml"
    overlay_path = tmp_path / "lane.bifrost.yaml"
    target = tmp_path / "rendered.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "backends": [
                    {
                        "backend_id": "local-embedding",
                        "model_name": "text-embedding-qwen3",
                        "endpoint_url": None,
                        "tier": "local",
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    overlay_path.write_text(
        "schema_version: bifrost_lane_overlay.v3\n"
        "lane: dev\n"
        "locale: lab\n"
        "backends:\n"
        "  - backend_id: local-embedding\n"
        f"    endpoint_url: {_EMBEDDING_ENDPOINT!r}\n"
        "    served_model_id: text-embedding-qwen3\n"
        "    parameter_count: 0.6B\n"
        "    context_window: 8192\n"
        "    max_tokens: 1\n"
        "    timeout_ms: 30000\n",
        encoding="utf-8",
    )

    rendered = render_bifrost_delegation_contract(
        source_path=source, overlay_path=overlay_path, target_path=target
    )

    assert rendered == target
    contract = yaml.safe_load(target.read_text(encoding="utf-8"))
    by_id = {backend["backend_id"]: backend for backend in contract["backends"]}
    assert by_id["local-embedding"]["endpoint_url"] == _EMBEDDING_ENDPOINT


@pytest.mark.unit
def test_missing_lane_overlay_pin_fails_naming_the_lane(tmp_path: Path) -> None:
    """OMN-17150: no pin means fail loudly — never another lane's overlay.

    The old hardcoded ``dev.bifrost.yaml`` default sent every lane without its
    own mounted overlay through the dev lane's routing config. A lane that
    renders without ``BIFROST_LANE_OVERLAY_PATH`` must abort with an error that
    names the lane, before any overlay is read.
    """
    source = tmp_path / "base.yaml"
    target = tmp_path / "rendered.yaml"
    _write_base_contract(source)

    with pytest.raises(
        ProtocolConfigurationError,
        match=r"BIFROST_LANE_OVERLAY_PATH is not bound for lane 'lakshman'",
    ):
        render_bifrost_delegation_contract(
            source_path=source,
            target_path=target,
            environ={"ONEX_ENVIRONMENT": "lakshman"},
        )
    assert not target.exists()

    # A blank pin is the same defect as an absent one, and the lane name still
    # surfaces even when ONEX_ENVIRONMENT is itself unset.
    with pytest.raises(
        ProtocolConfigurationError,
        match=r"BIFROST_LANE_OVERLAY_PATH is not bound for lane "
        r"'<ONEX_ENVIRONMENT unset>'",
    ):
        render_bifrost_delegation_contract(
            source_path=source,
            target_path=target,
            environ={"BIFROST_LANE_OVERLAY_PATH": "   "},
        )
    assert not target.exists()


@pytest.mark.unit
def test_lane_overlay_pin_resolves_from_the_environment(tmp_path: Path) -> None:
    """The per-lane env pin is the production path (compose sets it per lane)."""
    source = tmp_path / "base.yaml"
    target = tmp_path / "rendered.yaml"
    _write_base_contract(source)

    rendered = render_bifrost_delegation_contract(
        source_path=source,
        target_path=target,
        environ={"BIFROST_LANE_OVERLAY_PATH": str(_OVERLAY)},
    )
    assert rendered == target

    # A pinned-but-absent lane file fails naming exactly that lane's file.
    with pytest.raises(
        ProtocolConfigurationError,
        match=r"overlay not found: .*lakshman\.bifrost\.yaml",
    ):
        render_bifrost_delegation_contract(
            source_path=source,
            target_path=tmp_path / "rendered2.yaml",
            environ={
                "BIFROST_LANE_OVERLAY_PATH": str(tmp_path / "lakshman.bifrost.yaml")
            },
        )


@pytest.mark.unit
def test_missing_or_malformed_overlay_fails_before_dispatch(tmp_path: Path) -> None:
    source = tmp_path / "base.yaml"
    invalid_overlay = tmp_path / "invalid-overlay.yaml"
    target = tmp_path / "rendered.yaml"
    _write_base_contract(source)
    invalid_overlay.write_text(
        # A lab lane with zero backends: schema-valid YAML, contract-invalid
        # overlay (OMN-16833 silent-degradation shape). A CLOUD lane with zero
        # backends is the one legal empty overlay — see the cloud tests below.
        "schema_version: bifrost_lane_overlay.v3\n"
        "lane: dev\n"
        "locale: lab\n"
        "backends: []\n",
        encoding="utf-8",
    )

    with pytest.raises(ProtocolConfigurationError, match="overlay is invalid"):
        render_bifrost_delegation_contract(
            source_path=source,
            overlay_path=invalid_overlay,
            target_path=target,
        )
    assert not target.exists()

    with pytest.raises(ProtocolConfigurationError, match="overlay not found"):
        render_bifrost_delegation_contract(
            source_path=source,
            overlay_path=tmp_path / "missing.yaml",
            target_path=target,
        )


@pytest.mark.unit
def test_overlay_model_overrides_a_different_base_model_and_records_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "base.yaml"
    target = tmp_path / "rendered.yaml"
    _write_base_contract(source, coder_model="Qwen3.8-27B-27b")

    render_bifrost_delegation_contract(
        source_path=source,
        overlay_path=_OVERLAY,
        target_path=target,
        verify_endpoints=False,
    )

    contract = yaml.safe_load(target.read_text(encoding="utf-8"))
    by_id = {backend["backend_id"]: backend for backend in contract["backends"]}
    assert by_id["local-coder"]["model_name"] == "Qwen3.8-27B"
    assert capsys.readouterr().out == (
        "[entrypoint] Bifrost backend 'local-coder' model_name 'Qwen3.8-27B-27b' "
        "overridden by lane overlay 'dev' served_model_id 'Qwen3.8-27B'\n"
    )


@pytest.mark.unit
def test_base_backend_without_a_model_name_key_is_still_malformed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "base.yaml"
    target = tmp_path / "rendered.yaml"
    _write_base_contract(source)
    base = yaml.safe_load(source.read_text(encoding="utf-8"))
    del base["backends"][0]["model_name"]
    source.write_text(yaml.safe_dump(base), encoding="utf-8")

    with pytest.raises(ProtocolConfigurationError, match="must declare model_name"):
        render_bifrost_delegation_contract(
            source_path=source,
            overlay_path=_OVERLAY,
            target_path=target,
        )
    assert not target.exists()


@pytest.mark.unit
def test_dark_rung_keeps_the_overlay_model_when_it_differs_from_the_base(
    tmp_path: Path,
) -> None:
    source = tmp_path / "base.yaml"
    overlay_path = tmp_path / "lane.yaml"
    target = tmp_path / "rendered.yaml"
    _write_base_contract(source)
    overlay = yaml.safe_load(_OVERLAY.read_text(encoding="utf-8"))
    for binding in overlay["backends"]:
        if binding["backend_id"] == "local-coder":
            binding["served_model_id"] = "operator-chosen-model"
            binding["serving"] = False
    overlay_path.write_text(yaml.safe_dump(overlay), encoding="utf-8")
    probed_models: list[str] = []

    def probe(
        endpoint_url: str, model_name: str, timeout: float
    ) -> ModelBifrostEndpointProbeFailure | None:
        probed_models.append(model_name)
        return None

    render_bifrost_delegation_contract(
        source_path=source,
        overlay_path=overlay_path,
        target_path=target,
        verify_endpoints=True,
        endpoint_probe=probe,
    )

    contract = yaml.safe_load(target.read_text(encoding="utf-8"))
    by_id = {backend["backend_id"]: backend for backend in contract["backends"]}
    assert by_id["local-coder"]["model_name"] == "operator-chosen-model"
    assert by_id["local-coder"]["endpoint_url"] is None
    assert by_id["local-coder"]["max_tokens"] == 65_536
    assert by_id["local-coder"]["timeout_ms"] == 300_000
    assert "operator-chosen-model" not in probed_models


@pytest.mark.unit
def test_unbound_base_model_is_materialized_from_the_typed_served_id(
    tmp_path: Path,
) -> None:
    source = tmp_path / "base.yaml"
    target = tmp_path / "rendered.yaml"
    _write_base_contract(source, coder_model=None)

    render_bifrost_delegation_contract(
        source_path=source,
        overlay_path=_OVERLAY,
        target_path=target,
    )

    contract = yaml.safe_load(target.read_text(encoding="utf-8"))
    by_id = {backend["backend_id"]: backend for backend in contract["backends"]}
    assert by_id["local-coder"]["model_name"] == "Qwen3.8-27B"


@pytest.mark.unit
def test_endpoint_probe_requires_advertised_served_id(tmp_path: Path) -> None:
    source = tmp_path / "base.yaml"
    _write_base_contract(source)

    def rejected_probe(
        endpoint_url: str, model_name: str, timeout: float
    ) -> ModelBifrostEndpointProbeFailure | None:
        assert (endpoint_url, model_name) in {
            (_ENDPOINT, "Qwen3.8-27B"),
            (_DS_V4_ENDPOINT, "deepseek-v4-flash"),
        }
        assert timeout > 0
        return ModelBifrostEndpointProbeFailure(
            kind=EnumBifrostEndpointProbeFailureKind.REFUSED,
            detail=f"model endpoint did not advertise {model_name}",
        )

    with pytest.raises(ProtocolConfigurationError, match="failed verification"):
        render_bifrost_delegation_contract(
            source_path=source,
            overlay_path=_OVERLAY,
            target_path=tmp_path / "rendered.yaml",
            verify_endpoints=True,
            endpoint_probe=rejected_probe,
        )


# ---------------------------------------------------------------------------
# OMN-17502: cloud-locale lanes. onex-dev runs in the cluster, where the .201 /
# .200 lab endpoints are not reachable (live probe from inside the namespace,
# 2026-09-01: ConnectionRefused on both). Its delegation has always been
# cloud-only; before this ticket the renderer had no way to say so, so the
# fail-closed OMN-17150 overlay requirement crash-looped the lane.
#
# The assertions below encode what the CONSUMER requires, read out of
# omnimarket rather than assumed:
#   * ``config_loader_bifrost_delegation.load_bifrost_delegation_config``
#     raises ``... references undeclared backend(s)`` when a routing rule or
#     ``default_backends`` names an id the contract does not declare — hence
#     disabled-not-deleted;
#   * ``handler_delegation_routing._load_bifrost_endpoints`` skips a backend
#     whose ``endpoint_url`` or ``model_name`` is empty, and
#     ``_tier_can_route_task`` then skips a tier with no resolvable backend —
#     hence a null endpoint is exactly "no local rung offered".
#
# The EXECUTABLE cross-repo seam test lives in omnimarket, next to its
# OMN-15628 sibling (``tests/integration/node_delegation_routing_reducer/
# test_omn15628_bifrost_renderer_reducer_seam.py``): omnimarket depends on
# omnibase_infra, not the reverse, and this repo's venv-purity gate (OMN-15620)
# refuses an omnimarket install into the test environment. It lands there on the
# next omnibase-infra repin. The seam was proven live for this change before
# landing — see the PR body's evidence block.
# ---------------------------------------------------------------------------

_CLOUD_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)


def _write_mixed_base_contract(
    path: Path, *, cloud_endpoint: str | None = _CLOUD_ENDPOINT
) -> None:
    """A base contract shaped like the packaged omnimarket one: local backends
    unbound (``endpoint_url: null`` + an env hint), cloud backends carrying a
    complete URL, and routing rules that REFERENCE the local ids."""
    path.write_text(
        yaml.safe_dump(
            {
                "backends": [
                    {
                        "backend_id": "local-coder",
                        "model_name": "Qwen3.8-27B",
                        "endpoint_url_env": "BIFROST_LOCAL_CODER_ENDPOINT_URL",
                        "endpoint_url": None,
                        "tier": "local",
                    },
                    {
                        "backend_id": "local-heavy-reasoning",
                        "model_name": "Qwen3.8-27B",
                        "endpoint_url_env": "BIFROST_LOCAL_CODER_ENDPOINT_URL",
                        "endpoint_url": None,
                        "tier": "local",
                    },
                    {
                        "backend_id": "local-ds-v4-flash",
                        "model_name": "deepseek-v4-flash",
                        "endpoint_url_env": "BIFROST_LOCAL_DS_V4_FLASH_ENDPOINT_URL",
                        "endpoint_url": None,
                        "tier": "local",
                    },
                    {
                        "backend_id": "local-embedding",
                        "model_name": "text-embedding-qwen3",
                        "endpoint_url_env": "BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL",
                        "endpoint_url": None,
                        "tier": "local",
                    },
                    {
                        "backend_id": "cloud-gemini-pro",
                        "model_name": "gemini-2.5-flash",
                        "endpoint_url": cloud_endpoint,
                        "tier": "frontier_api",
                    },
                ],
                "routing_rules": [
                    {
                        "rule_id": "d4e5f6a7-0001-4000-8000-000000000001",
                        "task_class": "code_generation",
                        "backend_ids": ["local-coder", "cloud-gemini-pro"],
                    }
                ],
                "default_backends": ["local-coder", "cloud-gemini-pro"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_cloud_overlay(path: Path, **binding_overrides: object) -> None:
    overlay = yaml.safe_load(_CLOUD_OVERLAY.read_text(encoding="utf-8"))
    overlay["backends"] = [
        {
            "backend_id": "cloud-gemini-pro",
            "endpoint_url": "https://cloud.example.invalid/v1/chat/completions",
            "served_model_id": "operator-chosen-gemini",
            "parameter_count": "unknown",
            "context_window": 32_768,
            "max_tokens": 16_384,
            "timeout_ms": 30_000,
            **binding_overrides,
        }
    ]
    path.write_text(yaml.safe_dump(overlay), encoding="utf-8")


@pytest.mark.unit
def test_cloud_overlay_rebinds_a_base_cloud_backends_model_and_endpoint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "base.yaml"
    overlay_path = tmp_path / "cloud.yaml"
    target = tmp_path / "rendered.yaml"
    _write_mixed_base_contract(source)
    _write_cloud_overlay(overlay_path)

    render_bifrost_delegation_contract(
        source_path=source,
        overlay_path=overlay_path,
        target_path=target,
        verify_endpoints=False,
    )

    contract = yaml.safe_load(target.read_text(encoding="utf-8"))
    by_id = {backend["backend_id"]: backend for backend in contract["backends"]}
    assert by_id["cloud-gemini-pro"]["model_name"] == "operator-chosen-gemini"
    assert by_id["cloud-gemini-pro"]["endpoint_url"] == (
        "https://cloud.example.invalid/v1/chat/completions"
    )
    assert by_id["cloud-gemini-pro"]["tier"] == "frontier_api"
    assert by_id["local-coder"]["endpoint_url"] is None


@pytest.mark.unit
def test_cloud_overlay_adds_a_fully_declared_cloud_backend(tmp_path: Path) -> None:
    source = tmp_path / "base.yaml"
    overlay_path = tmp_path / "cloud.yaml"
    target = tmp_path / "rendered.yaml"
    # The added endpoint must count as active even when the base has none.
    _write_mixed_base_contract(source, cloud_endpoint=None)
    _write_cloud_overlay(
        overlay_path,
        backend_id="cloud-custom",
        provider="gemini",
        tier="frontier_api",
        credential={"kind": "secret_ref", "secret_ref": "llm.custom.api_key"},
    )

    render_bifrost_delegation_contract(
        source_path=source,
        overlay_path=overlay_path,
        target_path=target,
        verify_endpoints=False,
    )

    contract = yaml.safe_load(target.read_text(encoding="utf-8"))
    backend = contract["backends"][-1]
    assert backend["backend_id"] == "cloud-custom"
    assert backend["provider"] == "gemini"
    assert backend["tier"] == "frontier_api"
    assert backend["secret_ref"] == "llm.custom.api_key"
    assert backend["model_name"] == "operator-chosen-gemini"
    assert backend["endpoint_url"] == (
        "https://cloud.example.invalid/v1/chat/completions"
    )


@pytest.mark.unit
@pytest.mark.parametrize("declared", [False, True])
@pytest.mark.parametrize("serving", [False, True])
def test_cloud_overlay_refuses_added_or_base_local_backends(
    tmp_path: Path,
    declared: bool,
    serving: bool,
) -> None:
    source = tmp_path / "base.yaml"
    overlay_path = tmp_path / "cloud.yaml"
    target = tmp_path / "rendered.yaml"
    _write_mixed_base_contract(source)
    backend_id = "added-local" if declared else "local-coder"
    declaration: dict[str, object] = (
        {"provider": "openai", "tier": "local", "credential": {"kind": "none"}}
        if declared
        else {}
    )
    _write_cloud_overlay(
        overlay_path,
        backend_id=backend_id,
        serving=serving,
        **declaration,
    )

    with pytest.raises(ProtocolConfigurationError) as excinfo:
        render_bifrost_delegation_contract(
            source_path=source,
            overlay_path=overlay_path,
            target_path=target,
        )

    message = str(excinfo.value)
    assert "lane 'onex-dev'" in message
    assert "locale 'cloud'" in message
    assert backend_id in message
    assert "local" in message
    assert "OMN-17502" in message
    assert not target.exists()


@pytest.mark.unit
def test_cloud_locale_renders_cloud_backends_and_disables_every_local_one(
    tmp_path: Path,
) -> None:
    source = tmp_path / "base.yaml"
    target = tmp_path / "rendered.yaml"
    _write_mixed_base_contract(source)

    rendered = render_bifrost_delegation_contract(
        source_path=source,
        overlay_path=_CLOUD_OVERLAY,
        target_path=target,
        # A poisoned environment must not resurrect a local rung: the overlay,
        # not the env, is the authority (OMN-15807) and this overlay says none.
        environ={
            "BIFROST_LOCAL_CODER_ENDPOINT_URL": "http://192.168.86.201:8000/v1/chat/completions",
            "BIFROST_LOCAL_DS_V4_FLASH_ENDPOINT_URL": "http://192.168.86.200:8101/v1/chat/completions",
        },
    )

    assert rendered == target
    contract = yaml.safe_load(target.read_text(encoding="utf-8"))
    by_id = {backend["backend_id"]: backend for backend in contract["backends"]}

    # The cloud rung is the only one carrying an endpoint.
    assert by_id["cloud-gemini-pro"]["endpoint_url"] == _CLOUD_ENDPOINT
    for local_id in (
        "local-coder",
        "local-heavy-reasoning",
        "local-ds-v4-flash",
        "local-embedding",
    ):
        assert by_id[local_id]["endpoint_url"] is None, local_id

    # Env transport hints are stripped for every backend, exactly as on a lab lane.
    assert all("endpoint_url_env" not in backend for backend in by_id.values())
    assert all("required" not in backend for backend in by_id.values())


@pytest.mark.unit
def test_cloud_locale_keeps_local_backend_ids_declared(tmp_path: Path) -> None:
    """Disabled, never deleted.

    ``load_bifrost_delegation_config`` (omnimarket, OMN-15628) raises
    ``Rule <id> references undeclared backend(s)`` / ``default_backends
    references undeclared backend(s)`` when a routing rule names a backend the
    contract does not declare — and the base contract's ``code_generation``
    rule and ``default_backends`` both name ``local-coder``. So a cloud lane's
    rendered contract must keep every local id DECLARED with a null endpoint
    (the shape ``_load_bifrost_endpoints`` drops), not drop the entries.
    """
    source = tmp_path / "base.yaml"
    target = tmp_path / "rendered.yaml"
    _write_mixed_base_contract(source)

    render_bifrost_delegation_contract(
        source_path=source, overlay_path=_CLOUD_OVERLAY, target_path=target
    )

    contract = yaml.safe_load(target.read_text(encoding="utf-8"))
    declared = {backend["backend_id"] for backend in contract["backends"]}
    for rule in contract["routing_rules"]:
        assert set(rule["backend_ids"]) <= declared
    assert set(contract["default_backends"]) <= declared
    assert "local-coder" in declared


@pytest.mark.unit
def test_cloud_locale_fails_closed_when_the_base_has_no_cloud_endpoint(
    tmp_path: Path,
) -> None:
    """A cloud lane with nothing to route to is a misconfiguration, not a lane
    with zero backends — the render must refuse rather than write a contract
    whose every rung is dead."""
    source = tmp_path / "base.yaml"
    target = tmp_path / "rendered.yaml"
    _write_mixed_base_contract(source, cloud_endpoint=None)

    with pytest.raises(ProtocolConfigurationError, match="no active endpoint"):
        render_bifrost_delegation_contract(
            source_path=source, overlay_path=_CLOUD_OVERLAY, target_path=target
        )
    assert not target.exists()


@pytest.mark.unit
def test_cloud_lane_overlay_still_resolves_only_from_its_own_pin(
    tmp_path: Path,
) -> None:
    """OMN-17150 is not relaxed by OMN-17502: a cloud lane that renders without
    ``BIFROST_LANE_OVERLAY_PATH`` still fails, naming the lane."""
    source = tmp_path / "base.yaml"
    target = tmp_path / "rendered.yaml"
    _write_mixed_base_contract(source)

    with pytest.raises(
        ProtocolConfigurationError,
        match=r"BIFROST_LANE_OVERLAY_PATH is not bound for lane 'onex-dev'",
    ):
        render_bifrost_delegation_contract(
            source_path=source,
            target_path=target,
            environ={"ONEX_ENVIRONMENT": "onex-dev"},
        )
    assert not target.exists()

    rendered = render_bifrost_delegation_contract(
        source_path=source,
        target_path=target,
        environ={"BIFROST_LANE_OVERLAY_PATH": str(_CLOUD_OVERLAY)},
    )
    assert rendered == target


# ---------------------------------------------------------------------------
# OMN-19432: a typed-decision backend (TypeSafe Jev) renders without the chat
# path, is never counted as a delegation rung, and a bare base is still refused.
# ---------------------------------------------------------------------------

_TYPED_DECISION_ENDPOINT = "https://decisions.example.invalid/v1/systemone"


def _write_base_with_typed_decision_backend(
    path: Path, *, decision_endpoint: str, cloud_endpoint: str | None = _CLOUD_ENDPOINT
) -> None:
    _write_mixed_base_contract(path, cloud_endpoint=cloud_endpoint)
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    contract["backends"].append(
        {
            "backend_id": "cloud-typed-decision",
            "provider": "typesafe",
            "model_name": "jev-latest",
            "endpoint_url": decision_endpoint,
            "secret_ref": "llm.typesafe.api_key",
            "tier": "typed_decision",
        }
    )
    path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")


@pytest.mark.unit
def test_typed_decision_backend_renders_its_operation_url_verbatim(
    tmp_path: Path,
) -> None:
    source = tmp_path / "base.yaml"
    target = tmp_path / "rendered.yaml"
    _write_base_with_typed_decision_backend(
        source, decision_endpoint=_TYPED_DECISION_ENDPOINT
    )

    render_bifrost_delegation_contract(
        source_path=source, overlay_path=_CLOUD_OVERLAY, target_path=target
    )

    contract = yaml.safe_load(target.read_text(encoding="utf-8"))
    by_id = {backend["backend_id"]: backend for backend in contract["backends"]}
    assert by_id["cloud-typed-decision"]["endpoint_url"] == _TYPED_DECISION_ENDPOINT


@pytest.mark.unit
def test_chat_backend_without_the_chat_path_is_still_refused(tmp_path: Path) -> None:
    """Positive control for the test above: the same non-chat URL on a backend
    that is NOT declared typed_decision still fails the render."""
    source = tmp_path / "base.yaml"
    target = tmp_path / "rendered.yaml"
    _write_mixed_base_contract(source, cloud_endpoint=_TYPED_DECISION_ENDPOINT)

    with pytest.raises(ProtocolConfigurationError, match="must be complete"):
        render_bifrost_delegation_contract(
            source_path=source, overlay_path=_CLOUD_OVERLAY, target_path=target
        )
    assert not target.exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    "bare",
    [
        "https://decisions.example.invalid/v1",
        "https://decisions.example.invalid/v1beta/",
        "https://decisions.example.invalid",
        "http://decisions.example.invalid/v1/systemone",
    ],
)
def test_typed_decision_backend_bare_base_or_plain_http_is_refused(
    tmp_path: Path, bare: str
) -> None:
    source = tmp_path / "base.yaml"
    target = tmp_path / "rendered.yaml"
    _write_base_with_typed_decision_backend(source, decision_endpoint=bare)

    with pytest.raises(ProtocolConfigurationError, match="typed-decision endpoint"):
        render_bifrost_delegation_contract(
            source_path=source, overlay_path=_CLOUD_OVERLAY, target_path=target
        )
    assert not target.exists()


@pytest.mark.unit
def test_typed_decision_backend_is_not_an_active_delegation_endpoint(
    tmp_path: Path,
) -> None:
    """A contract whose only endpoint is a typed-decision one has no rung to
    delegate to, so the render still refuses it as having no active endpoint."""
    source = tmp_path / "base.yaml"
    target = tmp_path / "rendered.yaml"
    _write_base_with_typed_decision_backend(
        source, decision_endpoint=_TYPED_DECISION_ENDPOINT, cloud_endpoint=None
    )

    with pytest.raises(ProtocolConfigurationError, match="no active endpoint"):
        render_bifrost_delegation_contract(
            source_path=source, overlay_path=_CLOUD_OVERLAY, target_path=target
        )
    assert not target.exists()
