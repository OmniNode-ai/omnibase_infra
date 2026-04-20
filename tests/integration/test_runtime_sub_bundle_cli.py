# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for the runtime sub-bundle CLI (OMN-9342, follow-up to OMN-9332).

The OMN-9332 decomposition introduced four new bundles:
  - runtime-core
  - runtime-integrations
  - runtime-observability-projections
  - runtime-infrastructure

Plus a composed `runtime` bundle that pulls all four in via `includes:`.

Unit tests (`tests/unit/infra/test_runtime_bundle_decomposition.py`) cover the
static YAML/resolver invariants. These integration tests exercise the CLI
itself — `generate` and `validate` subcommands — to prove the operator-facing
contract works end-to-end without requiring a live Docker daemon for every
case.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

_CLI = ["uv", "run", "python", "-m", "omnibase_infra.docker.catalog.cli"]

_RUNTIME_CORE_SERVICES = frozenset(
    {
        "omninode-runtime",
        "runtime-effects",
        "agent-actions-consumer",
        "skill-lifecycle-consumer",
        "context-audit-consumer",
        "omninode-contract-resolver",
        "intelligence-api",
    }
)

_RUNTIME_INTEGRATIONS_REQUIRED_ENV = frozenset(
    {
        "CI_CALLBACK_TOKEN",
        "LINEAR_WEBHOOK_SECRET",
        "WAITLIST_NOTIFIER_SLACK_BOT_TOKEN",
    }
)

_SUB_BUNDLES = (
    "runtime-core",
    "runtime-integrations",
    "runtime-observability-projections",
    "runtime-infrastructure",
)


def _run_cli(
    args: list[str], *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Invoke the catalog CLI with the given args and return the CompletedProcess."""
    return subprocess.run(
        [*_CLI, *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
        env=env,
    )


def _generate_to(bundle: str, output: Path) -> subprocess.CompletedProcess[str]:
    return _run_cli(["generate", bundle, "--output", str(output)])


def _load_compose(path: Path) -> dict[str, object]:
    with open(path) as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), f"generated compose is not a mapping: {path}"
    return data


def _service_names(compose: dict[str, object]) -> set[str]:
    services = compose.get("services", {})
    assert isinstance(services, dict), (
        f"`services` block is not a mapping: {type(services).__name__}"
    )
    return set(services.keys())


@pytest.mark.integration
def test_generate_runtime_core_declares_seven_services(tmp_path: Path) -> None:
    """`catalog.cli generate runtime-core` emits the 7 core runtime services."""
    output = tmp_path / "runtime-core.yml"
    result = _generate_to("runtime-core", output)
    assert result.returncode == 0, f"generate failed: {result.stderr}"
    assert output.exists(), "output compose file was not written"

    compose = _load_compose(output)
    services = _service_names(compose)

    missing = _RUNTIME_CORE_SERVICES - services
    assert not missing, (
        f"runtime-core compose is missing services: {sorted(missing)}. "
        f"Full service set: {sorted(services)}"
    )


@pytest.mark.integration
def test_validate_runtime_integrations_reports_required_env(tmp_path: Path) -> None:
    """`catalog.cli validate runtime-integrations` surfaces the 3 required secrets.

    We strip the integration-specific secrets from the environment and redirect
    HOME so the CLI's `~/.omnibase/.env` auto-load can't silently satisfy them.
    The CLI must fail with the missing vars named in stderr.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in _RUNTIME_INTEGRATIONS_REQUIRED_ENV
    }
    env["HOME"] = str(tmp_path)

    result = _run_cli(["validate", "runtime-integrations"], env=env)
    assert result.returncode != 0, (
        "validate runtime-integrations unexpectedly succeeded with required vars unset"
    )
    for var in _RUNTIME_INTEGRATIONS_REQUIRED_ENV:
        assert var in result.stderr, (
            f"expected '{var}' in validate stderr, got:\n{result.stderr}"
        )


@pytest.mark.integration
def test_generate_composed_runtime_includes_all_sub_bundles(tmp_path: Path) -> None:
    """`catalog.cli generate runtime` emits the union of all four sub-bundles
    plus the transitive core + tracing services — preserving the operator
    contract after decomposition.
    """
    runtime_output = tmp_path / "runtime.yml"
    result = _generate_to("runtime", runtime_output)
    assert result.returncode == 0, f"generate runtime failed: {result.stderr}"
    runtime_compose = _load_compose(runtime_output)
    runtime_services = _service_names(runtime_compose)

    # Every sub-bundle's services must appear in the composed runtime stack.
    for sub in _SUB_BUNDLES:
        sub_output = tmp_path / f"{sub}.yml"
        sub_result = _generate_to(sub, sub_output)
        assert sub_result.returncode == 0, f"generate {sub} failed: {sub_result.stderr}"
        sub_services = _service_names(_load_compose(sub_output))
        missing = sub_services - runtime_services
        assert not missing, (
            f"composed runtime is missing services from {sub}: {sorted(missing)}"
        )

    # Transitive includes: core + tracing bring these in.
    for transitively_required in ("postgres", "redpanda", "phoenix"):
        assert transitively_required in runtime_services, (
            f"composed runtime missing transitively-included service "
            f"'{transitively_required}'"
        )


@pytest.mark.integration
@pytest.mark.xfail(
    strict=True,
    reason="OMN-9345: CatalogResolver iterates a set[str] over bundle names, "
    "producing non-deterministic service ordering in the generated compose. "
    "Remove this marker once OMN-9345 lands.",
)
def test_generate_runtime_is_deterministic_across_invocations(tmp_path: Path) -> None:
    """Two back-to-back `generate runtime` runs must produce byte-identical
    compose output. Resolver non-determinism (e.g. set iteration order) would
    cause operator machines to see spurious compose-file diffs.
    """
    out_a = tmp_path / "runtime-a.yml"
    out_b = tmp_path / "runtime-b.yml"

    result_a = _generate_to("runtime", out_a)
    result_b = _generate_to("runtime", out_b)
    assert result_a.returncode == 0, f"first generate failed: {result_a.stderr}"
    assert result_b.returncode == 0, f"second generate failed: {result_b.stderr}"

    content_a = out_a.read_text()
    content_b = out_b.read_text()
    assert content_a == content_b, (
        "generate runtime is non-deterministic — "
        "back-to-back invocations produced different output"
    )

    # Also assert the service-key ordering is stable (defensive against dict
    # reordering that happens to round-trip to identical YAML by accident).
    loaded_a = yaml.safe_load(content_a)
    loaded_b = yaml.safe_load(content_b)
    assert isinstance(loaded_a, dict) and isinstance(loaded_b, dict)
    svcs_a = loaded_a["services"]
    svcs_b = loaded_b["services"]
    assert isinstance(svcs_a, dict) and isinstance(svcs_b, dict)
    services_a = list(svcs_a.keys())
    services_b = list(svcs_b.keys())
    assert services_a == services_b, (
        f"service key ordering drift:\n  A: {json.dumps(services_a)}\n  B: {json.dumps(services_b)}"
    )
