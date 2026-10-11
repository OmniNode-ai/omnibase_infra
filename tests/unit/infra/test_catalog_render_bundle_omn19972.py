# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The public bundle renderer and its port bridge (OMN-19972).

``render_bundle`` is what ``onex local up|down|status`` calls: it renders a
catalog bundle from an explicit checkout root, names the env files compose must
interpolate from, and bridges the port-publishing services into the
``ModelDeploymentTopology`` the local provision node takes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_core.enums.enum_deployment_mode import EnumDeploymentMode
from omnibase_infra.docker.catalog import cli
from omnibase_infra.docker.catalog.generator import (
    read_env_file,
    render_bundle,
    topology_from_resolved_stack,
)
from omnibase_infra.docker.catalog.resolver import CatalogResolver
from omnibase_infra.errors import ProtocolConfigurationError

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CATALOG = _REPO_ROOT / "docker" / "catalog"


def _local_required_env() -> set[str]:
    return CatalogResolver(catalog_dir=str(_CATALOG)).resolve(["local"]).required_env


def _env_file(tmp_path: Path, *, skip: str | None = None, extra: str = "") -> Path:
    lines = [f"{var}=x{i}" for i, var in enumerate(sorted(_local_required_env()))]
    lines = [line for line in lines if skip is None or not line.startswith(skip + "=")]
    path = tmp_path / "local.env"
    path.write_text("\n".join(lines) + "\n" + extra, encoding="utf-8")
    return path


@pytest.mark.unit
class TestRenderBundle:
    def test_renders_local_bundle_and_bridges_ported_services(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "compose.yml"
        rendered = render_bundle(
            ["local"],
            infra_root=_REPO_ROOT,
            env_file=_env_file(tmp_path),
            output_path=out,
            process_environment={},
        )

        compose = yaml.safe_load(out.read_text(encoding="utf-8"))
        assert compose["name"] == rendered.project == "omnibase-infra-local"
        assert rendered.compose_file_path == str(out.resolve())
        assert set(rendered.services) == set(compose["services"])

        topo = rendered.topology.local_services()
        ports = {name: svc.local.host_port for name, svc in topo.items() if svc.local}
        # The bundle's port override wins over the manifest (3102, not 3002).
        assert ports["projection-api"] == 3102
        assert ports["omninode-runtime"] == 8085
        assert ports["runtime-effects"] == 8086
        assert all(
            svc.mode == EnumDeploymentMode.LOCAL
            and svc.local is not None
            and svc.local.compose_service == name
            for name, svc in topo.items()
        )
        # Services with no published port are in the render, not the topology.
        for portless in ("forward-migration", "migration-gate"):
            assert portless in rendered.services
            assert portless not in topo

    def test_env_files_are_policy_then_operator(self, tmp_path: Path) -> None:
        env = _env_file(tmp_path)
        rendered = render_bundle(
            ["local"],
            infra_root=_REPO_ROOT,
            env_file=env,
            output_path=tmp_path / "c.yml",
            process_environment={},
        )
        assert rendered.env_files == (
            str((_REPO_ROOT / "docker" / "runtime-policy.env").resolve()),
            str(env.resolve()),
        )

    def test_missing_env_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ProtocolConfigurationError, match="Env file not found"):
            render_bundle(
                ["local"],
                infra_root=_REPO_ROOT,
                env_file=tmp_path / "absent.env",
                output_path=tmp_path / "c.yml",
            )

    def test_template_placeholder_is_refused_by_name(self, tmp_path: Path) -> None:
        env = _env_file(tmp_path, extra="SOME_KEY=__REPLACE_WITH_SECRET__\n")
        with pytest.raises(ProtocolConfigurationError, match="SOME_KEY"):
            render_bundle(
                ["local"],
                infra_root=_REPO_ROOT,
                env_file=env,
                output_path=tmp_path / "c.yml",
            )

    def test_missing_required_var_names_it_and_no_value(self, tmp_path: Path) -> None:
        var = sorted(_local_required_env())[0]
        with pytest.raises(ProtocolConfigurationError, match=var) as exc:
            render_bundle(
                ["local"],
                infra_root=_REPO_ROOT,
                env_file=_env_file(tmp_path, skip=var),
                output_path=tmp_path / "c.yml",
                process_environment={},
            )
        assert "x0" not in str(exc.value)

    def test_process_environment_satisfies_a_required_var(self, tmp_path: Path) -> None:
        var = sorted(_local_required_env())[0]
        render_bundle(
            ["local"],
            infra_root=_REPO_ROOT,
            env_file=_env_file(tmp_path, skip=var),
            output_path=tmp_path / "c.yml",
            process_environment={var: "set-in-shell"},
        )

    def test_root_without_catalog_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ProtocolConfigurationError, match="not an omnibase_infra"):
            render_bundle(["local"], infra_root=tmp_path)

    def test_bridge_skips_portless_services(self) -> None:
        resolved = CatalogResolver(catalog_dir=str(_CATALOG)).resolve(["local"])
        topo = topology_from_resolved_stack(resolved)
        assert set(topo.services) == {
            name for name, m in resolved.manifests.items() if m.ports is not None
        }


@pytest.mark.unit
def test_read_env_file_matches_cli_loader(tmp_path: Path) -> None:
    path = tmp_path / "e.env"
    path.write_text(
        "# comment\n\nexport A='1'\nB=\"two\"\nA=ignored\nnoequals\n",
        encoding="utf-8",
    )
    assert read_env_file(path) == {"A": "1", "B": "two"}


@pytest.mark.unit
class TestRuntimeImageBuildArgs:
    def test_quad_for_the_named_checkout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[Path] = []

        def fake_sha(root: Path) -> str:
            seen.append(root)
            return "cafef00d"

        monkeypatch.setattr(cli, "_git_sha_at", fake_sha)
        pairs = dict(p.split("=", 1) for p in cli.runtime_image_build_args(_REPO_ROOT))
        assert seen == [_REPO_ROOT]
        assert pairs["GIT_SHA"] == pairs["VCS_REF"] == "cafef00d"
        assert pairs["RUNTIME_VERSION"] == cli._runtime_version_at(_REPO_ROOT)
        assert pairs["BUILD_DATE"]

    def test_unknown_sha_fails_fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli, "_git_sha_at", lambda _root: "unknown")
        with pytest.raises(RuntimeError, match="OMN-12965"):
            cli.runtime_image_build_args(_REPO_ROOT)
