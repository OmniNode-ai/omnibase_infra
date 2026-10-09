# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20830: the exposure-reader gate sees the readers that exist.

The omnimarket contract pin has been held at f859c7af since 10-07 because
``exposure-reader-coverage`` reported three exposures as unread that the
omnidash local dashboard renders, and refused a fourth for a field omnimarket's
own reader model declares.

Each test names the failure it exists to catch:

* a component bound to a topic on an omnidash local page contract is not
  counted as a reader of it;
* a local page binding counts for a topic it does not name;
* a missing, empty or unparseable local pages directory reads as "no readers"
  instead of failing closed;
* ``read_all_rows``, a field of ``ModelProjectionBackendReader``, is refused
  as an unknown key;
* a key the model does not declare is accepted, or ``read_all_rows`` takes a
  non-boolean.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from omnibase_infra.validators.bus_backed_exposure_readers import (
    Finding,
    ReaderSurfaceError,
    check_exposure_readers,
    collect_backend_reader_surface,
    collect_bus_backed_exposures,
    collect_local_page_readers,
    evaluate,
)

pytestmark = pytest.mark.unit

_CREDENTIALS = "onex.snapshot.projection.tenant-credentials.v1"
_ACTIVITY = "onex.snapshot.projection.topic-activity.v1"


def _surface(root: Path) -> Path:
    """The Market projection surface, with the reader model as omnimarket declares it."""
    surface = root / "projection"
    surface.mkdir(parents=True, exist_ok=True)
    (surface / "models.py").write_text(
        "from typing import Literal\n\n\n"
        "class ModelProjectionBackendReader(BaseModel):\n"
        "    model_config = ConfigDict(frozen=True, extra='forbid')\n\n"
        "    id: str\n"
        '    kind: Literal["projection_status_page"]\n'
        "    route: str\n"
        "    projection_slot: str\n"
        "    read_all_rows: bool = False\n",
        encoding="utf-8",
    )
    (surface / "morning_page.py").write_text(
        "def build_morning_page(topic_map, cache):\n"
        "    return read_backend_projection(\n"
        "        topic_map,\n"
        "        cache,\n"
        '        reader_id="onex_status_page",\n'
        '        projection_slot="topic_activity",\n'
        '        route="/",\n'
        "    )\n",
        encoding="utf-8",
    )
    (surface / "api_server.py").write_text(
        '@app.get("/", response_class=HTMLResponse)\n'
        "async def status_page() -> HTMLResponse:\n"
        "    return _render_status_page()\n",
        encoding="utf-8",
    )
    return surface


def _contract(root: Path, node: str, exposure: dict[str, object]) -> Path:
    node_dir = root / "contracts" / node
    node_dir.mkdir(parents=True, exist_ok=True)
    body = ["name: " + node, "projection_api:"]
    body += [f"  {key}: {json.dumps(value)}" for key, value in exposure.items()]
    (node_dir / "contract.yaml").write_text("\n".join(body) + "\n", encoding="utf-8")
    return root / "contracts"


def _pages(root: Path, bindings: dict[str, list[str]]) -> Path:
    pages = root / "pages-local"
    pages.mkdir(parents=True, exist_ok=True)
    components = [
        {
            "component_id": component_id,
            "data_bindings": [
                {"binding_id": f"{component_id}-{i}", "projection_topic": topic}
                for i, topic in enumerate(topics)
            ],
        }
        for component_id, topics in bindings.items()
    ]
    (pages / "credentials.contracts.yaml").write_text(
        json.dumps({"components": components}), encoding="utf-8"
    )
    return pages


def _omnidash(root: Path) -> tuple[Path, Path]:
    registry = root / "registry.json"
    registry.write_text(json.dumps({"components": {}}), encoding="utf-8")
    layouts = root / "templates"
    layouts.mkdir(parents=True, exist_ok=True)
    return registry, layouts


def _gate(root: Path, contracts: Path, pages: Path) -> tuple[int, str]:
    registry, layouts = _omnidash(root)
    out = io.StringIO()
    code = check_exposure_readers(
        [contracts], registry, layouts, _surface(root), pages, stream=out
    )
    return code, out.getvalue()


def test_a_local_page_binding_is_a_reader(tmp_path: Path) -> None:
    contracts = _contract(
        tmp_path,
        "node_projection_tenant_credentials",
        {"expose": True, "topic": _CREDENTIALS, "bus_backed": True},
    )
    code, report = _gate(
        tmp_path, contracts, _pages(tmp_path, {"credentials-keys": [_CREDENTIALS]})
    )
    assert code == 0, report
    assert f"{_CREDENTIALS} :: local-page:credentials/credentials-keys" in report


def test_a_local_page_binding_counts_only_for_the_topic_it_names(
    tmp_path: Path,
) -> None:
    contracts = _contract(
        tmp_path,
        "node_projection_tenant_credentials",
        {"expose": True, "topic": _CREDENTIALS, "bus_backed": True},
    )
    code, report = _gate(
        tmp_path,
        contracts,
        _pages(tmp_path, {"other": ["onex.snapshot.projection.other.v1"]}),
    )
    assert code == 1
    assert "no_reader" in report


def test_a_stale_opt_out_is_caught_through_a_local_page_reader(tmp_path: Path) -> None:
    contracts = _contract(
        tmp_path,
        "node_projection_tenant_credentials",
        {
            "expose": True,
            "topic": _CREDENTIALS,
            "bus_backed": True,
            "consumers": "none",
            "consumers_reason": "nothing renders this exposure yet",
        },
    )
    code, report = _gate(
        tmp_path, contracts, _pages(tmp_path, {"credentials-keys": [_CREDENTIALS]})
    )
    assert code == 1
    assert "stale_opt_out" in report


def test_a_missing_local_pages_dir_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ReaderSurfaceError, match="does not exist"):
        collect_local_page_readers(tmp_path / "no-such-dir")


def test_a_local_pages_dir_with_no_contracts_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ReaderSurfaceError, match="holds no"):
        collect_local_page_readers(tmp_path / "empty")


def test_an_unparseable_local_page_contract_fails_closed(tmp_path: Path) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "x.contracts.yaml").write_text("components: [unclosed", encoding="utf-8")
    with pytest.raises(ReaderSurfaceError):
        collect_local_page_readers(pages)


def _backend_exposure(tmp_path: Path, **extra: object) -> list[Finding]:
    reader = {
        "id": "onex_status_page",
        "kind": "projection_status_page",
        "route": "/",
        "projection_slot": "topic_activity",
        **extra,
    }
    contracts = _contract(
        tmp_path,
        "node_projection_topic_activity",
        {
            "expose": True,
            "topic": _ACTIVITY,
            "bus_backed": True,
            "backend_readers": [reader],
        },
    )
    surface = collect_backend_reader_surface(_surface(tmp_path))
    return evaluate(collect_bus_backed_exposures([contracts], surface), {})


def test_read_all_rows_is_a_field_of_the_reader_model(tmp_path: Path) -> None:
    assert _backend_exposure(tmp_path, read_all_rows=True) == []


def test_a_key_the_reader_model_does_not_declare_is_still_refused(
    tmp_path: Path,
) -> None:
    (finding,) = _backend_exposure(tmp_path, page_size=50)
    assert finding.code == "invalid_backend_reader"
    assert "unknown keys page_size" in finding.reason


def test_read_all_rows_must_be_a_boolean(tmp_path: Path) -> None:
    (finding,) = _backend_exposure(tmp_path, read_all_rows="yes")
    assert finding.code == "invalid_backend_reader"
    assert "read_all_rows must be true or false" in finding.reason


_WORKFLOW = (
    Path(__file__).resolve().parents[3]
    / ".github/workflows/exposure-reader-coverage.yml"
)
_PINS = Path(__file__).resolve().parents[3] / ".github/sibling-pins.yaml"
#: The omnidash commit the pin stood at before this change, which predates every
#: local page binding and registry reader the unread exposures have.
_STALE_OMNIDASH_PIN = "cc628179717f4830bdd6c5a5e1b8d0c7bc4042ff"


def test_the_workflow_checks_out_and_passes_the_local_pages() -> None:
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert "            src/pages/local\n" in text
    assert "--local-pages-dir omnidash/src/pages/local" in text
    assert "test -d omnidash/src/pages/local" in text


def test_the_omnidash_pin_moved_past_the_commit_with_no_readers() -> None:
    pins = _PINS.read_text(encoding="utf-8")
    assert f"omnidash: {_STALE_OMNIDASH_PIN}" not in pins
    assert _STALE_OMNIDASH_PIN not in _WORKFLOW.read_text(encoding="utf-8")
