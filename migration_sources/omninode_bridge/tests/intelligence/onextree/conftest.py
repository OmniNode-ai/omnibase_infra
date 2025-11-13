"""
Pytest fixtures for OnexTree tests.

Provides sample project structures and pre-loaded query engines.
"""

from pathlib import Path

import pytest


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    """
    Create sample project structure for testing.

    Structure:
    - src/
      - services/
        - api.py
        - worker.py
      - models/
        - user.py
        - product.py
      - utils/
        - helper.py
    - tests/
      - test_api.py
      - test_worker.py
    - docs/
      - README.md
    - config.yaml
    """
    # Create src directory
    src = tmp_path / "src"
    src.mkdir()

    # Create services
    services = src / "services"
    services.mkdir()
    (services / "__init__.py").write_text("# Services")
    (services / "api.py").write_text("# API service\nclass ApiService:\n    pass")
    (services / "worker.py").write_text("# Worker service\nclass Worker:\n    pass")

    # Create models
    models = src / "models"
    models.mkdir()
    (models / "__init__.py").write_text("# Models")
    (models / "user.py").write_text("# User model\nclass User:\n    pass")
    (models / "product.py").write_text("# Product model\nclass Product:\n    pass")

    # Create utils
    utils = src / "utils"
    utils.mkdir()
    (utils / "__init__.py").write_text("# Utils")
    (utils / "helper.py").write_text("# Helper functions\ndef help():\n    pass")

    # Create tests
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("# Tests")
    (tests / "test_api.py").write_text("# API tests\ndef test_api():\n    pass")
    (tests / "test_worker.py").write_text(
        "# Worker tests\ndef test_worker():\n    pass"
    )

    # Create docs
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "README.md").write_text("# Documentation\n\nProject docs here.")

    # Create root files
    (tmp_path / "config.yaml").write_text("# Configuration\nkey: value")
    (tmp_path / "README.md").write_text("# Project\n\nMain readme.")

    return tmp_path


@pytest.fixture
async def loaded_query_engine(sample_project: Path):
    """
    Query engine with loaded tree from sample project.

    Returns:
        Tuple of (query_engine, tree_root)
    """
    from omninode_bridge.intelligence.onextree.generator import OnexTreeGenerator
    from omninode_bridge.intelligence.onextree.query_engine import OnexTreeQueryEngine

    generator = OnexTreeGenerator(sample_project)
    tree_root = await generator.generate_tree()

    engine = OnexTreeQueryEngine()
    await engine.load_tree(tree_root)

    return engine, tree_root


@pytest.fixture
def large_project(tmp_path: Path) -> Path:
    """
    Create large project structure for performance testing.

    Creates 1000+ files for testing scalability.
    """
    # Create 10 modules with 10 files each
    for i in range(10):
        module_dir = tmp_path / f"module_{i}"
        module_dir.mkdir()

        for j in range(100):
            file_path = module_dir / f"file_{j}.py"
            file_path.write_text(f"# File {i}-{j}\nclass Class{i}{j}:\n    pass")

    return tmp_path
