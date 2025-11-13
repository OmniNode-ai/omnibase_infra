"""Pre-built workflow templates for common CI/CD scenarios."""

from ..models.github_actions import (
    CacheAction,
    CheckoutAction,
    PythonVersion,
    SetupPythonAction,
    UploadArtifactAction,
)
from ..models.workflow import (
    EventType,
    MatrixStrategy,
    PermissionLevel,
    PermissionSet,
    WorkflowConfig,
    WorkflowJob,
    WorkflowStep,
)


class WorkflowTemplates:
    """Collection of pre-built workflow templates."""

    @staticmethod
    def python_ci_template(
        name: str = "Python CI",
        python_versions: list[str] = None,
        test_command: str = "pytest",
        lint_commands: list[str] = None,
        coverage_threshold: int = 80,
    ) -> WorkflowConfig:
        """Create a comprehensive Python CI workflow.

        Args:
            name: Workflow name
            python_versions: Python versions to test against
            test_command: Command to run tests
            lint_commands: Commands for linting (defaults to flake8, black, isort)
            coverage_threshold: Minimum coverage percentage

        Returns:
            Complete Python CI workflow
        """
        if python_versions is None:
            python_versions = [PythonVersion.PYTHON_3_11, PythonVersion.PYTHON_3_12]

        if lint_commands is None:
            lint_commands = [
                "flake8 src tests",
                "black --check src tests",
                "isort --check-only src tests",
            ]

        # Define job steps
        steps = [
            WorkflowStep(name="Checkout code", **CheckoutAction().model_dump()),
            WorkflowStep(
                name="Set up Python ${{ matrix.python-version }}",
                **SetupPythonAction("${{ matrix.python-version }}").model_dump(),
            ),
            WorkflowStep(
                name="Cache dependencies",
                **CacheAction(
                    key="pip-${{ runner.os }}-${{ matrix.python-version }}-${{ hashFiles('**/requirements*.txt') }}",
                    path=["~/.cache/pip"],
                    restore_keys=["pip-${{ runner.os }}-${{ matrix.python-version }}-"],
                ).model_dump(),
            ),
            WorkflowStep(
                name="Install dependencies",
                run="""
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
                """.strip(),
            ),
            WorkflowStep(name="Install package", run="pip install -e ."),
        ]

        # Add linting steps
        for i, lint_cmd in enumerate(lint_commands):
            steps.append(
                WorkflowStep(name=f"Run linting ({lint_cmd.split()[0]})", run=lint_cmd)
            )

        # Add test and coverage steps
        steps.extend(
            [
                WorkflowStep(
                    name="Run tests with coverage",
                    run=f"{test_command} --cov=src --cov-report=xml --cov-report=term-missing",
                ),
                WorkflowStep(
                    name="Check coverage threshold",
                    run=f"coverage report --fail-under={coverage_threshold}",
                ),
                WorkflowStep(
                    name="Upload coverage reports",
                    **UploadArtifactAction(
                        name="coverage-${{ matrix.python-version }}",
                        path="coverage.xml",
                    ).model_dump(),
                    if_="success()",
                ),
            ]
        )

        # Create matrix strategy
        strategy = MatrixStrategy(
            matrix={"python-version": python_versions}, fail_fast=False
        )

        # Create test job
        test_job = WorkflowJob(
            name="Test", runs_on="ubuntu-latest", strategy=strategy, steps=steps
        )

        return WorkflowConfig(
            name=name,
            on=[EventType.PUSH, EventType.PULL_REQUEST],
            jobs={"test": test_job},
        )

    @staticmethod
    def docker_build_template(
        name: str = "Docker Build",
        dockerfile_path: str = "Dockerfile",
        image_name: str = "${{ github.repository }}",
        platforms: list[str] = None,
        push_to_registry: bool = True,
    ) -> WorkflowConfig:
        """Create a Docker build and push workflow.

        Args:
            name: Workflow name
            dockerfile_path: Path to Dockerfile
            image_name: Docker image name
            platforms: Target platforms (defaults to linux/amd64,linux/arm64)
            push_to_registry: Whether to push to registry

        Returns:
            Docker build workflow
        """
        if platforms is None:
            platforms = ["linux/amd64", "linux/arm64"]

        steps = [
            WorkflowStep(name="Checkout code", **CheckoutAction().model_dump()),
            WorkflowStep(
                name="Set up Docker Buildx", uses="docker/setup-buildx-action@v3"
            ),
            WorkflowStep(
                name="Log in to Docker Hub",
                uses="docker/login-action@v3",
                with_={
                    "username": "${{ secrets.DOCKER_USERNAME }}",
                    "password": "${{ secrets.DOCKER_PASSWORD }}",
                },
                if_="github.event_name != 'pull_request'",
            ),
            WorkflowStep(
                name="Extract metadata",
                id="meta",
                uses="docker/metadata-action@v5",
                with_={
                    "images": image_name,
                    "tags": [
                        "type=ref,event=branch",
                        "type=ref,event=pr",
                        "type=sha,prefix={{branch}}-",
                        "type=raw,value=latest,enable={{is_default_branch}}",
                    ],
                },
            ),
            WorkflowStep(
                name="Build and push",
                uses="docker/build-push-action@v5",
                with_={
                    "context": ".",
                    "file": dockerfile_path,
                    "platforms": ",".join(platforms),
                    "push": str(push_to_registry).lower(),
                    "tags": "${{ steps.meta.outputs.tags }}",
                    "labels": "${{ steps.meta.outputs.labels }}",
                    "cache-from": "type=gha",
                    "cache-to": "type=gha,mode=max",
                },
            ),
        ]

        build_job = WorkflowJob(
            name="Build",
            runs_on="ubuntu-latest",
            permissions=PermissionSet(
                contents=PermissionLevel.READ,
                id_token=PermissionLevel.WRITE,  # For OIDC
            ),
            steps=steps,
        )

        return WorkflowConfig(
            name=name,
            on=[EventType.PUSH, EventType.PULL_REQUEST],
            jobs={"build": build_job},
        )

    @staticmethod
    def security_scan_template(
        name: str = "Security Scan",
        scan_tools: list[str] = None,
        fail_on_high: bool = True,
    ) -> WorkflowConfig:
        """Create a security scanning workflow.

        Args:
            name: Workflow name
            scan_tools: Security tools to use (defaults to bandit, safety, semgrep)
            fail_on_high: Whether to fail on high severity issues

        Returns:
            Security scanning workflow
        """
        if scan_tools is None:
            scan_tools = ["bandit", "safety", "semgrep"]

        steps = [
            WorkflowStep(name="Checkout code", **CheckoutAction().model_dump()),
            WorkflowStep(name="Set up Python", **SetupPythonAction().model_dump()),
            WorkflowStep(
                name="Install security tools", run=f"pip install {' '.join(scan_tools)}"
            ),
        ]

        # Add tool-specific steps
        if "bandit" in scan_tools:
            steps.append(
                WorkflowStep(
                    name="Run Bandit security scan",
                    run="bandit -r src/ -f json -o bandit-report.json",
                    continue_on_error=not fail_on_high,
                )
            )

        if "safety" in scan_tools:
            steps.append(
                WorkflowStep(
                    name="Run Safety dependency scan",
                    run="safety check --json --output safety-report.json",
                    continue_on_error=not fail_on_high,
                )
            )

        if "semgrep" in scan_tools:
            steps.append(
                WorkflowStep(
                    name="Run Semgrep scan",
                    uses="returntocorp/semgrep-action@v1",
                    with_={"config": "auto"},
                )
            )

        # Upload security reports
        steps.append(
            WorkflowStep(
                name="Upload security reports",
                **UploadArtifactAction(
                    name="security-reports", path="*-report.json"
                ).model_dump(),
                if_="always()",
            )
        )

        security_job = WorkflowJob(
            name="Security Scan",
            runs_on="ubuntu-latest",
            permissions=PermissionSet(
                contents=PermissionLevel.READ, security_events=PermissionLevel.WRITE
            ),
            steps=steps,
        )

        return WorkflowConfig(
            name=name,
            on=[EventType.PUSH, EventType.PULL_REQUEST],
            jobs={"security": security_job},
        )

    @staticmethod
    def release_template(
        name: str = "Release",
        python_version: str = PythonVersion.PYTHON_3_12,
        build_artifacts: bool = True,
        publish_pypi: bool = True,
        create_github_release: bool = True,
    ) -> WorkflowConfig:
        """Create a release workflow that triggers on tags.

        Args:
            name: Workflow name
            python_version: Python version for building
            build_artifacts: Whether to build distribution artifacts
            publish_pypi: Whether to publish to PyPI
            create_github_release: Whether to create GitHub release

        Returns:
            Release workflow
        """
        steps = [
            WorkflowStep(
                name="Checkout code", **CheckoutAction(fetch_depth=0).model_dump()
            ),
            WorkflowStep(
                name="Set up Python", **SetupPythonAction(python_version).model_dump()
            ),
            WorkflowStep(
                name="Install build dependencies", run="pip install build twine"
            ),
        ]

        if build_artifacts:
            steps.extend(
                [
                    WorkflowStep(name="Build distribution", run="python -m build"),
                    WorkflowStep(name="Check distribution", run="twine check dist/*"),
                    WorkflowStep(
                        name="Upload artifacts",
                        **UploadArtifactAction(name="dist", path="dist/").model_dump(),
                    ),
                ]
            )

        if publish_pypi:
            steps.append(
                WorkflowStep(
                    name="Publish to PyPI",
                    uses="pypa/gh-action-pypi-publish@release/v1",
                    with_={"password": "${{ secrets.PYPI_API_TOKEN }}"},
                )
            )

        if create_github_release:
            steps.append(
                WorkflowStep(
                    name="Create GitHub Release",
                    uses="softprops/action-gh-release@v1",
                    with_={
                        "files": "dist/*" if build_artifacts else "",
                        "generate_release_notes": True,
                    },
                    # Use GH_PAT for GitHub API auth in generated workflows
                    env={"GITHUB_TOKEN": "${{ secrets.GH_PAT }}"},
                )
            )

        release_job = WorkflowJob(
            name="Release",
            runs_on="ubuntu-latest",
            permissions=PermissionSet(
                contents=PermissionLevel.WRITE, id_token=PermissionLevel.WRITE
            ),
            steps=steps,
        )

        return WorkflowConfig(
            name=name, on={"push": {"tags": ["v*"]}}, jobs={"release": release_job}
        )

    @staticmethod
    def performance_test_template(
        name: str = "Performance Tests",
        benchmark_command: str = "pytest --benchmark-only",
        baseline_branch: str = "main",
    ) -> WorkflowConfig:
        """Create a performance testing workflow.

        Args:
            name: Workflow name
            benchmark_command: Command to run benchmarks
            baseline_branch: Branch to compare against

        Returns:
            Performance testing workflow
        """
        steps = [
            WorkflowStep(name="Checkout code", **CheckoutAction().model_dump()),
            WorkflowStep(name="Set up Python", **SetupPythonAction().model_dump()),
            WorkflowStep(
                name="Install dependencies",
                run="""
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest-benchmark
                """.strip(),
            ),
            WorkflowStep(
                name="Run performance tests",
                run=f"{benchmark_command} --benchmark-json=benchmark.json",
            ),
            WorkflowStep(
                name="Upload benchmark results",
                **UploadArtifactAction(
                    name="benchmark-results", path="benchmark.json"
                ).model_dump(),
            ),
            WorkflowStep(
                name="Performance regression check",
                uses="benchmark-action/github-action-benchmark@v1",
                with_={
                    "tool": "pytest",
                    "output-file-path": "benchmark.json",
                    # Use GH_PAT for GitHub API auth in generated workflows
                    "github-token": "${{ secrets.GH_PAT }}",
                    "auto-push": True,
                    "comment-on-alert": True,
                    "alert-threshold": "200%",
                    "fail-on-alert": True,
                },
            ),
        ]

        perf_job = WorkflowJob(
            name="Performance Tests", runs_on="ubuntu-latest", steps=steps
        )

        return WorkflowConfig(
            name=name,
            on=[EventType.PUSH, EventType.PULL_REQUEST],
            jobs={"performance": perf_job},
        )

    @staticmethod
    def multi_os_template(
        name: str = "Multi-OS CI",
        operating_systems: list[str] = None,
        python_versions: list[str] = None,
    ) -> WorkflowConfig:
        """Create a multi-OS testing workflow.

        Args:
            name: Workflow name
            operating_systems: OS versions to test (defaults to ubuntu, windows, macos)
            python_versions: Python versions to test

        Returns:
            Multi-OS testing workflow
        """
        if operating_systems is None:
            operating_systems = ["ubuntu-latest", "windows-latest", "macos-latest"]

        if python_versions is None:
            python_versions = [PythonVersion.PYTHON_3_11, PythonVersion.PYTHON_3_12]

        steps = [
            WorkflowStep(name="Checkout code", **CheckoutAction().model_dump()),
            WorkflowStep(
                name="Set up Python ${{ matrix.python-version }}",
                **SetupPythonAction("${{ matrix.python-version }}").model_dump(),
            ),
            WorkflowStep(
                name="Install dependencies",
                run="""
python -m pip install --upgrade pip
pip install -r requirements.txt
                """.strip(),
            ),
            WorkflowStep(name="Run tests", run="pytest"),
        ]

        strategy = MatrixStrategy(
            matrix={"os": operating_systems, "python-version": python_versions},
            fail_fast=False,
        )

        test_job = WorkflowJob(
            name="Test", runs_on="${{ matrix.os }}", strategy=strategy, steps=steps
        )

        return WorkflowConfig(
            name=name,
            on=[EventType.PUSH, EventType.PULL_REQUEST],
            jobs={"test": test_job},
        )


def get_available_templates() -> dict[str, str]:
    """Get list of available workflow templates.

    Returns:
        Dictionary mapping template names to descriptions
    """
    return {
        "python_ci": "Comprehensive Python CI with testing, linting, and coverage",
        "docker_build": "Docker build and push workflow with multi-platform support",
        "security_scan": "Security scanning with multiple tools (bandit, safety, semgrep)",
        "release": "Release workflow that triggers on tags and publishes to PyPI",
        "performance_test": "Performance testing with benchmark comparison",
        "multi_os": "Multi-OS testing across Ubuntu, Windows, and macOS",
    }


def create_template(template_name: str, **kwargs) -> WorkflowConfig:
    """Create a workflow from a template.

    Args:
        template_name: Name of the template to create
        **kwargs: Template-specific parameters

    Returns:
        Workflow configuration

    Raises:
        ValueError: If template name is not recognized
    """
    templates = WorkflowTemplates()

    template_map = {
        "python_ci": templates.python_ci_template,
        "docker_build": templates.docker_build_template,
        "security_scan": templates.security_scan_template,
        "release": templates.release_template,
        "performance_test": templates.performance_test_template,
        "multi_os": templates.multi_os_template,
    }

    if template_name not in template_map:
        available = ", ".join(template_map.keys())
        raise ValueError(f"Unknown template '{template_name}'. Available: {available}")

    return template_map[template_name](**kwargs)
