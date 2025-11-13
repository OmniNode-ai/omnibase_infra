#!/usr/bin/env python3
"""Pydantic-based CI workflow generator for Claude workflows - Fixed version."""

from pathlib import Path
from typing import Any

import yaml

from .models.workflow import (
    PermissionLevel,
    PermissionSet,
    WorkflowConfig,
    WorkflowJob,
    WorkflowStep,
)


class ClaudeWorkflowGenerator:
    """Generator for Claude-based GitHub Actions workflows using Pydantic models."""

    def __init__(self, project_root: Path | None = None):
        """Initialize the workflow generator."""
        if project_root is None:
            current_file = Path(__file__)
            project_root = current_file.parents[3]

        self.project_root = Path(project_root)
        self.workflows_dir = self.project_root / ".github" / "workflows"

    def create_claude_workflow(self) -> WorkflowConfig:
        """Create the main Claude workflow configuration using Pydantic models."""

        # Define workflow permissions
        permissions = PermissionSet(
            contents=PermissionLevel.READ,
            pull_requests=PermissionLevel.READ,
            issues=PermissionLevel.READ,
            id_token=PermissionLevel.WRITE,
            actions=PermissionLevel.READ,
        )

        # Create checkout step
        checkout_step = WorkflowStep(
            name="Checkout repository",
            uses="actions/checkout@v4",
            with_={"fetch-depth": 1},
        )

        # Create Claude Code step
        claude_step = WorkflowStep(
            name="Run Claude Code",
            id="claude",
            uses="anthropics/claude-code-action@v1",
            with_={
                "claude_code_oauth_token": "${{ secrets.CLAUDE_CODE_OAUTH_TOKEN }}",
                "additional_permissions": "actions: read",
            },
        )

        # Define the job with proper if condition
        claude_job = WorkflowJob(
            runs_on="ubuntu-latest",
            permissions=permissions,
            if_=(
                "(github.event_name == 'issue_comment' && contains(github.event.comment.body, '@claude')) ||\n"
                "(github.event_name == 'pull_request_review_comment' && contains(github.event.comment.body, '@claude')) ||\n"
                "(github.event_name == 'pull_request_review' && contains(github.event.review.body, '@claude')) ||\n"
                "(github.event_name == 'issues' && (contains(github.event.issue.body, '@claude') || contains(github.event.issue.title, '@claude')))"
            ),
            steps=[checkout_step, claude_step],
        )

        # Define workflow triggers
        workflow_triggers = {
            "issue_comment": {"types": ["created"]},
            "pull_request_review_comment": {"types": ["created"]},
            "issues": {"types": ["opened", "assigned"]},
            "pull_request_review": {"types": ["submitted"]},
        }

        # Create the complete workflow
        workflow = WorkflowConfig(
            name="Claude Code", on=workflow_triggers, jobs={"claude": claude_job}
        )

        return workflow

    def create_claude_code_review_workflow(self) -> WorkflowConfig:
        """Create the Claude Code Review workflow configuration using Pydantic models."""

        # Define workflow permissions
        permissions = PermissionSet(
            contents=PermissionLevel.READ,
            pull_requests=PermissionLevel.READ,
            issues=PermissionLevel.READ,
            id_token=PermissionLevel.WRITE,
        )

        # Create checkout step
        checkout_step = WorkflowStep(
            name="Checkout repository",
            uses="actions/checkout@v4",
            with_={"fetch-depth": 1},
        )

        # Create Claude Code Review step
        claude_review_step = WorkflowStep(
            name="Run Claude Code Review",
            id="claude-review",
            uses="anthropics/claude-code-action@v1",
            with_={
                "claude_code_oauth_token": "${{ secrets.CLAUDE_CODE_OAUTH_TOKEN }}",
                "prompt": (
                    "Please review this pull request and provide feedback on:\n"
                    "- Code quality and best practices\n"
                    "- Potential bugs or issues\n"
                    "- Performance considerations\n"
                    "- Security concerns\n"
                    "- Test coverage\n\n"
                    "Use the repository's CLAUDE.md for guidance on style and conventions. "
                    "Be constructive and helpful in your feedback.\n\n"
                    "Use `gh pr comment` with your Bash tool to leave your review as a comment on the PR."
                ),
                "claude_args": (
                    '--allowed-tools "Bash(gh issue view:*),Bash(gh search:*),Bash(gh issue list:*),'
                    'Bash(gh pr comment:*),Bash(gh pr diff:*),Bash(gh pr view:*),Bash(gh pr list:*)"'
                ),
            },
        )

        # Define the job
        claude_review_job = WorkflowJob(
            runs_on="ubuntu-latest",
            permissions=permissions,
            steps=[checkout_step, claude_review_step],
        )

        # Define workflow triggers
        workflow_triggers = {"pull_request": {"types": ["opened", "synchronize"]}}

        # Create the complete workflow
        workflow = WorkflowConfig(
            name="Claude Code Review",
            on=workflow_triggers,
            jobs={"claude-review": claude_review_job},
        )

        return workflow

    def validate_workflow(self, workflow: WorkflowConfig) -> bool:
        """Validate a workflow configuration."""
        try:
            # The Pydantic model validation will catch most issues
            workflow.model_validate(workflow.model_dump())

            # Additional custom validation
            if not workflow.jobs:
                raise ValueError("Workflow must have at least one job")

            for job_id, job in workflow.jobs.items():
                if not job.steps:
                    raise ValueError(f"Job '{job_id}' must have at least one step")

                # Validate that each step has either 'uses' or 'run'
                for i, step in enumerate(job.steps):
                    if not step.uses and not step.run:
                        raise ValueError(
                            f"Step {i} in job '{job_id}' must have either 'uses' or 'run'"
                        )
                    if step.uses and step.run:
                        raise ValueError(
                            f"Step {i} in job '{job_id}' cannot have both 'uses' and 'run'"
                        )

            return True

        except Exception as e:
            print(f"Workflow validation failed: {e}")
            raise

    def generate_yaml(self, workflow: WorkflowConfig) -> str:
        """Generate YAML content from workflow configuration."""
        # Convert to dictionary with proper serialization
        workflow_dict = self._serialize_workflow(workflow)

        # Generate YAML with proper formatting
        yaml_content = yaml.dump(
            workflow_dict,
            default_flow_style=False,
            sort_keys=False,
            indent=2,
            width=120,
            allow_unicode=True,
        )

        # Post-process YAML to fix formatting
        yaml_content = self._clean_yaml_output(yaml_content)

        return yaml_content

    def _serialize_workflow(self, workflow: WorkflowConfig) -> dict[str, Any]:
        """Serialize workflow to dictionary with proper type handling."""
        # Get the base dictionary
        workflow_dict = workflow.model_dump(by_alias=True, exclude_none=True)

        # Fix permissions serialization
        for job_name, job_data in workflow_dict.get("jobs", {}).items():
            if "permissions" in job_data and isinstance(job_data["permissions"], dict):
                permissions = job_data["permissions"]
                fixed_permissions = {}
                for key, value in permissions.items():
                    if hasattr(value, "value"):
                        fixed_permissions[key] = value.value
                    elif value is not None:
                        fixed_permissions[key] = str(value)
                job_data["permissions"] = fixed_permissions

        return workflow_dict

    def _clean_yaml_output(self, yaml_content: str) -> str:
        """Clean up YAML output formatting."""
        # Fix quoted 'on' key
        yaml_content = yaml_content.replace("'on':", "on:")

        # Handle multiline if conditions
        lines = yaml_content.split("\n")
        processed_lines = []

        for line in lines:
            # Convert single-line if with || to multiline format
            if "if:" in line and "||" in line:
                indent = len(line) - len(line.lstrip())
                # Use literal block scalar for multiline if
                processed_lines.append(f"{' ' * indent}if: |")
                if_content = line.split("if:", 1)[1].strip()
                # Remove quotes and split by ||
                if_content = if_content.strip("'\"")
                conditions = [cond.strip() for cond in if_content.split("||")]

                for i, condition in enumerate(conditions):
                    if i == len(conditions) - 1:
                        processed_lines.append(f"{' ' * (indent + 2)}{condition}")
                    else:
                        processed_lines.append(f"{' ' * (indent + 2)}{condition} ||")
            else:
                processed_lines.append(line)

        return "\n".join(processed_lines)

    def write_workflow_file(self, workflow: WorkflowConfig, filename: str) -> Path:
        """Write workflow to a YAML file."""
        # Validate before writing
        self.validate_workflow(workflow)

        # Ensure workflows directory exists
        self.workflows_dir.mkdir(parents=True, exist_ok=True)

        # Generate YAML content
        yaml_content = self.generate_yaml(workflow)

        # Write to file
        file_path = self.workflows_dir / f"{filename}.yml"
        file_path.write_text(yaml_content, encoding="utf-8")

        print(f"Generated workflow: {file_path}")
        return file_path

    def generate_all_workflows(self) -> list[Path]:
        """Generate all Claude workflows."""
        generated_files = []

        # Generate main Claude workflow
        claude_workflow = self.create_claude_workflow()
        claude_file = self.write_workflow_file(claude_workflow, "claude")
        generated_files.append(claude_file)

        # Generate Claude Code Review workflow
        review_workflow = self.create_claude_code_review_workflow()
        review_file = self.write_workflow_file(review_workflow, "claude-code-review")
        generated_files.append(review_file)

        return generated_files


def main():
    """Main entry point for the workflow generator."""
    generator = ClaudeWorkflowGenerator()

    try:
        print("Generating Claude workflows using Pydantic models...")
        generated_files = generator.generate_all_workflows()

        print("\nSuccessfully generated workflows:")
        for file_path in generated_files:
            print(f"  - {file_path.relative_to(generator.project_root)}")

        print("\nAll workflows have been validated and generated successfully!")

    except Exception as e:
        print(f"Error generating workflows: {e}")
        raise


if __name__ == "__main__":
    main()
