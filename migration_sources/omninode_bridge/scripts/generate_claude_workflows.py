#!/usr/bin/env python3
"""Generate Claude workflows using Pydantic CI system."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from omninode_bridge.ci import WorkflowConfig, WorkflowJob, WorkflowStep


def create_claude_code_review_workflow():
    """Create the claude-code-review.yml workflow using Pydantic models."""

    # Create steps
    checkout_step = WorkflowStep(
        name="Checkout repository", uses="actions/checkout@v4", with_={"fetch-depth": 1}
    )

    claude_step = WorkflowStep(
        name="Run Claude Code Review",
        id="claude-review",
        uses="anthropics/claude-code-action@v1",
        with_={
            "claude_code_oauth_token": "${{ secrets.CLAUDE_CODE_OAUTH_TOKEN }}",
            "prompt": """Please review this pull request and provide feedback on:
- Code quality and best practices
- Potential bugs or issues
- Performance considerations
- Security concerns
- Test coverage

Use the repository's CLAUDE.md for guidance on style and conventions. Be constructive and helpful in your feedback.

Use `gh pr comment` with your Bash tool to leave your review as a comment on the PR.""",
            "claude_args": '--allowed-tools "Bash(gh issue view:*),Bash(gh search:*),Bash(gh issue list:*),Bash(gh pr comment:*),Bash(gh pr diff:*),Bash(gh pr view:*),Bash(gh pr list:*)"',
        },
    )

    # Create permissions - use string values directly instead of enums
    permissions = {
        "contents": "read",
        "pull-requests": "read",
        "issues": "read",
        "id-token": "write",
    }

    # Create job
    claude_review_job = WorkflowJob(
        runs_on="ubuntu-latest",
        permissions=permissions,
        steps=[checkout_step, claude_step],
    )

    # Create workflow
    workflow = WorkflowConfig(
        name="Claude Code Review",
        on={"pull_request": {"types": ["opened", "synchronize"]}},
        jobs={"claude-review": claude_review_job},
    )

    return workflow


def create_claude_workflow():
    """Create the claude.yml workflow using Pydantic models."""

    # Create steps
    checkout_step = WorkflowStep(
        name="Checkout repository", uses="actions/checkout@v4", with_={"fetch-depth": 1}
    )

    claude_step = WorkflowStep(
        name="Run Claude Code",
        id="claude",
        uses="anthropics/claude-code-action@v1",
        with_={
            "claude_code_oauth_token": "${{ secrets.CLAUDE_CODE_OAUTH_TOKEN }}",
            "additional_permissions": "actions: read",
        },
    )

    # Create permissions - use string values directly instead of enums
    permissions = {
        "contents": "read",
        "pull-requests": "read",
        "issues": "read",
        "id-token": "write",
        "actions": "read",
    }

    # Create job with complex conditional
    claude_job = WorkflowJob(
        runs_on="ubuntu-latest",
        permissions=permissions,
        if_="""(github.event_name == 'issue_comment' && contains(github.event.comment.body, '@claude')) ||
(github.event_name == 'pull_request_review_comment' && contains(github.event.comment.body, '@claude')) ||
(github.event_name == 'pull_request_review' && contains(github.event.review.body, '@claude')) ||
(github.event_name == 'issues' && (contains(github.event.issue.body, '@claude') || contains(github.event.issue.title, '@claude')))""",
        steps=[checkout_step, claude_step],
    )

    # Create workflow
    workflow = WorkflowConfig(
        name="Claude Code",
        on={
            "issue_comment": {"types": ["created"]},
            "pull_request_review_comment": {"types": ["created"]},
            "issues": {"types": ["opened", "assigned"]},
            "pull_request_review": {"types": ["submitted"]},
        },
        jobs={"claude": claude_job},
    )

    return workflow


def main():
    """Generate both Claude workflows."""
    print("Generating Claude workflows using Pydantic CI system...")

    # Generate claude-code-review.yml
    claude_review_workflow = create_claude_code_review_workflow()
    claude_review_workflow.to_yaml_file(".github/workflows/claude-code-review.yml")
    print("✅ Generated .github/workflows/claude-code-review.yml")

    # Generate claude.yml
    claude_workflow = create_claude_workflow()
    claude_workflow.to_yaml_file(".github/workflows/claude.yml")
    print("✅ Generated .github/workflows/claude.yml")

    print("✅ Claude workflows generated successfully using Pydantic models!")


if __name__ == "__main__":
    main()
