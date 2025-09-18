# Development Scripts

This directory contains development and debugging utilities for the ONEX Infrastructure project.

## Directory Structure

- `dev/` - Developer tools and debugging scripts

## Scripts in dev/

### simple_slack_test.py
**Purpose**: Tests basic HTTP connectivity to Slack webhooks before testing the full Hook Node.

**Usage**:
```bash
export SLACK_WEBHOOK_URL='https://hooks.slack.com/services/...your-url...'
cd scripts/dev
python simple_slack_test.py
```

**Requirements**:
- `aiohttp` package
- Valid Slack webhook URL in environment variable

**What it does**:
- Validates Slack webhook connectivity
- Sends test message to verify webhook configuration
- Useful for debugging webhook issues before Hook Node testing

## Environment Variables

- `SLACK_WEBHOOK_URL` - Your Slack incoming webhook URL (required for Slack tests)

## Security Note

These scripts are for development use only and should never be included in production builds. The packaging configuration excludes the entire `scripts/` directory from distribution.
