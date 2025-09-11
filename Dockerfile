FROM python:3.12-slim

# GitHub token will be provided via Docker secret

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Configure git with GitHub token for private repos (using mount to avoid layer exposure)
RUN --mount=type=secret,id=github_token \
    GITHUB_TOKEN=$(cat /run/secrets/github_token) && \
    git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"

# Install Poetry
RUN pip install poetry

# Copy poetry files and README
COPY pyproject.toml poetry.lock README.md ./

# Copy source code first
COPY src/ ./src/

# Configure poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --only main

# Set Python path
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 8080

# Run the PostgreSQL adapter
CMD ["python", "-m", "omnibase_infra.nodes.node_postgres_adapter_effect.v1_0_0.node"]