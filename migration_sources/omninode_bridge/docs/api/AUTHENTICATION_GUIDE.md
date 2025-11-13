# OmniNode Bridge API Authentication Guide

## Overview

This comprehensive guide covers all authentication methods, security best practices, and implementation examples for the OmniNode Bridge API.

## Table of Contents

1. [Authentication Methods](#authentication-methods)
2. [API Key Management](#api-key-management)
3. [Implementation Examples](#implementation-examples)
4. [Security Best Practices](#security-best-practices)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)

## Authentication Methods

### 1. API Key Authentication

OmniNode Bridge supports two methods for API key authentication:

#### Bearer Token (Recommended)
```http
Authorization: Bearer omninode-bridge-a1b2c3d4e5f6789-2024
```

#### Custom Header
```http
X-API-Key: omninode-bridge-a1b2c3d4e5f6789-2024
```

### 2. Authentication Endpoints

All API endpoints require authentication except:
- `/health` - Health check endpoint
- `/metrics` - Prometheus metrics (restricted by IP)
- `/docs` - API documentation

## API Key Management

### Obtaining API Keys

#### Production Environment
Contact your system administrator or use the management portal:

```bash
# Via CLI (if available)
omninode-cli auth create-key --name "my-application" --permissions "read,write"

# Via API (admin only)
curl -X POST https://api.omninode.ai/admin/api-keys \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-application",
    "permissions": ["hooks:write", "metrics:read", "workflows:execute"],
    "expires_at": "2024-12-31T23:59:59Z"
  }'
```

#### Development Environment
```bash
# Generate development key
docker-compose exec hook-receiver python -c "
import secrets
import datetime
key = f'omninode-bridge-dev-{secrets.token_hex(16)}-{datetime.datetime.now().year}'
print(f'Development API Key: {key}')
"
```

### Key Format and Structure

```
omninode-bridge-{environment}-{random-hex}-{year}

Examples:
- omninode-bridge-prod-a1b2c3d4e5f6789-2024    (Production)
- omninode-bridge-dev-1a2b3c4d5e6f789-2024     (Development)
- omninode-bridge-test-9z8y7x6w5v4u321-2024    (Testing)
```

### Key Permissions

API keys can have granular permissions:

```yaml
Permissions:
  hooks:read      # Read hook data and history
  hooks:write     # Submit new hooks
  hooks:delete    # Delete hook data

  metrics:read    # Read model performance metrics
  metrics:write   # Submit performance data

  workflows:read  # Read workflow status
  workflows:write # Create and modify workflows
  workflows:execute # Execute workflows

  admin:read      # Administrative read access
  admin:write     # Administrative write access
```

## Implementation Examples

### 1. Python Implementation

#### Using requests library
```python
import requests
import json
from typing import Dict, Any, Optional

class OmniNodeBridgeClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'OmniNodeBridge-Python-Client/1.0'
        })

    def submit_hook(self, hook_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit a hook event to OmniNode Bridge.

        Args:
            hook_data: Hook event data

        Returns:
            Response from the API

        Raises:
            requests.exceptions.HTTPError: For HTTP errors
            requests.exceptions.RequestException: For request errors
        """
        url = f"{self.base_url}/hooks"

        try:
            response = self.session.post(url, json=hook_data, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise AuthenticationError("Invalid API key")
            elif e.response.status_code == 429:
                raise RateLimitError("Rate limit exceeded")
            else:
                raise APIError(f"HTTP {e.response.status_code}: {e.response.text}")
        except requests.exceptions.RequestException as e:
            raise APIError(f"Request failed: {str(e)}")

    def get_model_health(self) -> Dict[str, Any]:
        """Check AI lab health status."""
        url = f"{self.base_url}/lab/health"
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        return response.json()

    def execute_task(self, task_type: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Execute an AI task with intelligent model selection.

        Args:
            task_type: Type of task (code_generation, debugging, etc.)
            prompt: Task prompt
            **kwargs: Additional task parameters

        Returns:
            Task execution result
        """
        url = f"{self.base_url}/execute"

        task_data = {
            'task_type': task_type,
            'prompt': prompt,
            **kwargs
        }

        response = self.session.post(url, json=task_data, timeout=60)
        response.raise_for_status()
        return response.json()

# Custom exceptions
class APIError(Exception):
    pass

class AuthenticationError(APIError):
    pass

class RateLimitError(APIError):
    pass

# Usage example
if __name__ == "__main__":
    client = OmniNodeBridgeClient(
        base_url="https://api.omninode.ai",
        api_key="omninode-bridge-prod-a1b2c3d4e5f6789-2024"  # pragma: allowlist secret
    )

    # Submit a hook
    hook_result = client.submit_hook({
        "source": "my-application",
        "action": "user_signup",
        "resource": "user",
        "resource_id": "user-12345",
        "data": {
            "email": "user@example.com",
            "plan": "premium"
        }
    })
    print(f"Hook submitted: {hook_result['event_id']}")

    # Execute an AI task
    task_result = client.execute_task(
        task_type="code_generation",
        prompt="Create a Python function to validate email addresses",
        complexity="moderate"
    )
    print(f"Generated code: {task_result['response']}")
```

#### Using aiohttp (async)
```python
import aiohttp
import asyncio
from typing import Dict, Any

class AsyncOmniNodeBridgeClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'OmniNodeBridge-AsyncPython-Client/1.0'
        }

    async def submit_hook(self, hook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Async hook submission."""
        url = f"{self.base_url}/hooks"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=hook_data,
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 401:
                    raise AuthenticationError("Invalid API key")
                elif response.status == 429:
                    raise RateLimitError("Rate limit exceeded")
                elif response.status >= 400:
                    text = await response.text()
                    raise APIError(f"HTTP {response.status}: {text}")

                return await response.json()

# Usage example
async def main():
    client = AsyncOmniNodeBridgeClient(
        base_url="https://api.omninode.ai",
        api_key="omninode-bridge-prod-a1b2c3d4e5f6789-2024"  # pragma: allowlist secret
    )

    result = await client.submit_hook({
        "source": "async-app",
        "action": "data_processed",
        "resource": "dataset",
        "resource_id": "dataset-789"
    })
    print(f"Async hook result: {result}")

# Run async example
# asyncio.run(main())
```

### 2. JavaScript/Node.js Implementation

#### Using fetch API
```javascript
class OmniNodeBridgeClient {
    constructor(baseUrl, apiKey) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.apiKey = apiKey;
        this.defaultHeaders = {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json',
            'User-Agent': 'OmniNodeBridge-JS-Client/1.0'
        };
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const config = {
            headers: { ...this.defaultHeaders, ...options.headers },
            ...options
        };

        try {
            const response = await fetch(url, config);

            if (response.status === 401) {
                throw new AuthenticationError('Invalid API key');
            } else if (response.status === 429) {
                throw new RateLimitError('Rate limit exceeded');
            } else if (!response.ok) {
                const errorText = await response.text();
                throw new APIError(`HTTP ${response.status}: ${errorText}`);
            }

            return await response.json();
        } catch (error) {
            if (error instanceof TypeError && error.message.includes('fetch')) {
                throw new APIError(`Network error: ${error.message}`);
            }
            throw error;
        }
    }

    async submitHook(hookData) {
        return this.request('/hooks', {
            method: 'POST',
            body: JSON.stringify(hookData)
        });
    }

    async getModelHealth() {
        return this.request('/lab/health');
    }

    async executeTask(taskType, prompt, options = {}) {
        const taskData = {
            task_type: taskType,
            prompt: prompt,
            ...options
        };

        return this.request('/execute', {
            method: 'POST',
            body: JSON.stringify(taskData)
        });
    }
}

// Custom error classes
class APIError extends Error {
    constructor(message) {
        super(message);
        this.name = 'APIError';
    }
}

class AuthenticationError extends APIError {
    constructor(message) {
        super(message);
        this.name = 'AuthenticationError';
    }
}

class RateLimitError extends APIError {
    constructor(message) {
        super(message);
        this.name = 'RateLimitError';
    }
}

// Usage example
async function example() {
    const client = new OmniNodeBridgeClient(
        'https://api.omninode.ai',
        'omninode-bridge-prod-a1b2c3d4e5f6789-2024'
    );

    try {
        // Submit a hook
        const hookResult = await client.submitHook({
            source: 'web-app',
            action: 'page_view',
            resource: 'page',
            resource_id: '/dashboard',
            data: {
                user_id: 'user-456',
                session_id: 'session-789'
            }
        });
        console.log('Hook submitted:', hookResult.event_id);

        // Execute AI task
        const taskResult = await client.executeTask(
            'code_generation',
            'Create a JavaScript function to debounce user input',
            { complexity: 'simple' }
        );
        console.log('Generated code:', taskResult.response);

    } catch (error) {
        if (error instanceof AuthenticationError) {
            console.error('Authentication failed:', error.message);
        } else if (error instanceof RateLimitError) {
            console.error('Rate limit exceeded, retry later');
        } else {
            console.error('API error:', error.message);
        }
    }
}
```

#### Using axios
```javascript
const axios = require('axios');

class OmniNodeBridgeAxiosClient {
    constructor(baseUrl, apiKey) {
        this.client = axios.create({
            baseURL: baseUrl.replace(/\/$/, ''),
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json',
                'User-Agent': 'OmniNodeBridge-Axios-Client/1.0'
            },
            timeout: 30000
        });

        // Add response interceptor for error handling
        this.client.interceptors.response.use(
            response => response,
            error => {
                if (error.response) {
                    const { status, data } = error.response;
                    if (status === 401) {
                        throw new AuthenticationError('Invalid API key');
                    } else if (status === 429) {
                        throw new RateLimitError('Rate limit exceeded');
                    } else {
                        throw new APIError(`HTTP ${status}: ${data.message || data}`);
                    }
                } else if (error.request) {
                    throw new APIError('Network error: No response received');
                } else {
                    throw new APIError(`Request error: ${error.message}`);
                }
            }
        );
    }

    async submitHook(hookData) {
        const response = await this.client.post('/hooks', hookData);
        return response.data;
    }

    async getModelHealth() {
        const response = await this.client.get('/lab/health');
        return response.data;
    }

    async executeTask(taskType, prompt, options = {}) {
        const taskData = {
            task_type: taskType,
            prompt: prompt,
            ...options
        };
        const response = await this.client.post('/execute', taskData);
        return response.data;
    }
}
```

### 3. curl Examples

#### Basic Hook Submission
```bash
#!/bin/bash

API_KEY="omninode-bridge-prod-a1b2c3d4e5f6789-2024"  # pragma: allowlist secret
BASE_URL="https://api.omninode.ai"

# Submit a hook using Bearer token
curl -X POST "${BASE_URL}/hooks" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -H "User-Agent: curl-script/1.0" \
  -d '{
    "source": "deployment-script",
    "action": "deploy",
    "resource": "application",
    "resource_id": "omninode-bridge-v1.2.3",
    "data": {
      "version": "1.2.3",
      "environment": "production",
      "deployed_by": "ci-cd-pipeline"
    }
  }' \
  --max-time 30 \
  --retry 3 \
  --retry-delay 5

# Alternative using X-API-Key header
curl -X POST "${BASE_URL}/hooks" \
  -H "X-API-Key: ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "monitoring-script",
    "action": "alert",
    "resource": "service",
    "resource_id": "database",
    "data": {
      "alert_type": "high_cpu",
      "value": 85.2,
      "threshold": 80
    }
  }'
```

#### Health Check Script
```bash
#!/bin/bash

API_KEY="omninode-bridge-prod-a1b2c3d4e5f6789-2024"  # pragma: allowlist secret
BASE_URL="https://api.omninode.ai"

# Function to check service health
check_health() {
    local service=$1
    local endpoint=$2

    echo "Checking ${service} health..."

    response=$(curl -s -w "%{http_code}" \
      -H "Authorization: Bearer ${API_KEY}" \
      "${BASE_URL}${endpoint}")

    http_code="${response: -3}"
    body="${response%???}"

    if [ "$http_code" = "200" ]; then
        echo "✓ ${service} is healthy"
        echo "  Response: ${body}" | jq .
    else
        echo "✗ ${service} health check failed (HTTP ${http_code})"
        echo "  Response: ${body}"
    fi
    echo
}

# Check all services
check_health "HookReceiver" "/health"
check_health "ModelMetrics" "/lab/health"
check_health "WorkflowCoordinator" "/workflows/health"
```

### 4. Go Implementation

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "time"
)

type OmniNodeBridgeClient struct {
    BaseURL    string
    APIKey     string
    HTTPClient *http.Client
}

type HookData struct {
    Source     string                 `json:"source"`
    Action     string                 `json:"action"`
    Resource   string                 `json:"resource"`
    ResourceID string                 `json:"resource_id"`
    Data       map[string]interface{} `json:"data,omitempty"`
}

type HookResponse struct {
    Success          bool    `json:"success"`
    Message          string  `json:"message"`
    EventID          string  `json:"event_id"`
    ProcessingTimeMs float64 `json:"processing_time_ms"`
}

func NewOmniNodeBridgeClient(baseURL, apiKey string) *OmniNodeBridgeClient {
    return &OmniNodeBridgeClient{
        BaseURL: baseURL,
        APIKey:  apiKey,
        HTTPClient: &http.Client{
            Timeout: 30 * time.Second,
        },
    }
}

func (c *OmniNodeBridgeClient) makeRequest(method, endpoint string, body interface{}) (*http.Response, error) {
    var reqBody io.Reader

    if body != nil {
        jsonData, err := json.Marshal(body)
        if err != nil {
            return nil, fmt.Errorf("failed to marshal request body: %w", err)
        }
        reqBody = bytes.NewBuffer(jsonData)
    }

    req, err := http.NewRequest(method, c.BaseURL+endpoint, reqBody)
    if err != nil {
        return nil, fmt.Errorf("failed to create request: %w", err)
    }

    req.Header.Set("Authorization", "Bearer "+c.APIKey)
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("User-Agent", "OmniNodeBridge-Go-Client/1.0")

    return c.HTTPClient.Do(req)
}

func (c *OmniNodeBridgeClient) SubmitHook(hookData HookData) (*HookResponse, error) {
    resp, err := c.makeRequest("POST", "/hooks", hookData)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    if resp.StatusCode == 401 {
        return nil, fmt.Errorf("authentication failed: invalid API key")
    } else if resp.StatusCode == 429 {
        return nil, fmt.Errorf("rate limit exceeded")
    } else if resp.StatusCode >= 400 {
        body, _ := io.ReadAll(resp.Body)
        return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
    }

    var hookResponse HookResponse
    if err := json.NewDecoder(resp.Body).Decode(&hookResponse); err != nil {
        return nil, fmt.Errorf("failed to decode response: %w", err)
    }

    return &hookResponse, nil
}

// Usage example
func main() {
    client := NewOmniNodeBridgeClient(
        "https://api.omninode.ai",
        "omninode-bridge-prod-a1b2c3d4e5f6789-2024",
    )

    hookData := HookData{
        Source:     "go-application",
        Action:     "process_completed",
        Resource:   "job",
        ResourceID: "job-12345",
        Data: map[string]interface{}{
            "duration_ms": 5432,
            "status":      "success",
            "records":     1000,
        },
    }

    response, err := client.SubmitHook(hookData)
    if err != nil {
        fmt.Printf("Error submitting hook: %v\n", err)
        return
    }

    fmt.Printf("Hook submitted successfully: %s\n", response.EventID)
    fmt.Printf("Processing time: %.2fms\n", response.ProcessingTimeMs)
}
```

## Security Best Practices

### 1. API Key Security

#### Storage
```bash
# ✅ Good: Environment variables
export OMNINODE_API_KEY="omninode-bridge-prod-a1b2c3d4e5f6789-2024"  # pragma: allowlist secret

# ✅ Good: Secure configuration files (with proper permissions)
chmod 600 /etc/omninode/config.json

# ❌ Bad: Hardcoded in source code
api_key = "omninode-bridge-prod-a1b2c3d4e5f6789-2024"  # Never do this!  # pragma: allowlist secret

# ❌ Bad: In version control
git add config.py  # containing API keys
```

#### Rotation
```python
# API key rotation example
class RotatingAPIKeyClient:
    def __init__(self, primary_key: str, backup_key: str = None):
        self.primary_key = primary_key
        self.backup_key = backup_key
        self.current_key = primary_key

    def make_request(self, endpoint: str, data: dict):
        try:
            return self._request_with_key(self.current_key, endpoint, data)
        except AuthenticationError:
            if self.backup_key:
                print("Primary key failed, trying backup key...")
                self.current_key = self.backup_key
                return self._request_with_key(self.current_key, endpoint, data)
            raise
```

### 2. Network Security

#### TLS/SSL Configuration
```python
import ssl
import certifi
import requests

# ✅ Good: Proper SSL verification
session = requests.Session()
session.verify = certifi.where()  # Use system CA bundle

# ✅ Good: Custom CA bundle if needed
session.verify = '/path/to/custom-ca-bundle.pem'

# ❌ Bad: Disabling SSL verification
session.verify = False  # Never do this in production!
```

#### Request Timeouts
```python
# ✅ Good: Appropriate timeouts
response = requests.post(
    url,
    json=data,
    timeout=(5, 30)  # (connect_timeout, read_timeout)
)

# ❌ Bad: No timeout (can hang indefinitely)
response = requests.post(url, json=data)  # No timeout!
```

### 3. Error Handling

#### Secure Error Messages
```python
def handle_api_error(response):
    """Handle API errors without exposing sensitive information."""
    if response.status_code == 401:
        # Don't expose the actual API key in error messages
        raise AuthenticationError("Invalid authentication credentials")
    elif response.status_code == 403:
        raise PermissionError("Insufficient permissions for this operation")
    elif response.status_code == 429:
        retry_after = response.headers.get('Retry-After', '60')
        raise RateLimitError(f"Rate limit exceeded. Retry after {retry_after} seconds")
    else:
        # Log detailed error server-side, return generic message to client
        logger.error(f"API error: {response.status_code} - {response.text}")
        raise APIError("An error occurred while processing your request")
```

## Error Handling

### Common Error Responses

#### 401 Unauthorized
```json
{
  "error": "INVALID_API_KEY",
  "message": "Invalid or missing API key. Provide via Authorization: Bearer <key> or X-API-Key header",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

#### 403 Forbidden
```json
{
  "error": "INSUFFICIENT_PERMISSIONS",
  "message": "API key does not have permission for this operation",
  "required_permissions": ["hooks:write"],
  "current_permissions": ["hooks:read"],
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

#### 429 Rate Limited
```json
{
  "error": "RATE_LIMIT_EXCEEDED",
  "message": "Rate limit exceeded for this API key",
  "limit": 100,
  "window": "1 minute",
  "retry_after": 60,
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

### Error Handling Patterns

#### Exponential Backoff
```python
import time
import random

def exponential_backoff_retry(func, max_retries=3, base_delay=1):
    """Retry function with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return func()
        except RateLimitError as e:
            if attempt == max_retries:
                raise

            # Extract retry_after from error if available
            retry_after = getattr(e, 'retry_after', None)
            if retry_after:
                delay = float(retry_after)
            else:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)

            print(f"Rate limited, retrying in {delay:.2f} seconds...")
            time.sleep(delay)
        except (APIError, AuthenticationError):
            # Don't retry on authentication or other API errors
            raise
```

## Rate Limiting

### Rate Limit Information

| Endpoint Category | Rate Limit | Window | Burst Limit |
|------------------|------------|--------|-------------|
| Hook Processing | 100 req/min | 1 minute | 150 |
| Model Execution | 20 req/min | 1 minute | 30 |
| Model Comparison | 5 req/min | 1 minute | 8 |
| Health Checks | No limit | - | - |
| Metrics | No limit | - | - |

### Rate Limit Headers

All API responses include rate limiting headers:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1640995200
X-RateLimit-Window: 60
```

### Handling Rate Limits

#### Python Example
```python
def check_rate_limit(response):
    """Check rate limit headers and handle accordingly."""
    remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
    reset_time = int(response.headers.get('X-RateLimit-Reset', 0))

    if remaining < 10:  # Warning threshold
        reset_in = reset_time - time.time()
        print(f"Rate limit warning: {remaining} requests remaining, resets in {reset_in:.0f}s")

    if response.status_code == 429:
        retry_after = int(response.headers.get('Retry-After', 60))
        raise RateLimitError(f"Rate limit exceeded, retry after {retry_after} seconds")
```

This comprehensive authentication guide provides everything needed to securely integrate with the OmniNode Bridge API.
