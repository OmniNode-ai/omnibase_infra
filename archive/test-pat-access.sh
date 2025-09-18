#!/bin/bash
# Test script to validate OMNI_CI_PAT configuration
# Usage: ./test-pat-access.sh YOUR_NEW_PAT_TOKEN

if [ -z "$1" ]; then
    echo "Usage: ./test-pat-access.sh YOUR_NEW_PAT_TOKEN"
    exit 1
fi

PAT_TOKEN="$1"
echo "=== Testing PAT Access to Required Repositories ==="

echo "Testing omnibase_spi access..."
RESPONSE=$(curl -H "Authorization: token $PAT_TOKEN" \
    https://api.github.com/repos/OmniNode-ai/omnibase_spi 2>/dev/null)
if echo "$RESPONSE" | jq -e '.name' > /dev/null 2>&1; then
    echo "✅ omnibase_spi: Access granted"
else
    echo "❌ omnibase_spi: Access denied"
    echo "Response: $(echo "$RESPONSE" | jq -r '.message // "Unknown error"')"
fi

echo "Testing omnibase_core access..."
RESPONSE=$(curl -H "Authorization: token $PAT_TOKEN" \
    https://api.github.com/repos/OmniNode-ai/omnibase_core 2>/dev/null)
if echo "$RESPONSE" | jq -e '.name' > /dev/null 2>&1; then
    echo "✅ omnibase_core: Access granted"
else
    echo "❌ omnibase_core: Access denied"
    echo "Response: $(echo "$RESPONSE" | jq -r '.message // "Unknown error"')"
fi

echo "Testing Git clone simulation..."
git config --global url."https://$PAT_TOKEN@github.com/".insteadOf "https://github.com/"
if git ls-remote https://github.com/OmniNode-ai/omnibase_spi.git HEAD > /dev/null 2>&1; then
    echo "✅ Git clone simulation: Success"
else
    echo "❌ Git clone simulation: Failed"
fi

echo "=== Test Complete ==="
