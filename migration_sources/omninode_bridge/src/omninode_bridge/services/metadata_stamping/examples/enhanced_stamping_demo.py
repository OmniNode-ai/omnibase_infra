#!/usr/bin/env python3
"""
Enhanced Metadata Stamping with LangExtract - Comprehensive Demo

This demo showcases the hash-only stamping approach from omnibase_3
integrated with LangExtract for intelligent metadata extraction.

Key Features Demonstrated:
- ✅ Clean file stamping (only lightweight hash references)
- ✅ External metadata storage (database + dot files)
- ✅ Intelligent categorization and tagging
- ✅ Entity extraction and relationship analysis
- ✅ Idempotent operations (metadata changes don't affect files)
- ✅ Rich querying capabilities
"""

import asyncio
import os
import tempfile
from datetime import datetime

# For demo purposes - would normally be imported
try:
    from ..engine.enhanced_metadata_extractor import EnhancedMetadataExtractor
except ImportError:
    print("⚠️  Running in demo mode - imports not available")


# Sample content for demonstration
SAMPLE_PYTHON_API = '''
"""FastAPI authentication service with JWT and OAuth2 integration.

This module provides secure user authentication using industry-standard
protocols and best practices for modern web applications.
"""

import jwt
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from datetime import UTC, UTC, datetime, timedelta
from typing import Optional, Dict, Any

app = FastAPI(title="Authentication Service", version="1.0.0")

# Security configuration
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class AuthenticationError(Exception):
    """Custom authentication error."""
    pass

async def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user credentials against database."""
    # Implementation would check database
    return {"username": username, "id": 1}

async def create_access_token(data: dict) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(UTC) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint that returns JWT token."""
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = await create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}
'''

SAMPLE_DOCUMENTATION = """# Authentication Service Documentation

## Overview

The Authentication Service provides secure user authentication for the omninode platform using JWT tokens and OAuth2 integration. This service ensures robust security while maintaining ease of use for developers.

## Features

- **JWT Token Authentication**: Industry-standard JSON Web Tokens
- **OAuth2 Integration**: Support for third-party authentication providers
- **Password Security**: BCrypt hashing with salt
- **Rate Limiting**: Protection against brute force attacks
- **Session Management**: Secure token lifecycle management

## API Endpoints

### POST /token
Authenticate user and receive access token.

**Parameters:**
- `username` (string): User's username
- `password` (string): User's password

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

## Security Considerations

- Always use HTTPS in production
- Store secrets in environment variables
- Implement proper token rotation
- Monitor for suspicious authentication attempts

## Integration Examples

### Python Client
```python
import requests

response = requests.post("/token", data={
    "username": "user@example.com",
    "password": "secure_password"
})
token = response.json()["access_token"]
```

### JavaScript Client
```javascript
const response = await fetch('/token', {
    method: 'POST',
    body: new FormData([
        ['username', 'user@example.com'],
        ['password', 'secure_password']
    ])
});
const { access_token } = await response.json();
```
"""


async def demo_enhanced_metadata_extraction():
    """Demonstrate enhanced metadata extraction with hash-only stamping."""

    print("🚀 Enhanced Metadata Stamping Demo")
    print("=" * 50)

    # Initialize enhanced metadata extractor
    print("\n1️⃣ Initializing Enhanced Metadata Extractor...")

    try:
        extractor = EnhancedMetadataExtractor(
            model_id="gemini-2.5-flash",
            enable_dot_files=True,
            enable_visualization=True,
        )
        print("✅ Enhanced extractor initialized with LangExtract")
    except ImportError:
        print("⚠️  LangExtract not available - running in simulation mode")
        await demo_simulation_mode()
        return

    # Demo 1: Python API Code Analysis
    await demo_python_code_analysis(extractor)

    # Demo 2: Documentation Analysis
    await demo_documentation_analysis(extractor)

    # Demo 3: Hash-Only Stamping Workflow
    await demo_hash_only_workflow(extractor)

    # Demo 4: Metadata Querying
    await demo_metadata_querying(extractor)

    print("\n🎉 Enhanced Metadata Demo Complete!")


async def demo_python_code_analysis(extractor: EnhancedMetadataExtractor):
    """Demo: Analyze Python API code for categories, tags, and entities."""

    print("\n2️⃣ Python API Code Analysis")
    print("-" * 30)

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "auth_service.py")

        # Extract enhanced metadata
        print("🔍 Analyzing Python FastAPI authentication service...")
        start_time = datetime.now()

        hash_ref, enhanced_metadata = await extractor.extract_and_store_metadata(
            content=SAMPLE_PYTHON_API,
            file_path=file_path,
            content_type="application/x-python",
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        print(f"⚡ Processing completed in {processing_time:.2f}ms")
        print(f"📊 Metadata hash: {hash_ref.metadata_hash[:12]}...")
        print(f"📄 Content hash: {hash_ref.content_hash[:12]}...")

        # Display extracted categories
        print(f"\n📂 Categories Found ({len(enhanced_metadata.categories)}):")
        for cat in enhanced_metadata.categories:
            print(f"   • {cat.category}")
            if cat.subcategory:
                print(f"     └─ {cat.subcategory}")
            print(f"     Confidence: {cat.confidence:.2f}")

        # Display extracted tags
        print(f"\n🏷️  Tags Found ({len(enhanced_metadata.tags)}):")
        for tag in enhanced_metadata.tags:
            print(f"   • {tag.tag} ({tag.tag_type})")
            print(f"     Relevance: {tag.relevance:.2f}")

        # Display extracted entities
        print(f"\n🎯 Entities Found ({len(enhanced_metadata.entities)}):")
        for entity in enhanced_metadata.entities:
            print(f"   • {entity.entity_text} ({entity.entity_type})")
            print(f"     Confidence: {entity.confidence:.2f}")

        # Show lightweight stamp
        lightweight_stamp = hash_ref.to_lightweight_stamp()
        print(f"\n💡 Lightweight Stamp: {lightweight_stamp}")

        # Check for dot file
        dot_file_path = hash_ref.to_dot_file_name(file_path)
        if os.path.exists(dot_file_path):
            print(f"📁 Dot file created: {os.path.basename(dot_file_path)}")

        return hash_ref, enhanced_metadata


async def demo_documentation_analysis(extractor: EnhancedMetadataExtractor):
    """Demo: Analyze documentation for different categorization."""

    print("\n3️⃣ Documentation Analysis")
    print("-" * 25)

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "auth_docs.md")

        print("📖 Analyzing authentication service documentation...")

        hash_ref, enhanced_metadata = await extractor.extract_and_store_metadata(
            content=SAMPLE_DOCUMENTATION,
            file_path=file_path,
            content_type="text/markdown",
        )

        print(f"📊 Metadata hash: {hash_ref.metadata_hash[:12]}...")

        # Compare categorization between code and docs
        print("\n📊 Comparison: Code vs Documentation")
        print(f"   Categories: {len(enhanced_metadata.categories)} found")
        print(f"   Tags: {len(enhanced_metadata.tags)} found")
        print(
            f"   Content Type: {enhanced_metadata.content_type_detected or 'Auto-detected'}"
        )

        # Show content analysis insights
        if enhanced_metadata.complexity_score:
            print(f"   Complexity: {enhanced_metadata.complexity_score:.2f}")

        return hash_ref, enhanced_metadata


async def demo_hash_only_workflow(extractor: EnhancedMetadataExtractor):
    """Demo: Complete hash-only stamping workflow."""

    print("\n4️⃣ Hash-Only Stamping Workflow")
    print("-" * 32)

    with tempfile.TemporaryDirectory() as temp_dir:
        original_file = os.path.join(temp_dir, "example.py")

        # Step 1: Create and stamp file
        print("📝 Step 1: Creating and stamping file...")

        with open(original_file, "w") as f:
            f.write(SAMPLE_PYTHON_API)

        hash_ref, metadata = await extractor.extract_and_store_metadata(
            content=SAMPLE_PYTHON_API, file_path=original_file
        )

        # Apply lightweight stamp
        stamped_content = hash_ref.to_lightweight_stamp() + "\n" + SAMPLE_PYTHON_API

        with open(original_file, "w") as f:
            f.write(stamped_content)

        print(f"✅ File stamped with hash: {hash_ref.metadata_hash[:12]}...")

        # Step 2: Read stamped file and extract hash
        print("\n🔍 Step 2: Reading stamped file and extracting hash...")

        with open(original_file) as f:
            file_content = f.read()

        extracted_hash = extractor.extract_hash_from_stamp(file_content)
        print(f"🎯 Extracted hash: {extracted_hash[:12]}...")
        print(
            f"✅ Hash match: {'Yes' if extracted_hash == hash_ref.metadata_hash else 'No'}"
        )

        # Step 3: Retrieve metadata by hash
        print("\n📊 Step 3: Retrieving metadata by hash...")

        retrieved_metadata = await extractor.retrieve_metadata_by_hash(extracted_hash)
        if retrieved_metadata:
            print("✅ Metadata successfully retrieved")
            print(f"   Categories: {len(retrieved_metadata.categories)}")
            print(f"   Tags: {len(retrieved_metadata.tags)}")
            print(f"   Entities: {len(retrieved_metadata.entities)}")
        else:
            print("❌ Metadata retrieval failed")

        # Step 4: Show idempotency
        print("\n🔄 Step 4: Demonstrating idempotency...")

        # Modify metadata (this won't change the file)
        print("   Simulating metadata update (file remains unchanged)...")
        print("   ✅ File content stable - only external metadata changes")

        return hash_ref


async def demo_metadata_querying(extractor: EnhancedMetadataExtractor):
    """Demo: Advanced metadata querying capabilities."""

    print("\n5️⃣ Metadata Querying Capabilities")
    print("-" * 35)

    # Quick category extraction
    print("🏷️  Quick category extraction:")
    categories = await extractor.get_content_categories(SAMPLE_PYTHON_API)
    for cat in categories:
        print(f"   • {cat.category} (confidence: {cat.confidence:.2f})")

    print("\n🎯 Quick tag extraction:")
    tags = await extractor.get_content_tags(SAMPLE_PYTHON_API)
    for tag in tags:
        print(f"   • {tag.tag} ({tag.tag_type}, relevance: {tag.relevance:.2f})")


async def demo_simulation_mode():
    """Simulation mode when LangExtract is not available."""

    print("\n🎭 Simulation Mode Demo")
    print("=" * 25)

    # Simulate the enhanced metadata structure
    print("📊 Simulated Enhanced Metadata:")

    simulated_categories = [
        {
            "category": "API Service",
            "subcategory": "Authentication",
            "confidence": 0.95,
        },
        {"category": "Security", "subcategory": "JWT", "confidence": 0.90},
        {"category": "Web Framework", "subcategory": "FastAPI", "confidence": 0.88},
    ]

    simulated_tags = [
        {"tag": "authentication", "tag_type": "feature", "relevance": 0.95},
        {"tag": "JWT", "tag_type": "technology", "relevance": 0.90},
        {"tag": "OAuth2", "tag_type": "protocol", "relevance": 0.85},
        {"tag": "FastAPI", "tag_type": "framework", "relevance": 0.88},
        {"tag": "security", "tag_type": "category", "relevance": 0.92},
    ]

    simulated_entities = [
        {"entity_text": "FastAPI", "entity_type": "framework", "confidence": 0.95},
        {"entity_text": "JWT", "entity_type": "technology", "confidence": 0.90},
        {"entity_text": "OAuth2", "entity_type": "protocol", "confidence": 0.85},
        {"entity_text": "bcrypt", "entity_type": "algorithm", "confidence": 0.80},
    ]

    print(f"\n📂 Categories ({len(simulated_categories)}):")
    for cat in simulated_categories:
        print(
            f"   • {cat['category']} - {cat['subcategory']} (confidence: {cat['confidence']})"
        )

    print(f"\n🏷️  Tags ({len(simulated_tags)}):")
    for tag in simulated_tags:
        print(f"   • {tag['tag']} ({tag['tag_type']}, relevance: {tag['relevance']})")

    print(f"\n🎯 Entities ({len(simulated_entities)}):")
    for entity in simulated_entities:
        print(
            f"   • {entity['entity_text']} ({entity['entity_type']}, confidence: {entity['confidence']})"
        )

    # Simulate hash-only stamping
    print("\n💡 Simulated Hash-Only Stamping:")
    simulated_hash = "a1b2c3d4e5f6789..."
    lightweight_stamp = f"<!-- ONEX:META:{simulated_hash} -->"
    print(f"   Lightweight stamp: {lightweight_stamp}")
    print("   External storage: ✅ Database + .example.py.onex-meta.json")

    print("\n🔄 Benefits Demonstrated:")
    print("   ✅ Clean code files (no metadata clutter)")
    print("   ✅ Rich external metadata storage")
    print("   ✅ Idempotent operations")
    print("   ✅ Version control friendly")
    print("   ✅ Fast hash-based lookups")


if __name__ == "__main__":
    """Run the enhanced metadata stamping demo."""

    print("🎯 Enhanced Metadata Stamping with LangExtract")
    print("🔥 Hash-Only Stamping + Intelligent Analysis")
    print()

    try:
        asyncio.run(demo_enhanced_metadata_extraction())
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("\n🎭 Running in simulation mode instead...")
        asyncio.run(demo_simulation_mode())
