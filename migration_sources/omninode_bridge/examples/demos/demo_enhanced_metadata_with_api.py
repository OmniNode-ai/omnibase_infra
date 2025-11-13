#!/usr/bin/env python3
"""
Demonstration of enhanced metadata extraction with real API integration.

This script shows how to use the enhanced metadata extractor with real API keys
when available, and demonstrates the full capabilities of the system.

To test with real API:
1. Set GOOGLE_API_KEY environment variable
2. Run: poetry run python demo_enhanced_metadata_with_api.py
"""

import asyncio
import json
import os
import sys
import tempfile

# Add src to path
sys.path.insert(0, "src")


async def demo_with_api_key():
    """Demonstrate enhanced metadata extraction with real API key."""

    print("🚀 ENHANCED METADATA EXTRACTION DEMO")
    print("=" * 45)

    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("⚠️  No GOOGLE_API_KEY found in environment")
        print("💡 To test with real API:")
        print(
            "   export GOOGLE_API_KEY='your-api-key-here'"  # pragma: allowlist secret
        )
        print("   poetry run python demo_enhanced_metadata_with_api.py")
        print("\n🎭 Running in simulation mode instead...")
        return await demo_simulation_mode()

    print(f"✅ API Key found: ***{api_key[-4:]}")

    try:
        from omninode_bridge.services.metadata_stamping.engine.enhanced_metadata_extractor import (
            EnhancedMetadataExtractor,
        )

        # Create extractor with real API key
        extractor = EnhancedMetadataExtractor(
            model_id="gemini-2.5-flash",
            api_key=api_key,
            enable_visualization=False,  # Disable for demo
            enable_dot_files=True,
        )

        print("✅ Enhanced extractor initialized with real API")

        # Test content - comprehensive Python microservice
        test_content = '''
# User Management Microservice
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field, validator
from typing import List, Optional, Dict, Any
from datetime import UTC, UTC, datetime, timedelta
import uuid
import bcrypt
import jwt
import redis
import asyncpg
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'Request duration')

# Security
security = HTTPBearer()
SECRET_KEY = "your-jwt-secret-key"  # pragma: allowlist secret
ALGORITHM = "HS256"

# Models
class UserBase(BaseModel):
    """Base user model."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = None

    @validator('username')
    def username_alphanumeric(cls, v):
        assert v.isalnum(), 'Username must be alphanumeric'
        return v

class UserCreate(UserBase):
    """User creation model."""
    password: str = Field(..., min_length=8)
    confirm_password: str

    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v

class UserResponse(UserBase):
    """User response model."""
    id: uuid.UUID
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool = True
    role: str = "user"

class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: str

# Database operations
class UserRepository:
    """User repository with PostgreSQL backend."""

    def __init__(self, connection_pool):
        self.pool = connection_pool

    async def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create new user in database."""
        user_id = uuid.uuid4()
        hashed_password = bcrypt.hashpw(user_data.password.encode(), bcrypt.gensalt())

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO users (id, email, username, full_name, password_hash, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                user_id, user_data.email, user_data.username,
                user_data.full_name, hashed_password.decode(), datetime.now(UTC)
            )

        return UserResponse(
            id=user_id,
            email=user_data.email,
            username=user_data.username,
            full_name=user_data.full_name,
            created_at=datetime.now(UTC),
            last_login=None
        )

    async def authenticate_user(self, username: str, password: str) -> Optional[UserResponse]:
        """Authenticate user credentials."""
        async with self.pool.acquire() as conn:
            user_record = await conn.fetchrow(
                "SELECT * FROM users WHERE username = $1 AND is_active = true",
                username
            )

            if user_record and bcrypt.checkpw(password.encode(), user_record['password_hash'].encode()):
                # Update last login
                await conn.execute(
                    "UPDATE users SET last_login = $1 WHERE id = $2",
                    datetime.now(UTC), user_record['id']
                )

                return UserResponse(**dict(user_record))

        return None

# FastAPI app
app = FastAPI(
    title="User Management Microservice",
    description="Comprehensive user management with authentication, authorization, and monitoring",
    version="2.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Redis client for session management
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Dependency injection
async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Get current authenticated user."""
    token = credentials.credentials

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")

        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        # Check if token is blacklisted
        if redis_client.get(f"blacklist:{token}"):
            raise HTTPException(status_code=401, detail="Token revoked")

        return username

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# API endpoints
@app.post("/api/v1/users/register", response_model=UserResponse)
async def register_user(user_data: UserCreate, user_repo: UserRepository = Depends()):
    """Register a new user account."""
    REQUEST_COUNT.labels(method='POST', endpoint='/users/register').inc()

    with REQUEST_DURATION.time():
        try:
            # Check if user already exists
            existing_user = await user_repo.get_user_by_username(user_data.username)
            if existing_user:
                raise HTTPException(status_code=400, detail="Username already registered")

            # Create user
            new_user = await user_repo.create_user(user_data)

            return new_user

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/api/v1/auth/login", response_model=TokenResponse)
async def login(username: str, password: str, user_repo: UserRepository = Depends()):
    """Authenticate user and return JWT tokens."""
    REQUEST_COUNT.labels(method='POST', endpoint='/auth/login').inc()

    with REQUEST_DURATION.time():
        user = await user_repo.authenticate_user(username, password)

        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Generate tokens
        access_token_expires = timedelta(hours=1)
        refresh_token_expires = timedelta(days=7)

        access_token = jwt.encode(
            {"sub": user.username, "exp": datetime.now(UTC) + access_token_expires},
            SECRET_KEY, algorithm=ALGORITHM
        )

        refresh_token = jwt.encode(
            {"sub": user.username, "type": "refresh", "exp": datetime.now(UTC) + refresh_token_expires},
            SECRET_KEY, algorithm=ALGORITHM
        )

        # Store refresh token in Redis
        redis_client.setex(f"refresh:{user.username}", refresh_token_expires, refresh_token)

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=3600
        )

@app.get("/api/v1/users/me", response_model=UserResponse)
async def get_current_user_info(current_user: str = Depends(get_current_user)):
    """Get current user information."""
    REQUEST_COUNT.labels(method='GET', endpoint='/users/me').inc()

    # Fetch user details from database
    # Implementation depends on user repository
    pass

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
'''

        print(f"📄 Content size: {len(test_content)} characters")
        print("🔍 Content type: Advanced Python microservice")

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "user_management_service.py")

            print("\n🚀 Extracting enhanced metadata with real API...")
            start_time = asyncio.get_event_loop().time()

            # Extract metadata with real API
            hash_ref, metadata = await extractor.extract_and_store_metadata(
                content=test_content, file_path=file_path
            )

            total_time = (asyncio.get_event_loop().time() - start_time) * 1000

            # Display results
            print("✅ Enhanced metadata extraction completed!")
            print(f"   ⏱️  Total processing time: {total_time:.2f}ms")
            print(f"   🧠 Model used: {metadata.extraction_model}")
            print(f"   📊 Extraction confidence: {metadata.extraction_confidence:.2f}")
            print(f"   🔗 Content hash: {metadata.content_hash[:12]}...")
            print(f"   🔗 Metadata hash: {hash_ref.metadata_hash[:12]}...")

            # Show extracted intelligence
            print(f"\n📂 Categories Found: {len(metadata.categories)}")
            for i, category in enumerate(metadata.categories):
                print(f"   {i+1}. {category.category}")
                if category.subcategory:
                    print(
                        f"      └─ {category.subcategory} (confidence: {category.confidence:.2f})"
                    )

            print(f"\n🔖 Tags Found: {len(metadata.tags)}")
            tag_groups = {}
            for tag in metadata.tags:
                if tag.tag_type not in tag_groups:
                    tag_groups[tag.tag_type] = []
                tag_groups[tag.tag_type].append(tag)

            for tag_type, tags in tag_groups.items():
                print(f"   {tag_type.title()}: {', '.join(t.tag for t in tags[:5])}")
                if len(tags) > 5:
                    print(f"      ... and {len(tags) - 5} more")

            print(f"\n🎯 Entities Found: {len(metadata.entities)}")
            entity_groups = {}
            for entity in metadata.entities:
                if entity.entity_type not in entity_groups:
                    entity_groups[entity.entity_type] = []
                entity_groups[entity.entity_type].append(entity)

            for entity_type, entities in entity_groups.items():
                print(
                    f"   {entity_type.title()}: {', '.join(e.entity_text for e in entities[:5])}"
                )
                if len(entities) > 5:
                    print(f"      ... and {len(entities) - 5} more")

            # Validate dot file
            dot_file_path = hash_ref.to_dot_file_name(file_path)
            if os.path.exists(dot_file_path):
                print(f"\n📁 Dot file created: {os.path.basename(dot_file_path)}")

                with open(dot_file_path) as f:
                    dot_content = json.load(f)

                print(
                    f"   📦 Metadata version: {dot_content.get('onex_metadata_version')}"
                )
                print(f"   🔧 Generated by: {dot_content.get('generated_by')}")
                print(f"   📊 File size: {os.path.getsize(dot_file_path)} bytes")

            # Test lightweight stamping
            stamp = hash_ref.to_lightweight_stamp()
            stamped_content = stamp + "\n" + test_content

            print(f"\n🔖 Lightweight stamp: {stamp}")
            print(f"📏 Stamped content size: {len(stamped_content)} characters")
            print(
                f"📈 Overhead: {len(stamp) / len(test_content) * 100:.3f}% size increase"
            )

            # Test hash extraction
            extracted_hash = extractor.extract_hash_from_stamp(stamped_content)
            print(
                f"🔍 Hash extraction: {extracted_hash[:12]}... ({'✅ Matches' if extracted_hash == hash_ref.metadata_hash else '❌ Mismatch'})"
            )

            return {
                "api_used": True,
                "model": metadata.extraction_model,
                "confidence": metadata.extraction_confidence,
                "categories": len(metadata.categories),
                "tags": len(metadata.tags),
                "entities": len(metadata.entities),
                "processing_time_ms": total_time,
                "dot_file_created": os.path.exists(dot_file_path),
                "stamp_overhead_percent": len(stamp) / len(test_content) * 100,
            }

    except Exception as e:
        print(f"❌ Real API demo failed: {e}")
        print("🔄 Falling back to simulation mode...")
        return await demo_simulation_mode()


async def demo_simulation_mode():
    """Demonstrate simulation mode functionality."""

    print("\n🎭 SIMULATION MODE DEMO")
    print("=" * 25)

    try:
        from omninode_bridge.services.metadata_stamping.engine.enhanced_metadata_extractor import (
            EnhancedMetadataExtractor,
        )

        # Create extractor in simulation mode
        extractor = EnhancedMetadataExtractor(
            model_id="gemini-2.5-flash",
            api_key=None,  # No API key - simulation mode
            enable_visualization=False,
            enable_dot_files=True,
        )

        print("✅ Enhanced extractor initialized in simulation mode")

        # Smaller test content for simulation
        test_content = '''
# Authentication API
from fastapi import FastAPI, Depends
from pydantic import BaseModel

class User(BaseModel):
    username: str
    email: str

app = FastAPI(title="Auth API")

@app.post("/login")
async def login(user: User):
    """User authentication endpoint."""
    return {"token": "jwt_token_here", "user": user.username}

@app.get("/profile")
async def get_profile(user_id: str):
    """Get user profile information."""
    return {"user_id": user_id, "profile": "user_data"}
'''

        print(f"📄 Content size: {len(test_content)} characters")

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "auth_api.py")

            print("\n🚀 Extracting metadata in simulation mode...")
            start_time = asyncio.get_event_loop().time()

            # Extract metadata (will use simulation/fallback)
            hash_ref, metadata = await extractor.extract_and_store_metadata(
                content=test_content, file_path=file_path
            )

            total_time = (asyncio.get_event_loop().time() - start_time) * 1000

            print("✅ Simulation extraction completed!")
            print(f"   ⏱️  Processing time: {total_time:.2f}ms")
            print(
                f"   📊 Confidence: {metadata.extraction_confidence:.2f} (expected 0.0 for simulation)"
            )
            print(f"   🔗 Content hash: {metadata.content_hash[:12]}...")
            print(f"   🔗 Metadata hash: {hash_ref.metadata_hash[:12]}...")

            # Validate basic functionality still works
            stamp = hash_ref.to_lightweight_stamp()
            stamped_content = stamp + "\n" + test_content

            extracted_hash = extractor.extract_hash_from_stamp(stamped_content)
            hash_match = extracted_hash == hash_ref.metadata_hash

            print("\n🔖 Stamp functionality:")
            print(f"   Lightweight stamp: {stamp}")
            print(f"   Hash extraction: {'✅ Working' if hash_match else '❌ Failed'}")

            # Check dot file
            dot_file_path = hash_ref.to_dot_file_name(file_path)
            dot_file_exists = os.path.exists(dot_file_path)

            print(
                f"   Dot file: {'✅ Created' if dot_file_exists else '❌ Not created'}"
            )

            if dot_file_exists:
                with open(dot_file_path) as f:
                    dot_content = json.load(f)
                print(f"   Dot file size: {os.path.getsize(dot_file_path)} bytes")

            return {
                "simulation_mode": True,
                "processing_time_ms": total_time,
                "confidence": metadata.extraction_confidence,
                "hash_extraction_works": hash_match,
                "dot_file_created": dot_file_exists,
                "content_hash": metadata.content_hash,
                "metadata_hash": hash_ref.metadata_hash,
            }

    except Exception as e:
        print(f"❌ Simulation demo failed: {e}")
        return {"error": str(e)}


async def main():
    """Run the demonstration."""

    print("🎪 ENHANCED METADATA EXTRACTION DEMONSTRATION")
    print("=" * 50)
    print("This demo shows the enhanced metadata extraction capabilities")
    print("with LangExtract integration and graceful fallback handling.")
    print()

    # Check environment
    print("🔍 Environment Check:")
    print(f"   Python version: {sys.version.split()[0]}")
    print(f"   Working directory: {os.getcwd()}")

    # Try to import dependencies
    try:
        from omninode_bridge.services.metadata_stamping.engine.enhanced_metadata_extractor import (
            LANGEXTRACT_AVAILABLE,
        )

        print(f"   LangExtract available: {LANGEXTRACT_AVAILABLE}")
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return

    # Run demo
    result = await demo_with_api_key()

    # Summary
    print("\n📊 DEMO SUMMARY")
    print("=" * 20)

    if "error" in result:
        print(f"❌ Demo failed: {result['error']}")
    else:
        if result.get("api_used"):
            print("✅ Real API integration demonstrated successfully")
            print(f"   Model: {result.get('model', 'unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.2f}")
            print(f"   Categories: {result.get('categories', 0)}")
            print(f"   Tags: {result.get('tags', 0)}")
            print(f"   Entities: {result.get('entities', 0)}")
        else:
            print("✅ Simulation mode demonstrated successfully")
            print(f"   Processing time: {result.get('processing_time_ms', 0):.2f}ms")
            print(
                f"   Hash extraction: {'Working' if result.get('hash_extraction_works') else 'Failed'}"
            )
            print(
                f"   Dot file: {'Created' if result.get('dot_file_created') else 'Not created'}"
            )

        print("\n💡 To enable full LangExtract features:")
        print("   1. Set GOOGLE_API_KEY environment variable")
        print("   2. Restart the service")
        print("   3. Enhanced categorization and tagging will be available")

    return result


if __name__ == "__main__":
    result = asyncio.run(main())
