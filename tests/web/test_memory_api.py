"""Test cases for memory API endpoints."""

import os
import tempfile
from datetime import datetime
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.memory.base import MemoryStore
from xagent.core.memory.core import MemoryNote
from xagent.core.memory.remote import (
    RemoteMemoryStoreError,
    RemoteMemoryStoreIncompleteError,
)
from xagent.web.api.auth import auth_router, hash_password
from xagent.web.api.memory import MemoryManagementRouter
from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY
from xagent.web.models.database import Base, get_db
from xagent.web.models.user import User

# Create temporary directory for database
temp_dir = tempfile.mkdtemp()
temp_db_path = os.path.join(temp_dir, "test.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{temp_db_path}"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = None
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        if db is not None:
            db.close()


@pytest.fixture(scope="function")
def test_db():
    """Create test database"""
    Base.metadata.create_all(bind=engine)
    # Create admin user
    session = TestingSessionLocal()
    try:
        admin_user = User(
            username="admin", password_hash=hash_password("admin"), is_admin=True
        )
        session.add(admin_user)
        session.commit()
        yield admin_user
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="session", autouse=True)
def cleanup_global_test_db():
    """Cleanup global test database after all tests"""
    yield
    try:
        import shutil

        shutil.rmtree(temp_dir)
    except OSError:
        pass


@pytest.fixture(scope="function")
def auth_headers(test_db):
    """Authentication headers for admin user"""
    # Create a valid JWT token directly
    from datetime import datetime, timedelta, timezone

    import jwt

    payload = {
        "sub": "admin",
        "type": "access",
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        "iat": datetime.now(timezone.utc),
        "user_id": test_db.id,  # Use actual user ID from test_db fixture
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="function")
def regular_user(test_db):
    """A non-admin account, so the personal-scope path keeps coverage."""
    session = TestingSessionLocal()
    try:
        user = User(
            username="member",
            password_hash=hash_password("member"),
            is_admin=False,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user
    finally:
        session.close()


@pytest.fixture(scope="function")
def regular_auth_headers(regular_user):
    """Authentication headers for a regular (non-admin) user."""
    from datetime import datetime, timedelta, timezone

    import jwt

    payload = {
        "sub": "member",
        "type": "access",
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        "iat": datetime.now(timezone.utc),
        "user_id": regular_user.id,
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def mock_memory_store():
    """Create a mock memory store for testing."""
    store = Mock(spec=MemoryStore)
    return store


@pytest.fixture
def memory_router(mock_memory_store):
    """Create memory management router with mock store."""
    return MemoryManagementRouter(lambda: mock_memory_store)


@pytest.fixture
def client(memory_router):
    """Create test client with memory router and authentication."""
    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(memory_router.get_router())
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


@pytest.fixture
def sample_memory_notes():
    """Create sample memory notes for testing."""
    return [
        MemoryNote(
            id="memory_1",
            content="Test memory 1",
            keywords=["test", "sample"],
            tags=["important"],
            category="general",
            metadata={"source": "test"},
            timestamp=datetime.now(),
            mime_type="text/plain",
        ),
        MemoryNote(
            id="memory_2",
            content="Test memory 2",
            keywords=["test", "example"],
            tags=["normal"],
            category="system",
            metadata={"source": "system"},
            timestamp=datetime.now(),
            mime_type="text/plain",
        ),
    ]


class TestMemoryListEndpoint:
    """Test cases for memory list endpoint."""

    def test_list_memories_success(
        self, client, mock_memory_store, sample_memory_notes, auth_headers
    ):
        """Test successful memory listing."""
        # Setup mock response - memory API expects list_all to return a list-like object
        mock_memory_store.list_all.return_value = sample_memory_notes

        response = client.get("/api/memory/list", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "memories" in data
        assert "total_count" in data
        assert "filters_used" in data
        assert data["total_count"] == 2
        assert len(data["memories"]) == 2

        # Verify memory structure
        memory = data["memories"][0]
        assert "id" in memory
        assert "content" in memory
        assert "keywords" in memory
        assert "tags" in memory
        assert "category" in memory
        assert "timestamp" in memory
        assert "metadata" in memory
        # Ensure context field is not present
        assert "context" not in memory

    def test_list_memories_with_category_filter(
        self, client, mock_memory_store, auth_headers
    ):
        """Test memory listing with category filter."""
        filtered_memories = [
            MemoryNote(
                id="filtered_1",
                content="Filtered memory",
                keywords=["test"],
                category="general",
                metadata={"source": "test"},
            )
        ]
        mock_memory_store.list_all.return_value = filtered_memories

        response = client.get("/api/memory/list?category=general", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert len(data["memories"]) == 1
        assert data["memories"][0]["category"] == "general"

    def test_list_memories_with_tags_filter(
        self, client, mock_memory_store, auth_headers
    ):
        """Test memory listing with tags filter."""
        mock_memory_store.list_all.return_value = []

        response = client.get(
            "/api/memory/list?tags=important,normal", headers=auth_headers
        )

        assert response.status_code == 200
        # Verify that the mock was called with correct filters
        mock_memory_store.list_all.assert_called_once()
        call_args = mock_memory_store.list_all.call_args[0]
        filters = call_args[0] if call_args else {}
        assert "tags" in filters
        assert set(filters["tags"]) == {"important", "normal"}

    def test_list_memories_with_keywords_filter(
        self, client, mock_memory_store, auth_headers
    ):
        """Test memory listing with keywords filter."""
        mock_memory_store.list_all.return_value = []

        response = client.get(
            "/api/memory/list?keywords=test,example", headers=auth_headers
        )

        assert response.status_code == 200
        # Verify that the mock was called with correct filters
        mock_memory_store.list_all.assert_called_once()
        call_args = mock_memory_store.list_all.call_args[0]
        filters = call_args[0] if call_args else {}
        assert "keywords" in filters
        assert set(filters["keywords"]) == {"test", "example"}

    def test_list_memories_with_date_filters(
        self, client, mock_memory_store, auth_headers
    ):
        """Test memory listing with date range filters."""
        mock_memory_store.list_all.return_value = []

        response = client.get(
            "/api/memory/list?date_from=2024-01-01&date_to=2024-12-31",
            headers=auth_headers,
        )

        assert response.status_code == 200
        # Verify that the mock was called with correct filters
        mock_memory_store.list_all.assert_called_once()
        call_args = mock_memory_store.list_all.call_args[0]
        filters = call_args[0] if call_args else {}
        assert "date_from" in filters
        assert "date_to" in filters

    def test_list_memories_with_limit_and_offset(
        self, client, mock_memory_store, auth_headers
    ):
        """Test memory listing with pagination."""
        # Create more memories than limit
        many_memories = [
            MemoryNote(
                id=f"memory_{i}",
                content=f"Test memory {i}",
                keywords=["test"],
                category="general",
            )
            for i in range(10)
        ]
        mock_memory_store.list_all.return_value = many_memories

        response = client.get("/api/memory/list?limit=5&offset=3", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert len(data["memories"]) == 5
        assert data["total_count"] == 10

    def test_list_memories_error_handling(
        self, client, mock_memory_store, auth_headers
    ):
        """Test error handling in memory listing."""
        # Setup mock to raise exception
        mock_memory_store.list_all.side_effect = Exception("Database error")

        response = client.get("/api/memory/list", headers=auth_headers)

        assert response.status_code == 503
        assert response.json()["detail"] == "Memory storage is temporarily unavailable."
        assert "Database error" not in response.text

    def test_search_memories_returns_empty_success_for_no_match(
        self, client, mock_memory_store, auth_headers
    ):
        mock_memory_store.search.return_value = []

        response = client.get("/api/memory/list?search=missing", headers=auth_headers)

        assert response.status_code == 200
        assert response.json()["memories"] == []
        assert response.json()["total_count"] == 0

    def test_incomplete_remote_list_never_reports_a_total(
        self, client, mock_memory_store, auth_headers
    ):
        mock_memory_store.list_all.side_effect = RemoteMemoryStoreIncompleteError(
            "Remote memory list is incomplete"
        )
        response = client.get("/api/memory/list?limit=5", headers=auth_headers)
        assert response.status_code == 503
        assert "total_count" not in response.json()
        assert "memories" not in response.json()

    def test_search_memories_sanitizes_backend_failure(
        self, client, mock_memory_store, auth_headers
    ):
        mock_memory_store.search.side_effect = RuntimeError(
            "lancedb://internal-host/private-table"
        )

        response = client.get(
            "/api/memory/list?search=preferences", headers=auth_headers
        )

        assert response.status_code == 503
        assert response.json()["detail"] == "Memory storage is temporarily unavailable."
        assert "internal-host" not in response.text


class TestMemoryGetEndpoint:
    """Test cases for memory get endpoint."""

    def test_get_memory_success(
        self, client, mock_memory_store, sample_memory_notes, auth_headers
    ):
        """Test successful memory retrieval."""
        mock_memory_store.get.return_value = Mock(
            success=True, memory_id="memory_1", content=sample_memory_notes[0]
        )

        response = client.get("/api/memory/memory_1", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "memory_1"
        assert data["content"] == "Test memory 1"
        assert "context" not in data  # Ensure context field is not present

    def test_get_memory_not_found(self, client, mock_memory_store, auth_headers):
        """Test memory retrieval for non-existent ID."""
        mock_memory_store.get.return_value = Mock(
            success=False, error="Memory not found", memory_id="non_existent"
        )

        response = client.get("/api/memory/non_existent", headers=auth_headers)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_memory_error_handling(self, client, mock_memory_store, auth_headers):
        """Test error handling in memory retrieval."""
        # Setup mock to raise exception
        mock_memory_store.get.side_effect = Exception("Database error")

        response = client.get("/api/memory/test_id", headers=auth_headers)

        assert response.status_code == 500
        assert "Failed to get memory" in response.json()["detail"]


class TestMemoryCreateEndpoint:
    """Test cases for memory create endpoint."""

    def test_create_memory_success(self, client, mock_memory_store, auth_headers):
        """Test successful memory creation."""
        # Setup mock response
        mock_memory_store.add.return_value = Mock(
            success=True, memory_id="new_memory_123"
        )

        memory_data = {
            "content": "New test memory",
            "keywords": ["test", "new"],
            "tags": ["example"],
            "category": "general",
            "metadata": {"source": "test"},
        }

        response = client.post("/api/memory/", json=memory_data, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["memory_id"] == "new_memory_123"
        # The receipt and the GET response must name the note the same way;
        # clients used to receive only memory_id here and id everywhere else.
        assert data["id"] == data["memory_id"]
        assert "Memory created successfully" in data["message"]

        # Verify that add was called with MemoryNote
        mock_memory_store.add.assert_called_once()
        call_args = mock_memory_store.add.call_args[0][0]
        assert isinstance(call_args, MemoryNote)
        assert call_args.content == "New test memory"
        assert call_args.category == "general"

    def test_create_memory_with_defaults(self, client, mock_memory_store, auth_headers):
        """Test memory creation with default values."""
        mock_memory_store.add.return_value = Mock(
            success=True, memory_id="default_memory"
        )

        # Minimal memory data
        memory_data = {"content": "Minimal memory"}

        response = client.post("/api/memory/", json=memory_data, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify default values were applied
        call_args = mock_memory_store.add.call_args[0][0]
        assert isinstance(call_args, MemoryNote)
        assert call_args.content == "Minimal memory"
        assert call_args.category == "general"  # Default category
        assert call_args.keywords == []  # Default empty list
        assert call_args.tags == []  # Default empty list
        assert call_args.metadata == {}  # Default empty dict

    def test_create_memory_error_handling(
        self, client, mock_memory_store, auth_headers
    ):
        """Test error handling in memory creation."""
        # Setup mock to return failure
        mock_memory_store.add.return_value = Mock(success=False, error="Storage failed")

        memory_data = {"content": "Test memory"}

        response = client.post("/api/memory/", json=memory_data, headers=auth_headers)

        assert response.status_code == 500
        # The route raises ``response.error or "Failed to create memory"``, so
        # the store's own message is the intended detail. It used to be caught
        # by the route's own broad handler and re-wrapped, status code and all,
        # into "Failed to create memory: 500: Storage failed"; that wrapper is
        # gone because it also swallowed the lifecycle 503.
        assert response.json()["detail"] == "Storage failed"

    def test_create_memory_error_handling_without_a_store_message(
        self, client, mock_memory_store, auth_headers
    ):
        """A refusal that carries no message still answers the generic 500."""
        mock_memory_store.add.return_value = Mock(success=False, error=None)

        response = client.post(
            "/api/memory/", json={"content": "Test memory"}, headers=auth_headers
        )

        assert response.status_code == 500
        assert response.json()["detail"] == "Failed to create memory"

    def test_create_surfaces_upstream_409(
        self, client, mock_memory_store, auth_headers
    ):
        """The create route forwards a 409 like delete/update do."""
        mock_memory_store.add.side_effect = RemoteMemoryStoreError(
            "delivered notes are append-only", status_code=409
        )

        response = client.post(
            "/api/memory/", json={"content": "Test memory"}, headers=auth_headers
        )

        assert response.status_code == 409
        assert "append-only" in response.json()["detail"]


class TestMemoryUpdateEndpoint:
    """Test cases for memory update endpoint."""

    def test_update_memory_success(
        self, client, mock_memory_store, sample_memory_notes, auth_headers
    ):
        """Test successful memory update."""
        # Setup mock responses
        mock_memory_store.get.return_value = Mock(
            success=True, content=sample_memory_notes[0]
        )
        mock_memory_store.update.return_value = Mock(success=True)

        update_data = {
            "content": "Updated content",
            "keywords": ["updated"],
            "tags": ["modified"],
            "category": "general",
            "metadata": {"source": "test", "updated": True},
        }

        response = client.put(
            "/api/memory/memory_1", json=update_data, headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Memory updated successfully" in data["message"]

        # Verify that update was called with updated MemoryNote
        mock_memory_store.update.assert_called_once()
        call_args = mock_memory_store.update.call_args[0][0]
        assert isinstance(call_args, MemoryNote)
        assert call_args.content == "Updated content"
        assert call_args.keywords == ["updated"]
        assert call_args.id == "memory_1"

    def test_update_memory_not_found(self, client, mock_memory_store, auth_headers):
        """Test memory update for non-existent ID."""
        mock_memory_store.get.return_value = Mock(
            success=False, error="Memory not found"
        )

        update_data = {"content": "Updated content"}

        response = client.put(
            "/api/memory/non_existent", json=update_data, headers=auth_headers
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_update_memory_partial_update(
        self, client, mock_memory_store, sample_memory_notes, auth_headers
    ):
        """Test partial memory update."""
        mock_memory_store.get.return_value = Mock(
            success=True, content=sample_memory_notes[0]
        )
        mock_memory_store.update.return_value = Mock(success=True)

        # Update only content
        update_data = {"content": "Only content updated"}

        response = client.put(
            "/api/memory/memory_1", json=update_data, headers=auth_headers
        )

        assert response.status_code == 200

        # Verify that only content was updated, other fields preserved
        call_args = mock_memory_store.update.call_args[0][0]
        assert call_args.content == "Only content updated"
        # X-06: a keyword that only the replaced body supported ("test" was in
        # "Test memory 1") is cleared so the service cannot refill tags from it;
        # "sample" never appeared in the body and is kept.
        assert call_args.keywords == ["sample"]
        assert call_args.tags == ["important"]  # Original values preserved

    def test_update_memory_drops_the_tag_the_new_content_replaced(self, client, mock_memory_store, auth_headers):
        """X-06: a content-only PUT must not keep the old plate in `tags`.

        The store keeps whichever tags a request carries (probed 2026-10-29),
        so a body rewritten from 京A8888 to 京A9999 left the old value in tags.
        """
        mock_memory_store.get.return_value = Mock(
            success=True,
            content=MemoryNote(
                id="memory_1",
                content="Y-C7-PLATE 我的车牌号是 京A8888，停车月卡登记用。",
                keywords=["车牌号", "京A8888"],
                tags=["车牌号", "京A8888"],
                category="general",
            ),
        )
        mock_memory_store.update.return_value = Mock(success=True)

        response = client.put(
            "/api/memory/memory_1",
            json={"content": "Y-C7-PLATE 我的车牌号是 京A9999，停车月卡登记用。"},
            headers=auth_headers,
        )

        assert response.status_code == 200
        written = mock_memory_store.update.call_args[0][0]
        assert written.content.endswith("京A9999，停车月卡登记用。")
        assert written.tags == ["车牌号"]

    def test_update_memory_honours_tags_the_caller_supplies(self, client, mock_memory_store, auth_headers):
        """An explicit tag list is the caller's decision, not the client's."""
        mock_memory_store.get.return_value = Mock(
            success=True,
            content=MemoryNote(
                id="memory_1", content="车牌 京A8888", tags=["京A8888"]
            ),
        )
        mock_memory_store.update.return_value = Mock(success=True)

        response = client.put(
            "/api/memory/memory_1",
            json={"content": "车牌 京A9999", "tags": ["2026-10-29校正"]},
            headers=auth_headers,
        )

        assert response.status_code == 200
        assert mock_memory_store.update.call_args[0][0].tags == ["2026-10-29校正"]

    def test_update_memory_clears_stale_keywords_without_tags(self, client, mock_memory_store, auth_headers):
        """X-06: an empty tag list must not be refilled from old keywords.

        memsrv treats an empty ``tags`` list as "use keywords", so a PUT whose
        tags are already empty kept ``京A8888`` alive through ``keywords``
        (evidence ``probe_x06_tags3.out``).
        """
        mock_memory_store.get.return_value = Mock(
            success=True,
            content=MemoryNote(
                id="memory_1",
                content="Y-C7-PLATE 我的车牌号是 京A8888，停车月卡登记用。",
                keywords=["车牌号", "京A8888"],
                tags=[],
                category="general",
            ),
        )
        mock_memory_store.update.return_value = Mock(success=True)

        response = client.put(
            "/api/memory/memory_1",
            json={"content": "Y-C7-PLATE 我的车牌号是 京A9999，停车月卡登记用。"},
            headers=auth_headers,
        )

        assert response.status_code == 200
        written = mock_memory_store.update.call_args[0][0]
        assert written.tags == []
        assert written.keywords == ["车牌号"]

    def test_update_memory_drops_echoed_old_tags_but_keeps_classification(
        self, client, mock_memory_store, auth_headers
    ):
        """The UI prefills the old tags; echoing them is not a decision.

        A value only the old body supported (京A8888) is dropped, a value the
        new body still supports (车牌号) stays, and a classification tag that
        never appeared in the body (高频) is untouched.
        """
        mock_memory_store.get.return_value = Mock(
            success=True,
            content=MemoryNote(
                id="memory_1",
                content="Y-C7-PLATE 我的车牌号是 京A8888，停车月卡登记用。",
                tags=["车牌号", "京A8888", "高频"],
                keywords=["车牌号", "京A8888"],
                category="general",
            ),
        )
        mock_memory_store.update.return_value = Mock(success=True)

        response = client.put(
            "/api/memory/memory_1",
            json={
                "content": "Y-C7-PLATE 我的车牌号是 京A9999，停车月卡登记用。",
                "tags": ["车牌号", "京A8888", "高频"],
            },
            headers=auth_headers,
        )

        assert response.status_code == 200
        written = mock_memory_store.update.call_args[0][0]
        assert written.tags == ["车牌号", "高频"]
        assert "京A8888" not in written.keywords

    def test_update_memory_error_handling(
        self, client, mock_memory_store, auth_headers
    ):
        """Test error handling in memory update."""
        # Setup mock to raise exception
        mock_memory_store.get.side_effect = Exception("Database error")

        update_data = {"content": "Updated content"}

        response = client.put(
            "/api/memory/test_id", json=update_data, headers=auth_headers
        )

        assert response.status_code == 500
        assert "Failed to update memory" in response.json()["detail"]

    def test_update_delivered_memory_surfaces_upstream_409(
        self, client, mock_memory_store, sample_memory_notes, auth_headers
    ):
        """A delivered note is append-only upstream; 409 must not become a 500."""
        mock_memory_store.get.return_value = Mock(
            success=True, content=sample_memory_notes[0]
        )
        mock_memory_store.update.side_effect = RemoteMemoryStoreError(
            "delivered notes are append-only and cannot be updated",
            status_code=409,
        )

        response = client.put(
            "/api/memory/fb83ec9f",
            json={"content": "rewrite the delivered note"},
            headers=auth_headers,
        )

        assert response.status_code == 409
        assert "append-only" in response.json()["detail"]

    def test_update_other_upstream_failure_keeps_the_500_wording(
        self, client, mock_memory_store, sample_memory_notes, auth_headers
    ):
        """Only 409 is forwarded; every other upstream status keeps the 500."""
        mock_memory_store.get.return_value = Mock(
            success=True, content=sample_memory_notes[0]
        )
        mock_memory_store.update.side_effect = RemoteMemoryStoreError(
            "upstream exploded", status_code=502
        )

        response = client.put(
            "/api/memory/memory_1",
            json={"content": "Updated content"},
            headers=auth_headers,
        )

        assert response.status_code == 500
        assert response.json()["detail"] == "Failed to update memory: upstream exploded"


class TestMemoryDeleteEndpoint:
    """Test cases for memory delete endpoint."""

    def test_delete_memory_success(self, client, mock_memory_store, auth_headers):
        """Test successful memory deletion."""
        mock_memory_store.delete.return_value = Mock(success=True)

        response = client.delete("/api/memory/memory_1", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Memory deleted successfully" in data["message"]

        # Verify delete was called with correct ID
        mock_memory_store.delete.assert_called_once_with("memory_1")

    def test_delete_memory_not_found(self, client, mock_memory_store, auth_headers):
        """Test memory deletion for non-existent ID."""
        mock_memory_store.delete.return_value = Mock(
            success=False, error="Memory not found"
        )

        response = client.delete("/api/memory/non_existent", headers=auth_headers)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_delete_memory_error_handling(
        self, client, mock_memory_store, auth_headers
    ):
        """Test error handling in memory deletion."""
        # Setup mock to raise exception
        mock_memory_store.delete.side_effect = Exception("Database error")

        response = client.delete("/api/memory/test_id", headers=auth_headers)

        assert response.status_code == 500
        assert "Failed to delete memory" in response.json()["detail"]

    def test_delete_delivered_memory_surfaces_upstream_409(
        self, client, mock_memory_store, auth_headers
    ):
        """A delivered note is append-only upstream; 409 must not become a 500."""
        mock_memory_store.delete.side_effect = RemoteMemoryStoreError(
            "delivered notes are append-only and cannot be deleted",
            status_code=409,
        )

        response = client.delete("/api/memory/fb83ec9f", headers=auth_headers)

        assert response.status_code == 409
        assert "append-only" in response.json()["detail"]

    def test_delete_other_upstream_failure_keeps_the_500_wording(
        self, client, mock_memory_store, auth_headers
    ):
        """Only 409 is forwarded; every other upstream status keeps the 500."""
        mock_memory_store.delete.side_effect = RemoteMemoryStoreError(
            "upstream exploded", status_code=502
        )

        response = client.delete("/api/memory/memory_1", headers=auth_headers)

        assert response.status_code == 500
        assert response.json()["detail"] == "Failed to delete memory: upstream exploded"


class TestMemoryStatsEndpoint:
    """Test cases for memory stats endpoint."""

    def test_get_stats_success(self, client, mock_memory_store, regular_auth_headers):
        """Test successful stats retrieval."""
        mock_stats = {
            "total_count": 10,
            "category_counts": {"general": 5, "system": 3, "test": 2},
            "tag_counts": {"important": 3, "normal": 4, "test": 5},
            "memory_store_type": "in_memory",
        }
        mock_memory_store.get_stats.return_value = mock_stats

        response = client.get("/api/memory/stats", headers=regular_auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 10
        assert data["category_counts"]["general"] == 5
        assert data["tag_counts"]["important"] == 3
        assert data["memory_store_type"] == "in_memory"
        assert data["scope"] == "personal"

    def test_get_stats_with_error(
        self, client, mock_memory_store, regular_auth_headers
    ):
        """Test stats retrieval with error."""
        mock_stats = {
            "total_count": 0,
            "category_counts": {},
            "tag_counts": {},
            "memory_store_type": "in_memory",
            "error": "Database connection failed",
        }
        mock_memory_store.get_stats.return_value = mock_stats

        response = client.get("/api/memory/stats", headers=regular_auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 0
        assert "error" in data
        assert data["error"] == "Database connection failed"

    def test_get_stats_error_handling(
        self, client, mock_memory_store, regular_auth_headers
    ):
        """Test error handling in stats retrieval."""
        # Setup mock to raise exception
        mock_memory_store.get_stats.side_effect = Exception("Database error")

        response = client.get("/api/memory/stats", headers=regular_auth_headers)

        assert response.status_code == 500
        assert "Failed to get memory stats" in response.json()["detail"]

    def test_admin_stats_aggregates_the_notes_the_list_returns(
        self, client, mock_memory_store, auth_headers
    ):
        """X-02: an admin's org figure must be the list's own notes.

        Probed 2026-10-29 on the live deployment: the admin list answered 17
        organisation notes while stats answered 4 (that identity's own), so the
        panel contradicted itself. Stats now aggregates ``list_all()`` -- the
        same call the list port makes -- over the organisation scope.
        """
        org_notes = [
            MemoryNote(
                id="mem-1",
                content="2 号的报销额度是 8800 元。",
                tags=["报销额度"],
                category="general",
                metadata={"user_id": 2},
            ),
            MemoryNote(
                id="mem-2",
                content="2 号的停车月卡登记用 京A8888。",
                tags=["e2e"],
                category="react_memory",
                metadata={"user_id": 2},
            ),
            MemoryNote(
                id="mem-3",
                content="3 号的车牌号是 京A9999。",
                tags=["e2e"],
                category="general",
                metadata={"user_id": 3},
            ),
        ]
        mock_memory_store.list_all.return_value = org_notes
        # The service's own stats stay personal; only the store type is taken.
        mock_memory_store.get_stats.return_value = {
            "total_count": 1,
            "category_counts": {"react_memory": 1},
            "tag_counts": {},
            "memory_store_type": "remote",
        }

        stats_response = client.get("/api/memory/stats", headers=auth_headers)
        list_response = client.get("/api/memory/list", headers=auth_headers)

        assert stats_response.status_code == 200
        assert list_response.status_code == 200
        stats, listing = stats_response.json(), list_response.json()
        assert stats["total_count"] == listing["total_count"] == 3
        assert sum(stats["category_counts"].values()) == stats["total_count"]
        assert stats["category_counts"] == {"general": 2, "react_memory": 1}
        assert stats["tag_counts"] == {"报销额度": 1, "e2e": 2}
        assert stats["scope"] == listing["scope"] == "org"
        assert stats["memory_store_type"] == "remote"

    def test_regular_user_stats_and_list_stay_personal_and_agree(
        self, client, mock_memory_store, regular_user, regular_auth_headers
    ):
        """A regular user keeps the personal surface, and the two ports agree."""
        own_notes = [
            MemoryNote(
                id="own-1",
                content="我的报销额度是 8800 元。",
                tags=["报销额度"],
                category="general",
                metadata={"user_id": regular_user.id},
            ),
            MemoryNote(
                id="own-2",
                content="我的停车月卡登记车牌 京A8888。",
                tags=["e2e"],
                category="react_memory",
                metadata={"user_id": regular_user.id},
            ),
        ]
        mock_memory_store.list_all.return_value = own_notes
        mock_memory_store.get_stats.return_value = {
            "total_count": 2,
            "category_counts": {"general": 1, "react_memory": 1},
            "tag_counts": {"报销额度": 1, "e2e": 1},
            "memory_store_type": "remote",
        }

        stats_response = client.get("/api/memory/stats", headers=regular_auth_headers)
        list_response = client.get("/api/memory/list", headers=regular_auth_headers)

        assert stats_response.status_code == 200
        assert list_response.status_code == 200
        stats, listing = stats_response.json(), list_response.json()
        assert stats["scope"] == listing["scope"] == "personal"
        assert stats["total_count"] == listing["total_count"] == 2


class TestMemoryApiIntegration:
    """Integration tests for memory API endpoints."""

    def test_crd_cycle(
        self, client, mock_memory_store, sample_memory_notes, auth_headers
    ):
        """Test Create-Read-Delete cycle."""

        # Setup mock responses
        def mock_add_side_effect(note):
            # Return the memory with the provided content
            return Mock(success=True, memory_id="test_memory_123")

        def mock_get_side_effect(memory_id):
            if memory_id == "test_memory_123":
                # Create a memory note with the test data
                from xagent.core.memory.core import MemoryNote

                test_memory = MemoryNote(
                    id="test_memory_123",
                    content="CRD test memory",
                    keywords=["test", "crd"],
                    tags=[],
                    category="general",
                    metadata={},
                )
                return Mock(success=True, content=test_memory)
            return Mock(success=False, error="Memory not found")

        mock_memory_store.add.side_effect = mock_add_side_effect
        mock_memory_store.get.side_effect = mock_get_side_effect
        mock_memory_store.delete.return_value = Mock(success=True)

        # Create memory
        memory_data = {
            "content": "CRD test memory",
            "keywords": ["test", "crd"],
            "category": "general",
        }

        create_response = client.post(
            "/api/memory/", json=memory_data, headers=auth_headers
        )
        assert create_response.status_code == 200
        memory_id = create_response.json()["memory_id"]

        # Read memory
        get_response = client.get(f"/api/memory/{memory_id}", headers=auth_headers)
        assert get_response.status_code == 200
        assert get_response.json()["content"] == "CRD test memory"

        # Delete memory
        delete_response = client.delete(
            f"/api/memory/{memory_id}", headers=auth_headers
        )
        assert delete_response.status_code == 200

    def test_query_parameter_parsing(self, client, mock_memory_store, auth_headers):
        """Test that query parameters are parsed correctly."""
        mock_memory_store.list_all.return_value = []

        response = client.get(
            "/api/memory/list?category=general&tags=tag1,tag2&keywords=kw1,kw2&limit=10&offset=5",
            headers=auth_headers,
        )

        assert response.status_code == 200

        # Verify that list_all was called with correct filters
        mock_memory_store.list_all.assert_called_once()
        call_args = mock_memory_store.list_all.call_args[0]
        filters = call_args[0] if call_args else {}

        assert filters["category"] == "general"
        assert set(filters["tags"]) == {"tag1", "tag2"}
        assert set(filters["keywords"]) == {"kw1", "kw2"}

    def test_invalid_parameters(self, client, mock_memory_store, auth_headers):
        """Test handling of invalid parameters."""
        mock_memory_store.list_all.return_value = []

        # Test with invalid limit
        response = client.get("/api/memory/list?limit=invalid", headers=auth_headers)
        # Should still work due to FastAPI validation
        assert response.status_code == 422  # Validation error

    def test_content_type_validation(self, client, mock_memory_store, auth_headers):
        """Test content type validation for POST/PUT requests."""
        # Setup mock to return valid response for successful creation
        mock_memory_store.add.return_value = Mock(success=True, memory_id="test_123")

        # Test with invalid content type
        response = client.post(
            "/api/memory/", content="invalid content", headers=auth_headers
        )
        assert response.status_code == 422  # Validation error

        # Test with valid JSON but invalid structure
        response = client.post(
            "/api/memory/", json={"invalid": "data"}, headers=auth_headers
        )
        # Should fail due to missing required content field
        assert response.status_code == 422


@pytest.mark.parametrize("truncated", [False, True])
def test_memsrv_envelope_through_adapter_and_web(
    client, mock_memory_store, auth_headers, test_db, truncated
):
    import json

    import requests

    from xagent.core.memory.remote import RemoteMemoryStore

    calls = []

    class Session:
        def request(self, method, url, **kwargs):
            calls.append({"method": method, "url": url, **kwargs})
            response = requests.Response()
            response.status_code = 200
            response._content = json.dumps(
                {
                    "results": [
                        {"id": "a", "content": "one"},
                        {"id": "b", "content": "two"},
                    ],
                    "limit": 10000,
                    "truncated": truncated,
                }
            ).encode()
            return response

    adapter = RemoteMemoryStore("http://memsrv:8080", "runtime-key", session=Session())
    mock_memory_store.list_all.side_effect = adapter.list_all
    response = client.get(
        "/api/memory/list?category=general&limit=1", headers=auth_headers
    )
    assert calls[0]["headers"]["X-User-External-Id"] == str(test_db.id)
    assert calls[0]["json"] == {"filters": {"category": "general"}}
    assert len(calls) == 1
    if truncated:
        assert response.status_code == 503
        assert "total_count" not in response.json()
        assert "memories" not in response.json()
    else:
        assert response.status_code == 200
        assert response.json()["total_count"] == 2
        assert [note["id"] for note in response.json()["memories"]] == ["a"]
