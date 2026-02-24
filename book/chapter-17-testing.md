# Chapter 17: Testing AI Systems — Determinism in Probabilistic Systems

## The Testing Paradox

AI systems are probabilistic. Tests require determinism. How do we test something that might give different answers each time?

The answer lies in our architecture: **AI proposes, the system commits**.

We test:
1. **Deterministic layers** — validation, persistence, business logic
2. **AI contracts** — structure of outputs, not exact content
3. **Integration seams** — using mocks and fixtures
4. **Behavior bounds** — AI outputs stay within expected ranges

## Testing Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                      TESTING PYRAMID                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                        ┌───────────┐                             │
│                        │  E2E Tests │  Few, slow, realistic      │
│                        └─────┬─────┘                             │
│                              │                                   │
│                    ┌─────────┴─────────┐                         │
│                    │ Integration Tests  │  AI mocked, seams tested│
│                    └─────────┬─────────┘                         │
│                              │                                   │
│         ┌────────────────────┴────────────────────┐              │
│         │            Unit Tests                    │  Fast,      │
│         │  Validation, Models, Repositories        │  deterministic│
│         └──────────────────────────────────────────┘              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Project Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── unit/
│   ├── __init__.py
│   ├── test_models.py       # Pydantic models
│   ├── test_validation.py   # Validation logic
│   ├── test_repository.py   # Database operations
│   └── test_date_parsing.py # Date service
├── integration/
│   ├── __init__.py
│   ├── test_task_service.py # Service with mocked AI
│   ├── test_rag_pipeline.py # RAG with fixtures
│   └── test_cli.py          # CLI commands
├── ai/
│   ├── __init__.py
│   ├── test_ai_contracts.py # AI output structure
│   └── test_ai_behavior.py  # Behavioral tests
└── fixtures/
    ├── ai_responses.json    # Canned AI responses
    └── sample_tasks.json    # Test data
```

## Fixtures and Configuration

```python
# tests/conftest.py
"""
Shared test fixtures and configuration.
"""

import pytest
import sqlite3
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from ai_todo.models.task import Task, TaskProposal
from ai_todo.models.enums import Priority, Category, TaskStatus
from ai_todo.ai.client import OllamaClient, AIResponse
from ai_todo.ai.service import AIService
from ai_todo.persistence.repositories import TaskRepository
from ai_todo.services.task_service import TaskService


# ==============================================================================
# EVENT LOOP
# ==============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ==============================================================================
# DATABASE FIXTURES
# ==============================================================================

@pytest.fixture
def temp_db():
    """Create temporary SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    yield conn
    
    conn.close()
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def task_repository(temp_db):
    """Create TaskRepository with temp database."""
    return TaskRepository(temp_db)


# ==============================================================================
# MOCK AI CLIENT
# ==============================================================================

@pytest.fixture
def mock_ai_client():
    """
    Create mock AI client that returns predictable responses.
    
    This is the key to testing AI-integrated code deterministically.
    """
    client = AsyncMock(spec=OllamaClient)
    
    # Default response
    client.generate.return_value = AIResponse(
        content='{"title": "Test task", "priority": "medium", "category": "work"}',
        model="test-model",
        tokens_used=10,
        duration_ms=100
    )
    
    return client


@pytest.fixture
def mock_ai_service(mock_ai_client):
    """Create AIService with mocked client."""
    return AIService(mock_ai_client)


# ==============================================================================
# SAMPLE DATA
# ==============================================================================

@pytest.fixture
def sample_task():
    """Create sample task for testing."""
    return Task(
        id="test-id-123",
        title="Test Task",
        description="A test task description",
        priority=Priority.MEDIUM,
        category=Category.WORK,
        due_date=datetime(2024, 1, 20, 17, 0),
        status=TaskStatus.PENDING,
        completed=False,
        created_at=datetime(2024, 1, 15, 10, 0),
        updated_at=datetime(2024, 1, 15, 10, 0)
    )


@pytest.fixture
def sample_proposal():
    """Create sample task proposal."""
    return TaskProposal(
        title="Test Task",
        description="A test task description",
        priority=Priority.MEDIUM,
        category=Category.WORK,
        due_date=datetime(2024, 1, 20, 17, 0)
    )


@pytest.fixture
def sample_tasks():
    """Create list of sample tasks."""
    base_time = datetime(2024, 1, 15, 10, 0)
    
    return [
        Task(
            id=f"task-{i}",
            title=f"Task {i}",
            priority=Priority.MEDIUM,
            category=Category.WORK,
            status=TaskStatus.PENDING,
            completed=False,
            created_at=base_time,
            updated_at=base_time
        )
        for i in range(5)
    ]


# ==============================================================================
# AI RESPONSE FIXTURES
# ==============================================================================

@pytest.fixture
def ai_response_fixture():
    """Factory for creating AI responses."""
    def _make_response(content: str, tokens: int = 50):
        return AIResponse(
            content=content,
            model="test-model",
            tokens_used=tokens,
            duration_ms=100
        )
    return _make_response


@pytest.fixture
def task_extraction_responses():
    """Canned AI responses for task extraction tests."""
    return {
        "simple": '{"title": "Buy groceries", "priority": "low", "category": "errands"}',
        "urgent": '{"title": "Fix production bug", "priority": "urgent", "category": "work"}',
        "with_date": '{"title": "Call mom", "priority": "medium", "category": "personal", "due_date": "2024-01-16T14:00:00"}',
        "invalid_json": 'This is not JSON',
        "missing_fields": '{"title": "Incomplete"}',
        "invalid_priority": '{"title": "Test", "priority": "super-urgent", "category": "work"}',
    }
```

## Unit Tests: Deterministic Layers

### Testing Models

```python
# tests/unit/test_models.py
"""
Unit tests for Pydantic models.

These tests are fully deterministic - no AI involved.
"""

import pytest
from datetime import datetime

from ai_todo.models.task import Task, TaskProposal, TaskInput
from ai_todo.models.enums import Priority, Category, TaskStatus


class TestTaskProposal:
    """Test TaskProposal validation."""
    
    def test_valid_proposal(self):
        """Valid proposal should pass."""
        proposal = TaskProposal(
            title="Test task",
            priority=Priority.HIGH,
            category=Category.WORK
        )
        
        assert proposal.title == "Test task"
        assert proposal.priority == Priority.HIGH
        assert proposal.category == Category.WORK
    
    def test_title_too_short(self):
        """Title under 2 chars should fail."""
        with pytest.raises(ValueError):
            TaskProposal(
                title="X",
                priority=Priority.MEDIUM,
                category=Category.WORK
            )
    
    def test_title_too_long(self):
        """Title over 200 chars should fail."""
        with pytest.raises(ValueError):
            TaskProposal(
                title="X" * 201,
                priority=Priority.MEDIUM,
                category=Category.WORK
            )
    
    def test_invalid_priority_string(self):
        """Invalid priority string should fail."""
        with pytest.raises(ValueError):
            TaskProposal(
                title="Test",
                priority="super-urgent",  # Not a valid priority
                category=Category.WORK
            )
    
    def test_priority_from_string(self):
        """Priority should accept valid string values."""
        proposal = TaskProposal(
            title="Test task",
            priority="urgent",
            category="work"
        )
        
        assert proposal.priority == Priority.URGENT
        assert proposal.category == Category.WORK
    
    def test_optional_description(self):
        """Description should be optional."""
        proposal = TaskProposal(
            title="Test",
            priority=Priority.LOW,
            category=Category.PERSONAL
        )
        
        assert proposal.description is None
    
    def test_optional_due_date(self):
        """Due date should be optional."""
        proposal = TaskProposal(
            title="Test",
            priority=Priority.LOW,
            category=Category.PERSONAL
        )
        
        assert proposal.due_date is None


class TestTask:
    """Test Task entity."""
    
    def test_from_proposal(self, sample_proposal):
        """Task should be created from proposal."""
        task = Task.from_proposal(sample_proposal)
        
        assert task.title == sample_proposal.title
        assert task.priority == sample_proposal.priority
        assert task.category == sample_proposal.category
        assert task.id is not None
        assert task.status == TaskStatus.PENDING
        assert task.completed is False
    
    def test_mark_completed(self, sample_task):
        """Marking complete should update status."""
        completed = sample_task.mark_completed()
        
        assert completed.completed is True
        assert completed.status == TaskStatus.COMPLETED
        assert completed.updated_at > sample_task.updated_at
    
    def test_immutability(self, sample_task):
        """Original task should not be modified."""
        original_completed = sample_task.completed
        _ = sample_task.mark_completed()
        
        assert sample_task.completed == original_completed


class TestTaskInput:
    """Test TaskInput validation."""
    
    def test_valid_input(self):
        """Valid input should pass."""
        input = TaskInput(raw_input="call mom tomorrow")
        assert input.raw_input == "call mom tomorrow"
    
    def test_empty_input(self):
        """Empty input should fail."""
        with pytest.raises(ValueError):
            TaskInput(raw_input="")
    
    def test_whitespace_only(self):
        """Whitespace-only input should fail."""
        with pytest.raises(ValueError):
            TaskInput(raw_input="   ")
```

### Testing Repository

```python
# tests/unit/test_repository.py
"""
Unit tests for TaskRepository.

Uses temporary SQLite database.
"""

import pytest
from datetime import datetime, timedelta

from ai_todo.models.task import Task
from ai_todo.models.enums import Priority, Category, TaskStatus


class TestTaskRepository:
    """Test repository operations."""
    
    @pytest.mark.asyncio
    async def test_save_and_get(self, task_repository, sample_task):
        """Should save and retrieve task."""
        await task_repository.save(sample_task)
        
        retrieved = await task_repository.get(sample_task.id)
        
        assert retrieved is not None
        assert retrieved.id == sample_task.id
        assert retrieved.title == sample_task.title
    
    @pytest.mark.asyncio
    async def test_get_nonexistent(self, task_repository):
        """Should return None for nonexistent task."""
        result = await task_repository.get("nonexistent-id")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_list_all(self, task_repository, sample_tasks):
        """Should list all tasks."""
        for task in sample_tasks:
            await task_repository.save(task)
        
        result = await task_repository.list()
        
        assert len(result) == len(sample_tasks)
    
    @pytest.mark.asyncio
    async def test_list_by_status(self, task_repository, sample_tasks):
        """Should filter by status."""
        # Save tasks
        for task in sample_tasks:
            await task_repository.save(task)
        
        # Complete one
        completed = sample_tasks[0].mark_completed()
        await task_repository.update(completed)
        
        # Filter pending
        pending = await task_repository.list(status=TaskStatus.PENDING)
        
        assert len(pending) == len(sample_tasks) - 1
    
    @pytest.mark.asyncio
    async def test_list_by_priority(self, task_repository):
        """Should filter by priority."""
        tasks = [
            Task(
                id=f"task-{i}",
                title=f"Task {i}",
                priority=Priority.URGENT if i == 0 else Priority.LOW,
                category=Category.WORK,
                status=TaskStatus.PENDING,
                completed=False,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            for i in range(3)
        ]
        
        for task in tasks:
            await task_repository.save(task)
        
        urgent = await task_repository.list(priority=Priority.URGENT)
        
        assert len(urgent) == 1
        assert urgent[0].priority == Priority.URGENT
    
    @pytest.mark.asyncio
    async def test_delete(self, task_repository, sample_task):
        """Should delete task."""
        await task_repository.save(sample_task)
        
        deleted = await task_repository.delete(sample_task.id)
        
        assert deleted is True
        assert await task_repository.get(sample_task.id) is None
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, task_repository):
        """Should return False for nonexistent delete."""
        result = await task_repository.delete("nonexistent-id")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_update(self, task_repository, sample_task):
        """Should update task."""
        await task_repository.save(sample_task)
        
        updated = sample_task.mark_completed()
        await task_repository.update(updated)
        
        retrieved = await task_repository.get(sample_task.id)
        
        assert retrieved.completed is True
    
    @pytest.mark.asyncio
    async def test_get_due_before(self, task_repository):
        """Should get tasks due before a date."""
        now = datetime.now()
        
        tasks = [
            Task(
                id="due-soon",
                title="Due soon",
                priority=Priority.HIGH,
                category=Category.WORK,
                due_date=now + timedelta(hours=1),
                status=TaskStatus.PENDING,
                completed=False,
                created_at=now,
                updated_at=now
            ),
            Task(
                id="due-later",
                title="Due later",
                priority=Priority.LOW,
                category=Category.WORK,
                due_date=now + timedelta(days=7),
                status=TaskStatus.PENDING,
                completed=False,
                created_at=now,
                updated_at=now
            ),
        ]
        
        for task in tasks:
            await task_repository.save(task)
        
        due_soon = await task_repository.get_due_before(now + timedelta(hours=2))
        
        assert len(due_soon) == 1
        assert due_soon[0].id == "due-soon"
```

### Testing Date Parsing

```python
# tests/unit/test_date_parsing.py
"""
Unit tests for DateService parsing.

Rule-based parsing is deterministic and testable.
"""

import pytest
from datetime import datetime, timedelta

from ai_todo.services.date_service import DateService


class TestDateParsing:
    """Test deterministic date parsing."""
    
    @pytest.fixture
    def date_service(self):
        return DateService()
    
    @pytest.fixture
    def reference_time(self):
        # Monday, January 15, 2024, 10:00 AM
        return datetime(2024, 1, 15, 10, 0, 0)
    
    # ========================================
    # RELATIVE DATES
    # ========================================
    
    def test_parse_today(self, date_service, reference_time):
        """Should parse 'today'."""
        result = date_service.parse("finish report today", reference_time)
        
        assert result.date is not None
        assert result.date.date() == reference_time.date()
        assert result.method == "relative"
        assert result.confidence >= 0.8
    
    def test_parse_tomorrow(self, date_service, reference_time):
        """Should parse 'tomorrow'."""
        result = date_service.parse("call mom tomorrow", reference_time)
        
        expected = reference_time + timedelta(days=1)
        assert result.date.date() == expected.date()
    
    def test_parse_in_n_days(self, date_service, reference_time):
        """Should parse 'in N days'."""
        result = date_service.parse("in 3 days", reference_time)
        
        expected = reference_time + timedelta(days=3)
        assert result.date.date() == expected.date()
    
    def test_parse_next_week(self, date_service, reference_time):
        """Should parse 'next week'."""
        result = date_service.parse("next week", reference_time)
        
        expected = reference_time + timedelta(weeks=1)
        assert result.date.date() == expected.date()
    
    # ========================================
    # WEEKDAYS
    # ========================================
    
    def test_parse_weekday(self, date_service, reference_time):
        """Should parse weekday name."""
        # Reference is Monday
        result = date_service.parse("by Friday", reference_time)
        
        # Should be this Friday (Jan 19)
        assert result.date.weekday() == 4  # Friday
        assert result.date.date() == datetime(2024, 1, 19).date()
    
    def test_parse_next_weekday(self, date_service, reference_time):
        """Should parse 'next' weekday."""
        result = date_service.parse("next Monday", reference_time)
        
        # Should be next Monday (Jan 22), not today
        assert result.date.weekday() == 0  # Monday
        assert result.date > reference_time
    
    # ========================================
    # TIME OF DAY
    # ========================================
    
    def test_parse_afternoon(self, date_service, reference_time):
        """Should apply 'afternoon' modifier."""
        result = date_service.parse("tomorrow afternoon", reference_time)
        
        assert result.date.hour == 14  # 2 PM
    
    def test_parse_morning(self, date_service, reference_time):
        """Should apply 'morning' modifier."""
        result = date_service.parse("tomorrow morning", reference_time)
        
        assert result.date.hour == 9
    
    def test_parse_eod(self, date_service, reference_time):
        """Should apply 'EOD' modifier."""
        result = date_service.parse("EOD today", reference_time)
        
        assert result.date.hour == 17  # 5 PM
    
    def test_parse_explicit_time(self, date_service, reference_time):
        """Should parse explicit time."""
        result = date_service.parse("tomorrow at 3pm", reference_time)
        
        assert result.date.hour == 15
    
    # ========================================
    # EXPLICIT DATES
    # ========================================
    
    def test_parse_iso_date(self, date_service, reference_time):
        """Should parse ISO date format."""
        result = date_service.parse("by 2024-01-25", reference_time)
        
        assert result.date.date() == datetime(2024, 1, 25).date()
        assert result.method == "explicit"
        assert result.confidence >= 0.9
    
    def test_parse_month_day(self, date_service, reference_time):
        """Should parse 'Month Day' format."""
        result = date_service.parse("by January 25", reference_time)
        
        assert result.date.month == 1
        assert result.date.day == 25
    
    # ========================================
    # NO DATE
    # ========================================
    
    def test_no_date_found(self, date_service, reference_time):
        """Should return None when no date found."""
        result = date_service.parse("buy groceries", reference_time)
        
        assert result.date is None
        assert result.method == "none"
        assert result.confidence == 0.0
```

## Integration Tests: Mocked AI

```python
# tests/integration/test_task_service.py
"""
Integration tests for TaskService.

AI is mocked to ensure deterministic behavior.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock

from ai_todo.models.task import TaskInput
from ai_todo.services.task_service import TaskService
from ai_todo.ai.client import AIResponse


class TestTaskServiceIntegration:
    """Test TaskService with mocked AI."""
    
    @pytest.fixture
    def task_service(self, mock_ai_service, task_repository):
        """Create TaskService with mocked AI."""
        return TaskService(
            ai_service=mock_ai_service,
            repository=task_repository,
            embedding_store=None
        )
    
    @pytest.mark.asyncio
    async def test_create_task_full_flow(
        self,
        task_service,
        mock_ai_client,
        task_extraction_responses
    ):
        """Test complete task creation flow."""
        # Configure mock
        mock_ai_client.generate.return_value = AIResponse(
            content=task_extraction_responses["with_date"],
            model="test",
            tokens_used=50,
            duration_ms=100
        )
        
        # Create task
        result = await task_service.create_task(
            TaskInput(raw_input="call mom tomorrow afternoon")
        )
        
        # Verify
        assert result.success
        assert result.task.title == "Call mom"
        assert result.task.priority.value == "medium"
        assert result.task.category.value == "personal"
        assert result.ai_duration_ms > 0
    
    @pytest.mark.asyncio
    async def test_create_task_urgent(
        self,
        task_service,
        mock_ai_client,
        task_extraction_responses
    ):
        """Test urgent task creation."""
        mock_ai_client.generate.return_value = AIResponse(
            content=task_extraction_responses["urgent"],
            model="test",
            tokens_used=50,
            duration_ms=100
        )
        
        result = await task_service.create_task(
            TaskInput(raw_input="urgent: fix production bug")
        )
        
        assert result.task.priority.value == "urgent"
        assert result.task.category.value == "work"
    
    @pytest.mark.asyncio
    async def test_task_persisted(
        self,
        task_service,
        task_repository,
        mock_ai_client,
        task_extraction_responses
    ):
        """Verify task is persisted to database."""
        mock_ai_client.generate.return_value = AIResponse(
            content=task_extraction_responses["simple"],
            model="test",
            tokens_used=50,
            duration_ms=100
        )
        
        result = await task_service.create_task(
            TaskInput(raw_input="buy groceries")
        )
        
        # Verify in database
        persisted = await task_repository.get(result.task.id)
        
        assert persisted is not None
        assert persisted.title == result.task.title
    
    @pytest.mark.asyncio
    async def test_complete_task(self, task_service, mock_ai_client, task_extraction_responses):
        """Test task completion."""
        mock_ai_client.generate.return_value = AIResponse(
            content=task_extraction_responses["simple"],
            model="test",
            tokens_used=50,
            duration_ms=100
        )
        
        # Create
        result = await task_service.create_task(
            TaskInput(raw_input="test task")
        )
        
        # Complete
        completed = await task_service.complete_task(result.task.id)
        
        assert completed.completed is True
    
    @pytest.mark.asyncio
    async def test_list_tasks(self, task_service, mock_ai_client, task_extraction_responses):
        """Test task listing."""
        mock_ai_client.generate.return_value = AIResponse(
            content=task_extraction_responses["simple"],
            model="test",
            tokens_used=50,
            duration_ms=100
        )
        
        # Create multiple
        for i in range(3):
            await task_service.create_task(
                TaskInput(raw_input=f"task {i}")
            )
        
        # List
        tasks = await task_service.list_tasks()
        
        assert len(tasks) == 3
```

## AI Contract Tests

```python
# tests/ai/test_ai_contracts.py
"""
Tests for AI output contracts.

These tests verify that AI outputs conform to expected structure,
even when content varies.
"""

import pytest
import json
from ai_todo.ai.parsers import parse_json_safe, extract_task_from_response
from ai_todo.models.task import TaskProposal


class TestAIContracts:
    """Test AI output structure contracts."""
    
    def test_parse_json_safe_valid(self):
        """Should parse valid JSON."""
        content = '{"title": "Test", "priority": "high"}'
        result = parse_json_safe(content, {})
        
        assert result["title"] == "Test"
        assert result["priority"] == "high"
    
    def test_parse_json_safe_with_markdown(self):
        """Should handle markdown code blocks."""
        content = '''Here's the JSON:
        ```json
        {"title": "Test", "priority": "medium"}
        ```
        '''
        result = parse_json_safe(content, {})
        
        assert result["title"] == "Test"
    
    def test_parse_json_safe_invalid(self):
        """Should return default for invalid JSON."""
        content = "This is not JSON at all"
        default = {"error": True}
        
        result = parse_json_safe(content, default)
        
        assert result == default
    
    def test_task_proposal_contract(self):
        """Verify TaskProposal accepts valid AI output."""
        # Typical AI output
        ai_output = {
            "title": "Review quarterly report",
            "description": "Check numbers and formatting",
            "priority": "high",
            "category": "work",
            "due_date": "2024-01-20T17:00:00"
        }
        
        # Should not raise
        proposal = TaskProposal.model_validate(ai_output)
        
        assert proposal.title == ai_output["title"]
    
    def test_task_proposal_minimal(self):
        """TaskProposal should accept minimal output."""
        ai_output = {
            "title": "Simple task",
            "priority": "low",
            "category": "personal"
        }
        
        proposal = TaskProposal.model_validate(ai_output)
        
        assert proposal.description is None
        assert proposal.due_date is None
    
    def test_task_proposal_rejects_invalid_priority(self):
        """TaskProposal should reject invalid priority."""
        ai_output = {
            "title": "Test",
            "priority": "super-urgent",  # Invalid
            "category": "work"
        }
        
        with pytest.raises(ValueError):
            TaskProposal.model_validate(ai_output)
    
    def test_task_proposal_normalizes_case(self):
        """TaskProposal should handle case variations."""
        ai_output = {
            "title": "Test",
            "priority": "HIGH",  # Uppercase
            "category": "WORK"
        }
        
        # Should normalize to lowercase
        proposal = TaskProposal.model_validate(ai_output)
        
        assert proposal.priority.value == "high"
```

## Behavioral Bounds Tests

```python
# tests/ai/test_ai_behavior.py
"""
Behavioral tests for AI outputs.

These tests verify AI outputs stay within expected bounds,
even with real AI calls (for CI/CD pipelines).
"""

import pytest
from datetime import datetime, timedelta

from ai_todo.ai.client import OllamaClient
from ai_todo.ai.service import AIService
from ai_todo.models.enums import Priority, Category


# Skip if Ollama not available
pytestmark = pytest.mark.skipif(
    not pytest.importorskip("httpx"),
    reason="Requires Ollama running"
)


class TestAIBehavioralBounds:
    """
    Test AI behavior stays within bounds.
    
    These are slower tests that use real AI.
    Run with: pytest -m "ai_behavior" --slow
    """
    
    @pytest.fixture
    async def ai_service(self):
        """Create real AI service."""
        client = OllamaClient()
        await client.__aenter__()
        service = AIService(client)
        yield service
        await client.__aexit__(None, None, None)
    
    @pytest.mark.slow
    @pytest.mark.ai_behavior
    @pytest.mark.asyncio
    async def test_priority_extraction_bounds(self, ai_service):
        """AI should always return valid priority."""
        test_cases = [
            "urgent task asap",
            "do this sometime",
            "important meeting",
            "maybe later",
            "CRITICAL BUG FIX NOW"
        ]
        
        for text in test_cases:
            proposal = await ai_service.extract_task(text)
            
            # Priority must be in valid enum
            assert proposal.priority in Priority
    
    @pytest.mark.slow
    @pytest.mark.ai_behavior
    @pytest.mark.asyncio
    async def test_category_extraction_bounds(self, ai_service):
        """AI should always return valid category."""
        test_cases = [
            "buy groceries",
            "finish report",
            "go to gym",
            "pay bills",
            "learn python"
        ]
        
        for text in test_cases:
            proposal = await ai_service.extract_task(text)
            
            # Category must be in valid enum
            assert proposal.category in Category
    
    @pytest.mark.slow
    @pytest.mark.ai_behavior
    @pytest.mark.asyncio
    async def test_title_extraction_quality(self, ai_service):
        """AI should extract reasonable titles."""
        test_cases = [
            ("remind me to call mom tomorrow", ["call", "mom"]),
            ("urgent fix the login bug", ["login", "bug"]),
            ("buy milk and eggs", ["buy", "milk"]),
        ]
        
        for text, expected_keywords in test_cases:
            proposal = await ai_service.extract_task(text)
            
            title_lower = proposal.title.lower()
            
            # At least one keyword should appear
            assert any(kw in title_lower for kw in expected_keywords), \
                f"Title '{proposal.title}' missing keywords {expected_keywords}"
    
    @pytest.mark.slow
    @pytest.mark.ai_behavior
    @pytest.mark.asyncio
    async def test_due_date_reasonableness(self, ai_service):
        """AI-suggested due dates should be reasonable."""
        now = datetime.now()
        
        proposal = await ai_service.extract_task(
            "finish report by tomorrow",
            current_datetime=now
        )
        
        if proposal.due_date:
            # Should not be in the past
            assert proposal.due_date >= now - timedelta(hours=1)
            
            # Should not be more than a year out
            assert proposal.due_date <= now + timedelta(days=365)
```

## Running Tests

```bash
# Run all unit tests (fast, deterministic)
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Run with coverage
pytest tests/ --cov=ai_todo --cov-report=html

# Run AI behavioral tests (requires Ollama)
pytest tests/ai/ -m ai_behavior --slow

# Run specific test
pytest tests/unit/test_models.py::TestTaskProposal::test_valid_proposal -v
```

## pytest Configuration

```ini
# pytest.ini
[pytest]
asyncio_mode = auto
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    ai_behavior: tests that require real AI (deselect with '-m "not ai_behavior"')

testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

## Summary

In this chapter we built:

1. **Test fixtures** for database, AI mocks, and sample data
2. **Unit tests** for deterministic layers (models, repository, parsing)
3. **Integration tests** with mocked AI for service layer
4. **Contract tests** verifying AI output structure
5. **Behavioral bounds tests** ensuring AI stays within limits

The key insight: **test determinism, bound randomness**.

- Deterministic code (validation, persistence) gets traditional unit tests
- AI integration gets mocked for speed and determinism
- AI behavior gets bounded tests that verify outputs stay valid

---

**Previous**: [Chapter 16: CLI Interface](./chapter-16-cli.md)  
**Next**: [Chapter 18: Error Handling](./chapter-18-error-handling.md)
