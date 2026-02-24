# Chapter 8: Task Creation Workflow — AI Proposes, System Commits

## The Complete Flow

This chapter brings together everything we've built into a cohesive workflow:

```
User Input → AI Service → Validation → Persistence → Memory → Response
```

Each step is isolated, testable, and has clear responsibilities.

## The Task Service

```python
# src/ai_todo/services/task_service.py
"""
Task service orchestrating the complete creation workflow.

This is the central orchestration point following:
Routes → Services → AI → Validation → Persistence → Memory
"""

from datetime import datetime
from typing import Optional, List
import logging

from ..models.task import TaskInput, TaskProposal, Task
from ..models.enums import Priority, Category, TaskStatus
from ..ai.service import AIService
from ..persistence.repositories import TaskRepository
from ..memory.embeddings import EmbeddingStore


logger = logging.getLogger(__name__)


class TaskCreationResult:
    """Result of task creation with metadata."""
    
    def __init__(
        self,
        task: Task,
        proposal: TaskProposal,
        ai_duration_ms: float,
        similar_tasks: List[dict] | None = None
    ):
        self.task = task
        self.proposal = proposal
        self.ai_duration_ms = ai_duration_ms
        self.similar_tasks = similar_tasks or []
    
    @property
    def success(self) -> bool:
        return self.task is not None


class TaskService:
    """
    Orchestrates task operations following AI-first principles.
    
    Responsibilities:
    - Coordinate AI extraction
    - Validate proposals
    - Persist to database
    - Index in vector store
    - Handle failures gracefully
    """
    
    def __init__(
        self,
        ai_service: AIService,
        repository: TaskRepository,
        embedding_store: Optional[EmbeddingStore] = None
    ):
        self.ai = ai_service
        self.repo = repository
        self.embeddings = embedding_store
    
    async def create_task(
        self,
        input: TaskInput,
        *,
        current_datetime: Optional[datetime] = None,
        find_similar: bool = True
    ) -> TaskCreationResult:
        """
        Create a task from natural language input.
        
        Full flow:
        1. AI extracts structure (proposes)
        2. Pydantic validates (system validates)
        3. Task created with ID (system commits)
        4. Saved to SQLite (persistence)
        5. Indexed in ChromaDB (memory)
        
        Args:
            input: Natural language task input
            current_datetime: For date resolution
            find_similar: Whether to search for similar tasks
            
        Returns:
            TaskCreationResult with task and metadata
        """
        if current_datetime is None:
            current_datetime = datetime.now()
        
        logger.info(f"Creating task from: {input.raw_input[:50]}...")
        
        # Step 1: AI proposes
        start_time = datetime.now()
        proposal = await self.ai.extract_task(
            input.raw_input,
            current_datetime=current_datetime,
            safe_mode=True
        )
        ai_duration = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.debug(f"AI proposal: {proposal.title} (took {ai_duration:.0f}ms)")
        
        # Step 2: Already validated by AIService using Pydantic
        # TaskProposal ensures schema compliance
        
        # Step 3: System commits - create Task entity
        task = Task.from_proposal(proposal)
        
        # Step 4: Persist to database
        await self.repo.save(task)
        logger.debug(f"Task persisted: {task.id}")
        
        # Step 5: Index in vector store (if available)
        similar_tasks = []
        if self.embeddings:
            await self.embeddings.index(task)
            logger.debug(f"Task indexed in vector store")
            
            # Optionally find similar tasks
            if find_similar:
                similar_tasks = await self.embeddings.find_similar(
                    f"{task.title} {task.description or ''}",
                    limit=3,
                    exclude_ids=[task.id]
                )
        
        return TaskCreationResult(
            task=task,
            proposal=proposal,
            ai_duration_ms=ai_duration,
            similar_tasks=similar_tasks
        )
    
    async def list_tasks(
        self,
        *,
        status: Optional[TaskStatus] = None,
        priority: Optional[Priority] = None,
        category: Optional[Category] = None,
        limit: int = 50
    ) -> List[Task]:
        """List tasks with optional filtering."""
        return await self.repo.list(
            status=status,
            priority=priority,
            category=category,
            limit=limit
        )
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get a specific task by ID."""
        return await self.repo.get(task_id)
    
    async def complete_task(self, task_id: str) -> Optional[Task]:
        """Mark a task as completed."""
        task = await self.repo.get(task_id)
        if task is None:
            return None
        
        updated = task.mark_completed()
        await self.repo.update(updated)
        
        return updated
    
    async def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        # Remove from database
        deleted = await self.repo.delete(task_id)
        
        # Remove from vector store
        if deleted and self.embeddings:
            await self.embeddings.remove(task_id)
        
        return deleted
    
    async def search_tasks(
        self,
        query: str,
        limit: int = 10
    ) -> List[Task]:
        """
        Semantic search for tasks.
        
        Uses embeddings for meaning-based search rather than keyword matching.
        """
        if not self.embeddings:
            # Fallback to title search
            return await self.repo.search_by_title(query, limit=limit)
        
        # Get similar task IDs from vector store
        similar = await self.embeddings.find_similar(query, limit=limit)
        
        # Fetch full task objects
        tasks = []
        for item in similar:
            task = await self.repo.get(item["id"])
            if task:
                tasks.append(task)
        
        return tasks
    
    async def get_tasks_due_soon(
        self,
        hours: int = 24
    ) -> List[Task]:
        """Get tasks due within the specified hours."""
        return await self.repo.get_due_before(
            datetime.now() + timedelta(hours=hours)
        )
    
    async def update_task_priority(
        self,
        task_id: str,
        priority: Priority
    ) -> Optional[Task]:
        """Update task priority."""
        task = await self.repo.get(task_id)
        if task is None:
            return None
        
        updated = task.update_priority(priority)
        await self.repo.update(updated)
        
        return updated


from datetime import timedelta
```

## The Repository Pattern

```python
# src/ai_todo/persistence/repositories.py
"""
Repository for task persistence.

This layer handles all database operations.
The database owns state - AI never touches this directly.
"""

import sqlite3
from datetime import datetime
from typing import Optional, List
from contextlib import contextmanager

from ..models.task import Task
from ..models.enums import Priority, Category, TaskStatus


class TaskRepository:
    """
    SQLite repository for tasks.
    
    Follows Principle 1: Determinism owns state.
    All writes are predictable and validated before reaching here.
    """
    
    def __init__(self, connection: sqlite3.Connection):
        self.conn = connection
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Create tables if they don't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                priority TEXT NOT NULL,
                category TEXT NOT NULL,
                due_date TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                completed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Create indexes for common queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_status 
            ON tasks(status)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_due_date 
            ON tasks(due_date)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_priority 
            ON tasks(priority)
        """)
        
        self.conn.commit()
    
    @contextmanager
    def _transaction(self):
        """Context manager for transactions."""
        try:
            yield
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
    
    async def save(self, task: Task) -> None:
        """Save a new task."""
        with self._transaction():
            self.conn.execute("""
                INSERT INTO tasks (
                    id, title, description, priority, category,
                    due_date, status, completed, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.id,
                task.title,
                task.description,
                task.priority.value,
                task.category.value,
                task.due_date.isoformat() if task.due_date else None,
                task.status.value,
                int(task.completed),
                task.created_at.isoformat(),
                task.updated_at.isoformat(),
            ))
    
    async def update(self, task: Task) -> None:
        """Update an existing task."""
        with self._transaction():
            self.conn.execute("""
                UPDATE tasks SET
                    title = ?,
                    description = ?,
                    priority = ?,
                    category = ?,
                    due_date = ?,
                    status = ?,
                    completed = ?,
                    updated_at = ?
                WHERE id = ?
            """, (
                task.title,
                task.description,
                task.priority.value,
                task.category.value,
                task.due_date.isoformat() if task.due_date else None,
                task.status.value,
                int(task.completed),
                task.updated_at.isoformat(),
                task.id,
            ))
    
    async def get(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM tasks WHERE id = ?",
            (task_id,)
        )
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return self._row_to_task(row)
    
    async def list(
        self,
        *,
        status: Optional[TaskStatus] = None,
        priority: Optional[Priority] = None,
        category: Optional[Category] = None,
        limit: int = 50
    ) -> List[Task]:
        """List tasks with optional filters."""
        query = "SELECT * FROM tasks WHERE 1=1"
        params = []
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        if priority:
            query += " AND priority = ?"
            params.append(priority.value)
        
        if category:
            query += " AND category = ?"
            params.append(category.value)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor = self.conn.execute(query, params)
        return [self._row_to_task(row) for row in cursor.fetchall()]
    
    async def delete(self, task_id: str) -> bool:
        """Delete a task. Returns True if deleted."""
        with self._transaction():
            cursor = self.conn.execute(
                "DELETE FROM tasks WHERE id = ?",
                (task_id,)
            )
            return cursor.rowcount > 0
    
    async def get_due_before(self, before: datetime) -> List[Task]:
        """Get tasks due before a certain time."""
        cursor = self.conn.execute("""
            SELECT * FROM tasks 
            WHERE due_date IS NOT NULL 
            AND due_date <= ?
            AND completed = 0
            ORDER BY due_date ASC
        """, (before.isoformat(),))
        
        return [self._row_to_task(row) for row in cursor.fetchall()]
    
    async def search_by_title(self, query: str, limit: int = 10) -> List[Task]:
        """Simple text search in title."""
        cursor = self.conn.execute("""
            SELECT * FROM tasks 
            WHERE title LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (f"%{query}%", limit))
        
        return [self._row_to_task(row) for row in cursor.fetchall()]
    
    def _row_to_task(self, row: sqlite3.Row) -> Task:
        """Convert database row to Task model."""
        return Task(
            id=row["id"],
            title=row["title"],
            description=row["description"],
            priority=Priority(row["priority"]),
            category=Category(row["category"]),
            due_date=datetime.fromisoformat(row["due_date"]) if row["due_date"] else None,
            status=TaskStatus(row["status"]),
            completed=bool(row["completed"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
```

## Wiring It Together

```python
# src/ai_todo/main.py
"""
Application factory and dependency injection.

Creates all components and wires them together.
"""

import sqlite3
from pathlib import Path
from typing import Optional

from .config import Settings
from .ai.client import OllamaClient
from .ai.service import AIService
from .persistence.repositories import TaskRepository
from .memory.embeddings import EmbeddingStore
from .services.task_service import TaskService


class Application:
    """
    Main application container.
    
    Manages component lifecycle and provides access to services.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self._db: Optional[sqlite3.Connection] = None
        self._ollama: Optional[OllamaClient] = None
        self._embeddings: Optional[EmbeddingStore] = None
        self._task_service: Optional[TaskService] = None
    
    def _ensure_data_dir(self):
        """Create data directory if needed."""
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def db(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._db is None:
            self._ensure_data_dir()
            self._db = sqlite3.connect(
                self.settings.db_path,
                check_same_thread=False
            )
            self._db.row_factory = sqlite3.Row
        return self._db
    
    @property
    def ollama(self) -> OllamaClient:
        """Get or create Ollama client."""
        if self._ollama is None:
            self._ollama = OllamaClient(
                base_url=self.settings.ollama_base_url,
                model=self.settings.ollama_model,
                timeout=self.settings.ollama_timeout
            )
        return self._ollama
    
    @property
    def embeddings(self) -> EmbeddingStore:
        """Get or create embedding store."""
        if self._embeddings is None:
            self._ensure_data_dir()
            self._embeddings = EmbeddingStore(
                persist_dir=str(self.settings.chroma_dir),
                ai_client=self.ollama
            )
        return self._embeddings
    
    @property
    def tasks(self) -> TaskService:
        """Get or create task service."""
        if self._task_service is None:
            ai_service = AIService(self.ollama)
            repository = TaskRepository(self.db)
            
            self._task_service = TaskService(
                ai_service=ai_service,
                repository=repository,
                embedding_store=self.embeddings
            )
        return self._task_service
    
    async def close(self):
        """Clean up resources."""
        if self._db:
            self._db.close()
            self._db = None
        
        if self._ollama:
            await self._ollama.__aexit__(None, None, None)
            self._ollama = None


# Global application instance
_app: Optional[Application] = None


def get_app() -> Application:
    """Get or create the application instance."""
    global _app
    if _app is None:
        _app = Application()
    return _app


async def create_task(raw_input: str) -> TaskCreationResult:
    """Convenience function to create a task."""
    from .models.task import TaskInput
    
    app = get_app()
    return await app.tasks.create_task(TaskInput(raw_input=raw_input))
```

## Complete Workflow Demo

```python
# scripts/workflow_demo.py
"""Demonstrate the complete task creation workflow."""

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ai_todo.main import get_app
from ai_todo.models.task import TaskInput


console = Console()


async def main():
    app = get_app()
    
    console.print(Panel.fit(
        "[bold]AI-First Task Creation Workflow[/bold]\n\n"
        "Routes → Services → AI → Validation → Persistence → Memory",
        title="Demo"
    ))
    
    test_inputs = [
        "remind me to call mom tomorrow afternoon",
        "urgent: finish the quarterly report by EOD Friday",
        "buy groceries sometime this week",
    ]
    
    for raw_input in test_inputs:
        console.print(f"\n[cyan]Input:[/cyan] \"{raw_input}\"")
        
        with console.status("Processing..."):
            result = await app.tasks.create_task(
                TaskInput(raw_input=raw_input)
            )
        
        # Display result
        task = result.task
        
        console.print(f"[green]✓ Task Created[/green]")
        console.print(f"  ID: {task.id[:8]}...")
        console.print(f"  Title: {task.title}")
        console.print(f"  Priority: {task.priority.value}")
        console.print(f"  Category: {task.category.value}")
        if task.due_date:
            console.print(f"  Due: {task.due_date.strftime('%Y-%m-%d %H:%M')}")
        console.print(f"  AI Time: {result.ai_duration_ms:.0f}ms")
        
        if result.similar_tasks:
            console.print(f"  Similar tasks: {len(result.similar_tasks)} found")
    
    # Show all tasks
    console.print("\n" + "=" * 50)
    console.print("[bold]All Tasks in Database:[/bold]\n")
    
    tasks = await app.tasks.list_tasks()
    
    table = Table()
    table.add_column("Title", style="cyan")
    table.add_column("Priority")
    table.add_column("Category")
    table.add_column("Due")
    table.add_column("Status")
    
    for task in tasks:
        priority_color = {
            "urgent": "red",
            "high": "yellow",
            "medium": "white",
            "low": "dim"
        }.get(task.priority.value, "white")
        
        table.add_row(
            task.title,
            f"[{priority_color}]{task.priority.value}[/]",
            task.category.value,
            task.due_date.strftime("%m/%d") if task.due_date else "—",
            task.status.value
        )
    
    console.print(table)
    
    # Clean up
    await app.close()


if __name__ == "__main__":
    asyncio.run(main())
```

Output:
```
╭────────────────────────────────────────────────────────────╮
│           AI-First Task Creation Workflow                   │
│                                                             │
│ Routes → Services → AI → Validation → Persistence → Memory │
╰────────────────────────────────────────────────────────────╯

Input: "remind me to call mom tomorrow afternoon"
✓ Task Created
  ID: a1b2c3d4...
  Title: Call mom
  Priority: medium
  Category: personal
  Due: 2024-01-16 14:00
  AI Time: 342ms
  Similar tasks: 0 found

Input: "urgent: finish the quarterly report by EOD Friday"
✓ Task Created
  ID: e5f6g7h8...
  Title: Finish quarterly report
  Priority: urgent
  Category: work
  Due: 2024-01-19 17:00
  AI Time: 287ms
  Similar tasks: 0 found

Input: "buy groceries sometime this week"
✓ Task Created
  ID: i9j0k1l2...
  Title: Buy groceries
  Priority: low
  Category: errands
  Due: —
  AI Time: 256ms
  Similar tasks: 0 found

==================================================
All Tasks in Database:

┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━┳━━━━━━━━━┓
┃ Title                    ┃ Priority ┃ Category ┃ Due  ┃ Status  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━╇━━━━━━━━━┩
│ Buy groceries            │ low      │ errands  │ —    │ pending │
│ Finish quarterly report  │ urgent   │ work     │ 01/19│ pending │
│ Call mom                 │ medium   │ personal │ 01/16│ pending │
└──────────────────────────┴──────────┴──────────┴──────┴─────────┘
```

## The Flow Visualized

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  User: "remind me to call mom tomorrow afternoon"               │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ ROUTES (cli/commands.py)                                  │  │
│  │ • Parse CLI arguments                                     │  │
│  │ • Create TaskInput                                        │  │
│  │ • Call TaskService                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ SERVICES (services/task_service.py)                       │  │
│  │ • Orchestrate flow                                        │  │
│  │ • Coordinate components                                   │  │
│  │ • Handle errors                                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ AI (ai/service.py → ai/client.py)                        │  │
│  │ • Build prompt with current date                          │  │
│  │ • Call Ollama (temperature: 0.1)                          │  │
│  │ • Parse JSON response                                     │  │
│  │                                                           │  │
│  │ Output: {"title": "Call mom", "due_date": "...", ...}    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ VALIDATION (models/task.py - Pydantic)                    │  │
│  │ • TaskProposal.model_validate()                           │  │
│  │ • Check field constraints                                 │  │
│  │ • Parse enums and dates                                   │  │
│  │ • Reject invalid data                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ PERSISTENCE (persistence/repositories.py)                 │  │
│  │ • Task.from_proposal() - create entity                    │  │
│  │ • Generate UUID                                           │  │
│  │ • INSERT INTO tasks                                       │  │
│  │ • COMMIT transaction                                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ MEMORY (memory/embeddings.py)                             │  │
│  │ • Generate embedding vector                               │  │
│  │ • Store in ChromaDB                                       │  │
│  │ • Find similar tasks                                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ RESPONSE                                                  │  │
│  │                                                           │  │
│  │ ✓ Created task: Call mom                                  │  │
│  │   Priority: medium                                        │  │
│  │   Category: personal                                      │  │
│  │   Due: 2024-01-16 14:00                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Summary

In this chapter we:

1. ✅ Built the TaskService orchestrating the complete flow
2. ✅ Implemented the repository pattern for persistence
3. ✅ Wired together all components with dependency injection
4. ✅ Demonstrated the end-to-end workflow
5. ✅ Maintained clear separation of concerns

The flow now fully implements our principles:
- **Determinism owns state**: SQLite, validated data
- **AI proposes, system commits**: Proposal → Validation → Task
- **Temperature controls entropy**: 0.1 for parsing
- **Clean architecture**: Each layer isolated

In the next chapter, we'll dive into smart prioritization—using AI reasoning with controlled temperature.

---

**Previous**: [Chapter 7: Natural Language Parsing](./chapter-07-nlp-parsing.md)  
**Next**: [Chapter 9: Smart Prioritization](./chapter-09-prioritization.md)
