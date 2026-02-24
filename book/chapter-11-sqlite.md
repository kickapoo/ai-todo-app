# Chapter 11: SQLite Persistence — Determinism Owns State

## The Principle

Recall Principle 1: **Determinism Owns State**.

Authentication, validation, and database writes must remain **predictable and stable**. The database is the source of truth—not the AI, not the cache, not the embeddings.

SQLite provides:
- ACID transactions
- Predictable behavior
- Zero configuration
- Single-file portability

## Database Design

### Schema

```sql
-- schema.sql

-- Core tasks table
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    priority TEXT NOT NULL CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
    category TEXT NOT NULL,
    due_date TEXT,  -- ISO 8601 format
    status TEXT NOT NULL DEFAULT 'pending' 
        CHECK (status IN ('pending', 'in_progress', 'completed', 'cancelled')),
    completed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority);
CREATE INDEX IF NOT EXISTS idx_tasks_due_date ON tasks(due_date);
CREATE INDEX IF NOT EXISTS idx_tasks_category ON tasks(category);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at);

-- Task history for audit trail
CREATE TABLE IF NOT EXISTS task_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    field_name TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    changed_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_task_history_task_id ON task_history(task_id);

-- Metadata table for app settings
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

## Database Manager

```python
# src/ai_todo/persistence/database.py
"""
Database management for SQLite.

Handles connection, migrations, and transactions.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, List
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)


class Database:
    """
    SQLite database manager.
    
    Provides connection management, migrations, and transaction support.
    """
    
    SCHEMA_VERSION = 1
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None
    
    @property
    def connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._connection is None:
            self._connection = self._create_connection()
            self._initialize_schema()
        return self._connection
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with proper settings."""
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,  # We handle thread safety
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Use WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode = WAL")
        
        # Row factory for dict-like access
        conn.row_factory = sqlite3.Row
        
        logger.info(f"Connected to database: {self.db_path}")
        return conn
    
    def _initialize_schema(self):
        """Initialize database schema."""
        conn = self._connection
        
        # Check current version
        try:
            cursor = conn.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            )
            row = cursor.fetchone()
            current_version = int(row["value"]) if row else 0
        except sqlite3.OperationalError:
            current_version = 0
        
        if current_version < self.SCHEMA_VERSION:
            self._run_migrations(current_version)
    
    def _run_migrations(self, from_version: int):
        """Run database migrations."""
        conn = self._connection
        
        if from_version < 1:
            # Initial schema
            conn.executescript("""
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
                );
                
                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
                CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority);
                CREATE INDEX IF NOT EXISTS idx_tasks_due_date ON tasks(due_date);
                
                CREATE TABLE IF NOT EXISTS task_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    changed_at TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
                );
                
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)
            
            # Set schema version
            now = datetime.utcnow().isoformat()
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value, updated_at) VALUES (?, ?, ?)",
                ("schema_version", str(self.SCHEMA_VERSION), now)
            )
            conn.commit()
            
            logger.info(f"Database migrated to version {self.SCHEMA_VERSION}")
    
    @contextmanager
    def transaction(self):
        """
        Context manager for transactions.
        
        Usage:
            with db.transaction():
                db.execute("INSERT ...")
                db.execute("UPDATE ...")
            # Auto-commit on success, rollback on exception
        """
        try:
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
    
    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute SQL and return cursor."""
        return self.connection.execute(sql, params)
    
    def executemany(self, sql: str, params_list: List[tuple]) -> sqlite3.Cursor:
        """Execute SQL for multiple parameter sets."""
        return self.connection.executemany(sql, params_list)
    
    def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
```

## Repository Implementation

```python
# src/ai_todo/persistence/repositories.py
"""
Repository pattern for data access.

Provides clean interface between domain and database.
"""

from datetime import datetime
from typing import Optional, List
import logging

from .database import Database
from ..models.task import Task
from ..models.enums import Priority, Category, TaskStatus

logger = logging.getLogger(__name__)


class TaskRepository:
    """
    Repository for Task persistence.
    
    All database operations go through this class.
    Ensures consistent data handling and query patterns.
    """
    
    def __init__(self, database: Database):
        self.db = database
    
    async def save(self, task: Task) -> None:
        """
        Save a new task to the database.
        
        Uses INSERT - fails if task already exists.
        """
        with self.db.transaction():
            self.db.execute("""
                INSERT INTO tasks (
                    id, title, description, priority, category,
                    due_date, status, completed, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, self._task_to_tuple(task))
        
        logger.debug(f"Saved task: {task.id}")
    
    async def update(self, task: Task) -> None:
        """
        Update an existing task.
        
        Records changes in history table.
        """
        # Get current state for history
        current = await self.get(task.id)
        
        with self.db.transaction():
            # Update task
            self.db.execute("""
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
                task.id
            ))
            
            # Record history if changed
            if current:
                self._record_history(current, task)
        
        logger.debug(f"Updated task: {task.id}")
    
    async def get(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        cursor = self.db.execute(
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
        completed: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "created_at",
        order_dir: str = "DESC"
    ) -> List[Task]:
        """
        List tasks with filtering and pagination.
        
        Args:
            status: Filter by status
            priority: Filter by priority
            category: Filter by category
            completed: Filter by completion
            limit: Maximum results
            offset: Results offset for pagination
            order_by: Column to order by
            order_dir: ASC or DESC
        """
        # Build query dynamically
        conditions = []
        params = []
        
        if status is not None:
            conditions.append("status = ?")
            params.append(status.value)
        
        if priority is not None:
            conditions.append("priority = ?")
            params.append(priority.value)
        
        if category is not None:
            conditions.append("category = ?")
            params.append(category.value)
        
        if completed is not None:
            conditions.append("completed = ?")
            params.append(int(completed))
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # Validate order_by to prevent SQL injection
        allowed_columns = {"created_at", "updated_at", "due_date", "priority", "title"}
        if order_by not in allowed_columns:
            order_by = "created_at"
        
        order_dir = "DESC" if order_dir.upper() == "DESC" else "ASC"
        
        query = f"""
            SELECT * FROM tasks 
            WHERE {where_clause}
            ORDER BY {order_by} {order_dir}
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        
        cursor = self.db.execute(query, tuple(params))
        return [self._row_to_task(row) for row in cursor.fetchall()]
    
    async def delete(self, task_id: str) -> bool:
        """
        Delete a task.
        
        Returns True if task was deleted, False if not found.
        """
        with self.db.transaction():
            cursor = self.db.execute(
                "DELETE FROM tasks WHERE id = ?",
                (task_id,)
            )
            deleted = cursor.rowcount > 0
        
        if deleted:
            logger.debug(f"Deleted task: {task_id}")
        
        return deleted
    
    async def get_due_before(self, before: datetime) -> List[Task]:
        """Get incomplete tasks due before a certain time."""
        cursor = self.db.execute("""
            SELECT * FROM tasks
            WHERE due_date IS NOT NULL
            AND due_date <= ?
            AND completed = 0
            ORDER BY due_date ASC
        """, (before.isoformat(),))
        
        return [self._row_to_task(row) for row in cursor.fetchall()]
    
    async def get_overdue(self) -> List[Task]:
        """Get all overdue incomplete tasks."""
        return await self.get_due_before(datetime.now())
    
    async def search_by_title(self, query: str, limit: int = 10) -> List[Task]:
        """Simple text search in title."""
        cursor = self.db.execute("""
            SELECT * FROM tasks
            WHERE title LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (f"%{query}%", limit))
        
        return [self._row_to_task(row) for row in cursor.fetchall()]
    
    async def count(
        self,
        *,
        status: Optional[TaskStatus] = None,
        completed: Optional[bool] = None
    ) -> int:
        """Count tasks with optional filters."""
        conditions = []
        params = []
        
        if status is not None:
            conditions.append("status = ?")
            params.append(status.value)
        
        if completed is not None:
            conditions.append("completed = ?")
            params.append(int(completed))
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        cursor = self.db.execute(
            f"SELECT COUNT(*) as count FROM tasks WHERE {where_clause}",
            tuple(params)
        )
        return cursor.fetchone()["count"]
    
    async def get_history(self, task_id: str) -> List[dict]:
        """Get change history for a task."""
        cursor = self.db.execute("""
            SELECT * FROM task_history
            WHERE task_id = ?
            ORDER BY changed_at DESC
        """, (task_id,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def _record_history(self, old: Task, new: Task) -> None:
        """Record changes between old and new task state."""
        now = datetime.utcnow().isoformat()
        changes = []
        
        if old.title != new.title:
            changes.append(("title", old.title, new.title))
        if old.priority != new.priority:
            changes.append(("priority", old.priority.value, new.priority.value))
        if old.category != new.category:
            changes.append(("category", old.category.value, new.category.value))
        if old.status != new.status:
            changes.append(("status", old.status.value, new.status.value))
        if old.completed != new.completed:
            changes.append(("completed", str(old.completed), str(new.completed)))
        
        for field, old_val, new_val in changes:
            self.db.execute("""
                INSERT INTO task_history (task_id, field_name, old_value, new_value, changed_at)
                VALUES (?, ?, ?, ?, ?)
            """, (new.id, field, old_val, new_val, now))
    
    def _task_to_tuple(self, task: Task) -> tuple:
        """Convert Task to tuple for INSERT."""
        return (
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
        )
    
    def _row_to_task(self, row: dict) -> Task:
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

## Using the Repository

```python
# Example usage in task service

class TaskService:
    def __init__(self, repository: TaskRepository, ...):
        self.repo = repository
    
    async def create_task(self, input: TaskInput) -> Task:
        # ... AI processing and validation ...
        
        # Deterministic: create task with UUID
        task = Task.from_proposal(proposal)
        
        # Deterministic: save to database
        await self.repo.save(task)
        
        return task
    
    async def get_dashboard_data(self) -> dict:
        """Get task statistics for dashboard."""
        return {
            "total": await self.repo.count(),
            "pending": await self.repo.count(status=TaskStatus.PENDING),
            "completed": await self.repo.count(completed=True),
            "overdue": len(await self.repo.get_overdue()),
        }
```

## Backup and Recovery

```python
# src/ai_todo/persistence/backup.py
"""Database backup utilities."""

import shutil
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BackupManager:
    """Manage database backups."""
    
    def __init__(self, db_path: Path, backup_dir: Path):
        self.db_path = db_path
        self.backup_dir = backup_dir
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self, suffix: str = "") -> Path:
        """Create a backup of the database."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"tasks_{timestamp}{suffix}.db"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(self.db_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        
        return backup_path
    
    def restore_backup(self, backup_path: Path) -> None:
        """Restore database from backup."""
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        
        # Create safety backup of current state
        self.create_backup(suffix="_before_restore")
        
        # Restore
        shutil.copy2(backup_path, self.db_path)
        logger.info(f"Restored from backup: {backup_path}")
    
    def list_backups(self) -> list[Path]:
        """List available backups."""
        return sorted(
            self.backup_dir.glob("tasks_*.db"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
    
    def cleanup_old_backups(self, keep: int = 10) -> int:
        """Remove old backups, keeping the most recent."""
        backups = self.list_backups()
        to_delete = backups[keep:]
        
        for backup in to_delete:
            backup.unlink()
            logger.debug(f"Deleted old backup: {backup}")
        
        return len(to_delete)
```

## Summary

In this chapter we implemented:

1. ✅ SQLite database with proper schema
2. ✅ Migration system for schema evolution
3. ✅ Repository pattern for clean data access
4. ✅ Transaction support for consistency
5. ✅ Change history for audit trail
6. ✅ Backup and recovery utilities

This implements Principle 1: **Determinism Owns State**.

Key properties:
- **ACID transactions** - data is never in inconsistent state
- **Schema enforcement** - CHECK constraints on valid values
- **Audit trail** - history table tracks all changes
- **Single source of truth** - SQLite is authoritative

In the next chapter, we'll add ChromaDB for semantic memory—implementing Principle 4.

---

**Previous**: [Chapter 10: Validation Patterns](./chapter-10-validation.md)  
**Next**: [Chapter 12: ChromaDB Embeddings](./chapter-12-chromadb.md)
