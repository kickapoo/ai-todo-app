# Chapter 6: The Task Data Model — Pydantic Schemas

## The Role of Models

Models are the **contract** between all layers of our system. They define:
- What data looks like
- What constraints apply
- How validation happens

In AI-first systems, models are especially critical because they're the boundary between probabilistic AI output and deterministic system behavior.

## The Model Hierarchy

We define three levels of task representation:

```
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL HIERARCHY                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   TaskInput                                                      │
│   ┌─────────────────────────────────────────┐                   │
│   │ raw_input: str                          │  ← User provides  │
│   │ "call mom tomorrow at 3pm"              │                   │
│   └──────────────────┬──────────────────────┘                   │
│                      │                                           │
│                      │ AI Processing                             │
│                      ▼                                           │
│   TaskProposal                                                   │
│   ┌─────────────────────────────────────────┐                   │
│   │ title: str                              │  ← AI generates   │
│   │ priority: Priority                      │    (untrusted)    │
│   │ category: Category                      │                   │
│   │ due_date: datetime | None               │                   │
│   └──────────────────┬──────────────────────┘                   │
│                      │                                           │
│                      │ Validation                                │
│                      ▼                                           │
│   Task                                                           │
│   ┌─────────────────────────────────────────┐                   │
│   │ id: str                                 │  ← System creates │
│   │ title: str                              │    (trusted)      │
│   │ priority: Priority                      │                   │
│   │ category: Category                      │                   │
│   │ due_date: datetime | None               │                   │
│   │ completed: bool                         │                   │
│   │ created_at: datetime                    │                   │
│   │ updated_at: datetime                    │                   │
│   └─────────────────────────────────────────┘                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Defining Enums

First, let's define our constrained choices:

```python
# src/ai_todo/models/enums.py
"""Enum definitions for task properties."""

from enum import Enum


class Priority(str, Enum):
    """Task priority levels.
    
    Inheriting from str makes JSON serialization seamless.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    
    @classmethod
    def from_string(cls, value: str) -> "Priority":
        """Parse priority from various string representations."""
        normalized = value.lower().strip()
        
        # Handle common variations
        mapping = {
            "low": cls.LOW,
            "l": cls.LOW,
            "1": cls.LOW,
            "medium": cls.MEDIUM,
            "med": cls.MEDIUM,
            "m": cls.MEDIUM,
            "2": cls.MEDIUM,
            "normal": cls.MEDIUM,
            "high": cls.HIGH,
            "h": cls.HIGH,
            "3": cls.HIGH,
            "important": cls.HIGH,
            "urgent": cls.URGENT,
            "u": cls.URGENT,
            "4": cls.URGENT,
            "critical": cls.URGENT,
            "asap": cls.URGENT,
        }
        
        if normalized in mapping:
            return mapping[normalized]
        
        return cls.MEDIUM  # Default


class Category(str, Enum):
    """Task category for organization."""
    WORK = "work"
    PERSONAL = "personal"
    HEALTH = "health"
    FINANCE = "finance"
    LEARNING = "learning"
    ERRANDS = "errands"
    HOME = "home"
    SOCIAL = "social"
    OTHER = "other"
    
    @classmethod
    def from_string(cls, value: str) -> "Category":
        """Parse category from various string representations."""
        normalized = value.lower().strip()
        
        mapping = {
            # Work
            "work": cls.WORK,
            "job": cls.WORK,
            "office": cls.WORK,
            "professional": cls.WORK,
            "business": cls.WORK,
            
            # Personal
            "personal": cls.PERSONAL,
            "family": cls.PERSONAL,
            "self": cls.PERSONAL,
            
            # Health
            "health": cls.HEALTH,
            "medical": cls.HEALTH,
            "fitness": cls.HEALTH,
            "exercise": cls.HEALTH,
            "gym": cls.HEALTH,
            "doctor": cls.HEALTH,
            
            # Finance
            "finance": cls.FINANCE,
            "financial": cls.FINANCE,
            "money": cls.FINANCE,
            "bills": cls.FINANCE,
            "payment": cls.FINANCE,
            
            # Learning
            "learning": cls.LEARNING,
            "education": cls.LEARNING,
            "study": cls.LEARNING,
            "course": cls.LEARNING,
            "reading": cls.LEARNING,
            
            # Errands
            "errands": cls.ERRANDS,
            "shopping": cls.ERRANDS,
            "groceries": cls.ERRANDS,
            "chores": cls.ERRANDS,
            
            # Home
            "home": cls.HOME,
            "house": cls.HOME,
            "maintenance": cls.HOME,
            "cleaning": cls.HOME,
            
            # Social
            "social": cls.SOCIAL,
            "friends": cls.SOCIAL,
            "events": cls.SOCIAL,
            "party": cls.SOCIAL,
        }
        
        if normalized in mapping:
            return mapping[normalized]
        
        return cls.OTHER


class TaskStatus(str, Enum):
    """Task completion status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
```

## Input Model

What the user provides:

```python
# src/ai_todo/models/task.py
"""Task models following the three-tier hierarchy."""

from datetime import datetime
from typing import Optional, Annotated
from pydantic import BaseModel, Field, field_validator
import uuid

from .enums import Priority, Category, TaskStatus


class TaskInput(BaseModel):
    """
    Raw user input for task creation.
    
    This is the entry point - natural language from the user.
    Minimal validation here; the AI will parse this.
    """
    raw_input: Annotated[
        str,
        Field(
            min_length=1,
            max_length=1000,
            description="Natural language task description"
        )
    ]
    
    @field_validator("raw_input")
    @classmethod
    def clean_input(cls, v: str) -> str:
        """Clean and normalize input."""
        # Strip whitespace
        v = v.strip()
        # Collapse multiple spaces
        v = " ".join(v.split())
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"raw_input": "call mom tomorrow at 3pm"},
                {"raw_input": "urgent: finish quarterly report by EOD"},
                {"raw_input": "buy groceries sometime this week"},
            ]
        }
    }
```

## Proposal Model

What the AI generates (untrusted until validated):

```python
class TaskProposal(BaseModel):
    """
    AI-generated task proposal.
    
    This is what the AI extracts from natural language.
    Must be validated before creating a Task.
    
    Note: extra="forbid" rejects unexpected fields from AI.
    """
    title: Annotated[
        str,
        Field(
            min_length=1,
            max_length=200,
            description="Concise task title"
        )
    ]
    
    description: Annotated[
        Optional[str],
        Field(
            default=None,
            max_length=1000,
            description="Additional task details"
        )
    ]
    
    priority: Annotated[
        Priority,
        Field(
            default=Priority.MEDIUM,
            description="Task priority level"
        )
    ]
    
    category: Annotated[
        Category,
        Field(
            default=Category.OTHER,
            description="Task category"
        )
    ]
    
    due_date: Annotated[
        Optional[datetime],
        Field(
            default=None,
            description="When the task is due"
        )
    ]
    
    @field_validator("title")
    @classmethod
    def clean_title(cls, v: str) -> str:
        """Clean and capitalize title."""
        v = v.strip()
        # Capitalize first letter if not already
        if v and v[0].islower():
            v = v[0].upper() + v[1:]
        return v
    
    @field_validator("priority", mode="before")
    @classmethod
    def parse_priority(cls, v):
        """Accept various priority formats."""
        if isinstance(v, Priority):
            return v
        if isinstance(v, str):
            return Priority.from_string(v)
        return Priority.MEDIUM
    
    @field_validator("category", mode="before")
    @classmethod
    def parse_category(cls, v):
        """Accept various category formats."""
        if isinstance(v, Category):
            return v
        if isinstance(v, str):
            return Category.from_string(v)
        return Category.OTHER
    
    @field_validator("due_date", mode="before")
    @classmethod
    def parse_due_date(cls, v):
        """Parse various date formats."""
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            # Try ISO format first
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                pass
            
            # Try common formats
            from dateutil import parser
            try:
                return parser.parse(v)
            except (ValueError, TypeError):
                return None
        return None
    
    model_config = {
        "extra": "forbid",  # Reject unexpected fields
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Call mom",
                    "priority": "medium",
                    "category": "personal",
                    "due_date": "2024-01-16T15:00:00"
                }
            ]
        }
    }
```

## Task Model

The fully validated, persistent entity:

```python
class Task(BaseModel):
    """
    Validated, persistent task entity.
    
    This is what gets stored in the database.
    Created by the system after validating a TaskProposal.
    """
    id: Annotated[
        str,
        Field(
            description="Unique task identifier"
        )
    ]
    
    title: Annotated[
        str,
        Field(
            min_length=1,
            max_length=200
        )
    ]
    
    description: Optional[str] = None
    
    priority: Priority
    
    category: Category
    
    due_date: Optional[datetime] = None
    
    status: TaskStatus = TaskStatus.PENDING
    
    completed: bool = False
    
    created_at: datetime
    
    updated_at: datetime
    
    @classmethod
    def from_proposal(cls, proposal: TaskProposal) -> "Task":
        """Create a Task from a validated proposal."""
        now = datetime.utcnow()
        return cls(
            id=str(uuid.uuid4()),
            title=proposal.title,
            description=proposal.description,
            priority=proposal.priority,
            category=proposal.category,
            due_date=proposal.due_date,
            status=TaskStatus.PENDING,
            completed=False,
            created_at=now,
            updated_at=now
        )
    
    def mark_completed(self) -> "Task":
        """Return a copy marked as completed."""
        return self.model_copy(update={
            "completed": True,
            "status": TaskStatus.COMPLETED,
            "updated_at": datetime.utcnow()
        })
    
    def update_priority(self, priority: Priority) -> "Task":
        """Return a copy with updated priority."""
        return self.model_copy(update={
            "priority": priority,
            "updated_at": datetime.utcnow()
        })
    
    def to_db_dict(self) -> dict:
        """Convert to dictionary for database storage."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "category": self.category.value,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "status": self.status.value,
            "completed": int(self.completed),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_db_row(cls, row: dict) -> "Task":
        """Create Task from database row."""
        return cls(
            id=row["id"],
            title=row["title"],
            description=row["description"],
            priority=Priority(row["priority"]),
            category=Category(row["category"]),
            due_date=datetime.fromisoformat(row["due_date"]) if row["due_date"] else None,
            status=TaskStatus(row["status"]) if "status" in row else TaskStatus.PENDING,
            completed=bool(row["completed"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "title": "Call mom",
                    "description": None,
                    "priority": "medium",
                    "category": "personal",
                    "due_date": "2024-01-16T15:00:00",
                    "status": "pending",
                    "completed": False,
                    "created_at": "2024-01-15T10:30:00",
                    "updated_at": "2024-01-15T10:30:00"
                }
            ]
        }
    }
```

## Validation in Action

Let's see how validation works:

```python
# scripts/validation_demo.py
"""Demonstrate Pydantic validation for task models."""

import json
from ai_todo.models.task import TaskProposal, Task
from pydantic import ValidationError


def demo_valid_proposal():
    """Valid AI response gets parsed correctly."""
    print("1. Valid Proposal")
    print("-" * 40)
    
    ai_response = {
        "title": "call mom",
        "priority": "medium",
        "category": "personal",
        "due_date": "2024-01-16T15:00:00"
    }
    
    proposal = TaskProposal.model_validate(ai_response)
    print(f"   Title: {proposal.title}")  # Capitalized: "Call mom"
    print(f"   Priority: {proposal.priority}")
    print(f"   Category: {proposal.category}")
    print(f"   Due date: {proposal.due_date}")
    
    # Create Task from proposal
    task = Task.from_proposal(proposal)
    print(f"   Task ID: {task.id[:8]}...")
    print()


def demo_fuzzy_parsing():
    """AI might use various formats - we handle them."""
    print("2. Fuzzy Parsing")
    print("-" * 40)
    
    ai_response = {
        "title": "finish report",
        "priority": "HIGH",  # uppercase
        "category": "job",   # synonym for 'work'
        "due_date": "January 20, 2024 5pm"  # natural date
    }
    
    proposal = TaskProposal.model_validate(ai_response)
    print(f"   Priority 'HIGH' → {proposal.priority}")
    print(f"   Category 'job' → {proposal.category}")
    print(f"   Due date parsed: {proposal.due_date}")
    print()


def demo_missing_fields():
    """Missing optional fields get defaults."""
    print("3. Missing Fields (Defaults Applied)")
    print("-" * 40)
    
    ai_response = {
        "title": "buy groceries"
        # No priority, category, or due_date
    }
    
    proposal = TaskProposal.model_validate(ai_response)
    print(f"   Title: {proposal.title}")
    print(f"   Priority (default): {proposal.priority}")
    print(f"   Category (default): {proposal.category}")
    print(f"   Due date (default): {proposal.due_date}")
    print()


def demo_validation_error():
    """Invalid data gets rejected."""
    print("4. Validation Errors")
    print("-" * 40)
    
    # Missing required field
    try:
        TaskProposal.model_validate({})
    except ValidationError as e:
        print(f"   Empty object: {e.error_count()} error(s)")
        print(f"   → {e.errors()[0]['msg']}")
    
    # Title too long
    try:
        TaskProposal.model_validate({"title": "x" * 300})
    except ValidationError as e:
        print(f"   Title too long: {e.errors()[0]['msg']}")
    
    # Extra field (forbidden)
    try:
        TaskProposal.model_validate({
            "title": "test",
            "unknown_field": "value"
        })
    except ValidationError as e:
        print(f"   Extra field: {e.errors()[0]['msg']}")
    print()


def demo_ai_to_task_flow():
    """Complete flow from AI response to Task."""
    print("5. Complete Flow: AI → Proposal → Task")
    print("-" * 40)
    
    # Simulate AI JSON response
    ai_json = '{"title": "urgent: call client", "priority": "urgent", "category": "work"}'
    
    # Parse JSON
    raw = json.loads(ai_json)
    print(f"   AI output: {raw}")
    
    # Validate as Proposal
    proposal = TaskProposal.model_validate(raw)
    print(f"   Proposal validated: {proposal.title}")
    
    # Create Task
    task = Task.from_proposal(proposal)
    print(f"   Task created: {task.id[:8]}...")
    print(f"   Ready for DB: {task.to_db_dict()['title']}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("PYDANTIC VALIDATION DEMO")
    print("=" * 50 + "\n")
    
    demo_valid_proposal()
    demo_fuzzy_parsing()
    demo_missing_fields()
    demo_validation_error()
    demo_ai_to_task_flow()
```

Output:
```
==================================================
PYDANTIC VALIDATION DEMO
==================================================

1. Valid Proposal
----------------------------------------
   Title: Call mom
   Priority: Priority.MEDIUM
   Category: Category.PERSONAL
   Due date: 2024-01-16 15:00:00

   Task ID: 550e8400...

2. Fuzzy Parsing
----------------------------------------
   Priority 'HIGH' → Priority.HIGH
   Category 'job' → Category.WORK
   Due date parsed: 2024-01-20 17:00:00

3. Missing Fields (Defaults Applied)
----------------------------------------
   Title: Buy groceries
   Priority (default): Priority.MEDIUM
   Category (default): Category.OTHER
   Due date (default): None

4. Validation Errors
----------------------------------------
   Empty object: 1 error(s)
   → Field required
   Title too long: String should have at most 200 characters
   Extra field: Extra inputs are not permitted

5. Complete Flow: AI → Proposal → Task
----------------------------------------
   AI output: {'title': 'urgent: call client', 'priority': 'urgent', 'category': 'work'}
   Proposal validated: Urgent: call client
   Task created: a1b2c3d4...
   Ready for DB: Urgent: call client
```

## Why This Structure Matters

### 1. Defense in Depth

```
AI Output (untrusted)
       ↓
JSON Parse (can fail)
       ↓
Pydantic Validation (can fail)
       ↓
Task Creation (deterministic)
       ↓
Database (trusted)
```

Each layer catches different errors.

### 2. Clear Contracts

The models define exactly what data looks like. Any layer can import and use these definitions.

### 3. Automatic Documentation

Pydantic generates JSON schemas:

```python
print(TaskProposal.model_json_schema())
```

```json
{
  "title": "TaskProposal",
  "type": "object",
  "required": ["title"],
  "properties": {
    "title": {
      "type": "string",
      "minLength": 1,
      "maxLength": 200
    },
    "priority": {
      "enum": ["low", "medium", "high", "urgent"],
      "default": "medium"
    },
    ...
  }
}
```

### 4. Type Safety

IDEs understand the types and provide autocomplete, catch errors before runtime.

## Module Exports

```python
# src/ai_todo/models/__init__.py
"""Model exports."""

from .enums import Priority, Category, TaskStatus
from .task import TaskInput, TaskProposal, Task

__all__ = [
    "Priority",
    "Category", 
    "TaskStatus",
    "TaskInput",
    "TaskProposal",
    "Task",
]
```

## Summary

In this chapter we:

1. ✅ Defined the three-tier model hierarchy (Input → Proposal → Task)
2. ✅ Created enums for constrained choices
3. ✅ Added fuzzy parsing for AI variations
4. ✅ Implemented validation with clear error messages
5. ✅ Built conversion methods for database storage

The model hierarchy enforces our core principle:
- **AI proposes** (TaskProposal)
- **System validates** (Pydantic)
- **System commits** (Task)

In the next chapter, we'll build the natural language parsing system that transforms user input into structured proposals.

---

**Previous**: [Chapter 5: Your First LLM Call](./chapter-05-first-llm-call.md)  
**Next**: [Chapter 7: Natural Language Parsing](./chapter-07-nlp-parsing.md)
