# Chapter 4: Python Project Structure — Clean Architecture

## The Architecture Principle

Recall Principle 5: **Clean Architecture Enables Safe Intelligence**.

```
Routes → Services → AI → Validation → Persistence → Memory
```

Each layer has a single responsibility. No layer skips another. This structure isn't bureaucracy—it's what makes AI systems debuggable and maintainable.

## Project Layout

```
ai-todo-app/
├── src/
│   └── ai_todo/
│       ├── __init__.py
│       ├── main.py              # Entry point
│       ├── config.py            # Configuration
│       │
│       ├── models/              # Pydantic schemas
│       │   ├── __init__.py
│       │   └── task.py
│       │
│       ├── services/            # Business logic
│       │   ├── __init__.py
│       │   ├── task_service.py
│       │   └── ai_service.py
│       │
│       ├── ai/                  # LLM interactions
│       │   ├── __init__.py
│       │   ├── client.py
│       │   ├── prompts.py
│       │   └── parsers.py
│       │
│       ├── persistence/         # Database operations
│       │   ├── __init__.py
│       │   ├── database.py
│       │   └── repositories.py
│       │
│       ├── memory/              # Vector store
│       │   ├── __init__.py
│       │   └── embeddings.py
│       │
│       └── cli/                 # User interface
│           ├── __init__.py
│           └── commands.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_ai.py
│   ├── test_services.py
│   └── test_persistence.py
│
├── data/                        # Runtime data (gitignored)
│   ├── tasks.db
│   └── chroma/
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Creating the Structure

```bash
cd ai-todo-app

# Create directories
mkdir -p src/ai_todo/{models,services,ai,persistence,memory,cli}
mkdir -p tests
mkdir -p data

# Create __init__.py files
touch src/ai_todo/__init__.py
touch src/ai_todo/models/__init__.py
touch src/ai_todo/services/__init__.py
touch src/ai_todo/ai/__init__.py
touch src/ai_todo/persistence/__init__.py
touch src/ai_todo/memory/__init__.py
touch src/ai_todo/cli/__init__.py
touch tests/__init__.py

# Create placeholder files
touch src/ai_todo/main.py
touch src/ai_todo/config.py
```

## Layer Responsibilities

### 1. Models Layer (`models/`)

Defines data structures using Pydantic. This is the **contract** between all layers.

```python
# src/ai_todo/models/task.py
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Category(str, Enum):
    WORK = "work"
    PERSONAL = "personal"
    HEALTH = "health"
    FINANCE = "finance"
    LEARNING = "learning"
    ERRANDS = "errands"
    OTHER = "other"


class TaskCreate(BaseModel):
    """Input from user (natural language)."""
    raw_input: str


class TaskProposal(BaseModel):
    """AI-generated proposal (needs validation)."""
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    priority: Priority = Priority.MEDIUM
    category: Category = Category.OTHER
    due_date: Optional[datetime] = None
    
    model_config = {"extra": "forbid"}


class Task(BaseModel):
    """Validated, persisted task."""
    id: str
    title: str
    description: Optional[str] = None
    priority: Priority
    category: Category
    due_date: Optional[datetime] = None
    completed: bool = False
    created_at: datetime
    updated_at: datetime
```

The model hierarchy enforces our principle:
- `TaskCreate`: Raw user input
- `TaskProposal`: AI-generated (untrustworthy)
- `Task`: Validated and persisted (trustworthy)

### 2. AI Layer (`ai/`)

Handles all LLM interactions. Never touches the database directly.

```python
# src/ai_todo/ai/client.py
import httpx
from typing import Any


class OllamaClient:
    """Low-level Ollama API client."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: float = 60.0
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
    
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        format: str | None = "json",
        max_tokens: int = 512
    ) -> str:
        """Generate text completion."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system,
                    "format": format,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    }
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["response"]
    
    async def embed(self, text: str) -> list[float]:
        """Generate embedding vector."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["embedding"]
```

### 3. Services Layer (`services/`)

Orchestrates operations. This is where the flow happens.

```python
# src/ai_todo/services/task_service.py
from datetime import datetime
import uuid

from ..models.task import TaskCreate, TaskProposal, Task
from ..ai.client import OllamaClient
from ..ai.parsers import parse_task_proposal
from ..persistence.repositories import TaskRepository
from ..memory.embeddings import EmbeddingStore


class TaskService:
    """Orchestrates task creation following AI-first principles."""
    
    def __init__(
        self,
        ai_client: OllamaClient,
        repository: TaskRepository,
        embedding_store: EmbeddingStore
    ):
        self.ai = ai_client
        self.repo = repository
        self.embeddings = embedding_store
    
    async def create_task(self, input: TaskCreate) -> Task:
        """
        Full flow: Routes → Services → AI → Validation → Persistence → Memory
        """
        # 1. AI proposes (temperature 0.1 for consistency)
        proposal = await self._generate_proposal(input.raw_input)
        
        # 2. System validates (Pydantic enforces schema)
        validated = TaskProposal.model_validate(proposal)
        
        # 3. System commits (create Task entity)
        now = datetime.utcnow()
        task = Task(
            id=str(uuid.uuid4()),
            title=validated.title,
            description=validated.description,
            priority=validated.priority,
            category=validated.category,
            due_date=validated.due_date,
            completed=False,
            created_at=now,
            updated_at=now
        )
        
        # 4. Persist to SQLite
        await self.repo.save(task)
        
        # 5. Index in vector store
        await self.embeddings.index(task)
        
        return task
    
    async def _generate_proposal(self, raw_input: str) -> dict:
        """AI generates structured proposal from natural language."""
        from ..ai.prompts import TASK_EXTRACTION_PROMPT
        
        prompt = TASK_EXTRACTION_PROMPT.format(user_input=raw_input)
        response = await self.ai.generate(
            prompt=prompt,
            temperature=0.1,  # Low entropy for parsing
            format="json"
        )
        return parse_task_proposal(response)
```

### 4. Persistence Layer (`persistence/`)

Handles SQLite operations. Deterministic and predictable.

```python
# src/ai_todo/persistence/database.py
import sqlite3
from pathlib import Path


def init_database(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite database with schema."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT,
            priority TEXT NOT NULL,
            category TEXT NOT NULL,
            due_date TEXT,
            completed INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn
```

### 5. Memory Layer (`memory/`)

Handles ChromaDB for semantic retrieval.

```python
# src/ai_todo/memory/embeddings.py
import chromadb
from chromadb.config import Settings

from ..models.task import Task
from ..ai.client import OllamaClient


class EmbeddingStore:
    """Semantic memory using ChromaDB."""
    
    def __init__(self, persist_dir: str, ai_client: OllamaClient):
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_dir
        ))
        self.collection = self.client.get_or_create_collection("tasks")
        self.ai = ai_client
    
    async def index(self, task: Task) -> None:
        """Index task for semantic retrieval."""
        # Generate embedding
        text = f"{task.title} {task.description or ''}"
        embedding = await self.ai.embed(text)
        
        # Store in ChromaDB
        self.collection.add(
            ids=[task.id],
            embeddings=[embedding],
            metadatas=[{
                "title": task.title,
                "category": task.category.value,
                "priority": task.priority.value
            }]
        )
    
    async def find_similar(self, query: str, limit: int = 5) -> list[dict]:
        """Find semantically similar tasks."""
        embedding = await self.ai.embed(query)
        
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=limit
        )
        
        return results["metadatas"][0] if results["metadatas"] else []
```

### 6. CLI Layer (`cli/`)

User interface. Accepts input, displays output.

```python
# src/ai_todo/cli/commands.py
from rich.console import Console
from rich.table import Table

from ..models.task import TaskCreate
from ..services.task_service import TaskService


console = Console()


async def add_task(service: TaskService, raw_input: str) -> None:
    """Add a new task from natural language."""
    with console.status("Thinking..."):
        task = await service.create_task(TaskCreate(raw_input=raw_input))
    
    console.print(f"\n[green]✓[/green] Created task: {task.title}")
    console.print(f"  Priority: {task.priority.value}")
    console.print(f"  Category: {task.category.value}")
    if task.due_date:
        console.print(f"  Due: {task.due_date.strftime('%Y-%m-%d %H:%M')}")
```

## Configuration

```python
# src/ai_todo/config.py
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration."""
    
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_timeout: float = 60.0
    
    # Temperature settings (Principle 3)
    temperature_parsing: float = 0.1      # Low for structure
    temperature_reasoning: float = 0.3    # Moderate for inference
    temperature_creative: float = 0.6     # Higher for suggestions
    
    # Paths
    data_dir: Path = Path("data")
    db_path: Path = Path("data/tasks.db")
    chroma_dir: Path = Path("data/chroma")
    
    model_config = {"env_prefix": "AI_TODO_"}


settings = Settings()
```

## Dependency Injection

```python
# src/ai_todo/main.py
from .config import settings
from .ai.client import OllamaClient
from .persistence.database import init_database
from .persistence.repositories import TaskRepository
from .memory.embeddings import EmbeddingStore
from .services.task_service import TaskService


def create_app() -> TaskService:
    """Wire up all dependencies."""
    # Ensure data directory exists
    settings.data_dir.mkdir(exist_ok=True)
    
    # Create components
    ai_client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=settings.ollama_timeout
    )
    
    db = init_database(settings.db_path)
    repository = TaskRepository(db)
    
    embedding_store = EmbeddingStore(
        persist_dir=str(settings.chroma_dir),
        ai_client=ai_client
    )
    
    # Compose service
    return TaskService(
        ai_client=ai_client,
        repository=repository,
        embedding_store=embedding_store
    )
```

## The Flow Visualized

```
┌─────────────────────────────────────────────────────────────────┐
│                          USER INPUT                              │
│              "remind me to call mom tomorrow 3pm"                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  CLI LAYER                cli/commands.py                        │
│  ─────────────────────────────────────────────────────────────  │
│  • Parse command-line arguments                                  │
│  • Display status and results                                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  SERVICE LAYER            services/task_service.py               │
│  ─────────────────────────────────────────────────────────────  │
│  • Orchestrate the full flow                                     │
│  • Coordinate AI, validation, persistence, memory                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  AI LAYER                 ai/client.py, ai/prompts.py            │
│  ─────────────────────────────────────────────────────────────  │
│  • Send prompt to Ollama                                         │
│  • Parse JSON response                                           │
│  • Temperature: 0.1                                              │
│                                                                  │
│  OUTPUT: {"title": "Call mom", "due_date": "...", ...}          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  VALIDATION               models/task.py (Pydantic)              │
│  ─────────────────────────────────────────────────────────────  │
│  • Validate against TaskProposal schema                          │
│  • Enforce field constraints                                     │
│  • Reject invalid proposals                                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  PERSISTENCE              persistence/repositories.py            │
│  ─────────────────────────────────────────────────────────────  │
│  • Create Task entity with UUID                                  │
│  • Save to SQLite                                                │
│  • What happened: stored                                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  MEMORY                   memory/embeddings.py                   │
│  ─────────────────────────────────────────────────────────────  │
│  • Generate embedding                                            │
│  • Store in ChromaDB                                             │
│  • What it means: indexed                                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                          RESPONSE                                │
│                                                                  │
│  ✓ Created task: Call mom                                        │
│    Priority: medium                                              │
│    Category: personal                                            │
│    Due: 2024-01-16 15:00                                         │
└─────────────────────────────────────────────────────────────────┘
```

## Why This Structure Matters

1. **Testability**: Each layer can be tested in isolation with mocks
2. **Debuggability**: Failures are localized to specific layers
3. **Evolvability**: Swap SQLite for Postgres without touching AI layer
4. **Safety**: AI never directly modifies state

## Summary

We've established a clean architecture:

| Layer | Files | Responsibility |
|-------|-------|----------------|
| Models | `models/*.py` | Data contracts |
| AI | `ai/*.py` | LLM interactions |
| Services | `services/*.py` | Orchestration |
| Persistence | `persistence/*.py` | State storage |
| Memory | `memory/*.py` | Semantic index |
| CLI | `cli/*.py` | User interface |

In the next chapter, we'll write our first actual LLM call and see the system come alive.

---

**Previous**: [Chapter 3: Understanding Ollama](./chapter-03-ollama-basics.md)  
**Next**: [Chapter 5: Your First LLM Call](./chapter-05-first-llm-call.md)
