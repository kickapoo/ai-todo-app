# Chapter 12: ChromaDB Embeddings — Semantic Memory

## The Memory Principle

Recall Principle 4: **Memory Creates Behavior**.

```
Without memory, AI is reactive.
With retrieval, AI becomes contextual and behavioral.
```

SQLite stores **what happened** (facts, records).
ChromaDB stores **what it means** (embeddings, semantic relationships).

Together, they create a memory system that makes the AI contextual.

## Understanding Embeddings

An embedding is a vector representation of text that captures meaning:

```
"buy groceries"     → [0.12, -0.34, 0.56, ...]  (4096 dimensions)
"pick up food"      → [0.11, -0.33, 0.55, ...]  (similar vector)
"fix server bug"    → [-0.45, 0.22, -0.18, ...] (different vector)
```

Similar meanings = similar vectors. This enables:
- **Semantic search**: Find tasks by meaning, not keywords
- **Duplicate detection**: Identify similar existing tasks
- **Context retrieval**: Augment AI prompts with relevant history

## ChromaDB Setup

```python
# src/ai_todo/memory/embeddings.py
"""
Embedding store using ChromaDB.

Stores semantic representations of tasks for retrieval.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Optional
import logging
from datetime import datetime

from ..models.task import Task
from ..ai.client import OllamaClient

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """
    ChromaDB-based embedding store.
    
    Stores embeddings for semantic search and retrieval.
    Used for:
    - Finding similar tasks
    - Duplicate detection
    - Context retrieval for AI prompts
    """
    
    COLLECTION_NAME = "tasks"
    
    def __init__(
        self,
        persist_dir: str,
        ai_client: OllamaClient
    ):
        self.persist_dir = persist_dir
        self.ai = ai_client
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        logger.info(f"Embedding store initialized: {persist_dir}")
        logger.info(f"Collection '{self.COLLECTION_NAME}' has {self.collection.count()} items")
    
    async def index(self, task: Task) -> None:
        """
        Index a task for semantic search.
        
        Generates embedding and stores in ChromaDB.
        """
        # Create text representation for embedding
        text = self._task_to_text(task)
        
        # Generate embedding via Ollama
        embedding = await self.ai.embed(text)
        
        # Prepare metadata (ChromaDB requires simple types)
        metadata = {
            "title": task.title,
            "priority": task.priority.value,
            "category": task.category.value,
            "status": task.status.value,
            "created_at": task.created_at.isoformat(),
        }
        if task.due_date:
            metadata["due_date"] = task.due_date.isoformat()
        
        # Upsert (update if exists, insert if not)
        self.collection.upsert(
            ids=[task.id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[text]
        )
        
        logger.debug(f"Indexed task: {task.id}")
    
    async def remove(self, task_id: str) -> None:
        """Remove a task from the index."""
        try:
            self.collection.delete(ids=[task_id])
            logger.debug(f"Removed from index: {task_id}")
        except Exception as e:
            logger.warning(f"Failed to remove from index: {e}")
    
    async def find_similar(
        self,
        query: str,
        limit: int = 5,
        exclude_ids: Optional[List[str]] = None,
        min_score: float = 0.0
    ) -> List[dict]:
        """
        Find tasks similar to a query.
        
        Args:
            query: Text to search for
            limit: Maximum results
            exclude_ids: IDs to exclude from results
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of matches with id, score, and metadata
        """
        # Generate query embedding
        query_embedding = await self.ai.embed(query)
        
        # Build where filter for exclusions
        where = None
        if exclude_ids:
            # ChromaDB doesn't have direct ID exclusion, so we query more
            limit = limit + len(exclude_ids)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["metadatas", "documents", "distances"]
        )
        
        # Process results
        matches = []
        if results["ids"] and results["ids"][0]:
            for i, task_id in enumerate(results["ids"][0]):
                # Skip excluded IDs
                if exclude_ids and task_id in exclude_ids:
                    continue
                
                # Convert distance to similarity score (cosine distance)
                # ChromaDB returns distance, we want similarity
                distance = results["distances"][0][i]
                score = 1 - distance  # Convert to similarity
                
                if score < min_score:
                    continue
                
                matches.append({
                    "id": task_id,
                    "score": score,
                    "metadata": results["metadatas"][0][i],
                    "text": results["documents"][0][i] if results["documents"] else None
                })
        
        # Re-limit after exclusions
        return matches[:limit] if len(matches) > limit else matches
    
    async def find_duplicates(
        self,
        text: str,
        threshold: float = 0.85
    ) -> List[dict]:
        """
        Find potential duplicate tasks.
        
        Args:
            text: Task text to check
            threshold: Similarity threshold for duplicates (0.85 = 85% similar)
            
        Returns:
            List of potential duplicates
        """
        return await self.find_similar(
            query=text,
            limit=5,
            min_score=threshold
        )
    
    async def get_context_for_task(
        self,
        task: Task,
        limit: int = 3
    ) -> str:
        """
        Get context from similar past tasks.
        
        Used for RAG - augmenting prompts with relevant history.
        """
        similar = await self.find_similar(
            query=self._task_to_text(task),
            limit=limit,
            exclude_ids=[task.id]
        )
        
        if not similar:
            return ""
        
        context_parts = ["Similar past tasks:"]
        for match in similar:
            meta = match["metadata"]
            context_parts.append(
                f"- {meta['title']} (priority: {meta['priority']}, category: {meta['category']})"
            )
        
        return "\n".join(context_parts)
    
    async def rebuild_index(self, tasks: List[Task]) -> int:
        """
        Rebuild the entire index from tasks.
        
        Use after bulk imports or index corruption.
        """
        # Clear existing
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
        
        # Recreate collection
        self.collection = self.client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Re-index all tasks
        count = 0
        for task in tasks:
            try:
                await self.index(task)
                count += 1
            except Exception as e:
                logger.error(f"Failed to index task {task.id}: {e}")
        
        logger.info(f"Rebuilt index with {count} tasks")
        return count
    
    def _task_to_text(self, task: Task) -> str:
        """Convert task to text for embedding."""
        parts = [task.title]
        
        if task.description:
            parts.append(task.description)
        
        # Add category context
        parts.append(f"category: {task.category.value}")
        
        return " | ".join(parts)
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "collection": self.COLLECTION_NAME,
            "count": self.collection.count(),
            "persist_dir": self.persist_dir
        }
```

## Retrieval-Augmented Generation (RAG)

RAG combines retrieval with generation. We retrieve relevant context and include it in prompts:

```python
# src/ai_todo/ai/rag.py
"""
Retrieval-Augmented Generation for context-aware AI.
"""

from typing import List, Optional
from datetime import datetime

from ..models.task import Task
from ..memory.embeddings import EmbeddingStore


class RAGService:
    """
    Retrieval-Augmented Generation service.
    
    Enhances AI prompts with relevant context from memory.
    """
    
    def __init__(self, embedding_store: EmbeddingStore):
        self.embeddings = embedding_store
    
    async def build_context(
        self,
        query: str,
        *,
        max_items: int = 5,
        include_metadata: bool = True
    ) -> str:
        """
        Build context string from similar tasks.
        
        This context is injected into AI prompts to make
        responses more relevant and consistent.
        """
        similar = await self.embeddings.find_similar(
            query=query,
            limit=max_items,
            min_score=0.3  # Only include reasonably similar items
        )
        
        if not similar:
            return ""
        
        lines = ["Relevant past tasks:"]
        for match in similar:
            meta = match["metadata"]
            score_pct = int(match["score"] * 100)
            
            line = f"- {meta['title']}"
            if include_metadata:
                line += f" ({meta['category']}, {meta['priority']} priority)"
            line += f" [{score_pct}% match]"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    async def augment_prompt(
        self,
        base_prompt: str,
        context_query: str,
        *,
        context_label: str = "CONTEXT"
    ) -> str:
        """
        Augment a prompt with retrieved context.
        
        Args:
            base_prompt: The original prompt
            context_query: Query to use for retrieval
            context_label: Label for context section
            
        Returns:
            Augmented prompt with context
        """
        context = await self.build_context(context_query)
        
        if not context:
            return base_prompt
        
        return f"""{context_label}:
{context}

{base_prompt}"""
    
    async def get_categorization_context(
        self,
        task_title: str
    ) -> str:
        """
        Get context for category inference.
        
        Shows how similar tasks were categorized.
        """
        similar = await self.embeddings.find_similar(
            query=task_title,
            limit=3,
            min_score=0.4
        )
        
        if not similar:
            return "No similar past tasks found."
        
        # Group by category
        by_category = {}
        for match in similar:
            cat = match["metadata"]["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(match["metadata"]["title"])
        
        lines = ["Similar tasks and their categories:"]
        for cat, tasks in by_category.items():
            lines.append(f"- {cat}: {', '.join(tasks)}")
        
        return "\n".join(lines)
    
    async def get_priority_context(
        self,
        task_title: str
    ) -> str:
        """
        Get context for priority inference.
        
        Shows how similar tasks were prioritized.
        """
        similar = await self.embeddings.find_similar(
            query=task_title,
            limit=5,
            min_score=0.4
        )
        
        if not similar:
            return "No similar past tasks found."
        
        # Group by priority
        by_priority = {}
        for match in similar:
            pri = match["metadata"]["priority"]
            if pri not in by_priority:
                by_priority[pri] = []
            by_priority[pri].append(match["metadata"]["title"])
        
        lines = ["Similar tasks and their priorities:"]
        for pri, tasks in by_priority.items():
            lines.append(f"- {pri}: {', '.join(tasks[:3])}")
        
        return "\n".join(lines)
```

## Using RAG in Task Creation

```python
# Enhanced task service with RAG

class TaskService:
    def __init__(
        self,
        ai_service: AIService,
        repository: TaskRepository,
        embedding_store: EmbeddingStore
    ):
        self.ai = ai_service
        self.repo = repository
        self.embeddings = embedding_store
        self.rag = RAGService(embedding_store)
    
    async def create_task_with_context(
        self,
        input: TaskInput,
        *,
        use_rag: bool = True
    ) -> TaskCreationResult:
        """
        Create task with RAG-enhanced AI processing.
        
        The AI sees relevant past tasks, making categorization
        and prioritization more consistent.
        """
        # Build context from similar tasks
        context = ""
        if use_rag:
            context = await self.rag.build_context(input.raw_input)
        
        # AI processes with context
        proposal = await self.ai.extract_task_with_context(
            raw_input=input.raw_input,
            context=context
        )
        
        # ... rest of creation flow ...
```

## Demo: Semantic Search

```python
# scripts/semantic_search_demo.py
"""Demonstrate semantic search capabilities."""

import asyncio
from rich.console import Console
from rich.table import Table

from ai_todo.main import get_app
from ai_todo.models.task import TaskInput


console = Console()


async def main():
    app = get_app()
    
    console.print("[bold]Semantic Search Demo[/bold]\n")
    
    # Create some test tasks
    test_tasks = [
        "buy groceries for the week",
        "pick up vegetables from the market",
        "call mom about dinner plans",
        "schedule dentist appointment",
        "review quarterly sales report",
        "send email to marketing team",
        "exercise at the gym",
        "go for a morning run",
    ]
    
    console.print("Creating test tasks...")
    for raw in test_tasks:
        await app.tasks.create_task(TaskInput(raw_input=raw))
        console.print(f"  Created: {raw}")
    
    console.print()
    
    # Semantic search queries
    queries = [
        ("food shopping", "Should find grocery/market tasks"),
        ("health", "Should find gym/exercise tasks"),
        ("communication", "Should find call/email tasks"),
        ("documents", "Should find report task"),
    ]
    
    for query, description in queries:
        console.print(f"\n[cyan]Query: \"{query}\"[/cyan] ({description})")
        
        similar = await app.tasks.embeddings.find_similar(query, limit=3)
        
        table = Table()
        table.add_column("Task")
        table.add_column("Score")
        
        for match in similar:
            score_pct = f"{match['score']*100:.0f}%"
            table.add_row(match["metadata"]["title"], score_pct)
        
        console.print(table)
    
    # Clean up
    await app.close()


if __name__ == "__main__":
    asyncio.run(main())
```

Output:
```
Semantic Search Demo

Creating test tasks...
  Created: buy groceries for the week
  Created: pick up vegetables from the market
  Created: call mom about dinner plans
  ...

Query: "food shopping" (Should find grocery/market tasks)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Task                             ┃ Score ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Buy groceries for the week       │ 78%   │
│ Pick up vegetables from market   │ 71%   │
│ Call mom about dinner plans      │ 42%   │
└──────────────────────────────────┴───────┘

Query: "health" (Should find gym/exercise tasks)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Task                             ┃ Score ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Exercise at the gym              │ 72%   │
│ Go for a morning run             │ 65%   │
│ Schedule dentist appointment     │ 58%   │
└──────────────────────────────────┴───────┘
```

## Summary

In this chapter we implemented:

1. ✅ ChromaDB integration for semantic storage
2. ✅ Embedding generation via Ollama
3. ✅ Similarity search with scoring
4. ✅ Duplicate detection
5. ✅ RAG service for context-aware prompts

This implements Principle 4: **Memory Creates Behavior**.

The memory architecture:
- **SQLite**: What happened (facts)
- **ChromaDB**: What it means (semantics)
- **RAG**: Retrieval augments future reasoning

In the next chapter, we'll use RAG for intelligent categorization.

---

**Previous**: [Chapter 11: SQLite Persistence](./chapter-11-sqlite.md)  
**Next**: [Chapter 13: Retrieval-Augmented Generation](./chapter-13-rag.md)
