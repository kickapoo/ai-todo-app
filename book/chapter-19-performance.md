# Chapter 19: Performance Optimization — Speed Without Compromise

## The Performance Challenge

AI operations are slow compared to traditional code:

| Operation | Typical Latency |
|-----------|-----------------|
| Database query | 1-10ms |
| Embedding lookup | 10-50ms |
| LLM inference | 200-2000ms |
| Full task creation | 500-3000ms |

This chapter optimizes without sacrificing correctness.

## Optimization Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                   OPTIMIZATION PRIORITY                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. AVOID WORK                   Highest impact                 │
│      └── Skip unnecessary AI calls                               │
│      └── Cache repeated requests                                 │
│                                                                  │
│   2. REDUCE WORK                  Medium impact                  │
│      └── Smaller prompts                                         │
│      └── Batch operations                                        │
│                                                                  │
│   3. PARALLELIZE                  Lower impact                   │
│      └── Concurrent requests                                     │
│      └── Async I/O                                               │
│                                                                  │
│   4. OPTIMIZE EXECUTION           Lowest impact                  │
│      └── Smaller models                                          │
│      └── Hardware acceleration                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Caching Layer

### Response Cache

```python
# src/ai_todo/cache/response_cache.py
"""
Cache for AI responses to avoid redundant inference.

Caches are keyed by prompt hash + temperature.
"""

import hashlib
import json
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
import logging


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cached response entry."""
    content: str
    created_at: float
    hits: int = 0
    model: str = ""
    tokens_used: int = 0


class ResponseCache:
    """
    LRU cache for AI responses.
    
    Features:
    - Configurable TTL (time-to-live)
    - Persistent storage with SQLite
    - Memory limit with LRU eviction
    """
    
    def __init__(
        self,
        max_entries: int = 1000,
        ttl_seconds: float = 3600,  # 1 hour default
        persist_path: Optional[Path] = None
    ):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.persist_path = persist_path
        
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._db: Optional[sqlite3.Connection] = None
        
        if persist_path:
            self._init_db()
    
    def _init_db(self):
        """Initialize SQLite cache storage."""
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._db = sqlite3.connect(self.persist_path)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                model TEXT,
                tokens_used INTEGER,
                created_at REAL NOT NULL,
                hits INTEGER DEFAULT 0
            )
        """)
        self._db.commit()
    
    def _make_key(
        self,
        prompt: str,
        temperature: float,
        model: str,
        format: Optional[str] = None
    ) -> str:
        """Create cache key from request parameters."""
        # Include all parameters that affect output
        data = f"{model}|{temperature}|{format or ''}|{prompt}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]
    
    def get(
        self,
        prompt: str,
        temperature: float,
        model: str,
        format: Optional[str] = None
    ) -> Optional[CacheEntry]:
        """
        Get cached response if available and fresh.
        
        Returns None if not cached or expired.
        """
        key = self._make_key(prompt, temperature, model, format)
        now = time.time()
        
        # Check memory cache first
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            
            if now - entry.created_at < self.ttl_seconds:
                entry.hits += 1
                logger.debug(f"Cache hit (memory): {key[:8]}")
                return entry
            else:
                # Expired
                del self._memory_cache[key]
        
        # Check persistent cache
        if self._db:
            cursor = self._db.execute(
                "SELECT content, model, tokens_used, created_at, hits FROM cache WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            
            if row:
                created_at = row[3]
                
                if now - created_at < self.ttl_seconds:
                    entry = CacheEntry(
                        content=row[0],
                        model=row[1],
                        tokens_used=row[2],
                        created_at=created_at,
                        hits=row[4] + 1
                    )
                    
                    # Update hit count
                    self._db.execute(
                        "UPDATE cache SET hits = ? WHERE key = ?",
                        (entry.hits, key)
                    )
                    self._db.commit()
                    
                    # Promote to memory
                    self._memory_cache[key] = entry
                    
                    logger.debug(f"Cache hit (disk): {key[:8]}")
                    return entry
                else:
                    # Expired - delete
                    self._db.execute("DELETE FROM cache WHERE key = ?", (key,))
                    self._db.commit()
        
        logger.debug(f"Cache miss: {key[:8]}")
        return None
    
    def set(
        self,
        prompt: str,
        temperature: float,
        model: str,
        content: str,
        tokens_used: int = 0,
        format: Optional[str] = None
    ):
        """Store response in cache."""
        key = self._make_key(prompt, temperature, model, format)
        now = time.time()
        
        entry = CacheEntry(
            content=content,
            model=model,
            tokens_used=tokens_used,
            created_at=now
        )
        
        # Store in memory
        self._memory_cache[key] = entry
        
        # Evict if over limit
        if len(self._memory_cache) > self.max_entries:
            self._evict_lru()
        
        # Persist
        if self._db:
            self._db.execute("""
                INSERT OR REPLACE INTO cache 
                (key, content, model, tokens_used, created_at, hits)
                VALUES (?, ?, ?, ?, ?, 0)
            """, (key, content, model, tokens_used, now))
            self._db.commit()
        
        logger.debug(f"Cached: {key[:8]}")
    
    def _evict_lru(self):
        """Evict least recently used entries."""
        # Sort by created_at + hits (simple LRU approximation)
        sorted_keys = sorted(
            self._memory_cache.keys(),
            key=lambda k: (
                self._memory_cache[k].created_at,
                self._memory_cache[k].hits
            )
        )
        
        # Remove oldest 10%
        to_remove = max(1, len(sorted_keys) // 10)
        
        for key in sorted_keys[:to_remove]:
            del self._memory_cache[key]
        
        logger.debug(f"Evicted {to_remove} cache entries")
    
    def clear(self):
        """Clear all cache entries."""
        self._memory_cache.clear()
        
        if self._db:
            self._db.execute("DELETE FROM cache")
            self._db.commit()
    
    def stats(self) -> dict:
        """Get cache statistics."""
        memory_count = len(self._memory_cache)
        
        disk_count = 0
        if self._db:
            cursor = self._db.execute("SELECT COUNT(*) FROM cache")
            disk_count = cursor.fetchone()[0]
        
        total_hits = sum(e.hits for e in self._memory_cache.values())
        
        return {
            "memory_entries": memory_count,
            "disk_entries": disk_count,
            "total_hits": total_hits
        }
```

### Embedding Cache

```python
# src/ai_todo/cache/embedding_cache.py
"""
Cache for embedding vectors.

Embeddings are expensive to compute but highly cacheable.
"""

import hashlib
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path
import sqlite3
import json


class EmbeddingCache:
    """
    Cache for embedding vectors.
    
    Embeddings are deterministic for same input + model,
    making them ideal candidates for caching.
    """
    
    def __init__(self, persist_path: Optional[Path] = None):
        self.persist_path = persist_path
        self._db: Optional[sqlite3.Connection] = None
        
        if persist_path:
            self._init_db()
    
    def _init_db(self):
        """Initialize SQLite storage."""
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._db = sqlite3.connect(self.persist_path)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                key TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                model TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        self._db.commit()
    
    def _make_key(self, text: str, model: str) -> str:
        """Create cache key."""
        data = f"{model}|{text}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]
    
    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get cached embedding."""
        if not self._db:
            return None
        
        key = self._make_key(text, model)
        
        cursor = self._db.execute(
            "SELECT vector FROM embeddings WHERE key = ?",
            (key,)
        )
        row = cursor.fetchone()
        
        if row:
            # Deserialize numpy array
            vector = np.frombuffer(row[0], dtype=np.float32).tolist()
            return vector
        
        return None
    
    def set(self, text: str, model: str, vector: List[float]):
        """Cache embedding."""
        if not self._db:
            return
        
        key = self._make_key(text, model)
        
        # Serialize as numpy bytes
        vector_bytes = np.array(vector, dtype=np.float32).tobytes()
        
        self._db.execute("""
            INSERT OR REPLACE INTO embeddings (key, vector, model, created_at)
            VALUES (?, ?, ?, ?)
        """, (key, vector_bytes, model, time.time()))
        self._db.commit()
    
    def get_batch(
        self,
        texts: List[str],
        model: str
    ) -> Dict[str, Optional[List[float]]]:
        """Get multiple embeddings at once."""
        results = {}
        
        for text in texts:
            results[text] = self.get(text, model)
        
        return results
    
    def set_batch(
        self,
        embeddings: Dict[str, List[float]],
        model: str
    ):
        """Cache multiple embeddings."""
        for text, vector in embeddings.items():
            self.set(text, model, vector)


import time
```

## Integrating Cache with AI Client

```python
# src/ai_todo/ai/cached_client.py
"""
AI client with caching layer.
"""

from typing import Optional

from .client import OllamaClient, AIResponse
from ..cache.response_cache import ResponseCache


class CachedOllamaClient:
    """
    Ollama client wrapper with response caching.
    
    For deterministic requests (low temperature),
    caching can eliminate redundant inference.
    """
    
    # Only cache low-temperature requests
    CACHE_TEMPERATURE_THRESHOLD = 0.3
    
    def __init__(
        self,
        client: OllamaClient,
        cache: ResponseCache
    ):
        self.client = client
        self.cache = cache
    
    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        format: Optional[str] = None,
        max_tokens: Optional[int] = None,
        bypass_cache: bool = False
    ) -> AIResponse:
        """
        Generate with caching for low-temperature requests.
        """
        # Only cache deterministic requests
        should_cache = (
            not bypass_cache and
            temperature <= self.CACHE_TEMPERATURE_THRESHOLD
        )
        
        if should_cache:
            cached = self.cache.get(
                prompt=prompt,
                temperature=temperature,
                model=self.client.model,
                format=format
            )
            
            if cached:
                return AIResponse(
                    content=cached.content,
                    model=cached.model,
                    tokens_used=cached.tokens_used,
                    duration_ms=0,  # Cached = instant
                    cached=True
                )
        
        # Call actual model
        response = await self.client.generate(
            prompt=prompt,
            temperature=temperature,
            format=format,
            max_tokens=max_tokens
        )
        
        # Cache the response
        if should_cache:
            self.cache.set(
                prompt=prompt,
                temperature=temperature,
                model=self.client.model,
                content=response.content,
                tokens_used=response.tokens_used,
                format=format
            )
        
        return response
    
    async def __aenter__(self):
        await self.client.__aenter__()
        return self
    
    async def __aexit__(self, *args):
        await self.client.__aexit__(*args)
```

## Batching Operations

```python
# src/ai_todo/utils/batching.py
"""
Batch processing utilities.
"""

import asyncio
from typing import TypeVar, List, Callable, Awaitable
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchConfig:
    """Batch processing configuration."""
    max_batch_size: int = 10
    max_wait_ms: float = 100
    max_concurrent: int = 3


class BatchProcessor:
    """
    Collect items and process in batches.
    
    Benefits:
    - Reduced overhead per item
    - Better throughput for bulk operations
    - Controlled concurrency
    """
    
    def __init__(
        self,
        processor: Callable[[List[T]], Awaitable[List[R]]],
        config: BatchConfig = None
    ):
        self.processor = processor
        self.config = config or BatchConfig()
        
        self._queue: List[tuple[T, asyncio.Future]] = []
        self._lock = asyncio.Lock()
        self._process_task: Optional[asyncio.Task] = None
    
    async def submit(self, item: T) -> R:
        """
        Submit item for batch processing.
        
        Returns result when batch is processed.
        """
        future = asyncio.get_event_loop().create_future()
        
        async with self._lock:
            self._queue.append((item, future))
            
            # Start processing task if needed
            if self._process_task is None or self._process_task.done():
                self._process_task = asyncio.create_task(self._process_loop())
        
        return await future
    
    async def _process_loop(self):
        """Process batches from queue."""
        while True:
            # Wait for batch to fill or timeout
            await asyncio.sleep(self.config.max_wait_ms / 1000)
            
            async with self._lock:
                if not self._queue:
                    return
                
                # Take batch
                batch_size = min(len(self._queue), self.config.max_batch_size)
                batch = self._queue[:batch_size]
                self._queue = self._queue[batch_size:]
            
            # Process batch
            items = [item for item, _ in batch]
            futures = [future for _, future in batch]
            
            try:
                results = await self.processor(items)
                
                for future, result in zip(futures, results):
                    future.set_result(result)
                    
            except Exception as e:
                for future in futures:
                    future.set_exception(e)


class EmbeddingBatcher:
    """
    Batch embedding requests for efficiency.
    """
    
    def __init__(self, ai_client, config: BatchConfig = None):
        self.client = ai_client
        self.config = config or BatchConfig(max_batch_size=20)
        
        self._processor = BatchProcessor(
            self._process_batch,
            self.config
        )
    
    async def embed(self, text: str) -> List[float]:
        """Get embedding for single text."""
        return await self._processor.submit(text)
    
    async def _process_batch(self, texts: List[str]) -> List[List[float]]:
        """Process batch of texts."""
        logger.debug(f"Processing embedding batch of {len(texts)}")
        
        # Call Ollama batch embedding endpoint
        response = await self.client.embed_batch(texts)
        
        return response.embeddings
```

## Async Parallelization

```python
# src/ai_todo/utils/parallel.py
"""
Parallel execution utilities.
"""

import asyncio
from typing import TypeVar, List, Callable, Awaitable, Tuple
import logging


logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


async def parallel_map(
    items: List[T],
    func: Callable[[T], Awaitable[R]],
    *,
    max_concurrent: int = 5
) -> List[R]:
    """
    Apply async function to items with bounded concurrency.
    
    Args:
        items: Items to process
        func: Async function to apply
        max_concurrent: Maximum concurrent tasks
        
    Returns:
        Results in same order as items
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def bounded_call(item: T) -> R:
        async with semaphore:
            return await func(item)
    
    tasks = [bounded_call(item) for item in items]
    return await asyncio.gather(*tasks)


async def parallel_first(
    operations: List[Callable[[], Awaitable[R]]],
    *,
    timeout: Optional[float] = None
) -> Tuple[int, R]:
    """
    Run operations in parallel, return first successful result.
    
    Useful for trying multiple approaches and using fastest.
    
    Returns:
        Tuple of (index, result) for first success
    """
    tasks = [asyncio.create_task(op()) for op in operations]
    
    try:
        done, pending = await asyncio.wait(
            tasks,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending
        for task in pending:
            task.cancel()
        
        # Return first successful
        for i, task in enumerate(tasks):
            if task in done and not task.exception():
                return (i, task.result())
        
        # All failed - raise first exception
        raise done.pop().exception()
        
    except asyncio.TimeoutError:
        for task in tasks:
            task.cancel()
        raise


from typing import Optional
```

## Optimized Task Service

```python
# src/ai_todo/services/optimized_task_service.py
"""
Performance-optimized task service.
"""

import asyncio
from typing import List, Optional
from datetime import datetime

from ..models.task import TaskInput, Task
from ..ai.cached_client import CachedOllamaClient
from ..cache.response_cache import ResponseCache
from ..utils.parallel import parallel_map


class OptimizedTaskService:
    """
    Task service with performance optimizations.
    
    Features:
    - Response caching
    - Parallel operations
    - Smart batching
    """
    
    def __init__(
        self,
        ai_client: CachedOllamaClient,
        repository,
        embedding_store=None,
        response_cache: Optional[ResponseCache] = None
    ):
        self.ai = ai_client
        self.repo = repository
        self.embeddings = embedding_store
        self.cache = response_cache
    
    async def create_tasks_batch(
        self,
        inputs: List[TaskInput],
        *,
        max_concurrent: int = 3
    ) -> List[Task]:
        """
        Create multiple tasks efficiently.
        
        Parallelizes AI calls with bounded concurrency.
        """
        # Process AI extraction in parallel
        proposals = await parallel_map(
            inputs,
            self._extract_single,
            max_concurrent=max_concurrent
        )
        
        # Create tasks
        tasks = [Task.from_proposal(p) for p in proposals]
        
        # Batch save to database
        for task in tasks:
            await self.repo.save(task)
        
        # Batch index in vector store
        if self.embeddings:
            await self._batch_index(tasks)
        
        return tasks
    
    async def _extract_single(self, input: TaskInput):
        """Extract single task (called in parallel)."""
        return await self.ai.extract_task(input.raw_input)
    
    async def _batch_index(self, tasks: List[Task]):
        """Index multiple tasks efficiently."""
        texts = [f"{t.title} {t.description or ''}" for t in tasks]
        ids = [t.id for t in tasks]
        
        await self.embeddings.index_batch(texts, ids)
    
    async def search_with_cache(
        self,
        query: str,
        limit: int = 10
    ) -> List[Task]:
        """
        Search with embedding cache.
        """
        if not self.embeddings:
            return await self.repo.search_by_title(query, limit=limit)
        
        # Embeddings are cached automatically
        similar = await self.embeddings.find_similar(query, limit=limit)
        
        # Fetch tasks in parallel
        task_ids = [item["id"] for item in similar]
        
        tasks = await parallel_map(
            task_ids,
            self.repo.get,
            max_concurrent=10
        )
        
        return [t for t in tasks if t is not None]
```

## Performance Monitoring

```python
# src/ai_todo/utils/metrics.py
"""
Performance metrics collection.
"""

import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager
import logging


logger = logging.getLogger(__name__)


@dataclass
class OperationMetrics:
    """Metrics for a single operation type."""
    count: int = 0
    total_ms: float = 0
    min_ms: float = float("inf")
    max_ms: float = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0


class MetricsCollector:
    """
    Collect performance metrics.
    """
    
    def __init__(self):
        self._metrics: Dict[str, OperationMetrics] = {}
    
    @contextmanager
    def measure(self, operation: str, cached: bool = False):
        """
        Context manager to measure operation duration.
        
        Usage:
            with metrics.measure("ai_extraction"):
                result = await ai.extract(...)
        """
        if operation not in self._metrics:
            self._metrics[operation] = OperationMetrics()
        
        metrics = self._metrics[operation]
        
        start = time.perf_counter()
        
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            
            metrics.count += 1
            metrics.total_ms += duration_ms
            metrics.min_ms = min(metrics.min_ms, duration_ms)
            metrics.max_ms = max(metrics.max_ms, duration_ms)
            
            if cached:
                metrics.cache_hits += 1
            else:
                metrics.cache_misses += 1
            
            logger.debug(f"{operation}: {duration_ms:.1f}ms (cached={cached})")
    
    def record_cache_hit(self, operation: str):
        """Record a cache hit."""
        if operation not in self._metrics:
            self._metrics[operation] = OperationMetrics()
        self._metrics[operation].cache_hits += 1
    
    def get_summary(self) -> Dict[str, dict]:
        """Get metrics summary."""
        return {
            name: {
                "count": m.count,
                "avg_ms": round(m.avg_ms, 1),
                "min_ms": round(m.min_ms, 1) if m.min_ms != float("inf") else 0,
                "max_ms": round(m.max_ms, 1),
                "cache_hit_rate": round(m.cache_hit_rate * 100, 1)
            }
            for name, m in self._metrics.items()
        }
    
    def log_summary(self):
        """Log metrics summary."""
        summary = self.get_summary()
        
        logger.info("Performance Summary:")
        for name, stats in summary.items():
            logger.info(
                f"  {name}: {stats['count']} calls, "
                f"avg={stats['avg_ms']}ms, "
                f"cache={stats['cache_hit_rate']}%"
            )


# Global metrics instance
metrics = MetricsCollector()
```

## CLI with Performance Stats

```python
# src/ai_todo/cli/app.py (performance additions)
"""
CLI with performance reporting.
"""

@app.command()
def stats():
    """Show performance statistics."""
    from ..utils.metrics import metrics
    from ..cache.response_cache import ResponseCache
    
    console.print("\n[bold]Performance Statistics[/bold]\n")
    
    # Operation metrics
    summary = metrics.get_summary()
    
    if summary:
        table = Table(title="Operation Metrics")
        table.add_column("Operation")
        table.add_column("Count", justify="right")
        table.add_column("Avg (ms)", justify="right")
        table.add_column("Min (ms)", justify="right")
        table.add_column("Max (ms)", justify="right")
        table.add_column("Cache %", justify="right")
        
        for name, stats in summary.items():
            table.add_row(
                name,
                str(stats["count"]),
                f"{stats['avg_ms']:.1f}",
                f"{stats['min_ms']:.1f}",
                f"{stats['max_ms']:.1f}",
                f"{stats['cache_hit_rate']:.0f}%"
            )
        
        console.print(table)
    else:
        console.print("[dim]No metrics collected yet[/dim]")
    
    # Cache statistics
    console.print()
    
    app = get_app()
    if hasattr(app, "response_cache"):
        cache_stats = app.response_cache.stats()
        
        console.print("[bold]Cache Statistics[/bold]")
        console.print(f"  Memory entries: {cache_stats['memory_entries']}")
        console.print(f"  Disk entries: {cache_stats['disk_entries']}")
        console.print(f"  Total hits: {cache_stats['total_hits']}")


@app.command()
def benchmark(
    count: int = typer.Option(10, help="Number of tasks to create"),
    parallel: bool = typer.Option(False, help="Use parallel processing")
):
    """Run performance benchmark."""
    import time
    
    async def _benchmark():
        app = get_app()
        
        test_inputs = [
            "buy groceries tomorrow",
            "finish report by Friday",
            "call mom in the afternoon",
            "schedule dentist appointment",
            "review pull request urgent",
        ]
        
        # Generate enough inputs
        inputs = [
            TaskInput(raw_input=test_inputs[i % len(test_inputs)])
            for i in range(count)
        ]
        
        console.print(f"\n[bold]Benchmark: Creating {count} tasks[/bold]")
        console.print(f"Mode: {'parallel' if parallel else 'sequential'}\n")
        
        start = time.perf_counter()
        
        if parallel:
            tasks = await app.tasks.create_tasks_batch(inputs, max_concurrent=3)
        else:
            tasks = []
            for input in inputs:
                result = await app.tasks.create_task(input)
                tasks.append(result.task)
        
        duration = time.perf_counter() - start
        
        console.print(f"[green]✓ Created {len(tasks)} tasks[/green]")
        console.print(f"\nTotal time: {duration:.2f}s")
        console.print(f"Average per task: {duration/count*1000:.0f}ms")
        console.print(f"Throughput: {count/duration:.1f} tasks/sec")
        
        # Clean up benchmark tasks
        for task in tasks:
            await app.tasks.delete_task(task.id)
        
        await app.close()
    
    run_async(_benchmark())
```

## Summary

In this chapter we built:

1. **Response cache** — Avoid redundant AI calls for identical prompts
2. **Embedding cache** — Cache expensive vector computations
3. **Cached AI client** — Transparent caching for low-temperature requests
4. **Batch processing** — Efficient bulk operations
5. **Parallel execution** — Bounded concurrency for throughput
6. **Performance metrics** — Track and report operation latencies

Key optimization principles:
- **Cache deterministic operations** (low temperature, embeddings)
- **Batch when possible** (database writes, embeddings)
- **Parallelize independent work** (AI calls, searches)
- **Measure everything** (can't optimize what you don't measure)

---

**Previous**: [Chapter 18: Error Handling](./chapter-18-error-handling.md)  
**Next**: [Chapter 20: Conclusion](./chapter-20-conclusion.md)
