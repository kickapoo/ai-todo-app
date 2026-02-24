# Chapter 18: Error Handling — Graceful Degradation

## The Reality of AI Systems

AI systems fail in ways traditional software doesn't:

- **Model unavailable** — Ollama not running, model not pulled
- **Timeout** — Large prompts, slow hardware
- **Invalid output** — JSON parsing fails, missing fields
- **Semantic errors** — Valid structure, wrong meaning
- **Resource exhaustion** — Memory limits, token limits

This chapter builds robust error handling that degrades gracefully.

## Error Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                      ERROR CATEGORIES                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   INFRASTRUCTURE           AI LAYER              BUSINESS        │
│   ├── Connection           ├── Timeout           ├── Validation  │
│   ├── Model missing        ├── Parse error       ├── Constraint  │
│   └── Service down         ├── Token limit       └── Business rule│
│                            └── Semantic error                    │
│                                                                  │
│   ↓ Recovery               ↓ Recovery            ↓ Recovery      │
│   Wait + Retry             Fallback + Retry      Reject + Report │
│   Alert user               Use defaults                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Custom Exceptions

```python
# src/ai_todo/errors.py
"""
Application-specific exceptions.

Each exception type has clear semantics and recovery strategies.
"""

from typing import Optional, Any
from dataclasses import dataclass


# ==============================================================================
# BASE EXCEPTIONS
# ==============================================================================

class AITodoError(Exception):
    """Base exception for all application errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            return f"{self.message} | {self.details}"
        return self.message


class RetryableError(AITodoError):
    """
    Error that may succeed on retry.
    
    The system should wait and try again.
    """
    
    def __init__(
        self,
        message: str,
        retry_after_seconds: float = 1.0,
        max_retries: int = 3,
        details: Optional[dict] = None
    ):
        super().__init__(message, details)
        self.retry_after_seconds = retry_after_seconds
        self.max_retries = max_retries


class FatalError(AITodoError):
    """
    Error that cannot be recovered from.
    
    The operation should fail immediately.
    """
    pass


# ==============================================================================
# INFRASTRUCTURE ERRORS
# ==============================================================================

class InfrastructureError(RetryableError):
    """Base class for infrastructure failures."""
    pass


class OllamaConnectionError(InfrastructureError):
    """Failed to connect to Ollama service."""
    
    def __init__(self, url: str, original_error: Optional[Exception] = None):
        super().__init__(
            f"Cannot connect to Ollama at {url}",
            retry_after_seconds=2.0,
            max_retries=3,
            details={
                "url": url,
                "original_error": str(original_error) if original_error else None,
                "hint": "Is Ollama running? Try: ollama serve"
            }
        )


class ModelNotFoundError(FatalError):
    """Requested model is not available."""
    
    def __init__(self, model: str):
        super().__init__(
            f"Model '{model}' not found",
            details={
                "model": model,
                "hint": f"Pull the model with: ollama pull {model}"
            }
        )


class DatabaseError(InfrastructureError):
    """Database operation failed."""
    
    def __init__(self, operation: str, original_error: Exception):
        super().__init__(
            f"Database {operation} failed: {original_error}",
            retry_after_seconds=0.5,
            max_retries=2,
            details={"operation": operation}
        )


# ==============================================================================
# AI LAYER ERRORS
# ==============================================================================

class AIError(AITodoError):
    """Base class for AI-related errors."""
    pass


class AITimeoutError(RetryableError):
    """AI operation timed out."""
    
    def __init__(self, timeout_seconds: float, prompt_length: int):
        super().__init__(
            f"AI request timed out after {timeout_seconds}s",
            retry_after_seconds=1.0,
            max_retries=2,
            details={
                "timeout_seconds": timeout_seconds,
                "prompt_length": prompt_length,
                "hint": "Try a shorter prompt or increase timeout"
            }
        )


class AIParseError(AIError):
    """Failed to parse AI response."""
    
    def __init__(self, expected_format: str, actual_content: str):
        # Truncate content for error message
        truncated = actual_content[:100] + "..." if len(actual_content) > 100 else actual_content
        
        super().__init__(
            f"Failed to parse AI response as {expected_format}",
            details={
                "expected": expected_format,
                "received": truncated
            }
        )


class AIValidationError(AIError):
    """AI output failed validation."""
    
    def __init__(self, validation_errors: list, ai_output: dict):
        super().__init__(
            "AI output failed validation",
            details={
                "errors": validation_errors,
                "output": ai_output
            }
        )


class TokenLimitError(AIError):
    """Request exceeds token limit."""
    
    def __init__(self, estimated_tokens: int, limit: int):
        super().__init__(
            f"Prompt exceeds token limit ({estimated_tokens} > {limit})",
            details={
                "estimated_tokens": estimated_tokens,
                "limit": limit,
                "hint": "Reduce prompt size or context"
            }
        )


# ==============================================================================
# BUSINESS ERRORS
# ==============================================================================

class BusinessError(AITodoError):
    """Base class for business rule violations."""
    pass


class TaskNotFoundError(BusinessError):
    """Requested task does not exist."""
    
    def __init__(self, task_id: str):
        super().__init__(
            f"Task not found: {task_id}",
            details={"task_id": task_id}
        )


class InvalidInputError(BusinessError):
    """User input is invalid."""
    
    def __init__(self, field: str, reason: str):
        super().__init__(
            f"Invalid {field}: {reason}",
            details={"field": field, "reason": reason}
        )


class DuplicateTaskError(BusinessError):
    """Task already exists."""
    
    def __init__(self, task_id: str, existing_title: str):
        super().__init__(
            f"Duplicate task detected",
            details={
                "task_id": task_id,
                "existing_title": existing_title
            }
        )
```

## Retry Logic

```python
# src/ai_todo/utils/retry.py
"""
Retry utilities for resilient operations.
"""

import asyncio
import logging
from typing import TypeVar, Callable, Awaitable, Optional
from functools import wraps

from ..errors import RetryableError, FatalError


logger = logging.getLogger(__name__)

T = TypeVar("T")


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    exponential_backoff: bool = True,
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> T:
    """
    Retry an async operation with configurable backoff.
    
    Args:
        operation: Async function to retry
        max_retries: Maximum retry attempts
        base_delay: Initial delay between retries
        exponential_backoff: Double delay on each retry
        on_retry: Callback for retry events
        
    Returns:
        Result of successful operation
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await operation()
        
        except FatalError:
            # Don't retry fatal errors
            raise
        
        except RetryableError as e:
            last_exception = e
            
            if attempt < max_retries:
                delay = e.retry_after_seconds
                if exponential_backoff:
                    delay *= (2 ** attempt)
                
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {e.message}"
                )
                
                if on_retry:
                    on_retry(e, attempt + 1)
                
                await asyncio.sleep(delay)
            
        except Exception as e:
            # Unexpected error - don't retry
            raise
    
    raise last_exception


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    exponential_backoff: bool = True
):
    """
    Decorator for retryable async functions.
    
    Usage:
        @with_retry(max_retries=3)
        async def fetch_data():
            ...
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            async def operation():
                return await func(*args, **kwargs)
            
            return await retry_async(
                operation,
                max_retries=max_retries,
                base_delay=base_delay,
                exponential_backoff=exponential_backoff
            )
        return wrapper
    return decorator
```

## Fallback Strategies

```python
# src/ai_todo/utils/fallback.py
"""
Fallback strategies for graceful degradation.
"""

from typing import TypeVar, Callable, Awaitable, Optional
from datetime import datetime
import logging

from ..models.task import TaskProposal
from ..models.enums import Priority, Category


logger = logging.getLogger(__name__)

T = TypeVar("T")


async def with_fallback(
    operation: Callable[[], Awaitable[T]],
    fallback: Callable[[], T],
    *,
    log_warning: bool = True
) -> T:
    """
    Execute operation with fallback on failure.
    
    Args:
        operation: Primary async operation
        fallback: Fallback function (sync)
        log_warning: Whether to log fallback usage
        
    Returns:
        Result from operation or fallback
    """
    try:
        return await operation()
    except Exception as e:
        if log_warning:
            logger.warning(f"Using fallback due to: {e}")
        return fallback()


class TaskExtractionFallback:
    """
    Fallback for task extraction when AI fails.
    
    Uses simple heuristics instead of AI.
    """
    
    # Keywords suggesting priority
    URGENT_KEYWORDS = {"urgent", "asap", "critical", "emergency", "now"}
    HIGH_KEYWORDS = {"important", "high", "priority", "must"}
    LOW_KEYWORDS = {"sometime", "eventually", "maybe", "whenever"}
    
    # Keywords suggesting category
    CATEGORY_KEYWORDS = {
        Category.WORK: {"work", "meeting", "report", "deadline", "project", "client"},
        Category.PERSONAL: {"call", "mom", "dad", "family", "friend"},
        Category.ERRANDS: {"buy", "groceries", "shop", "pick up", "return"},
        Category.HEALTH: {"gym", "doctor", "dentist", "exercise", "workout"},
        Category.FINANCE: {"pay", "bill", "bank", "invoice", "budget"},
        Category.LEARNING: {"learn", "study", "read", "course", "tutorial"},
    }
    
    @classmethod
    def extract(cls, raw_input: str) -> TaskProposal:
        """
        Extract task using heuristics when AI unavailable.
        
        This provides degraded but functional behavior.
        """
        text_lower = raw_input.lower()
        words = set(text_lower.split())
        
        # Infer priority
        priority = cls._infer_priority(words)
        
        # Infer category
        category = cls._infer_category(words)
        
        # Clean title
        title = cls._clean_title(raw_input)
        
        logger.info(f"Fallback extraction: '{title}' -> {priority.value}/{category.value}")
        
        return TaskProposal(
            title=title,
            description=None,
            priority=priority,
            category=category,
            due_date=None
        )
    
    @classmethod
    def _infer_priority(cls, words: set) -> Priority:
        """Infer priority from keywords."""
        if words & cls.URGENT_KEYWORDS:
            return Priority.URGENT
        if words & cls.HIGH_KEYWORDS:
            return Priority.HIGH
        if words & cls.LOW_KEYWORDS:
            return Priority.LOW
        return Priority.MEDIUM
    
    @classmethod
    def _infer_category(cls, words: set) -> Category:
        """Infer category from keywords."""
        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            if words & keywords:
                return category
        return Category.PERSONAL  # Default
    
    @classmethod
    def _clean_title(cls, raw_input: str) -> str:
        """Clean up title by removing common prefixes."""
        # Remove common prefixes
        prefixes = [
            "remind me to ",
            "reminder to ",
            "need to ",
            "i need to ",
            "urgent: ",
            "urgent ",
            "todo: ",
            "task: ",
        ]
        
        title = raw_input.strip()
        title_lower = title.lower()
        
        for prefix in prefixes:
            if title_lower.startswith(prefix):
                title = title[len(prefix):]
                break
        
        # Capitalize first letter
        if title:
            title = title[0].upper() + title[1:]
        
        return title[:200]  # Enforce max length
```

## Error Handling in Services

```python
# src/ai_todo/services/task_service.py (error handling additions)
"""
Task service with comprehensive error handling.
"""

import logging
from typing import Optional

from ..errors import (
    AIError, AITimeoutError, AIParseError, AIValidationError,
    TaskNotFoundError, OllamaConnectionError
)
from ..utils.retry import retry_async
from ..utils.fallback import TaskExtractionFallback, with_fallback


logger = logging.getLogger(__name__)


class TaskService:
    # ... existing code ...
    
    async def create_task_safe(
        self,
        input: TaskInput,
        *,
        use_fallback: bool = True,
        max_retries: int = 2
    ) -> TaskCreationResult:
        """
        Create task with comprehensive error handling.
        
        Features:
        - Retries on transient failures
        - Falls back to heuristics if AI fails
        - Always returns a result (never raises to caller)
        """
        try:
            # Try with retries
            async def ai_operation():
                return await self.ai.extract_task(
                    input.raw_input,
                    safe_mode=True
                )
            
            proposal = await retry_async(
                ai_operation,
                max_retries=max_retries,
                base_delay=1.0
            )
            
        except OllamaConnectionError as e:
            logger.error(f"Ollama unavailable: {e}")
            
            if use_fallback:
                proposal = TaskExtractionFallback.extract(input.raw_input)
            else:
                raise
                
        except AITimeoutError as e:
            logger.error(f"AI timeout: {e}")
            
            if use_fallback:
                proposal = TaskExtractionFallback.extract(input.raw_input)
            else:
                raise
                
        except (AIParseError, AIValidationError) as e:
            logger.error(f"AI output invalid: {e}")
            
            if use_fallback:
                proposal = TaskExtractionFallback.extract(input.raw_input)
            else:
                raise
        
        # Proceed with validated proposal
        task = Task.from_proposal(proposal)
        await self.repo.save(task)
        
        return TaskCreationResult(
            task=task,
            proposal=proposal,
            ai_duration_ms=0,  # No AI timing in fallback
            similar_tasks=[]
        )
    
    async def get_task_or_raise(self, task_id: str) -> Task:
        """Get task or raise TaskNotFoundError."""
        task = await self.repo.get(task_id)
        
        if task is None:
            raise TaskNotFoundError(task_id)
        
        return task
```

## Error Handling in AI Client

```python
# src/ai_todo/ai/client.py (error handling additions)
"""
Ollama client with error handling.
"""

import httpx
from typing import Optional

from ..errors import (
    OllamaConnectionError, ModelNotFoundError,
    AITimeoutError, AIParseError
)


class OllamaClient:
    # ... existing code ...
    
    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        format: Optional[str] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> AIResponse:
        """
        Generate completion with error handling.
        """
        effective_timeout = timeout or self.timeout
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens or 512,
                    },
                    **({"format": format} if format else {})
                },
                timeout=effective_timeout
            )
            
        except httpx.ConnectError as e:
            raise OllamaConnectionError(self.base_url, e)
        
        except httpx.TimeoutException:
            raise AITimeoutError(effective_timeout, len(prompt))
        
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ModelNotFoundError(self.model)
            raise OllamaConnectionError(self.base_url, e)
        
        # Parse response
        try:
            data = response.json()
        except Exception as e:
            raise AIParseError("JSON", response.text)
        
        if "error" in data:
            error_msg = data["error"]
            if "model" in error_msg.lower() and "not found" in error_msg.lower():
                raise ModelNotFoundError(self.model)
            raise OllamaConnectionError(self.base_url, Exception(error_msg))
        
        return AIResponse(
            content=data.get("response", ""),
            model=self.model,
            tokens_used=data.get("eval_count", 0),
            duration_ms=data.get("total_duration", 0) / 1_000_000
        )
```

## CLI Error Display

```python
# src/ai_todo/cli/errors.py
"""
CLI error display with Rich formatting.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..errors import (
    AITodoError, RetryableError, FatalError,
    OllamaConnectionError, ModelNotFoundError,
    AITimeoutError, AIParseError,
    TaskNotFoundError
)


console = Console()


def display_error(error: Exception):
    """
    Display error with appropriate formatting.
    
    Different error types get different treatments.
    """
    if isinstance(error, OllamaConnectionError):
        _display_connection_error(error)
    elif isinstance(error, ModelNotFoundError):
        _display_model_error(error)
    elif isinstance(error, AITimeoutError):
        _display_timeout_error(error)
    elif isinstance(error, TaskNotFoundError):
        _display_not_found_error(error)
    elif isinstance(error, AITodoError):
        _display_generic_app_error(error)
    else:
        _display_unexpected_error(error)


def _display_connection_error(error: OllamaConnectionError):
    """Display connection error with recovery steps."""
    content = f"""[red bold]Cannot connect to Ollama[/red bold]

{error.message}

[bold]Recovery steps:[/bold]
1. Check if Ollama is running: [cyan]ollama serve[/cyan]
2. Verify the URL: {error.details.get('url', 'unknown')}
3. Check your network connection

[dim]The application will use fallback mode if available.[/dim]"""
    
    console.print(Panel(content, title="Connection Error", border_style="red"))


def _display_model_error(error: ModelNotFoundError):
    """Display model error with installation steps."""
    model = error.details.get("model", "unknown")
    
    content = f"""[red bold]Model not found: {model}[/red bold]

[bold]To install the model:[/bold]
[cyan]ollama pull {model}[/cyan]

[bold]Available models:[/bold]
[cyan]ollama list[/cyan]

[dim]Common models: llama3.2, llama3.2:1b, mistral, codellama[/dim]"""
    
    console.print(Panel(content, title="Model Error", border_style="red"))


def _display_timeout_error(error: AITimeoutError):
    """Display timeout error."""
    content = f"""[yellow bold]AI request timed out[/yellow bold]

Timeout: {error.details.get('timeout_seconds', '?')} seconds
Prompt length: {error.details.get('prompt_length', '?')} characters

[bold]Possible solutions:[/bold]
• Use a shorter prompt
• Try a smaller model (e.g., llama3.2:1b)
• Increase timeout setting

[dim]Retrying automatically...[/dim]"""
    
    console.print(Panel(content, title="Timeout", border_style="yellow"))


def _display_not_found_error(error: TaskNotFoundError):
    """Display task not found error."""
    task_id = error.details.get("task_id", "unknown")
    
    content = f"""[yellow]Task not found: {task_id}[/yellow]

[dim]Use 'todo list' to see available tasks.[/dim]"""
    
    console.print(Panel(content, border_style="yellow"))


def _display_generic_app_error(error: AITodoError):
    """Display generic application error."""
    content = f"[red]{error.message}[/red]"
    
    if error.details:
        content += "\n\n[dim]Details:[/dim]"
        for key, value in error.details.items():
            content += f"\n  {key}: {value}"
    
    console.print(Panel(content, title="Error", border_style="red"))


def _display_unexpected_error(error: Exception):
    """Display unexpected error."""
    content = f"""[red bold]Unexpected error[/red bold]

{type(error).__name__}: {error}

[dim]This is likely a bug. Please report it.[/dim]"""
    
    console.print(Panel(content, title="Error", border_style="red"))
```

## Integrating Error Handling in CLI

```python
# src/ai_todo/cli/app.py (error handling integration)
"""
CLI with comprehensive error handling.
"""

import typer
from .errors import display_error
from ..errors import AITodoError


@app.command()
def add(task: str = typer.Argument(...)):
    """Add a new task."""
    
    async def _add():
        try:
            application = get_app()
            
            # Use safe method with fallback
            result = await application.tasks.create_task_safe(
                TaskInput(raw_input=task),
                use_fallback=True
            )
            
            console.print(f"[green]✓ Task created[/green]")
            _display_task_card(result.task)
            
        except AITodoError as e:
            display_error(e)
            raise typer.Exit(1)
        
        except Exception as e:
            display_error(e)
            raise typer.Exit(1)
        
        finally:
            await application.close()
    
    run_async(_add())
```

## Logging Configuration

```python
# src/ai_todo/logging_config.py
"""
Logging configuration for error tracking.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None
):
    """
    Configure logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs
    """
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Configure root logger
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))
    
    # Console handler (errors only in production)
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.WARNING)
    console.setFormatter(formatter)
    root.addHandler(console)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def log_error_context(error: Exception, context: dict):
    """
    Log error with additional context.
    
    Useful for debugging production issues.
    """
    logger = logging.getLogger("ai_todo.errors")
    
    logger.error(
        f"{type(error).__name__}: {error}",
        extra={"context": context},
        exc_info=True
    )
```

## Summary

In this chapter we built:

1. **Exception hierarchy** — Clear categories: Infrastructure, AI, Business
2. **Retry logic** — Exponential backoff for transient failures
3. **Fallback strategies** — Heuristic-based degradation when AI fails
4. **Rich error display** — User-friendly error messages in CLI
5. **Logging configuration** — Error tracking for debugging

The key principle: **fail gracefully, degrade intelligently**.

When AI is unavailable:
1. Retry with backoff
2. Fall back to heuristics
3. Inform user clearly
4. Log for debugging

The system remains functional even when AI components fail.

---

**Previous**: [Chapter 17: Testing AI Systems](./chapter-17-testing.md)  
**Next**: [Chapter 19: Performance Optimization](./chapter-19-performance.md)
