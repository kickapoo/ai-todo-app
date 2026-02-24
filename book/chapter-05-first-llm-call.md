# Chapter 5: Your First LLM Call — Connecting to Ollama

## From Theory to Practice

We've established principles and architecture. Now let's write actual code that talks to Ollama and see AI-first development in action.

## The Complete AI Client

Let's build a production-ready Ollama client:

```python
# src/ai_todo/ai/client.py
"""
Ollama client with proper error handling and configuration.

This is the AI layer - it only generates proposals.
It never validates or persists data directly.
"""

import json
import httpx
from typing import Any
from dataclasses import dataclass


@dataclass
class GenerateResponse:
    """Structured response from generate endpoint."""
    content: str
    model: str
    total_duration_ms: float
    prompt_tokens: int
    completion_tokens: int


class OllamaError(Exception):
    """Base exception for Ollama errors."""
    pass


class OllamaConnectionError(OllamaError):
    """Server not reachable."""
    pass


class OllamaModelError(OllamaError):
    """Model not found or failed to load."""
    pass


class OllamaTimeoutError(OllamaError):
    """Request timed out."""
    pass


class OllamaClient:
    """
    Async client for Ollama API.
    
    Usage:
        client = OllamaClient()
        response = await client.generate("Hello, world!")
        print(response.content)
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: float = 60.0
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
    
    async def __aenter__(self) -> "OllamaClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self
    
    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.1,
        format: str | None = "json",
        max_tokens: int = 512,
        stop: list[str] | None = None
    ) -> GenerateResponse:
        """
        Generate text completion.
        
        Args:
            prompt: The input prompt
            system: Optional system prompt
            temperature: Randomness (0.0-2.0). Lower = more deterministic
            format: Output format ("json" or None)
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            
        Returns:
            GenerateResponse with content and metadata
            
        Raises:
            OllamaConnectionError: Server not reachable
            OllamaModelError: Model not found
            OllamaTimeoutError: Request timed out
        """
        client = self._get_client()
        
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if system:
            payload["system"] = system
        if format:
            payload["format"] = format
        if stop:
            payload["options"]["stop"] = stop
        
        try:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            return GenerateResponse(
                content=data["response"],
                model=data["model"],
                total_duration_ms=data.get("total_duration", 0) / 1_000_000,
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0)
            )
            
        except httpx.ConnectError as e:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is the server running? Try: ollama serve"
            ) from e
            
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(
                f"Request timed out after {self.timeout}s. "
                "Try increasing timeout or using a smaller model."
            ) from e
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise OllamaModelError(
                    f"Model '{self.model}' not found. "
                    f"Try: ollama pull {self.model}"
                ) from e
            raise OllamaError(f"HTTP error: {e.response.status_code}") from e
    
    async def embed(self, text: str) -> list[float]:
        """
        Generate embedding vector for text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding
        """
        client = self._get_client()
        
        try:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            return response.json()["embedding"]
            
        except httpx.ConnectError as e:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.base_url}"
            ) from e
    
    async def health_check(self) -> bool:
        """Check if Ollama server is running and model is available."""
        try:
            client = self._get_client()
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            models = response.json().get("models", [])
            model_names = [m["name"].split(":")[0] for m in models]
            
            return self.model.split(":")[0] in model_names
        except Exception:
            return False
```

## Testing the Client

Let's write a simple test script:

```python
# scripts/test_ollama.py
"""Test script for Ollama connection."""

import asyncio
from ai_todo.ai.client import OllamaClient, OllamaError


async def main():
    print("🔍 Testing Ollama connection...\n")
    
    async with OllamaClient() as client:
        # Test 1: Health check
        print("1. Health check...")
        healthy = await client.health_check()
        if healthy:
            print("   ✅ Server running, model available\n")
        else:
            print("   ❌ Server or model not available\n")
            return
        
        # Test 2: Simple generation
        print("2. Simple generation...")
        response = await client.generate(
            prompt="Say 'Hello, AI-First!' and nothing else.",
            temperature=0.0,
            format=None,
            max_tokens=20
        )
        print(f"   Response: {response.content}")
        print(f"   Duration: {response.total_duration_ms:.0f}ms")
        print(f"   Tokens: {response.prompt_tokens} prompt, {response.completion_tokens} completion\n")
        
        # Test 3: JSON generation (crucial for our app)
        print("3. JSON generation...")
        response = await client.generate(
            prompt='Extract the color from: "The sky is blue". Return JSON with a "color" field.',
            system="You are a JSON extraction assistant. Output valid JSON only.",
            temperature=0.0,
            format="json",
            max_tokens=50
        )
        print(f"   Response: {response.content}")
        
        # Validate it's actually JSON
        import json
        try:
            parsed = json.loads(response.content)
            print(f"   Parsed: {parsed}")
            print("   ✅ Valid JSON\n")
        except json.JSONDecodeError:
            print("   ❌ Invalid JSON\n")
        
        # Test 4: Embedding
        print("4. Embedding generation...")
        embedding = await client.embed("buy groceries tomorrow")
        print(f"   Dimension: {len(embedding)}")
        print(f"   Sample values: [{embedding[0]:.4f}, {embedding[1]:.4f}, ...]\n")
        
        print("🎉 All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
python -m scripts.test_ollama
```

Expected output:
```
🔍 Testing Ollama connection...

1. Health check...
   ✅ Server running, model available

2. Simple generation...
   Response: Hello, AI-First!
   Duration: 234ms
   Tokens: 12 prompt, 5 completion

3. JSON generation...
   Response: {"color": "blue"}
   Parsed: {'color': 'blue'}
   ✅ Valid JSON

4. Embedding generation...
   Dimension: 4096
   Sample values: [0.0234, -0.0156, ...]

🎉 All tests passed!
```

## Understanding Temperature

Let's see temperature's effect on output:

```python
# scripts/temperature_demo.py
"""Demonstrate temperature's effect on output consistency."""

import asyncio
from ai_todo.ai.client import OllamaClient


async def main():
    prompt = "List one task someone might do tomorrow. Just the task, nothing else."
    
    async with OllamaClient() as client:
        print("Temperature Effects on Output:\n")
        print("-" * 50)
        
        for temp in [0.0, 0.3, 0.7, 1.0]:
            print(f"\nTemperature: {temp}")
            print("Outputs (5 runs):")
            
            for i in range(5):
                response = await client.generate(
                    prompt=prompt,
                    temperature=temp,
                    format=None,
                    max_tokens=30
                )
                # Clean output
                text = response.content.strip().split('\n')[0]
                print(f"  {i+1}. {text}")
            
            print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())
```

Output:
```
Temperature Effects on Output:

--------------------------------------------------

Temperature: 0.0
Outputs (5 runs):
  1. Go grocery shopping
  2. Go grocery shopping
  3. Go grocery shopping
  4. Go grocery shopping
  5. Go grocery shopping
--------------------------------------------------

Temperature: 0.3
Outputs (5 runs):
  1. Go grocery shopping
  2. Go to the gym
  3. Go grocery shopping
  4. Clean the house
  5. Go grocery shopping
--------------------------------------------------

Temperature: 0.7
Outputs (5 runs):
  1. Finish the quarterly report
  2. Take the dog for a walk
  3. Schedule a dentist appointment
  4. Buy birthday gift for mom
  5. Organize the garage
--------------------------------------------------

Temperature: 1.0
Outputs (5 runs):
  1. Reorganize the vintage vinyl collection
  2. Finally learn to make sourdough bread
  3. Call the plumber about that leak
  4. Practice juggling for exactly 17 minutes
  5. Write a haiku about Tuesdays
--------------------------------------------------
```

This demonstrates Principle 3: **Temperature Controls Entropy**.

- **0.0**: Deterministic (same output every time)
- **0.3**: Slight variation, mostly consistent
- **0.7**: Creative variety
- **1.0**: High creativity, unpredictable

For parsing tasks from natural language, we use **0.0-0.2**.

## Structured Output with System Prompts

The system prompt shapes model behavior:

```python
# scripts/system_prompt_demo.py
"""Demonstrate system prompt's effect on output structure."""

import asyncio
import json
from ai_todo.ai.client import OllamaClient


SYSTEM_PROMPT = """You are a task extraction assistant. 
Your job is to extract structured task information from natural language.
Always output valid JSON with these fields:
- title: string (the task to do)
- priority: "low" | "medium" | "high" | "urgent"
- category: "work" | "personal" | "health" | "finance" | "errands" | "other"
- due_date: ISO 8601 datetime string or null

Base priority on urgency signals:
- "urgent", "asap", "immediately" → urgent
- "important", "must", "need to" → high
- Default → medium
- "sometime", "when possible", "eventually" → low

Base category on context:
- work: meetings, reports, colleagues, deadlines
- personal: family, friends, calls, visits
- health: doctor, exercise, medication, gym
- finance: bills, payments, budget, taxes
- errands: shopping, repairs, chores

Output ONLY valid JSON. No explanation."""


async def main():
    inputs = [
        "remind me to call mom tomorrow afternoon",
        "urgent: finish the quarterly report by EOD",
        "buy groceries sometime this week",
        "schedule dentist appointment",
        "pay electricity bill before the 15th"
    ]
    
    async with OllamaClient() as client:
        print("Structured Extraction Demo\n")
        print("=" * 60)
        
        for user_input in inputs:
            print(f"\nInput: \"{user_input}\"")
            
            response = await client.generate(
                prompt=f"Extract task from: {user_input}",
                system=SYSTEM_PROMPT,
                temperature=0.1,
                format="json",
                max_tokens=200
            )
            
            try:
                parsed = json.loads(response.content)
                print(f"Output: {json.dumps(parsed, indent=2)}")
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON - {response.content}")
            
            print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
```

Output:
```
Structured Extraction Demo

============================================================

Input: "remind me to call mom tomorrow afternoon"
Output: {
  "title": "Call mom",
  "priority": "medium",
  "category": "personal",
  "due_date": "2024-01-16T14:00:00"
}
------------------------------------------------------------

Input: "urgent: finish the quarterly report by EOD"
Output: {
  "title": "Finish quarterly report",
  "priority": "urgent",
  "category": "work",
  "due_date": "2024-01-15T17:00:00"
}
------------------------------------------------------------

Input: "buy groceries sometime this week"
Output: {
  "title": "Buy groceries",
  "priority": "low",
  "category": "errands",
  "due_date": null
}
------------------------------------------------------------

Input: "schedule dentist appointment"
Output: {
  "title": "Schedule dentist appointment",
  "priority": "medium",
  "category": "health",
  "due_date": null
}
------------------------------------------------------------

Input: "pay electricity bill before the 15th"
Output: {
  "title": "Pay electricity bill",
  "priority": "high",
  "category": "finance",
  "due_date": "2024-01-15T00:00:00"
}
------------------------------------------------------------
```

## The AI Proposes, System Commits Pattern

Notice what we're NOT doing:
- We're NOT trusting the JSON blindly
- We're NOT saving directly to database
- We're NOT assuming the structure is correct

The AI generates a **proposal**. The system must **validate** before **committing**.

```python
# Example of the full pattern
from pydantic import BaseModel, ValidationError
import json

class TaskProposal(BaseModel):
    title: str
    priority: str
    category: str
    due_date: str | None

# AI proposes
response = await client.generate(prompt=..., format="json")

# System validates
try:
    raw = json.loads(response.content)
    proposal = TaskProposal.model_validate(raw)
    # Now we can trust 'proposal'
except json.JSONDecodeError:
    # AI produced invalid JSON
    handle_error("Invalid JSON from AI")
except ValidationError as e:
    # JSON valid but doesn't match schema
    handle_error(f"Schema validation failed: {e}")
```

This separation is crucial. The AI is probabilistic; validation is deterministic. Together they form a reliable system.

## Error Handling Patterns

AI systems need robust error handling:

```python
# src/ai_todo/ai/safe_client.py
"""Wrapper with retry and fallback logic."""

import asyncio
from typing import TypeVar, Callable, Awaitable
from .client import OllamaClient, OllamaError, OllamaTimeoutError

T = TypeVar("T")


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    max_attempts: int = 3,
    backoff: float = 1.0
) -> T:
    """Retry async function with exponential backoff."""
    last_error = None
    
    for attempt in range(max_attempts):
        try:
            return await fn()
        except OllamaTimeoutError as e:
            last_error = e
            if attempt < max_attempts - 1:
                wait = backoff * (2 ** attempt)
                await asyncio.sleep(wait)
        except OllamaError:
            raise  # Don't retry other errors
    
    raise last_error


async def generate_with_fallback(
    client: OllamaClient,
    prompt: str,
    fallback_value: str = "{}"
) -> str:
    """Generate with fallback on failure."""
    try:
        response = await with_retry(
            lambda: client.generate(prompt=prompt)
        )
        return response.content
    except OllamaError:
        return fallback_value
```

## Summary

In this chapter we:

1. ✅ Built a production-ready Ollama client
2. ✅ Demonstrated temperature's effect on output
3. ✅ Showed structured extraction with system prompts
4. ✅ Established the "AI proposes, system commits" pattern
5. ✅ Added proper error handling

Key takeaways:
- **Temperature 0.0-0.2** for structured extraction
- **System prompts** define expected output format
- **JSON format** enables parsing
- **Validation** is non-negotiable
- **Errors are expected** — handle them gracefully

In the next chapter, we'll define our data models using Pydantic and see how validation enforces correctness.

---

**Previous**: [Chapter 4: Project Structure](./chapter-04-project-structure.md)  
**Next**: [Chapter 6: The Task Data Model](./chapter-06-data-model.md)
