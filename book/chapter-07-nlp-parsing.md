# Chapter 7: Natural Language Parsing — From Text to Structure

## The Core Challenge

Users say things like:
- "call mom tomorrow"
- "remind me to finish the report by Friday 5pm"
- "urgent - buy birthday present for Sarah"
- "maybe clean the garage sometime"

We need to extract:
- **What**: The task itself
- **When**: Due date (if mentioned)
- **Priority**: Urgency signals
- **Category**: Context clues

This is where AI shines—and where we need careful engineering.

## Prompt Design Principles

### 1. Be Specific About Output Format

```python
# Bad: Vague instruction
prompt = "Extract the task from this text"

# Good: Exact specification
prompt = """Extract task information and return valid JSON with these exact fields:
- title: string (the action to take)
- priority: "low" | "medium" | "high" | "urgent"  
- category: "work" | "personal" | "health" | "finance" | "errands" | "other"
- due_date: ISO 8601 datetime string or null"""
```

### 2. Provide Classification Rules

```python
# Bad: Let AI guess
"Determine the priority"

# Good: Explicit rules
"""Priority classification:
- urgent: Contains "urgent", "asap", "immediately", "emergency"
- high: Contains "important", "must", "need to", "critical"
- medium: Default for most tasks
- low: Contains "sometime", "eventually", "when possible", "maybe"
"""
```

### 3. Include Examples

```python
"""Examples:
Input: "call mom tomorrow afternoon"
Output: {"title": "Call mom", "priority": "medium", "category": "personal", "due_date": "2024-01-16T14:00:00"}

Input: "URGENT: fix production bug"
Output: {"title": "Fix production bug", "priority": "urgent", "category": "work", "due_date": null}
"""
```

## The Task Extraction Prompt

```python
# src/ai_todo/ai/prompts.py
"""Prompts for AI task extraction."""

from datetime import datetime


def get_task_extraction_prompt(current_datetime: datetime | None = None) -> str:
    """
    Generate the task extraction system prompt.
    
    Including current datetime helps the model resolve relative dates.
    """
    if current_datetime is None:
        current_datetime = datetime.now()
    
    current_date_str = current_datetime.strftime("%Y-%m-%d")
    current_time_str = current_datetime.strftime("%H:%M")
    current_weekday = current_datetime.strftime("%A")
    
    return f"""You are a task extraction assistant. Your job is to parse natural language task descriptions into structured JSON.

CURRENT CONTEXT:
- Today's date: {current_date_str} ({current_weekday})
- Current time: {current_time_str}

OUTPUT FORMAT:
Return valid JSON with exactly these fields:
{{
    "title": "string - concise task description (2-8 words)",
    "description": "string or null - additional details if present",
    "priority": "low" | "medium" | "high" | "urgent",
    "category": "work" | "personal" | "health" | "finance" | "learning" | "errands" | "home" | "social" | "other",
    "due_date": "ISO 8601 datetime string or null"
}}

PRIORITY RULES:
- urgent: "urgent", "asap", "immediately", "emergency", "critical now"
- high: "important", "must", "need to", "don't forget", "critical"
- medium: Default for most tasks
- low: "sometime", "eventually", "when possible", "maybe", "if I have time"

CATEGORY RULES:
- work: meetings, reports, colleagues, clients, deadlines, projects, emails
- personal: family, friends, calls, messages, relationships
- health: doctor, dentist, gym, exercise, medication, appointments, wellness
- finance: bills, payments, budget, taxes, banking, invoices
- learning: study, course, book, read, learn, tutorial, practice
- errands: shopping, groceries, pickup, drop-off, returns
- home: cleaning, repairs, maintenance, organizing, chores
- social: party, event, gathering, dinner, meetup
- other: anything that doesn't fit above

DATE PARSING RULES:
- "today" → {current_date_str}
- "tomorrow" → next day from today
- "next week" → 7 days from today
- "Monday", "Tuesday", etc. → next occurrence of that day
- "morning" → 09:00
- "afternoon" → 14:00
- "evening" → 18:00
- "EOD" / "end of day" → 17:00
- "next month" → first of next month
- No date mentioned → null

TITLE RULES:
- Start with a verb when possible (Call, Buy, Send, Complete, etc.)
- Remove filler words ("I need to", "remind me to", "don't forget to")
- Keep it concise: 2-8 words
- Capitalize first letter

OUTPUT ONLY VALID JSON. No explanations, no markdown, no extra text."""


TASK_EXTRACTION_USER_PROMPT = """Parse this task: "{user_input}"

Return JSON only:"""


# Additional prompts for specific operations

PRIORITY_INFERENCE_PROMPT = """Analyze this task and determine its priority based on:
1. Explicit urgency words
2. Implied urgency (deadlines, consequences)
3. Task nature (health/safety = higher priority)

Task: "{task_title}"
Context: "{context}"

Return JSON: {{"priority": "low|medium|high|urgent", "reasoning": "brief explanation"}}"""


CATEGORY_INFERENCE_PROMPT = """Categorize this task based on its content and context.

Task: "{task_title}"
Description: "{description}"

Categories: work, personal, health, finance, learning, errands, home, social, other

Return JSON: {{"category": "category_name", "confidence": 0.0-1.0}}"""
```

## The Parser Implementation

```python
# src/ai_todo/ai/parsers.py
"""Parse AI responses into structured data."""

import json
import re
from typing import Any
from datetime import datetime

from ..models.task import TaskProposal
from pydantic import ValidationError


class ParseError(Exception):
    """Failed to parse AI response."""
    pass


def extract_json(text: str) -> dict[str, Any]:
    """
    Extract JSON from AI response text.
    
    Handles common issues:
    - JSON wrapped in markdown code blocks
    - Leading/trailing whitespace
    - Multiple JSON objects (takes first)
    """
    text = text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```"):
        # Find the JSON content
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
        else:
            # Try removing just the backticks
            text = re.sub(r"```(?:json)?", "", text).strip()
    
    # Try to find JSON object
    try:
        # Direct parse
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON object from text
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # Try to fix common issues
    # Single quotes instead of double
    fixed = text.replace("'", '"')
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    raise ParseError(f"Could not extract JSON from: {text[:100]}...")


def parse_task_proposal(ai_response: str) -> TaskProposal:
    """
    Parse AI response into a validated TaskProposal.
    
    Args:
        ai_response: Raw text from Ollama
        
    Returns:
        Validated TaskProposal
        
    Raises:
        ParseError: If JSON extraction fails
        ValidationError: If data doesn't match schema
    """
    # Extract JSON
    data = extract_json(ai_response)
    
    # Validate and create proposal
    return TaskProposal.model_validate(data)


def parse_task_proposal_safe(
    ai_response: str,
    fallback_title: str = "Untitled task"
) -> TaskProposal:
    """
    Parse AI response with fallback for failures.
    
    Always returns a valid TaskProposal, using defaults if parsing fails.
    """
    try:
        return parse_task_proposal(ai_response)
    except (ParseError, ValidationError):
        # Return minimal valid proposal
        return TaskProposal(title=fallback_title)


def parse_json_safe(text: str, default: dict | None = None) -> dict:
    """Extract JSON with fallback to default."""
    try:
        return extract_json(text)
    except ParseError:
        return default or {}
```

## The AI Service

```python
# src/ai_todo/ai/service.py
"""AI service for task extraction and inference."""

from datetime import datetime
from typing import Optional

from .client import OllamaClient
from .prompts import get_task_extraction_prompt, TASK_EXTRACTION_USER_PROMPT
from .parsers import parse_task_proposal, parse_task_proposal_safe
from ..models.task import TaskProposal


class AIService:
    """
    Service for AI-powered task operations.
    
    Handles all LLM interactions with appropriate temperature settings.
    """
    
    def __init__(self, client: OllamaClient):
        self.client = client
    
    async def extract_task(
        self,
        user_input: str,
        *,
        current_datetime: Optional[datetime] = None,
        safe_mode: bool = True
    ) -> TaskProposal:
        """
        Extract structured task from natural language.
        
        Args:
            user_input: Natural language task description
            current_datetime: For resolving relative dates
            safe_mode: If True, return fallback on parse failure
            
        Returns:
            TaskProposal with extracted information
        """
        if current_datetime is None:
            current_datetime = datetime.now()
        
        # Build prompts
        system_prompt = get_task_extraction_prompt(current_datetime)
        user_prompt = TASK_EXTRACTION_USER_PROMPT.format(user_input=user_input)
        
        # Call LLM with low temperature for consistent parsing
        response = await self.client.generate(
            prompt=user_prompt,
            system=system_prompt,
            temperature=0.1,  # Low entropy for structured output
            format="json",
            max_tokens=256
        )
        
        # Parse response
        if safe_mode:
            return parse_task_proposal_safe(
                response.content,
                fallback_title=user_input[:50]
            )
        else:
            return parse_task_proposal(response.content)
    
    async def infer_priority(
        self,
        task_title: str,
        context: str = ""
    ) -> tuple[str, str]:
        """
        Infer priority for an existing task.
        
        Returns:
            Tuple of (priority, reasoning)
        """
        from .prompts import PRIORITY_INFERENCE_PROMPT
        from .parsers import parse_json_safe
        
        prompt = PRIORITY_INFERENCE_PROMPT.format(
            task_title=task_title,
            context=context
        )
        
        response = await self.client.generate(
            prompt=prompt,
            temperature=0.2,  # Slightly higher for reasoning
            format="json",
            max_tokens=128
        )
        
        data = parse_json_safe(response.content, {"priority": "medium", "reasoning": ""})
        return data.get("priority", "medium"), data.get("reasoning", "")
    
    async def infer_category(
        self,
        task_title: str,
        description: str = ""
    ) -> tuple[str, float]:
        """
        Infer category for an existing task.
        
        Returns:
            Tuple of (category, confidence)
        """
        from .prompts import CATEGORY_INFERENCE_PROMPT
        from .parsers import parse_json_safe
        
        prompt = CATEGORY_INFERENCE_PROMPT.format(
            task_title=task_title,
            description=description
        )
        
        response = await self.client.generate(
            prompt=prompt,
            temperature=0.2,
            format="json",
            max_tokens=64
        )
        
        data = parse_json_safe(response.content, {"category": "other", "confidence": 0.5})
        return data.get("category", "other"), data.get("confidence", 0.5)
```

## Testing the Parser

```python
# tests/test_parsing.py
"""Tests for natural language parsing."""

import pytest
from datetime import datetime
from ai_todo.ai.service import AIService
from ai_todo.ai.client import OllamaClient


@pytest.fixture
async def ai_service():
    """Create AI service for testing."""
    client = OllamaClient()
    return AIService(client)


class TestTaskExtraction:
    """Test task extraction from natural language."""
    
    @pytest.mark.asyncio
    async def test_simple_task(self, ai_service):
        """Simple task with no date."""
        result = await ai_service.extract_task("buy groceries")
        
        assert "grocer" in result.title.lower() or "buy" in result.title.lower()
        assert result.category.value == "errands"
    
    @pytest.mark.asyncio
    async def test_task_with_date(self, ai_service):
        """Task with relative date."""
        result = await ai_service.extract_task(
            "call mom tomorrow",
            current_datetime=datetime(2024, 1, 15, 10, 0)
        )
        
        assert "call" in result.title.lower() or "mom" in result.title.lower()
        assert result.due_date is not None
        assert result.due_date.date() == datetime(2024, 1, 16).date()
    
    @pytest.mark.asyncio
    async def test_urgent_task(self, ai_service):
        """Task with urgency signal."""
        result = await ai_service.extract_task("URGENT: fix the server")
        
        assert result.priority.value in ["urgent", "high"]
        assert result.category.value == "work"
    
    @pytest.mark.asyncio
    async def test_low_priority_task(self, ai_service):
        """Task with low priority signal."""
        result = await ai_service.extract_task("maybe clean the garage sometime")
        
        assert result.priority.value == "low"
    
    @pytest.mark.asyncio
    async def test_health_category(self, ai_service):
        """Health-related task."""
        result = await ai_service.extract_task("schedule dentist appointment")
        
        assert result.category.value == "health"
    
    @pytest.mark.asyncio
    async def test_finance_category(self, ai_service):
        """Finance-related task."""
        result = await ai_service.extract_task("pay electricity bill")
        
        assert result.category.value == "finance"
```

## Demo Script

```python
# scripts/parsing_demo.py
"""Demonstrate natural language parsing."""

import asyncio
from datetime import datetime
from rich.console import Console
from rich.table import Table

from ai_todo.ai.client import OllamaClient
from ai_todo.ai.service import AIService


console = Console()


async def main():
    test_inputs = [
        "call mom tomorrow afternoon",
        "URGENT: fix production bug immediately",
        "buy groceries sometime this week",
        "schedule dentist appointment for next Tuesday",
        "pay electricity bill before the 15th",
        "maybe clean the garage if I have time",
        "important: finish quarterly report by Friday EOD",
        "pick up dry cleaning",
        "read that book on machine learning",
        "dinner with Sarah on Saturday at 7pm",
    ]
    
    async with OllamaClient() as client:
        service = AIService(client)
        
        table = Table(title="Natural Language Parsing Results")
        table.add_column("Input", style="cyan", width=40)
        table.add_column("Title", style="green")
        table.add_column("Priority", style="yellow")
        table.add_column("Category", style="blue")
        table.add_column("Due Date", style="magenta")
        
        current_dt = datetime.now()
        
        for user_input in test_inputs:
            with console.status(f"Parsing: {user_input[:30]}..."):
                proposal = await service.extract_task(
                    user_input,
                    current_datetime=current_dt
                )
            
            due_str = (
                proposal.due_date.strftime("%m/%d %H:%M")
                if proposal.due_date else "—"
            )
            
            table.add_row(
                user_input,
                proposal.title,
                proposal.priority.value,
                proposal.category.value,
                due_str
            )
        
        console.print(table)


if __name__ == "__main__":
    asyncio.run(main())
```

Output:
```
              Natural Language Parsing Results
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Input                        ┃ Title            ┃ Priority ┃ Category ┃ Due Date   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━┩
│ call mom tomorrow afternoon  │ Call mom         │ medium   │ personal │ 01/16 14:00│
│ URGENT: fix production bug   │ Fix production   │ urgent   │ work     │ —          │
│ buy groceries sometime       │ Buy groceries    │ low      │ errands  │ —          │
│ schedule dentist for Tuesday │ Schedule dentist │ medium   │ health   │ 01/21 09:00│
│ pay electricity bill         │ Pay electricity  │ high     │ finance  │ 01/15 00:00│
│ maybe clean the garage       │ Clean garage     │ low      │ home     │ —          │
│ important: finish report     │ Finish quarterly │ high     │ work     │ 01/19 17:00│
│ pick up dry cleaning         │ Pick up dry      │ medium   │ errands  │ —          │
│ read ML book                 │ Read machine     │ medium   │ learning │ —          │
│ dinner with Sarah Saturday   │ Dinner with      │ medium   │ social   │ 01/20 19:00│
└──────────────────────────────┴──────────────────┴──────────┴──────────┴────────────┘
```

## Handling Edge Cases

```python
# src/ai_todo/ai/parsers.py (additions)

def sanitize_title(title: str) -> str:
    """Clean up extracted title."""
    # Remove common prefixes
    prefixes = [
        "remind me to ",
        "don't forget to ",
        "i need to ",
        "i have to ",
        "i should ",
        "remember to ",
        "make sure to ",
        "please ",
    ]
    
    lower = title.lower()
    for prefix in prefixes:
        if lower.startswith(prefix):
            title = title[len(prefix):]
            break
    
    # Capitalize first letter
    if title and title[0].islower():
        title = title[0].upper() + title[1:]
    
    # Truncate if too long
    if len(title) > 100:
        title = title[:97] + "..."
    
    return title.strip()
```

## Summary

In this chapter we:

1. ✅ Designed prompts for consistent task extraction
2. ✅ Built robust JSON parsing with fallbacks
3. ✅ Created the AIService for centralized LLM operations
4. ✅ Handled edge cases in parsing
5. ✅ Demonstrated the parsing pipeline

Key principles applied:
- **Low temperature (0.1)** for structured extraction
- **Explicit rules** in prompts for consistent behavior
- **Safe parsing** with fallbacks for reliability
- **Validation** separates AI output from system state

In the next chapter, we'll build the complete task creation workflow, integrating all the pieces.

---

**Previous**: [Chapter 6: The Task Data Model](./chapter-06-data-model.md)  
**Next**: [Chapter 8: Task Creation Workflow](./chapter-08-task-creation.md)
