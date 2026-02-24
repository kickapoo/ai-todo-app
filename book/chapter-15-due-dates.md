# Chapter 15: Due Date Intelligence — Pattern Recognition

## Understanding Time in Tasks

Due dates come in many forms:
- **Explicit**: "by Friday 5pm", "January 15th"
- **Relative**: "tomorrow", "next week", "in 3 days"
- **Implicit**: "ASAP", "when possible", "eventually"
- **Contextual**: "before the meeting", "after lunch"

This chapter builds intelligent date handling using patterns and AI.

## The Date Service

```python
# src/ai_todo/services/date_service.py
"""
Intelligent date handling for task due dates.

Combines rule-based parsing with AI inference.
"""

import re
from datetime import datetime, timedelta, time
from typing import Optional, Tuple, List
from dataclasses import dataclass
from dateutil import parser as dateutil_parser
from dateutil.relativedelta import relativedelta

from ..ai.client import OllamaClient


@dataclass
class DateParseResult:
    """Result of date parsing."""
    date: Optional[datetime]
    confidence: float
    method: str  # "explicit", "relative", "ai", "none"
    original_text: str
    interpretation: str


class DateService:
    """
    Date parsing and suggestion service.
    
    Uses a cascade approach:
    1. Explicit date patterns (highest confidence)
    2. Relative date expressions
    3. AI inference for complex cases
    """
    
    # Time of day defaults
    TIME_DEFAULTS = {
        "morning": time(9, 0),
        "noon": time(12, 0),
        "afternoon": time(14, 0),
        "evening": time(18, 0),
        "night": time(20, 0),
        "eod": time(17, 0),
        "end of day": time(17, 0),
        "cob": time(17, 0),  # Close of business
    }
    
    # Relative date patterns
    RELATIVE_PATTERNS = {
        r"\btoday\b": lambda now: now.replace(hour=17, minute=0),
        r"\btomorrow\b": lambda now: (now + timedelta(days=1)).replace(hour=17, minute=0),
        r"\byesterday\b": lambda now: (now - timedelta(days=1)).replace(hour=17, minute=0),
        r"\bnext week\b": lambda now: now + timedelta(weeks=1),
        r"\bnext month\b": lambda now: now + relativedelta(months=1),
        r"\bin (\d+) days?\b": lambda now, d: now + timedelta(days=int(d)),
        r"\bin (\d+) hours?\b": lambda now, h: now + timedelta(hours=int(h)),
        r"\bin (\d+) weeks?\b": lambda now, w: now + timedelta(weeks=int(w)),
    }
    
    # Day of week patterns
    WEEKDAYS = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
        "mon": 0, "tue": 1, "wed": 2, "thu": 3,
        "fri": 4, "sat": 5, "sun": 6,
    }
    
    def __init__(self, ai_client: Optional[OllamaClient] = None):
        self.ai = ai_client
    
    def parse(
        self,
        text: str,
        reference_time: Optional[datetime] = None
    ) -> DateParseResult:
        """
        Parse date/time from text.
        
        Args:
            text: Text potentially containing date information
            reference_time: Base time for relative dates
            
        Returns:
            DateParseResult with parsed date and metadata
        """
        if reference_time is None:
            reference_time = datetime.now()
        
        text_lower = text.lower()
        
        # Try explicit parsing first
        result = self._try_explicit(text_lower, reference_time)
        if result.date:
            return result
        
        # Try relative patterns
        result = self._try_relative(text_lower, reference_time)
        if result.date:
            return result
        
        # Try weekday patterns
        result = self._try_weekday(text_lower, reference_time)
        if result.date:
            return result
        
        # No date found
        return DateParseResult(
            date=None,
            confidence=0.0,
            method="none",
            original_text=text,
            interpretation="No date information found"
        )
    
    def _try_explicit(
        self,
        text: str,
        reference: datetime
    ) -> DateParseResult:
        """Try explicit date parsing with dateutil."""
        try:
            # Look for date-like patterns
            date_patterns = [
                r"\d{4}-\d{2}-\d{2}",  # ISO format
                r"\d{1,2}/\d{1,2}/\d{2,4}",  # US format
                r"\d{1,2}-\d{1,2}-\d{2,4}",  # EU format
                r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2}",
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    parsed = dateutil_parser.parse(match.group(), default=reference)
                    
                    # Check for time-of-day indicators
                    parsed = self._apply_time_of_day(text, parsed)
                    
                    return DateParseResult(
                        date=parsed,
                        confidence=0.95,
                        method="explicit",
                        original_text=text,
                        interpretation=f"Parsed as {parsed.strftime('%Y-%m-%d %H:%M')}"
                    )
        except Exception:
            pass
        
        return DateParseResult(
            date=None, confidence=0.0, method="explicit",
            original_text=text, interpretation=""
        )
    
    def _try_relative(
        self,
        text: str,
        reference: datetime
    ) -> DateParseResult:
        """Try relative date patterns."""
        for pattern, func in self.RELATIVE_PATTERNS.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    groups = match.groups()
                    if groups:
                        date = func(reference, *groups)
                    else:
                        date = func(reference)
                    
                    # Apply time of day
                    date = self._apply_time_of_day(text, date)
                    
                    return DateParseResult(
                        date=date,
                        confidence=0.9,
                        method="relative",
                        original_text=text,
                        interpretation=f"'{match.group()}' → {date.strftime('%Y-%m-%d %H:%M')}"
                    )
                except Exception:
                    pass
        
        return DateParseResult(
            date=None, confidence=0.0, method="relative",
            original_text=text, interpretation=""
        )
    
    def _try_weekday(
        self,
        text: str,
        reference: datetime
    ) -> DateParseResult:
        """Try weekday patterns like 'next Monday'."""
        for day_name, day_num in self.WEEKDAYS.items():
            if day_name in text:
                # Find next occurrence of this weekday
                current_day = reference.weekday()
                days_ahead = day_num - current_day
                
                # If "next" specified or day has passed, go to next week
                if "next" in text or days_ahead <= 0:
                    days_ahead += 7
                
                target = reference + timedelta(days=days_ahead)
                target = target.replace(hour=17, minute=0, second=0)
                
                # Apply time of day
                target = self._apply_time_of_day(text, target)
                
                return DateParseResult(
                    date=target,
                    confidence=0.85,
                    method="relative",
                    original_text=text,
                    interpretation=f"'{day_name}' → {target.strftime('%A, %Y-%m-%d %H:%M')}"
                )
        
        return DateParseResult(
            date=None, confidence=0.0, method="relative",
            original_text=text, interpretation=""
        )
    
    def _apply_time_of_day(
        self,
        text: str,
        date: datetime
    ) -> datetime:
        """Apply time-of-day modifiers."""
        text_lower = text.lower()
        
        for time_word, default_time in self.TIME_DEFAULTS.items():
            if time_word in text_lower:
                return date.replace(
                    hour=default_time.hour,
                    minute=default_time.minute
                )
        
        # Look for explicit time like "3pm" or "15:00"
        time_patterns = [
            (r"(\d{1,2}):(\d{2})", lambda h, m: (int(h), int(m))),
            (r"(\d{1,2})\s*(?:am)", lambda h: (int(h), 0)),
            (r"(\d{1,2})\s*(?:pm)", lambda h: (int(h) + 12 if int(h) < 12 else int(h), 0)),
        ]
        
        for pattern, parser in time_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    result = parser(*match.groups())
                    if isinstance(result, tuple):
                        hour, minute = result
                    else:
                        hour, minute = result, 0
                    return date.replace(hour=hour, minute=minute)
                except Exception:
                    pass
        
        return date
    
    async def suggest_due_date(
        self,
        task_title: str,
        category: Optional[str] = None
    ) -> DateParseResult:
        """
        AI-powered due date suggestion when no date is specified.
        
        Uses task characteristics to suggest appropriate deadlines.
        """
        if not self.ai:
            return DateParseResult(
                date=None, confidence=0.0, method="none",
                original_text=task_title,
                interpretation="AI not available for date suggestion"
            )
        
        now = datetime.now()
        
        prompt = f"""Suggest an appropriate due date for this task.

Task: "{task_title}"
Category: {category or "unknown"}
Current date: {now.strftime('%Y-%m-%d %A')}

Consider:
- Urgency implied by the task
- Typical timeframes for similar tasks
- Category norms (work tasks often have shorter deadlines)

Return JSON:
{{
    "suggested_date": "YYYY-MM-DD",
    "suggested_time": "HH:MM",
    "reasoning": "brief explanation",
    "urgency": "immediate|short|medium|long|flexible"
}}

If no due date makes sense, return {{"suggested_date": null}}"""

        try:
            response = await self.ai.generate(
                prompt=prompt,
                temperature=0.3,
                format="json",
                max_tokens=128
            )
            
            from ..ai.parsers import parse_json_safe
            result = parse_json_safe(response.content, {})
            
            if not result.get("suggested_date"):
                return DateParseResult(
                    date=None, confidence=0.3, method="ai",
                    original_text=task_title,
                    interpretation="AI suggests no specific deadline"
                )
            
            date_str = result["suggested_date"]
            time_str = result.get("suggested_time", "17:00")
            
            suggested = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
            
            # Confidence based on urgency
            confidence_map = {
                "immediate": 0.8,
                "short": 0.7,
                "medium": 0.6,
                "long": 0.5,
                "flexible": 0.4
            }
            confidence = confidence_map.get(result.get("urgency", "medium"), 0.5)
            
            return DateParseResult(
                date=suggested,
                confidence=confidence,
                method="ai",
                original_text=task_title,
                interpretation=result.get("reasoning", "AI suggestion")
            )
            
        except Exception as e:
            return DateParseResult(
                date=None, confidence=0.0, method="ai",
                original_text=task_title,
                interpretation=f"AI suggestion failed: {e}"
            )


class DateSuggestionService:
    """
    Learn due date patterns from historical tasks.
    """
    
    def __init__(self, embedding_store):
        self.embeddings = embedding_store
    
    async def get_typical_duration(
        self,
        task_title: str,
        category: str
    ) -> Optional[timedelta]:
        """
        Estimate typical duration based on similar past tasks.
        
        Looks at how long similar tasks took to complete.
        """
        similar = await self.embeddings.find_similar(
            task_title,
            limit=10,
            min_score=0.4
        )
        
        durations = []
        for item in similar:
            meta = item["metadata"]
            if meta.get("due_date") and meta.get("created_at"):
                try:
                    due = datetime.fromisoformat(meta["due_date"])
                    created = datetime.fromisoformat(meta["created_at"])
                    durations.append(due - created)
                except Exception:
                    pass
        
        if not durations:
            return None
        
        # Return median duration
        durations.sort()
        return durations[len(durations) // 2]
```

## Demo: Date Intelligence

```python
# scripts/date_demo.py
"""Demonstrate date parsing and suggestion."""

import asyncio
from datetime import datetime
from rich.console import Console
from rich.table import Table

from ai_todo.services.date_service import DateService
from ai_todo.ai.client import OllamaClient


console = Console()


async def main():
    console.print("[bold]Date Intelligence Demo[/bold]\n")
    
    # Test parsing
    console.print("[cyan]Date Parsing:[/cyan]\n")
    
    date_service = DateService()
    
    test_cases = [
        "finish report by Friday",
        "call mom tomorrow afternoon",
        "meeting next Monday at 3pm",
        "submit taxes by April 15",
        "in 3 days",
        "EOD today",
        "sometime next week",
    ]
    
    table = Table()
    table.add_column("Input", style="cyan")
    table.add_column("Parsed Date")
    table.add_column("Confidence")
    table.add_column("Method")
    
    for text in test_cases:
        result = date_service.parse(text)
        
        date_str = result.date.strftime("%a %m/%d %H:%M") if result.date else "—"
        conf_str = f"{result.confidence*100:.0f}%"
        
        table.add_row(text, date_str, conf_str, result.method)
    
    console.print(table)
    
    # Test AI suggestions
    console.print("\n[cyan]AI Date Suggestions:[/cyan]\n")
    
    async with OllamaClient() as client:
        ai_date_service = DateService(ai_client=client)
        
        suggestion_cases = [
            ("schedule dentist appointment", "health"),
            ("buy birthday present", "personal"),
            ("fix critical bug in production", "work"),
            ("learn to play guitar", "learning"),
        ]
        
        table2 = Table()
        table2.add_column("Task", style="cyan")
        table2.add_column("Category")
        table2.add_column("Suggested Date")
        table2.add_column("Reasoning")
        
        for task, category in suggestion_cases:
            result = await ai_date_service.suggest_due_date(task, category)
            
            date_str = result.date.strftime("%m/%d") if result.date else "No deadline"
            
            table2.add_row(task, category, date_str, result.interpretation[:40])
        
        console.print(table2)


if __name__ == "__main__":
    asyncio.run(main())
```

Output:
```
Date Intelligence Demo

Date Parsing:

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Input                         ┃ Parsed Date     ┃ Confidence ┃ Method   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ finish report by Friday       │ Fri 01/19 17:00 │ 85%        │ relative │
│ call mom tomorrow afternoon   │ Tue 01/16 14:00 │ 90%        │ relative │
│ meeting next Monday at 3pm    │ Mon 01/22 15:00 │ 85%        │ relative │
│ submit taxes by April 15      │ Mon 04/15 17:00 │ 95%        │ explicit │
│ in 3 days                     │ Thu 01/18 17:00 │ 90%        │ relative │
│ EOD today                     │ Mon 01/15 17:00 │ 90%        │ relative │
│ sometime next week            │ Mon 01/22 17:00 │ 90%        │ relative │
└───────────────────────────────┴─────────────────┴────────────┴──────────┘

AI Date Suggestions:

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Task                           ┃ Category ┃ Suggested Date┃ Reasoning                        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ schedule dentist appointment   │ health   │ 01/22         │ Health appointments typically... │
│ buy birthday present           │ personal │ 01/18         │ Gifts should be purchased 2-3... │
│ fix critical bug in production │ work     │ 01/15         │ Critical bugs require immediate..│
│ learn to play guitar           │ learning │ No deadline   │ Learning goals are ongoing, no.. │
└────────────────────────────────┴──────────┴───────────────┴──────────────────────────────────┘
```

## Summary

In this chapter we built:

1. ✅ Rule-based date parsing (explicit, relative, weekday)
2. ✅ Time-of-day modifiers
3. ✅ AI-powered date suggestions
4. ✅ Pattern learning from historical tasks
5. ✅ Confidence scoring for parsed dates

This completes Part 3: Memory and Retrieval.

The memory architecture is now complete:
- **SQLite**: Facts (what happened)
- **ChromaDB**: Semantics (what it means)
- **RAG**: Context (informed generation)
- **Learning**: Patterns from history

In Part 4, we'll build production-ready features: CLI, testing, and deployment.

---

**Previous**: [Chapter 14: Auto-Categorization](./chapter-14-categorization.md)  
**Next**: [Chapter 16: CLI Interface](./chapter-16-cli.md)
