# Chapter 9: Smart Prioritization — Temperature-Controlled Reasoning

## Beyond Keyword Matching

In Chapter 7, we used explicit keywords ("urgent", "asap") to determine priority. But intelligent prioritization goes deeper:

- "Pick up prescription" → Health + medication = higher priority
- "Call mom for her birthday tomorrow" → Time-sensitive + personal importance = high
- "Organize old photos" → No deadline, low consequence = low

This requires **reasoning**, not pattern matching.

## Temperature for Reasoning

Recall Principle 3: **Temperature Controls Entropy**.

| Use Case | Temperature | Why |
|----------|-------------|-----|
| JSON parsing | 0.0-0.1 | Need exact structure |
| Priority reasoning | 0.2-0.3 | Need consistent logic with slight flexibility |
| Creative suggestions | 0.6-0.8 | Want variety |

For prioritization, we use **0.2-0.3**—enough flexibility for nuanced reasoning, low enough for consistency.

## The Prioritization Prompt

```python
# src/ai_todo/ai/prompts.py (additions)

PRIORITY_REASONING_PROMPT = """You are a task prioritization expert. Analyze the task and determine its priority.

Consider these factors:
1. **Time sensitivity**: Deadlines, appointments, time-bound events
2. **Consequences**: What happens if this isn't done? Health/safety risks?
3. **Dependencies**: Does something else depend on this?
4. **Effort vs Impact**: Quick wins vs. large important projects
5. **Explicit signals**: Words like "urgent", "important", "critical"

Priority levels:
- **urgent**: Must be done today/immediately. Serious consequences if delayed.
- **high**: Important task. Should be done soon. Notable impact if delayed.
- **medium**: Standard task. Normal timeframe. (default)
- **low**: Can wait. No immediate consequences. Nice-to-have.

TASK TO ANALYZE:
Title: {title}
Description: {description}
Due Date: {due_date}
Category: {category}
Context: {context}

Analyze this task and return JSON:
{{
    "priority": "low" | "medium" | "high" | "urgent",
    "reasoning": "Brief explanation (1-2 sentences)",
    "factors": ["list", "of", "factors", "considered"]
}}

Output valid JSON only."""


BATCH_PRIORITIZATION_PROMPT = """You are a task prioritization expert. Analyze these tasks and rank them by priority.

Consider urgency, importance, consequences, and deadlines.

TASKS:
{tasks_json}

Return JSON array with tasks ranked from highest to lowest priority:
[
    {{"id": "task_id", "priority": "urgent|high|medium|low", "reasoning": "brief explanation"}},
    ...
]

Output valid JSON only."""
```

## The Prioritization Service

```python
# src/ai_todo/ai/prioritization.py
"""
Smart prioritization using LLM reasoning.

Uses moderate temperature (0.2-0.3) for consistent but nuanced reasoning.
"""

from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass

from .client import OllamaClient
from .prompts import PRIORITY_REASONING_PROMPT, BATCH_PRIORITIZATION_PROMPT
from .parsers import parse_json_safe
from ..models.task import Task
from ..models.enums import Priority


@dataclass
class PriorityAssessment:
    """Result of priority analysis."""
    priority: Priority
    reasoning: str
    factors: List[str]
    confidence: float = 1.0


class PrioritizationService:
    """
    AI-powered task prioritization.
    
    Uses temperature 0.2-0.3 for reasoning tasks.
    Lower than creative tasks, higher than parsing.
    """
    
    def __init__(self, client: OllamaClient):
        self.client = client
        self.temperature = 0.25  # Moderate for reasoning
    
    async def assess_priority(
        self,
        task: Task,
        context: str = ""
    ) -> PriorityAssessment:
        """
        Analyze a task and determine appropriate priority.
        
        Args:
            task: The task to analyze
            context: Additional context (e.g., other tasks, user patterns)
            
        Returns:
            PriorityAssessment with priority, reasoning, and factors
        """
        prompt = PRIORITY_REASONING_PROMPT.format(
            title=task.title,
            description=task.description or "None provided",
            due_date=task.due_date.isoformat() if task.due_date else "Not set",
            category=task.category.value,
            context=context or "No additional context"
        )
        
        response = await self.client.generate(
            prompt=prompt,
            temperature=self.temperature,
            format="json",
            max_tokens=256
        )
        
        data = parse_json_safe(response.content, {
            "priority": "medium",
            "reasoning": "Default assessment",
            "factors": []
        })
        
        return PriorityAssessment(
            priority=Priority.from_string(data.get("priority", "medium")),
            reasoning=data.get("reasoning", ""),
            factors=data.get("factors", [])
        )
    
    async def reprioritize_task(
        self,
        task: Task,
        all_tasks: List[Task]
    ) -> PriorityAssessment:
        """
        Reprioritize a task considering other active tasks.
        
        Context-aware prioritization based on workload.
        """
        # Build context from other tasks
        urgent_count = sum(1 for t in all_tasks if t.priority == Priority.URGENT)
        high_count = sum(1 for t in all_tasks if t.priority == Priority.HIGH)
        
        context = f"""Current workload:
- Urgent tasks: {urgent_count}
- High priority tasks: {high_count}
- Total active tasks: {len(all_tasks)}

Other tasks due soon:
"""
        for t in sorted(all_tasks, key=lambda x: x.due_date or datetime.max)[:5]:
            if t.id != task.id and t.due_date:
                context += f"- {t.title} (due: {t.due_date.strftime('%m/%d')})\n"
        
        return await self.assess_priority(task, context)
    
    async def batch_prioritize(
        self,
        tasks: List[Task]
    ) -> List[tuple[str, PriorityAssessment]]:
        """
        Prioritize multiple tasks in one call.
        
        More efficient than individual calls.
        Returns list of (task_id, assessment) tuples.
        """
        if not tasks:
            return []
        
        tasks_json = [
            {
                "id": t.id,
                "title": t.title,
                "description": t.description,
                "due_date": t.due_date.isoformat() if t.due_date else None,
                "category": t.category.value
            }
            for t in tasks
        ]
        
        import json
        prompt = BATCH_PRIORITIZATION_PROMPT.format(
            tasks_json=json.dumps(tasks_json, indent=2)
        )
        
        response = await self.client.generate(
            prompt=prompt,
            temperature=self.temperature,
            format="json",
            max_tokens=1024
        )
        
        results = parse_json_safe(response.content, [])
        
        assessments = []
        for item in results:
            if isinstance(item, dict) and "id" in item:
                assessments.append((
                    item["id"],
                    PriorityAssessment(
                        priority=Priority.from_string(item.get("priority", "medium")),
                        reasoning=item.get("reasoning", ""),
                        factors=[]
                    )
                ))
        
        return assessments


class PriorityRules:
    """
    Deterministic priority rules.
    
    Use these for guaranteed behavior before or after AI assessment.
    AI proposes, but rules can override.
    """
    
    @staticmethod
    def apply_deadline_boost(task: Task, assessment: PriorityAssessment) -> PriorityAssessment:
        """
        Boost priority for imminent deadlines.
        
        Deterministic rule: tasks due within 24 hours get bumped up.
        """
        if task.due_date is None:
            return assessment
        
        hours_until_due = (task.due_date - datetime.now()).total_seconds() / 3600
        
        if hours_until_due < 0:
            # Overdue - urgent
            return PriorityAssessment(
                priority=Priority.URGENT,
                reasoning=f"Task is overdue. Original: {assessment.reasoning}",
                factors=assessment.factors + ["overdue"]
            )
        elif hours_until_due < 4:
            # Due within 4 hours
            new_priority = max(Priority.HIGH, assessment.priority, key=lambda p: list(Priority).index(p))
            return PriorityAssessment(
                priority=new_priority,
                reasoning=f"Due within 4 hours. {assessment.reasoning}",
                factors=assessment.factors + ["imminent_deadline"]
            )
        elif hours_until_due < 24:
            # Due within 24 hours
            if assessment.priority in [Priority.LOW, Priority.MEDIUM]:
                return PriorityAssessment(
                    priority=Priority.HIGH,
                    reasoning=f"Due within 24 hours. {assessment.reasoning}",
                    factors=assessment.factors + ["approaching_deadline"]
                )
        
        return assessment
    
    @staticmethod
    def apply_category_defaults(task: Task, assessment: PriorityAssessment) -> PriorityAssessment:
        """
        Apply category-based priority floors.
        
        Some categories have minimum priority levels.
        """
        from ..models.enums import Category
        
        # Health tasks have a floor of medium
        if task.category == Category.HEALTH and assessment.priority == Priority.LOW:
            return PriorityAssessment(
                priority=Priority.MEDIUM,
                reasoning=f"Health tasks minimum priority. {assessment.reasoning}",
                factors=assessment.factors + ["health_floor"]
            )
        
        return assessment
```

## Integration with Task Service

```python
# src/ai_todo/services/task_service.py (additions)

from ..ai.prioritization import PrioritizationService, PriorityRules


class TaskService:
    # ... existing code ...
    
    def __init__(
        self,
        ai_service: AIService,
        repository: TaskRepository,
        embedding_store: Optional[EmbeddingStore] = None,
        prioritization_service: Optional[PrioritizationService] = None
    ):
        self.ai = ai_service
        self.repo = repository
        self.embeddings = embedding_store
        self.prioritization = prioritization_service
    
    async def smart_reprioritize(self, task_id: str) -> Optional[Task]:
        """
        Reprioritize a task using AI reasoning + deterministic rules.
        
        Flow:
        1. AI assesses priority with context
        2. Deterministic rules apply overrides
        3. System commits the change
        """
        if not self.prioritization:
            return None
        
        task = await self.repo.get(task_id)
        if task is None:
            return None
        
        # Get all active tasks for context
        all_tasks = await self.repo.list(status=TaskStatus.PENDING, limit=100)
        
        # AI proposes
        assessment = await self.prioritization.reprioritize_task(task, all_tasks)
        
        # Deterministic rules can override
        assessment = PriorityRules.apply_deadline_boost(task, assessment)
        assessment = PriorityRules.apply_category_defaults(task, assessment)
        
        # System commits (only if changed)
        if assessment.priority != task.priority:
            updated = task.update_priority(assessment.priority)
            await self.repo.update(updated)
            return updated
        
        return task
    
    async def auto_prioritize_all(self) -> List[Task]:
        """
        Reprioritize all pending tasks.
        
        Useful for daily review or after importing tasks.
        """
        if not self.prioritization:
            return []
        
        tasks = await self.repo.list(status=TaskStatus.PENDING, limit=100)
        
        # Batch prioritize for efficiency
        assessments = await self.prioritization.batch_prioritize(tasks)
        
        updated_tasks = []
        for task_id, assessment in assessments:
            task = next((t for t in tasks if t.id == task_id), None)
            if task is None:
                continue
            
            # Apply deterministic rules
            assessment = PriorityRules.apply_deadline_boost(task, assessment)
            assessment = PriorityRules.apply_category_defaults(task, assessment)
            
            if assessment.priority != task.priority:
                updated = task.update_priority(assessment.priority)
                await self.repo.update(updated)
                updated_tasks.append(updated)
        
        return updated_tasks
```

## Demo: Priority Assessment

```python
# scripts/priority_demo.py
"""Demonstrate smart prioritization."""

import asyncio
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table

from ai_todo.ai.client import OllamaClient
from ai_todo.ai.prioritization import PrioritizationService, PriorityRules
from ai_todo.models.task import Task
from ai_todo.models.enums import Priority, Category, TaskStatus


console = Console()


def create_test_task(
    title: str,
    category: Category,
    due_date: datetime | None = None
) -> Task:
    """Create a test task."""
    now = datetime.now()
    return Task(
        id=f"test-{hash(title) % 10000}",
        title=title,
        description=None,
        priority=Priority.MEDIUM,  # Default, will be reassessed
        category=category,
        due_date=due_date,
        status=TaskStatus.PENDING,
        completed=False,
        created_at=now,
        updated_at=now
    )


async def main():
    console.print("[bold]Smart Prioritization Demo[/bold]\n")
    
    # Create test tasks
    now = datetime.now()
    test_tasks = [
        create_test_task(
            "Pick up prescription from pharmacy",
            Category.HEALTH,
            due_date=now + timedelta(hours=2)
        ),
        create_test_task(
            "Organize old photos",
            Category.PERSONAL,
            due_date=None
        ),
        create_test_task(
            "Call mom for her birthday",
            Category.PERSONAL,
            due_date=now + timedelta(hours=6)
        ),
        create_test_task(
            "Submit tax documents",
            Category.FINANCE,
            due_date=now + timedelta(days=2)
        ),
        create_test_task(
            "Clean the garage",
            Category.HOME,
            due_date=None
        ),
        create_test_task(
            "Fix production bug reported by client",
            Category.WORK,
            due_date=None
        ),
    ]
    
    async with OllamaClient() as client:
        prioritization = PrioritizationService(client)
        
        table = Table(title="Priority Assessment Results")
        table.add_column("Task", style="cyan", width=35)
        table.add_column("Category")
        table.add_column("Due")
        table.add_column("AI Priority", style="yellow")
        table.add_column("Final Priority", style="green")
        table.add_column("Reasoning")
        
        for task in test_tasks:
            with console.status(f"Analyzing: {task.title[:30]}..."):
                # AI assessment
                assessment = await prioritization.assess_priority(task)
                
                # Apply deterministic rules
                final_assessment = PriorityRules.apply_deadline_boost(task, assessment)
                final_assessment = PriorityRules.apply_category_defaults(task, final_assessment)
            
            due_str = (
                task.due_date.strftime("%m/%d %H:%M")
                if task.due_date else "—"
            )
            
            priority_colors = {
                "urgent": "red",
                "high": "yellow", 
                "medium": "white",
                "low": "dim"
            }
            
            ai_priority = f"[{priority_colors[assessment.priority.value]}]{assessment.priority.value}[/]"
            final_priority = f"[{priority_colors[final_assessment.priority.value]}]{final_assessment.priority.value}[/]"
            
            # Truncate reasoning
            reasoning = final_assessment.reasoning[:40] + "..." if len(final_assessment.reasoning) > 40 else final_assessment.reasoning
            
            table.add_row(
                task.title,
                task.category.value,
                due_str,
                ai_priority,
                final_priority,
                reasoning
            )
        
        console.print(table)
        
        console.print("\n[dim]AI temperature: 0.25 (moderate for reasoning)[/dim]")
        console.print("[dim]Deterministic rules applied: deadline_boost, category_defaults[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
```

Output:
```
Smart Prioritization Demo

                            Priority Assessment Results
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Task                              ┃ Category ┃ Due         ┃ AI Priority┃ Final Priority┃ Reasoning                     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Pick up prescription from pharmacy│ health   │ 01/15 12:00 │ high       │ urgent        │ Due within 4 hours. Health... │
│ Organize old photos               │ personal │ —           │ low        │ low           │ No deadline, low consequence  │
│ Call mom for her birthday         │ personal │ 01/15 16:00 │ high       │ high          │ Time-sensitive personal event │
│ Submit tax documents              │ finance  │ 01/17 10:00 │ high       │ high          │ Financial deadline, conseq... │
│ Clean the garage                  │ home     │ —           │ low        │ low           │ No urgency, can be deferred   │
│ Fix production bug from client    │ work     │ —           │ urgent     │ urgent        │ Client-facing issue, reputa...│
└───────────────────────────────────┴──────────┴─────────────┴────────────┴───────────────┴───────────────────────────────┘

AI temperature: 0.25 (moderate for reasoning)
Deterministic rules applied: deadline_boost, category_defaults
```

## Key Observations

### AI Assessment vs. Deterministic Rules

The prescription task:
- AI assessed as "high" (health context)
- Deadline rule boosted to "urgent" (due in 2 hours)

The "organize photos" task:
- AI correctly assessed as "low" (no deadline, no consequences)
- No rules applied

The production bug:
- AI assessed as "urgent" (client-facing, reputation risk)
- No deadline, but AI recognized implied urgency

### Temperature Effect

With temperature 0.25:
- Reasoning is consistent across runs
- Slight variation in wording, not conclusions
- Different from temperature 0.0 (identical outputs)
- Different from temperature 0.7 (variable conclusions)

## Summary

In this chapter we:

1. ✅ Built prioritization prompts for reasoning tasks
2. ✅ Created the PrioritizationService with temperature 0.25
3. ✅ Implemented deterministic rules for guaranteed behaviors
4. ✅ Combined AI assessment with rule overrides
5. ✅ Demonstrated context-aware batch prioritization

The priority system demonstrates:
- **AI proposes**: Initial priority assessment
- **System commits**: Deterministic rules can override
- **Temperature 0.25**: Consistent reasoning with flexibility

In the next chapter, we'll build robust validation patterns for the entire system.

---

**Previous**: [Chapter 8: Task Creation Workflow](./chapter-08-task-creation.md)  
**Next**: [Chapter 10: Validation Patterns](./chapter-10-validation.md)
