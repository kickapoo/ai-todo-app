# Chapter 13: Retrieval-Augmented Generation — Context-Aware AI

## Beyond Static Prompts

In earlier chapters, our AI prompts were static templates. RAG makes them dynamic by injecting relevant context:

```
Static Prompt:
"Categorize this task: buy groceries"

RAG-Enhanced Prompt:
"CONTEXT:
Similar past tasks and their categories:
- grocery shopping → errands
- pick up vegetables → errands
- order food delivery → errands

Categorize this task: buy groceries"
```

The AI now sees patterns from history, making it more consistent and accurate.

## The RAG Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                      RAG PIPELINE                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. USER INPUT                                                  │
│   ┌────────────────────────────────────────┐                    │
│   │ "schedule dentist appointment"          │                    │
│   └───────────────────┬────────────────────┘                    │
│                       │                                          │
│   2. RETRIEVAL        ▼                                          │
│   ┌────────────────────────────────────────┐                    │
│   │ Query ChromaDB for similar tasks        │                    │
│   │                                         │                    │
│   │ Results:                                │                    │
│   │ - "doctor appointment" (health, 82%)    │                    │
│   │ - "eye exam" (health, 75%)              │                    │
│   │ - "physical therapy" (health, 70%)      │                    │
│   └───────────────────┬────────────────────┘                    │
│                       │                                          │
│   3. AUGMENTATION     ▼                                          │
│   ┌────────────────────────────────────────┐                    │
│   │ Build context-enhanced prompt:          │                    │
│   │                                         │                    │
│   │ "Similar tasks and categories:          │                    │
│   │  - doctor appointment: health           │                    │
│   │  - eye exam: health                     │                    │
│   │  - physical therapy: health             │                    │
│   │                                         │                    │
│   │ Based on patterns, categorize:          │                    │
│   │ schedule dentist appointment"           │                    │
│   └───────────────────┬────────────────────┘                    │
│                       │                                          │
│   4. GENERATION       ▼                                          │
│   ┌────────────────────────────────────────┐                    │
│   │ LLM generates response:                 │                    │
│   │ {"category": "health", "confidence": 95}│                    │
│   └────────────────────────────────────────┘                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Complete RAG Implementation

```python
# src/ai_todo/ai/rag.py
"""
Complete RAG implementation for context-aware task processing.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .client import OllamaClient
from ..memory.embeddings import EmbeddingStore
from ..models.task import Task, TaskProposal
from ..models.enums import Category, Priority


@dataclass
class RetrievalResult:
    """Result from retrieval step."""
    items: List[dict]
    query: str
    retrieval_time_ms: float


@dataclass
class RAGResult:
    """Complete RAG result with provenance."""
    response: dict
    context_used: str
    sources: List[dict]
    retrieval_time_ms: float
    generation_time_ms: float


class RAGPipeline:
    """
    Full RAG pipeline for task processing.
    
    Combines retrieval from ChromaDB with LLM generation.
    """
    
    def __init__(
        self,
        ai_client: OllamaClient,
        embedding_store: EmbeddingStore
    ):
        self.ai = ai_client
        self.embeddings = embedding_store
    
    async def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        min_score: float = 0.3
    ) -> RetrievalResult:
        """
        Retrieve relevant context from memory.
        
        Args:
            query: Search query
            limit: Max items to retrieve
            min_score: Minimum similarity threshold
            
        Returns:
            RetrievalResult with items and metadata
        """
        start = datetime.now()
        
        items = await self.embeddings.find_similar(
            query=query,
            limit=limit,
            min_score=min_score
        )
        
        elapsed = (datetime.now() - start).total_seconds() * 1000
        
        return RetrievalResult(
            items=items,
            query=query,
            retrieval_time_ms=elapsed
        )
    
    def format_context(
        self,
        items: List[dict],
        *,
        format_type: str = "categorization"
    ) -> str:
        """
        Format retrieved items as context for prompt.
        
        Args:
            items: Retrieved items
            format_type: How to format (categorization, prioritization, general)
            
        Returns:
            Formatted context string
        """
        if not items:
            return ""
        
        if format_type == "categorization":
            return self._format_for_categorization(items)
        elif format_type == "prioritization":
            return self._format_for_prioritization(items)
        else:
            return self._format_general(items)
    
    def _format_for_categorization(self, items: List[dict]) -> str:
        """Format context for category inference."""
        lines = ["CONTEXT - How similar tasks were categorized:"]
        
        # Group by category
        by_category = {}
        for item in items:
            cat = item["metadata"]["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append({
                "title": item["metadata"]["title"],
                "score": item["score"]
            })
        
        for cat, tasks in by_category.items():
            task_list = ", ".join(t["title"] for t in tasks[:3])
            lines.append(f"• {cat}: {task_list}")
        
        return "\n".join(lines)
    
    def _format_for_prioritization(self, items: List[dict]) -> str:
        """Format context for priority inference."""
        lines = ["CONTEXT - How similar tasks were prioritized:"]
        
        for item in items[:5]:
            meta = item["metadata"]
            lines.append(
                f"• \"{meta['title']}\" → {meta['priority']} priority"
            )
        
        return "\n".join(lines)
    
    def _format_general(self, items: List[dict]) -> str:
        """General context formatting."""
        lines = ["CONTEXT - Related past tasks:"]
        
        for item in items[:5]:
            meta = item["metadata"]
            score_pct = int(item["score"] * 100)
            lines.append(
                f"• {meta['title']} ({meta['category']}, {meta['priority']}) [{score_pct}% similar]"
            )
        
        return "\n".join(lines)
    
    async def categorize_with_rag(
        self,
        task_title: str,
        description: Optional[str] = None
    ) -> RAGResult:
        """
        Categorize a task using RAG.
        
        Retrieves similar tasks and uses their categories as context.
        """
        # Step 1: Retrieve
        query = f"{task_title} {description or ''}"
        retrieval = await self.retrieve(query, limit=5, min_score=0.3)
        
        # Step 2: Format context
        context = self.format_context(
            retrieval.items,
            format_type="categorization"
        )
        
        # Step 3: Build prompt
        if context:
            prompt = f"""{context}

Based on the patterns above, categorize this new task:
Title: "{task_title}"
Description: {description or "None"}

Return JSON: {{"category": "category_name", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""
        else:
            prompt = f"""Categorize this task:
Title: "{task_title}"
Description: {description or "None"}

Categories: work, personal, health, finance, learning, errands, home, social, other

Return JSON: {{"category": "category_name", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""
        
        # Step 4: Generate
        gen_start = datetime.now()
        response = await self.ai.generate(
            prompt=prompt,
            temperature=0.2,  # Low for consistency
            format="json",
            max_tokens=128
        )
        gen_time = (datetime.now() - gen_start).total_seconds() * 1000
        
        # Parse response
        from .parsers import parse_json_safe
        result = parse_json_safe(response.content, {"category": "other", "confidence": 0.5})
        
        return RAGResult(
            response=result,
            context_used=context,
            sources=retrieval.items,
            retrieval_time_ms=retrieval.retrieval_time_ms,
            generation_time_ms=gen_time
        )
    
    async def prioritize_with_rag(
        self,
        task_title: str,
        due_date: Optional[datetime] = None,
        category: Optional[str] = None
    ) -> RAGResult:
        """
        Prioritize a task using RAG.
        
        Considers how similar tasks were prioritized.
        """
        # Retrieve
        retrieval = await self.retrieve(task_title, limit=5, min_score=0.3)
        
        # Format context
        context = self.format_context(
            retrieval.items,
            format_type="prioritization"
        )
        
        # Build prompt
        due_info = f"Due: {due_date.isoformat()}" if due_date else "No due date"
        cat_info = f"Category: {category}" if category else ""
        
        if context:
            prompt = f"""{context}

Based on the patterns above, determine priority for this new task:
Title: "{task_title}"
{due_info}
{cat_info}

Priority levels: low, medium, high, urgent

Return JSON: {{"priority": "level", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""
        else:
            prompt = f"""Determine priority for this task:
Title: "{task_title}"
{due_info}
{cat_info}

Priority levels:
- urgent: Must be done immediately, serious consequences if delayed
- high: Important, should be done soon
- medium: Normal priority (default)
- low: Can wait, no immediate consequences

Return JSON: {{"priority": "level", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""
        
        # Generate
        gen_start = datetime.now()
        response = await self.ai.generate(
            prompt=prompt,
            temperature=0.25,
            format="json",
            max_tokens=128
        )
        gen_time = (datetime.now() - gen_start).total_seconds() * 1000
        
        from .parsers import parse_json_safe
        result = parse_json_safe(response.content, {"priority": "medium", "confidence": 0.5})
        
        return RAGResult(
            response=result,
            context_used=context,
            sources=retrieval.items,
            retrieval_time_ms=retrieval.retrieval_time_ms,
            generation_time_ms=gen_time
        )
    
    async def extract_task_with_rag(
        self,
        raw_input: str,
        current_datetime: Optional[datetime] = None
    ) -> Tuple[TaskProposal, RAGResult]:
        """
        Full task extraction with RAG-enhanced categorization and priority.
        
        1. Basic extraction (title, due date)
        2. RAG-enhanced categorization
        3. RAG-enhanced prioritization
        """
        from .prompts import get_task_extraction_prompt, TASK_EXTRACTION_USER_PROMPT
        from .parsers import parse_task_proposal
        
        if current_datetime is None:
            current_datetime = datetime.now()
        
        # Step 1: Basic extraction
        system_prompt = get_task_extraction_prompt(current_datetime)
        user_prompt = TASK_EXTRACTION_USER_PROMPT.format(user_input=raw_input)
        
        response = await self.ai.generate(
            prompt=user_prompt,
            system=system_prompt,
            temperature=0.1,
            format="json"
        )
        
        proposal = parse_task_proposal(response.content)
        
        # Step 2: RAG-enhanced categorization
        cat_result = await self.categorize_with_rag(
            proposal.title,
            proposal.description
        )
        
        # Step 3: RAG-enhanced prioritization
        pri_result = await self.prioritize_with_rag(
            proposal.title,
            proposal.due_date,
            cat_result.response.get("category")
        )
        
        # Update proposal with RAG-enhanced values
        # Only override if RAG is confident
        if cat_result.response.get("confidence", 0) > 0.6:
            proposal = proposal.model_copy(update={
                "category": Category.from_string(cat_result.response["category"])
            })
        
        if pri_result.response.get("confidence", 0) > 0.6:
            proposal = proposal.model_copy(update={
                "priority": Priority.from_string(pri_result.response["priority"])
            })
        
        # Return both proposal and RAG info
        combined_result = RAGResult(
            response={
                "category": cat_result.response,
                "priority": pri_result.response
            },
            context_used=f"{cat_result.context_used}\n\n{pri_result.context_used}",
            sources=cat_result.sources + pri_result.sources,
            retrieval_time_ms=cat_result.retrieval_time_ms + pri_result.retrieval_time_ms,
            generation_time_ms=cat_result.generation_time_ms + pri_result.generation_time_ms
        )
        
        return proposal, combined_result
```

## Demo: RAG in Action

```python
# scripts/rag_demo.py
"""Demonstrate RAG for task categorization."""

import asyncio
from datetime import datetime
from rich.console import Console
from rich.panel import Panel

from ai_todo.main import get_app
from ai_todo.ai.rag import RAGPipeline
from ai_todo.models.task import TaskInput


console = Console()


async def main():
    app = get_app()
    
    # First, create some tasks to build context
    console.print(Panel.fit("[bold]Phase 1: Building Context[/bold]"))
    
    seed_tasks = [
        ("buy groceries", "errands"),
        ("pick up dry cleaning", "errands"),
        ("grocery shopping for party", "errands"),
        ("call mom", "personal"),
        ("call dad about weekend", "personal"),
        ("phone call with sister", "personal"),
        ("doctor checkup", "health"),
        ("dentist appointment", "health"),
        ("gym workout", "health"),
        ("quarterly report", "work"),
        ("team meeting prep", "work"),
        ("send email to client", "work"),
    ]
    
    for title, expected_cat in seed_tasks:
        result = await app.tasks.create_task(TaskInput(raw_input=title))
        console.print(f"  Created: {title} → {result.task.category.value}")
    
    console.print()
    
    # Now test RAG categorization
    console.print(Panel.fit("[bold]Phase 2: RAG-Enhanced Categorization[/bold]"))
    
    rag = RAGPipeline(app.ollama, app.embeddings)
    
    test_cases = [
        "order food online",           # Should match errands (shopping)
        "video call with friend",      # Should match personal (calls)
        "annual physical exam",        # Should match health
        "presentation slides",         # Should match work
    ]
    
    for query in test_cases:
        console.print(f"\n[cyan]Input: \"{query}\"[/cyan]")
        
        result = await rag.categorize_with_rag(query)
        
        console.print(f"  Category: [green]{result.response['category']}[/green]")
        console.print(f"  Confidence: {result.response.get('confidence', 0)*100:.0f}%")
        console.print(f"  Reasoning: {result.response.get('reasoning', '')}")
        console.print(f"  Retrieval: {result.retrieval_time_ms:.0f}ms, Generation: {result.generation_time_ms:.0f}ms")
        
        if result.sources:
            console.print("  Context sources:")
            for src in result.sources[:3]:
                console.print(f"    - {src['metadata']['title']} ({src['score']*100:.0f}%)")
    
    await app.close()


if __name__ == "__main__":
    asyncio.run(main())
```

Output:
```
╭──────────────────────────────────╮
│ Phase 1: Building Context        │
╰──────────────────────────────────╯
  Created: buy groceries → errands
  Created: pick up dry cleaning → errands
  Created: grocery shopping for party → errands
  Created: call mom → personal
  ...

╭──────────────────────────────────────╮
│ Phase 2: RAG-Enhanced Categorization │
╰──────────────────────────────────────╯

Input: "order food online"
  Category: errands
  Confidence: 87%
  Reasoning: Similar to grocery shopping and food-related errands
  Retrieval: 12ms, Generation: 245ms
  Context sources:
    - Buy groceries (74%)
    - Grocery shopping for party (71%)
    - Pick up dry cleaning (52%)

Input: "video call with friend"
  Category: personal
  Confidence: 91%
  Reasoning: Similar to other personal calls with family/friends
  Retrieval: 8ms, Generation: 198ms
  Context sources:
    - Call mom (78%)
    - Phone call with sister (76%)
    - Call dad about weekend (73%)
```

## When RAG Helps vs. Hurts

### RAG Helps When:
- You have relevant past data
- Patterns exist in history
- Consistency with past decisions matters
- User expects similar inputs to get similar outputs

### RAG Can Hurt When:
- No relevant context exists (adds noise)
- Past data has errors (propagates mistakes)
- Novel inputs need fresh thinking
- Context is misleading

### Mitigation Strategies:

```python
# Only use context if highly relevant
if retrieval.items and retrieval.items[0]["score"] > 0.5:
    # Good context, use it
    prompt = build_prompt_with_context(...)
else:
    # Poor context, use base prompt
    prompt = build_base_prompt(...)

# Confidence thresholds
if result.confidence < 0.6:
    # Low confidence, fall back to rules
    category = apply_rule_based_categorization(task)
```

## Summary

In this chapter we built:

1. ✅ Complete RAG pipeline (retrieve → augment → generate)
2. ✅ Context formatting for different tasks
3. ✅ RAG-enhanced categorization
4. ✅ RAG-enhanced prioritization
5. ✅ Confidence-based fallback logic

RAG implements Principle 4 in action:
- **Memory** (ChromaDB) creates **context**
- **Context** enables **consistent behavior**
- **Behavior** emerges from **patterns**, not just rules

In the next chapter, we'll apply RAG to auto-categorization in depth.

---

**Previous**: [Chapter 12: ChromaDB Embeddings](./chapter-12-chromadb.md)  
**Next**: [Chapter 14: Auto-Categorization](./chapter-14-categorization.md)
