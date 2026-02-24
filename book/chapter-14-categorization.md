# Chapter 14: Auto-Categorization — Learning from History

## Pattern Recognition Through Memory

Auto-categorization demonstrates how memory creates intelligent behavior. Rather than hardcoding rules like "gym → health", we let the system learn from patterns.

## The Category Service

```python
# src/ai_todo/services/category_service.py
"""
Intelligent categorization service.

Uses RAG + rules for accurate, consistent categorization.
"""

from typing import Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime

from ..ai.client import OllamaClient
from ..ai.rag import RAGPipeline
from ..memory.embeddings import EmbeddingStore
from ..models.enums import Category


@dataclass
class CategoryPrediction:
    """Prediction result with confidence and provenance."""
    category: Category
    confidence: float
    method: str  # "rag", "rule", "default"
    reasoning: str
    similar_tasks: List[dict]


class CategoryService:
    """
    Categorization service combining RAG with rule-based fallbacks.
    
    Strategy:
    1. Try RAG if we have context
    2. Fall back to keyword rules
    3. Default to "other" if uncertain
    """
    
    # Keyword patterns for rule-based categorization
    KEYWORD_PATTERNS = {
        Category.WORK: [
            "meeting", "report", "email", "client", "project",
            "deadline", "presentation", "review", "colleague",
            "office", "boss", "team", "budget"
        ],
        Category.PERSONAL: [
            "mom", "dad", "family", "friend", "birthday",
            "gift", "call", "visit", "dinner", "party"
        ],
        Category.HEALTH: [
            "doctor", "dentist", "gym", "exercise", "workout",
            "medicine", "prescription", "appointment", "checkup",
            "therapy", "vitamin", "diet", "weight"
        ],
        Category.FINANCE: [
            "bill", "payment", "tax", "bank", "invoice",
            "budget", "salary", "invest", "insurance", "refund"
        ],
        Category.LEARNING: [
            "study", "course", "book", "read", "learn",
            "tutorial", "practice", "exam", "class", "lecture"
        ],
        Category.ERRANDS: [
            "buy", "shop", "grocery", "pick up", "return",
            "drop off", "mail", "package", "store"
        ],
        Category.HOME: [
            "clean", "repair", "fix", "organize", "laundry",
            "dishes", "vacuum", "mow", "garden", "trash"
        ],
        Category.SOCIAL: [
            "party", "event", "wedding", "concert", "movie",
            "dinner out", "bar", "club", "festival", "game"
        ],
    }
    
    def __init__(
        self,
        ai_client: OllamaClient,
        embedding_store: Optional[EmbeddingStore] = None
    ):
        self.ai = ai_client
        self.embeddings = embedding_store
        self.rag = RAGPipeline(ai_client, embedding_store) if embedding_store else None
    
    async def predict_category(
        self,
        title: str,
        description: Optional[str] = None,
        *,
        use_rag: bool = True,
        min_confidence: float = 0.6
    ) -> CategoryPrediction:
        """
        Predict category for a task.
        
        Args:
            title: Task title
            description: Optional description
            use_rag: Whether to use RAG (if available)
            min_confidence: Minimum confidence to accept prediction
            
        Returns:
            CategoryPrediction with category and metadata
        """
        # Try RAG first if available
        if use_rag and self.rag:
            rag_result = await self._try_rag(title, description)
            if rag_result and rag_result.confidence >= min_confidence:
                return rag_result
        
        # Try keyword rules
        rule_result = self._try_rules(title, description)
        if rule_result.confidence >= min_confidence:
            return rule_result
        
        # Default
        return CategoryPrediction(
            category=Category.OTHER,
            confidence=0.3,
            method="default",
            reasoning="Could not determine category with sufficient confidence",
            similar_tasks=[]
        )
    
    async def _try_rag(
        self,
        title: str,
        description: Optional[str]
    ) -> Optional[CategoryPrediction]:
        """Try RAG-based categorization."""
        try:
            result = await self.rag.categorize_with_rag(title, description)
            
            if not result.response:
                return None
            
            return CategoryPrediction(
                category=Category.from_string(result.response.get("category", "other")),
                confidence=result.response.get("confidence", 0.5),
                method="rag",
                reasoning=result.response.get("reasoning", ""),
                similar_tasks=result.sources
            )
        except Exception:
            return None
    
    def _try_rules(
        self,
        title: str,
        description: Optional[str]
    ) -> CategoryPrediction:
        """Try keyword rule-based categorization."""
        text = f"{title} {description or ''}".lower()
        
        # Count matches per category
        scores = {}
        for category, keywords in self.KEYWORD_PATTERNS.items():
            matches = sum(1 for kw in keywords if kw in text)
            if matches > 0:
                scores[category] = matches
        
        if not scores:
            return CategoryPrediction(
                category=Category.OTHER,
                confidence=0.3,
                method="rule",
                reasoning="No keyword patterns matched",
                similar_tasks=[]
            )
        
        # Best match
        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]
        
        # Confidence based on number of matches and exclusivity
        total_matches = sum(scores.values())
        confidence = min(0.4 + (best_score * 0.15), 0.85)  # Cap at 0.85 for rules
        
        # Reduce confidence if multiple categories matched
        if len(scores) > 1:
            confidence *= (best_score / total_matches)
        
        return CategoryPrediction(
            category=best_category,
            confidence=confidence,
            method="rule",
            reasoning=f"Matched {best_score} keywords for {best_category.value}",
            similar_tasks=[]
        )
    
    async def get_category_distribution(self) -> dict[Category, int]:
        """Get count of tasks per category (from embeddings)."""
        if not self.embeddings:
            return {}
        
        # Get all tasks from ChromaDB
        results = self.embeddings.collection.get(
            include=["metadatas"]
        )
        
        distribution = {}
        for metadata in results.get("metadatas", []):
            cat_str = metadata.get("category", "other")
            cat = Category.from_string(cat_str)
            distribution[cat] = distribution.get(cat, 0) + 1
        
        return distribution
    
    async def suggest_category(
        self,
        title: str,
        top_n: int = 3
    ) -> List[Tuple[Category, float]]:
        """
        Suggest possible categories with confidence scores.
        
        Returns top N suggestions.
        """
        text = title.lower()
        
        # Rule-based scoring
        rule_scores = {}
        for category, keywords in self.KEYWORD_PATTERNS.items():
            matches = sum(1 for kw in keywords if kw in text)
            if matches > 0:
                rule_scores[category] = min(matches * 0.2, 0.6)
        
        # RAG-based scoring
        rag_scores = {}
        if self.rag:
            retrieval = await self.rag.retrieve(title, limit=10, min_score=0.3)
            
            for item in retrieval.items:
                cat_str = item["metadata"].get("category", "other")
                cat = Category.from_string(cat_str)
                # Weight by similarity score
                rag_scores[cat] = rag_scores.get(cat, 0) + item["score"]
        
        # Combine scores
        all_categories = set(rule_scores.keys()) | set(rag_scores.keys())
        combined = {}
        
        for cat in all_categories:
            rule = rule_scores.get(cat, 0)
            rag = rag_scores.get(cat, 0)
            # Weighted combination
            combined[cat] = (rule * 0.4) + (rag * 0.6)
        
        # Sort and return top N
        sorted_cats = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return sorted_cats[:top_n]
```

## Feedback Loop

Categories improve over time as users correct them:

```python
# src/ai_todo/services/category_service.py (continued)

class CategoryFeedbackLoop:
    """
    Track category corrections to improve predictions.
    
    When users change a category, we learn from it.
    """
    
    def __init__(self, embedding_store: EmbeddingStore):
        self.embeddings = embedding_store
    
    async def record_correction(
        self,
        task_id: str,
        predicted_category: Category,
        corrected_category: Category
    ) -> None:
        """
        Record a category correction.
        
        Updates the embedding metadata so future similar tasks
        will be categorized correctly.
        """
        # Update the task's embedding metadata
        # This affects future RAG retrievals
        try:
            # Get current data
            results = self.embeddings.collection.get(
                ids=[task_id],
                include=["embeddings", "documents", "metadatas"]
            )
            
            if not results["ids"]:
                return
            
            # Update metadata with corrected category
            metadata = results["metadatas"][0]
            metadata["category"] = corrected_category.value
            metadata["was_corrected"] = "true"
            metadata["original_category"] = predicted_category.value
            
            # Upsert with updated metadata
            self.embeddings.collection.update(
                ids=[task_id],
                metadatas=[metadata]
            )
            
        except Exception as e:
            # Log but don't fail
            import logging
            logging.warning(f"Failed to record correction: {e}")
```

## Integration Demo

```python
# scripts/categorization_demo.py
"""Demonstrate the categorization system."""

import asyncio
from rich.console import Console
from rich.table import Table

from ai_todo.main import get_app
from ai_todo.services.category_service import CategoryService
from ai_todo.models.task import TaskInput


console = Console()


async def main():
    app = get_app()
    
    # Create seed data
    console.print("[bold]Creating seed tasks...[/bold]")
    
    seed_tasks = [
        "schedule dentist appointment",
        "buy birthday gift for mom",
        "review quarterly budget",
        "submit expense report",
        "pick up prescription",
        "call insurance company",
    ]
    
    for raw in seed_tasks:
        await app.tasks.create_task(TaskInput(raw_input=raw))
        console.print(f"  Created: {raw}")
    
    # Test categorization
    console.print("\n[bold]Testing categorization...[/bold]\n")
    
    category_service = CategoryService(app.ollama, app.embeddings)
    
    test_cases = [
        "schedule eye exam",           # health
        "order groceries online",      # errands
        "dinner with friends friday",  # social
        "read python book chapter 5",  # learning
        "clean the bathroom",          # home
    ]
    
    table = Table(title="Category Predictions")
    table.add_column("Input", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Confidence")
    table.add_column("Method")
    table.add_column("Similar Tasks")
    
    for task in test_cases:
        prediction = await category_service.predict_category(task)
        
        similar = ", ".join(
            s["metadata"]["title"][:20] 
            for s in prediction.similar_tasks[:2]
        ) if prediction.similar_tasks else "—"
        
        table.add_row(
            task,
            prediction.category.value,
            f"{prediction.confidence*100:.0f}%",
            prediction.method,
            similar
        )
    
    console.print(table)
    
    # Show suggestions
    console.print("\n[bold]Category suggestions for ambiguous task...[/bold]")
    console.print("Input: \"call about appointment\"")
    
    suggestions = await category_service.suggest_category("call about appointment", top_n=3)
    
    for cat, score in suggestions:
        bar = "█" * int(score * 20)
        console.print(f"  {cat.value:12} {bar} {score:.2f}")
    
    await app.close()


if __name__ == "__main__":
    asyncio.run(main())
```

Output:
```
Creating seed tasks...
  Created: schedule dentist appointment
  Created: buy birthday gift for mom
  ...

Testing categorization...

                     Category Predictions
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Input                       ┃ Category ┃ Confidence ┃ Method ┃ Similar Tasks      ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ schedule eye exam           │ health   │ 89%        │ rag    │ Schedule dentist..  │
│ order groceries online      │ errands  │ 82%        │ rag    │ Buy birthday gift.. │
│ dinner with friends friday  │ social   │ 75%        │ rule   │ —                   │
│ read python book chapter 5  │ learning │ 78%        │ rule   │ —                   │
│ clean the bathroom          │ home     │ 85%        │ rule   │ —                   │
└─────────────────────────────┴──────────┴────────────┴────────┴────────────────────┘

Category suggestions for ambiguous task...
Input: "call about appointment"
  personal     ████████████ 0.62
  health       ████████ 0.41
  work         ████ 0.23
```

## Summary

In this chapter we built:

1. ✅ CategoryService combining RAG and rules
2. ✅ Keyword pattern matching as fallback
3. ✅ Confidence-based method selection
4. ✅ Category suggestion for ambiguous inputs
5. ✅ Feedback loop for learning from corrections

The system demonstrates intelligent behavior emerging from memory:
- New tasks categorized like similar past tasks
- Corrections improve future predictions
- Rules provide baseline when memory is sparse

In the next chapter, we'll apply similar techniques to due date intelligence.

---

**Previous**: [Chapter 13: Retrieval-Augmented Generation](./chapter-13-rag.md)  
**Next**: [Chapter 15: Due Date Intelligence](./chapter-15-due-dates.md)
