# Chapter 16: CLI Interface — User Interaction Design

## The Interface Layer

The CLI is the **Routes** layer in our architecture—it accepts input and returns output. It never contains business logic. It never calls the AI directly. It orchestrates services and formats responses.

```
User → CLI → Services → AI → Validation → Persistence → Memory
```

This chapter builds a professional terminal interface using **Rich**.

## Why Rich?

Rich provides:
- **Beautiful tables** for task lists
- **Progress spinners** for AI operations
- **Colored output** for priorities and status
- **Panels and boxes** for structured display
- **Markdown rendering** for help text

All without external UI dependencies.

## CLI Structure

```python
# src/ai_todo/cli/__init__.py
"""
CLI package for AI-first todo application.

Commands:
- add: Create a task from natural language
- list: Show tasks with filtering
- complete: Mark task as done
- delete: Remove a task
- search: Semantic search
- show: Display task details
"""

from .app import app

__all__ = ["app"]
```

## The Main Application

```python
# src/ai_todo/cli/app.py
"""
Main CLI application using Typer and Rich.

Typer provides CLI argument parsing.
Rich provides beautiful output.
"""

import asyncio
from typing import Optional
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm

from ..main import get_app
from ..models.task import TaskInput
from ..models.enums import Priority, Category, TaskStatus


# Initialize
app = typer.Typer(
    name="todo",
    help="AI-powered todo application",
    no_args_is_help=True
)
console = Console()


def run_async(coro):
    """Run async function in sync context."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ==============================================================================
# ADD COMMAND
# ==============================================================================

@app.command()
def add(
    task: str = typer.Argument(
        ...,
        help="Natural language task description"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show detailed AI processing info"
    )
):
    """
    Add a new task using natural language.
    
    Examples:
        todo add "call mom tomorrow afternoon"
        todo add "urgent: finish report by Friday EOD"
        todo add "buy groceries sometime this week"
    """
    async def _add():
        application = get_app()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            progress.add_task("Processing with AI...", total=None)
            
            result = await application.tasks.create_task(
                TaskInput(raw_input=task)
            )
        
        # Display result
        created = result.task
        
        console.print()
        console.print(f"[green]✓ Task created[/green]")
        console.print()
        
        _display_task_card(created)
        
        if verbose:
            console.print()
            console.print(f"[dim]AI processing time: {result.ai_duration_ms:.0f}ms[/dim]")
            
            if result.similar_tasks:
                console.print(f"[dim]Similar tasks found: {len(result.similar_tasks)}[/dim]")
        
        await application.close()
    
    run_async(_add())


# ==============================================================================
# LIST COMMAND
# ==============================================================================

@app.command("list")
def list_tasks(
    status: Optional[str] = typer.Option(
        None, "--status", "-s",
        help="Filter by status: pending, completed, cancelled"
    ),
    priority: Optional[str] = typer.Option(
        None, "--priority", "-p",
        help="Filter by priority: urgent, high, medium, low"
    ),
    category: Optional[str] = typer.Option(
        None, "--category", "-c",
        help="Filter by category: work, personal, errands, health, finance, learning"
    ),
    limit: int = typer.Option(
        20, "--limit", "-n",
        help="Maximum number of tasks to show"
    ),
    all_tasks: bool = typer.Option(
        False, "--all", "-a",
        help="Include completed tasks"
    )
):
    """
    List tasks with optional filtering.
    
    Examples:
        todo list
        todo list --status pending --priority urgent
        todo list --category work --limit 5
        todo list --all
    """
    async def _list():
        application = get_app()
        
        # Parse filters
        status_filter = None
        if status:
            try:
                status_filter = TaskStatus(status.lower())
            except ValueError:
                console.print(f"[red]Invalid status: {status}[/red]")
                raise typer.Exit(1)
        elif not all_tasks:
            # Default to pending only
            status_filter = TaskStatus.PENDING
        
        priority_filter = None
        if priority:
            try:
                priority_filter = Priority(priority.lower())
            except ValueError:
                console.print(f"[red]Invalid priority: {priority}[/red]")
                raise typer.Exit(1)
        
        category_filter = None
        if category:
            try:
                category_filter = Category(category.lower())
            except ValueError:
                console.print(f"[red]Invalid category: {category}[/red]")
                raise typer.Exit(1)
        
        tasks = await application.tasks.list_tasks(
            status=status_filter,
            priority=priority_filter,
            category=category_filter,
            limit=limit
        )
        
        if not tasks:
            console.print("[dim]No tasks found[/dim]")
            await application.close()
            return
        
        _display_task_table(tasks)
        
        await application.close()
    
    run_async(_list())


# ==============================================================================
# COMPLETE COMMAND
# ==============================================================================

@app.command()
def complete(
    task_id: str = typer.Argument(
        ...,
        help="Task ID (or prefix) to mark as complete"
    )
):
    """
    Mark a task as completed.
    
    You can use the full ID or just the first few characters.
    
    Example:
        todo complete a1b2c3
    """
    async def _complete():
        application = get_app()
        
        # Find task by ID or prefix
        task = await _find_task_by_id_prefix(application, task_id)
        
        if task is None:
            console.print(f"[red]Task not found: {task_id}[/red]")
            raise typer.Exit(1)
        
        if task.completed:
            console.print(f"[yellow]Task already completed[/yellow]")
            raise typer.Exit(0)
        
        updated = await application.tasks.complete_task(task.id)
        
        console.print(f"[green]✓ Completed: {updated.title}[/green]")
        
        await application.close()
    
    run_async(_complete())


# ==============================================================================
# DELETE COMMAND
# ==============================================================================

@app.command()
def delete(
    task_id: str = typer.Argument(
        ...,
        help="Task ID (or prefix) to delete"
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Skip confirmation"
    )
):
    """
    Delete a task.
    
    Example:
        todo delete a1b2c3
        todo delete a1b2c3 --force
    """
    async def _delete():
        application = get_app()
        
        task = await _find_task_by_id_prefix(application, task_id)
        
        if task is None:
            console.print(f"[red]Task not found: {task_id}[/red]")
            raise typer.Exit(1)
        
        if not force:
            console.print()
            _display_task_card(task)
            console.print()
            
            if not Confirm.ask("Delete this task?"):
                console.print("[dim]Cancelled[/dim]")
                raise typer.Exit(0)
        
        deleted = await application.tasks.delete_task(task.id)
        
        if deleted:
            console.print(f"[green]✓ Deleted: {task.title}[/green]")
        else:
            console.print(f"[red]Failed to delete task[/red]")
        
        await application.close()
    
    run_async(_delete())


# ==============================================================================
# SEARCH COMMAND
# ==============================================================================

@app.command()
def search(
    query: str = typer.Argument(
        ...,
        help="Search query (semantic search)"
    ),
    limit: int = typer.Option(
        10, "--limit", "-n",
        help="Maximum results"
    )
):
    """
    Search tasks using semantic similarity.
    
    Uses AI embeddings to find related tasks, not just keyword matching.
    
    Examples:
        todo search "meetings about budget"
        todo search "health related"
    """
    async def _search():
        application = get_app()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            progress.add_task("Searching...", total=None)
            
            tasks = await application.tasks.search_tasks(query, limit=limit)
        
        if not tasks:
            console.print(f"[dim]No tasks matching '{query}'[/dim]")
            await application.close()
            return
        
        console.print(f"\n[bold]Results for:[/bold] {query}\n")
        _display_task_table(tasks)
        
        await application.close()
    
    run_async(_search())


# ==============================================================================
# SHOW COMMAND
# ==============================================================================

@app.command()
def show(
    task_id: str = typer.Argument(
        ...,
        help="Task ID (or prefix) to display"
    )
):
    """
    Show detailed information about a task.
    
    Example:
        todo show a1b2c3
    """
    async def _show():
        application = get_app()
        
        task = await _find_task_by_id_prefix(application, task_id)
        
        if task is None:
            console.print(f"[red]Task not found: {task_id}[/red]")
            raise typer.Exit(1)
        
        console.print()
        _display_task_detail(task)
        
        await application.close()
    
    run_async(_show())


# ==============================================================================
# DUE COMMAND
# ==============================================================================

@app.command()
def due(
    hours: int = typer.Option(
        24, "--hours", "-h",
        help="Show tasks due within this many hours"
    )
):
    """
    Show tasks due soon.
    
    Examples:
        todo due
        todo due --hours 48
    """
    async def _due():
        application = get_app()
        
        tasks = await application.tasks.get_tasks_due_soon(hours=hours)
        
        if not tasks:
            console.print(f"[dim]No tasks due in the next {hours} hours[/dim]")
            await application.close()
            return
        
        console.print(f"\n[bold]Tasks due in next {hours} hours:[/bold]\n")
        _display_task_table(tasks, show_due_urgency=True)
        
        await application.close()
    
    run_async(_due())


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

async def _find_task_by_id_prefix(application, id_prefix: str):
    """Find task by full ID or prefix."""
    # Try exact match first
    task = await application.tasks.get_task(id_prefix)
    if task:
        return task
    
    # Try prefix match
    all_tasks = await application.tasks.list_tasks(limit=1000)
    matches = [t for t in all_tasks if t.id.startswith(id_prefix)]
    
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        console.print(f"[yellow]Multiple tasks match '{id_prefix}':[/yellow]")
        for t in matches[:5]:
            console.print(f"  {t.id[:8]} - {t.title}")
        return None
    
    return None


def _display_task_card(task):
    """Display a single task as a card."""
    priority_style = _get_priority_style(task.priority)
    status_style = "green" if task.completed else "white"
    
    lines = [
        f"[bold]{task.title}[/bold]",
        "",
        f"Priority: [{priority_style}]{task.priority.value}[/]",
        f"Category: {task.category.value}",
        f"Status: [{status_style}]{task.status.value}[/]",
    ]
    
    if task.due_date:
        due_str = task.due_date.strftime("%Y-%m-%d %H:%M")
        lines.append(f"Due: {due_str}")
    
    if task.description:
        lines.append("")
        lines.append(f"[dim]{task.description}[/dim]")
    
    content = "\n".join(lines)
    
    console.print(Panel(
        content,
        title=f"[dim]{task.id[:8]}[/dim]",
        border_style="blue"
    ))


def _display_task_detail(task):
    """Display detailed task information."""
    priority_style = _get_priority_style(task.priority)
    
    console.print(Panel.fit(
        f"[bold]{task.title}[/bold]",
        border_style="blue"
    ))
    
    console.print()
    
    table = Table(show_header=False, box=None)
    table.add_column("Field", style="dim")
    table.add_column("Value")
    
    table.add_row("ID", task.id)
    table.add_row("Priority", f"[{priority_style}]{task.priority.value}[/]")
    table.add_row("Category", task.category.value)
    table.add_row("Status", task.status.value)
    table.add_row("Completed", "Yes" if task.completed else "No")
    
    if task.due_date:
        due_str = task.due_date.strftime("%A, %B %d, %Y at %H:%M")
        table.add_row("Due Date", due_str)
    else:
        table.add_row("Due Date", "[dim]Not set[/dim]")
    
    table.add_row("Created", task.created_at.strftime("%Y-%m-%d %H:%M"))
    table.add_row("Updated", task.updated_at.strftime("%Y-%m-%d %H:%M"))
    
    console.print(table)
    
    if task.description:
        console.print()
        console.print(Panel(
            task.description,
            title="Description",
            border_style="dim"
        ))


def _display_task_table(tasks, show_due_urgency=False):
    """Display tasks as a table."""
    table = Table()
    
    table.add_column("ID", style="dim", width=8)
    table.add_column("Title", style="cyan", max_width=40)
    table.add_column("Priority", justify="center")
    table.add_column("Category")
    table.add_column("Due")
    table.add_column("Status")
    
    now = datetime.now()
    
    for task in tasks:
        priority_style = _get_priority_style(task.priority)
        
        # Format due date
        if task.due_date:
            if show_due_urgency:
                delta = task.due_date - now
                if delta.total_seconds() < 0:
                    due_str = f"[red]OVERDUE[/red]"
                elif delta.total_seconds() < 3600:
                    due_str = f"[red]{int(delta.total_seconds() / 60)}m[/red]"
                elif delta.total_seconds() < 86400:
                    due_str = f"[yellow]{int(delta.total_seconds() / 3600)}h[/yellow]"
                else:
                    due_str = task.due_date.strftime("%m/%d")
            else:
                due_str = task.due_date.strftime("%m/%d %H:%M")
        else:
            due_str = "[dim]—[/dim]"
        
        # Format status
        if task.completed:
            status_str = "[green]done[/green]"
        elif task.status == TaskStatus.CANCELLED:
            status_str = "[dim]cancelled[/dim]"
        else:
            status_str = task.status.value
        
        table.add_row(
            task.id[:8],
            task.title[:40],
            f"[{priority_style}]{task.priority.value}[/]",
            task.category.value,
            due_str,
            status_str
        )
    
    console.print(table)
    console.print(f"\n[dim]{len(tasks)} task(s)[/dim]")


def _get_priority_style(priority: Priority) -> str:
    """Get Rich style for priority."""
    return {
        Priority.URGENT: "red bold",
        Priority.HIGH: "yellow",
        Priority.MEDIUM: "white",
        Priority.LOW: "dim",
    }.get(priority, "white")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
```

## Setting Up the Entry Point

```toml
# pyproject.toml

[project.scripts]
todo = "ai_todo.cli.app:main"
```

After installing the package:

```bash
pip install -e .
```

You can run:

```bash
todo add "call mom tomorrow"
todo list
todo complete a1b2
```

## Interactive Mode

For a more immersive experience, add an interactive mode:

```python
# src/ai_todo/cli/interactive.py
"""
Interactive mode for continuous task management.
"""

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.live import Live
from rich.table import Table

from ..main import get_app
from ..models.task import TaskInput


console = Console()


async def interactive_session():
    """Run interactive session."""
    console.print(Panel.fit(
        "[bold]AI-First Todo[/bold]\n\n"
        "Commands:\n"
        "  [cyan]add[/cyan] <task>     Add a new task\n"
        "  [cyan]list[/cyan]           Show all tasks\n"
        "  [cyan]done[/cyan] <id>      Complete a task\n"
        "  [cyan]search[/cyan] <query> Search tasks\n"
        "  [cyan]quit[/cyan]           Exit\n",
        title="Interactive Mode"
    ))
    
    app = get_app()
    
    while True:
        try:
            console.print()
            user_input = Prompt.ask("[bold cyan]>[/bold cyan]")
            
            if not user_input.strip():
                continue
            
            parts = user_input.strip().split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            if command in ("quit", "exit", "q"):
                console.print("[dim]Goodbye![/dim]")
                break
            
            elif command == "add":
                if not args:
                    console.print("[red]Usage: add <task description>[/red]")
                    continue
                
                result = await app.tasks.create_task(TaskInput(raw_input=args))
                task = result.task
                console.print(f"[green]✓[/green] Created: {task.title}")
                console.print(f"  [dim]Priority: {task.priority.value}, Category: {task.category.value}[/dim]")
            
            elif command == "list":
                tasks = await app.tasks.list_tasks(limit=10)
                if not tasks:
                    console.print("[dim]No tasks[/dim]")
                else:
                    _show_quick_list(tasks)
            
            elif command == "done":
                if not args:
                    console.print("[red]Usage: done <task-id>[/red]")
                    continue
                
                task = await app.tasks.complete_task(args)
                if task:
                    console.print(f"[green]✓[/green] Completed: {task.title}")
                else:
                    console.print(f"[red]Task not found: {args}[/red]")
            
            elif command == "search":
                if not args:
                    console.print("[red]Usage: search <query>[/red]")
                    continue
                
                tasks = await app.tasks.search_tasks(args, limit=5)
                if not tasks:
                    console.print("[dim]No matches[/dim]")
                else:
                    _show_quick_list(tasks)
            
            else:
                # Treat unknown commands as task additions
                console.print(f"[dim]Unknown command. Adding as task...[/dim]")
                result = await app.tasks.create_task(TaskInput(raw_input=user_input))
                console.print(f"[green]✓[/green] Created: {result.task.title}")
        
        except KeyboardInterrupt:
            console.print("\n[dim]Use 'quit' to exit[/dim]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    await app.close()


def _show_quick_list(tasks):
    """Show compact task list."""
    for task in tasks:
        priority_marker = {
            "urgent": "[red]![/red]",
            "high": "[yellow]↑[/yellow]",
            "medium": " ",
            "low": "[dim]↓[/dim]",
        }.get(task.priority.value, " ")
        
        status = "[green]✓[/green]" if task.completed else " "
        
        console.print(f"  {status} {priority_marker} [{task.id[:6]}] {task.title}")


# Entry point for interactive mode
def run_interactive():
    """Run interactive mode."""
    import asyncio
    asyncio.run(interactive_session())
```

## Demo Output

```
$ todo add "finish quarterly report by Friday EOD - urgent"

✓ Task created

╭── a1b2c3d4 ───────────────────────────────────────────────╮
│ Finish quarterly report                                   │
│                                                           │
│ Priority: urgent                                          │
│ Category: work                                            │
│ Status: pending                                           │
│ Due: 2024-01-19 17:00                                     │
╰───────────────────────────────────────────────────────────╯

$ todo list

┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ ID       ┃ Title                    ┃ Priority ┃ Category ┃ Due         ┃ Status  ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━┩
│ a1b2c3d4 │ Finish quarterly report  │ urgent   │ work     │ 01/19 17:00 │ pending │
│ e5f6g7h8 │ Call mom                 │ medium   │ personal │ 01/16 14:00 │ pending │
│ i9j0k1l2 │ Buy groceries            │ low      │ errands  │ —           │ pending │
└──────────┴──────────────────────────┴──────────┴──────────┴─────────────┴─────────┘

3 task(s)

$ todo due --hours 48

Tasks due in next 48 hours:

┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┓
┃ ID       ┃ Title                    ┃ Priority ┃ Category ┃ Due   ┃ Status  ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━┩
│ e5f6g7h8 │ Call mom                 │ medium   │ personal │ 18h   │ pending │
│ a1b2c3d4 │ Finish quarterly report  │ urgent   │ work     │ 01/19 │ pending │
└──────────┴──────────────────────────┴──────────┴──────────┴───────┴─────────┘

2 task(s)

$ todo complete e5f6

✓ Completed: Call mom

$ todo search "budget stuff"

Results for: budget stuff

┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ ID       ┃ Title                    ┃ Priority ┃ Category ┃ Due         ┃ Status  ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━┩
│ a1b2c3d4 │ Finish quarterly report  │ urgent   │ work     │ 01/19 17:00 │ pending │
│ m3n4o5p6 │ Review department budget │ high     │ work     │ 01/25 17:00 │ pending │
└──────────┴──────────────────────────┴──────────┴──────────┴─────────────┴─────────┘

2 task(s)
```

## CLI Design Principles

### 1. Progressive Disclosure

Simple commands for simple needs:

```bash
todo add "buy milk"          # Minimal
todo list                    # Just show tasks
```

Options for power users:

```bash
todo add "buy milk" --verbose
todo list --priority urgent --category work --limit 50
```

### 2. Forgiving Input

The CLI accepts ID prefixes:

```bash
todo complete a1b2           # Instead of full UUID
todo show a1                 # Even shorter (if unique)
```

### 3. Confirmation for Destructive Actions

```bash
$ todo delete a1b2c3d4

╭── a1b2c3d4 ──────────────────╮
│ Finish quarterly report       │
│                               │
│ Priority: urgent              │
│ Category: work                │
╰───────────────────────────────╯

Delete this task? [y/n]: 
```

Use `--force` to skip:

```bash
todo delete a1b2c3d4 --force
```

### 4. Visual Hierarchy

- **Colors** indicate importance (red=urgent, yellow=high)
- **Icons** provide quick scanning (✓=done, !=urgent)
- **Tables** organize information
- **Panels** highlight single items

### 5. Consistent Output

All commands follow the same patterns:
- Success: `[green]✓[/green] Action: description`
- Error: `[red]Error message[/red]`
- Info: `[dim]Additional context[/dim]`

## Error States

```python
# src/ai_todo/cli/errors.py
"""
CLI error handling.
"""

from rich.console import Console
from rich.panel import Panel


console = Console()


def display_error(message: str, hint: str = None):
    """Display formatted error."""
    content = f"[red]{message}[/red]"
    
    if hint:
        content += f"\n\n[dim]Hint: {hint}[/dim]"
    
    console.print(Panel(content, title="Error", border_style="red"))


def display_ai_error(error: Exception):
    """Display AI-related error."""
    display_error(
        f"AI processing failed: {error}",
        hint="Is Ollama running? Try: ollama serve"
    )


def display_not_found(item_type: str, identifier: str):
    """Display not found error."""
    display_error(
        f"{item_type} not found: {identifier}",
        hint=f"Use 'todo list' to see available {item_type.lower()}s"
    )
```

## Summary

In this chapter we built:

1. **Complete CLI** with Typer and Rich
2. **Six commands**: add, list, complete, delete, search, show, due
3. **Interactive mode** for continuous use
4. **Visual formatting** with colors, tables, and panels
5. **User-friendly features**: ID prefixes, confirmations, progress indicators

The CLI is purely a presentation layer—it contains no business logic. All intelligence comes from the services layer.

---

**Previous**: [Chapter 15: Due Date Intelligence](./chapter-15-due-dates.md)  
**Next**: [Chapter 17: Testing AI Systems](./chapter-17-testing.md)
