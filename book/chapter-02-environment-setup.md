# Chapter 2: Setting Up Your Development Environment

## Overview

A solid development environment is crucial for productive AI development. In this chapter, we'll set up everything you need to build our AI-first todo application.

## System Requirements

### Minimum Requirements
- **CPU**: Any modern multi-core processor
- **RAM**: 8GB (for smaller models like Phi-3 or Gemma 2B)
- **Storage**: 10GB free space
- **OS**: macOS, Linux, or Windows with WSL2

### Recommended Requirements
- **CPU**: Apple Silicon (M1/M2/M3) or modern x86 with AVX2
- **RAM**: 16GB or more
- **Storage**: 20GB free space
- **GPU**: Optional but helpful (NVIDIA with CUDA or Apple Silicon)

## Installing Ollama

Ollama is our local LLM inference engine. It makes running large language models as simple as running any other command-line tool.

### macOS Installation

```bash
# Using the official installer (recommended)
curl -fsSL https://ollama.com/install.sh | sh

# Or using Homebrew
brew install ollama
```

### Linux Installation

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows Installation

1. Download the installer from [ollama.com](https://ollama.com)
2. Run the installer
3. Ollama will be available in your terminal

### Verify Installation

```bash
ollama --version
```

You should see output like:
```
ollama version 0.1.29
```

## Starting the Ollama Server

Ollama runs as a background service. Start it with:

```bash
ollama serve
```

On macOS and Windows, Ollama typically starts automatically. On Linux, you may need to start it manually or set up a systemd service.

### Verify the Server is Running

```bash
curl http://localhost:11434/api/tags
```

You should see a JSON response (possibly with an empty models list).

## Downloading Your First Model

Let's download a model suitable for our todo application. We'll use **Llama 3.2** as our primary model:

```bash
# Download Llama 3.2 3B (recommended for most systems)
ollama pull llama3.2

# Or for systems with more RAM, use the 8B variant
ollama pull llama3.2:8b
```

### Alternative Models

Depending on your hardware, consider these alternatives:

| Model | Size | RAM Needed | Best For |
|-------|------|------------|----------|
| `phi3:mini` | 2.3GB | 4GB | Low-resource systems |
| `llama3.2` | 2.0GB | 4GB | Balanced performance |
| `llama3.2:8b` | 4.7GB | 8GB | Better reasoning |
| `mistral` | 4.1GB | 8GB | Strong all-around |
| `llama3.1:8b` | 4.7GB | 8GB | Excellent instruction following |

### Test Your Model

```bash
ollama run llama3.2 "Say hello in one sentence"
```

You should see a greeting response. Press Ctrl+D to exit.

## Setting Up Python

### Install Python 3.11+

**macOS (using Homebrew):**
```bash
brew install python@3.12
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv
```

**Windows:**
Download from [python.org](https://python.org) or use winget:
```powershell
winget install Python.Python.3.12
```

### Verify Python Installation

```bash
python3 --version
# Should output: Python 3.12.x (or 3.11.x)
```

## Creating the Project

### Create Project Directory

```bash
mkdir ai-todo-app
cd ai-todo-app
```

### Set Up Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
.\venv\Scripts\activate
```

Your prompt should now show `(venv)` indicating the virtual environment is active.

### Initialize Project Structure

```bash
mkdir -p src/ai_todo tests
touch src/__init__.py
touch src/ai_todo/__init__.py
touch pyproject.toml
touch README.md
```

## Installing Dependencies

Create a `requirements.txt` file:

```bash
cat > requirements.txt << 'EOF'
# Core dependencies
httpx>=0.27.0          # HTTP client for Ollama API
pydantic>=2.6.0        # Data validation
rich>=13.7.0           # Beautiful terminal output

# Development dependencies
pytest>=8.0.0          # Testing
pytest-asyncio>=0.23.0 # Async test support
ruff>=0.3.0            # Linting and formatting

# Optional but recommended
python-dateutil>=2.8.0 # Date parsing
EOF
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Creating pyproject.toml

```toml
[project]
name = "ai-todo-app"
version = "0.1.0"
description = "An AI-first todo application using local LLMs"
requires-python = ">=3.11"
dependencies = [
    "httpx>=0.27.0",
    "pydantic>=2.6.0",
    "rich>=13.7.0",
    "python-dateutil>=2.8.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.3.0",
]

[project.scripts]
todo = "ai_todo.cli:main"

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

## Verify Everything Works

Let's create a simple verification script:

```bash
cat > verify_setup.py << 'EOF'
"""Verify that the development environment is correctly set up."""

import sys
import subprocess

def check_python():
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print("❌ Python 3.11+ required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_ollama():
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ Ollama installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    print("❌ Ollama not found")
    return False

def check_ollama_server():
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama server running ({len(models)} models available)")
            return True
    except Exception:
        pass
    print("❌ Ollama server not running (run 'ollama serve')")
    return False

def check_dependencies():
    required = ["httpx", "pydantic", "rich"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        return False
    print("✅ All required packages installed")
    return True

if __name__ == "__main__":
    print("\n🔍 Checking development environment...\n")
    
    results = [
        check_python(),
        check_ollama(),
        check_ollama_server(),
        check_dependencies(),
    ]
    
    print()
    if all(results):
        print("🎉 Environment is ready! You can proceed to the next chapter.")
    else:
        print("⚠️  Please fix the issues above before continuing.")
        sys.exit(1)
EOF
```

Run the verification:

```bash
python verify_setup.py
```

You should see all green checkmarks:

```
🔍 Checking development environment...

✅ Python 3.12.2
✅ Ollama installed: ollama version 0.1.29
✅ Ollama server running (1 models available)
✅ All required packages installed

🎉 Environment is ready! You can proceed to the next chapter.
```

## Troubleshooting

### Ollama Server Won't Start

```bash
# Check if something is using port 11434
lsof -i :11434

# Kill any existing process and restart
pkill ollama
ollama serve
```

### Model Download Fails

```bash
# Check available disk space
df -h

# Try downloading a smaller model
ollama pull phi3:mini
```

### Python Import Errors

```bash
# Ensure you're in the virtual environment
which python  # Should point to venv/bin/python

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

## Project Structure So Far

```
ai-todo-app/
├── src/
│   ├── __init__.py
│   └── ai_todo/
│       └── __init__.py
├── tests/
├── venv/
├── pyproject.toml
├── requirements.txt
├── verify_setup.py
└── README.md
```

## Summary

In this chapter, we:

1. ✅ Installed Ollama and downloaded a language model
2. ✅ Set up Python 3.11+ with a virtual environment
3. ✅ Created the project structure
4. ✅ Installed all required dependencies
5. ✅ Verified everything works together

## What's Next

In Chapter 3, we'll explore Ollama in depth—understanding how it works, the API it exposes, and how to interact with it programmatically from Python.

---

**Previous**: [Chapter 1: Introduction](./chapter-01-introduction.md)  
**Next**: [Chapter 3: Understanding Ollama](./chapter-03-ollama-basics.md)
