# Chapter 3: Understanding Ollama — Local LLM Architecture

## What is Ollama?

Ollama is a local LLM inference engine that abstracts away the complexity of running large language models. It handles:

- Model downloading and storage
- Memory management and quantization
- GPU/CPU inference optimization
- A REST API for programmatic access

For AI-first development, Ollama provides a critical guarantee: **your data never leaves your machine**.

## How Ollama Works

```
┌────────────────────────────────────────────────────────────────┐
│                        OLLAMA ARCHITECTURE                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐      ┌─────────────────────────────────────┐ │
│   │ Your Python │      │           Ollama Server             │ │
│   │    Code     │─────▶│         localhost:11434             │ │
│   └─────────────┘      └─────────────┬───────────────────────┘ │
│                                      │                         │
│                                      ▼                         │
│                        ┌─────────────────────────────────────┐ │
│                        │         Model Runner                │ │
│                        │                                     │ │
│                        │  ┌─────────────────────────────┐   │ │
│                        │  │   llama.cpp / Metal / CUDA  │   │ │
│                        │  └─────────────────────────────┘   │ │
│                        └─────────────────────────────────────┘ │
│                                      │                         │
│                                      ▼                         │
│                        ┌─────────────────────────────────────┐ │
│                        │         Model Files                 │ │
│                        │    ~/.ollama/models/                │ │
│                        │                                     │ │
│                        │  • llama3.2 (2.0GB)                │ │
│                        │  • mistral (4.1GB)                 │ │
│                        │  • phi3:mini (2.3GB)               │ │
│                        └─────────────────────────────────────┘ │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Components

1. **Ollama Server**: HTTP server listening on port 11434
2. **Model Runner**: Inference engine (llama.cpp) optimized for your hardware
3. **Model Files**: Quantized model weights stored locally

## The Ollama API

Ollama exposes a REST API with several endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/generate` | POST | Generate text completion |
| `/api/chat` | POST | Chat completion (multi-turn) |
| `/api/embeddings` | POST | Generate embeddings |
| `/api/tags` | GET | List available models |
| `/api/show` | POST | Model information |
| `/api/pull` | POST | Download a model |

### The Generate Endpoint

This is the core endpoint we'll use:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "What is the capital of France?",
  "stream": false
}'
```

Response:
```json
{
  "model": "llama3.2",
  "response": "The capital of France is Paris.",
  "done": true,
  "total_duration": 1234567890,
  "load_duration": 123456789,
  "prompt_eval_count": 8,
  "eval_count": 12
}
```

### Key Parameters

```json
{
  "model": "llama3.2",           // Required: model name
  "prompt": "...",               // Required: input text
  "stream": false,               // Whether to stream response
  "options": {
    "temperature": 0.1,          // Randomness (0.0-2.0)
    "top_p": 0.9,                // Nucleus sampling
    "top_k": 40,                 // Top-k sampling
    "num_predict": 256,          // Max tokens to generate
    "stop": ["\n", "###"]        // Stop sequences
  },
  "system": "You are a helpful assistant.",  // System prompt
  "format": "json"               // Force JSON output
}
```

## Temperature in Practice

Remember Principle 3: **Temperature Controls Entropy**.

Let's see this in action:

```bash
# Temperature 0.0 - Deterministic
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Name one color:",
  "options": {"temperature": 0.0},
  "stream": false
}'
# Always returns: "Blue"

# Temperature 1.0 - High variance
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2", 
  "prompt": "Name one color:",
  "options": {"temperature": 1.0},
  "stream": false
}'
# Returns: "Blue", "Crimson", "Teal", "Amber"... varies each time
```

For structured data extraction (our primary use case), we use **temperature 0.0-0.2**.

## The Format Parameter

When `"format": "json"` is specified, Ollama constrains the model's output to valid JSON:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Extract the task from: \"buy milk tomorrow\". Return JSON with title and due_date fields.",
  "format": "json",
  "options": {"temperature": 0.1},
  "stream": false
}'
```

Response:
```json
{
  "response": "{\"title\": \"buy milk\", \"due_date\": \"tomorrow\"}"
}
```

This is crucial for Principle 2: **AI Proposes, The System Commits**. JSON output is parseable and validatable.

## Embeddings for Semantic Memory

The `/api/embeddings` endpoint generates vector representations:

```bash
curl http://localhost:11434/api/embeddings -d '{
  "model": "llama3.2",
  "prompt": "buy groceries"
}'
```

Response:
```json
{
  "embedding": [0.123, -0.456, 0.789, ...]  // 4096-dimensional vector
}
```

These embeddings power Principle 4: **Memory Creates Behavior**. Similar tasks have similar vectors, enabling semantic search.

## Model Selection Strategy

Different models have different strengths:

### For Structured Output (Parsing, Extraction)
```
llama3.2      - Excellent instruction following
mistral       - Good balance of speed and quality
phi3:mini     - Fastest, acceptable quality
```

### For Reasoning (Prioritization, Categorization)
```
llama3.1:8b   - Strong reasoning
llama3.2:8b   - Better context window
mistral       - Good general reasoning
```

### For Embeddings
```
nomic-embed-text  - Purpose-built for embeddings
llama3.2          - Acceptable quality
```

For this tutorial, we'll use `llama3.2` for generation and embedding to keep things simple.

## Ollama Server Management

### Starting the Server
```bash
ollama serve
```

### Running in Background (Linux)
```bash
# Create systemd service
sudo systemctl enable ollama
sudo systemctl start ollama
```

### Environment Variables
```bash
# Change host/port
OLLAMA_HOST=127.0.0.1:11434

# Set model directory
OLLAMA_MODELS=/path/to/models

# GPU settings (NVIDIA)
CUDA_VISIBLE_DEVICES=0

# Memory limits
OLLAMA_NUM_PARALLEL=2
OLLAMA_MAX_LOADED_MODELS=1
```

## Connection Patterns

### Blocking Request
```python
import httpx

response = httpx.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2",
        "prompt": "Hello",
        "stream": False
    },
    timeout=60.0
)
result = response.json()
```

### Streaming Request
```python
import httpx

with httpx.stream(
    "POST",
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2",
        "prompt": "Hello",
        "stream": True
    },
    timeout=60.0
) as response:
    for line in response.iter_lines():
        chunk = json.loads(line)
        print(chunk["response"], end="", flush=True)
```

### Async Request
```python
import httpx

async def generate(prompt: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            },
            timeout=60.0
        )
        return response.json()["response"]
```

## Error Handling

Ollama can fail in several ways:

```python
import httpx

def generate_safe(prompt: str) -> str | None:
    try:
        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False},
            timeout=60.0
        )
        response.raise_for_status()
        return response.json()["response"]
    
    except httpx.ConnectError:
        print("Error: Ollama server not running")
        return None
    
    except httpx.TimeoutException:
        print("Error: Request timed out")
        return None
    
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            print("Error: Model not found")
        else:
            print(f"Error: HTTP {e.response.status_code}")
        return None
```

## Model Loading Behavior

When you first call a model, Ollama loads it into memory. This takes time:

```
First request:  Load model (5-30s) + Inference (1-5s)
Second request: Inference only (1-5s)
```

Models stay loaded until:
- Another model is requested (memory pressure)
- The server restarts
- A timeout expires (default: 5 minutes of inactivity)

### Warming Up Models

For production, pre-load models:

```python
async def warmup_model(model: str = "llama3.2"):
    """Load model into memory without generating output."""
    async with httpx.AsyncClient() as client:
        await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": "warmup",
                "options": {"num_predict": 1}
            },
            timeout=120.0
        )
```

## Practical Tips

### 1. Always Set Timeouts
LLM inference can be slow. Set appropriate timeouts:
```python
timeout=60.0  # Minimum for generation
timeout=120.0 # For complex prompts or first load
```

### 2. Use JSON Format for Structured Output
```python
json={"format": "json", ...}
```

### 3. Limit Token Generation
```python
"options": {"num_predict": 256}  # Don't let it ramble
```

### 4. Use Stop Sequences
```python
"options": {"stop": ["\n\n", "```"]}  # Stop at natural boundaries
```

### 5. System Prompts for Consistency
```python
"system": "You are a task parser. Output valid JSON only."
```

## Summary

Ollama provides:
- Local LLM inference (privacy-first)
- Simple REST API
- Temperature control for entropy management
- JSON output mode for structured responses
- Embeddings for semantic memory

In the next chapter, we'll set up our Python project structure following clean architecture principles.

---

**Previous**: [Chapter 2: Environment Setup](./chapter-02-environment-setup.md)  
**Next**: [Chapter 4: Python Project Structure](./chapter-04-project-structure.md)
