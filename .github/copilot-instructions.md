# Neural Recall v3 - AI Coding Agent Instructions

## Architecture Overview

**Neural Recall v3** implements a **Dual Pipeline Architecture** for intelligent memory management in OpenWebUI:

- **Retrieval Pipeline** (Synchronous): Runs in `inlet()` to inject relevant memories into conversation context via vector search + LLM re-ranking
- **Consolidation Pipeline** (Asynchronous): Runs in `outlet()` as background task to optimize memory database through merging, splitting, and cleaning operations

## Core Components

### 1. Filter Class (`neural_recall_v3.py`)
The main orchestrator implementing OpenWebUI's filter interface:
```python
class Filter:
    async def inlet(self, body, __event_emitter__, __user__):  # Retrieval Pipeline
    async def outlet(self, body, __event_emitter__, __user__): # Consolidation Pipeline
```

### 2. Dual Pipeline Flow
```
User Message → inlet() → Vector Search → LLM Re-ranking → Context Injection → AI Response
                ↓
           outlet() → Background Consolidation → Memory Optimization
```

### 3. Configuration System
- **Config class**: Centralized constants (timeouts, thresholds, cache sizes)
- **Valves**: Runtime configuration (API endpoints, model names, semantic thresholds)
- **Token-optimized LLM prompts**: `MEMORY_RERANKING_PROMPT` (~1146 tokens), `MEMORY_CONSOLIDATION_PROMPT` (~1366 tokens)

## Key Patterns & Conventions

### Memory Operations
Use `MemoryOperation` Pydantic models with strict validation:
```python
{"operation": "CREATE|UPDATE|DELETE", "id": "mem-123", "content": "User..."}
```
All memory content must start with "User" or "User's" and be ≤500 chars.

### Intelligent Skip Logic
Messages are filtered through `_should_skip_memory_operations()` which detects and skips:
- Code blocks, SQL, stack traces, structured data, URL dumps
- Messages too long (>3000 chars) or too short (<10 chars)
- Symbol-heavy content (>50% non-alphabetic)

### Error Handling
Custom exception hierarchy:
```python
NeuralRecallError → ModelLoadError, EmbeddingError, MemoryOperationError, ValidationError
```

### Asynchronous Patterns
- Database operations wrapped in `asyncio.to_thread()` with timeouts
- Background consolidation tasks with proper exception handling
- LRU caching with async locks for thread safety

## Development Workflows

### Testing
Run comprehensive test suite:
```bash
python neural_recall_v3_test.py
```
Tests cover: skip logic, memory operations, LLM prompts, caching, edge cases, configuration validation.

### Token Optimization
Monitor prompt token usage:
```bash
python3 -c "
import re
with open('neural_recall_v3.py', 'r') as f: content = f.read()
# Calculate tokens for both prompts (target: reranking 900-1200, consolidation 1200-1500)
"
```

### Configuration Management
Key settings in `Config` class:
- `SEMANTIC_THRESHOLD = 0.50` (similarity cutoff)
- `MAX_MEMORIES_RETURNED = 15` (context injection limit)
- `CONSOLIDATION_CANDIDATE_SIZE = 50` (optimization scope)
- `RETRIEVAL_TIMEOUT = 5.0` vs `CONSOLIDATION_TIMEOUT = 30.0`

### Integration Points
- **OpenWebUI Dependencies**: `open_webui.models.users.Users`, `open_webui.routers.memories.Memories`
- **External Models**: Sentence transformers for embeddings, OpenAI-compatible API for LLM operations
- **Database**: Async operations through OpenWebUI's memory router

## Critical Implementation Details

### Embedding Generation
Batch processing with user-specific LRU caching:
```python
await self._generate_embeddings_batch(texts, user_id)  # Efficient batch + cache
```

### Memory Consolidation Rules
LLM enforces strict consolidation criteria:
- **MERGE**: Fragmented facts about same entity
- **SPLIT**: Topic contamination (work + personal mixed)
- **DELETE**: Temporal conflicts, duplicates
- **NO-OP**: Already optimally organized

### Status Emission
Real-time progress updates via `__event_emitter__`:
```python
await self._emit_status(emitter, "🎯 Selected 5 memories", done=True)
```

## Common Pitfalls

1. **Token Limits**: LLM prompts are heavily optimized - avoid expanding without recalculating tokens
2. **Async Context**: Always use `asyncio.to_thread()` for database operations to avoid blocking
3. **Memory Format**: Content must start with "User" for semantic consistency
4. **Cache Invalidation**: Call `_invalidate_user_cache()` after memory modifications
5. **Exception Handling**: Use custom exceptions for proper error categorization

## File References
- `neural_recall_v3.py`: Main implementation (1361 lines)
- `neural_recall_v3_test.py.py`: Comprehensive test suite (1366 lines)
- Core classes: `Filter`, `Config`, `MemoryOperation`, `LRUCache`
