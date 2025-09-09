"""
title: Memory System
version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import re
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError, model_validator
from sentence_transformers import SentenceTransformer

from open_webui.models.users import Users
from open_webui.routers.memories import Memories

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("MemorySystem")
logger.setLevel(logging.INFO)


class MemoryOperationType(Enum):
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


class MemorySkipReason(Enum):
    SKIP_EMPTY = "SKIP_EMPTY"
    SKIP_TOO_LONG = "SKIP_TOO_LONG"
    SKIP_CODE = "SKIP_CODE"
    SKIP_STRUCTURED = "SKIP_STRUCTURED"
    SKIP_SYMBOLS = "SKIP_SYMBOLS"
    SKIP_LOGS = "SKIP_LOGS"
    SKIP_STACKTRACE = "SKIP_STACKTRACE"
    SKIP_URL_DUMP = "SKIP_URL_DUMP"


class AsyncOperationManager:
    @staticmethod
    async def execute_with_timeout(operation, timeout: float, error_context: str = "Operation"):
        try:
            return await asyncio.wait_for(operation, timeout=timeout)
        except asyncio.TimeoutError:
            raise MemorySystemError(f"{error_context} timed out after {timeout}s")
        except Exception as e:
            raise MemorySystemError(f"{error_context} failed: {str(e)}")

    @staticmethod
    async def execute_db_operation(operation, timeout: float = None, context: str = "Database operation"):
        if timeout is None:
            timeout = ConfigManager.SystemConfig.TIMEOUT_DATABASE_OPERATION
        return await AsyncOperationManager.execute_with_timeout(asyncio.to_thread(operation), timeout, context)


class MemorySystemError(Exception):
    """Exception for Memory System operations with optional error context."""
    
    def __init__(self, message: str, context: str = None):
        self.context = context
        super().__init__(f"{context}: {message}" if context else message)


class ConfigManager:
    class SkipThresholds:
        MAX_MESSAGE_LENGTH = 3000
        MIN_QUERY_LENGTH = 10
        JSON_KEY_VALUE_THRESHOLD = 5
        STRUCTURED_LINE_COUNT_MIN = 3
        STRUCTURED_PERCENTAGE_THRESHOLD = 0.6
        STRUCTURED_BULLET_MIN = 4
        STRUCTURED_PIPE_MIN = 2
        SYMBOL_CHECK_MIN_LENGTH = 10
        SYMBOL_RATIO_THRESHOLD = 0.5
        URL_COUNT_THRESHOLD = 3
        LOGS_LINE_COUNT_MIN = 2
        LOGS_MIN_MATCHES = 1
        LOGS_MATCH_PERCENTAGE = 0.30

    class SystemConfig:
        CACHE_MAX_SIZE = 5000
        MAX_USER_CACHES = 500
        MAX_MEMORY_CONTENT_LENGTH = 500
        TIMEOUT_SESSION_REQUEST = 30
        TIMEOUT_DATABASE_OPERATION = 10
        TIMEOUT_USER_LOOKUP = 5
        DEFAULT_MAX_MEMORIES_RETURNED = 15
        MIN_DYNAMIC_THRESHOLD = 0.45
        MAX_DYNAMIC_THRESHOLD = 0.90
        PERCENTILE_THRESHOLD = 90
        CONSOLIDATION_RELAXED_MULTIPLIER = 0.90
        MIN_BATCH_SIZE = 8
        MAX_BATCH_SIZE = 32
        RETRIEVAL_MULTIPLIER = 2.0
        RETRIEVAL_TIMEOUT = 10.0
        CONSOLIDATION_TIMEOUT = 30.0
        STATUS_EMIT_TIMEOUT = 10.0

        STATUS_MESSAGES = {
            MemorySkipReason.SKIP_EMPTY: "✉️ Empty or Short Content Detected, Skipping Memory Operations",
            MemorySkipReason.SKIP_TOO_LONG: "📄 Oversized Content Detected, Skipping Memory Operations",
            MemorySkipReason.SKIP_CODE: "💻 Code Content Detected, Skipping Memory Operations",
            MemorySkipReason.SKIP_STRUCTURED: "📊 Structured Data Detected, Skipping Memory Operations",
            MemorySkipReason.SKIP_SYMBOLS: "🔢 Symbol Heavy Content Detected, Skipping Memory Operations",
            MemorySkipReason.SKIP_LOGS: "📝 Log Content Detected, Skipping Memory Operations",
            MemorySkipReason.SKIP_STACKTRACE: "⚠️ Stack Trace Detected, Skipping Memory Operations",
            MemorySkipReason.SKIP_URL_DUMP: "🔗 URL List Detected, Skipping Memory Operations",
        }


class ContentPatternDetector:
    @staticmethod
    def _check_patterns(message: str, patterns: List[str]) -> bool:
        return any(re.search(pattern, message, re.MULTILINE | re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def _count_line_matches(lines: List[str], pattern: str) -> int:
        return sum(1 for line in lines if re.match(pattern, line))

    @staticmethod
    def is_code_content(message: str) -> bool:
        patterns = [
            r"```[\s\S]*?```",
            r"`[^`\n]+`",
            r"^\s*(def|class|var|let|const)\s+[\w_][\w\d_]*\s*[=\(\{;:]",
            r"^\s*(import|from)\s+[\w\.\-_]+",
            r"^\s*function\s+\w+\s*\(",
            r"if\s*\(.*\)\s*\{",
            r"for\s*\(.*\)\s*\{",
            r"while\s*\(.*\)\s*\{",
            r"#include\s*<",
            r"using\s+namespace",
            r"SELECT\s+.+\s+FROM\s+\w+",
            r"INSERT\s+INTO\s+\w+",
            r"UPDATE\s+\w+\s+SET\s+\w+\s*=",
            r"DELETE\s+FROM\s+\w+",
            r"^\s*[A-Za-z_]\w*\s*[=:]\s*[A-Za-z_]\w*\s*\(",
            r"\w+\s*=\s*\([^)]*\)\s*=>\s*\{",
            r"^\s*\{\s*\}\s*$",
            r"<[^>]+>[\s\S]*</[^>]+>",
        ]
        return ContentPatternDetector._check_patterns(message, patterns)

    @staticmethod
    def is_structured_data(message: str) -> bool:
        json_patterns = [
            r"^\s*[\{\[].*[\}\]]\s*$",
            r'"[^"]*"\s*:\s*["\{\[\d]',
            r'["\']?\w+["\']?\s*:\s*["\{\[\d].*["\'\}\]\d]',
            r'\{\s*["\']?\w+["\']?\s*:\s*\{',
            r"\[\s*\{.*\}\s*\]",
        ]

        if ContentPatternDetector._check_patterns(message, json_patterns):
            return True

        if len(re.findall(r'["\']?\w+["\']?\s*:\s*["\{\[\d\w]', message)) >= ConfigManager.SkipThresholds.JSON_KEY_VALUE_THRESHOLD:
            return True

        lines = message.split("\n")
        line_count = len(lines)

        if line_count < ConfigManager.SkipThresholds.STRUCTURED_LINE_COUNT_MIN:
            return False

        threshold = max(ConfigManager.SkipThresholds.STRUCTURED_LINE_COUNT_MIN, line_count * ConfigManager.SkipThresholds.STRUCTURED_PERCENTAGE_THRESHOLD)

        checks = [
            ContentPatternDetector._count_line_matches(lines, r"^\s*[A-Za-z0-9_]+:\s+\S+") >= threshold,
            ContentPatternDetector._count_line_matches(lines, r"^\s*[-\*\+]\s+.+") >= max(ConfigManager.SkipThresholds.STRUCTURED_BULLET_MIN, threshold),
            ContentPatternDetector._count_line_matches(lines, r"^\s*\d+\.\s") >= threshold,
            ContentPatternDetector._count_line_matches(lines, r"^\s*[a-zA-Z]\)\s") >= threshold,
            sum(1 for line in lines if "|" in line and line.count("|") >= ConfigManager.SkipThresholds.STRUCTURED_PIPE_MIN)
            >= max(ConfigManager.SkipThresholds.STRUCTURED_PIPE_MIN, threshold),
        ]

        return any(checks)

    @staticmethod
    def is_mostly_symbols(message: str) -> bool:
        if len(message) <= ConfigManager.SkipThresholds.SYMBOL_CHECK_MIN_LENGTH:
            return False
        alpha_space_count = sum(1 for c in message if c.isalpha() or c.isspace())
        return alpha_space_count / len(message) < ConfigManager.SkipThresholds.SYMBOL_RATIO_THRESHOLD

    @staticmethod
    def _get_error_patterns() -> List[str]:
        return [
            r"(NameError|TypeError|ValueError|AttributeError|KeyError|SyntaxError|IndentationError).*:",
            r"Exception\s+in\s+thread",
            r'^\s*(at|File)\s+["\']?[^"\']*["\']?,?\s*line\s+\d+',
            r"^\s*at\s+\w+.*\([^)]*:\d+:\d+\)",
        ]

    @staticmethod
    def is_log_content(message: str) -> bool:
        lines = message.split("\n")
        line_count = len(lines)

        error_patterns = ContentPatternDetector._get_error_patterns()
        log_patterns = [r"\d{2,4}[-/]\d{1,2}[-/]\d{1,2}[\s\d:]+\b(ERROR|WARN|INFO|DEBUG)\b", r"\bERROR\b.*:.*"]

        if ContentPatternDetector._check_patterns(message, error_patterns + log_patterns):
            return True

        if line_count < ConfigManager.SkipThresholds.LOGS_LINE_COUNT_MIN:
            return False

        datetime_matches = sum(1 for line in lines if re.search(r"\d{2,4}[-/]\d{1,2}[-/]\d{1,2}.*\d{1,2}:\d{2}", line))
        return datetime_matches >= max(ConfigManager.SkipThresholds.LOGS_MIN_MATCHES, line_count * ConfigManager.SkipThresholds.LOGS_MATCH_PERCENTAGE)

    @staticmethod
    def is_stack_trace(message: str) -> bool:
        lines = message.split("\n")
        error_patterns = ContentPatternDetector._get_error_patterns()
        stack_patterns = [r"^\s*(Traceback|Exception|Error).*:\d+", r"Traceback\s*\(most recent call", r'^\s*File\s*"[^"]+",\s*line\s*\d+']
        return any(any(re.search(pattern, line) for pattern in error_patterns + stack_patterns) for line in lines)

    @staticmethod
    def is_url_dump(message: str) -> bool:
        return len(re.findall(r"https?://[^\s]+", message)) >= ConfigManager.SkipThresholds.URL_COUNT_THRESHOLD


class MemoryTransformer:
    @staticmethod
    def to_dict(memory, relevance: float = None) -> Dict[str, Any]:
        memory_dict = {
            "id": str(memory.id),
            "content": memory.content,
        }
        
        if relevance is not None:
            memory_dict["relevance"] = relevance
            
        if hasattr(memory, "created_at") and memory.created_at:
            memory_dict["created_at"] = datetime.fromtimestamp(memory.created_at, tz=timezone.utc).isoformat()
        if hasattr(memory, "updated_at") and memory.updated_at:
            memory_dict["updated_at"] = datetime.fromtimestamp(memory.updated_at, tz=timezone.utc).isoformat()
            
        return memory_dict

    @staticmethod
    def to_context_format(memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return ""
            
        memory_count = len(memories)
        fact_word = "fact" if memory_count == 1 else f"{memory_count} facts"
        memory_header = f"BACKGROUND: You naturally know {'this' if memory_count == 1 else 'these'} {fact_word}. Never mention {'its' if memory_count == 1 else 'their'} source."

        formatted_memories = []
        for memory in memories:
            clean_content = " ".join(memory["content"].replace("\n", " ").replace("\r", " ").split())
            formatted_memories.append(f"- {clean_content}")
            
        return f"{memory_header}\n{chr(10).join(formatted_memories)}"


class SimilarityService:
    def __init__(self, embedding_generator, cache_service):
        self.embedding_generator = embedding_generator
        self.cache_service = cache_service

    async def compute_similarities(self, user_message: str, user_id: str, user_memories: List, limit_multiplier: float = 1.0) -> Tuple[List[Dict], float, List[Dict]]:
        if not user_memories:
            return [], ConfigManager.SystemConfig.MIN_DYNAMIC_THRESHOLD, []

        query_embedding = await self.embedding_generator._generate_embedding(user_message, user_id)
        memory_contents = [memory.content for memory in user_memories]
        memory_embeddings = await self.embedding_generator._generate_embeddings_batch(memory_contents, user_id)

        if len(memory_embeddings) != len(user_memories):
            raise MemorySystemError(f"Embedding count mismatch: {len(memory_embeddings)} vs {len(user_memories)}", "Similarity computation")

        similarity_scores = []
        memory_data = []

        for memory_index, memory in enumerate(user_memories):
            memory_embedding = memory_embeddings[memory_index]
            if memory_embedding is None:
                continue

            similarity = float(np.dot(query_embedding, memory_embedding))
            similarity_scores.append(similarity)
            memory_data.append(MemoryTransformer.to_dict(memory, similarity))

        if not similarity_scores:
            return [], ConfigManager.SystemConfig.MIN_DYNAMIC_THRESHOLD, []

        memory_data.sort(key=lambda x: x["relevance"], reverse=True)
        
        is_consolidation = limit_multiplier > 1.0
        dynamic_threshold = self._calculate_dynamic_threshold(similarity_scores, is_consolidation)

        filtered_memories = [m for m in memory_data if m["relevance"] >= dynamic_threshold]
        limit = int(ConfigManager.SystemConfig.DEFAULT_MAX_MEMORIES_RETURNED * limit_multiplier)
        filtered_memories = filtered_memories[:limit]

        return filtered_memories, dynamic_threshold, memory_data

    def _calculate_dynamic_threshold(self, similarity_scores: List[float], for_consolidation: bool = False) -> float:
        if not similarity_scores:
            return ConfigManager.SystemConfig.MIN_DYNAMIC_THRESHOLD

        percentile_threshold = float(np.percentile(similarity_scores, ConfigManager.SystemConfig.PERCENTILE_THRESHOLD))
        base_threshold = max(
            ConfigManager.SystemConfig.MIN_DYNAMIC_THRESHOLD, 
            min(ConfigManager.SystemConfig.MAX_DYNAMIC_THRESHOLD, percentile_threshold)
        )

        if for_consolidation:
            return base_threshold * ConfigManager.SystemConfig.CONSOLIDATION_RELAXED_MULTIPLIER
        return base_threshold


class CacheService:
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager

    async def get_or_generate_embedding(self, text: str, user_id: str, generator_func) -> Optional[np.ndarray]:
        if not text or len(text.strip()) < ConfigManager.SkipThresholds.MIN_QUERY_LENGTH:
            return None
            
        cache = await self.cache_manager._create_user_cache(user_id)
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        cached_embedding = await cache.get(text_hash)
        
        if cached_embedding is not None:
            return cached_embedding
            
        generated_embedding = await generator_func(text)
        if generated_embedding is not None:
            await cache.put(text_hash, generated_embedding)
            
        return generated_embedding

    async def batch_get_or_generate(self, texts: List[str], user_id: str, batch_generator_func) -> List[Optional[np.ndarray]]:
        cache = await self.cache_manager._create_user_cache(user_id)
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []

        for text_index, text in enumerate(texts):
            if not text or len(text.strip()) < ConfigManager.SkipThresholds.MIN_QUERY_LENGTH:
                continue

            text_hash = hashlib.sha256(text.encode()).hexdigest()
            cached_embedding = await cache.get(text_hash)

            if cached_embedding is not None:
                cached_embeddings[text_index] = cached_embedding
            else:
                uncached_texts.append(text)
                uncached_indices.append(text_index)

        new_embeddings = {}
        if uncached_texts:
            batch_embeddings = await batch_generator_func(uncached_texts)
            for batch_index, embedding in enumerate(batch_embeddings):
                if embedding is not None:
                    text_idx = uncached_indices[batch_index]
                    text_hash = hashlib.sha256(uncached_texts[batch_index].encode()).hexdigest()
                    await cache.put(text_hash, embedding)
                    new_embeddings[text_idx] = embedding

        result_embeddings = []
        for text_index in range(len(texts)):
            if text_index in cached_embeddings:
                result_embeddings.append(cached_embeddings[text_index])
            elif text_index in new_embeddings:
                result_embeddings.append(new_embeddings[text_index])
            else:
                result_embeddings.append(None)

        return result_embeddings


class ErrorHandler:
    @staticmethod
    def handle_operation_error(operation_data: Dict, operation_index: int, error: Exception) -> str:
        operation_type = operation_data.get("operation", "UNKNOWN")
        content_preview = ""
        
        if "content" in operation_data:
            content = operation_data.get("content", "")
            content_preview = f" - Content: {content[:60]}..." if len(content) > 60 else f" - Content: {content}"
        elif "id" in operation_data:
            content_preview = f" - ID: {operation_data['id']}"
            
        error_message = f"❌ Failed {operation_type} operation {operation_index+1}{content_preview}: {str(error)}"
        logger.error(error_message)
        return error_message


MEMORY_CONSOLIDATION_PROMPT = """You are the Memory System Consolidator, a meticulous component of the Consolidation Pipeline. Your primary function is to enrich the user's memory profile with high-value, fact-based memories derived from conversation. Your secondary function is to perform database hygiene (merging, updating, splitting) with bias towards **information preservation**.

# PRIMARY GOAL
Your objective is to build a high-fidelity, long-term user profile. The memories you create are the foundation for all future AI interactions. Quality, accuracy, and relevance are paramount. A well-maintained memory store enables the AI to be a truly helpful and context-aware assistant.

# CORE PRINCIPLES
1. **Quality First** — Act only when you have high confidence that the operation improves accuracy, completeness, or organization. Your default action is NO-OP (`[]`) when uncertain.
2. **Verifiable Factuality** — You must only record what is **explicitly stated** by the user. Do not infer, assume, or interpret facts not present in the text.
3. **Single Distinct Concept** — Each memory contains ONE distinct concept or fact category - never mix unrelated information types.
4. **Temporal Precedence** — New, dated information always supersedes older, conflicting information. For non-conflicting memories, recent memories take priority when they provide more specific or actionable context.
5. **Contextual Grounding** — Use the provided `Current Date/Time` to convert relative time references. For ambiguous cases: "yesterday" = current date minus 1 day, "next week" = start of following week (Monday), "soon" = within 30 days.

# EXECUTION RULES
1. **Language Mandate** — All memory `content` MUST be in **English**. Translate core personal facts if conversation is in another language.
2. **First-Person Format** — Use "I", "My", "Me" with specific names and details when available (e.g., "My wife Jane", "My daughter Sarah").
3. **Cross-Reference Enrichment** — During UPDATE operations, actively enhance memories by cross-referencing related memories to add specific names/details. Only add names when:
   - Memories explicitly refer to same role/relationship
   - Another memory directly links a name to that role
4. **Date Integration** — Include dates using these simple formats:
   - **Full date:** "September 1 2025" (Month Day Year)  
   - **Month/Year:** "September 2025" (when specific day unknown)
   - **Year only:** "2025" (when only year is relevant)
   - **Wrong formats:** Avoid "Sep 1 2025", "9/1/2025", "01-Sep-2025"
5. **Value Filter** — You **MUST IGNORE** and **NEVER** create memories from: User curiosity about general topics, questions seeking information/advice, conversational filler, transient states, speculation, or hypotheticals. **ONLY CREATE** memories for concrete facts about the user's life, experiences, or preferences.
6. **Concrete Facts Only** — Only create memories for definitive, concrete facts - avoid speculation or uncertain statements.

# CONSOLIDATION OPERATIONS
Analyze the user's message and candidate memories to determine if any of the following operations are justified.

## CREATE (New, Atomic Fact)
- **Justification** — The conversation reveals a new, high-value, personal fact that passes the Value Filter and is NOT already captured in existing memories.
- **Single Distinct Concept** — Create separate memories for distinct concepts (schedule changes vs. education milestones).
- **Example:**
  - **Conversation:** "My daughter is starting kindergarten next month, so I'm adjusting my work schedule to be free in the afternoons."
  - **Current Date/Time:** "August 15 2025"
  - **Existing Memories:** `["mem-101: I have a daughter"]`
  - **Output:** `[{{"operation":"CREATE","content":"My daughter is starting kindergarten in September 2025"}}, {{"operation":"CREATE","content":"I am adjusting my work schedule to be free in afternoons starting September 2025"}}]`

## UPDATE (Temporal Progression OR Cross-Reference Enrichment)
- **Justification** — New information supersedes existing memory OR you can add specific names/details from cross-referencing related memories.
- **Example (Cross-Reference):**
  - **Conversation:** "My wife picked up groceries today."
  - **Existing Memories:** `["mem-301: My wife works at the hospital", "mem-302: Jane works as a nurse"]`
  - **Output:** `[{{"operation":"UPDATE","id":"mem-301","content":"My wife Jane works at the hospital as a nurse"}}]`

## DELETE (User Retraction)
- **Justification** — User explicitly states previous information should be forgotten or was completely incorrect.
- **Example:**
  - **Conversation:** "Please forget what I said about changing jobs - I'm staying at my current company."
  - **Existing Memories:** `["mem-201: I am actively job searching"]`
  - **Output:** `[{{"operation":"DELETE","id":"mem-201"}}]`

## MERGE (Consolidate Fragmented Information)
- **Justification** — Multiple memories contain fragmented pieces about the SAME entity that should be unified.
- **Decision Rule** — Only MERGE when:
  - Memories refer to same entity
  - Combined content stays under {max_memory_length} characters
  - Result maintains Single Distinct Concept
- **Example:**
  - **Existing Memories:** `["mem-401: I am working on Project Phoenix", "mem-402: Project Phoenix is a data migration initiative"]`
  - **Output:** `[{{"operation":"UPDATE","id":"mem-401","content":"I am working on Project Phoenix, a data migration initiative"}}, {{"operation":"DELETE","id":"mem-402"}}]`

## SPLIT (Enforce Single Distinct Concept)
- **Example:** `["mem-501: I am a vegetarian and my favorite movie is Blade Runner"]` → `[{{"operation":"UPDATE","id":"mem-501","content":"I am a vegetarian"}}, {{"operation":"CREATE","content":"My favorite movie is Blade Runner"}}]`

# CHARACTER LIMIT
Every memory content MUST be under {max_memory_length} characters.

**ATOMIC MEMORY STRATEGY:**
- **Single Distinct Concept** — Each memory contains ONE retrievable fact - never mix unrelated information
- **Dense Information** — Include specific names, dates, and details when available
- **Default to Creation** — When memories are long, create separate new memories rather than updating existing ones

# OUTPUT CONTRACT
- **RETURN ONLY A JSON OBJECT** with the following structure:
```json
{{
  "operations": [
    {{"operation": "CREATE", "content": "memory content"}},
    {{"operation": "UPDATE", "id": "mem-123", "content": "updated content"}},
    {{"operation": "DELETE", "id": "mem-456"}}
  ]
}}
```
- The `operations` array contains memory operations to execute.
- If no operations are needed, return `{{"operations": []}}`.
- **ABSOLUTELY NO COMMENTARY, EXPLANATIONS, OR CODE FENCES.**
"""


class MemoryOperation(BaseModel):
    """Pydantic model for memory operations with validation."""

    operation: MemoryOperationType = Field(description="Type of memory operation to perform")
    content: Optional[str] = Field(None, description="Memory content for CREATE/UPDATE operations")
    id: Optional[str] = Field(None, description="Memory ID for UPDATE/DELETE operations")

    @model_validator(mode="after")
    def validate_operation_requirements(self):
        """Validate operation-specific requirements."""
        if self.operation == MemoryOperationType.CREATE:
            if not self.content or not self.content.strip():
                raise ValueError("CREATE operation requires non-empty content")
            if len(self.content.strip()) > ConfigManager.SystemConfig.MAX_MEMORY_CONTENT_LENGTH:
                raise ValueError(f"Content too long (max {ConfigManager.SystemConfig.MAX_MEMORY_CONTENT_LENGTH} characters)")

        elif self.operation == MemoryOperationType.UPDATE:
            if not self.id:
                raise ValueError("UPDATE operation requires memory ID")
            if not self.content or not self.content.strip():
                raise ValueError("UPDATE operation requires non-empty content")
            if len(self.content.strip()) > ConfigManager.SystemConfig.MAX_MEMORY_CONTENT_LENGTH:
                raise ValueError(f"Content too long (max {ConfigManager.SystemConfig.MAX_MEMORY_CONTENT_LENGTH} characters)")

        elif self.operation == MemoryOperationType.DELETE:
            if not self.id:
                raise ValueError("DELETE operation requires memory ID")

        return self

    def validate_operation(self, existing_memory_ids: set = None) -> bool:
        """Validate the memory operation against existing memory IDs."""
        if existing_memory_ids is None:
            existing_memory_ids = set()

        if self.operation == MemoryOperationType.CREATE:
            return True
        elif self.operation in [MemoryOperationType.UPDATE, MemoryOperationType.DELETE]:
            return self.id in existing_memory_ids
        return False


class ConsolidationResponse(BaseModel):
    """Pydantic model for memory consolidation LLM response."""

    operations: List[MemoryOperation] = Field(default_factory=list, description="List of memory operations to execute")


class LRUCache:
    """Enhanced thread-safe LRU cache implementation using OrderedDict with comprehensive metrics."""

    def __init__(self, max_size: int) -> None:
        """Initialize LRU cache with specified maximum size."""
        self.max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.RLock()

    async def get(self, key: str) -> Optional[Any]:
        """Get an item from cache, moving it to end (most recently used)."""
        def _get():
            with self._lock:
                if key in self._cache:
                    value = self._cache.pop(key)
                    self._cache[key] = value
                    return value
                return None
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _get)

    async def put(self, key: str, value: Any) -> None:
        """Put an item in cache, evicting least recently used if necessary."""
        def _put():
            with self._lock:
                if key in self._cache:
                    self._cache.pop(key)
                elif len(self._cache) >= self.max_size:
                    self._cache.popitem(last=False)
                self._cache[key] = value
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _put)

    async def clear(self) -> None:
        """Clear cache and return number of entries cleared."""
        def _clear():
            with self._lock:
                self._cache.clear()
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _clear)


class Filter:
    """Enhanced multi-model embedding and memory filter with LRU caching."""

    class Valves(BaseModel):
        """Configuration valves for the Memory System."""

        api_url: str = Field(
            default="http://host.docker.internal:11434/v1",
            description="OpenAI-compatible API URL for LLM operations",
        )
        api_key: str = Field(
            default="sk-memory-system-key",
            description="API key for LLM service authentication",
        )
        model: str = Field(
            default="llama3.2:3b",
            description="Model name for LLM operations",
        )

        embedding_model: str = Field(
            default="Alibaba-NLP/gte-multilingual-base",
            description="Sentence transformer model for embeddings",
        )

        max_memories_returned: int = Field(
            default=ConfigManager.SystemConfig.DEFAULT_MAX_MEMORIES_RETURNED,
            description="Maximum number of memories to return in context",
        )

        min_dynamic_threshold: float = Field(default=ConfigManager.SystemConfig.MIN_DYNAMIC_THRESHOLD, description="Minimum allowed dynamic threshold value")
        max_dynamic_threshold: float = Field(default=ConfigManager.SystemConfig.MAX_DYNAMIC_THRESHOLD, description="Maximum allowed dynamic threshold value")
        percentile_threshold: int = Field(default=ConfigManager.SystemConfig.PERCENTILE_THRESHOLD, description="Percentile to use for dynamic threshold calculation")

    def __init__(self):
        """Initialize the Memory System filter with production validation."""
        self.valves = self.Valves()
        self._validate_system_configuration()

        self._model = None
        self._model_load_lock = asyncio.Lock()
        self._embedding_cache: OrderedDict[str, LRUCache] = OrderedDict()
        self._cache_lock = threading.RLock()
        self._aiohttp_session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        self._background_tasks: set = set()
        self._shutdown_event = asyncio.Event()
        self._retrieval_cache: OrderedDict[str, Dict] = OrderedDict()
        
        self._cache_service = CacheService(self)
        self._similarity_service = SimilarityService(self, self._cache_service)
        self._error_handler = ErrorHandler()

    def _validate_system_configuration(self) -> None:
        """Validate configuration and fail if invalid."""
        if not self.valves.api_url or not self.valves.api_url.strip():
            raise MemorySystemError("API URL not configured", "Configuration validation")

        if not self.valves.api_key or not self.valves.api_key.strip():
            raise MemorySystemError("API key not configured", "Configuration validation")

        if not self.valves.model or not self.valves.model.strip():
            raise MemorySystemError("Model not specified", "Configuration validation")

        if not 1 <= self.valves.percentile_threshold <= 99:
            raise MemorySystemError(f"Invalid percentile threshold: {self.valves.percentile_threshold}", "Configuration validation")

        if not 0.0 <= self.valves.min_dynamic_threshold <= 1.0:
            raise MemorySystemError(f"Invalid min dynamic threshold: {self.valves.min_dynamic_threshold}", "Configuration validation")

        if not 0.0 <= self.valves.max_dynamic_threshold <= 1.0:
            raise MemorySystemError(f"Invalid max dynamic threshold: {self.valves.max_dynamic_threshold}", "Configuration validation")

        if self.valves.min_dynamic_threshold >= self.valves.max_dynamic_threshold:
            raise MemorySystemError(f"Min threshold ({self.valves.min_dynamic_threshold}) must be less than max threshold ({self.valves.max_dynamic_threshold})", "Configuration validation")

        if self.valves.max_memories_returned <= 0:
            raise MemorySystemError(f"Invalid max memories returned: {self.valves.max_memories_returned}", "Configuration validation")

        logger.info("✅ Configuration validated")

    async def _load_embedding_model(self):
        """Get or load the sentence transformer model with thread safety."""
        if self._model is None:
            async with self._model_load_lock:
                if self._model is None:
                    try:
                        logger.info(f"🤖 Loading embedding model: {self.valves.embedding_model}")

                        def load_model():
                            return SentenceTransformer(
                                self.valves.embedding_model,
                                device="cpu",
                                trust_remote_code=True,
                            )

                        loop = asyncio.get_event_loop()
                        self._model = await loop.run_in_executor(None, load_model)
                        logger.info("✅ Embedding model loaded successfully")

                    except Exception as e:
                        error_msg = f"Failed to load embedding model: {str(e)}"
                        logger.error(f"❌ {error_msg}")
                        raise MemorySystemError(error_msg, "Model loading")

        return self._model

    async def _create_user_cache(self, user_id: str) -> LRUCache:
        """Get or create user-specific embedding cache with global user limit."""
        def _create_cache():
            with self._cache_lock:
                if user_id in self._embedding_cache:
                    self._embedding_cache.move_to_end(user_id)
                    return self._embedding_cache[user_id]

                if len(self._embedding_cache) >= ConfigManager.SystemConfig.MAX_USER_CACHES:
                    self._embedding_cache.popitem(last=False)

                self._embedding_cache[user_id] = LRUCache(ConfigManager.SystemConfig.CACHE_MAX_SIZE)
                self._embedding_cache.move_to_end(user_id)
                return self._embedding_cache[user_id]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _create_cache)

    async def _generate_embedding(self, text: str, user_id: str) -> np.ndarray:
        """Generate embedding for a single text using the batch method."""
        embeddings = await self._generate_embeddings_batch([text], user_id)
        if embeddings and embeddings[0] is not None:
            return embeddings[0]
        raise MemorySystemError("Failed to generate embedding for the given text", "Embedding generation")

    async def _generate_embeddings_batch(self, texts: List[str], user_id: str) -> List[np.ndarray]:
        """Generate embeddings for multiple texts efficiently with batch processing and caching."""
        if not texts:
            return []

        cache = await self._create_user_cache(user_id)
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []

        for text_index, text in enumerate(texts):
            if not text or len(text.strip()) < ConfigManager.SkipThresholds.MIN_QUERY_LENGTH:
                continue

            text_hash = hashlib.sha256(text.encode()).hexdigest()
            cached_embedding = await cache.get(text_hash)

            if cached_embedding is not None:
                cached_embeddings[text_index] = cached_embedding
            else:
                uncached_texts.append(text)
                uncached_indices.append(text_index)

        new_embeddings = {}
        if uncached_texts:
            model = await self._load_embedding_model()
            batch_size = min(ConfigManager.SystemConfig.MAX_BATCH_SIZE, max(ConfigManager.SystemConfig.MIN_BATCH_SIZE, len(uncached_texts)))

            def generate_batch_embeddings(batch_texts):
                embeddings = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
                return [embedding / np.linalg.norm(embedding) if np.linalg.norm(embedding) > 0 else embedding for embedding in embeddings]

            loop = asyncio.get_event_loop()

            for batch_start in range(0, len(uncached_texts), batch_size):
                batch_texts = uncached_texts[batch_start : batch_start + batch_size]
                batch_indices = uncached_indices[batch_start : batch_start + batch_size]
                batch_embeddings = await loop.run_in_executor(None, generate_batch_embeddings, batch_texts)

                for batch_index, embedding in enumerate(batch_embeddings):
                    text_idx = batch_indices[batch_index]
                    text_hash = hashlib.sha256(batch_texts[batch_index].encode()).hexdigest()
                    await cache.put(text_hash, embedding)
                    new_embeddings[text_idx] = embedding

        result_embeddings = []
        for text_index in range(len(texts)):
            if text_index in cached_embeddings:
                result_embeddings.append(cached_embeddings[text_index])
            elif text_index in new_embeddings:
                result_embeddings.append(new_embeddings[text_index])
            else:
                result_embeddings.append(None)

        valid_count = sum(1 for emb in result_embeddings if emb is not None)
        logger.info(f"🚀 Batch embedding: {len(cached_embeddings)} cached, {len(new_embeddings)} new, {valid_count}/{len(texts)} valid")
        return result_embeddings

    async def _create_http_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with connection pooling and production settings."""
        if self._aiohttp_session is None or self._aiohttp_session.closed:
            async with self._session_lock:
                if self._aiohttp_session is None or self._aiohttp_session.closed:
                    timeout = aiohttp.ClientTimeout(total=ConfigManager.SystemConfig.TIMEOUT_SESSION_REQUEST)
                    connector = aiohttp.TCPConnector(
                        limit=100,
                        limit_per_host=30,
                        keepalive_timeout=300,
                        enable_cleanup_closed=True,
                        ttl_dns_cache=300,
                    )
                    self._aiohttp_session = aiohttp.ClientSession(
                        timeout=timeout,
                        connector=connector,
                        headers={"User-Agent": "MemorySystem/1.0.0"},
                    )
        return self._aiohttp_session

    def _should_skip_memory_operations(self, user_message: str) -> Tuple[bool, str]:
        """
        Enhanced gating for memory operations with comprehensive pattern detection.
        Skips: empty, too long, code, logs, structured data, URL dumps, or symbol spam.
        """
        if not user_message or not user_message.strip():
            return True, ConfigManager.SystemConfig.STATUS_MESSAGES[MemorySkipReason.SKIP_EMPTY]

        trimmed_message = user_message.strip()

        if len(trimmed_message) < ConfigManager.SkipThresholds.MIN_QUERY_LENGTH:
            return True, ConfigManager.SystemConfig.STATUS_MESSAGES[MemorySkipReason.SKIP_EMPTY]

        if len(trimmed_message) > ConfigManager.SkipThresholds.MAX_MESSAGE_LENGTH:
            return True, ConfigManager.SystemConfig.STATUS_MESSAGES[MemorySkipReason.SKIP_TOO_LONG]

        if ContentPatternDetector.is_code_content(trimmed_message):
            return True, ConfigManager.SystemConfig.STATUS_MESSAGES[MemorySkipReason.SKIP_CODE]

        if ContentPatternDetector.is_structured_data(trimmed_message):
            return True, ConfigManager.SystemConfig.STATUS_MESSAGES[MemorySkipReason.SKIP_STRUCTURED]

        if ContentPatternDetector.is_mostly_symbols(trimmed_message):
            return True, ConfigManager.SystemConfig.STATUS_MESSAGES[MemorySkipReason.SKIP_SYMBOLS]

        if ContentPatternDetector.is_log_content(trimmed_message):
            return True, ConfigManager.SystemConfig.STATUS_MESSAGES[MemorySkipReason.SKIP_LOGS]

        if ContentPatternDetector.is_stack_trace(trimmed_message):
            return True, ConfigManager.SystemConfig.STATUS_MESSAGES[MemorySkipReason.SKIP_STACKTRACE]

        if ContentPatternDetector.is_url_dump(trimmed_message):
            return True, ConfigManager.SystemConfig.STATUS_MESSAGES[MemorySkipReason.SKIP_URL_DUMP]

        return False, ""

    def _validate_and_extract_user_message(self, body: Dict[str, Any]) -> Tuple[Optional[str], bool, str]:
        """Extract user message and determine if memory operations should be skipped."""
        user_message = self._extract_user_message(body.get("messages", []))
        should_skip, skip_reason = (
            self._should_skip_memory_operations(user_message) if user_message else (True, ConfigManager.SystemConfig.STATUS_MESSAGES[MemorySkipReason.SKIP_EMPTY])
        )
        return user_message, should_skip, skip_reason

    async def _get_user_memories(self, user_id: str, timeout: float = None) -> List:
        """Get user memories with timeout handling."""
        if timeout is None:
            timeout = ConfigManager.SystemConfig.RETRIEVAL_TIMEOUT
        return await AsyncOperationManager.execute_with_timeout(asyncio.to_thread(Memories.get_memories_by_user_id, user_id), timeout, "Memory retrieval")

    def _extract_text_content(self, content: Union[str, List[Dict], Dict]) -> str:
        """Extract only text content from various message content formats, ignoring images/files."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return " ".join(text_parts)
        elif isinstance(content, dict) and "text" in content:
            return content["text"]
        return ""

    def _extract_user_message(self, messages: List[Dict]) -> Optional[str]:
        """Extract the last user message from conversation messages."""
        if not messages:
            return None

        for message in reversed(messages):
            if message.get("role") == "user":
                return self._extract_text_content(message.get("content", ""))
        return None

    async def _compute_memory_similarities(
        self, user_message: str, user_id: str, user_memories: List, limit_multiplier: float = 1.0
    ) -> Tuple[List[Dict], float, List[Dict]]:
        """Compute similarities between user message and memories, return filtered results, threshold, and all similarities."""
        return await self._similarity_service.compute_similarities(
            user_message, user_id, user_memories, limit_multiplier
        )

    def format_current_datetime(self) -> str:
        try:
            now = datetime.now(timezone.utc)
            return now.strftime("%A %B %d %Y at %H:%M:%S UTC")
        except Exception as e:
            raise MemorySystemError(f"Failed to format datetime: {e}")

    async def _emit_status(self, emitter: Optional[Callable], description: str, done: bool = True) -> None:
        """Emit status messages for memory operations with error handling."""
        if not emitter:
            return

        payload = {"type": "status", "data": {"description": description, "done": done}}

        try:
            result = emitter(payload)
            if asyncio.iscoroutine(result):
                await AsyncOperationManager.execute_with_timeout(result, ConfigManager.SystemConfig.STATUS_EMIT_TIMEOUT, "Status emit")
            logger.info(f"📌 Status emitted: {description}")
        except Exception as e:
            logger.warning(f"❌ Status emit failed for '{description[:50]}': {e}")

    async def _retrieve_relevant_memories(self, user_message: str, user_id: str, user_memories: List = None) -> Dict[str, Any]:
        """Retrieve memories for injection using simplified similarity computation."""
        if user_memories is None:
            user_memories = await AsyncOperationManager.execute_with_timeout(
                asyncio.to_thread(Memories.get_memories_by_user_id, user_id), ConfigManager.SystemConfig.RETRIEVAL_TIMEOUT, "Memory retrieval"
            )

        if not user_memories:
            logger.info("📭 No memories found for user")
            return {"memories": [], "dynamic_threshold": None}

        logger.info(f"🔍 Found {len(user_memories)} total memories for analysis")

        memories, threshold, all_similarities = await self._compute_memory_similarities(user_message, user_id, user_memories)

        logger.info(f"🎯 Selected {len(memories)} injection memories (threshold: {threshold:.3f})")

        if memories:
            logger.info("📊 Memories ranked by semantic similarity:")
            for rank, memory in enumerate(memories, 1):
                content_preview = memory["content"][:120] + "..." if len(memory["content"]) > 120 else memory["content"]
                logger.info(f"  {rank}. Score: {memory['relevance']:.4f} | Content: {content_preview}")

        return {"memories": memories, "dynamic_threshold": threshold, "all_similarities": all_similarities}

    async def _add_memory_context(
        self, body: Dict[str, Any], memories: List[Dict[str, Any]] = None, user_id: str = None, emitter: Optional[Callable] = None
    ) -> None:
        """Add memory context to request body with simplified logic."""
        if not body or "messages" not in body or not body["messages"]:
            logger.info("❌ Invalid body or no messages found")
            return

        content_parts = [f"Current Date/Time: {self.format_current_datetime()}"]

        if memories and user_id:
            memory_count = len(memories)
            fact_word = "fact" if memory_count == 1 else f"{memory_count} facts"
            memory_header = f"BACKGROUND: You naturally know {'this' if memory_count == 1 else 'these'} {fact_word}. Never mention {'its' if memory_count == 1 else 'their'} source."

            formatted_memories = []
            for memory in memories:
                clean_content = " ".join(memory["content"].replace("\n", " ").replace("\r", " ").split())
                formatted_memories.append(f"- {clean_content}")
            content_parts.append(f"{memory_header}\n{chr(10).join(formatted_memories)}")

            if emitter:
                description = f"🧠 Retrieved {memory_count} {'Memory' if memory_count == 1 else 'Memories'} for Context"
                await self._emit_status(emitter, description, done=True)

        memory_context = "\n\n".join(content_parts)

        system_index = next((i for i, msg in enumerate(body["messages"]) if msg.get("role") == "system"), None)

        if system_index is not None:
            body["messages"][system_index]["content"] = f"{body['messages'][system_index].get('content', '')}\n\n{memory_context}"
        else:
            body["messages"].insert(0, {"role": "system", "content": memory_context})

    async def _collect_consolidation_candidates(
        self, user_message: str, user_id: str, cached_similarities: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Collect candidate memories for consolidation analysis using cached or computed similarities."""
        if cached_similarities is not None:
            logger.info(f"🚀 Reusing cached similarities for {len(cached_similarities)} memories")
            similarity_scores = [mem["relevance"] for mem in cached_similarities]
            consolidation_threshold = self._similarity_service._calculate_dynamic_threshold(similarity_scores, for_consolidation=True)
            candidates = [mem for mem in cached_similarities if mem["relevance"] >= consolidation_threshold]
            logger.info(f"🎯 Found {len(candidates)} candidate memories for consolidation (threshold: {consolidation_threshold:.3f})")
            return candidates

        user_memories = await self._get_user_memories(user_id, ConfigManager.SystemConfig.CONSOLIDATION_TIMEOUT)

        if not user_memories:
            logger.info("📭 No existing memories found for user")
            return []

        logger.info(f"🗃️ Found {len(user_memories)} existing memories for consolidation analysis")

        _, _, all_similarities = await self._compute_memory_similarities(
            user_message, user_id, user_memories, limit_multiplier=1.0
        )
        
        if all_similarities:
            similarity_scores = [mem["relevance"] for mem in all_similarities]
            consolidation_threshold = self._similarity_service._calculate_dynamic_threshold(similarity_scores, for_consolidation=True)
            candidates = [mem for mem in all_similarities if mem["relevance"] >= consolidation_threshold]
            limit = int(ConfigManager.SystemConfig.DEFAULT_MAX_MEMORIES_RETURNED * ConfigManager.SystemConfig.RETRIEVAL_MULTIPLIER)
            candidates = candidates[:limit]
        else:
            candidates = []

        logger.info(f"🎯 Found {len(candidates)} candidate memories for consolidation (threshold: {consolidation_threshold if all_similarities else 'N/A'})")
        return candidates

    async def _generate_consolidation_plan(self, user_message: str, candidate_memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate consolidation plan using LLM with simplified logic."""
        if candidate_memories:
            memory_context = "## CANDIDATE MEMORIES FOR CONSOLIDATION\n"
            for memory in candidate_memories:
                memory_context += f"ID: {memory['id']}\nContent: {memory['content']}\n"
                if memory.get("created_at"):
                    memory_context += f"Created: {memory['created_at']}\n"
                if memory.get("updated_at"):
                    memory_context += f"Updated: {memory['updated_at']}\n"
                memory_context += "\n"
        else:
            memory_context = "## CANDIDATE MEMORIES FOR CONSOLIDATION\nNone - Focus on extracting new memories from the user message below.\n\n"

        date_context = f"## CURRENT DATE/TIME\n{self.format_current_datetime()}\n\n"
        user_message_section = f"## USER MESSAGE\n{user_message}\n\n"
        system_prompt = (
            MEMORY_CONSOLIDATION_PROMPT.format(max_memory_length=ConfigManager.SystemConfig.MAX_MEMORY_CONTENT_LENGTH)
            + f"\n\n{date_context}{memory_context}{user_message_section}"
        )

        response = await asyncio.wait_for(
            self._query_llm(system_prompt, "Analyze the memories and generate a consolidation plan.", response_model=ConsolidationResponse),
            timeout=ConfigManager.SystemConfig.CONSOLIDATION_TIMEOUT,
        )

        operations = response.operations
        existing_memory_ids = {memory["id"] for memory in candidate_memories}

        total_operations = len(operations)
        delete_operations = [op for op in operations if op.operation == MemoryOperationType.DELETE]
        delete_ratio = len(delete_operations) / total_operations if total_operations > 0 else 0

        if delete_ratio > 0.5 and total_operations >= 3:
            logger.warning(
                f"⚠️ Consolidation safety: {len(delete_operations)}/{total_operations} operations are deletions ({delete_ratio*100:.1f}%) - rejecting plan"
            )
            return []

        valid_operations = [op.model_dump() for op in operations if op.validate_operation(existing_memory_ids)]

        if valid_operations:
            create_count = sum(1 for op in valid_operations if op.get("operation") == "CREATE")
            update_count = sum(1 for op in valid_operations if op.get("operation") == "UPDATE")
            delete_count = sum(1 for op in valid_operations if op.get("operation") == "DELETE")

            operation_details = []
            if create_count > 0:
                operation_details.append(f"{create_count} CREATE")
            if update_count > 0:
                operation_details.append(f"{update_count} UPDATE")
            if delete_count > 0:
                operation_details.append(f"{delete_count} DELETE")

            logger.info(f"🎯 Planned {len(valid_operations)} memory optimization operations: {', '.join(operation_details)}")
        else:
            logger.info("🎯 No valid memory optimization operations planned")

        return valid_operations

    async def _execute_memory_operations(self, operations: List[Dict[str, Any]], user_id: str, emitter: Optional[Callable] = None) -> None:
        """Execute consolidation operations with simplified tracking."""
        if not operations:
            return

        user = await AsyncOperationManager.execute_with_timeout(
            asyncio.to_thread(Users.get_user_by_id, user_id), ConfigManager.SystemConfig.TIMEOUT_USER_LOOKUP, "User lookup"
        )

        if not user:
            raise MemorySystemError(f"User not found for consolidation: {user_id}", "Memory operations")

        created_count = updated_count = deleted_count = failed_count = 0

        for operation_index, operation_data in enumerate(operations):
            try:
                operation = MemoryOperation(**operation_data)
                result = await self._execute_single_operation(operation, user)

                if result == MemoryOperationType.CREATE.value:
                    created_count += 1
                elif result == MemoryOperationType.UPDATE.value:
                    updated_count += 1
                elif result == MemoryOperationType.DELETE.value:
                    deleted_count += 1

            except Exception as e:
                failed_count += 1
                self._error_handler.handle_operation_error(operation_data, operation_index, e)

        total_executed = created_count + updated_count + deleted_count
        logger.info(
            f"✅ Memory optimization completed {total_executed}/{len(operations)} operations (Created {created_count}, Updated {updated_count}, Deleted {deleted_count}, Failed {failed_count})"
        )

        if total_executed > 0:
            if emitter:
                operation_details = []
                if created_count > 0:
                    operation_details.append(f"Created {created_count}")
                if updated_count > 0:
                    operation_details.append(f"Updated {updated_count}")
                if deleted_count > 0:
                    operation_details.append(f"Deleted {deleted_count}")

                description = f"🔄 Memory Operations: {', '.join(operation_details)}"
                if failed_count > 0:
                    description += f" ({failed_count} Failed)"
                await self._emit_status(emitter, description, done=True)

            await self._clear_user_cache(user_id)
            await self._warm_user_cache(user_id)
        else:
            if emitter and failed_count > 0:
                await self._emit_status(emitter, f"⚠️ Memory Operations: All {failed_count} operations failed", done=True)

    async def _run_consolidation_pipeline(
        self, user_message: str, user_id: str, emitter: Optional[Callable] = None, cached_similarities: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Complete consolidation pipeline with simplified flow."""
        try:
            if self._shutdown_event.is_set():
                return

            candidates = await self._collect_consolidation_candidates(user_message, user_id, cached_similarities)
            if self._shutdown_event.is_set():
                return

            operations = await self._generate_consolidation_plan(user_message, candidates)
            if self._shutdown_event.is_set():
                return

            if operations:
                await self._execute_memory_operations(operations, user_id, emitter)

        except Exception as e:
            logger.error(f"❌ Memory consolidation failed: {str(e)}")
            if emitter:
                try:
                    await self._emit_status(emitter, "⚠️ Memory Optimization Failed", done=False)
                except Exception as emit_error:
                    logger.warning(f"❌ Failed to emit consolidation error status: {emit_error}")
            raise

    async def inlet(self, body: Dict[str, Any], __event_emitter__: Optional[Callable] = None, __user__: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Simplified inlet processing for memory retrieval and injection."""
        if not (body and __user__):
            return body

        user_id = __user__["id"]

        user_message, should_skip, skip_reason = self._validate_and_extract_user_message(body)

        if not user_message or should_skip:
            if __event_emitter__ and skip_reason:
                await self._emit_status(__event_emitter__, skip_reason, done=True)
            await self._add_memory_context(body, None, user_id, __event_emitter__)
            return body

        try:
            user_memories = await self._get_user_memories(user_id)

            if user_memories:
                retrieval_result = await self._retrieve_relevant_memories(user_message, user_id, user_memories)
                memories = retrieval_result.get("memories", [])
                threshold = retrieval_result.get("dynamic_threshold")
                all_similarities = retrieval_result.get("all_similarities", [])

                if all_similarities:
                    cache_key = f"{user_id}:{hashlib.sha256(user_message.encode('utf-8')).hexdigest()[:16]}"
                    self._retrieval_cache[cache_key] = all_similarities

                    if len(self._retrieval_cache) > 100:
                        self._retrieval_cache.popitem(last=False)
            else:
                memories = []
                threshold = None

            await self._add_memory_context(body, memories, user_id, __event_emitter__)

        except Exception as e:
            logger.error(f"❌ Memory retrieval failed: {str(e)}")
            await self._add_memory_context(body, None, user_id, __event_emitter__)

        return body

    async def outlet(self, body: dict, __event_emitter__: Optional[Callable] = None, __user__: Optional[dict] = None) -> dict:
        """Simplified outlet processing for background memory consolidation."""
        if not (body and __user__):
            return body

        user_id = __user__["id"]

        user_message, should_skip, skip_reason = self._validate_and_extract_user_message(body)

        if not user_message or should_skip:
            if __event_emitter__ and skip_reason:
                await self._emit_status(__event_emitter__, skip_reason, done=True)
            return body

        cache_key = f"{user_id}:{hashlib.sha256(user_message.encode('utf-8')).hexdigest()[:16]}"
        cached_similarities = self._retrieval_cache.get(cache_key)

        if cached_similarities:
            self._retrieval_cache.move_to_end(cache_key)

        task = asyncio.create_task(self._run_consolidation_pipeline(user_message, user_id, __event_emitter__, cached_similarities))
        self._background_tasks.add(task)
        task.add_done_callback(lambda t: self._background_tasks.discard(t))

        if cached_similarities:
            self._retrieval_cache.pop(cache_key, None)

        return body

    async def shutdown(self) -> None:
        """Cleanup method to properly shutdown background tasks."""
        self._shutdown_event.set()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

        if self._aiohttp_session and not self._aiohttp_session.closed:
            await self._aiohttp_session.close()

    async def _clear_user_cache(self, user_id: str) -> None:
        """Invalidate all cache entries for a specific user."""
        def _clear_cache():
            with self._cache_lock:
                if user_id in self._embedding_cache:
                    user_cache = self._embedding_cache[user_id]
                    return user_cache
                return None

        loop = asyncio.get_event_loop()
        user_cache = await loop.run_in_executor(None, _clear_cache)
        
        if user_cache:
            await user_cache.clear()
            logger.info(f"🧹 Embedding cache cleared for user {user_id}")

    async def _warm_user_cache(self, user_id: str) -> None:
        """Pre-populate cache with embeddings for all user memories."""
        start_time = datetime.now()
        try:
            user_memories = await self._get_user_memories(user_id)

            if not user_memories:
                logger.info(f"🔥 No memories to warm cache for user {user_id}")
                return

            memory_contents = [
                memory.content for memory in user_memories 
                if memory.content and len(memory.content.strip()) >= ConfigManager.SkipThresholds.MIN_QUERY_LENGTH
            ]

            if memory_contents:
                await self._generate_embeddings_batch(memory_contents, user_id)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"🔥 Cache warmed with {len(memory_contents)} embeddings for user {user_id} in {duration:.2f}s")
            else:
                logger.info(f"🔥 No valid memory content to warm cache for user {user_id}")

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.warning(f"⚠️ Failed to warm cache for user {user_id} after {duration:.2f}s: {str(e)}")

    async def _execute_single_operation(self, operation: MemoryOperation, user: Any) -> str:
        """Execute a single memory operation."""
        try:
            if operation.operation == MemoryOperationType.CREATE:
                await AsyncOperationManager.execute_db_operation(
                    lambda: Memories.insert_new_memory(user.id, operation.content.strip()), context="Memory creation"
                )
                content_preview = operation.content[:80] + "..." if len(operation.content) > 80 else operation.content
                logger.info(f"✅ Created memory: {content_preview}")
                return MemoryOperationType.CREATE.value

            elif operation.operation == MemoryOperationType.UPDATE and operation.id:
                await AsyncOperationManager.execute_db_operation(
                    lambda: Memories.update_memory_by_id_and_user_id(operation.id, user.id, operation.content.strip()), context="Memory update"
                )
                content_preview = operation.content[:80] + "..." if len(operation.content) > 80 else operation.content
                logger.info(f"🔄 Updated memory {operation.id}: {content_preview}")
                return MemoryOperationType.UPDATE.value

            elif operation.operation == MemoryOperationType.DELETE and operation.id:
                await AsyncOperationManager.execute_db_operation(
                    lambda: Memories.delete_memory_by_id_and_user_id(operation.id, user.id), context="Memory deletion"
                )
                logger.info(f"🗑️ Deleted memory {operation.id}")
                return MemoryOperationType.DELETE.value

            else:
                raise MemorySystemError(f"Unsupported operation: {operation}", "Memory operations")

        except MemorySystemError:
            raise
        except Exception as e:
            raise MemorySystemError(f"Database operation failed: {str(e)}", "Memory operations")

    async def _query_llm(self, system_prompt: str, user_prompt: str, response_model: Optional[BaseModel] = None) -> Union[str, BaseModel]:
        """Query the OpenAI API or compatible endpoints with Pydantic model parsing."""
        if not self.valves.api_key or not self.valves.api_key.strip():
            raise MemorySystemError("API key is required but not provided.")

        session = await self._create_http_session()
        url = f"{self.valves.api_url.rstrip('/')}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.api_key}",
            "Accept": "application/json",
        }

        payload = {
            "model": self.valves.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 4096,
        }

        if response_model:
            payload["response_format"] = {"type": "json_object"}

        async with session.post(url, json=payload, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()

            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]

                if response_model:
                    try:
                        parsed_data = json.loads(content.strip())
                        return response_model.model_validate(parsed_data)
                    except json.JSONDecodeError as e:
                        raise MemorySystemError(f"Invalid JSON from LLM: {e}")
                    except PydanticValidationError as e:
                        raise MemorySystemError(f"LLM response validation failed: {e}")

                if not content or content.strip() == "":
                    raise MemorySystemError("Empty response from LLM")
                return content

            raise MemorySystemError(f"Unexpected API response format: {data}")
