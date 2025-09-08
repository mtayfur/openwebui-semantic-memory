"""
title: Neural Recall
version: 3.0.0
"""

import asyncio
import hashlib
import json
import logging
import re
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Tuple, Union

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

logger = logging.getLogger("NeuralRecallV3")
logger.setLevel(logging.INFO)


class NeuralRecallError(Exception):
    """Base exception for Neural Recall operations."""
    pass


class ModelLoadError(NeuralRecallError):
    """Raised when sentence transformer model fails to load."""
    pass


class EmbeddingError(NeuralRecallError):
    """Raised when embedding generation fails."""
    pass


class MemoryOperationError(NeuralRecallError):
    """Raised when memory operations fail."""
    pass


class ValidationError(NeuralRecallError):
    """Raised when data validation fails."""
    pass


class MemorySkipThresholds:
    """Threshold configuration for memory skip logic detection patterns."""

    # Message length and query limits
    MAX_MESSAGE_LENGTH = 3000  # Max characters allowed for an incoming message
    MIN_QUERY_LENGTH = 10  # Minimum characters considered a valid query

    # JSON and structured data detection
    JSON_KEY_VALUE_THRESHOLD = 5  # Min key:value pairs to consider JSON/structured
    STRUCTURED_LINE_COUNT_MIN = 3  # Min number of lines before structured heuristics apply
    STRUCTURED_PERCENTAGE_THRESHOLD = 0.6  # Fraction of lines matching pattern to trigger structured skip
    STRUCTURED_BULLET_MIN = 4  # Min bullet-like lines to treat as structured list
    STRUCTURED_PIPE_MIN = 2  # Min pipe-count per line to treat as table-like structure

    # Symbol and character ratio detection
    SYMBOL_CHECK_MIN_LENGTH = 10  # Min message length before symbol-ratio check runs
    SYMBOL_RATIO_THRESHOLD = 0.5  # Minimum fraction of alpha/space characters to avoid symbol-skip

    # URL and log detection
    URL_COUNT_THRESHOLD = 3  # Number of URLs considered a URL dump
    LOGS_LINE_COUNT_MIN = 2  # Min lines before log-detection heuristics apply
    LOGS_MIN_MATCHES = 1  # Minimum matched log-like lines to trigger log skip
    LOGS_MATCH_PERCENTAGE = 0.30  # Fraction of lines matching log pattern to trigger skip


class MemorySystemConfig:
    """Core system configuration constants for Neural Recall."""

    # Cache and user limits
    CACHE_MAX_SIZE = 2500  # LRU cache max entries per user
    MAX_USER_CACHES = 500  # Global limit of per-user caches
    MAX_MEMORY_CONTENT_LENGTH = 500  # Max characters allowed when creating a memory

    # Network and DB timeouts (seconds)
    TIMEOUT_SESSION_REQUEST = 30  # aiohttp session total timeout
    TIMEOUT_DATABASE_OPERATION = 10  # Timeout for DB operations
    TIMEOUT_USER_LOOKUP = 5  # Timeout for user lookups

    # Semantic retrieval defaults
    DEFAULT_MAX_MEMORIES_RETURNED = 15  # Default max memories returned
    
    # Dynamic threshold configuration
    PERCENTILE_THRESHOLD = 75  # Use 75th percentile for dynamic threshold
    MIN_DYNAMIC_THRESHOLD = 0.40  # Bottom limit for dynamic threshold
    MAX_DYNAMIC_THRESHOLD = 0.80  # Top limit for dynamic threshold
    CONSOLIDATION_RELAXED_MULTIPLIER = 0.8  # Relaxed threshold multiplier for consolidation

    # Embedding batch sizing
    MIN_BATCH_SIZE = 8  # Minimum embedding batch size
    MAX_BATCH_SIZE = 32  # Maximum embedding batch size

    # Pipeline configuration
    RETRIEVAL_MULTIPLIER = 2.0  # Multiplier for candidate memory retrieval for consolidation
    RETRIEVAL_TIMEOUT = 10.0  # Timeout for retrieval operations
    CONSOLIDATION_TIMEOUT = 30.0  # Timeout for consolidation operations
    STATUS_EMIT_TIMEOUT = 10.0  # Timeout for status emit operations

    # Status messages for skip operations
    STATUS_MESSAGES = {
        "SKIP_EMPTY": "🔍 Message too short to process",
        "SKIP_TOO_LONG": "📄 Message too long to process",
        "SKIP_CODE": "💻 Code content detected, skipping memory operations",
        "SKIP_STRUCTURED": "📊 Structured data detected, skipping memory operations",
        "SKIP_SYMBOLS": "🔢 Symbol heavy content detected, skipping memory operations",
        "SKIP_LOGS": "📝 Log content detected, skipping memory operations",
        "SKIP_STACKTRACE": "⚠️ Stack trace detected, skipping memory operations",
        "SKIP_URL_DUMP": "🔗 URL list detected, skipping memory operations",
    }


MEMORY_CONSOLIDATION_PROMPT = f"""You are the Neural Recall Consolidator, a meticulous component of the Consolidation Pipeline. Your primary function is to enrich the user's memory profile with high-value, fact-based memories derived from conversation. Your secondary function is to perform database hygiene (merging, updating, splitting) with bias towards **information preservation**.

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
  - Combined content stays under {MemorySystemConfig.MAX_MEMORY_CONTENT_LENGTH} characters
  - Result maintains Single Distinct Concept
- **Example:**
  - **Existing Memories:** `["mem-401: I am working on Project Phoenix", "mem-402: Project Phoenix is a data migration initiative"]`
  - **Output:** `[{{"operation":"UPDATE","id":"mem-401","content":"I am working on Project Phoenix, a data migration initiative"}}, {{"operation":"DELETE","id":"mem-402"}}]`

## SPLIT (Enforce Single Distinct Concept)
- **Example:** `["mem-501: I am a vegetarian and my favorite movie is Blade Runner"]` → `[{{"operation":"UPDATE","id":"mem-501","content":"I am a vegetarian"}}, {{"operation":"CREATE","content":"My favorite movie is Blade Runner"}}]`

# CHARACTER LIMIT
Every memory content MUST be under {MemorySystemConfig.MAX_MEMORY_CONTENT_LENGTH} characters.

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

    operation: Literal["CREATE", "UPDATE", "DELETE"]
    content: Optional[str] = Field(None, description="Memory content for CREATE/UPDATE operations")
    id: Optional[str] = Field(None, description="Memory ID for UPDATE/DELETE operations")

    @model_validator(mode="after")
    def validate_operation_requirements(self):
        """Validate operation-specific requirements."""
        if self.operation == "CREATE":
            if not self.content or not self.content.strip():
                raise ValueError("CREATE operation requires non-empty content")
            if len(self.content.strip()) > MemorySystemConfig.MAX_MEMORY_CONTENT_LENGTH:
                raise ValueError(f"Content too long (max {MemorySystemConfig.MAX_MEMORY_CONTENT_LENGTH} characters)")

        elif self.operation == "UPDATE":
            if not self.id:
                raise ValueError("UPDATE operation requires memory ID")
            if not self.content or not self.content.strip():
                raise ValueError("UPDATE operation requires non-empty content")
            if len(self.content.strip()) > MemorySystemConfig.MAX_MEMORY_CONTENT_LENGTH:
                raise ValueError(f"Content too long (max {MemorySystemConfig.MAX_MEMORY_CONTENT_LENGTH} characters)")

        elif self.operation == "DELETE":
            if not self.id:
                raise ValueError("DELETE operation requires memory ID")

        return self

    def validate_operation(self, existing_memory_ids: set = None) -> bool:
        """Validate the memory operation against existing memory IDs."""
        if existing_memory_ids is None:
            existing_memory_ids = set()

        if self.operation == "CREATE":
            return True
        elif self.operation in ["UPDATE", "DELETE"]:
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
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get an item from cache, moving it to end (most recently used)."""
        async with self._lock:
            if key in self._cache:
                value = self._cache.pop(key)
                self._cache[key] = value
                return value
            return None

    async def put(self, key: str, value: Any) -> None:
        """Put an item in cache, evicting least recently used if necessary."""
        async with self._lock:
            if key in self._cache:
                self._cache.pop(key)
            elif len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = value

    async def clear(self) -> int:
        """Clear cache and return number of entries cleared."""
        async with self._lock:
            size = len(self._cache)
            self._cache.clear()
            return size


class Filter:
    """Enhanced multi-model embedding and memory filter with LRU caching."""

    class Valves(BaseModel):
        """Configuration valves for the Neural Recall system."""

        api_url: str = Field(
            default="http://host.docker.internal:11434/v1",
            description="OpenAI-compatible API URL for LLM operations",
        )
        api_key: str = Field(
            default="sk-neural-recall-key",
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
            default=MemorySystemConfig.DEFAULT_MAX_MEMORIES_RETURNED,
            description="Maximum number of memories to return in context",
        )
        
        # Dynamic threshold configuration
        percentile_threshold: int = Field(
            default=MemorySystemConfig.PERCENTILE_THRESHOLD,
            description="Percentile to use for dynamic threshold calculation (1-99)"
        )
        min_dynamic_threshold: float = Field(
            default=MemorySystemConfig.MIN_DYNAMIC_THRESHOLD,
            description="Minimum allowed dynamic threshold value"
        )
        max_dynamic_threshold: float = Field(
            default=MemorySystemConfig.MAX_DYNAMIC_THRESHOLD,
            description="Maximum allowed dynamic threshold value"
        )

    def __init__(self):
        """Initialize the Neural Recall filter with production validation."""
        self.valves = self.Valves()
        self._validate_system_configuration()

        self._model = None
        self._model_load_lock = asyncio.Lock()
        self._embedding_cache: Dict[str, LRUCache] = {}
        self._cache_access_order: List[str] = []
        self._cache_lock = asyncio.Lock()
        self._aiohttp_session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        self._background_tasks: set = set()
        self._shutdown_event = asyncio.Event()

    def _validate_system_configuration(self) -> None:
        """Validate configuration and fail if invalid."""
        if not self.valves.api_url or not self.valves.api_url.strip():
            raise ValidationError("API URL not configured")

        if not self.valves.model or not self.valves.model.strip():
            raise ValidationError("Model not specified")

        if not 1 <= self.valves.percentile_threshold <= 99:
            raise ValidationError(f"Invalid percentile threshold: {self.valves.percentile_threshold}")

        if not 0.0 <= self.valves.min_dynamic_threshold <= 1.0:
            raise ValidationError(f"Invalid min dynamic threshold: {self.valves.min_dynamic_threshold}")

        if not 0.0 <= self.valves.max_dynamic_threshold <= 1.0:
            raise ValidationError(f"Invalid max dynamic threshold: {self.valves.max_dynamic_threshold}")

        if self.valves.min_dynamic_threshold >= self.valves.max_dynamic_threshold:
            raise ValidationError(f"Min threshold ({self.valves.min_dynamic_threshold}) must be less than max threshold ({self.valves.max_dynamic_threshold})")

        if self.valves.max_memories_returned <= 0:
            raise ValidationError(f"Invalid max memories returned: {self.valves.max_memories_returned}")

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
                        raise ModelLoadError(error_msg)

        return self._model

    async def _create_user_cache(self, user_id: str) -> LRUCache:
        """Get or create user-specific embedding cache with global user limit."""
        async with self._cache_lock:
            if user_id in self._embedding_cache:
                try:
                    self._cache_access_order.remove(user_id)
                except ValueError:
                    pass
                self._cache_access_order.append(user_id)
                return self._embedding_cache[user_id]

            if len(self._embedding_cache) >= MemorySystemConfig.MAX_USER_CACHES:
                if self._cache_access_order:
                    lru_user_id = self._cache_access_order.pop(0)
                    if lru_user_id in self._embedding_cache:
                        del self._embedding_cache[lru_user_id]
                        logger.info(f"🧹 Cache evicted for user {lru_user_id}")

            self._embedding_cache[user_id] = LRUCache(MemorySystemConfig.CACHE_MAX_SIZE)
            self._cache_access_order.append(user_id)
            return self._embedding_cache[user_id]

    async def _generate_embedding(self, text: str, user_id: str) -> np.ndarray:
        """Generate embedding for a single text using the batch method."""
        embeddings = await self._generate_embeddings_batch([text], user_id)
        if embeddings and embeddings[0] is not None:
            return embeddings[0]
        raise EmbeddingError("Failed to generate embedding for the given text.")

    async def _generate_embeddings_batch(self, texts: List[str], user_id: str) -> List[np.ndarray]:
        """Generate embeddings for multiple texts efficiently with batch processing and caching."""
        if not texts:
            return []

        cache = await self._create_user_cache(user_id)

        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []

        for text_index, text in enumerate(texts):
            if not text or len(text.strip()) < MemorySkipThresholds.MIN_QUERY_LENGTH:
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
            try:
                model = await self._load_embedding_model()

                batch_size = min(
                    MemorySystemConfig.MAX_BATCH_SIZE,
                    max(MemorySystemConfig.MIN_BATCH_SIZE, len(uncached_texts)),
                )

                def generate_batch_embeddings(batch_texts):
                    embeddings = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
                    normalized_embeddings = []
                    for embedding in embeddings:
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = embedding / norm
                        normalized_embeddings.append(embedding)
                    return normalized_embeddings

                loop = asyncio.get_event_loop()

                for batch_start in range(0, len(uncached_texts), batch_size):
                    batch_texts = uncached_texts[batch_start : batch_start + batch_size]
                    batch_indices = uncached_indices[batch_start : batch_start + batch_size]

                    batch_embeddings = await loop.run_in_executor(None, generate_batch_embeddings, batch_texts)

                    for batch_index, embedding in enumerate(batch_embeddings):
                        text_idx = batch_indices[batch_index]
                        text = batch_texts[batch_index]
                        text_hash = hashlib.sha256(text.encode()).hexdigest()

                        await cache.put(text_hash, embedding)
                        new_embeddings[text_idx] = embedding

            except Exception as e:
                error_msg = f"Batch embedding generation failed: {str(e)}"
                logger.error(f"❌ {error_msg}")
                raise EmbeddingError(error_msg)

        result_embeddings = []
        for text_index in range(len(texts)):
            if text_index in cached_embeddings:
                result_embeddings.append(cached_embeddings[text_index])
            elif text_index in new_embeddings:
                result_embeddings.append(new_embeddings[text_index])
            else:
                result_embeddings.append(None)

        valid_count = len([emb for emb in result_embeddings if emb is not None])
        logger.info(f"🚀 Batch embedding: {len(cached_embeddings)} cached, {len(new_embeddings)} new, {valid_count}/{len(texts)} valid")
        return result_embeddings

    async def _create_http_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with connection pooling and production settings."""
        if self._aiohttp_session is None or self._aiohttp_session.closed:
            async with self._session_lock:
                if self._aiohttp_session is None or self._aiohttp_session.closed:
                    timeout = aiohttp.ClientTimeout(total=MemorySystemConfig.TIMEOUT_SESSION_REQUEST)
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
                        headers={"User-Agent": "NeuralRecallV3/3.0.0"},
                    )
        return self._aiohttp_session

    def _should_skip_memory_operations(self, user_message: str) -> Tuple[bool, str]:
        """
        Enhanced gating for memory operations with comprehensive pattern detection.
        Skips: empty, too long, code, logs, structured data, URL dumps, or symbol spam.
        """
        if not user_message or not user_message.strip():
            return True, MemorySystemConfig.STATUS_MESSAGES["SKIP_EMPTY"]

        trimmed_message = user_message.strip()

        if len(trimmed_message) < MemorySkipThresholds.MIN_QUERY_LENGTH:
            return True, MemorySystemConfig.STATUS_MESSAGES["SKIP_EMPTY"]

        if len(trimmed_message) > MemorySkipThresholds.MAX_MESSAGE_LENGTH:
            return True, MemorySystemConfig.STATUS_MESSAGES["SKIP_TOO_LONG"]

        if self._is_code_content(trimmed_message):
            return True, MemorySystemConfig.STATUS_MESSAGES["SKIP_CODE"]

        if self._is_structured_data(trimmed_message):
            return True, MemorySystemConfig.STATUS_MESSAGES["SKIP_STRUCTURED"]

        if self._is_mostly_symbols(trimmed_message):
            return True, MemorySystemConfig.STATUS_MESSAGES["SKIP_SYMBOLS"]

        if self._is_log_content(trimmed_message):
            return True, MemorySystemConfig.STATUS_MESSAGES["SKIP_LOGS"]

        if self._is_stack_trace(trimmed_message):
            return True, MemorySystemConfig.STATUS_MESSAGES["SKIP_STACKTRACE"]

        if self._is_url_dump(trimmed_message):
            return True, MemorySystemConfig.STATUS_MESSAGES["SKIP_URL_DUMP"]

        return False, ""

    def _is_code_content(self, message: str) -> bool:
        """Check if message contains code patterns."""
        code_patterns = [
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
        return any(re.search(pattern, message, re.MULTILINE | re.IGNORECASE) for pattern in code_patterns)

    def _is_structured_data(self, message: str) -> bool:
        """Check if message contains structured data patterns."""
        json_indicators = [
            r"^\s*[\{\[].*[\}\]]\s*$",
            r'"[^"]*"\s*:\s*["\{\[\d]',
            r'["\']?\w+["\']?\s*:\s*["\{\[\d].*["\'\}\]\d]',
            r'\{\s*["\']?\w+["\']?\s*:\s*\{',
            r"\[\s*\{.*\}\s*\]",
        ]

        if any(re.search(pattern, message, re.DOTALL | re.IGNORECASE) for pattern in json_indicators):
            return True

        json_kv_pattern = r'["\']?\w+["\']?\s*:\s*["\{\[\d\w]'
        if len(re.findall(json_kv_pattern, message)) >= MemorySkipThresholds.JSON_KEY_VALUE_THRESHOLD:
            return True

        message_lines = message.split("\n")
        line_count = len(message_lines)

        if line_count < MemorySkipThresholds.STRUCTURED_LINE_COUNT_MIN:
            return False

        count_lines = lambda pattern: sum(1 for line in message_lines if re.match(pattern, line))
        threshold = max(
            MemorySkipThresholds.STRUCTURED_LINE_COUNT_MIN,
            line_count * MemorySkipThresholds.STRUCTURED_PERCENTAGE_THRESHOLD,
        )

        structured_checks = [
            count_lines(r"^\s*[A-Za-z0-9_]+:\s+\S+") >= threshold,
            count_lines(r"^\s*[-\*\+]\s+.+") >= max(MemorySkipThresholds.STRUCTURED_BULLET_MIN, threshold),
            count_lines(r"^\s*\d+\.\s") >= threshold,
            count_lines(r"^\s*[a-zA-Z]\)\s") >= threshold,
            sum(1 for line in message_lines if "|" in line and line.count("|") >= MemorySkipThresholds.STRUCTURED_PIPE_MIN)
            >= max(MemorySkipThresholds.STRUCTURED_PIPE_MIN, threshold),
        ]

        return any(structured_checks)

    def _is_mostly_symbols(self, message: str) -> bool:
        """Check if message is mostly symbols/numbers."""
        if len(message) <= MemorySkipThresholds.SYMBOL_CHECK_MIN_LENGTH:
            return False

        alpha_space_count = sum(1 for c in message if c.isalpha() or c.isspace())
        return alpha_space_count / len(message) < MemorySkipThresholds.SYMBOL_RATIO_THRESHOLD

    def _is_log_content(self, message: str) -> bool:
        """Check if message contains log patterns."""
        message_lines = message.split("\n")
        line_count = len(message_lines)

        single_line_error_patterns = [
            r"(NameError|TypeError|ValueError|AttributeError|KeyError|SyntaxError|IndentationError).*:",
            r"\d{2,4}[-/]\d{1,2}[-/]\d{1,2}[\s\d:]+\b(ERROR|WARN|INFO|DEBUG)\b",
            r"\bERROR\b.*:.*",
            r'^\s*(at|File)\s+["\']?[^"\']*["\']?,?\s*line\s+\d+',
            r"^\s*at\s+\w+.*\([^)]*:\d+:\d+\)",
            r"Exception\s+in\s+thread",
        ]

        if any(re.search(pattern, message, re.IGNORECASE) for pattern in single_line_error_patterns):
            return True

        if line_count < MemorySkipThresholds.LOGS_LINE_COUNT_MIN:
            return False

        datetime_matches = sum(1 for line in message_lines if re.search(r"\d{2,4}[-/]\d{1,2}[-/]\d{1,2}.*\d{1,2}:\d{2}", line))
        if datetime_matches >= max(
            MemorySkipThresholds.LOGS_MIN_MATCHES,
            line_count * MemorySkipThresholds.LOGS_MATCH_PERCENTAGE,
        ):
            return True

        return False

    def _is_stack_trace(self, message: str) -> bool:
        """Check if message contains stack trace patterns."""
        message_lines = message.split("\n")

        stack_patterns = [
            r"^\s*(at|File|Traceback|Exception|Error).*:\d+",
            r"Traceback\s*\(most recent call",
            r'^\s*File\s*"[^"]+",\s*line\s*\d+',
            r"^\s*at\s+\w+.*\([^)]*:\d+:\d+\)",
            r"Exception\s+in\s+thread",
            r"^\s*(NameError|TypeError|ValueError|AttributeError|KeyError):",
        ]

        return any(any(re.search(pattern, line) for pattern in stack_patterns) for line in message_lines)

    def _is_url_dump(self, message: str) -> bool:
        """Check if message contains multiple URLs (URL dump)."""
        return len(re.findall(r"https?://[^\s]+", message)) >= MemorySkipThresholds.URL_COUNT_THRESHOLD

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

    def format_current_datetime(self) -> str:
        try:
            now = datetime.now(timezone.utc)
            return now.strftime("%A %B %d %Y at %H:%M:%S UTC")
        except Exception as e:
            raise NeuralRecallError(f"Failed to format datetime: {e}")

    async def _send_status_update(
        self,
        emitter: Optional[Callable[[Any], Awaitable[None]]],
        message: str,
        done: bool = False,
    ) -> None:
        """Emit status message through the event emitter with robust error handling."""
        if not emitter:
            return

        payload = {
            "type": "status",
            "data": {"description": message, "done": done},
        }

        try:
            result = emitter(payload)
            if asyncio.iscoroutine(result):
                await asyncio.wait_for(result, timeout=MemorySystemConfig.STATUS_EMIT_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(f"⏱️ Status emit timeout after {MemorySystemConfig.STATUS_EMIT_TIMEOUT}s for message: {message[:50]}")
        except Exception as e:
            logger.warning(f"❌ Status emit failed for message '{message[:50]}': {e}")

    async def _retrieve_relevant_memories(
        self,
        user_message: str,
        user_id: str,
        emitter: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> List[Dict[str, Any]]:
        """Step 1 of Retrieval Pipeline: Perform broad vector search for candidate memories."""
        candidate_limit = int(self.valves.max_memories_returned * MemorySystemConfig.RETRIEVAL_MULTIPLIER)

        user_memories = await asyncio.wait_for(
            asyncio.to_thread(Memories.get_memories_by_user_id, user_id),
            timeout=MemorySystemConfig.RETRIEVAL_TIMEOUT,
        )

        if not user_memories:
            logger.info("📭 No memories found for user")
            return []

        logger.info(f"🔍 Found {len(user_memories)} total memories for analysis")

        query_embedding = await self._generate_embedding(user_message, user_id)

        memory_contents = [memory.content for memory in user_memories]

        memory_embeddings = await self._generate_embeddings_batch(memory_contents, user_id)

        if len(memory_embeddings) != len(user_memories):
            error_msg = f"Embedding count mismatch: {len(memory_embeddings)} vs {len(user_memories)}"
            logger.error(f"❌ {error_msg}")
            raise EmbeddingError(error_msg)

        similarity_scores = []
        memory_data = []

        for memory_index, memory in enumerate(user_memories):
            memory_embedding = memory_embeddings[memory_index]
            if memory_embedding is None:
                continue

            similarity = float(np.dot(query_embedding, memory_embedding))
            similarity_scores.append(similarity)
            
            memory_dict = {
                "id": str(memory.id),
                "content": memory.content,
                "relevance": similarity,
            }
            memory_data.append(memory_dict)

        if not similarity_scores:
            logger.info("🎯 No valid similarity scores available, returning empty memory list")
            return []

        percentile_threshold = float(np.percentile(similarity_scores, self.valves.percentile_threshold))
        dynamic_threshold = max(
            self.valves.min_dynamic_threshold,
            min(self.valves.max_dynamic_threshold, percentile_threshold)
        )

        formatted_memories = [
            memory for memory in memory_data 
            if memory["relevance"] >= dynamic_threshold
        ]

        formatted_memories.sort(key=lambda x: x["relevance"], reverse=True)
        formatted_memories = formatted_memories[:candidate_limit]

        logger.info(f"🎯 Selected {len(formatted_memories)} candidate memories (dynamic threshold {dynamic_threshold:.3f}, {self.valves.percentile_threshold}th percentile: {percentile_threshold:.3f})")

        if formatted_memories:
            logger.info("📊 Memories ranked by semantic similarity (highest to lowest):")
            for rank, memory in enumerate(formatted_memories, 1):
                content_preview = memory["content"][:120] + "..." if len(memory["content"]) > 120 else memory["content"]
                logger.info(f"  {rank}. Score: {memory['relevance']:.4f} | Content: {content_preview}")

        return formatted_memories

    async def _add_memory_context(
        self,
        body: Dict[str, Any],
        memories: List[Dict[str, Any]] = None,
        user_id: str = None,
    ) -> None:
        if not body or "messages" not in body or not body["messages"]:
            logger.info("❌ Invalid body or no messages found")
            return

        logger.info(f"🧠 Preparing to provide {len(memories) if memories else 0} memories for user {user_id}")

        current_datetime = self.format_current_datetime()
        content_parts = [f"Current Date/Time: {current_datetime}"]

        if memories and user_id:
            valid_memories = [m for m in memories if m.get('content', '').strip()]
            
            if valid_memories:
                memory_count = len(valid_memories)
                fact_word = "fact" if memory_count == 1 else f"{memory_count} facts"
                memory_header = f"BACKGROUND: You naturally know {'this' if memory_count == 1 else 'these'} {fact_word}. Never mention {'its' if memory_count == 1 else 'their'} source."

                formatted_memories = []
                for memory in valid_memories:
                    content = memory['content'].strip()                
                    content = content.replace('\n', ' ').replace('\r', ' ')
                    content = content.replace('- ', '• ').replace('\n- ', '\n• ')                    
                    content = ' '.join(content.split())

                    formatted_memories.append(f"- {content}")
                
                memory_content = "\n".join(formatted_memories)
                content_parts.append(f"{memory_header}\n{memory_content}")
                logger.info(f"💭 Added {len(valid_memories)} memories to context (filtered from {len(memories)} total)")

        memory_context = "\n\n".join(content_parts)
        logger.info(f"🔧 Created memory context")

        existing_system_msg = None
        existing_system_index = None

        for i, message in enumerate(body["messages"]):
            if message.get("role") == "system":
                existing_system_msg = message
                existing_system_index = i
                break

        if existing_system_msg:
            original_content = existing_system_msg.get("content", "")
            updated_content = f"{original_content}\n\n{memory_context}"
            body["messages"][existing_system_index]["content"] = updated_content
            logger.info(f"✅ Memory context appended to existing system message at position {existing_system_index}")
        else:
            system_message = {"role": "system", "content": memory_context}
            body["messages"].insert(0, system_message)
            logger.info(f"✅ New system message with memory context inserted at position 0")

    async def _collect_consolidation_candidates(
        self,
        user_message: str,
        user_id: str,
        emitter: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> List[Dict[str, Any]]:
        """Step 1 of Consolidation Pipeline: Gather candidate memories for consolidation analysis."""
        user_memories = await asyncio.wait_for(
            asyncio.to_thread(Memories.get_memories_by_user_id, user_id),
            timeout=MemorySystemConfig.CONSOLIDATION_TIMEOUT,
        )

        if not user_memories:
            logger.info("📭 No existing memories found for user")
            await self._send_status_update(emitter, "💭 No memories to analyze", True)
            return []

        logger.info(f"🗃️ Found {len(user_memories)} existing memories for consolidation analysis")

        query_embedding = await self._generate_embedding(user_message, user_id)

        memory_contents = [memory.content for memory in user_memories]

        memory_embeddings = await self._generate_embeddings_batch(memory_contents, user_id)

        if len(memory_embeddings) != len(user_memories):
            error_msg = f"Consolidation embedding count mismatch: {len(memory_embeddings)} vs {len(user_memories)}"
            logger.error(f"❌ {error_msg}")
            raise EmbeddingError(error_msg)

        similarity_scores = []
        memory_data = []

        for memory_index, memory in enumerate(user_memories):
            memory_embedding = memory_embeddings[memory_index]
            if memory_embedding is None:
                continue

            similarity = float(np.dot(query_embedding, memory_embedding))
            similarity_scores.append(similarity)

            created_at_iso = None
            if hasattr(memory, "created_at") and memory.created_at:
                created_at_iso = datetime.fromtimestamp(memory.created_at, tz=timezone.utc).isoformat()

            updated_at_iso = None
            if hasattr(memory, "updated_at") and memory.updated_at:
                updated_at_iso = datetime.fromtimestamp(memory.updated_at, tz=timezone.utc).isoformat()

            memory_dict = {
                "id": str(memory.id),
                "content": memory.content,
                "created_at": created_at_iso,
                "updated_at": updated_at_iso,
                "relevance": similarity,
            }
            memory_data.append(memory_dict)

        if not similarity_scores:
            logger.info("🎯 No valid similarity scores available for consolidation, returning empty list")
            return []

        percentile_threshold = float(np.percentile(similarity_scores, self.valves.percentile_threshold))
        base_dynamic_threshold = max(
            self.valves.min_dynamic_threshold,
            min(self.valves.max_dynamic_threshold, percentile_threshold)
        )
        relaxed_threshold = base_dynamic_threshold * MemorySystemConfig.CONSOLIDATION_RELAXED_MULTIPLIER

        formatted_memories = [
            memory for memory in memory_data 
            if memory["relevance"] >= relaxed_threshold
        ]

        candidate_limit = int(self.valves.max_memories_returned * MemorySystemConfig.RETRIEVAL_MULTIPLIER)
        formatted_memories.sort(key=lambda x: x["relevance"], reverse=True)
        formatted_memories = formatted_memories[:candidate_limit]

        logger.info(f"🎯 Found {len(formatted_memories)} candidate memories for consolidation analysis with {relaxed_threshold:.3f})")

        if len(formatted_memories) >= candidate_limit and len(user_memories) > candidate_limit:
            await self._send_status_update(
                emitter,
                f"🔍 Analyzing {len(formatted_memories)} candidate memories from {len(user_memories)} total",
                False,
            )
        else:
            await self._send_status_update(
                emitter,
                f"🔍 Analyzing {len(formatted_memories)} candidate memories",
                False,
            )

        return formatted_memories

    async def _generate_consolidation_plan(
        self,
        user_message: str,
        candidate_memories: List[Dict[str, Any]],
        emitter: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> List[Dict[str, Any]]:
        """Step 2 of Consolidation Pipeline: Use LLM to generate consolidation plan."""

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

        system_prompt = MEMORY_CONSOLIDATION_PROMPT + f"\n\n{date_context}{memory_context}{user_message_section}"

        await self._send_status_update(emitter, "🔧 Analyzing memory patterns for optimization", False)
        response = await asyncio.wait_for(
            self._query_llm(
                system_prompt,
                "Analyze the memories and generate a consolidation plan.",
                response_model=ConsolidationResponse,
            ),
            timeout=MemorySystemConfig.CONSOLIDATION_TIMEOUT,
        )

        logger.info(f"🔧 Memory consolidation analysis completed with {len(response.operations)} operations")

        operations = response.operations
        existing_memory_ids = {memory["id"] for memory in candidate_memories}

        total_operations = len(operations)
        delete_operations = [op for op in operations if op.operation == "DELETE"]
        delete_ratio = len(delete_operations) / total_operations if total_operations > 0 else 0

        if delete_ratio > 0.5 and total_operations >= 3:
            logger.warning(f"⚠️ Consolidation safety trigger: {len(delete_operations)}/{total_operations} operations are deletions ({delete_ratio*100:.1f}%)")
            logger.warning("⚠️ Rejecting consolidation plan to prevent excessive memory loss")
            await self._send_status_update(
                emitter,
                "⚠️ Memory consolidation plan too aggressive, preserving existing memories",
                False,
            )
            return []

        valid_operations = []
        for operation in operations:
            if operation.validate_operation(existing_memory_ids):
                valid_operations.append(operation.model_dump())

        logger.info(f"🎯 Planned {len(valid_operations)} memory optimization operations")
        await self._send_status_update(emitter, f"🎯 Planning {len(valid_operations)} memory optimizations", False)
        return valid_operations

    async def _execute_memory_operations(
        self,
        operations: List[Dict[str, Any]],
        user_id: str,
        emitter: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> None:
        """Step 3 of Consolidation Pipeline: Execute consolidation operations against database."""
        if not operations:
            return

        user = await asyncio.wait_for(
            asyncio.to_thread(Users.get_user_by_id, user_id),
            timeout=MemorySystemConfig.TIMEOUT_USER_LOOKUP,
        )

        if not user:
            error_msg = f"User not found for consolidation: {user_id}"
            logger.error(f"❌ {error_msg}")
            raise MemoryOperationError(error_msg)

        created_count = 0
        updated_count = 0
        deleted_count = 0
        failed_operations = []

        for operation_index, operation_data in enumerate(operations):
            try:
                operation = MemoryOperation(**operation_data)
                result = await self._execute_single_operation(operation, user)
                if result == "CREATE":
                    created_count += 1
                elif result == "UPDATE":
                    updated_count += 1
                elif result == "DELETE":
                    deleted_count += 1
            except Exception as e:
                operation_type = operation_data.get("operation", "UNKNOWN")
                operation_id = operation_data.get("id", "no-id")
                error_details = {
                    "index": operation_index + 1,
                    "operation_type": operation_type,
                    "operation_id": operation_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                failed_operations.append(error_details)
                logger.error(f"❌ Failed to execute {operation_type} operation {operation_index+1}/{len(operations)} (ID: {operation_id}): {str(e)}")
                continue

        failed_count = len(failed_operations)
        total_executed = created_count + updated_count + deleted_count

        if failed_operations:
            logger.warning(f"🔥 Failed operations summary: {failed_operations}")

        logger.info(
            f"✅ Memory optimization completed {total_executed}/{len(operations)} operations (Created {created_count}, Updated {updated_count}, Deleted {deleted_count}, Failed {failed_count})"
        )

        if total_executed > 0:
            result_parts = []
            if created_count > 0:
                result_parts.append(f"Created {created_count}")
            if updated_count > 0:
                result_parts.append(f"Updated {updated_count}")
            if deleted_count > 0:
                result_parts.append(f"Deleted {deleted_count}")

            suffix = "memory" if total_executed == 1 else "memories"
            message = f"✅ " + ", ".join(result_parts) + f" {suffix}"

            if failed_count > 0:
                message += f" ({failed_count} failed)"

            await self._send_status_update(emitter, message, True)
            await self._clear_user_cache(user_id, "consolidation")
        elif failed_count > 0:
            await self._send_status_update(
                emitter,
                f"⚠️ {failed_count} of {len(operations)} operations failed",
                True,
            )
        else:
            await self._send_status_update(emitter, "✅ Memories already optimally organized", True)

    async def _run_consolidation_pipeline(
        self,
        user_message: str,
        user_id: str,
        emitter: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> None:
        """Complete Consolidation Pipeline as async background task (formerly Slow Path)."""
        try:
            if self._shutdown_event.is_set():
                logger.info("🛑 Shutdown requested, aborting consolidation pipeline")
                return
                
            logger.info("🔧 Starting memory consolidation analysis")
            await self._send_status_update(emitter, "🔧 Analyzing memory patterns", False)

            candidates = await self._collect_consolidation_candidates(user_message, user_id, emitter)

            if self._shutdown_event.is_set():
                return

            if not candidates:
                logger.info("🔧 No existing memories found, analyzing for new memory extraction")
                await self._send_status_update(emitter, "💭 Analyzing conversation for new memories", False)
            else:
                logger.info(f"🔧 Found {len(candidates)} consolidation candidates")

            operations = await self._generate_consolidation_plan(user_message, candidates, emitter)

            if self._shutdown_event.is_set():
                return

            if not operations:
                if candidates:
                    logger.info("🔧 Existing memories already optimally organized, no changes needed")
                    await self._send_status_update(emitter, "✅ Memories already optimally organized", True)
                else:
                    logger.info("🔧 No new memories extracted from conversation")
                    await self._send_status_update(emitter, "💭 No new memories to create", True)
                return

            await self._execute_memory_operations(operations, user_id, emitter)

            logger.info("🔧 Memory consolidation completed successfully")

        except Exception as e:
            logger.error(f"❌ Memory consolidation failed {str(e)}")
            await self._send_status_update(emitter, f"❌ Memory consolidation failed {str(e)[:50]}", True)
            raise

    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not body or not __user__:
            return body

        user_id = __user__["id"]
        memories = None

        logger.info(f"🚪 Processing request for user {user_id}")

        if "messages" in body and body["messages"]:
            logger.info(f"📨 Found {len(body['messages'])} messages in request")

            last_user_msg = next(
                (self._extract_text_content(m["content"]) for m in reversed(body.get("messages", [])) if m.get("role") == "user"),
                None,
            )

            logger.info(f"👤 User message extracted {'successfully' if last_user_msg else 'not found'}")

            if last_user_msg:
                should_skip, skip_reason = self._should_skip_memory_operations(last_user_msg)
                logger.info(f"🔍 Skip analysis {skip_reason if should_skip else 'proceeding with memory retrieval'}")

                if should_skip:
                    status_message = f"⏭️ {skip_reason}"
                else:
                    await self._send_status_update(__event_emitter__, "🚀 Searching for relevant memories", False)

                    candidate_memories = await self._retrieve_relevant_memories(last_user_msg, user_id, __event_emitter__)
                    logger.info(f"🔍 Found {len(candidate_memories) if candidate_memories else 0} candidate memories")

                    if candidate_memories:
                        memories = candidate_memories[: self.valves.max_memories_returned]
                        logger.info(f"🎯 Selected {len(memories)} memories based on semantic similarity")

                        if memories:
                            memory_count = len(memories)
                            if memory_count == 1:
                                status_message = "💡 Found 1 relevant memory"
                            else:
                                status_message = f"💡 Found {memory_count} relevant memories"
                        else:
                            status_message = "💭 No memories selected"
                    else:
                        status_message = "💭 No relevant memories found"

                await self._send_status_update(__event_emitter__, status_message, True)
                await self._add_memory_context(body, memories, user_id)
        else:
            await self._add_memory_context(body, None, user_id)

        logger.info(f"✅ Request processing complete, provided {len(memories) if memories else 0} memories")
        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        """
        CONSOLIDATION PIPELINE (Asynchronous, formerly Slow Path)
        Process outgoing messages to consolidate and optimize memory database.
        """
        if not body or not __user__:
            return body

        user_id = __user__["id"]

        logger.info(f"🚪 Starting background memory optimization for user {user_id}")

        if "messages" in body and body["messages"]:
            user_message = next(
                (self._extract_text_content(m["content"]) for m in reversed(body["messages"]) if m["role"] == "user"),
                None,
            )

            logger.info(f"👤 User message {'found' if user_message else 'not found'} for optimization analysis")

            if user_message:
                should_skip, skip_reason = self._should_skip_memory_operations(user_message)
                if should_skip:
                    logger.info(f"⏭️ Memory optimization skipped {skip_reason}")
                    return body

                logger.info("🔧 Starting background memory optimization task")
                task = asyncio.create_task(self._run_consolidation_pipeline(user_message, user_id, __event_emitter__))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

        return body

    async def _clear_user_cache(self, user_id: str, reason: str = "") -> None:
        """Invalidate all cache entries for a specific user."""
        async with self._cache_lock:
            if user_id in self._embedding_cache:
                user_cache = self._embedding_cache[user_id]
                await user_cache.clear()
                logger.info(f"🧹 Embedding cache cleared for user {user_id} (reason: {reason})")

                if user_id in self._cache_access_order:
                    self._cache_access_order.remove(user_id)

    async def _execute_database_operation(self, operation_function, *args, timeout: float = None) -> Any:
        """Execute database operation with timeout and error handling."""
        if timeout is None:
            timeout = MemorySystemConfig.TIMEOUT_DATABASE_OPERATION

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(operation_function, *args),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.error(f"⏰ Database operation timed out after {timeout}s. WARNING: The underlying synchronous operation may still be running.")
            logger.warning("⚠️ If you see frequent database timeouts, verify that OpenWebUI's database layer is thread-safe")
            raise

    async def _execute_single_operation(
        self,
        operation: MemoryOperation,
        user: Any,
    ) -> str:
        """Execute a single memory operation."""
        if operation.operation == "CREATE":
            await self._execute_database_operation(Memories.insert_new_memory, user.id, operation.content.strip())

            logger.info("✨ Memory created successfully")
            return "CREATE"

        elif operation.operation == "UPDATE" and operation.id:
            await self._execute_database_operation(
                Memories.update_memory_by_id_and_user_id,
                operation.id,
                user.id,
                operation.content.strip(),
            )

            logger.info(f"📝 Memory updated successfully")
            return "UPDATE"

        elif operation.operation == "DELETE" and operation.id:
            await self._execute_database_operation(Memories.delete_memory_by_id_and_user_id, operation.id, user.id)

            logger.info("🗑️ Memory deleted successfully")
            return "DELETE"

        else:
            raise MemoryOperationError(f"Unsupported operation: {operation}")

    async def _query_llm(self, system_prompt: str, user_prompt: str, response_model: Optional[BaseModel] = None) -> Union[str, BaseModel]:
        """Query the OpenAI API or compatible endpoints with Pydantic model parsing."""
        if not self.valves.api_key or not self.valves.api_key.strip():
            raise NeuralRecallError("API key is required but not provided.")

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
                        raise NeuralRecallError(f"Invalid JSON from LLM: {e}")
                    except PydanticValidationError as e:
                        raise NeuralRecallError(f"LLM response validation failed: {e}")

                if not content or content.strip() == "":
                    raise NeuralRecallError("Empty response from LLM")
                return content

            raise NeuralRecallError(f"Unexpected API response format: {data}")