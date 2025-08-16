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
import weakref
import numpy as np
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
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


class SkipThresholds:
    """Threshold configuration for skip logic detection patterns."""

    # Message length and query limits
    MAX_MESSAGE_LENGTH = 3000  # Max characters allowed for an incoming message
    MIN_QUERY_LENGTH = 10  # Minimum characters considered a valid query

    # JSON and structured data detection
    JSON_KEY_VALUE_THRESHOLD = 5  # Min key:value pairs to consider JSON/structured
    STRUCTURED_LINE_COUNT_MIN = (
        3  # Min number of lines before structured heuristics apply
    )
    STRUCTURED_PERCENTAGE_THRESHOLD = (
        0.6  # Fraction of lines matching pattern to trigger structured skip
    )
    STRUCTURED_BULLET_MIN = 4  # Min bullet-like lines to treat as structured list
    STRUCTURED_PIPE_MIN = 2  # Min pipe-count per line to treat as table-like structure

    # Symbol and character ratio detection
    SYMBOL_CHECK_MIN_LENGTH = 10  # Min message length before symbol-ratio check runs
    SYMBOL_RATIO_THRESHOLD = (
        0.5  # Minimum fraction of alpha/space characters to avoid symbol-skip
    )

    # URL and log detection
    URL_COUNT_THRESHOLD = 3  # Number of URLs considered a URL dump
    LOGS_LINE_COUNT_MIN = 2  # Min lines before log-detection heuristics apply
    LOGS_MIN_MATCHES = 1  # Minimum matched log-like lines to trigger log skip
    LOGS_MATCH_PERCENTAGE = (
        0.30  # Fraction of lines matching log pattern to trigger skip
    )


class Config:
    """Core system configuration constants for Neural Recall."""

    # Cache and user limits
    CACHE_MAX_SIZE = 2500  # LRU cache max entries per user
    MAX_USER_CACHES = 500  # Global limit of per-user caches
    MAX_MEMORY_CONTENT_LENGTH = 600  # Max characters allowed when creating a memory

    # Network and DB timeouts (seconds)
    TIMEOUT_SESSION_REQUEST = 30  # aiohttp session total timeout
    TIMEOUT_DATABASE_OPERATION = 10  # Timeout for DB operations
    TIMEOUT_USER_LOOKUP = 5  # Timeout for user lookups

    # Semantic retrieval defaults
    DEFAULT_SEMANTIC_THRESHOLD = 0.50  # Default similarity threshold for retrieval
    DEFAULT_MAX_MEMORIES_RETURNED = 15  # Default max memories injected into context

    # Conversation-based memory tracking
    MAX_CONVERSATION_CACHES = 200  # Max conversation caches per user
    CONVERSATION_CACHE_CLEANUP_THRESHOLD = (
        250  # Trigger cleanup when cache exceeds this
    )

    # Embedding batch sizing
    MIN_BATCH_SIZE = 8  # Minimum embedding batch size
    MAX_BATCH_SIZE = 32  # Maximum embedding batch size

    # Pipeline configuration
    RETRIEVAL_MULTIPLIER = 3.0  # Multiplier for candidate memory retrieval
    RETRIEVAL_TIMEOUT = 5.0  # Timeout for retrieval operations
    CONSOLIDATION_CANDIDATE_SIZE = 50  # Max memories for consolidation analysis
    CONSOLIDATION_TIMEOUT = 30.0  # Timeout for consolidation operations
    CONSOLIDATION_RELAXED_MULTIPLIER = (
        0.9  # Relaxed threshold multiplier for consolidation
    )

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


MEMORY_RERANKING_PROMPT = f"""You are the Neural Recall Re-ranker, a high-speed component of the Retrieval Pipeline. Your mission is to rapidly select a small, focused set of memories that will directly and materially improve the quality of the AI's upcoming response. Speed and precision are critical.

## 🎯 PRIMARY GOAL
Your objective is to construct the ideal short-term context for the AI. You are the bridge between the long-term memory store and the AI's immediate awareness. Your selections must be impactful and relevant, prioritizing the high-quality, dense, and factual memories created by the Consolidation Pipeline.

## 🏛️ CORE PRINCIPLES
1.  **Relevance is Paramount:** The primary filter is always the user's most recent message. A memory is only useful if it directly informs or contextualizes the current topic.
2.  **Factuality First:** Prioritize memories that represent concrete, verifiable facts over subjective statements.
3.  **Density is Value:** A single, informationally dense memory that consolidates multiple related facts is more valuable than several fragmented ones.
4.  **Temporal Currency:** The most recent, dated information is the most reliable. Actively seek memories with specific dates that reflect the latest user state.

---

### 📜 RANKING HEURISTICS
Apply these rules in strict order to build the final, ranked list of memory IDs:

1.  **Direct Relevance:** Always prioritize memories that directly answer or address a specific question in the user's message.
2.  **Contextual Enhancement:** After direct hits, select memories that provide essential background context (preferences, goals, constraints) needed for a personalized response.
3.  **Temporal Precedence:** When memories conflict, the one with the most recent, specific date **always wins**. A memory stating a project was "completed on August 15 2025" supersedes one saying it's "in progress."
4.  **Informational Density:** Prefer a single, well-consolidated memory over multiple fragmented ones if it covers the topic comprehensively and accurately.
5.  **Specificity Over Generality:** A specific memory ("User dislikes spicy Thai food") is more valuable than a general one ("User enjoys Asian cuisine").

### 🚫 EXCLUSION CRITERIA
Actively **EXCLUDE** memories that are:
- **Trivially Related:** General knowledge or personal trivia with no bearing on the current query.
- **Outdated/Superseded:** Information that has been clearly replaced by a more recent and accurate memory.
- **Transient:** Memories about temporary states like moods, weather, or speculative plans.
 - **Anti-Pattern:** Do NOT return memories that merely duplicate or lightly rephrase other high-quality memories; prefer the single most informative memory.

---

### OUTPUT CONTRACT
- **RETURN ONLY A JSON ARRAY OF STRINGS.**
- The strings must be the memory IDs, ordered from most to least relevant.
- If no candidate memories materially improve the response, return an empty array `[]`.
- **ABSOLUTELY NO COMMENTARY, EXPLANATIONS, OR CODE FENCES.**

---

### EXAMPLES

**1) Project Status Query ("What's the latest on the Phoenix project?")**
* **Candidates:** `["mem-101: User is the lead on Project Phoenix, a data migration initiative due on October 1 2025", "mem-102: User is a project manager with a PMP certification", "mem-103: User's main stakeholder for Project Phoenix is named Sarah Jenkins", "mem-104: User is experiencing a blocker on Project Phoenix related to API authentication as of August 14 2025"]`
* **Expected Output:** `["mem-104","mem-101","mem-103","mem-102"]`

**2) Cooking Advice ("What should I make for dinner tonight? I want something quick.")**
* **Candidates:** `["mem-010: User is vegetarian with a severe nut allergy", "mem-011: User enjoys Italian food", "mem-012: User works from home in Seattle", "mem-013: User bought fresh tomatoes and basil yesterday", "mem-014: User dislikes cleaning up complex recipes on weekdays"]`
* **Expected Output:** `["mem-010","mem-014","mem-013","mem-011"]`

**3) NEGATIVE EXAMPLE (Trivially Related)**
* **User Query:** "Can you recommend a good book about Roman history?"
* **Candidates:** `["mem-201: User enjoys reading science fiction novels", "mem-202: User has a degree in European History", "mem-203: User's favorite author is Neal Stephenson"]`
* **Expected Output:** `[]` // The memories are about the user's reading habits, but they don't help in recommending a *Roman history* book. They are not directly relevant to the specific request.
"""

MEMORY_CONSOLIDATION_PROMPT = f"""You are the Neural Recall Consolidator, a meticulous component of the Consolidation Pipeline. Your primary function is to enrich the user's memory profile with high-value, fact-based memories derived from conversation. Your secondary function is to perform database hygiene (merging, updating, splitting) with a strict bias towards **information preservation**.

## 🎯 PRIMARY GOAL
Your objective is to build a high-fidelity, long-term user profile. The memories you create are the foundation for all future AI interactions. Quality, accuracy, and relevance are paramount. A well-maintained memory store enables the AI to be a truly helpful and context-aware assistant.

## 🏛️ CORE PRINCIPLES
1.  **Preservation First:** When in doubt, do nothing. It is safer to have slightly redundant memories than to lose information. Your default action is NO-OP (`[]`).
2.  **Verifiable Factuality:** You must only record what is **explicitly stated** by the user. Do not infer, assume, or interpret facts not present in the text.
3.  **Informational Density:** Group closely related details into a single, dense memory. If a user mentions a project's name, its purpose, and its deadline in sequence, combine them into one consolidated fact.
4.  **Temporal Precedence:** New, dated information always supersedes older, conflicting information. This is the primary trigger for `UPDATE` operations.
5.  **Contextual Grounding:** Use the provided `Current Date/Time` to convert relative time references (e.g., "yesterday," "next week") into absolute dates (e.g., "August 14 2025," "August 22 2025").

---

### 📜 EXECUTION RULES
These rules are mandatory for every operation.

1.  **Language Mandate:** All memory `content` MUST be in **English**. If the conversation is in another language, you must translate the core, personal facts about the user into English.
2.  **Strict Prefixing:** Every `content` field **MUST** start with "User" or "User's". There are no exceptions.
3.  **Date Integration:** When temporal information is present or derivable, always include specific dates in the format "Month Day Year" (e.g., "August 15 2025").
4.  **Length Constraint:** Memory content must not exceed **{Config.MAX_MEMORY_CONTENT_LENGTH}** characters.
5.  **Value Filter / Content to Ignore:** You **MUST IGNORE** and **NEVER** create memories from:
    * **Questions for the AI:** "What is the capital of France?", "How does a neural network work?"
    * **Conversational Filler:** "Hmm," "let me see," "that's interesting," "oh, right."
    * **Transient States:** Temporary moods ("I'm feeling tired"), weather, or speculative plans ("I might go to the store later").
    * **General Knowledge & Opinions:** Impersonal facts, broad opinions, or philosophical statements.
    * **User's Internal Monologue:** Self-correction or thinking-out-loud phrases ("Wait, no, I meant...").

---

### ⚙️ CONSOLIDATION OPERATIONS
Analyze the user's message and candidate memories to determine if any of the following operations are justified.

#### `CREATE` (New, Atomic Fact)
- **Justification:** The conversation reveals a new, high-value, personal fact that passes the Value Filter and is NOT already captured in existing memories.
- **Anti-Pattern:** Do NOT create if similar information already exists in memories, even with different wording.
- **Example:**
    - **Conversation:** "My daughter is starting kindergarten next month, so I'm adjusting my work schedule to be free in the afternoons."
    - **Current Date/Time:** "August 15 2025"
    - **Existing Memories:** `["mem-101: User has a daughter"]`
    - **Output:** `[{{"operation":"CREATE","content":"User is adjusting their work schedule to be free in the afternoons for their daughter's kindergarten starting in September 2025"}}]`

#### `UPDATE` (Temporal Progression)
- **Justification:** New information clearly supersedes or refines an existing memory with SUBSTANTIAL change.
- **Anti-Pattern:** Do NOT update if the new content is essentially the same as existing memory with only minor rewording.
- **Example:**
    - **Conversation:** "I finally finished my PMP certification course yesterday."
    - **Current Date/Time:** "August 15 2025"
    - **Existing Memories:** `["mem-201: User is studying for a PMP certification"]`
    - **Output:** `[{{"operation":"UPDATE","id":"mem-201","content":"User completed their PMP certification on August 14 2025"}}]`

#### `MERGE` (Showcasing Informational Density)
- **Justification:** Multiple fragmented memories about the same entity can be consolidated into a single, dense, and more useful memory with substantial information gain.
- **Anti-Pattern:** Do NOT merge memories that are already well-organized or where merging would not significantly improve information density.
- **Example:**
    - **Conversation:** "The deadline for Project Phoenix is now October 1st. The main stakeholder is Sarah Jenkins. Yesterday, I hit a blocker with the API authentication."
    - **Current Date/Time:** "August 15 2025"
    - **Existing Memories:** `["mem-301: User is working on Project Phoenix", "mem-302: Project Phoenix is a data migration initiative"]`
    - **Output:** `[{{"operation":"UPDATE","id":"mem-301","content":"User is the lead on Project Phoenix, a data migration initiative due on October 1 2025"}}, {{"operation":"DELETE","id":"mem-302"}}, {{"operation":"CREATE","content":"User's main stakeholder for Project Phoenix is named Sarah Jenkins"}}, {{"operation":"CREATE","content":"User is experiencing a blocker on Project Phoenix related to API authentication as of August 14 2025"}}]`

#### `SPLIT` (Enforcing Atomicity)
- **Justification:** An existing memory bundles distinct atomic facts that should be separated for better precision. Prefer atomic, single-concept memories over merged ones unless topics are highly contextually dependent.
- **Anti-Pattern:** Only merge memories when topics are inseparably linked and splitting would fragment essential context. Default to atomic separation for better retrieval precision.
- **Example:**
    - **Existing Memories:** `["mem-401: User is a vegetarian and their favorite movie is Blade Runner"]`
    - **Output:** `[{{"operation":"UPDATE","id":"mem-401","content":"User is a vegetarian"}}, {{"operation":"CREATE","content":"User's favorite movie is Blade Runner"}}]`

#### **NEGATIVE EXAMPLE** (Ignoring Transient/Speculative Content)
- **Justification:** The user's message contains personal information, but it's speculative and not a stable fact, triggering the Value Filter.
- **Example:**
    - **Conversation:** "I'm so tired of this rain. I'm thinking I might take a trip to Spain next year to get some sun. Maybe in the spring."
    - **Existing Memories:** `["mem-501: User lives in Seattle"]`
    - **Output:** `[]` // "Thinking I might" and "Maybe" are speculative. This is not a confirmed plan and should not be stored as a memory.

---

### OUTPUT CONTRACT
- **RETURN ONLY A VALID JSON ARRAY OF OPERATION OBJECTS.**
- If no operations are needed, return an empty array `[]`.
- **ABSOLUTELY NO COMMENTARY, EXPLANATIONS, OR CODE FENCES.**
"""


class MemoryOperation(BaseModel):
    """Pydantic model for memory operations with validation."""

    operation: Literal["CREATE", "UPDATE", "DELETE"]
    content: Optional[str] = None
    id: Optional[str] = None

    def validate_operation(self, existing_memory_ids: set = None) -> bool:
        """Validate the memory operation according to business rules."""
        if existing_memory_ids is None:
            existing_memory_ids = set()

        if self.operation == "CREATE":
            return self.content is not None and len(self.content.strip()) > 0
        elif self.operation in ["UPDATE", "DELETE"]:
            if not self.id or self.id not in existing_memory_ids:
                return False
            if self.operation == "UPDATE":
                return self.content is not None and len(self.content.strip()) > 0
            return True
        return False


class LRUCache:
    """Enhanced thread-safe LRU cache implementation using OrderedDict with comprehensive metrics."""

    def __init__(self, max_size: int) -> None:
        """Initialize LRU cache with specified maximum size."""
        self.max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = asyncio.Lock()

        self._hits = 0
        self._misses = 0
        self._evictions = 0

    async def get(self, key: str) -> Optional[Any]:
        """Get an item from cache, moving it to end (most recently used)."""
        async with self._lock:
            if key in self._cache:
                value = self._cache.pop(key)
                self._cache[key] = value
                self._hits += 1
                return value
            self._misses += 1
            return None

    async def put(self, key: str, value: Any) -> None:
        """Put an item in cache, evicting least recently used if necessary."""
        async with self._lock:
            if key in self._cache:
                self._cache.pop(key)
            elif len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
                self._evictions += 1
            self._cache[key] = value

    async def size(self) -> int:
        """Get current cache size."""
        async with self._lock:
            return len(self._cache)

    async def clear(self) -> int:
        """Clear cache and return number of entries cleared."""
        async with self._lock:
            size = len(self._cache)
            self._cache.clear()
            self._reset_metrics()
            return size

    async def contains(self, key: str) -> bool:
        """Check if key exists in cache without updating access order."""
        async with self._lock:
            return key in self._cache

    async def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        async with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (
                (self._hits / total_requests * 100) if total_requests > 0 else 0.0
            )
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": hit_rate,
            }

    def _reset_metrics(self) -> None:
        """Reset cache metrics."""
        self._hits = 0
        self._misses = 0
        self._evictions = 0


class Filter:
    """Enhanced multi-model embedding and memory filter with LRU caching."""

    _model = None
    _model_load_lock = None
    _embedding_cache: Dict[str, LRUCache] = {}
    _cache_access_order: List[str] = []
    _cache_lock = None
    _aiohttp_session: Optional[aiohttp.ClientSession] = None
    _session_lock = None
    _conversation_memory_cache: Dict[str, Dict[str, set]] = {}
    _conversation_cache_lock = None

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
            default=Config.DEFAULT_MAX_MEMORIES_RETURNED,
            description="Maximum number of memories to inject into context",
        )
        semantic_threshold: float = Field(
            default=Config.DEFAULT_SEMANTIC_THRESHOLD,
            description="Minimum similarity threshold for memory retrieval",
        )

    def __init__(self):
        """Initialize the Neural Recall filter with production validation."""
        self.valves = self.Valves()
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate configuration and fail if invalid."""
        if not self.valves.api_url or not self.valves.api_url.strip():
            raise ValidationError("API URL not configured")

        if not self.valves.model or not self.valves.model.strip():
            raise ValidationError("Model not specified")

        if not 0.0 <= self.valves.semantic_threshold <= 1.0:
            raise ValidationError(
                f"Invalid semantic threshold: {self.valves.semantic_threshold}"
            )

        if self.valves.max_memories_returned <= 0:
            raise ValidationError(
                f"Invalid max memories returned: {self.valves.max_memories_returned}"
            )

        logger.info("✅ Configuration validated")

    async def _get_embedding_model(self):
        """Get or load the sentence transformer model with thread safety."""
        if Filter._model is None:
            if Filter._model_load_lock is None:
                Filter._model_load_lock = asyncio.Lock()

            async with Filter._model_load_lock:
                if Filter._model is None:
                    try:
                        logger.info(
                            f"🤖 Loading embedding model: {self.valves.embedding_model}"
                        )

                        def load_model():
                            return SentenceTransformer(
                                self.valves.embedding_model,
                                device="cpu",
                                trust_remote_code=True,
                            )

                        loop = asyncio.get_event_loop()
                        Filter._model = await loop.run_in_executor(None, load_model)
                        logger.info("✅ Embedding model loaded successfully")

                    except Exception as e:
                        error_msg = f"Failed to load embedding model: {str(e)}"
                        logger.error(f"❌ {error_msg}")
                        raise ModelLoadError(error_msg)

        return Filter._model

    async def _get_user_cache(self, user_id: str) -> LRUCache:
        """Get or create user-specific embedding cache with global user limit."""
        if Filter._cache_lock is None:
            Filter._cache_lock = asyncio.Lock()

        async with Filter._cache_lock:
            if user_id in Filter._embedding_cache:
                if user_id in Filter._cache_access_order:
                    Filter._cache_access_order.remove(user_id)
                Filter._cache_access_order.append(user_id)
                return Filter._embedding_cache[user_id]

            if len(Filter._embedding_cache) >= Config.MAX_USER_CACHES:
                if Filter._cache_access_order:
                    lru_user_id = Filter._cache_access_order.pop(0)
                    if lru_user_id in Filter._embedding_cache:
                        del Filter._embedding_cache[lru_user_id]
                        logger.info(f"🧹 Cache evicted for user {lru_user_id}")

            Filter._embedding_cache[user_id] = LRUCache(Config.CACHE_MAX_SIZE)
            Filter._cache_access_order.append(user_id)
            return Filter._embedding_cache[user_id]

    async def _generate_embedding(self, text: str, user_id: str) -> np.ndarray:
        """Generate embedding for text with caching support."""
        if not text or len(text.strip()) < SkipThresholds.MIN_QUERY_LENGTH:
            raise EmbeddingError("Text too short for embedding generation")

        text_hash = hashlib.sha256(text.encode()).hexdigest()
        cache = await self._get_user_cache(user_id)

        cached_embedding = await cache.get(text_hash)
        if cached_embedding is not None:
            return cached_embedding

        try:
            model = await self._get_embedding_model()

            def generate_embedding():
                embedding = model.encode([text], convert_to_numpy=True)[0]
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                return embedding

            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, generate_embedding)

            await cache.put(text_hash, embedding)
            return embedding

        except Exception as e:
            error_msg = f"Embedding generation failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            raise EmbeddingError(error_msg)

    async def _generate_embeddings_batch(
        self, texts: List[str], user_id: str
    ) -> List[np.ndarray]:
        """Generate embeddings for multiple texts efficiently with batch processing and caching."""
        if not texts:
            return []

        cache = await self._get_user_cache(user_id)

        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if not text or len(text.strip()) < SkipThresholds.MIN_QUERY_LENGTH:
                continue

            text_hash = hashlib.sha256(text.encode()).hexdigest()
            cached_embedding = await cache.get(text_hash)

            if cached_embedding is not None:
                cached_embeddings[i] = cached_embedding
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        new_embeddings = {}
        if uncached_texts:
            try:
                model = await self._get_embedding_model()

                batch_size = min(
                    Config.MAX_BATCH_SIZE,
                    max(Config.MIN_BATCH_SIZE, len(uncached_texts)),
                )

                def generate_batch_embeddings(batch_texts):
                    embeddings = model.encode(
                        batch_texts, convert_to_numpy=True, show_progress_bar=False
                    )
                    normalized_embeddings = []
                    for embedding in embeddings:
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = embedding / norm
                        normalized_embeddings.append(embedding)
                    return normalized_embeddings

                loop = asyncio.get_event_loop()

                for i in range(0, len(uncached_texts), batch_size):
                    batch_texts = uncached_texts[i : i + batch_size]
                    batch_indices = uncached_indices[i : i + batch_size]

                    batch_embeddings = await loop.run_in_executor(
                        None, generate_batch_embeddings, batch_texts
                    )

                    for j, embedding in enumerate(batch_embeddings):
                        text_idx = batch_indices[j]
                        text = batch_texts[j]
                        text_hash = hashlib.sha256(text.encode()).hexdigest()

                        await cache.put(text_hash, embedding)
                        new_embeddings[text_idx] = embedding

            except Exception as e:
                error_msg = f"Batch embedding generation failed: {str(e)}"
                logger.error(f"❌ {error_msg}")
                raise EmbeddingError(error_msg)

        result_embeddings = []
        for i in range(len(texts)):
            if i in cached_embeddings:
                result_embeddings.append(cached_embeddings[i])
            elif i in new_embeddings:
                result_embeddings.append(new_embeddings[i])
            else:
                result_embeddings.append(None)

        valid_count = len([emb for emb in result_embeddings if emb is not None])
        logger.info(
            f"🚀 Batch embedding: {len(cached_embeddings)} cached, {len(new_embeddings)} new, {valid_count}/{len(texts)} valid"
        )
        return result_embeddings

    async def _get_aiohttp_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with connection pooling and production settings."""
        if Filter._aiohttp_session is None or Filter._aiohttp_session.closed:
            if Filter._session_lock is None:
                Filter._session_lock = asyncio.Lock()

            async with Filter._session_lock:
                if Filter._aiohttp_session is None or Filter._aiohttp_session.closed:
                    timeout = aiohttp.ClientTimeout(
                        total=Config.TIMEOUT_SESSION_REQUEST
                    )
                    connector = aiohttp.TCPConnector(
                        limit=100,
                        limit_per_host=30,
                        keepalive_timeout=300,
                        enable_cleanup_closed=True,
                        ttl_dns_cache=300,
                    )
                    Filter._aiohttp_session = aiohttp.ClientSession(
                        timeout=timeout,
                        connector=connector,
                        headers={"User-Agent": "NeuralRecallV3/3.0.0"},
                    )
                    try:
                        weakref.finalize(
                            Filter._aiohttp_session,
                            Filter._auto_cleanup_finalizer,
                        )
                    except Exception:
                        logger.warning("⚠️ Failed to register aiohttp session finalizer")
        return Filter._aiohttp_session

    def _should_skip_memory_operations(self, user_message: str) -> Tuple[bool, str]:
        """
        Enhanced gating for memory operations with comprehensive pattern detection.
        Skips: empty, too long, code, logs, structured data, URL dumps, or symbol spam.
        """
        if not user_message or not user_message.strip():
            return True, Config.STATUS_MESSAGES["SKIP_EMPTY"]

        trimmed_message = user_message.strip()

        if len(trimmed_message) < SkipThresholds.MIN_QUERY_LENGTH:
            return True, Config.STATUS_MESSAGES["SKIP_EMPTY"]
        
        if len(trimmed_message) > SkipThresholds.MAX_MESSAGE_LENGTH:
            return (
                True,
                f"📄 Message too long ({len(trimmed_message)} characters exceeds {SkipThresholds.MAX_MESSAGE_LENGTH} limit)",
            )

        if self._is_code_content(trimmed_message):
            return True, Config.STATUS_MESSAGES["SKIP_CODE"]

        if self._is_structured_data(trimmed_message):
            return True, Config.STATUS_MESSAGES["SKIP_STRUCTURED"]

        if self._is_mostly_symbols(trimmed_message):
            return True, Config.STATUS_MESSAGES["SKIP_SYMBOLS"]

        if self._is_log_content(trimmed_message):
            return True, Config.STATUS_MESSAGES["SKIP_LOGS"]

        if self._is_stack_trace(trimmed_message):
            return True, Config.STATUS_MESSAGES["SKIP_STACKTRACE"]

        if self._is_url_dump(trimmed_message):
            return True, Config.STATUS_MESSAGES["SKIP_URL_DUMP"]

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
        if len(re.findall(json_kv_pattern, message)) >= SkipThresholds.JSON_KEY_VALUE_THRESHOLD:
            return True

        message_lines = message.split("\n")
        line_count = len(message_lines)
        
        if line_count < SkipThresholds.STRUCTURED_LINE_COUNT_MIN:
            return False

        count_lines = lambda pattern: sum(1 for line in message_lines if re.match(pattern, line))
        threshold = max(SkipThresholds.STRUCTURED_LINE_COUNT_MIN, line_count * SkipThresholds.STRUCTURED_PERCENTAGE_THRESHOLD)

        structured_checks = [
            count_lines(r"^\s*[A-Za-z0-9_]+:\s+\S+") >= threshold,
            count_lines(r"^\s*[-\*\+]\s+.+") >= max(SkipThresholds.STRUCTURED_BULLET_MIN, threshold),
            count_lines(r"^\s*\d+\.\s") >= threshold,
            count_lines(r"^\s*[a-zA-Z]\)\s") >= threshold,
            sum(1 for line in message_lines if "|" in line and line.count("|") >= SkipThresholds.STRUCTURED_PIPE_MIN) >= max(SkipThresholds.STRUCTURED_PIPE_MIN, threshold)
        ]
        
        return any(structured_checks)

    def _is_mostly_symbols(self, message: str) -> bool:
        """Check if message is mostly symbols/numbers."""
        if len(message) <= SkipThresholds.SYMBOL_CHECK_MIN_LENGTH:
            return False
        
        alpha_space_count = sum(1 for c in message if c.isalpha() or c.isspace())
        return alpha_space_count / len(message) < SkipThresholds.SYMBOL_RATIO_THRESHOLD

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
        
        if line_count < SkipThresholds.LOGS_LINE_COUNT_MIN:
            return False

        datetime_matches = sum(
            1 for line in message_lines 
            if re.search(r"\d{2,4}[-/]\d{1,2}[-/]\d{1,2}.*\d{1,2}:\d{2}", line)
        )
        if datetime_matches >= max(SkipThresholds.LOGS_MIN_MATCHES, line_count * SkipThresholds.LOGS_MATCH_PERCENTAGE):
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
        url_count = len(re.findall(r"https?://[^\s]+", message))
        return url_count >= SkipThresholds.URL_COUNT_THRESHOLD

    def _extract_text_from_message_content(
        self, content: Union[str, List[Dict], Dict]
    ) -> str:
        """Extract text from various message content formats."""
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

    def get_formatted_datetime_string(self) -> str:
        try:
            now = datetime.now(timezone.utc)
            weekday = now.strftime("%A")
            month = now.strftime("%B")
            day = now.day
            year = now.year
            time_str = now.strftime("%H:%M:%S UTC")
            return f"{weekday} {month} {day} {year} at {time_str}"
        except Exception as e:
            raise NeuralRecallError(f"Failed to format datetime: {e}")

    def _get_conversation_id(self, body: Dict[str, Any]) -> str:
        """Extract conversation ID from OpenWebUI request body using metadata.chat_id or first message ID."""
        if 'metadata' in body and isinstance(body['metadata'], dict):
            if 'chat_id' in body['metadata'] and body['metadata']['chat_id']:
                conversation_id = str(body['metadata']['chat_id'])
                logger.info(f"🆔 Using metadata.chat_id as conversation ID: {conversation_id}")
                return conversation_id 
        raise ValueError("No chat_id found in request body")


    def _generate_conversation_hash(self, body: Dict[str, Any], user_id: str) -> str:
        """Generate conversation hash using user_id + conversation_id."""
        conversation_id = self._get_conversation_id(body)
        conversation_key = f"{user_id}:{conversation_id}"
        conv_hash = hashlib.sha256(conversation_key.encode()).hexdigest()[:16]
        return conv_hash

    async def _get_conversation_memory_tracker(
        self, user_id: str, conversation_hash: str
    ) -> set:
        if Filter._conversation_cache_lock is None:
            Filter._conversation_cache_lock = asyncio.Lock()

        async with Filter._conversation_cache_lock:
            if user_id not in Filter._conversation_memory_cache:
                Filter._conversation_memory_cache[user_id] = {}

            user_cache = Filter._conversation_memory_cache[user_id]

            if len(user_cache) > Config.CONVERSATION_CACHE_CLEANUP_THRESHOLD:
                conversations_to_keep = list(user_cache.keys())[
                    -Config.MAX_CONVERSATION_CACHES :
                ]
                user_cache = {k: user_cache[k] for k in conversations_to_keep}
                Filter._conversation_memory_cache[user_id] = user_cache

            if conversation_hash not in user_cache:
                user_cache[conversation_hash] = set()

            return user_cache[conversation_hash]

    async def _mark_memories_as_injected(
        self, user_id: str, conversation_hash: str, memory_ids: List[str]
    ) -> None:
        injected_memories = await self._get_conversation_memory_tracker(
            user_id, conversation_hash
        )
        injected_memories.update(memory_ids)
        logger.info(
            f"📝 Marked {len(memory_ids)} memories as injected for conversation {conversation_hash}: {memory_ids}"
        )

    async def _mark_memory_as_created(
        self, user_id: str, conversation_hash: str, memory_id: str
    ) -> None:
        injected_memories = await self._get_conversation_memory_tracker(
            user_id, conversation_hash
        )
        injected_memories.add(memory_id)

    async def _mark_memory_as_updated(
        self, user_id: str, conversation_hash: str, memory_id: str
    ) -> None:
        injected_memories = await self._get_conversation_memory_tracker(
            user_id, conversation_hash
        )
        injected_memories.add(memory_id)

    async def _remove_memory_from_conversation_tracking(
        self, user_id: str, conversation_hash: str, memory_id: str
    ) -> None:
        injected_memories = await self._get_conversation_memory_tracker(
            user_id, conversation_hash
        )
        injected_memories.discard(memory_id)

    async def _filter_already_injected_memories(
        self, user_id: str, conversation_hash: str, memories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter out memories that have already been injected in this conversation.
        
        Returns filtered memories (excluding already injected ones).
        """
        injected_memories = await self._get_conversation_memory_tracker(
            user_id, conversation_hash
        )

        if not injected_memories:
            logger.info(
                f"� No previous memory injections found for this conversation"
            )
            return memories

        filtered_memories = []
        skipped_memories = []

        for memory in memories:
            memory_id = str(memory.get("id", ""))
            if memory_id not in injected_memories:
                filtered_memories.append(memory)
            else:
                skipped_memories.append(memory_id)

        if skipped_memories:
            n = len(skipped_memories)
            if n == 1:
                logger.info(f"� Filtered out 1 previously injected memory")
            else:
                logger.info(f"🔄 Filtered out {n} previously injected memories")

        logger.info(
            f"✅ {len(filtered_memories)} new memories available from {len(memories)} candidates"
        )
        return filtered_memories

    async def _emit_status(
        self,
        emitter: Optional[Callable[[Any], Awaitable[None]]],
        message: str,
        done: bool = False,
    ) -> None:
        """Emit status message through the event emitter with fault tolerance."""
        if not emitter:
            return

        payload = {
            "type": "status",
            "data": {"description": message, "done": done},
        }

        try:
            result = emitter(payload)
            if asyncio.iscoroutine(result):
                await asyncio.wait_for(result, timeout=1.0)
        except asyncio.TimeoutError:
            logger.info(f"⏳ Event emitter timed out for message {message}")
        except Exception as e:
            logger.info(f"⚠️ Event emitter failed for message '{message}' {e}")

    async def _broad_retrieval(
        self,
        user_message: str,
        user_id: str,
        emitter: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> List[Dict[str, Any]]:
        """Step 1 of Retrieval Pipeline: Perform broad vector search for candidate memories."""
        candidate_limit = int(
            self.valves.max_memories_returned * Config.RETRIEVAL_MULTIPLIER
        )

        user_memories = await asyncio.wait_for(
            asyncio.to_thread(Memories.get_memories_by_user_id, user_id),
            timeout=Config.RETRIEVAL_TIMEOUT,
        )

        if not user_memories:
            logger.info("� No memories found for user")
            return []

        logger.info(
            f"🔍 Found {len(user_memories)} total memories for analysis"
        )

        query_embedding = await self._generate_embedding(user_message, user_id)

        memory_contents = [memory.content for memory in user_memories]

        memory_embeddings = await self._generate_embeddings_batch(
            memory_contents, user_id
        )

        if len(memory_embeddings) != len(user_memories):
            error_msg = f"Embedding count mismatch: {len(memory_embeddings)} vs {len(user_memories)}"
            logger.error(f"❌ {error_msg}")
            raise EmbeddingError(error_msg)

        formatted_memories = []

        for i, memory in enumerate(user_memories):
            memory_embedding = memory_embeddings[i]
            if memory_embedding is None:
                continue

            similarity = float(np.dot(query_embedding, memory_embedding))

            if similarity >= self.valves.semantic_threshold:
                memory_dict = {
                    "id": str(memory.id),
                    "content": memory.content,
                    "relevance": similarity,
                }
                formatted_memories.append(memory_dict)

        formatted_memories.sort(key=lambda x: x["relevance"], reverse=True)
        formatted_memories = formatted_memories[:candidate_limit]

        logger.info(
            f"🎯 Selected {len(formatted_memories)} candidate memories (threshold {self.valves.semantic_threshold:.2f})"
        )

        if (
            len(formatted_memories) >= candidate_limit
            and len(user_memories) > candidate_limit
        ):
            await self._emit_status(
                emitter,
                f"� Found {len(formatted_memories)} relevant memories from {len(user_memories)} total",
                False,
            )
        else:
            await self._emit_status(
                emitter, f"� Found {len(formatted_memories)} relevant memories", False
            )

        return formatted_memories

    async def _llm_rerank_memories(
        self,
        user_message: str,
        candidate_memories: List[Dict[str, Any]],
        emitter: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> List[Dict[str, Any]]:
        """Step 2 of Retrieval Pipeline: Use LLM to re-rank candidate memories."""
        if not candidate_memories:
            return []

        memory_context = "## CANDIDATE MEMORIES\n"
        for memory in candidate_memories:
            memory_context += f"ID: {memory['id']}\nContent: {memory['content']}\n\n"

        system_prompt = MEMORY_RERANKING_PROMPT + f"\n\n{memory_context}"

        await self._emit_status(emitter, "� Analyzing memory relevance", False)
        response = await asyncio.wait_for(
            self._query_llm(system_prompt, user_message, json_response=True),
            timeout=Config.RETRIEVAL_TIMEOUT,
        )

        if not response or response.startswith("Error:"):
            error_msg = "LLM re-ranking failed - empty or error response"
            logger.error(f"❌ {error_msg}")
            raise NeuralRecallError(error_msg)

        ranked_ids = self._extract_and_parse_json(response)
        if not isinstance(ranked_ids, list):
            error_msg = "Invalid LLM re-ranking response - expected list"
            logger.error(f"❌ {error_msg}")
            raise NeuralRecallError(error_msg)

        memory_lookup = {mem["id"]: mem for mem in candidate_memories}
        ranked_memories = []

        for memory_id in ranked_ids:
            if memory_id in memory_lookup:
                ranked_memories.append(memory_lookup[memory_id])

        ranked_ids_set = set(ranked_ids)
        for memory in candidate_memories:
            if memory["id"] not in ranked_ids_set:
                ranked_memories.append(memory)

        final_memories = ranked_memories[: self.valves.max_memories_returned]

        logger.info(
            f"🎯 Selected {len(final_memories)} memories from {len(candidate_memories)} candidates after analysis"
        )

        if (
            len(final_memories) >= self.valves.max_memories_returned
            and len(ranked_memories) > self.valves.max_memories_returned
        ):
            await self._emit_status(
                emitter,
                f"🎯 Selected {len(final_memories)} most relevant memories",
                False,
            )
        else:
            await self._emit_status(
                emitter, f"🎯 Selected {len(final_memories)} relevant memories", False
            )

        return final_memories

    async def _inject_context(
        self,
        body: Dict[str, Any],
        memories: List[Dict[str, Any]] = None,
        user_id: str = None,
        conversation_hash: str = None,
    ) -> None:
        if not body or "messages" not in body or not body["messages"]:
            logger.info("❌ Invalid body or no messages found")
            return

        logger.info(
            f"� Preparing to inject {len(memories) if memories else 0} memories for user {user_id}"
        )

        current_datetime = self.get_formatted_datetime_string()
        content_parts = [f"Current Date/Time: {current_datetime}"]

        if memories and user_id:
            if not conversation_hash:
                conversation_hash = self._generate_conversation_hash(body, user_id)
            memory_ids = [str(memory.get("id", "")) for memory in memories]
            await self._mark_memories_as_injected(
                user_id, conversation_hash, memory_ids
            )
            logger.info(
                f"✅ Marked {len(memory_ids)} memories as injected for this conversation"
            )

            n = len(memories)
            if n == 1:
                memory_header = "BACKGROUND: You naturally know this fact. Never mention its source."
            else:
                memory_header = f"BACKGROUND: You naturally know these {n} facts. Never mention their source."

            memory_content = "\n".join(
                [f"- {memory['content']}" for memory in memories]
            )
            content_parts.append(f"{memory_header}\n{memory_content}")
            logger.info(f"💭 Added {len(memories)} memories to context injection")

        system_message = {"role": "system", "content": "\n\n".join(content_parts)}
        logger.info(f"🔧 Created system message with context")

        injection_position = None
        for i in range(len(body["messages"]) - 1, -1, -1):
            if body["messages"][i].get("role") == "user":
                body["messages"].insert(i, system_message)
                injection_position = i
                logger.info(
                    f"✅ Context injected successfully at position {i}"
                )
                break

        if injection_position is None:
            logger.warning("⚠️ No user message found for context injection")

    async def _gather_consolidation_candidates(
        self,
        user_message: str,
        user_id: str,
        emitter: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> List[Dict[str, Any]]:
        """Step 1 of Consolidation Pipeline: Gather candidate memories for consolidation analysis."""
        user_memories = await asyncio.wait_for(
            asyncio.to_thread(Memories.get_memories_by_user_id, user_id),
            timeout=Config.CONSOLIDATION_TIMEOUT,
        )

        if not user_memories:
            logger.info("�️ No existing memories found for user")
            await self._emit_status(emitter, "💭 No memories to analyze", True)
            return []

        logger.info(
            f"�️ Found {len(user_memories)} existing memories for consolidation analysis"
        )

        query_embedding = await self._generate_embedding(user_message, user_id)

        memory_contents = [memory.content for memory in user_memories]

        memory_embeddings = await self._generate_embeddings_batch(
            memory_contents, user_id
        )

        if len(memory_embeddings) != len(user_memories):
            error_msg = f"Consolidation embedding count mismatch: {len(memory_embeddings)} vs {len(user_memories)}"
            logger.error(f"❌ {error_msg}")
            raise EmbeddingError(error_msg)

        formatted_memories = []
        relaxed_threshold = (
            self.valves.semantic_threshold * Config.CONSOLIDATION_RELAXED_MULTIPLIER
        )

        for i, memory in enumerate(user_memories):
            memory_embedding = memory_embeddings[i]
            if memory_embedding is None:
                continue

            similarity = float(np.dot(query_embedding, memory_embedding))

            if similarity >= relaxed_threshold:
                created_at_iso = None
                if hasattr(memory, "created_at") and memory.created_at:
                    created_at_iso = datetime.fromtimestamp(
                        memory.created_at, tz=timezone.utc
                    ).isoformat()

                updated_at_iso = None
                if hasattr(memory, "updated_at") and memory.updated_at:
                    updated_at_iso = datetime.fromtimestamp(
                        memory.updated_at, tz=timezone.utc
                    ).isoformat()

                memory_dict = {
                    "id": str(memory.id),
                    "content": memory.content,
                    "created_at": created_at_iso,
                    "updated_at": updated_at_iso,
                    "relevance": similarity,
                }
                formatted_memories.append(memory_dict)

        formatted_memories.sort(key=lambda x: x["relevance"], reverse=True)
        formatted_memories = formatted_memories[: Config.CONSOLIDATION_CANDIDATE_SIZE]

        logger.info(
            f"� Found {len(formatted_memories)} candidate memories for consolidation analysis"
        )

        if (
            len(formatted_memories) >= Config.CONSOLIDATION_CANDIDATE_SIZE
            and len(user_memories) > Config.CONSOLIDATION_CANDIDATE_SIZE
        ):
            await self._emit_status(
                emitter,
                f"� Analyzing {len(formatted_memories)} candidate memories from {len(user_memories)} total",
                False,
            )
        else:
            await self._emit_status(
                emitter,
                f"� Analyzing {len(formatted_memories)} candidate memories",
                False,
            )

        return formatted_memories

    async def _llm_consolidate_memories(
        self,
        user_message: str,
        candidate_memories: List[Dict[str, Any]],
        emitter: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> List[Dict[str, Any]]:
        """Step 2 of Consolidation Pipeline: Use LLM to generate consolidation plan."""
        if not candidate_memories:
            return []

        memory_context = "## CANDIDATE MEMORIES FOR CONSOLIDATION\n"
        for memory in candidate_memories:
            memory_context += f"ID: {memory['id']}\nContent: {memory['content']}\n"
            if memory.get("created_at"):
                memory_context += f"Created: {memory['created_at']}\n"
            if memory.get("updated_at"):
                memory_context += f"Updated: {memory['updated_at']}\n"
            memory_context += "\n"

        context_section = f"## USER MESSAGE\n{user_message}\n\n"

        date_context = (
            f"## CURRENT DATE/TIME\n{self.get_formatted_datetime_string()}\n\n"
        )

        system_prompt = (
            MEMORY_CONSOLIDATION_PROMPT
            + f"\n\n{date_context}{context_section}{memory_context}"
        )

        await self._emit_status(
            emitter, "� Analyzing memory patterns for optimization", False
        )
        response = await asyncio.wait_for(
            self._query_llm(
                system_prompt,
                "Analyze the memories and generate a consolidation plan.",
                json_response=True,
            ),
            timeout=Config.CONSOLIDATION_TIMEOUT,
        )

        if not response or response.startswith("Error:"):
            error_msg = "LLM consolidation analysis failed - empty or error response"
            logger.error(f"❌ {error_msg}")
            raise NeuralRecallError(error_msg)

        logger.info(
            f"🔧 Memory consolidation analysis completed ({len(response)} characters)"
        )

        operations = self._extract_and_parse_json(response)
        if not isinstance(operations, list):
            if isinstance(operations, dict) and operations:
                operations = [operations]
                logger.info("🔧 Wrapped single operation in list")
            else:
                error_msg = f"Invalid LLM consolidation response - Expected list, got {type(operations)}"
                logger.error(f"❌ {error_msg}")
                raise NeuralRecallError(error_msg)

        existing_memory_ids = {memory["id"] for memory in candidate_memories}
        valid_operations = []

        total_operations = len(operations)
        delete_operations = [
            op
            for op in operations
            if isinstance(op, dict) and op.get("operation") == "DELETE"
        ]
        delete_ratio = (
            len(delete_operations) / total_operations if total_operations > 0 else 0
        )

        if delete_ratio > 0.5 and total_operations >= 3:
            logger.warning(
                f"⚠️ Consolidation safety trigger: {len(delete_operations)}/{total_operations} operations are deletions ({delete_ratio*100:.1f}%)"
            )
            logger.warning(
                "⚠️ Rejecting consolidation plan to prevent excessive memory loss"
            )
            await self._emit_status(
                emitter,
                "⚠️ Memory consolidation plan too aggressive, preserving existing memories",
                False,
            )
            return []

        for operation in operations:
            if isinstance(operation, dict):
                memory_operation = MemoryOperation(**operation)
                if memory_operation.validate_operation(existing_memory_ids):
                    valid_operations.append(operation)

        logger.info(
            f"🎯 Planned {len(valid_operations)} memory optimization operations"
        )
        await self._emit_status(
            emitter, f"🎯 Planning {len(valid_operations)} memory optimizations", False
        )
        return valid_operations

    async def _execute_consolidation_operations(
        self,
        operations: List[Dict[str, Any]],
        user_id: str,
        conversation_hash: str = None,
        emitter: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> None:
        """Step 3 of Consolidation Pipeline: Execute consolidation operations against database."""
        if not operations:
            return

        user = await asyncio.wait_for(
            asyncio.to_thread(Users.get_user_by_id, user_id),
            timeout=Config.TIMEOUT_USER_LOOKUP,
        )

        if not user:
            error_msg = f"User not found for consolidation: {user_id}"
            logger.error(f"❌ {error_msg}")
            raise MemoryOperationError(error_msg)

        created_count = 0
        updated_count = 0
        deleted_count = 0
        failed_count = 0

        for i, operation_data in enumerate(operations):
            try:
                operation = MemoryOperation(**operation_data)
                result = await self._execute_single_operation(
                    operation, user, user_id, conversation_hash
                )
                if result == "CREATE":
                    created_count += 1
                elif result == "UPDATE":
                    updated_count += 1
                elif result == "DELETE":
                    deleted_count += 1
            except Exception as e:
                failed_count += 1
                operation_type = operation_data.get("operation", "UNKNOWN")
                operation_id = operation_data.get("id", "no-id")
                logger.error(
                    f"❌ Failed to execute {operation_type} operation {i+1}/{len(operations)} (ID: {operation_id}): {str(e)}"
                )
                continue

        total_executed = created_count + updated_count + deleted_count
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

            await self._emit_status(emitter, message, True)

            await self._invalidate_user_cache(user_id, "consolidation")
        elif failed_count > 0:
            await self._emit_status(
                emitter, f"⚠️ {failed_count} of {len(operations)} operations failed", True
            )
        else:
            await self._emit_status(
                emitter, "✅ Memories already optimally organized", True
            )

    async def _consolidation_pipeline_task(
        self,
        user_message: str,
        user_id: str,
        conversation_hash: str = None,
        emitter: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> None:
        """Complete Consolidation Pipeline as async background task (formerly Slow Path)."""
        try:
            logger.info("🔧 Starting memory consolidation analysis")
            await self._emit_status(emitter, "🔧 Analyzing memory patterns", False)

            candidates = await self._gather_consolidation_candidates(
                user_message, user_id, emitter
            )

            if not candidates:
                logger.info("🔧 No consolidation candidates found")
                await self._emit_status(emitter, "💭 No memories need consolidation", True)
                return

            operations = await self._llm_consolidate_memories(
                user_message, candidates, emitter
            )

            if not operations:
                logger.info(
                    "🔧 Memories already optimally organized, no changes needed"
                )
                await self._emit_status(
                    emitter, "✅ Memories already well organized", True
                )
                return

            await self._execute_consolidation_operations(
                operations, user_id, conversation_hash, emitter
            )

            logger.info("🔧 Memory consolidation completed successfully")

        except Exception as e:
            logger.error(f"❌ Memory consolidation failed {str(e)}")
            await self._emit_status(
                emitter, f"❌ Memory consolidation failed {str(e)[:50]}", True
            )
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
            logger.info(f"� Found {len(body['messages'])} messages in request")

            last_user_msg = next(
                (
                    self._extract_text_from_message_content(m["content"])
                    for m in reversed(body.get("messages", []))
                    if m.get("role") == "user"
                ),
                None,
            )

            logger.info(
                f"👤 User message extracted {'successfully' if last_user_msg else 'not found'}"
            )

            if last_user_msg:
                should_skip, skip_reason = self._should_skip_memory_operations(
                    last_user_msg
                )
                logger.info(
                    f"🔍 Skip analysis {skip_reason if should_skip else 'proceeding with memory retrieval'}"
                )
                
                conversation_hash = self._generate_conversation_hash(body, user_id)
                memories = None
                status_message = ""
                
                if should_skip:
                    status_message = f"⏭️ {skip_reason}"
                else:
                    await self._emit_status(__event_emitter__, "🚀 Searching for relevant memories", False)
                    
                    candidate_memories = await self._broad_retrieval(
                        last_user_msg, user_id, __event_emitter__
                    )
                    logger.info(
                        f"� Found {len(candidate_memories) if candidate_memories else 0} candidate memories"
                    )
                    
                    if candidate_memories:
                        filtered_memories = await self._filter_already_injected_memories(
                            user_id, conversation_hash, candidate_memories
                        )
                        logger.info(f"✅ {len(filtered_memories)} new memories after filtering duplicates")
                        
                        if filtered_memories:
                            memories = await self._llm_rerank_memories(
                                last_user_msg, filtered_memories, __event_emitter__
                            )
                            logger.info(f"� Final selection completed {len(memories) if memories else 0} memories chosen")
                            
                            if memories:
                                n = len(memories)
                                if n == 1:
                                    status_message = "💡 Found 1 relevant memory"
                                elif n <= 3:
                                    status_message = f"💡 Found {n} relevant memories"
                                else:
                                    status_message = f"💡 Found {n} relevant memories"
                            else:
                                status_message = "💭 No memories selected by analysis"
                        else:
                            memories = []
                            if len(candidate_memories) == 1:
                                status_message = "🔄 1 memory already used in this conversation"
                            else:
                                status_message = f"🔄 All {len(candidate_memories)} memories already used in this conversation"
                    else:
                        status_message = "💭 No relevant memories found"
                
                await self._emit_status(__event_emitter__, status_message, True)
                await self._inject_context(body, memories, user_id, conversation_hash)
        else:
            await self._inject_context(body, None, user_id)

        logger.info(
            f"✅ Request processing complete, injected {len(memories) if memories else 0} memories"
        )
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
                (
                    self._extract_text_from_message_content(m["content"])
                    for m in reversed(body["messages"])
                    if m["role"] == "user"
                ),
                None,
            )

            logger.info(
                f"👤 User message {'found' if user_message else 'not found'} for optimization analysis"
            )

            if user_message:
                should_skip, skip_reason = self._should_skip_memory_operations(
                    user_message
                )
                if should_skip:
                    logger.info(f"⏭️ Memory optimization skipped {skip_reason}")
                    return body

                conversation_hash = self._generate_conversation_hash(body, user_id)

                logger.info(
                    "🔧 Starting background memory optimization task"
                )
                task = asyncio.create_task(
                    self._consolidation_pipeline_task(
                        user_message, user_id, conversation_hash, __event_emitter__
                    )
                )

                def handle_task_completion(completed_task):
                    if completed_task.exception():
                        exception = completed_task.exception()
                        logger.error(
                            f"❌ Background memory optimization failed {exception}"
                        )
                        if __event_emitter__:
                            try:
                                asyncio.create_task(
                                    self._emit_status(
                                        __event_emitter__,
                                        "❌ Consolidation failed",
                                        True,
                                    )
                                )
                            except Exception as emit_error:
                                logger.error(
                                    f"❌ Failed to emit final error status: {emit_error}"
                                )
                    else:
                        logger.info(
                            "✅ Consolidation pipeline task completed successfully"
                        )

                task.add_done_callback(handle_task_completion)

        return body

    async def _invalidate_user_cache(self, user_id: str, reason: str = "") -> None:
        """Invalidate all cache entries for a specific user."""
        if Filter._cache_lock is None:
            Filter._cache_lock = asyncio.Lock()

        async with Filter._cache_lock:
            if user_id in Filter._embedding_cache:
                user_cache = Filter._embedding_cache[user_id]
                await user_cache.clear()
                logger.info(f"🧹 Cache cleared for user {user_id}")

                if user_id in Filter._cache_access_order:
                    Filter._cache_access_order.remove(user_id)

    def _clean_memory_content(self, content: str) -> str:
        """Clean memory content and validate length limits."""
        clean_content = content.strip()
        if len(clean_content) > Config.MAX_MEMORY_CONTENT_LENGTH:
            raise ValueError(
                f"Memory content too long ({len(clean_content)} chars, max {Config.MAX_MEMORY_CONTENT_LENGTH})"
            )
        return clean_content

    async def _execute_database_operation(
        self, operation_func, *args, timeout: float = None
    ) -> Any:
        """Execute database operation with timeout and error handling."""
        if timeout is None:
            timeout = Config.TIMEOUT_DATABASE_OPERATION

        return await asyncio.wait_for(
            asyncio.to_thread(operation_func, *args),
            timeout=timeout,
        )

    async def _execute_single_operation(
        self,
        operation: MemoryOperation,
        user: Any,
        user_id: str = None,
        conversation_hash: str = None,
    ) -> str:
        """Execute a single memory operation with optional conversation tracking."""
        if operation.operation == "CREATE":
            clean_content = self._clean_memory_content(operation.content)

            create_result = await self._execute_database_operation(
                Memories.insert_new_memory, user.id, clean_content
            )

            if conversation_hash and user_id and create_result:
                await self._mark_memory_as_created(
                    user_id, conversation_hash, str(create_result.id)
                )

            logger.info("✨ Memory created successfully")
            return "CREATE"

        elif operation.operation == "UPDATE" and operation.id:
            clean_content = self._clean_memory_content(operation.content)

            await self._execute_database_operation(
                Memories.update_memory_by_id_and_user_id,
                operation.id,
                user.id,
                clean_content,
            )

            if conversation_hash and user_id and operation.id:
                await self._mark_memory_as_updated(
                    user_id, conversation_hash, operation.id
                )

            logger.info(f"� Memory updated successfully")
            return "UPDATE"

        elif operation.operation == "DELETE" and operation.id:
            await self._execute_database_operation(
                Memories.delete_memory_by_id_and_user_id, operation.id, user.id
            )

            if conversation_hash and user_id and operation.id:
                await self._remove_memory_from_conversation_tracking(
                    user_id, conversation_hash, operation.id
                )

            logger.info("🗑️ Memory deleted successfully")
            return "DELETE"

        else:
            raise MemoryOperationError(f"Unsupported operation: {operation}")

    async def _query_llm(
        self, system_prompt: str, user_prompt: str, json_response: bool = True
    ) -> str:
        """Query the OpenAI API or compatible endpoints with production-ready error handling."""
        if not self.valves.api_key or not self.valves.api_key.strip():
            raise NeuralRecallError("API key is required but not provided.")

        session = await self._get_aiohttp_session()
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

        if json_response:
            payload["response_format"] = {"type": "json_object"}

        try:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 429:
                    logger.warning("🚦 Rate limited by LLM API, retrying after delay")
                    await asyncio.sleep(1)
                    raise NeuralRecallError("Rate limited by API")

                if response.status == 503:
                    logger.warning("🚫 LLM service unavailable")
                    raise NeuralRecallError("LLM service unavailable")

                response.raise_for_status()
                data = await response.json()

                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0]["message"]["content"]
                    if not content or content.strip() == "":
                        raise NeuralRecallError("Empty response from LLM")
                    return content

                raise NeuralRecallError(f"Unexpected API response format: {data}")

        except aiohttp.ClientError as e:
            logger.error(f"❌ HTTP client error: {e}")
            raise NeuralRecallError(f"HTTP client error: {e}")
        except asyncio.TimeoutError:
            logger.error("❌ LLM API timeout")
            raise NeuralRecallError("LLM API timeout")
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON response from LLM API: {e}")
            raise NeuralRecallError(f"Invalid JSON response: {e}")

    def _extract_and_parse_json(self, text: str) -> Any:
        """Extract and parse JSON from text response with production-ready error handling."""
        if not text or not text.strip():
            logger.warning("⚠️ Empty text provided for JSON parsing")
            return []

        text = text.strip()

        def try_parse_json(json_str: str) -> Any:
            """Helper to parse JSON and normalize return type."""
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    return [parsed]
                if isinstance(parsed, list):
                    return parsed
                logger.warning(f"⚠️ Parsed JSON is not list/dict: {type(parsed)}")
                return []
            except json.JSONDecodeError:
                return None

        result = try_parse_json(text)
        if result is not None:
            return result

        code_fence_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if code_fence_match:
            result = try_parse_json(code_fence_match.group(1).strip())
            if result is not None:
                return result

        generic_fence = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if generic_fence:
            result = try_parse_json(generic_fence.group(1).strip())
            if result is not None:
                return result

        start = text.find("[")
        if start != -1:
            count = 0
            for i in range(start, len(text)):
                if text[i] == "[":
                    count += 1
                elif text[i] == "]":
                    count -= 1
                    if count == 0:
                        result = try_parse_json(text[start:i + 1])
                        if result is not None:
                            return result
                        break

        logger.warning(
            "⚠️ Unable to extract valid JSON from LLM response; ensure the LLM returns only JSON (see consolidation prompt)"
        )
        return []

    @classmethod
    async def cleanup(cls) -> None:
        """Cleanup resources for graceful shutdown (production requirement)."""
        try:
            if cls._aiohttp_session and not cls._aiohttp_session.closed:
                await cls._aiohttp_session.close()
                logger.info("✅ HTTP session closed gracefully")

        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

    @classmethod
    def _auto_cleanup_finalizer(cls) -> None:
        """Synchronous finalizer called when the session object is GC'd."""
        try:
            loop = None
            try:
                loop = asyncio.get_event_loop()
            except Exception:
                loop = None
            if loop and loop.is_running():
                loop.call_soon_threadsafe(asyncio.create_task, cls.cleanup())
            else:
                logger.warning(
                    "⚠️ aiohttp session finalized without running event loop; call await Filter.cleanup() for graceful shutdown"
                )
        except Exception as e:
            logger.error(f"❌ Error in session finalizer: {e}")

    def __del__(self):
        """Destructor - logs warning if cleanup wasn't called explicitly."""
        try:
            if (
                hasattr(self.__class__, "_aiohttp_session")
                and self.__class__._aiohttp_session
                and not self.__class__._aiohttp_session.closed
            ):
                logger.warning(
                    "Filter instance finalized; aiohttp session still open (finalizer may handle cleanup)"
                )
        except:
            pass
