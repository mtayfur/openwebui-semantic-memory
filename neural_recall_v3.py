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
    datefmt="%Y-%m-%d %H:%M:%S"
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
    MAX_MESSAGE_LENGTH = 3000                  # Max characters allowed for an incoming message
    MIN_QUERY_LENGTH = 10                      # Minimum characters considered a valid query
    
    # JSON and structured data detection
    JSON_KEY_VALUE_THRESHOLD = 5               # Min key:value pairs to consider JSON/structured
    STRUCTURED_LINE_COUNT_MIN = 3              # Min number of lines before structured heuristics apply
    STRUCTURED_PERCENTAGE_THRESHOLD = 0.6      # Fraction of lines matching pattern to trigger structured skip
    STRUCTURED_BULLET_MIN = 4                  # Min bullet-like lines to treat as structured list
    STRUCTURED_PIPE_MIN = 2                    # Min pipe-count per line to treat as table-like structure
    
    # Symbol and character ratio detection
    SYMBOL_CHECK_MIN_LENGTH = 10               # Min message length before symbol-ratio check runs
    SYMBOL_RATIO_THRESHOLD = 0.5               # Minimum fraction of alpha/space characters to avoid symbol-skip
    
    # URL and log detection
    URL_COUNT_THRESHOLD = 3                    # Number of URLs considered a URL dump
    LOGS_LINE_COUNT_MIN = 2                    # Min lines before log-detection heuristics apply
    LOGS_MIN_MATCHES = 1                       # Minimum matched log-like lines to trigger log skip
    LOGS_MATCH_PERCENTAGE = 0.30               # Fraction of lines matching log pattern to trigger skip


class Config:
    """Core system configuration constants for Neural Recall."""
    
    # Cache and user limits
    CACHE_MAX_SIZE = 2500                      # LRU cache max entries per user
    MAX_USER_CACHES = 500                      # Global limit of per-user caches
    MAX_MEMORY_CONTENT_LENGTH = 600            # Max characters allowed when creating a memory

    # Network and DB timeouts (seconds)
    TIMEOUT_SESSION_REQUEST = 30               # aiohttp session total timeout
    TIMEOUT_DATABASE_OPERATION = 10            # Timeout for DB operations
    TIMEOUT_USER_LOOKUP = 5                    # Timeout for user lookups

    # Semantic retrieval defaults
    DEFAULT_SEMANTIC_THRESHOLD = 0.50          # Default similarity threshold for retrieval
    DEFAULT_MAX_MEMORIES_RETURNED = 15         # Default max memories injected into context

    # Embedding batch sizing
    MIN_BATCH_SIZE = 8                         # Minimum embedding batch size
    MAX_BATCH_SIZE = 32                        # Maximum embedding batch size

    # Pipeline configuration
    RETRIEVAL_MULTIPLIER = 3.0                 # Multiplier for candidate memory retrieval
    RETRIEVAL_TIMEOUT = 5.0                    # Timeout for retrieval operations
    CONSOLIDATION_CANDIDATE_SIZE = 50          # Max memories for consolidation analysis
    CONSOLIDATION_TIMEOUT = 30.0               # Timeout for consolidation operations
    CONSOLIDATION_RELAXED_MULTIPLIER = 0.9     # Relaxed threshold multiplier for consolidation

    # Status messages for skip operations
    STATUS_MESSAGES = {
        'SKIP_EMPTY': 'empty message',
        'SKIP_TOO_LONG': 'message too long',
        'SKIP_CODE': 'detected code content',
        'SKIP_STRUCTURED': 'detected structured data',
        'SKIP_SYMBOLS': 'message mostly symbols/numbers',
        'SKIP_LOGS': 'detected log dump',
        'SKIP_STACKTRACE': 'detected stack trace',
        'SKIP_URL_DUMP': 'detected URL dump',
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
- **Justification:** An existing memory inappropriately bundles two or more unrelated atomic facts that would be more useful as separate memories.
- **Anti-Pattern:** Do NOT split memories that are naturally related or where splitting would reduce coherence without meaningful benefit.
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
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0
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
            raise ValidationError(f"Invalid semantic threshold: {self.valves.semantic_threshold}")
            
        if self.valves.max_memories_returned <= 0:
            raise ValidationError(f"Invalid max memories returned: {self.valves.max_memories_returned}")
            
        logger.info("✅ Configuration validated")

    async def _get_embedding_model(self):
        """Get or load the sentence transformer model with thread safety."""
        if Filter._model is None:
            if Filter._model_load_lock is None:
                Filter._model_load_lock = asyncio.Lock()
            
            async with Filter._model_load_lock:
                if Filter._model is None:
                    try:
                        logger.info(f"🤖 Loading embedding model: {self.valves.embedding_model}")
                        
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

    async def _generate_embeddings_batch(self, texts: List[str], user_id: str) -> List[np.ndarray]:
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
                
                batch_size = min(Config.MAX_BATCH_SIZE, max(Config.MIN_BATCH_SIZE, len(uncached_texts)))
                
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
                
                for i in range(0, len(uncached_texts), batch_size):
                    batch_texts = uncached_texts[i:i + batch_size]
                    batch_indices = uncached_indices[i:i + batch_size]
                    
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
        logger.info(f"🚀 Batch embedding: {len(cached_embeddings)} cached, {len(new_embeddings)} new, {valid_count}/{len(texts)} valid")
        return result_embeddings

    async def _get_aiohttp_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with connection pooling and production settings."""
        if Filter._aiohttp_session is None or Filter._aiohttp_session.closed:
            if Filter._session_lock is None:
                Filter._session_lock = asyncio.Lock()
                
            async with Filter._session_lock:
                if Filter._aiohttp_session is None or Filter._aiohttp_session.closed:
                    timeout = aiohttp.ClientTimeout(total=Config.TIMEOUT_SESSION_REQUEST)
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
                        headers={"User-Agent": "NeuralRecallV3/3.0.0"}
                    )
                    try:
                        weakref.finalize(
                            Filter._aiohttp_session,
                            Filter._auto_cleanup_finalizer,
                        )
                    except Exception:
                        logger.debug("⚠️ Failed to register aiohttp session finalizer")
        return Filter._aiohttp_session

    def _should_skip_memory_operations(self, user_message: str) -> Tuple[bool, str]:
        """
        Enhanced gating for memory operations with comprehensive pattern detection.
        Skips: empty, too long, code, logs, structured data, URL dumps, or symbol spam.
        """
        if not user_message or not user_message.strip():
            return True, Config.STATUS_MESSAGES['SKIP_EMPTY']

        trimmed_message = user_message.strip()

        if len(trimmed_message) > SkipThresholds.MAX_MESSAGE_LENGTH:
            return True, f"{Config.STATUS_MESSAGES['SKIP_TOO_LONG']} ({len(trimmed_message)} chars > {SkipThresholds.MAX_MESSAGE_LENGTH})"

        code_patterns = [
            r'```[\s\S]*?```',        
            r'`[^`\n]+`',                  
            r'^\s*(def|class|var|let|const)\s+[\w_][\w\d_]*\s*[=\(\{;:]',
            r'^\s*(import|from)\s+[\w\.\-_]+',       
            r'^\s*function\s+\w+\s*\(',         
            r'if\s*\(.*\)\s*\{',         
            r'for\s*\(.*\)\s*\{',            
            r'while\s*\(.*\)\s*\{',           
            r'#include\s*<',                 
            r'using\s+namespace',              
            r'SELECT\s+.+\s+FROM\s+\w+',         
            r'INSERT\s+INTO\s+\w+',             
            r'UPDATE\s+\w+\s+SET\s+\w+\s*=',    
            r'DELETE\s+FROM\s+\w+',           
            r'^\s*[A-Za-z_]\w*\s*[=:]\s*[A-Za-z_]\w*\s*\(',
            r'\w+\s*=\s*\([^)]*\)\s*=>\s*\{',  
            r'^\s*\{\s*\}\s*$',                 
            r'<[^>]+>[\s\S]*</[^>]+>',        
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, trimmed_message, re.MULTILINE | re.IGNORECASE):
                return True, Config.STATUS_MESSAGES['SKIP_CODE']
        
        message_lines = trimmed_message.split('\n')
        line_count = len(message_lines)
        
        count_lines = lambda pattern: sum(1 for line in message_lines if re.match(pattern, line))
        
        json_indicators = [
            r'^\s*[\{\[].*[\}\]]\s*$',
            r'"[^"]*"\s*:\s*["\{\[\d]',
            r'["\']?\w+["\']?\s*:\s*["\{\[\d].*["\'\}\]\d]',
            r'\{\s*["\']?\w+["\']?\s*:\s*\{',
            r'\[\s*\{.*\}\s*\]',
        ]
        
        for pattern in json_indicators:
            if re.search(pattern, trimmed_message, re.DOTALL | re.IGNORECASE):
                return True, Config.STATUS_MESSAGES['SKIP_STRUCTURED']
        
        json_kv_pattern = r'["\']?\w+["\']?\s*:\s*["\{\[\d\w]'
        json_kv_matches = len(re.findall(json_kv_pattern, trimmed_message))
        if json_kv_matches >= SkipThresholds.JSON_KEY_VALUE_THRESHOLD:
            return True, Config.STATUS_MESSAGES['SKIP_STRUCTURED']
        
        if ((line_count >= SkipThresholds.STRUCTURED_LINE_COUNT_MIN and count_lines(r'^\s*[A-Za-z0-9_]+:\s+\S+') >= max(SkipThresholds.STRUCTURED_LINE_COUNT_MIN, line_count * SkipThresholds.STRUCTURED_PERCENTAGE_THRESHOLD)) or
            (line_count >= SkipThresholds.STRUCTURED_BULLET_MIN and count_lines(r'^\s*[-\*\+]\s+.+') >= max(SkipThresholds.STRUCTURED_BULLET_MIN, line_count * SkipThresholds.STRUCTURED_PERCENTAGE_THRESHOLD))):
            return True, Config.STATUS_MESSAGES['SKIP_STRUCTURED']
        
        if len(trimmed_message) > SkipThresholds.SYMBOL_CHECK_MIN_LENGTH and sum(1 for c in trimmed_message if c.isalpha() or c.isspace()) / len(trimmed_message) < SkipThresholds.SYMBOL_RATIO_THRESHOLD:
            return True, Config.STATUS_MESSAGES['SKIP_SYMBOLS']
        
        if (line_count >= SkipThresholds.STRUCTURED_LINE_COUNT_MIN and 
            (sum(1 for line in message_lines if '|' in line and line.count('|') >= SkipThresholds.STRUCTURED_PIPE_MIN) >= max(SkipThresholds.STRUCTURED_PIPE_MIN, line_count * SkipThresholds.STRUCTURED_PERCENTAGE_THRESHOLD) or
             count_lines(r'^\s*\d+\.\s') >= max(SkipThresholds.STRUCTURED_LINE_COUNT_MIN, line_count * SkipThresholds.STRUCTURED_PERCENTAGE_THRESHOLD) or
             count_lines(r'^\s*[a-zA-Z]\)\s') >= max(SkipThresholds.STRUCTURED_LINE_COUNT_MIN, line_count * SkipThresholds.STRUCTURED_PERCENTAGE_THRESHOLD))):
            return True, Config.STATUS_MESSAGES['SKIP_STRUCTURED']
        
        if line_count >= SkipThresholds.LOGS_LINE_COUNT_MIN:
            if sum(1 for line in message_lines if re.search(r'\d{2,4}[-/]\d{1,2}[-/]\d{1,2}.*\d{1,2}:\d{2}', line)) >= max(SkipThresholds.LOGS_MIN_MATCHES, line_count * SkipThresholds.LOGS_MATCH_PERCENTAGE):
                return True, Config.STATUS_MESSAGES['SKIP_LOGS']
            
            stack_patterns = [
                r'^\s*(at|File|Traceback|Exception|Error).*:\d+',
                r'Traceback\s*\(most recent call',
                r'^\s*File\s*"[^"]+",\s*line\s*\d+',
                r'^\s*at\s+\w+.*\([^)]*:\d+:\d+\)',
                r'Exception\s+in\s+thread',
                r'^\s*(NameError|TypeError|ValueError|AttributeError|KeyError):'
            ]
            if any(any(re.search(pattern, line) for pattern in stack_patterns) for line in message_lines):
                return True, Config.STATUS_MESSAGES['SKIP_STACKTRACE']
        
        single_line_error_patterns = [
            r'(NameError|TypeError|ValueError|AttributeError|KeyError|SyntaxError|IndentationError).*:',
            r'\d{2,4}[-/]\d{1,2}[-/]\d{1,2}[\s\d:]+\b(ERROR|WARN|INFO|DEBUG)\b',
            r'\bERROR\b.*:.*',
            r'^\s*(at|File)\s+["\']?[^"\']*["\']?,?\s*line\s+\d+',
            r'^\s*at\s+\w+.*\([^)]*:\d+:\d+\)',
            r'Exception\s+in\s+thread',
        ]
        
        for pattern in single_line_error_patterns:
            if re.search(pattern, trimmed_message, re.IGNORECASE):
                return True, Config.STATUS_MESSAGES['SKIP_LOGS']
        
        if len(re.findall(r'https?://[^\s]+', trimmed_message)) >= SkipThresholds.URL_COUNT_THRESHOLD:
            return True, Config.STATUS_MESSAGES['SKIP_URL_DUMP']
        
        if len(trimmed_message) < SkipThresholds.MIN_QUERY_LENGTH:
            return True, Config.STATUS_MESSAGES['SKIP_EMPTY']

        return False, ""

    def _extract_text_from_message_content(self, content: Union[str, List[Dict], Dict]) -> str:
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

    def _inject_datetime_context(self, body: Dict[str, Any]) -> None:
        """Inject current datetime context into the conversation."""
        datetime_string = self.get_formatted_datetime_string()
        system_message = {
            "role": "system",
            "content": f"Current Date/Time: {datetime_string}",
        }

        if "messages" in body:
            existing_system_messages = [
                msg for msg in body["messages"] if msg.get("role") == "system"
            ]
            if not existing_system_messages:
                body["messages"].insert(0, system_message)
            else:
                for msg in existing_system_messages:
                    if "Current Date/Time:" not in msg.get("content", ""):
                        msg["content"] += f"\n{system_message['content']}"
                        break
                else:
                    body["messages"].insert(0, system_message)

    def get_formatted_datetime_string(self) -> str:
        """Get formatted datetime string for context injection."""
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
            logger.info(f"⏳ Event emitter timed out for message: {message}")
        except Exception as e:
            logger.info(f"⚠️ Event emitter failed for message '{message}': {e}")

    async def _broad_retrieval(self, user_message: str, user_id: str, emitter: Optional[Callable[[Any], Awaitable[None]]] = None) -> List[Dict[str, Any]]:
        """Step 1 of Retrieval Pipeline: Perform broad vector search for candidate memories."""
        candidate_limit = int(self.valves.max_memories_returned * Config.RETRIEVAL_MULTIPLIER)
        
        user_memories = await asyncio.wait_for(
            asyncio.to_thread(Memories.get_memories_by_user_id, user_id),
            timeout=Config.RETRIEVAL_TIMEOUT
        )
        
        if not user_memories:
            logger.info("🔍 Retrieval Pipeline: No memories found for user")
            return []
        
        logger.info(f"🔍 Retrieval Pipeline: Retrieved {len(user_memories)} total memories")
        
        query_embedding = await self._generate_embedding(user_message, user_id)
        
        memory_contents = [memory.content for memory in user_memories]
        
        memory_embeddings = await self._generate_embeddings_batch(memory_contents, user_id)
        
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
            
            if similarity >= self.valves.semantic_threshold :
                memory_dict = {
                    "id": str(memory.id),
                    "content": memory.content,
                    "relevance": similarity,
                }
                formatted_memories.append(memory_dict)
        
        formatted_memories.sort(key=lambda x: x["relevance"], reverse=True)
        formatted_memories = formatted_memories[:candidate_limit]
        
        logger.info(f"🔍 Retrieval Pipeline: {len(formatted_memories)} candidates from {len(user_memories)} total memories (threshold: {self.valves.semantic_threshold :.3f})")
        
        if len(formatted_memories) >= candidate_limit and len(user_memories) > candidate_limit:
            await self._emit_status(emitter, f"📚 Found {len(user_memories)} memories, selected {len(formatted_memories)} candidates", False)
        else:
            await self._emit_status(emitter, f"📚 Found {len(formatted_memories)} memory candidates", False)
        
        return formatted_memories

    async def _llm_rerank_memories(self, user_message: str, candidate_memories: List[Dict[str, Any]], emitter: Optional[Callable[[Any], Awaitable[None]]] = None) -> List[Dict[str, Any]]:
        """Step 2 of Retrieval Pipeline: Use LLM to re-rank candidate memories."""
        if not candidate_memories:
            return []
        
        memory_context = "## CANDIDATE MEMORIES\n"
        for memory in candidate_memories:
            memory_context += f"ID: {memory['id']}\nContent: {memory['content']}\n\n"
        
        system_prompt = MEMORY_RERANKING_PROMPT + f"\n\n{memory_context}"
        
        await self._emit_status(emitter, "🤖 LLM re-ranking memories...", False)
        response = await asyncio.wait_for(
            self._query_llm(system_prompt, user_message, json_response=True),
            timeout=Config.RETRIEVAL_TIMEOUT
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
        
        memory_lookup = {mem['id']: mem for mem in candidate_memories}
        ranked_memories = []
        
        for memory_id in ranked_ids:
            if memory_id in memory_lookup:
                ranked_memories.append(memory_lookup[memory_id])
        
        ranked_ids_set = set(ranked_ids)
        for memory in candidate_memories:
            if memory['id'] not in ranked_ids_set:
                ranked_memories.append(memory)
        
        final_memories = ranked_memories[:self.valves.max_memories_returned]
        
        logger.info(f"🎯 Retrieval Pipeline: {len(final_memories)} memories selected from {len(candidate_memories)} candidates after LLM re-ranking")
        
        if len(final_memories) >= self.valves.max_memories_returned and len(ranked_memories) > self.valves.max_memories_returned:
            await self._emit_status(emitter, f"🎯 Ranked {len(ranked_memories)} memories, selected {len(final_memories)}", False)
        else:
            await self._emit_status(emitter, f"🎯 Selected {len(final_memories)} memories", False)
        
        return final_memories

    def _inject_memories_into_context(self, body: Dict[str, Any], memories: List[Dict[str, Any]], emitter: Optional[Callable[[Any], Awaitable[None]]] = None) -> None:
        """Inject selected memories into the conversation context."""
        if not memories:
            return
        memory_header = f"## RETRIEVED MEMORIES\n"
        memory_content = "\n".join([f"- {memory['content']}" for memory in memories])
        memory_injection = f"{memory_header}{memory_content}\n"
        
        if "messages" in body and body["messages"]:
            for i in range(len(body["messages"]) - 1, -1, -1):
                if body["messages"][i].get("role") == "user":
                    memory_message = {
                        "role": "system",
                        "content": memory_injection
                    }
                    body["messages"].insert(i, memory_message)
                    logger.info(f"💡 Injected {len(memories)} memories into conversation context")
                    break

    async def _gather_consolidation_candidates(self, user_message: str, user_id: str, emitter: Optional[Callable[[Any], Awaitable[None]]] = None) -> List[Dict[str, Any]]:
        """Step 1 of Consolidation Pipeline: Gather candidate memories for consolidation analysis."""
        user_memories = await asyncio.wait_for(
            asyncio.to_thread(Memories.get_memories_by_user_id, user_id),
            timeout=Config.CONSOLIDATION_TIMEOUT
        )
        
        if not user_memories:
            logger.info("📊 Consolidation Pipeline: No memories found for user")
            await self._emit_status(emitter, "💭 No memories to analyze", True)
            return []

        logger.info(f"📊 Consolidation Pipeline: Retrieved {len(user_memories)} total memories for consolidation analysis")
        
        query_embedding = await self._generate_embedding(user_message, user_id)
        
        memory_contents = [memory.content for memory in user_memories]
        
        memory_embeddings = await self._generate_embeddings_batch(memory_contents, user_id)
        
        if len(memory_embeddings) != len(user_memories):
            error_msg = f"Consolidation embedding count mismatch: {len(memory_embeddings)} vs {len(user_memories)}"
            logger.error(f"❌ {error_msg}")
            raise EmbeddingError(error_msg)
        
        formatted_memories = []
        relaxed_threshold = self.valves.semantic_threshold * Config.CONSOLIDATION_RELAXED_MULTIPLIER
        
        for i, memory in enumerate(user_memories):
            memory_embedding = memory_embeddings[i]
            if memory_embedding is None:
                continue
            
            similarity = float(np.dot(query_embedding, memory_embedding))
            
            if similarity >= relaxed_threshold:
                created_at_iso = None
                if hasattr(memory, 'created_at') and memory.created_at:
                    created_at_iso = datetime.fromtimestamp(memory.created_at, tz=timezone.utc).isoformat()
                
                updated_at_iso = None
                if hasattr(memory, 'updated_at') and memory.updated_at:
                    updated_at_iso = datetime.fromtimestamp(memory.updated_at, tz=timezone.utc).isoformat()
                
                memory_dict = {
                    "id": str(memory.id),
                    "content": memory.content,
                    "created_at": created_at_iso,
                    "updated_at": updated_at_iso,
                    "relevance": similarity,
                }
                formatted_memories.append(memory_dict)
        
        formatted_memories.sort(key=lambda x: x["relevance"], reverse=True)
        formatted_memories = formatted_memories[:Config.CONSOLIDATION_CANDIDATE_SIZE]
        
        logger.info(f"📊 Consolidation Pipeline: {len(formatted_memories)} candidates from {len(user_memories)} total memories (threshold: {relaxed_threshold:.3f})")
        
        if len(formatted_memories) >= Config.CONSOLIDATION_CANDIDATE_SIZE and len(user_memories) > Config.CONSOLIDATION_CANDIDATE_SIZE:
            await self._emit_status(emitter, f"📊 Analyzing {len(user_memories)} memories, selected {len(formatted_memories)} candidates", False)
        else:
            await self._emit_status(emitter, f"📊 Analyzing {len(formatted_memories)} memory candidates", False)
        
        return formatted_memories

    async def _llm_consolidate_memories(self, user_message: str, candidate_memories: List[Dict[str, Any]], emitter: Optional[Callable[[Any], Awaitable[None]]] = None) -> List[Dict[str, Any]]:
        """Step 2 of Consolidation Pipeline: Use LLM to generate consolidation plan."""
        if not candidate_memories:
            return []
        
        memory_context = "## CANDIDATE MEMORIES FOR CONSOLIDATION\n"
        for memory in candidate_memories:
            memory_context += f"ID: {memory['id']}\nContent: {memory['content']}\n"
            if memory.get('created_at'):
                memory_context += f"Created: {memory['created_at']}\n"
            if memory.get('updated_at'):
                memory_context += f"Updated: {memory['updated_at']}\n"
            memory_context += "\n"
        
        context_section = f"## USER MESSAGE\n{user_message}\n\n"
        
        date_context = f"## CURRENT DATE/TIME\n{self.get_formatted_datetime_string()}\n\n"
        
        system_prompt = MEMORY_CONSOLIDATION_PROMPT + f"\n\n{date_context}{context_section}{memory_context}"
        
        await self._emit_status(emitter, "🤖 LLM analyzing memory consolidation...", False)
        response = await asyncio.wait_for(
            self._query_llm(system_prompt, "Analyze the memories and generate a consolidation plan.", json_response=True),
            timeout=Config.CONSOLIDATION_TIMEOUT
        )
        
        if not response or response.startswith("Error:"):
            error_msg = "LLM consolidation analysis failed - empty or error response"
            logger.error(f"❌ {error_msg}")
            raise NeuralRecallError(error_msg)
        
        logger.info(f"🔧 LLM consolidation response received: {len(response)} characters")
        
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
        delete_operations = [op for op in operations if isinstance(op, dict) and op.get("operation") == "DELETE"]
        delete_ratio = len(delete_operations) / total_operations if total_operations > 0 else 0
        
        if delete_ratio > 0.5 and total_operations >= 3:
            logger.warning(f"⚠️ Consolidation safety trigger: {len(delete_operations)}/{total_operations} operations are deletions ({delete_ratio*100:.1f}%)")
            logger.warning("⚠️ Rejecting consolidation plan to prevent excessive memory loss")
            await self._emit_status(emitter, f"⚠️ Consolidation plan too aggressive - preserving memories", False)
            return []
        
        for operation in operations:
            if isinstance(operation, dict):
                memory_operation = MemoryOperation(**operation)
                if memory_operation.validate_operation(existing_memory_ids):
                    valid_operations.append(operation)
        
        logger.info(f"🔧 Consolidation Pipeline: {len(valid_operations)} operations planned")
        await self._emit_status(emitter, f"🎯 {len(valid_operations)} memory operations planned", False)
        return valid_operations

    async def _execute_consolidation_operations(self, operations: List[Dict[str, Any]], user_id: str, emitter: Optional[Callable[[Any], Awaitable[None]]] = None) -> None:
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
                result = await self._execute_single_operation(operation, user)
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
                logger.error(f"❌ Failed to execute {operation_type} operation {i+1}/{len(operations)} (ID: {operation_id}): {str(e)}")
                continue
        
        total_executed = created_count + updated_count + deleted_count
        logger.info(f"✅ Consolidation Pipeline: Executed {total_executed}/{len(operations)} operations (Created: {created_count}, Updated: {updated_count}, Deleted: {deleted_count}, Failed: {failed_count})")
        
        if total_executed > 0:
            result_parts = []
            if created_count > 0:
                result_parts.append(f"✨ Created {created_count}")
            if updated_count > 0:
                result_parts.append(f"📝 Updated {updated_count}")
            if deleted_count > 0:
                result_parts.append(f"🗑️ Deleted {deleted_count}")

            suffix = "memory" if total_executed == 1 else "memories"
            message = ", ".join(result_parts) + f" {suffix}"
            
            if failed_count > 0:
                message += f" ({failed_count} failed)"
                
            await self._emit_status(emitter, message, True)
            
            await self._invalidate_user_cache(user_id, "consolidation")
        elif failed_count > 0:
            await self._emit_status(emitter, f"⚠️ {failed_count}/{len(operations)} operations failed", True)
        else:
            await self._emit_status(emitter, "✅ Memories already optimally organized", True)

    async def _consolidation_pipeline_task(self, user_message: str, user_id: str, emitter: Optional[Callable[[Any], Awaitable[None]]] = None) -> None:
        """Complete Consolidation Pipeline as async background task (formerly Slow Path)."""
        try:
            logger.info("🔧 Starting Consolidation Pipeline analysis")
            await self._emit_status(emitter, "🔧 Analyzing memory patterns...", False)
            
            candidates = await self._gather_consolidation_candidates(user_message, user_id, emitter)
            
            if not candidates:
                logger.info("🔧 Consolidation Pipeline: No candidates found")
                await self._emit_status(emitter, "💭 No consolidation candidates", True)
                return
            
            operations = await self._llm_consolidate_memories(user_message, candidates, emitter)
            
            if not operations:
                logger.info("🔧 Consolidation Pipeline: Memories already optimally organized - no consolidation required")
                await self._emit_status(emitter, "✅ Memories already well organized", True)
                return
            
            await self._execute_consolidation_operations(operations, user_id, emitter)
            
            logger.info("🔧 Consolidation Pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Consolidation Pipeline failed: {str(e)}")
            await self._emit_status(emitter, f"❌ Consolidation failed: {str(e)[:50]}", True)
            raise


    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        RETRIEVAL PIPELINE (Synchronous, formerly Fast Path)
        Process incoming messages to inject relevant memories into context.
        """
        if not body or not __user__:
            return body

        user_id = __user__["id"]

        if "messages" in body and body["messages"]:
            last_user_msg = next(
                (
                    self._extract_text_from_message_content(m["content"])
                    for m in reversed(body.get("messages", []))
                    if m.get("role") == "user"
                ),
                None,
            )
            
            if not last_user_msg:
                self._inject_datetime_context(body)
                return body
                
            user_message = last_user_msg

            should_skip, skip_reason = self._should_skip_memory_operations(user_message)
            if should_skip:
                logger.info(f"⏭️ Retrieval Pipeline skipped: {skip_reason}")
                await self._emit_status(__event_emitter__, f"⏭️ Skipping: {skip_reason}", True)
                self._inject_datetime_context(body)
                return body

            await self._emit_status(__event_emitter__, "🚀 Retrieving relevant memories...", False)
            
            candidate_memories = await self._broad_retrieval(user_message, user_id, __event_emitter__)
            
            if not candidate_memories:
                await self._emit_status(__event_emitter__, "💭 No relevant memories", True)
                self._inject_datetime_context(body)
                return body
            
            relevant_memories = await self._llm_rerank_memories(user_message, candidate_memories, __event_emitter__)
            
            if relevant_memories:
                count = len(relevant_memories)
                await self._emit_status(__event_emitter__, f"💡 Injected {count} memories", True)
                self._inject_memories_into_context(body, relevant_memories, __event_emitter__)
            else:
                await self._emit_status(__event_emitter__, "💭 No memories selected", True)

        self._inject_datetime_context(body)
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

        if "messages" in body and body["messages"]:
            user_message = next(
                (
                    self._extract_text_from_message_content(m["content"])
                    for m in reversed(body["messages"])
                    if m["role"] == "user"
                ),
                None,
            )

            if user_message:
                should_skip, skip_reason = self._should_skip_memory_operations(user_message)
                if should_skip:
                    logger.info(f"⏭️ Consolidation Pipeline skipped: {skip_reason}")
                    return body

                logger.info("🔧 Starting Consolidation Pipeline background consolidation")
                task = asyncio.create_task(
                    self._consolidation_pipeline_task(user_message, user_id, __event_emitter__)
                )
                
                def handle_task_completion(completed_task):
                    if completed_task.exception():
                        exception = completed_task.exception()
                        logger.error(f"❌ Consolidation pipeline task failed: {exception}")
                        if __event_emitter__:
                            try:
                                asyncio.create_task(self._emit_status(__event_emitter__, "❌ Consolidation failed", True))
                            except Exception as emit_error:
                                logger.error(f"❌ Failed to emit final error status: {emit_error}")
                    else:
                        logger.info("✅ Consolidation pipeline task completed successfully")
                
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
            raise ValueError(f"Memory content too long ({len(clean_content)} chars, max {Config.MAX_MEMORY_CONTENT_LENGTH})")
        return clean_content

    async def _execute_database_operation(self, operation_func, *args, timeout: float = None) -> Any:
        """Execute database operation with timeout and error handling."""
        if timeout is None:
            timeout = Config.TIMEOUT_DATABASE_OPERATION
        
        return await asyncio.wait_for(
            asyncio.to_thread(operation_func, *args),
            timeout=timeout,
        )

    async def _execute_single_operation(self, operation: MemoryOperation, user: Any) -> str:
        """Execute a single memory operation."""
        if operation.operation == "CREATE":
            clean_content = self._clean_memory_content(operation.content)
            
            await self._execute_database_operation(
                Memories.insert_new_memory, user.id, clean_content
            )
            logger.info(f"✨ Memory created")
            return "CREATE"

        elif operation.operation == "UPDATE" and operation.id:
            clean_content = self._clean_memory_content(operation.content)

            await self._execute_database_operation(
                Memories.update_memory_by_id_and_user_id,
                operation.id,
                user.id,
                clean_content
            )
            logger.info(f"🔄 Memory updated {operation.id}")
            return "UPDATE"

        elif operation.operation == "DELETE" and operation.id:
            await self._execute_database_operation(
                Memories.delete_memory_by_id_and_user_id,
                operation.id,
                user.id
            )
            logger.info(f"🗑️ Memory deleted {operation.id}")
            return "DELETE"

        else:
            raise MemoryOperationError(f"Unsupported operation: {operation}")

    async def _query_llm(self, system_prompt: str, user_prompt: str, json_response: bool = True) -> str:
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

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return [parsed]
            if isinstance(parsed, list):
                return parsed
            logger.warning(f"⚠️ Parsed JSON is not list/dict: {type(parsed)}")
            return []
        except json.JSONDecodeError:
            pass

        code_fence_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        if code_fence_match:
            inner = code_fence_match.group(1).strip()
            try:
                parsed = json.loads(inner)
                if isinstance(parsed, dict):
                    return [parsed]
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                logger.warning("⚠️ Failed to parse JSON inside ```json code fence")

        generic_fence = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if generic_fence:
            inner = generic_fence.group(1).strip()
            try:
                parsed = json.loads(inner)
                if isinstance(parsed, dict):
                    return [parsed]
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                logger.warning("⚠️ Failed to parse JSON inside generic code fence")

        start = text.find('[')
        if start != -1:
            count = 0
            for i in range(start, len(text)):
                if text[i] == '[':
                    count += 1
                elif text[i] == ']':
                    count -= 1
                    if count == 0:
                        candidate = text[start:i+1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, list):
                                return parsed
                        except json.JSONDecodeError:
                            break

        logger.warning("⚠️ Unable to extract valid JSON from LLM response; ensure the LLM returns only JSON (see consolidation prompt)")
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
                logger.warning("⚠️ aiohttp session finalized without running event loop; call await Filter.cleanup() for graceful shutdown")
        except Exception as e:
            logger.error(f"❌ Error in session finalizer: {e}")
    
    def __del__(self):
        """Destructor - logs warning if cleanup wasn't called explicitly."""
        try:
            if hasattr(self.__class__, '_aiohttp_session') and self.__class__._aiohttp_session and not self.__class__._aiohttp_session.closed:
                logger.debug("Filter instance finalized; aiohttp session still open (finalizer may handle cleanup)")
        except:
            pass
