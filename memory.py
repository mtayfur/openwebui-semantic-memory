"""
title: Neural Recall

This module implements a semantic memory system for OpenWebUI with advanced LRU caching.

Key Features:
- True LRU cache implementation using OrderedDict
- Per-user cache isolation with comprehensive metrics
- Real-time cache performance monitoring and logging
- Automatic performance warnings for low hit rates
- Context-aware cache statistics (operations, retrieval, search)

Cache Statistics Logging:
- Cache performance is logged after major operations
- Hit rates, eviction counts, and cache sizes are tracked
- Performance warnings are issued when hit rates drop below 50%
- Success indicators are shown when hit rates exceed 80%
"""

import asyncio
import hashlib
import json
import multiprocessing
import re
import time
import traceback
import unicodedata
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Tuple, Union

import aiohttp
import numpy as np
from fastapi.requests import Request
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from open_webui.main import app as webui_app
from open_webui.models.users import Users
from open_webui.routers.memories import Memories

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("NeuralRecall")


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

class Config:
    """Centralized configuration constants for Neural Recall system."""

    CACHE_MAX_SIZE = 1000

    MAX_MESSAGE_LENGTH = 5000
    MAX_MEMORY_CONTENT_LENGTH = 1000
    MIN_QUERY_LENGTH = 10

    TIMEOUT_SESSION_REQUEST = 30
    TIMEOUT_DATABASE_OPERATION = 10
    TIMEOUT_MEMORY_PROCESSING = 15
    TIMEOUT_USER_LOOKUP = 5

    QUERY_ENHANCEMENT_MIN_LENGTH = 40
    QUERY_ENHANCEMENT_MAX_SIMPLE_LENGTH = 120

    DEFAULT_SEMANTIC_THRESHOLD = 0.60
    DEFAULT_DUPLICATE_THRESHOLD = 0.90

    DEFAULT_MAX_MEMORIES_RETURNED = 10
    MIN_BATCH_SIZE = 8
    MAX_BATCH_SIZE = 32

    CODE_PATTERNS = [
        r'```[\s\S]*?```',
        r'`[^`\n]+`',
        r'function\s+\w+\s*\(',
        r'class\s+\w+',
        r'import\s+\w+',
        r'from\s+\w+\s+import',
        r'def\s+\w+\s*\(',
        r'const\s+\w+\s*=',
        r'let\s+\w+\s*=',
        r'var\s+\w+\s*=',
        r'if\s*\([^)]*\)\s*\{',
        r'for\s*\([^)]*\)\s*\{',
        r'while\s*\([^)]*\)\s*\{',
        r'#include\s*<',
        r'using\s+namespace',
        r'SELECT\s+.+\s+FROM',
        r'INSERT\s+INTO',
        r'UPDATE\s+.+\s+SET',
        r'DELETE\s+FROM',
    ]

    STRUCTURED_DATA_PATTERNS = [
        r'^\s*[\{\[]...[\}\]]\s*$',
        r'^\s*<[^>]+>[\s\S]*</[^>]+>\s*$',
        r'^\s*\|[^|]+\|[^|]+\|',
        r'^\s*[-\*\+]\s+.+(\n\s*[-\*\+]\s+.+){3,}',
        r'^\s*\d+\.\s+.+(\n\s*\d+\.\s+.+){3,}',
        r'^\s*[A-Za-z0-9_]+:\s*.+(\n\s*[A-Za-z0-9_]+:\s*.+){3,}',
    ]

    STATUS_MESSAGES = {
        'QUERY_ENHANCING': '🧠 Enhancing query for better recall...',
        'SEARCHING_MEMORY': '🔍 Searching memory vault...',
        'ANALYZING_MESSAGE': '🧪 Analyzing message for memorable facts...',
        'MEMORY_TIMEOUT': '⏰ Memory analysis timed out',
        'MEMORY_FAILED': '❌ Memory analysis failed',
        'SKIP_EMPTY': 'empty message',
        'SKIP_TOO_LONG': 'message too long',
        'SKIP_CODE': 'detected code content',
        'SKIP_STRUCTURED': 'detected structured data',
        'SKIP_SYMBOLS': 'message mostly symbols/numbers'
    }

MEMORY_IDENTIFICATION_PROMPT = """You are a Personal Information Structuring Agent. Your mission is to analyze user messages and convert explicitly stated personal facts into a structured JSON array of memory operations. You must operate with absolute precision and adherence to the following principles and rules.

    ## Core Principles
    1.  **Verifiable Factuality**: You must only record what is explicitly stated. Do not infer, assume, or interpret facts not present in the text.
    2.  **Informational Density**: Group closely related details into a single, dense memory. If a user mentions a project's name, its purpose, and its deadline in separate sentences, your final operation must combine them into one consolidated fact.
    3.  **Temporal Precedence**: New, dated information always supersedes older, conflicting information. Use this to determine when an `UPDATE` or `DELETE` is appropriate.
    4.  **Contextual Grounding**: Use the provided `Current Date/Time` to convert relative references (e.g., "yesterday", "next week") into absolute, semantic-friendly dates (e.g., "August 11 2025", "December 1 2025"). Always include specific dates when available or derivable.
    5.  **Rich Context Preservation**: Include relevant nouns, proper names, and pronouns in your memory content to maintain semantic searchability and context.

    ## Execution Rules
    - **Mandatory English**: All memory `content` must be translated into clear, concise English.
    - **Strict Prefixing**: Every `content` field MUST start with "User" or "User's". There are no exceptions.
    - **Date Integration**: When temporal information is present or derivable, always include specific dates in format "Month Day Year" (e.g., "August 12 2025") for optimal semantic retrieval.
    - **Fact Selection**: Only store significant, stable facts about the user's life, identity, and key relationships.
    - **Content to Ignore**: You must ignore all of the following: questions for the AI, conversational filler, transient states (moods, weather), speculative statements ("I might..."), general knowledge, and technical content.

    ## Operations
    - `NEW`: Creates an entirely new fact.
    - `UPDATE`: Corrects, expands, or adds detail to an existing fact. Use the provided `id`.
    - `DELETE`: Removes a fact that is explicitly stated to be incorrect or outdated. Use the provided `id`.

    Your entire output must be a single, valid JSON array. If no valid facts are found, output an empty array `[]`.
    ---
    ## Examples

    ### Example 1: Vehicle Transition with Rich Context (English)
    **Existing Memories**:
    - `{"id": "mem-077", "content": "User's car is a Ford Focus."}`
    **User Input**: "I finally sold my old Ford Focus last week. My new car is a midnight silver Tesla Model Y Performance that I picked up yesterday from the Seattle dealership."
    **Response**:
    [
    {"operation": "UPDATE", "id": "mem-077", "content": "User sold their Ford Focus."},
    {"operation": "NEW", "content": "User's new car is a midnight silver Tesla Model Y Performance from Seattle dealership."}
    ]

    ### Example 2: Family Information Updates with Temporal Context (English)
    **Context Provided**: Current Date/Time is `Friday September 15 2025`.
    **Existing Memories**:
    - `{"id": "mem-101", "content": "User's son Alex plays tennis."}`
    - `{"id": "mem-102", "content": "User's daughter Jessica attends high school."}`
    **User Input**: "Alex switched to soccer this fall season and is really enjoying it. Also, I need to correct something - my daughter's name is Jennifer, not Jessica, and she just started her senior year."
    **Response**:
    [
    {"operation": "UPDATE", "id": "mem-101", "content": "User's son Alex plays soccer and enjoys it."},
    {"operation": "UPDATE", "id": "mem-102", "content": "User's daughter Jennifer is a high school senior."}
    ]

    ### Example 3: Complex Professional Update with Multiple People (Spanish)
    **Context Provided**: Current Date/Time is `Monday October 7 2025`.
    **Existing Memories**:
    - `{"id": "mem-203", "content": "User works with colleague Maria on marketing projects."}`
    **User Input**: "Maria se fue de la empresa la semana pasada. Ahora trabajo con Carlos y Diana en el nuevo proyecto de marketing digital que empezamos ayer."
    **Response**:
    [
    {"operation": "UPDATE", "id": "mem-203", "content": "User's former colleague Maria left the company."},
    {"operation": "NEW", "content": "User works with colleagues Carlos and Diana on digital marketing project started October 6 2025."}
    ]

    ### Example 4: Health Information with Date Precision (German)
    **Context Provided**: Current Date/Time is `Wednesday November 20 2025`.
    **User Input**: "Gestern hatte ich endlich meinen Termin bei Dr. Schmidt wegen meiner Rückenschmerzen. Er hat mir Physiotherapie verschrieben und ich fange nächste Woche damit an."
    **Response**:
    [
    {"operation": "NEW", "content": "User saw Dr. Schmidt on November 19 2025 for back pain and was prescribed physiotherapy starting November 27 2025."}
    ]

    ### Example 5: Mixed Personal and Professional Information Filtering (French)
    **User Input**: "La France a gagné la Coupe du Monde en 2018, c'est de l'histoire. Mais pour moi personnellement, mon frère Jean travaille maintenant chez Airbus à Toulouse depuis janvier."
    **Response**:
    [
    {"operation": "NEW", "content": "User's brother Jean works at Airbus in Toulouse since January 2025."}
    ]

    OUTPUT: JSON array only."""


QUERY_ENHANCEMENT_PROMPT = """You are a Semantic Query Formulation Engine. Your mission is to deconstruct a user's conversational message into a precise, logical set of search queries for a vector database. These queries must be optimized to retrieve stored personal facts.

    ## Core Principles
    1.  **Intent Decomposition**: Analyze the user's underlying intent. If the query implies comparison, causality, or multi-step logic, break it down into multiple, distinct queries that retrieve the necessary component facts.
    2.  **Conceptual Search**: Expand keywords into related concepts to cast a wider semantic net. A query about a "trip" should include "travel," "flight," and "hotel."
    3.  **Preserve Key Context**: Retain important markers like pronouns ("he", "she", "they"), proper nouns (names, places, companies), and temporal references. Convert relative dates into absolute, specific terms using the provided context to ensure queries match the stored fact format.
    4.  **Temporal Precision**: Always include specific dates when available or derivable, formatted as "Month Day Year" (e.g., "August 12 2025") for optimal semantic matching.
    5.  **Noun and Pronoun Retention**: Include relevant nouns, proper names, and pronouns in queries to maintain semantic context and improve retrieval accuracy.

    ## Formatting Rules
    - **JSON Array Output**: Your entire response must be a single, valid JSON array of strings (`["query1", ...]`).
    - **Strict Prefixing**: Every query string MUST begin with "User" or "User's". There are no exceptions.
    - **Declarative & English**: All queries must be declarative statements, not questions, and must be in English.
    - **Topic Granularity**: Keep distinct topics as separate queries.
    - **Date Format**: Use "Month Day Year" format (e.g., "August 5 2025") for dates to match memory storage format.

    Your function is to formulate the best possible queries to find existing memories; you are not answering the user. If the input is not a question or command that requires memory, output an empty array `[]`.
    ---
    ## Examples

    ### Example 1: Complex Travel Query with Multi-Step Logic (English)
    **Context Provided**: Current Date/Time is `Tuesday August 12 2025`.
    **Original**: "Why was my flight to Denver from JFK last Tuesday cancelled, and what was my backup plan with the hotel booking?"
    **Enhanced**:
    [
    "User's flight to Denver from JFK on August 5 2025 cancellation reason details",
    "User's backup plan for Denver trip after flight cancellation",
    "User's hotel booking Denver trip August 5 2025 arrangements"
    ]

    ### Example 2: Professional Relationship and Decision Tracking (English)
    **Original**: "What did my manager Sarah and the client representative Mr. Johnson agree on during our Project Titan budget meeting last week?"
    **Enhanced**:
    [
    "User's manager Sarah Project Titan budget meeting decisions agreements",
    "User's client representative Mr. Johnson Project Titan budget discussion",
    "User's Project Titan budget meeting outcomes Sarah Johnson agreement"
    ]

    ### Example 3: Personal Goals and Timeline Comparison (French)
    **Context Provided**: Current Date/Time is `Wednesday September 18 2025`.
    **Original**: "Compare mes objectifs de fitness que j'avais fixés en janvier avec mes progrès actuels et ce que mon entraîneur Marc recommande."
    **Enhanced**:
    [
    "User's fitness goals set in January 2025 objectives targets",
    "User's current fitness progress September 2025 achievements",
    "User's trainer Marc fitness recommendations current advice"
    ]

    ### Example 4: Multi-Party Family Business Inquiry (German)
    **Original**: "Was haben mein Bruder Klaus und seine Geschäftspartnerin Anna über die Expansion ihres Restaurants nach Hamburg entschieden?"
    **Enhanced**:
    [
    "User's brother Klaus business partner Anna restaurant expansion Hamburg",
    "User's brother Klaus restaurant expansion decisions Hamburg plans",
    "User's brother Klaus Anna partnership restaurant business Hamburg"
    ]

    ### Example 5: Complex Professional Network Query (English)
    **Context Provided**: Current Date/Time is `Friday November 22 2025`.
    **Original**: "What was the outcome of the negotiation between my business partner Michael, the investor Dr. Chen, and the supplier representatives about our new product line launch timeline?"
    **Enhanced**:
    [
    "User's business partner Michael Dr. Chen investor negotiation outcome",
    "User's supplier representatives new product line negotiation results",
    "User's new product line launch timeline decisions Michael Dr. Chen suppliers",
    "User's business partner Michael investor Dr. Chen supplier meeting agreements"
    ]

    OUTPUT: JSON array of enhanced queries only, in English."""

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
        """Get comprehensive cache statistics."""
        async with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": round(hit_rate, 2),
                "total_requests": total_requests,
                "utilization": round((len(self._cache) / self.max_size) * 100, 2)
            }

    def _reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self._hits = 0
        self._misses = 0
        self._evictions = 0


class MemoryOperation(BaseModel):
    """Enhanced model for memory operations with better validation."""

    operation: Literal["NEW", "UPDATE", "DELETE"]
    id: Optional[str] = Field(None, description="Memory ID for UPDATE/DELETE operations")
    content: Optional[str] = Field(None, description="Memory content for NEW/UPDATE operations")

    def validate_operation(self) -> bool:
        """Validate operation based on its type."""
        if self.operation in ["NEW", "UPDATE"]:
            return self.content is not None and self.content.strip() != ""
        elif self.operation == "DELETE":
            return self.id is not None and self.id.strip() != ""
        return False


class Filter:
    """Enhanced Neural Recall semantic memory filter with improved architecture."""

    
    _global_sentence_model: Optional[SentenceTransformer] = None
    _model_lock = asyncio.Lock()
    _embedding_cache: Dict[str, LRUCache] = {}
    _cache_lock = asyncio.Lock()

    class Valves(BaseModel):
        """Enhanced configuration valves for the filter with better validation."""

        api_url: str = Field(
            default="https://api.openai.com/v1",
            description="API URL for OpenAI-compatible chat completions endpoint.",
        )
        api_key: str = Field(
            default="",
            description="API key for authentication (required).",
        )
        model: str = Field(
            default="gpt-4o-mini",
            description="Model name to use for chat completions.",
        )
        embedding_model: str = Field(
            default="Alibaba-NLP/gte-multilingual-base",
            description="Sentence transformer model for semantic similarity.",
        )
        semantic_threshold: float = Field(
            default=Config.DEFAULT_SEMANTIC_THRESHOLD,
            description="Minimum semantic similarity score (0-1) for initial memory filtering."
        )
        duplicate_threshold: float = Field(
            default=Config.DEFAULT_DUPLICATE_THRESHOLD,
            description="Similarity threshold (0-1) for detecting and preventing duplicate memories."
        )
        max_memories_returned: int = Field(
            default=Config.DEFAULT_MAX_MEMORIES_RETURNED,
            description="Maximum number of most relevant memories to inject into the context.",
        )
        timezone_hours: int = Field(
            default=0,
            description="Timezone offset in hours (e.g., 5 for UTC+5, -4 for UTC-4).",
            ge=-12,
            le=12,
        )

    class UserValves(BaseModel):
        """User-specific configuration valves."""

        enabled: bool = Field(
            default=True,
            description="Enable or disable the memory function."
        )

    def __init__(self) -> None:
        """Initialize the Filter with enhanced configuration and optimizations."""
        self.valves = self.Valves()
        self.user_valves = self.UserValves()
        self._aiohttp_session: Optional[aiohttp.ClientSession] = None

        cpu_count = multiprocessing.cpu_count()
        self._batch_size = max(
            Config.MIN_BATCH_SIZE,
            min(Config.MAX_BATCH_SIZE, cpu_count * 2)
        )

        logger.info(
            f"Neural Recall initialized - Batch size: {self._batch_size} "
            f"(CPU cores: {cpu_count}), Cache max: {Config.CACHE_MAX_SIZE}"
        )


    def get_formatted_datetime(self) -> datetime:
        """Get the current datetime object in the user's timezone with enhanced handling."""
        try:
            utc_time = datetime.now(timezone.utc)
            user_timezone = timezone(timedelta(hours=self.valves.timezone_hours))
            return utc_time.astimezone(user_timezone)
        except Exception as e:
            logger.error(f"Timezone conversion failed: {e}")
            raise NeuralRecallError(f"Failed to get formatted datetime: {e}") from e

    async def _get_sentence_model(self) -> SentenceTransformer:
        """Get or initialize the sentence transformer model using enhanced singleton pattern."""
        async with Filter._model_lock:
            if Filter._global_sentence_model is None:
                try:
                    logger.info(f"Loading sentence transformer: {self.valves.embedding_model}")

                    Filter._global_sentence_model = await asyncio.to_thread(
                        SentenceTransformer,
                        self.valves.embedding_model,
                        device="cpu",
                        trust_remote_code=True,
                    )

                    logger.info("Sentence transformer model loaded successfully")

                except Exception as e:
                    error_msg = f"Failed to load sentence transformer model: {e}"
                    logger.error(error_msg)
                    raise ModelLoadError(error_msg) from e

            return Filter._global_sentence_model


    async def _get_aiohttp_session(self) -> aiohttp.ClientSession:
        """Get or create an enhanced aiohttp session with proper configuration."""
        if self._aiohttp_session is None or self._aiohttp_session.closed:
            timeout = aiohttp.ClientTimeout(total=Config.TIMEOUT_SESSION_REQUEST)
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                enable_cleanup_closed=True
            )

            self._aiohttp_session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={"User-Agent": "Neural-Recall/2.0"},
            )

        return self._aiohttp_session

    async def _ensure_session_closed(self) -> None:
        """Ensure the aiohttp session is properly closed with enhanced error handling."""
        if self._aiohttp_session and not self._aiohttp_session.closed:
            try:
                await self._aiohttp_session.close()
                logger.info("HTTP session closed successfully")
            except Exception as e:
                logger.warning(f"Error closing HTTP session: {e}")
            finally:
                self._aiohttp_session = None

    def _generate_cache_key(self, text: str, user_id: Optional[str] = None) -> str:
        """Generate a stable cache key for text embedding with enhanced hashing."""
        cache_content = f"{text}|{self.valves.embedding_model}"
        if user_id:
            cache_content = f"{user_id}|{cache_content}"
        return hashlib.sha256(cache_content.encode('utf-8')).hexdigest()[:32]

    async def _get_cached_embedding(
        self, text: str, user_id: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Get cached embedding for text if available with enhanced cache management."""
        cache_key = self._generate_cache_key(text, user_id)
        cache_user_key = user_id or "global"

        async with Filter._cache_lock:
            if cache_user_key not in Filter._embedding_cache:
                Filter._embedding_cache[cache_user_key] = LRUCache(Config.CACHE_MAX_SIZE)

            user_cache = Filter._embedding_cache[cache_user_key]
            cached_data = await user_cache.get(cache_key)

            if cached_data and isinstance(cached_data, dict) and "embedding" in cached_data:
                return cached_data["embedding"]

        return None

    async def _cache_embedding(
        self, text: str, embedding: np.ndarray, user_id: Optional[str] = None
    ) -> None:
        """Cache embedding for text with enhanced metadata."""
        cache_key = self._generate_cache_key(text, user_id)
        cache_user_key = user_id or "global"

        async with Filter._cache_lock:
            if cache_user_key not in Filter._embedding_cache:
                Filter._embedding_cache[cache_user_key] = LRUCache(Config.CACHE_MAX_SIZE)

            user_cache = Filter._embedding_cache[cache_user_key]

            cache_data = {
                "embedding": embedding,
                "timestamp": datetime.now().timestamp(),
                "text_preview": text[:50],
                "text_length": len(text),
            }

            await user_cache.put(cache_key, cache_data)

    async def _get_embedding_with_cache(
        self, text: str, user_id: Optional[str] = None
    ) -> np.ndarray:
        """Get embedding for text with enhanced cache management and error handling."""
        if not text or not text.strip():
            raise EmbeddingError("Cannot generate embedding for empty text")

        cached_embedding = await self._get_cached_embedding(text, user_id)
        if cached_embedding is not None:
            return cached_embedding

        try:
            model = await self._get_sentence_model()
            embedding = await asyncio.to_thread(
                model.encode, [text], normalize_embeddings=True
            )
            embedding_array = embedding[0]

            await self._cache_embedding(text, embedding_array, user_id)
            return embedding_array

        except Exception as e:
            error_msg = f"Error generating embedding for text: {e}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e

    async def _get_embeddings_batch(
        self, texts: List[str], user_id: Optional[str] = None
    ) -> List[np.ndarray]:
        """Get embeddings for multiple texts using enhanced batch processing and caching."""
        if not texts:
            return []

        
        valid_texts = [text for text in texts if text and text.strip()]
        if len(valid_texts) != len(texts):
            logger.warning(f"Filtered out {len(texts) - len(valid_texts)} empty texts")

        cached_embeddings: Dict[int, np.ndarray] = {}
        texts_to_process: List[str] = []
        text_indices: Dict[int, int] = {}

        
        for i, text in enumerate(valid_texts):
            cached_embedding = await self._get_cached_embedding(text, user_id)
            if cached_embedding is not None:
                cached_embeddings[i] = cached_embedding
            else:
                text_indices[len(texts_to_process)] = i
                texts_to_process.append(text)

        
        new_embeddings: Dict[int, np.ndarray] = {}
        if texts_to_process:
            try:
                model = await self._get_sentence_model()

                for batch_start in range(0, len(texts_to_process), self._batch_size):
                    batch_end = min(batch_start + self._batch_size, len(texts_to_process))
                    batch_texts = texts_to_process[batch_start:batch_end]

                    batch_embeddings = await asyncio.to_thread(
                        model.encode, batch_texts, normalize_embeddings=True
                    )

                    for j, embedding in enumerate(batch_embeddings):
                        original_index = text_indices[batch_start + j]
                        new_embeddings[original_index] = embedding

                        
                        await self._cache_embedding(batch_texts[j], embedding, user_id)

            except Exception as e:
                error_msg = f"Error in batch embedding generation: {e}"
                logger.error(error_msg)
                raise EmbeddingError(error_msg) from e

        
        result: List[np.ndarray] = []
        for i in range(len(valid_texts)):
            if i in cached_embeddings:
                result.append(cached_embeddings[i])
            elif i in new_embeddings:
                result.append(new_embeddings[i])
            else:
                raise EmbeddingError(f"Missing embedding for text at index {i}")

        return result

    async def _get_cache_stats(self, user_id: Optional[str] = None) -> str:
        """Get enhanced cache statistics for monitoring and debugging."""
        cache_user_key = user_id or "global"

        async with Filter._cache_lock:
            if cache_user_key not in Filter._embedding_cache:
                user_stats = {
                    "size": 0, "max_size": Config.CACHE_MAX_SIZE,
                    "hits": 0, "misses": 0, "evictions": 0,
                    "hit_rate": 0, "utilization": 0
                }
            else:
                user_cache = Filter._embedding_cache[cache_user_key]
                user_stats = await user_cache.get_stats()

            if user_id is not None:
                return (f"user:{user_id} {user_stats['size']}/{user_stats['max_size']} "
                        f"(hit_rate: {user_stats['hit_rate']:.1f}%, "
                        f"utilization: {user_stats['utilization']:.1f}%)")

            
            total_size = 0
            total_hits = 0
            total_misses = 0
            total_evictions = 0
            cache_count = len(Filter._embedding_cache)

            for cache in Filter._embedding_cache.values():
                stats = await cache.get_stats()
                total_size += stats["size"]
                total_hits += stats["hits"]
                total_misses += stats["misses"]
                total_evictions += stats["evictions"]

            total_requests = total_hits + total_misses
            overall_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0

            return (f"user:{cache_user_key} {user_stats['size']}/{user_stats['max_size']} "
                    f"(hit_rate: {user_stats['hit_rate']:.1f}%), "
                    f"global: {total_size} entries, {cache_count} caches, "
                    f"hit_rate: {overall_hit_rate:.1f}%, evictions: {total_evictions}")

    async def log_cache_performance(
        self, user_id: Optional[str] = None, context: str = "general"
    ) -> None:
        """Log cache performance statistics with enhanced context and warnings."""
        try:
            cache_stats_str = await self._get_cache_stats(user_id)
            logger.info(f"📊 Cache performance [{context}]: {cache_stats_str}")

        except Exception as e:
            logger.error(f"Failed to log cache performance [{context}]: {e}")

    def _should_enhance_query(self, query: str) -> bool:
        """Decide if LLM enhancement is needed based on enhanced criteria."""
        if not query or not query.strip():
            return False

        query = query.strip()
        sentence_count = len(re.findall(r'[.!?]+', query))

        if sentence_count > 1:
            return (len(query) >= Config.QUERY_ENHANCEMENT_MIN_LENGTH or
                    sentence_count > 3)

        return len(query) > Config.QUERY_ENHANCEMENT_MAX_SIMPLE_LENGTH

    async def _enhance_query(self, query: str) -> List[str]:
        """Enhanced query enhancement using LLM for better semantic retrieval."""
        if not self._should_enhance_query(query):
            normalized = self._normalize_text(query)
            if self._is_query_meaningful(normalized):
                return [query.strip()]
            else:
                return []

        try:
            enhanced_prompt = QUERY_ENHANCEMENT_PROMPT
            try:
                date_context = f"\n\nCurrent Date/Time: {self.get_formatted_datetime().strftime('%A %B %d %Y at %H:%M')}"
                enhanced_prompt += date_context
            except NeuralRecallError as e:
                logger.warning(f"Failed to add datetime context to query enhancement: {e}")

            enhanced_response = await self._query_llm(
                enhanced_prompt, query, json_response=True
            )

            enhanced_queries = self._extract_and_parse_json(enhanced_response)

            if not isinstance(enhanced_queries, list):
                logger.error("Query enhancement returned non-list format")
                raise EmbeddingError("Query enhancement failed: invalid response format")

            valid_queries = []
            for enhanced_query in enhanced_queries:
                if isinstance(enhanced_query, str) and enhanced_query.strip():
                    cleaned = enhanced_query.strip()
                    cleaned = re.sub(r'^"|"$', "", cleaned)
                    normalized = self._normalize_text(cleaned)
                    if self._is_query_meaningful(normalized):
                        valid_queries.append(cleaned)

            if not valid_queries:
                logger.error("No valid queries after enhancement")
                raise EmbeddingError("Query enhancement failed: no valid queries generated")

            return valid_queries

        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")
            raise EmbeddingError(f"Query enhancement failed: {e}") from e

    async def _get_relevant_memories_multi_query(
        self, queries: List[str], user_id: str
    ) -> List[Dict[str, Any]]:
        """Get relevant memories for multiple queries and deduplicate results."""
        if not queries:
            return []

        all_memories = []
        memory_ids_seen = set()
        
        
        query_stats = {
            'total_queries': len(queries),
            'memories_per_query': [],
            'new_memories_per_query': [],
            'score_ranges': [],
            'threshold_passed_counts': [],
            'total_processed': 0,
            'start_time': time.time()
        }

        logger.info(f"🔍 Multi-Query Search: Processing {len(queries)} distinct queries")

        for i, query in enumerate(queries, 1):
            
            memories = await self.get_relevant_memories(query, user_id, threshold=None, verbose_logging=False)
            
            query_stats['memories_per_query'].append(len(memories))
            
            new_memories = []
            for memory in memories:
                memory_id = memory.get("id")
                if memory_id and memory_id not in memory_ids_seen:
                    memory_ids_seen.add(memory_id)
                    new_memories.append(memory)

            query_stats['new_memories_per_query'].append(len(new_memories))
            all_memories.extend(new_memories)
            
            
            if memories:
                scores = [mem.get('relevance', 0) for mem in memories]
                query_stats['score_ranges'].append((min(scores), max(scores)))
                
                
                threshold = self.valves.semantic_threshold
                passed_count = sum(1 for score in scores if score >= threshold)
                query_stats['threshold_passed_counts'].append(passed_count)
            else:
                query_stats['score_ranges'].append((0, 0))
                query_stats['threshold_passed_counts'].append(0)

        all_memories.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        final_memories = all_memories[:self.valves.max_memories_returned]
        
        
        processing_time = time.time() - query_stats['start_time']
        total_found = sum(query_stats['memories_per_query'])
        total_new = sum(query_stats['new_memories_per_query'])
        total_passed_threshold = sum(query_stats['threshold_passed_counts'])
        
        
        all_scores = [score for score_range in query_stats['score_ranges'] for score in score_range if score > 0]
        if all_scores:
            min_score = min(all_scores)
            max_score = max(all_scores)
            avg_score = sum(all_scores) / len(all_scores)
        else:
            min_score = max_score = avg_score = 0
            
        
        avg_memories_per_query = total_found / len(queries) if queries else 0
        avg_new_per_query = total_new / len(queries) if queries else 0
        
        logger.info(f"📊 Multi-Query Results Summary:")
        logger.info(f"   ├─ Queries processed: {len(queries)} in {processing_time:.2f}s")
        logger.info(f"   ├─ Total memories found: {total_found} (avg: {avg_memories_per_query:.1f} per query)")
        logger.info(f"   ├─ Unique memories: {len(all_memories)} (deduplication: {total_found - len(all_memories)} removed)")
        logger.info(f"   ├─ Threshold passed: {total_passed_threshold}/{total_found} ({total_passed_threshold/max(1,total_found)*100:.1f}%)")
        logger.info(f"   ├─ Score range: {min_score:.1%} - {max_score:.1%} (avg: {avg_score:.1%})")
        logger.info(f"   └─ Final selection: {len(final_memories)} memories (max: {self.valves.max_memories_returned})")

        return final_memories

    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process incoming messages to inject relevant memories into the context."""
        if not body or not __user__ or not self.user_valves.enabled:
            return body

        if "messages" in body and body["messages"]:
            last_user_msg = next(
                (
                    m["content"]
                    for m in reversed(body.get("messages", []))
                    if m.get("role") == "user"
                ),
                None,
            )
            if not last_user_msg:
                return body
            user_message = last_user_msg
            user_id = __user__["id"]

            
            should_skip, skip_reason = self._should_skip_memory_operations(user_message)
            if should_skip:
                logger.info(f"⏭️ Skipping memory operations: {skip_reason}")
                await self._emit_status(
                    __event_emitter__, f"⏭️ Skipping memory operations: {skip_reason}", True
                )
                
            else:
                
                user_messages = [msg for msg in body["messages"] if msg["role"] == "user"]
                is_first_message = len(user_messages) == 1

                retrieval_start_time = time.time()

                should_enhance = self._should_enhance_query(user_message)

                if should_enhance:
                    await self._emit_status(
                        __event_emitter__,
                        "🧠 Enhancing query for better recall...",
                        False,
                    )
                    try:
                        enhanced_queries = await self._enhance_query(user_message)
                        if len(enhanced_queries) > 1:
                            logger.info(f"🔍 Multi-query enhancement: '{user_message[:100]}...' → {len(enhanced_queries)} queries")
                        else:
                            logger.info(f"🔍 Query enhanced for semantic search: '{user_message[:100]}...' → '{enhanced_queries[0][:100] if enhanced_queries else 'empty'}...'")
                    except EmbeddingError as e:
                        logger.error(f"Query enhancement failed: {e}")
                        await self._emit_status(
                            __event_emitter__, "❌ Query enhancement failed", True
                        )
                        
                    else:
                        
                        await self._emit_status(
                            __event_emitter__, "🔍 Searching memory vault...", False
                        )

                        if len(enhanced_queries) > 1:
                            relevant_memories = await self._get_relevant_memories_multi_query(
                                enhanced_queries, user_id
                            )
                        elif enhanced_queries:
                            relevant_memories = await self.get_relevant_memories(
                                enhanced_queries[0], user_id, threshold=None
                            )
                        else:
                            relevant_memories = []

                        retrieval_duration = time.time() - retrieval_start_time

                        if relevant_memories:
                            count = len(relevant_memories)
                            memory_word = "memory" if count == 1 else "memories"
                            query_info = f" ({len(enhanced_queries)} queries)" if len(enhanced_queries) > 1 else ""
                            logger.info(f"✅ Memory Retrieval: Completed in {retrieval_duration:.2f}s - Found {count} relevant {memory_word}{query_info}")

                            await self._emit_status(
                                __event_emitter__, f"💡 Found {count} relevant {memory_word}"
                            )
                            self._inject_memories_into_context(body, relevant_memories)
                        else:
                            normalized_query = self._normalize_text(enhanced_queries[0]) if enhanced_queries else ""
                            if not self._is_query_meaningful(normalized_query):
                                logger.info(f"⚠️ Memory Retrieval: Completed in {retrieval_duration:.2f}s - Query too short ({len(normalized_query)} < 10 chars)")
                                await self._emit_status(
                                    __event_emitter__,
                                    f"💭 Query too short for memory ({len(normalized_query)} < 10)",
                                )
                            else:
                                query_info = f" ({len(enhanced_queries)} queries)" if len(enhanced_queries) > 1 else ""
                                logger.info(f"🔍 Memory Retrieval: Completed in {retrieval_duration:.2f}s - No relevant memories found{query_info}")
                                await self._emit_status(
                                    __event_emitter__, "💭 No relevant memories found"
                                )
                else:
                    
                    enhanced_queries = [user_message]
                    logger.info(f"🔍 Simple query, skipping enhancement: '{user_message[:100]}...'")

                    await self._emit_status(
                        __event_emitter__, "🔍 Searching memory vault...", False
                    )

                    relevant_memories = await self.get_relevant_memories(
                        enhanced_queries[0], user_id, threshold=None
                    )

                    retrieval_duration = time.time() - retrieval_start_time

                    if relevant_memories:
                        count = len(relevant_memories)
                        memory_word = "memory" if count == 1 else "memories"
                        logger.info(f"✅ Memory Retrieval: Completed in {retrieval_duration:.2f}s - Found {count} relevant {memory_word}")

                        await self._emit_status(
                            __event_emitter__, f"💡 Found {count} relevant {memory_word}"
                        )
                        self._inject_memories_into_context(body, relevant_memories)
                    else:
                        normalized_query = self._normalize_text(enhanced_queries[0]) if enhanced_queries else ""
                        if not self._is_query_meaningful(normalized_query):
                            logger.info(f"⚠️ Memory Retrieval: Completed in {retrieval_duration:.2f}s - Query too short ({len(normalized_query)} < 10 chars)")
                            await self._emit_status(
                                __event_emitter__,
                                f"💭 Query too short for memory ({len(normalized_query)} < 10)",
                            )
                        else:
                            logger.info(f"🔍 Memory Retrieval: Completed in {retrieval_duration:.2f}s - No relevant memories found")
                            await self._emit_status(
                                __event_emitter__, "💭 No relevant memories found"
                            )

        
        self._inject_datetime_context(body)
        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        """Process outgoing messages to identify and store new memories from user messages only."""
        if not body or not __user__ or not self.user_valves.enabled:
            return body

        if "messages" in body and body["messages"]:
            user_message = next(
                (
                    m["content"]
                    for m in reversed(body["messages"])
                    if m["role"] == "user"
                ),
                None,
            )

            if user_message:
                
                should_skip, skip_reason = self._should_skip_memory_operations(user_message)
                if should_skip:
                    logger.info(f"⏭️ Skipping memory analysis: {skip_reason}")
                    await self._emit_status(
                        __event_emitter__, f"⏭️ Skipping memory analysis: {skip_reason}", True
                    )
                    return body

                await self._emit_status(
                    __event_emitter__,
                    "🧪 Analyzing message for memorable facts...",
                    False,
                )

                try:
                    memory_creation_start_time = time.time()
                    await asyncio.wait_for(
                        self._process_user_memories(
                            user_message,
                            __user__["id"],
                            __event_emitter__,
                            memory_creation_start_time,
                        ),
                        timeout=Config.TIMEOUT_MEMORY_PROCESSING,
                    )

                    try:
                        await self.log_cache_performance(__user__["id"], "memory_operation_completed")
                    except Exception as cache_error:
                        logger.error(f"❌ Cache performance logging failed: {cache_error}")

                except asyncio.TimeoutError:
                    logger.error("⏰ Memory processing timed out after 15 seconds")
                    await self._emit_status(
                        __event_emitter__,
                        "⏰ Memory analysis timed out",
                        True,
                    )
                except Exception as e:
                    logger.error(f"❌ Error during memory processing: {e}")
                    await self._emit_status(
                        __event_emitter__,
                        "❌ Memory analysis failed",
                        True,
                    )
        return body

    async def _emit_status(
        self,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]],
        description: str,
        done: bool = True,
    ) -> None:
        """Enhanced helper method to emit status messages with better error handling."""
        if event_emitter:
            try:
                await event_emitter({
                    "type": "status",
                    "data": {
                        "description": description,
                        "done": done,
                    },
                })
            except Exception as e:
                logger.error(f"Event emitter failed: {e}")

    def _format_operation_status(self, executed_ops: List[Dict[str, Any]]) -> str:
        """Format memory operations into an enhanced status message."""
        if not executed_ops:
            return "🙈 No memory operations executed"

        
        op_emojis = {"NEW": "✨", "UPDATE": "🔄", "DELETE": "🗑️"}
        op_actions = {"NEW": "Created", "UPDATE": "Updated", "DELETE": "Removed"}

        ops_by_type = {"NEW": [], "UPDATE": [], "DELETE": []}
        for op in executed_ops:
            ops_by_type[op["operation"]].append(op)

        status_parts = []
        for op_type, ops in ops_by_type.items():
            if ops:
                emoji = op_emojis[op_type]
                action = op_actions[op_type]
                count = len(ops)
                memory_word = "memory" if count == 1 else "memories"
                status_parts.append(f"{emoji} {action} {count} {memory_word}")

        if len(status_parts) == 1:
            return f"🧠 {status_parts[0]}"
        else:
            return f"🧠 Memory updates: {' | '.join(status_parts)}"

    async def _get_formatted_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all memories for a user from the database with enhanced error handling."""
        try:
            user_memories = await asyncio.wait_for(
                asyncio.to_thread(Memories.get_memories_by_user_id, user_id=str(user_id)),
                timeout=Config.TIMEOUT_DATABASE_OPERATION,
            )

            memories_list = [
                {
                    "id": str(memory.id),
                    "memory": memory.content,
                }
                for memory in user_memories
            ]

            return memories_list

        except asyncio.TimeoutError:
            logger.error(f"Database timeout while fetching memories for user {user_id}")
            raise MemoryOperationError("Database timeout while fetching memories")
        except Exception as e:
            logger.error(f"Error fetching memories for user {user_id}: {e}")
            raise MemoryOperationError(f"Failed to fetch memories: {e}") from e

    def _inject_memories_into_context(
        self, body: Dict[str, Any], memories: List[Dict[str, Any]]
    ) -> None:
        """Inject relevant memories into the system message with enhanced formatting."""
        if not memories:
            return

        memory_context = self._format_memories_for_context(memories)
        system_message_found = False

        for message in body["messages"]:
            if message["role"] == "system":
                message["content"] = f"{message['content']}\n\n{memory_context}"
                system_message_found = True
                break

        if not system_message_found:
            body["messages"].insert(0, {"role": "system", "content": memory_context})

    def _inject_datetime_context(self, body: Dict[str, Any]) -> None:
        """Inject current date/time context into the system message with enhanced formatting."""
        try:
            current_time = self.get_formatted_datetime()

            timezone_offset = self.valves.timezone_hours
            if timezone_offset >= 0:
                timezone_str = f"UTC+{timezone_offset}" if timezone_offset > 0 else "UTC"
            else:
                timezone_str = f"UTC{timezone_offset}"

            
            formatted_date = current_time.strftime("%A %B %d %Y")
            formatted_time = current_time.strftime("%H:%M")

            datetime_context = (
                f"CURRENT DATE/TIME: {formatted_date} at {formatted_time} {timezone_str}"
            )

            system_message_found = False
            for message in body["messages"]:
                if message["role"] == "system":
                    message["content"] = f"{message['content']}\n\n{datetime_context}"
                    system_message_found = True
                    break

            if not system_message_found:
                body["messages"].insert(0, {"role": "system", "content": datetime_context})

        except Exception as e:
            logger.warning(f"Failed to inject datetime context: {e}")

    def _format_memories_for_context(self, memories: List[Dict[str, Any]]) -> str:
        """Format a list of memories into an enhanced string for LLM context."""
        if not memories:
            return ""

        header = (
            "BACKGROUND: You naturally know these facts. "
            "Never reference having this information or where it came from:"
        )
        formatted_mems = "\n".join([f"- {mem['memory']}" for mem in memories])
        return f"{header}\n{formatted_mems}"

    async def _process_user_memories(
        self,
        user_message: str,
        user_id: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]],
        start_time: float,
    ) -> List[Dict[str, Any]]:
        """Identify and execute memory operations based on the user's message."""

        try:
            existing_memories = await self._get_formatted_memories(user_id)
            relevant_memories = await self.get_relevant_memories(user_message, user_id, threshold=self.valves.semantic_threshold-0.1, max_memories=self.valves.max_memories_returned * 2)
            operations, validation_info = await self.identify_memory_operations(
                user_message, relevant_memories
            )

            if operations:
                operation_types = [op['operation'] for op in operations]
                logger.info(f"🧠 Memory Operations: Identified {len(operations)} operations ({', '.join(operation_types)})")
                executed_ops, skipped_info = await self.execute_memory_operations(
                    operations, user_id, existing_memories
                )

                memory_creation_duration = time.time() - start_time

                if executed_ops:
                    status_message = self._format_operation_status(executed_ops)
                    if skipped_info["duplicates"] > 0:
                        status_message += (
                            f" | 💡 Skipped {skipped_info['duplicates']} duplicates"
                        )
                    logger.info(f"✅ Memory Operations: Completed in {memory_creation_duration:.2f}s - {status_message}")

                    await self._emit_status(__event_emitter__, status_message, True)
                elif skipped_info["duplicates"] > 0:
                    logger.info(f"💡 Memory operations completed in {memory_creation_duration:.2f}s - Skipped {skipped_info['duplicates']} duplicate memories")
                    await self._emit_status(
                        __event_emitter__,
                        f"💡 Skipped {skipped_info['duplicates']} duplicate memories",
                        True,
                    )
                else:
                    logger.info(f"💭 Memory operations completed in {memory_creation_duration:.2f}s - No operations executed")
                    await self._emit_status(__event_emitter__, "💭 No operations executed", True)
                return executed_ops
            else:
                memory_creation_duration = time.time() - start_time
                if validation_info["validation_failed"] > 0:
                    logger.info(f"⚠️ Memory analysis completed in {memory_creation_duration:.2f}s - Skipped {validation_info['validation_failed']} invalid operations")
                    await self._emit_status(
                        __event_emitter__,
                        f"⚠️ Skipped {validation_info['validation_failed']} invalid operations",
                        True,
                    )
                else:
                    logger.info(f"💭 Memory analysis completed in {memory_creation_duration:.2f}s - No memorable facts found")
                    await self._emit_status(__event_emitter__, "💭 No memorable facts found", True)
            return []
        except Exception as e:
            memory_creation_duration = time.time() - start_time
            logger.error(f"❌ Unexpected error in memory processing after {memory_creation_duration:.2f}s: {e}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            await self._emit_status(
                __event_emitter__,
                "❌ Memory processing failed",
                True
            )
            return []

    async def identify_memory_operations(
        self,
        input_text: str,
        existing_memories: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Use an LLM to identify memory operations from the input text."""
        system_prompt = MEMORY_IDENTIFICATION_PROMPT

        if existing_memories:
            mem_context = (
                "\n\n## EXISTING USER MEMORIES\nRefer to these memories when determining UPDATE/DELETE operations. Use the exact `id` values for modifications:\n\n"
                + json.dumps(existing_memories, indent=2)
            )
            system_prompt += mem_context

        try:
            date_context = f"\n\nCurrent Date/Time: {self.get_formatted_datetime().strftime('%A %B %d %Y at %H:%M')}"
            system_prompt += date_context
        except NeuralRecallError as e:
            logger.warning(f"Failed to add datetime context: {e}")
            

        response = await self._query_llm(system_prompt, input_text, json_response=True)
        if not response or response.startswith("Error:"):
            logger.error(f"❌ LLM error during memory identification: {response}")
            return [], {"validation_failed": 0}

        logger.info(f"✅ LLM Analysis: Completed - Processing memory operations")
        operations = self._extract_and_parse_json(response)

        if not isinstance(operations, list):
            logger.warning(f"⚠️ LLM returned invalid format (expected list): {type(operations)}")
            return [], {"validation_failed": 0}

        valid_operations = [op for op in operations if op is not None and isinstance(op, dict)]

        if len(valid_operations) < len(operations):
            logger.info(f"🔧 Pre-filtered {len(operations) - len(valid_operations)} incomplete operations, keeping {len(valid_operations)}")

        operations = valid_operations

        logger.info(f"📊 Operation Parsing: Found {len(operations)} operations")

        valid_ops = []
        validation_failed_count = 0

        for i, op in enumerate(operations):
            if self._validate_memory_operation(op, existing_memories):
                valid_ops.append(op)
            else:
                validation_failed_count += 1
                logger.info(f"❌ Operation Validation: Rejected operation {i+1} - {op.get('operation', 'unknown')} - {op}")

        if validation_failed_count > 0:
            logger.info(f"⚠️ Operation Validation: Rejected {validation_failed_count} invalid operations")

        logger.info(f"✅ Operation Validation: {len(valid_ops)} operations ready for execution")
        return valid_ops, {"validation_failed": validation_failed_count}

    def _validate_memory_operation(
        self, op: Dict[str, Any], existing_memories: List[Dict[str, Any]]
    ) -> bool:
        """Enhanced validation of a single memory operation with detailed logging."""
        try:
            if not isinstance(op, dict):
                logger.warning(f"Operation must be a dictionary, got {type(op)}: {op}")
                return False

            if "operation" not in op:
                logger.warning(f"Operation missing 'operation' field: {op}")
                return False

            if op["operation"] not in ["NEW", "UPDATE", "DELETE"]:
                logger.warning(f"Invalid operation type: {op.get('operation')} (must be NEW, UPDATE, or DELETE)")
                return False

            
            try:
                memory_op = MemoryOperation(**op)
                if not memory_op.validate_operation():
                    logger.warning(f"Operation failed validation: {op}")
                    return False
            except PydanticValidationError as e:
                logger.warning(f"Invalid MemoryOperation structure: {e} - Operation: {op}")
                return False

            
            if op["operation"] in ["NEW", "UPDATE"]:
                content = op.get("content", "").strip()

                if not content:
                    logger.warning(f"Content is required for {op['operation']} operation but is empty: {op}")
                    return False

                if not (content.startswith("User ") or content.startswith("User's ")):
                    logger.warning(f"Memory content must start with 'User' or 'User's': {content[:50]}...")
                    return False

                if len(content) > Config.MAX_MEMORY_CONTENT_LENGTH:
                    logger.warning(f"Content exceeds maximum length ({len(content)} > {Config.MAX_MEMORY_CONTENT_LENGTH}): {content[:50]}...")
                    return False

            
            if op["operation"] in ["UPDATE", "DELETE"]:
                op_id = op.get("id")
                if not op_id:
                    logger.warning(f"Operation {op['operation']} has empty/None id: {repr(op_id)}")
                    return False

                existing_ids = {mem["id"] for mem in existing_memories}
                if op_id not in existing_ids:
                    available_preview = list(existing_ids)[:3]
                    logger.warning(
                        f"Invalid ID for {op['operation']} on non-existent memory: {op_id} "
                        f"(Available: {available_preview}{'...' if len(existing_ids) > 3 else ''})"
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"Unexpected error during operation validation: {e}")
            return False

    def _extract_and_parse_json(self, text: str) -> Union[List, Dict, None]:
        """Extract and parse JSON from text, handling common LLM response issues."""
        if not text:
            logger.warning("Empty text provided to JSON parser")
            return None

        try:
            result = json.loads(text.strip())
            return result
        except json.JSONDecodeError as e:
            pass

        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1)
            try:
                result = json.loads(text.strip())
                return result
            except json.JSONDecodeError as e:
                pass

        for pattern in [r"(\[[\s\S]*?\])", r"(\{[\s\S]*?\})"]:
            match = re.search(pattern, text)
            if match:
                json_text = match.group(1)
                try:
                    result = json.loads(json_text)
                    return result
                except json.JSONDecodeError as e:
                    continue

        logger.error(f"❌ JSON Parsing: Failed to extract valid JSON from response")
        return None

    async def _calculate_memory_similarity(
        self, text1: str, text2: str, user_id: Optional[str] = None
    ) -> float:
        """Calculate enhanced semantic similarity between two strings using sentence transformers."""
        if not text1 or not text2:
            return 0.0

        normalized_text1 = self._normalize_text(text1)
        normalized_text2 = self._normalize_text(text2)

        
        if normalized_text1 == normalized_text2:
            return 1.0

        try:
            embedding1 = await self._get_embedding_with_cache(normalized_text1, user_id)
            embedding2 = await self._get_embedding_with_cache(normalized_text2, user_id)

            similarity_matrix = cosine_similarity(np.array([embedding1]), np.array([embedding2]))
            similarity_score = float(max(0, min(1, similarity_matrix[0][0])))

            return similarity_score

        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            raise EmbeddingError(f"Failed to calculate similarity: {e}") from e

    def _normalize_text(self, text: str) -> str:
        """Enhanced text normalization for better semantic matching with multi-language support."""
        if not text or not text.strip():
            return ""

        try:
            normalized = unicodedata.normalize("NFKD", text).lower()
            normalized = "".join(c for c in normalized if not unicodedata.combining(c))
            normalized = re.sub(r"[^\w\s]", " ", normalized)
            normalized = re.sub(r"\s+", " ", normalized).strip()

            return normalized

        except Exception as e:
            logger.error(f"Text normalization failed: {e}")
            raise EmbeddingError(f"Text normalization failed: {e}") from e

    def _is_query_meaningful(self, text: str) -> bool:
        """Enhanced query validation for meaningful memory search."""
        if not text or not text.strip():
            return False
        return len(text.strip()) >= Config.MIN_QUERY_LENGTH

    def _calculate_dynamic_threshold(self, scored_memories: List[Dict[str, Any]], verbose_logging: bool = True) -> float:
        """Calculate a simple dynamic threshold using the median score, clamped to min/max bounds."""
        if not scored_memories:
            return self.valves.semantic_threshold

        scores = [mem["relevance"] for mem in scored_memories]
        scores_sorted = sorted(scores, reverse=True)
        n = len(scores_sorted)
        median_score = (
            (scores_sorted[n // 2 - 1] + scores_sorted[n // 2]) / 2
            if n % 2 == 0 else scores_sorted[n // 2]
        )

        min_threshold = max(0.0, self.valves.semantic_threshold - 0.1)
        max_threshold = min(1.0, self.valves.duplicate_threshold - 0.1)
        final_threshold = max(min_threshold, min(max_threshold, median_score))

        if verbose_logging:
            logger.info(f"Simple dynamic threshold: median={median_score:.2f}, bounds=({min_threshold:.2f}, {max_threshold:.2f}), final={final_threshold:.2f}")

        return final_threshold

    def _should_skip_memory_operations(self, message: str) -> Tuple[bool, str]:
        """
        Enhanced determination of whether memory operations should be skipped.
        Returns (should_skip, reason).

        Skips memory operations for:
        1. Very long messages (>5000 chars) that could overwhelm the LLM
        2. Non-textual content like code, structured data, or lists
        """
        if not message or not message.strip():
            return True, Config.STATUS_MESSAGES['SKIP_EMPTY']

        message = message.strip()

        
        if len(message) > Config.MAX_MESSAGE_LENGTH:
            return True, f"{Config.STATUS_MESSAGES['SKIP_TOO_LONG']} ({len(message)} chars > {Config.MAX_MESSAGE_LENGTH})"

        
        for pattern in Config.CODE_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE | re.MULTILINE):
                return True, Config.STATUS_MESSAGES['SKIP_CODE']

        
        for pattern in Config.STRUCTURED_DATA_PATTERNS:
            if re.search(pattern, message, re.MULTILINE):
                return True, Config.STATUS_MESSAGES['SKIP_STRUCTURED']

        
        words = re.findall(r'\b[a-zA-Z]{2,}\b', message)
        total_chars = len(message)
        word_chars = sum(len(word) for word in words)

        if total_chars > 200 and (word_chars / total_chars) < 0.3:
            return True, Config.STATUS_MESSAGES['SKIP_SYMBOLS']

        return False, ""

    async def get_relevant_memories(
        self, current_message: str, user_id: str, threshold: Optional[float] = None, max_memories: Optional[int] = None, verbose_logging: bool = True
    ) -> List[Dict[str, Any]]:
        """Get memories relevant to the current context using enhanced semantic similarity."""
        try:
            existing_memories = await self._get_formatted_memories(user_id)
            initial_memory_count = len(existing_memories)
            if verbose_logging:
                logger.info(f"Memory search initiated across {initial_memory_count} stored memories")

            if not existing_memories:
                if verbose_logging:
                    logger.info("No stored memories found")
                return []

            normalized_current_message = self._normalize_text(current_message)

            if not self._is_query_meaningful(normalized_current_message):
                if verbose_logging:
                    logger.info(f"Query too short ({len(normalized_current_message.strip())} chars < {Config.MIN_QUERY_LENGTH})")
                return []

            memory_texts = [
                self._normalize_text(mem["memory"]) for mem in existing_memories
            ]
            all_texts = [normalized_current_message] + memory_texts

            
            all_embeddings = await self._get_embeddings_batch(all_texts, user_id)
            current_embedding = all_embeddings[0]
            memory_embeddings = all_embeddings[1:]

            
            scored_memories = []
            for i, mem in enumerate(existing_memories):
                similarity_matrix = cosine_similarity(
                    np.array([current_embedding]), np.array([memory_embeddings[i]])
                )
                semantic_similarity = float(max(0, min(1, similarity_matrix[0][0])))
                scored_memories.append({**mem, "relevance": semantic_similarity})

            scored_memories.sort(key=lambda x: x["relevance"], reverse=True)
            if verbose_logging:
                logger.info(f"Computed similarity scores for {len(scored_memories)} memories")

            
            if threshold is not None:
                semantic_threshold = threshold
                if verbose_logging:
                    logger.info(f"Using provided threshold: {semantic_threshold:.1%}")
            else:
                semantic_threshold = self._calculate_dynamic_threshold(scored_memories, verbose_logging)

            filtered_memories = [
                mem for mem in scored_memories
                if mem["relevance"] >= semantic_threshold
            ]
            if verbose_logging:
                logger.info(f"Semantic filtering: {len(filtered_memories)}/{len(scored_memories)} memories passed {semantic_threshold:.1%} threshold")

            
            max_returned = max_memories if max_memories is not None else self.valves.max_memories_returned
            result = filtered_memories[:max_returned]
            if verbose_logging:
                logger.info(f"Final selection: {len(result)} memories (max: {max_returned})")

            if result and verbose_logging:
                sorted_scores = sorted([mem["relevance"] for mem in result])
                n = len(sorted_scores)
                median_score = (
                    (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
                    if n % 2 == 0 else sorted_scores[n // 2]
                )

                score_range = f"{result[-1]['relevance']:.1%} - {result[0]['relevance']:.1%}"
                logger.info(f"Score range: {score_range} (median: {median_score:.1%})")
            elif not result and verbose_logging:
                median_score = 0
                max_score = (
                    max(scored_memories, key=lambda x: x["relevance"])["relevance"]
                    if scored_memories else 0
                )
                logger.info(f"No memories met {semantic_threshold:.1%} threshold (best: {max_score:.1%})")
            else:
                
                if result:
                    sorted_scores = sorted([mem["relevance"] for mem in result])
                    n = len(sorted_scores)
                    median_score = (
                        (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
                        if n % 2 == 0 else sorted_scores[n // 2]
                    )
                else:
                    median_score = 0

            if verbose_logging:
                logger.info(f"Pipeline complete: {initial_memory_count} → {len(filtered_memories)} → {len(result)}")
            
            
            result_stats = {
                'initial_count': initial_memory_count,
                'filtered_count': len(filtered_memories),
                'final_count': len(result),
                'threshold': semantic_threshold,
                'score_range': (result[-1]['relevance'], result[0]['relevance']) if result else (0, 0),
                'median_score': median_score
            }
            
            
            if hasattr(result, '_stats'):
                result._stats = result_stats
            else:
                
                for mem in result:
                    if not hasattr(mem, '_query_stats'):
                        mem['_query_stats'] = result_stats
                        break
            
            return result

        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            raise MemoryOperationError(f"Failed to retrieve memories: {e}") from e

    async def execute_memory_operations(
        self,
        operations: List[Dict[str, Any]],
        user_id: str,
        existing_memories: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Execute a list of memory operations (NEW, UPDATE, DELETE)."""
        executed_operations = []
        skipped_info = {"duplicates": 0, "validation_failed": 0}

        user = await asyncio.wait_for(
            asyncio.to_thread(Users.get_user_by_id, str(user_id)),
            timeout=Config.TIMEOUT_USER_LOOKUP
        )
        if not user:
            raise ValueError(f"User not found: {user_id}")

        ops_to_execute = []
        existing_contents = {mem["memory"] for mem in existing_memories}

        new_operations = [op for op in operations if op["operation"] == "NEW"]
        other_operations = [op for op in operations if op["operation"] != "NEW"]

        if new_operations and existing_contents:
            new_contents = [
                self._normalize_text(op["content"]) for op in new_operations
            ]
            existing_list = [
                self._normalize_text(content) for content in existing_contents
            ]

            all_texts = new_contents + existing_list
            all_embeddings = await self._get_embeddings_batch(all_texts, user_id)

            new_embeddings = all_embeddings[: len(new_contents)]
            existing_embeddings = all_embeddings[len(new_contents) :]

            for i, op in enumerate(new_operations):
                is_duplicate = False
                new_embedding = new_embeddings[i]

                for existing_embedding in existing_embeddings:
                    similarity_matrix = cosine_similarity(
                        np.array([new_embedding]), np.array([existing_embedding])
                    )
                    similarity = float(similarity_matrix[0][0])

                    if similarity >= self.valves.duplicate_threshold:
                        is_duplicate = True
                        skipped_info["duplicates"] += 1
                        logger.info(f"💡 Detected duplicate memory (similarity: {similarity:.1%}): '{op['content'][:100]}...'")
                        break

                if not is_duplicate:
                    ops_to_execute.append(op)
        else:
            ops_to_execute.extend(new_operations)

        ops_to_execute.extend(other_operations)

        skipped_count = len(operations) - len(ops_to_execute)
        if skipped_count > 0:
            logger.info(f"💡 Duplicate Detection: Skipped {skipped_count} duplicate operations")

        logger.info(f"🚀 Memory Execution: Processing {len(ops_to_execute)} operations")

        
        operation_types = []

        for op_data in ops_to_execute:
            try:
                if not isinstance(op_data, dict):
                    raise ValueError(f"Operation data must be a dict, got {type(op_data)}: {op_data}")

                if "operation" not in op_data:
                    raise ValueError(f"Operation missing 'operation' field: {op_data}")

                if op_data["operation"] in ["NEW", "UPDATE"]:
                    if "content" not in op_data or not op_data["content"]:
                        raise ValueError(f"Operation {op_data['operation']} missing content: {op_data}")
                    if not op_data["content"].strip():
                        raise ValueError(f"Operation {op_data['operation']} has empty content: {op_data}")

                if op_data["operation"] in ["UPDATE", "DELETE"]:
                    if "id" not in op_data or not op_data["id"]:
                        raise ValueError(f"Operation {op_data['operation']} missing id: {op_data}")

                memory_operation = MemoryOperation(**op_data)

                operation_type = await self._execute_single_operation(memory_operation, user)
                executed_operations.append(op_data)
                operation_types.append(operation_type)

            except Exception as e:
                logger.error(f"❌ Failed to execute operation {op_data}: {e}")
                logger.error(f"❌ Operation data details: {op_data}")
                raise

        if executed_operations:
            logger.info(f"✅ Memory Execution: Completed {len(executed_operations)} operations successfully")
            await self._smart_cache_invalidation(operation_types, user_id)

        if skipped_info["duplicates"] > 0:
            logger.info(f"💡 Memory Execution: Skipped {skipped_info['duplicates']} duplicate memories")

        return executed_operations, skipped_info

    async def _invalidate_user_cache(self, user_id: str, reason: str = "") -> None:
        """Invalidate all cache entries for a specific user with enhanced logging."""
        try:
            cache_user_key = user_id or "global"
            async with Filter._cache_lock:
                if cache_user_key in Filter._embedding_cache:
                    user_cache = Filter._embedding_cache[cache_user_key]
                    cleared_count = await user_cache.clear()

                    if cleared_count > 0:
                        logger.info(f"Cache invalidated: {cleared_count} entries cleared for user {user_id} ({reason})")
                    else:
                        logger.info(f"Cache invalidation: No entries to clear for user {user_id} ({reason})")
                else:
                    logger.info(f"Cache invalidation: No cache found for user {user_id} ({reason})")

        except Exception as e:
            logger.error(f"Failed to invalidate cache for user {user_id}: {e}")

    async def _smart_cache_invalidation(
        self, operation_types: List[str], user_id: str
    ) -> None:
        """Smart cache invalidation based on operation types."""
        
        needs_invalidation = any(op_type in ["UPDATE", "DELETE"] for op_type in operation_types)

        if not needs_invalidation:
            new_count = operation_types.count("NEW")
            logger.info(f"Smart cache: {new_count} NEW operations - no cache invalidation needed for user {user_id}")
            return

        
        update_count = operation_types.count("UPDATE")
        delete_count = operation_types.count("DELETE")
        new_count = operation_types.count("NEW")

        await self._invalidate_user_cache(user_id, f"memory changes: {update_count} updates, {delete_count} deletes")

        if new_count > 0:
            logger.info(f"Smart cache: {new_count} NEW operations had no impact on cache")

    async def _execute_single_operation(
        self, operation: MemoryOperation, user: Any
    ) -> str:
        """Execute a single memory operation with enhanced error handling and validation.

        Returns:
            Operation type for smart cache invalidation
        """
        try:
            if operation.operation == "NEW":
                if not operation.content or not operation.content.strip():
                    raise MemoryOperationError(f"Content is required for NEW operation but got: {repr(operation.content)}")

                clean_content = operation.content.strip()

                if len(clean_content) > Config.MAX_MEMORY_CONTENT_LENGTH:
                    logger.warning(
                        f"Content truncated from {len(clean_content)} to {Config.MAX_MEMORY_CONTENT_LENGTH} chars"
                    )
                    clean_content = clean_content[:Config.MAX_MEMORY_CONTENT_LENGTH].strip()

                await asyncio.wait_for(
                    asyncio.to_thread(Memories.insert_new_memory, user.id, clean_content),
                    timeout=Config.TIMEOUT_DATABASE_OPERATION,
                )
                logger.info(f"✨ NEW: Created memory - '{clean_content[:100]}...'")
                return "NEW"

            elif operation.operation == "UPDATE" and operation.id:
                if not operation.content or not operation.content.strip():
                    raise MemoryOperationError(f"Content is required for UPDATE operation but got: {repr(operation.content)}")

                clean_content = operation.content.strip()

                if len(clean_content) > Config.MAX_MEMORY_CONTENT_LENGTH:
                    logger.warning(
                        f"UPDATE content truncated from {len(clean_content)} to {Config.MAX_MEMORY_CONTENT_LENGTH} chars"
                    )
                    clean_content = clean_content[:Config.MAX_MEMORY_CONTENT_LENGTH].strip()

                await asyncio.wait_for(
                    asyncio.to_thread(
                        Memories.update_memory_by_id_and_user_id,
                        operation.id,
                        user.id,
                        clean_content
                    ),
                    timeout=Config.TIMEOUT_DATABASE_OPERATION,
                )
                logger.info(f"🔄 UPDATE: Modified {operation.id} - '{clean_content[:100]}...'")
                return "UPDATE"

            elif operation.operation == "DELETE" and operation.id:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        Memories.delete_memory_by_id_and_user_id,
                        operation.id,
                        user.id
                    ),
                    timeout=Config.TIMEOUT_USER_LOOKUP
                )
                logger.info(f"🗑️ DELETE: Removed memory {operation.id}")
                return "DELETE"

            else:
                raise MemoryOperationError(f"Unsupported or invalid operation: {operation}")

        except asyncio.TimeoutError as e:
            error_msg = f"Database timeout during {operation.operation} operation"
            logger.error(error_msg)
            raise MemoryOperationError(error_msg) from e
        except Exception as e:
            error_msg = f"Error executing {operation.operation} operation: {e}"
            logger.error(error_msg)
            logger.error(f"Operation details: {operation}")
            if hasattr(e, '__traceback__'):
                logger.error(f"Traceback: {traceback.format_exc()}")
            raise MemoryOperationError(error_msg) from e

    async def _query_llm(
        self, system_prompt: str, user_prompt: str, json_response: bool = True
    ) -> str:
        """Query the OpenAI API or compatible endpoints with enhanced error handling."""
        if not self.valves.api_key or not self.valves.api_key.strip():
            raise NeuralRecallError("API key is required but not provided.")

        session = await self._get_aiohttp_session()
        url = f"{self.valves.api_url.rstrip('/')}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.api_key}"
        }

        payload = {
            "model": self.valves.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 2048,
        }

        if json_response:
            payload["response_format"] = {"type": "json_object"}

        try:
            async with session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()

                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0]["message"]["content"]
                    return content

                raise NeuralRecallError(f"Unexpected API response format: {data}")

        except aiohttp.ClientError as e:
            error_msg = f"HTTP client error during LLM query: {e}"
            logger.error(error_msg)
            raise NeuralRecallError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error during LLM query: {e}"
            logger.error(error_msg)
            raise NeuralRecallError(error_msg) from e