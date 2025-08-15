"""
title: Neural Recall v3 - Dual Pipeline Architecture
version: 3.0.0

Complete architectural refactor implementing Retrieval Pipeline / Consolidation Pipeline model:
- Retrieval Pipeline: Synchronous retrieval and re-ranking in inlet for context injection
- Consolidation Pipeline: Asynchronous consolidation in outlet for memory management
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


class Config:
    """Centralized configuration constants for Neural Recall system."""

    CACHE_MAX_SIZE = 2500
    MAX_USER_CACHES = 500

    MAX_MESSAGE_LENGTH = 3000
    MAX_MEMORY_CONTENT_LENGTH = 500
    MIN_QUERY_LENGTH = 10

    TIMEOUT_SESSION_REQUEST = 30
    TIMEOUT_DATABASE_OPERATION = 10
    TIMEOUT_USER_LOOKUP = 5

    DEFAULT_SEMANTIC_THRESHOLD = 0.50
    
    DEFAULT_MAX_MEMORIES_RETURNED = 15 
    MIN_BATCH_SIZE = 8
    MAX_BATCH_SIZE = 32

    RETRIEVAL_MULTIPLIER = 3.0
    RETRIEVAL_TIMEOUT = 5.0
    
    CONSOLIDATION_CANDIDATE_SIZE = 50
    CONSOLIDATION_TIMEOUT = 30.0
    CONSOLIDATION_RELAXED_MULTIPLIER = 0.9

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


MEMORY_RERANKING_PROMPT = f"""You are the Retrieval Pipeline memory re-ranker. Rapidly select focused, atomic memories that will materially improve the AI's next reply.

Priority memory types (seek these first):
- Project status, deadlines, and current work context
- Skills, expertise levels, and tools the user employs  
- Personal preferences that affect recommendations (dietary, accessibility, work style)
- Ongoing challenges, goals, or learning objectives
- Relationships, roles, and professional context
- Recent events or changes in circumstances
- Location, time zone, or environmental context that impacts suggestions
- Health conditions, medications, or wellness routines
- Financial constraints or goals affecting decisions
- Family dynamics and caregiving responsibilities

Ranking rules (apply in order):
1) **Direct relevance**: prefer memories that directly answer or inform the user's question.
2) **Context boost**: prefer memories that supply missing background (preferences, recent events, open tasks).
3) **Atomic focus**: favor single-topic memories over complex multi-fact entries that contain irrelevant information.
4) **Temporal relevance**: prefer recent facts when they change the answer or context significantly.
5) **Completeness**: ensure selected memories form coherent context without gaps.
6) **Quality over quantity**: exclude vague personal trivia or marginally related facts.
7) **Safety priority**: prioritize health, safety, or crisis-related context when relevant.

Output contract:
- Return ONLY a JSON array of memory IDs (strings) ordered by relevance, highest first.
- If none materially improve the response, return an empty array.

Examples (LLM must output only the JSON arrays shown):

1) Health advice request ("I've been having trouble sleeping lately")
Candidates: ["mem-001: User has chronic insomnia and takes melatonin supplements", "mem-002: User enjoys reading mystery novels", "mem-003: User works night shifts at the hospital as a nurse", "mem-004: User drinks 3 cups of coffee daily including one at 7pm", "mem-005: User has 2-year-old daughter who wakes up frequently", "mem-006: User prefers warm baths before bedtime", "mem-007: User lives in noisy downtown apartment"]
Expected output:
["mem-001","mem-003","mem-004","mem-005","mem-007"]

2) Cooking advice request ("What should I make for dinner tonight?")
Candidates: ["mem-010: User is vegetarian with severe nut allergy, dislikes spicy food", "mem-011: User enjoys Italian cuisine and prefers quick weeknight meals", "mem-012: User works remotely from home office in Seattle", "mem-013: User bought fresh tomatoes and basil yesterday", "mem-014: User has pasta-making tools and prefers homemade meals", "mem-015: User dislikes cleaning up complex recipes on weekdays"]
Expected output:
["mem-010","mem-011","mem-015","mem-013","mem-014"]

3) Parenting guidance ("How do I handle my teenager's anxiety?")
Candidates: ["mem-020: User's favorite vacation spot is Bali", "mem-021: User has 15-year-old son with social anxiety disorder", "mem-022: User experienced anxiety during college years", "mem-023: User's son started therapy 3 months ago", "mem-024: User teaches high school mathematics", "mem-025: User's family has history of anxiety disorders"]
Expected output:
["mem-021","mem-023","mem-022","mem-025","mem-024"]

4) General conversation with no specific memory relevance
Candidates: ["mem-030: User's favorite movie is Inception", "mem-031: User works remotely from Seattle", "mem-032: User owns a golden retriever named Max"]
Expected output:
[]

5) Educational planning inquiry ("Should I pursue a master's degree?")
Candidates: ["mem-040: User has bachelor's degree in psychology from 2018", "mem-041: User currently works as school counselor for 4 years", "mem-042: User wants to specialize in child psychology", "mem-043: User has student loans of $35,000 remaining", "mem-044: User enjoys gardening on weekends", "mem-045: User's employer offers tuition reimbursement program", "mem-046: User considering online vs campus programs"]
Expected output:
["mem-040","mem-041","mem-042","mem-043","mem-045","mem-046"]

CRITICAL: Response must be valid JSON (only the array). No commentary, no code fences, no extra fields."""


MEMORY_CONSOLIDATION_PROMPT = f"""You are the Consolidation Pipeline memory consolidator. **PRESERVATION IS THE PRIMARY GOAL**. Only consolidate when absolutely necessary for organization.

⚠️ SAFETY RULES - STRICTLY ENFORCE:
- **DELETE SPARINGLY**: Only for exact duplicates or direct contradictions
- **PRESERVE INFORMATION**: When in doubt, keep memories separate  
- **MINIMAL OPERATIONS**: Bias toward NO-OP (empty array) unless clear benefit exists
- **ATOMICITY PREFERENCE**: Favor granular, single-topic memories over merged ones
- **ENRICHMENT FOCUS**: Create valuable atomic memories from conversation that enhance future interactions

🎯 PRIMARY FUNCTIONS:
1. **Memory Enrichment**: Extract atomic, valuable facts from conversation to create new memories
2. **Organization**: Only consolidate when fragmentation genuinely hinders usefulness
3. **Preservation**: Protect existing information integrity above cosmetic improvements

CONSOLIDATION CRITERIA:

**CREATE ONLY WHEN** (memory enrichment from conversation):
- Conversation reveals new atomic facts about user (preferences, skills, context, goals)
- Information is specific, actionable, and will improve future AI responses
- Facts are not already captured in existing memories
- Content follows format: "User [specific atomic fact]"

**MERGE ONLY WHEN** (all conditions must be met):
- Same specific entity (person, project, event) with fragmented facts across multiple memories
- Information is clearly complementary, not conflicting or overlapping
- Results in genuinely better organization without any information loss
- Merged result maintains atomicity (single focused topic)

**SPLIT ONLY WHEN**:
- Multiple unrelated topics are clearly bundled together inappropriately
- Different life domains are mixed (work + personal, health + hobbies)
- Splitting creates more useful, focused atomic memories

**UPDATE ONLY WHEN**:
- New information genuinely supersedes or corrects existing memory
- Temporal progression requires updating status (learning → completed)
- Clarification improves specificity without losing context

**DELETE ONLY WHEN** (extremely rare):
- Exact duplicate with identical meaning exists
- Direct factual contradiction with clear temporal precedence
- Information explicitly superseded by user correction
- User explicitly requests removal

**PREFER NO-OP** - Return [] when:
- Memories are already well-organized and atomic
- Changes would be cosmetic rather than functional
- Any doubt exists about benefit vs. risk of information loss
- Conversation contains no extractable atomic facts
- Existing memories adequately cover conversation topics

Content constraints:
- Max length: {Config.MAX_MEMORY_CONTENT_LENGTH} characters
- Start with "User" or "User's" for consistency
- Preserve all important details and context
- Maintain temporal specificity when relevant
- Focus on actionable, specific information

CONSERVATIVE EXAMPLES:

1) Memory enrichment - CREATE justified:
Conversation: "I've been taking yoga classes twice a week and really love the meditation portion"
Existing: ["mem-400: User enjoys physical exercise"]
Output: [{{"operation":"CREATE","content":"User takes yoga classes twice weekly and enjoys meditation"}}]

2) Clear fragmentation - MERGE justified:
Input: ["mem-101: User works at Lincoln Elementary", "mem-102: User is 3rd grade teacher", "mem-103: User has 8 years teaching experience"]
Output: [{{"operation":"UPDATE","id":"mem-101","content":"User works at Lincoln Elementary as 3rd grade teacher with 8 years experience"}},{{"operation":"DELETE","id":"mem-102"}},{{"operation":"DELETE","id":"mem-103"}}]

3) Topic contamination - SPLIT justified:
Input: ["mem-201: User prefers Mediterranean diet and has golden retriever named Max"]
Output: [{{"operation":"UPDATE","id":"mem-201","content":"User prefers Mediterranean diet"}},{{"operation":"CREATE","content":"User has golden retriever named Max"}}]

4) Information progression - UPDATE justified:
Input: ["mem-301: User learning French"]
Conversation: "I just passed my French B1 certification exam yesterday"
Output: [{{"operation":"UPDATE","id":"mem-301","content":"User learning French and passed B1 certification exam"}}]

5) Well-organized - NO-OP preferred:
Input: ["mem-401: User is vegetarian", "mem-402: User lives in Portland", "mem-403: User works as school nurse"]
Output: [] // Different topics, already atomic

6) Preserve separation - NO-OP preferred:
Input: ["mem-501: User has diabetes", "mem-502: User takes daily vitamins", "mem-503: User enjoys hiking"]
Output: [] // Different health/lifestyle domains, maintain granularity

7) Duplicate check — UPDATE vs NO-OP
Conversation: "I recently moved to the pediatric intensive care unit at Children's Hospital and started full-time this month"
Existing: ["mem-800: User works as pediatric nurse at Children's Hospital"]
Output: [{{"operation":"UPDATE","id":"mem-800","content":"User works full-time in the pediatric intensive care unit at Children's Hospital"}}] // Prefer UPDATE only when the change is verified; if uncertain, return []

8) Financial planning enrichment - CREATE justified:
Conversation: "We're saving for our daughter's college fund and plan to contribute $300 monthly until she's 18"
Existing: ["mem-900: User has 8-year-old daughter"]
Output: [{{"operation":"CREATE","content":"User saving for daughter's college fund with $300 monthly contributions until age 18"}}]

🔒 PRESERVATION PRINCIPLES:
- When uncertain between merge vs. separate, choose separate
- When uncertain between update vs. create new, choose create new  
- When uncertain between delete vs. keep, choose keep
- When uncertain about any operation, choose NO-OP
- Atomicity trumps organization - prefer granular over consolidated
- User context richness is more valuable than database tidiness

CRITICAL: When uncertain, choose preservation over consolidation. Return valid JSON array only."""


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
        if not text or len(text.strip()) < Config.MIN_QUERY_LENGTH:
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
            if not text or len(text.strip()) < Config.MIN_QUERY_LENGTH:
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
        
        result_embeddings = [emb for emb in result_embeddings if emb is not None]
        
        logger.info(f"🚀 Batch embedding: {len(cached_embeddings)} cached, {len(new_embeddings)} new, {len(result_embeddings)} total")
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
        return Filter._aiohttp_session

    def _should_skip_memory_operations(self, user_message: str) -> Tuple[bool, str]:
        """
        Enhanced gating for memory operations with comprehensive pattern detection.
        Skips: empty, too long, code, logs, structured data, URL dumps, or symbol spam.
        """
        if not user_message or not user_message.strip():
            return True, Config.STATUS_MESSAGES['SKIP_EMPTY']

        trimmed_message = user_message.strip()

        if len(trimmed_message) > Config.MAX_MESSAGE_LENGTH:
            return True, f"{Config.STATUS_MESSAGES['SKIP_TOO_LONG']} ({len(trimmed_message)} chars > {Config.MAX_MESSAGE_LENGTH})"

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
        
        if ((line_count >= 3 and count_lines(r'^\s*[A-Za-z0-9_]+:\s+\S+') >= max(3, line_count * 0.6)) or
            (line_count >= 4 and count_lines(r'^\s*[-\*\+]\s+.+') >= max(4, line_count * 0.6)) or
            re.search(r'^\s*[\{\[].*[\}\]]\s*$', trimmed_message, re.DOTALL)):
            return True, Config.STATUS_MESSAGES['SKIP_STRUCTURED']
        
        if len(trimmed_message) > 10 and sum(1 for c in trimmed_message if c.isalpha() or c.isspace()) / len(trimmed_message) < 0.5:
            return True, Config.STATUS_MESSAGES['SKIP_SYMBOLS']
        
        if (line_count >= 3 and 
            (sum(1 for line in message_lines if '|' in line and line.count('|') >= 2) >= max(2, line_count * 0.6) or
             count_lines(r'^\s*\d+\.\s') >= max(3, line_count * 0.6) or
             count_lines(r'^\s*[a-zA-Z]\)\s') >= max(3, line_count * 0.6))):
            return True, Config.STATUS_MESSAGES['SKIP_STRUCTURED']
        
        if line_count >= 2:
            if sum(1 for line in message_lines if re.search(r'\d{2,4}[-/]\d{1,2}[-/]\d{1,2}.*\d{1,2}:\d{2}', line)) >= max(1, line_count * 0.3):
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
        
        if len(re.findall(r'https?://[^\s]+', trimmed_message)) >= 3:
            return True, Config.STATUS_MESSAGES['SKIP_URL_DUMP']
        
        if len(trimmed_message) < Config.MIN_QUERY_LENGTH:
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
        """Emit status message through the event emitter."""
        if emitter:
            await emitter(
                {
                    "type": "status",
                    "data": {"description": message, "done": done},
                }
            )

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

    async def _llm_consolidate_memories(self, user_message: str, conversation_context: str, candidate_memories: List[Dict[str, Any]], emitter: Optional[Callable[[Any], Awaitable[None]]] = None) -> List[Dict[str, Any]]:
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
        
        context_section = f"## CONVERSATION CONTEXT\nUser Message: {user_message}\nFull Context: {conversation_context}\n\n"
        
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
        
        # Safety check: Prevent excessive deletions that could cause memory loss
        total_operations = len(operations)
        delete_operations = [op for op in operations if isinstance(op, dict) and op.get("operation") == "DELETE"]
        delete_ratio = len(delete_operations) / total_operations if total_operations > 0 else 0
        
        # If more than 50% of operations are deletions, this is likely aggressive over-consolidation
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
        
        for operation_data in operations:
            operation = MemoryOperation(**operation_data)
            result = await self._execute_single_operation(operation, user)
            if result == "CREATE":
                created_count += 1
            elif result == "UPDATE":
                updated_count += 1
            elif result == "DELETE":
                deleted_count += 1
        
        total_executed = created_count + updated_count + deleted_count
        logger.info(f"✅ Consolidation Pipeline: Executed {total_executed}/{len(operations)} operations (Created: {created_count}, Updated: {updated_count}, Deleted: {deleted_count})")
        
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
            await self._emit_status(emitter, message, True)
            
            await self._invalidate_user_cache(user_id, "consolidation")
        else:
            await self._emit_status(emitter, "✅ Memories already optimally organized", True)

    async def _consolidation_pipeline_task(self, user_message: str, conversation_context: str, user_id: str, emitter: Optional[Callable[[Any], Awaitable[None]]] = None) -> None:
        """Complete Consolidation Pipeline as async background task (formerly Slow Path)."""
        logger.info("🔧 Starting Consolidation Pipeline analysis")
        await self._emit_status(emitter, "🔧 Analyzing memory patterns...", False)
        
        candidates = await self._gather_consolidation_candidates(user_message, user_id, emitter)
        
        if not candidates:
            logger.info("🔧 Consolidation Pipeline: No candidates found")
            await self._emit_status(emitter, "💭 No consolidation candidates", True)
            return
        
        operations = await self._llm_consolidate_memories(user_message, conversation_context, candidates, emitter)
        
        if not operations:
            logger.info("🔧 Consolidation Pipeline: Memories already optimally organized - no consolidation required")
            await self._emit_status(emitter, "✅ Memories already well organized", True)
            return
        
        await self._execute_consolidation_operations(operations, user_id, emitter)
        
        logger.info("🔧 Consolidation Pipeline completed successfully")


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
                return body
                
            user_message = last_user_msg
            user_id = __user__["id"]

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

                conversation_context = ""
                for message in body["messages"]:
                    role = message.get("role", "")
                    content = self._extract_text_from_message_content(message.get("content", ""))
                    if content:
                        conversation_context += f"[{role}]: {content}\n"

                user_id = __user__["id"]

                logger.info("🔧 Starting Consolidation Pipeline background consolidation")
                task = asyncio.create_task(
                    self._consolidation_pipeline_task(user_message, conversation_context, user_id, __event_emitter__)
                )
                task.add_done_callback(
                    lambda t: logger.error(f"❌ Consolidation pipeline task failed: {t.exception()}") 
                    if t.exception() else None
                )

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
                    Filter._cache_access_order.append(user_id)

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
            "max_tokens": 2048,
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
    
    def __del__(self):
        """Destructor to ensure cleanup (production safety)."""
        try:
            if hasattr(self, '_aiohttp_session') and self._aiohttp_session and not self._aiohttp_session.closed:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self.cleanup())
                except:
                    pass
        except:
            pass
