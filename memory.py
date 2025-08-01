"""
title: Neural Recall
"""

import json
import traceback
from datetime import datetime, timezone, timedelta
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Union
import logging
import re
import asyncio
from difflib import SequenceMatcher
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import multiprocessing

import aiohttp
from fastapi.requests import Request
from pydantic import BaseModel, Field

from open_webui.routers.memories import (
    add_memory,
    AddMemoryForm,
    delete_memory_by_id,
    Memories,
)
from open_webui.models.users import Users
from open_webui.main import app as webui_app

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("NeuralRecall")

MEMORY_IDENTIFICATION_PROMPT = """You are a specialized system for extracting and structuring personal facts about a user. Your sole function is to analyze user text and output a JSON array of memory operations.

## Core Directive
- **Analyze User Input**: Scrutinize the user's message for personal, factual information.
- **Translate to English**: All extracted memories MUST be in English, regardless of the input language.
- **Output JSON Only**: Your entire response must be a single, valid JSON array (`[]`). Do not include any explanatory text, markdown, or other content.

## Memory Criteria
### Store ONLY if:
- **Factual & Personal**: Objective, verifiable facts about the user's life, identity, or personal relationships (e.g., "User is a software engineer," "User's son is named Leo").
- **Significant & Stable**: Information that is likely to be relevant for future interactions.
- **Starts with "User" or "User's"**: This is a mandatory rule for all `content` fields.
- **About the User**: Must be information that describes the user personally, not their questions or interests in external topics.

### NEVER Store:
- **Questions**: Any form of question, inquiry, or request for information (e.g., "Why did X happen?", "What is Y?", "How to do Z?").
- **General Knowledge Inquiries**: Questions about celebrities, historical events, how-to guides, or any topic not about the user.
- **System Instructions**: Prompts, commands, AI behavior requests, or formatting guidelines.
- **Temporary States**: Current mood, weather, tasks, or fleeting circumstances.
- **Speculative Content**: Plans, intentions, considerations, or "might/could/thinking about" statements.
- **Technical Content**: Code snippets, documentation, how-to guides, or troubleshooting.
- **Meta-Content**: Lists of rules, principles, formatting symbols (##, -, ×, ✓), or instructional text.
- **Casual Conversations**: Greetings, acknowledgments, reactions, or conversational responses.
- **Transient Details**: Trivial facts, temporary preferences, or context-dependent information.

## Structuring Memories: Atomicity & Grouping
- **Atomic**: Each memory should represent a single, core fact.
- **Grouped**: Combine closely related atomic facts into a single, comprehensive memory for better context.
  - **Poor (too separate)**: `{"operation": "NEW", "content": "User has a cat."}`, `{"operation": "NEW", "content": "User's cat is named Luna."}`
  - **Good (grouped)**: `{"operation": "NEW", "content": "User has a cat named Luna."}`

## Operations
- `NEW`: For entirely new facts that don't overlap with existing memories.
- `UPDATE`: To correct, expand, or make an existing memory more specific. Requires a valid `id`.
- `DELETE`: To remove information that is outdated, incorrect, or superseded by a new memory. Requires a valid `id`.

---
## Examples

### Example 1: New, Grouped Facts (from German)
**User Input**: "Mein Bruder, der in Berlin lebt, ist Anwalt. Meine Schwester ist 15 und geht noch zur Schule."
**Response**:
[
  {"operation": "NEW", "content": "User's brother is a lawyer who lives in Berlin."},
  {"operation": "NEW", "content": "User's sister is 15 years old and attends school."}
]

### Example 2: Complex Update & Deletion
**Existing Memories**:
- `{"id": "mem021", "content": "User lives in San Francisco."}`
- `{"id": "mem022", "content": "User works as a Product Manager."}`
**User Input**: "I just moved to New York last week. I also got a promotion and am now the Director of Product at my company."
**Response**:
[
  {"operation": "UPDATE", "id": "mem021", "content": "User lives in New York."},
  {"operation": "UPDATE", "id": "mem022", "content": "User is the Director of Product."}
]

### Example 3: New Fact Superseding an Old One
**Existing Memories**:
- `{"id": "mem045", "content": "User's favorite hobby is hiking."}`
**User Input**: "I haven't been hiking in years. These days, my main passion is landscape photography."
**Response**:
[
  {"operation": "DELETE", "id": "mem045"},
  {"operation": "NEW", "content": "User's main passion is landscape photography."}
]

### Example 4: Rejecting Questions While Extracting Personal Facts
**User Input**: "I work as a data scientist at Microsoft. Can you explain how neural networks work? My daughter Emma is studying computer science at Stanford."
**Response**:
[
  {"operation": "NEW", "content": "User works as a data scientist at Microsoft."},
  {"operation": "NEW", "content": "User's daughter is named Emma and studies computer science at Stanford."}
]

### Example 5: Rejecting Questions and General Knowledge
**User Input**: "Who was the first person to climb Mount Everest? I'm curious about mountaineering history."
**Response**:
[]

### Example 6: Rejecting Speculative Content
**User Input**: "I might learn guitar next year. Maybe I'll take lessons. I'm thinking about getting a Fender."
**Response**:
[]
---

## Final Check
- **Format Validation**: Is the output a valid JSON array?
- **Content Standards**: Does every memory `content` start with "User" or "User's"?
- **Question Filter**: Have ALL questions, inquiries, and requests for information been rejected?
- **Personal Information Only**: Is all content about the user personally, not their interests in external topics?
- **Logical Structure**: Are related facts grouped logically and operations correctly linked to existing `id`s?
- **Factual Filter**: Is all content factual, lasting information about the user's personal life/identity?
- **Exclusion Compliance**: Have system instructions, temporary states, and speculative content been rejected?
- **Technical Filter**: Are code snippets, documentation, formatting symbols, and meta-content excluded?
- **Relevance Test**: Is transient, trivial, or general knowledge appropriately ignored?

OUTPUT: JSON array only."""


class MemoryOperation(BaseModel):
    """Model for memory operations"""

    operation: Literal["NEW", "UPDATE", "DELETE"]
    id: Optional[str] = None
    content: Optional[str] = None


class Filter:
    _global_sentence_model = None
    _model_lock = asyncio.Lock()
    _embedding_cache = {}
    _cache_lock = asyncio.Lock()
    
    CACHE_MAX_SIZE = 2000
    
    class Valves(BaseModel):
        """Configuration valves for the filter"""

        api_url: str = Field(
            default="https://api.openai.com/v1",
            description="OpenAI API URL for memory processing.",
        )
        api_key: str = Field(
            default="",
            description="OpenAI API key (required).",
        )
        model: str = Field(
            default="gpt-4o-mini",
            description="OpenAI model to use for identifying facts to remember.",
        )
        related_memories_n: int = Field(
            default=12,
            description="Number of most relevant memories to inject into the context.",
        )
        relevance_threshold: float = Field(
            default=0.6,
            description="Baseline minimum similarity score (0-1) for a memory to be considered relevant enough to be injected.",
        )
        similarity_threshold: float = Field(
            default=0.9,
            description="Similarity threshold (0-1) for detecting and preventing duplicate memories.",
        )
        embedding_model: str = Field(
            default="Snowflake/snowflake-arctic-embed-s",
            description="Sentence transformer model for semantic similarity.",
        ) 
        timezone_hours: int = Field(
            default=0,
            description="Timezone offset in hours (e.g., 5 for UTC+5, -4 for UTC-4).",
        )

    class UserValves(BaseModel):
        enabled: bool = Field(
            default=True, description="Enable or disable the memory function."
        )

    def __init__(self):
        self.valves = self.Valves()
        self.user_valves = self.UserValves()
        self._aiohttp_session = None
        
        # Calculate optimal batch size based on CPU cores
        cpu_count = multiprocessing.cpu_count()
        # Use 2x CPU cores for batch size, with min 4 and max 32
        self._batch_size = max(4, min(32, cpu_count * 2))
        logger.info(f"Initialized batch processing with size {self._batch_size} (CPU cores: {cpu_count})")

    def get_formatted_datetime(self):
        """Get the current datetime object in the user's timezone."""
        utc_time = datetime.now(timezone.utc)
        user_timezone = timezone(timedelta(hours=self.valves.timezone_hours))
        user_time = utc_time.astimezone(user_timezone)
        return user_time

    async def _get_sentence_model(self) -> SentenceTransformer:
        """Get or initialize the sentence transformer model using singleton pattern."""
        async with Filter._model_lock:
            if Filter._global_sentence_model is None:
                try:
                    logger.info(f"Loading sentence transformer model: {self.valves.embedding_model}")
                    Filter._global_sentence_model = await asyncio.to_thread(
                        SentenceTransformer, self.valves.embedding_model, trust_remote_code=True
                    )
                    logger.info("Sentence transformer model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load sentence transformer model: {e}")
                    try:
                        logger.info("Attempting to load fallback model: all-MiniLM-L6-v2")
                        Filter._global_sentence_model = await asyncio.to_thread(
                            SentenceTransformer, "all-MiniLM-L6-v2", trust_remote_code=True
                        )
                        logger.info("Fallback model loaded successfully")
                    except Exception as fallback_error:
                        logger.error(f"Failed to load fallback model: {fallback_error}")
                        raise RuntimeError("Could not load any sentence transformer model")
        return Filter._global_sentence_model

    async def _get_aiohttp_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._aiohttp_session is None or self._aiohttp_session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            self._aiohttp_session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={"User-Agent": "Neural-Recall/1.0"}
            )
        return self._aiohttp_session

    async def _ensure_session_closed(self):
        """Ensure the aiohttp session is properly closed."""
        if self._aiohttp_session and not self._aiohttp_session.closed:
            try:
                await self._aiohttp_session.close()
            except Exception as e:
                logger.warning(f"Error closing aiohttp session: {e}")
            finally:
                self._aiohttp_session = None

    def _generate_cache_key(self, text: str, user_id: str = None) -> str:
        """Generate a cache key for text embedding."""
        # Use text content and model name for cache key
        cache_content = f"{text}|{self.valves.embedding_model}"
        if user_id:
            cache_content = f"{user_id}|{cache_content}"
        return hashlib.md5(cache_content.encode()).hexdigest()

    async def _get_cached_embedding(self, text: str, user_id: str = None) -> Optional[np.ndarray]:
        """Get cached embedding for text if available."""
        cache_key = self._generate_cache_key(text, user_id)
        async with Filter._cache_lock:
            user_cache = Filter._embedding_cache.get(user_id or "global", {})
            if cache_key in user_cache:
                logger.debug(f"Cache hit for text: {text[:50]}...")
                return user_cache[cache_key]["embedding"]
        return None

    async def _cache_embedding(self, text: str, embedding: np.ndarray, user_id: str = None) -> None:
        """Cache embedding for text."""
        cache_key = self._generate_cache_key(text, user_id)
        cache_user_key = user_id or "global"
        
        async with Filter._cache_lock:
            if cache_user_key not in Filter._embedding_cache:
                Filter._embedding_cache[cache_user_key] = {}
            
            user_cache = Filter._embedding_cache[cache_user_key]
            
            if len(user_cache) >= self.CACHE_MAX_SIZE:
                entries_to_remove = max(1, len(user_cache) // 5)
                sorted_entries = sorted(user_cache.items(), key=lambda x: x[1]["timestamp"])
                for old_key, _ in sorted_entries[:entries_to_remove]:
                    del user_cache[old_key]
                logger.debug(f"Cleaned {entries_to_remove} old cache entries for user {cache_user_key}")
            
            user_cache[cache_key] = {
                "embedding": embedding,
                "timestamp": datetime.now().timestamp(),
                "text_preview": text[:50]
            }
            logger.debug(f"Cached embedding for text: {text[:50]}...")

    async def _get_embedding_with_cache(self, text: str, user_id: str = None) -> np.ndarray:
        """Get embedding for text, using cache if available."""
        cached_embedding = await self._get_cached_embedding(text, user_id)
        if cached_embedding is not None:
            return cached_embedding
        
        try:
            model = await self._get_sentence_model()
            embedding = await asyncio.to_thread(model.encode, [text])
            embedding_array = embedding[0]
            
            await self._cache_embedding(text, embedding_array, user_id)
            
            return embedding_array
        except Exception as e:
            logger.error(f"Error generating embedding for text: {e}")
            raise

    async def _get_embeddings_batch(self, texts: List[str], user_id: str = None) -> List[np.ndarray]:
        """Get embeddings for multiple texts using batch processing and caching."""
        if not texts:
            return []
        
        # Separate cached and non-cached texts
        cached_embeddings = {}
        texts_to_process = []
        text_indices = {}
        
        for i, text in enumerate(texts):
            cached_embedding = await self._get_cached_embedding(text, user_id)
            if cached_embedding is not None:
                cached_embeddings[i] = cached_embedding
            else:
                text_indices[len(texts_to_process)] = i
                texts_to_process.append(text)
        
        logger.debug(f"Batch processing: {len(cached_embeddings)} cached, {len(texts_to_process)} to process")
        
        # Process remaining texts in batches
        new_embeddings = {}
        if texts_to_process:
            try:
                model = await self._get_sentence_model()
                
                # Process in batches for better CPU utilization
                for batch_start in range(0, len(texts_to_process), self._batch_size):
                    batch_end = min(batch_start + self._batch_size, len(texts_to_process))
                    batch_texts = texts_to_process[batch_start:batch_end]
                    
                    # Generate embeddings for the batch
                    batch_embeddings = await asyncio.to_thread(model.encode, batch_texts)
                    
                    # Store results and cache them
                    for j, embedding in enumerate(batch_embeddings):
                        original_index = text_indices[batch_start + j]
                        new_embeddings[original_index] = embedding
                        
                        # Cache the embedding
                        await self._cache_embedding(batch_texts[j], embedding, user_id)
                
                logger.debug(f"Generated {len(new_embeddings)} new embeddings in batches of {self._batch_size}")
            
            except Exception as e:
                logger.error(f"Error in batch embedding generation: {e}")
                raise
        
        # Combine cached and new embeddings in original order
        result = []
        for i in range(len(texts)):
            if i in cached_embeddings:
                result.append(cached_embeddings[i])
            elif i in new_embeddings:
                result.append(new_embeddings[i])
            else:
                raise ValueError(f"Missing embedding for text at index {i}")
        
        return result

    async def _get_cache_stats(self, user_id: str = None) -> str:
        """Get cache statistics for debugging."""
        cache_user_key = user_id or "global"
        async with Filter._cache_lock:
            user_cache = Filter._embedding_cache.get(cache_user_key, {})
            total_cache_size = sum(len(cache) for cache in Filter._embedding_cache.values())
            return f"user: {len(user_cache)}, total: {total_cache_size}"

    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process incoming messages to inject relevant memories into the context."""
        if not body or not __user__ or not self.user_valves.enabled:
            return body

        try:
            if "messages" in body and body["messages"]:
                user_message = body["messages"][-1]["content"]
                user_id = __user__["id"]

                await self._emit_status(__event_emitter__, "🔍 Searching memory vault...", False)

                relevant_memories = await self.get_relevant_memories(
                    user_message, user_id
                )

                if relevant_memories:
                    count = len(relevant_memories)
                    memory_word = "memory" if count == 1 else "memories"
                    await self._emit_status(__event_emitter__, f"💡 Found {count} relevant {memory_word}")
                else:
                    await self._emit_status(__event_emitter__, "💭 No relevant memories found")

                if relevant_memories:
                    self._inject_memories_into_context(body, relevant_memories)

                self._inject_datetime_context(body)

        except Exception as e:
            logger.error(f"Error in inlet: {e}\n{traceback.format_exc()}")
            await self._emit_status(__event_emitter__, f"🙈 Error retrieving memories: {str(e)}")
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

        try:
            if "messages" in body and body["messages"]:
                user_message = next(
                    (
                        m["content"]
                        for m in reversed(body["messages"])
                        if m["role"] == "user"
                    ),
                    None,
                )

                logger.debug(f"User message for memory processing: {user_message}")
                if user_message:
                    await self._emit_status(__event_emitter__, "🧪 Analyzing message for memorable facts...", False)

                    try:
                        await asyncio.wait_for(
                            self._process_user_memories(
                                user_message,
                                __user__["id"],
                                __event_emitter__,
                            ),
                            timeout=15.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning("Memory processing timed out, continuing without waiting")
        except Exception as e:
            logger.error(f"Error in outlet: {e}\n{traceback.format_exc()}")
            await self._emit_status(__event_emitter__, f"🙈 Error processing memories: {str(e)}")
        return body

    async def _emit_status(
        self,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]],
        description: str,
        done: bool = True,
    ) -> None:
        """Helper method to emit status messages."""
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
                logger.error(f"Error in event emitter: {e}")

    def _format_operation_status(self, executed_ops: List[Dict[str, Any]]) -> str:
        """Format memory operations into a fancy status message."""
        if not executed_ops:
            return "🙈 No memory operations executed"
        
        op_emojis = {
            "NEW": "✨",
            "UPDATE": "🔄", 
            "DELETE": "🗑️"
        }
        
        op_actions = {
            "NEW": "Created",
            "UPDATE": "Updated",
            "DELETE": "Removed"
        }
        
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
        """Get all memories for a user from the database."""
        try:
            user_memories = await asyncio.wait_for(
                asyncio.to_thread(Memories.get_memories_by_user_id, user_id=str(user_id)),
                timeout=10.0
            )
            memories_list = [
                {
                    "id": str(memory.id),
                    "memory": memory.content,
                }
                for memory in user_memories
            ]
            logger.debug(f"Retrieved {len(memories_list)} memories for user {user_id}")
            return memories_list
        except asyncio.TimeoutError:
            logger.error(f"Database timeout while retrieving memories for user {user_id}")
            return []
        except Exception as e:
            logger.error(
                f"Error getting formatted memories for user {user_id}: {e}\n{traceback.format_exc()}"
            )
            return []

    def _inject_memories_into_context(
        self, body: Dict[str, Any], memories: List[Dict[str, Any]]
    ) -> None:
        """Inject relevant memories into the system message of the chat body."""
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
        """Inject current date/time context into the system message."""
        current_time = self.get_formatted_datetime()
        
        timezone_offset = self.valves.timezone_hours
        if timezone_offset >= 0:
            timezone_str = f"UTC+{timezone_offset}" if timezone_offset > 0 else "UTC"
        else:
            timezone_str = f"UTC{timezone_offset}"
        
        formatted_date = current_time.strftime("%B %d, %Y")
        formatted_time = current_time.strftime("%H:%M:%S")
        day_of_week = current_time.strftime("%A")
        
        datetime_context = f"CURRENT TIME: {day_of_week}, {formatted_date}, {formatted_time} {timezone_str}"
        
        system_message_found = False
        for message in body["messages"]:
            if message["role"] == "system":
                message["content"] = f"{message['content']}\n\n{datetime_context}"
                system_message_found = True
                break

        if not system_message_found:
            body["messages"].insert(0, {"role": "system", "content": datetime_context})

    def _format_memories_for_context(self, memories: List[Dict[str, Any]]) -> str:
        """Format a list of memories into a string for LLM context."""
        if not memories:
            return ""

        header = "BACKGROUND: You naturally know these facts. Never reference having this information or where it came from:"
        formatted_mems = "\n".join([f"- {mem['memory']}" for mem in memories])
        return f"{header}\n{formatted_mems}"

    async def _process_user_memories(
        self,
        user_message: str,
        user_id: str,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]],
    ) -> List[Dict[str, Any]]:
        """Identify and execute memory operations based on the user's message."""
        try:
            logger.debug(f"Processing memories for message: {user_message[:50]}...")
            existing_memories = await self._get_formatted_memories(user_id)
            operations = await self.identify_memory_operations(
                user_message, existing_memories
            )

            if operations:
                logger.info(f"Found {len(operations)} memory operations to process.")
                executed_ops = await self.execute_memory_operations(
                    operations, user_id, existing_memories
                )
                
                if executed_ops:
                    status_message = self._format_operation_status(executed_ops)
                    await self._emit_status(__event_emitter__, status_message)
                else:
                    await self._emit_status(__event_emitter__, "🙈 Failed to update memories")
                return executed_ops
            else:
                logger.debug("No new memory operations identified.")
                await self._emit_status(__event_emitter__, "💭 No memorable facts found")
            return []
        except asyncio.CancelledError:
            logger.info("Memory processing task was cancelled")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in memory processing: {e}\n{traceback.format_exc()}")
            return []

    async def identify_memory_operations(
        self,
        input_text: str,
        existing_memories: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Use an LLM to identify memory operations from the input text."""
        logger.debug("Starting memory identification.")
        system_prompt = MEMORY_IDENTIFICATION_PROMPT

        if existing_memories:
            mem_context = "\n\nExisting Memories:\n" + json.dumps(
                existing_memories, indent=2
            )
            system_prompt += mem_context

        date_context = f"\n\nCurrent Date/Time: {self.get_formatted_datetime().strftime('%A, %B %d, %Y %H:%M:%S %z')}"
        system_prompt += date_context

        response = await self._query_llm(system_prompt, input_text)
        if not response or response.startswith("Error:"):
            logger.error(f"Error from LLM during memory identification: {response}")
            return []

        logger.info(f"LLM response for memory identification: {response}")
        operations = self._extract_and_parse_json(response)

        if isinstance(operations, dict):
            logger.info(f"LLM returned single operation object, converting to list: {operations}")
            operations = [operations]

        if not isinstance(operations, list):
            logger.warning(
                f"LLM did not return a valid list of operations: {operations}"
            )
            return []

        logger.info(f"Parsed {len(operations)} operations from LLM response")

        valid_ops = [
            op
            for op in operations
            if self._validate_memory_operation(op, existing_memories)
        ]
        
        rejected_count = len(operations) - len(valid_ops)
        if rejected_count > 0:
            logger.info(f"Rejected {rejected_count} invalid memory operations (validation failed)")
        
        logger.info(f"Identified {len(valid_ops)} valid memory operations.")
        return valid_ops

    def _validate_memory_operation(
        self, op: Dict[str, Any], existing_memories: List[Dict[str, Any]]
    ) -> bool:
        """Validate a single memory operation."""
        try:
            MemoryOperation(**op)
            
            if op["operation"] in ["NEW", "UPDATE"] and op.get("content"):
                content = op["content"].strip()
                if not (content.startswith("User ") or content.startswith("User's ")):
                    logger.warning(
                        f"Memory content must start with 'User' or 'User's': {content[:50]}..."
                    )
                    return False
            
            if op["operation"] in ["UPDATE", "DELETE"]:
                existing_ids = {mem["id"] for mem in existing_memories}
                if op.get("id") not in existing_ids:
                    logger.warning(
                        f"Invalid or missing ID for {op['operation']} on non-existent memory: {op.get('id')}"
                    )
                    return False
            return True
        except Exception as e:
            logger.warning(f"Invalid memory operation format: {op}. Error: {e}")
            return False

    def _extract_and_parse_json(self, text: str) -> Union[List, Dict, None]:
        """Extract and parse JSON from text, handling common LLM response issues."""
        if not text:
            logger.warning("Empty text provided to JSON parser")
            return None

        original_text = text
        
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1)

        for start_char, end_char in [("[", "]"), ("{", "}")]:
            start_idx = text.find(start_char)
            if start_idx != -1:
                bracket_count = 0
                for i, char in enumerate(text[start_idx:], start_idx):
                    if char == start_char:
                        bracket_count += 1
                    elif char == end_char:
                        bracket_count -= 1
                        if bracket_count == 0:
                            json_text = text[start_idx:i+1]
                            try:
                                result = json.loads(json_text)
                                logger.info(f"Successfully parsed JSON: {type(result)}")
                                return result
                            except json.JSONDecodeError:
                                continue
        
        logger.error(f"Failed to parse JSON from: {original_text[:500]}...")
        return None

    async def _calculate_memory_similarity(self, text1: str, text2: str, user_id: str = None) -> float:
        """Calculate semantic similarity between two strings using cached sentence transformers."""
        try:
            embedding1 = await self._get_embedding_with_cache(text1, user_id)
            embedding2 = await self._get_embedding_with_cache(text2, user_id)
            
            similarity_matrix = cosine_similarity([embedding1], [embedding2])
            return float(max(0, min(1, similarity_matrix[0][0])))
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity, falling back to string similarity: {e}")
            return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _calculate_dynamic_threshold(self, scored_memories: List[Dict[str, Any]]) -> float:
        """Calculate a dynamic threshold based on score distribution while maintaining a baseline minimum."""
        if not scored_memories:
            return self.valves.relevance_threshold
        
        baseline_threshold = self.valves.relevance_threshold
        scores = [mem["relevance"] for mem in scored_memories]
        
        if len(scores) <= self.valves.related_memories_n:
            return baseline_threshold
        
        top_n_scores = scores[:self.valves.related_memories_n]
        avg_top_n = sum(top_n_scores) / len(top_n_scores)
        max_score = max(scores)
        
        dynamic_threshold = max(baseline_threshold, avg_top_n - 0.05)
        
        score_gap = max_score - avg_top_n
        if score_gap > 0.2: 
            dynamic_threshold = max(baseline_threshold, avg_top_n + 0.05)
        
        return min(dynamic_threshold, 0.95) 

    async def get_relevant_memories(
        self, current_message: str, user_id: str
    ) -> List[Dict[str, Any]]:
        """Get memories relevant to the current context using cached sentence similarity with dynamic threshold."""
        existing_memories = await self._get_formatted_memories(user_id)
        if not existing_memories:
            return []

        # Prepare texts for batch processing
        memory_texts = [mem["memory"] for mem in existing_memories]
        all_texts = [current_message] + memory_texts
        
        # Get embeddings in batch (much faster for many memories)
        try:
            all_embeddings = await self._get_embeddings_batch(all_texts, user_id)
            current_embedding = all_embeddings[0]
            memory_embeddings = all_embeddings[1:]
        except Exception as e:
            logger.warning(f"Batch processing failed, falling back to individual processing: {e}")
            # Fallback to individual processing
            scored_memories = []
            for mem in existing_memories:
                similarity = await self._calculate_memory_similarity(
                    current_message, mem["memory"], user_id
                )
                scored_memories.append({**mem, "relevance": similarity})
        else:
            # Calculate similarities using batch embeddings
            scored_memories = []
            for i, mem in enumerate(existing_memories):
                try:
                    similarity_matrix = cosine_similarity([current_embedding], [memory_embeddings[i]])
                    similarity = float(max(0, min(1, similarity_matrix[0][0])))
                except Exception as e:
                    logger.warning(f"Error calculating similarity for memory {i}, using fallback: {e}")
                    similarity = SequenceMatcher(None, current_message.lower(), mem["memory"].lower()).ratio()
                
                scored_memories.append({**mem, "relevance": similarity})

        scored_memories.sort(key=lambda x: x["relevance"], reverse=True)
        
        dynamic_threshold = self._calculate_dynamic_threshold(scored_memories)
        
        filtered_memories = [
            mem for mem in scored_memories 
            if mem["relevance"] >= dynamic_threshold
        ]
        
        result = filtered_memories[:self.valves.related_memories_n]

        cache_stats = await self._get_cache_stats(user_id)
        logger.info(
            f"Dynamic threshold: {dynamic_threshold:.3f} (baseline: {self.valves.relevance_threshold:.3f}). "
            f"Found {len(filtered_memories)} relevant memories above threshold. Returning top {len(result)}. "
            f"Cache stats: {cache_stats}"
        )
        return result

    async def execute_memory_operations(
        self,
        operations: List[Dict[str, Any]],
        user_id: str,
        existing_memories: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Execute a list of memory operations (NEW, UPDATE, DELETE)."""
        executed_operations = []
        try:
            user = await asyncio.wait_for(
                asyncio.to_thread(Users.get_user_by_id, user_id),
                timeout=5.0
            )
            if not user:
                logger.error(f"User not found: {user_id}")
                return executed_operations

            ops_to_execute = []
            existing_contents = {mem["memory"] for mem in existing_memories}
            
            # Collect all NEW operations for batch processing
            new_operations = [op for op in operations if op["operation"] == "NEW"]
            other_operations = [op for op in operations if op["operation"] != "NEW"]
            
            # Process NEW operations with batch similarity checking
            if new_operations and existing_contents:
                try:
                    # Prepare texts for batch processing
                    new_contents = [op["content"] for op in new_operations]
                    existing_list = list(existing_contents)
                    
                    # Get embeddings for all new and existing memories
                    all_texts = new_contents + existing_list
                    all_embeddings = await self._get_embeddings_batch(all_texts, user_id)
                    
                    new_embeddings = all_embeddings[:len(new_contents)]
                    existing_embeddings = all_embeddings[len(new_contents):]
                    
                    # Check each new operation for duplicates
                    for i, op in enumerate(new_operations):
                        is_duplicate = False
                        new_embedding = new_embeddings[i]
                        
                        for existing_embedding in existing_embeddings:
                            similarity_matrix = cosine_similarity([new_embedding], [existing_embedding])
                            similarity = float(similarity_matrix[0][0])
                            
                            if similarity >= self.valves.similarity_threshold:
                                is_duplicate = True
                                break
                        
                        if is_duplicate:
                            logger.debug(f"Skipping duplicate new memory: {op['content'][:50]}...")
                        else:
                            ops_to_execute.append(op)
                            
                except Exception as e:
                    logger.warning(f"Batch duplicate detection failed, falling back to individual: {e}")
                    # Fallback to individual processing
                    for op in new_operations:
                        is_duplicate = False
                        for existing_content in existing_contents:
                            similarity = await self._calculate_memory_similarity(
                                op["content"], existing_content, user_id
                            )
                            if similarity >= self.valves.similarity_threshold:
                                is_duplicate = True
                                break
                        
                        if is_duplicate:
                            logger.debug(f"Skipping duplicate new memory: {op['content'][:50]}...")
                        else:
                            ops_to_execute.append(op)
            else:
                # No existing memories to compare against, add all NEW operations
                ops_to_execute.extend(new_operations)
            
            # Add all non-NEW operations (no duplicate checking needed)
            ops_to_execute.extend(other_operations)
            
            logger.debug(
                f"Executing {len(ops_to_execute)} memory operations after filtering."
            )

            for op_data in ops_to_execute:
                success = await self._execute_single_operation(MemoryOperation(**op_data), user)
                if success:
                    executed_operations.append(op_data)

            logger.info(
                f"Successfully processed {len(executed_operations)} memory operations."
            )
            return executed_operations
        except asyncio.TimeoutError:
            logger.error(f"Database timeout while processing memory operations for user {user_id}")
            return executed_operations
        except Exception as e:
            logger.error(f"Error processing memories for user {user_id}: {e}\n{traceback.format_exc()}")
            return executed_operations

    async def _execute_single_operation(
        self, operation: MemoryOperation, user: Any
    ) -> bool:
        """Execute a single memory operation with improved error handling."""
        try:
            request = Request(scope={"type": "http", "app": webui_app})
            
            if operation.operation == "NEW":
                await asyncio.wait_for(
                    add_memory(
                        request=request,
                        form_data=AddMemoryForm(content=operation.content),
                        user=user,
                    ),
                    timeout=10.0
                )
                logger.info(f"NEW memory created: {operation.content[:50]}...")

            elif operation.operation == "UPDATE" and operation.id:
                await asyncio.wait_for(
                    delete_memory_by_id(operation.id, user=user),
                    timeout=5.0
                )
                await asyncio.wait_for(
                    add_memory(
                        request=request,
                        form_data=AddMemoryForm(content=operation.content),
                        user=user,
                    ),
                    timeout=10.0
                )
                logger.info(f"UPDATE memory {operation.id}: {operation.content[:50]}...")

            elif operation.operation == "DELETE" and operation.id:
                await asyncio.wait_for(
                    delete_memory_by_id(operation.id, user=user),
                    timeout=5.0
                )
                logger.info(f"DELETE memory {operation.id}")

            return True
        except asyncio.TimeoutError:
            logger.error(f"Database timeout executing {operation.operation} for ID {operation.id}")
            return False
        except Exception as e:
            logger.error(
                f"Error executing {operation.operation} for ID {operation.id}: {e}\n{traceback.format_exc()}"
            )
            return False

    async def _query_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Query the OpenAI API with improved error handling."""
        if not self.valves.api_key or not self.valves.api_key.strip():
            return "Error: API key is required but not provided."
            
        session = None
        try:
            session = await self._get_aiohttp_session()
            
            url = f"{self.valves.api_url.rstrip('/')}/chat/completions"
            headers = {"Content-Type": "application/json"}
            
            if self.valves.api_key:
                headers["Authorization"] = f"Bearer {self.valves.api_key}"
                    
            payload = {
                "model": self.valves.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "top_p": 0.8,
                "max_tokens": 4096,
                "response_format": {"type": "json_object"}
            }

            async with session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
                return f"Error: Unexpected OpenAI response format: {data}"
                
        except asyncio.TimeoutError:
            return "Error: OpenAI API request timed out."
        except aiohttp.ClientError as e:
            if session and session.closed:
                self._aiohttp_session = None
            return f"Error: OpenAI API connection error: {e}"
        except aiohttp.ServerTimeoutError:
            return "Error: OpenAI server timeout."
        except Exception as e:
            return f"Error: OpenAI API query failed: {e}"

    async def cleanup(self):
        """Clean up resources on shutdown."""
        logger.info("Cleaning up Neural Recall resources...")
        
        await self._ensure_session_closed()
        
        async with Filter._cache_lock:
            cache_size = sum(len(cache) for cache in Filter._embedding_cache.values())
            Filter._embedding_cache.clear()
            logger.info(f"Cleared embedding cache ({cache_size} entries)")
        
        logger.info("Cleanup complete.")
