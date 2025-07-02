import re
import asyncio
import logging
import math
from typing import Any, List, Dict, Optional
from datetime import datetime
from collections import Counter, OrderedDict
import os
import torch

from open_webui.models.memories import Memories
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util as st_util
import unicodedata

logger = logging.getLogger("openwebui.memory")


class MemoryValidationError(Exception):
    def __init__(self, message, payload=None):
        super().__init__(message)
        self.payload = payload or {"error": message}


class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = max(capacity, 1)

    def get(self, key: Any) -> Any:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key: Any, value: Any):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            evicted_key, _ = self.cache.popitem(last=False)
            logger.info(f"Cache evicted: {evicted_key}")

    def pop(self, key: Any, default: Any = None) -> Any:
        return self.cache.pop(key, default)


class Tools:
    class Valves(BaseModel):
        USE_MEMORY: bool = Field(
            default=True,
            description="Enable or disable all memory features globally. If False, memory is not stored or retrieved.",
        )
        SIMILARITY_BASE_THRESHOLD: float = Field(
            default=0.80,
            description="Base similarity score (0-1) for memory comparison. Actual thresholds are dynamically calculated from this value and input length.",
        )
        DUPLICATE_THRESHOLD_MULTIPLIER: float = Field(
            default=1.2,
            description="A memory is a duplicate if its similarity is this multiple of the dynamic threshold or higher.",
        )
        RETRIEVE_THRESHOLD_MULTIPLIER: float = Field(
            default=0.7,
            description="Multiplier for the dynamic threshold to make retrieval more inclusive. Lower values return more results.",
        )
        MAX_RELEVANT_CONTEXT_RESULTS: int = Field(
            default=20,
            description="Max number of relevant memories to return per query.",
        )
        GLOBAL_EMBEDDING_CACHE_SIZE: int = Field(
            default=4000,
            description="Max number of memory embeddings to cache globally across all users.",
        )
        SENTENCE_TRANSFORMER_MODEL: str = Field(
            default="Alibaba-NLP/gte-multilingual-base",
            description="HuggingFace model name for SentenceTransformer.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self._st_model: Optional[SentenceTransformer] = None
        self._st_model_lock = asyncio.Lock()
        self._embedding_cache = LRUCache(self.valves.GLOBAL_EMBEDDING_CACHE_SIZE)
        self._user_locks: Dict[str, asyncio.Lock] = {}
        self._embedding_semaphore = asyncio.Semaphore(max(1, os.cpu_count() // 2))

    async def _get_st_model(self) -> SentenceTransformer:
        if self._st_model is None:
            async with self._st_model_lock:
                if self._st_model is None:
                    logger.info("Loading SentenceTransformer model...")
                    self._st_model = await asyncio.get_running_loop().run_in_executor(
                        None,
                        lambda: SentenceTransformer(
                            self.valves.SENTENCE_TRANSFORMER_MODEL,
                            device="cpu",
                            trust_remote_code=True,
                        ),
                    )
                    logger.info("SentenceTransformer model loaded successfully.")
        return self._st_model

    async def _get_user_lock(self, user_id: str) -> asyncio.Lock:
        if user_id not in self._user_locks:
            self._user_locks[user_id] = asyncio.Lock()
        return self._user_locks[user_id]

    def _sanitize_input(self, text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = re.sub(r"^(user's |user )", "", text, flags=re.IGNORECASE)
        text = re.sub(r"[^\w\s.'’]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    async def _check_user(self, __user__: Optional[dict]) -> str:
        if not self.valves.USE_MEMORY:
            raise MemoryValidationError("Memory feature is disabled.")
        user_id = str((__user__ or {}).get("id") or "").strip()
        if not user_id:
            raise MemoryValidationError("Invalid or missing User ID.")
        return user_id

    async def _db_call(self, func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    async def _emit_event(self, emitter: Optional[Any], description: str, status: str, done: bool = False):
        logger.info(f"Event: {status} - {description}")
        if emitter:
            try:
                event_data = {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
                (await emitter(event_data) if asyncio.iscoroutinefunction(emitter) else emitter(event_data))
            except Exception as e:
                logger.warning(f"Failed to emit event: {e}")

    async def _error_response(self, emitter: Optional[Any], msg: str, payload: Optional[dict] = None) -> dict:
        logger.error(f"Error: {msg}")
        await self._emit_event(emitter, f"❌ {msg}", "error", True)
        return payload or {"error": msg}

    async def _get_user_memories(self, user_id: str) -> List[Any]:
        logger.info(f"Fetching memories for user {user_id}")
        memories = await self._db_call(Memories.get_memories_by_user_id, user_id)
        valid_memories = [mem for mem in memories or [] if getattr(mem, "id") and getattr(mem, "content")]
        return sorted(valid_memories, key=lambda m: m.id)

    async def _get_or_create_embeddings(self, user_id: str, memories: List[Any]) -> Optional[torch.Tensor]:
        if not memories:
            return None

        st_model = await self._get_st_model()

        cached_embeddings = {str(mem.id): self._embedding_cache.get((user_id, str(mem.id))) for mem in memories}
        uncached_mems = [mem for mem in memories if cached_embeddings.get(str(mem.id)) is None]
        hit_count = len(memories) - len(uncached_mems)
        if hit_count > 0:
            logger.info(f"Cache hits for user {user_id}: {hit_count} out of {len(memories)}")

        if uncached_mems:
            logger.info(f"Cache miss for {len(uncached_mems)} memories for user {user_id}. Computing now.")
            try:
                texts = [self._sanitize_input(m.content).lower() for m in uncached_mems]
                async with self._embedding_semaphore:
                    new_embeddings = await asyncio.get_running_loop().run_in_executor(
                        None,
                        lambda: st_model.encode(texts, convert_to_tensor=True, show_progress_bar=False),
                    )

                for mem, emb in zip(uncached_mems, new_embeddings):
                    cache_key = (user_id, str(mem.id))
                    self._embedding_cache.set(cache_key, emb)
                    cached_embeddings[str(mem.id)] = emb
            except Exception as e:
                logger.error(f"Error computing embeddings for user {user_id}: {e}. Some memories will be skipped.")

        embedding_dim = st_model.get_sentence_embedding_dimension()
        device = st_model.device
        final_embeddings = [cached_embeddings.get(str(mem.id), torch.zeros(embedding_dim, device=device)) for mem in memories]

        return torch.stack(final_embeddings) if final_embeddings else None

    async def _find_similar_memories(self, query: str, memories: List[Any], threshold: float, user_id: str) -> List[Dict[str, Any]]:
        if not memories or not query:
            return []

        query_for_model = self._sanitize_input(query).lower()

        try:
            st_model = await self._get_st_model()
            memory_embeddings = await self._get_or_create_embeddings(user_id, memories)
            if memory_embeddings is None:
                return []

            async with self._embedding_semaphore:
                query_embedding = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: st_model.encode(
                        [query_for_model],
                        convert_to_tensor=True,
                        show_progress_bar=False,
                    ),
                )

            scores = st_util.cos_sim(query_embedding[0], memory_embeddings)[0]

            results = [{"score": score.item(), "memory": memories[i]} for i, score in enumerate(scores) if score.item() > threshold]
            return sorted(results, key=lambda x: x["score"], reverse=True)
        except Exception as e:
            logger.error(f"Error during semantic search for user {user_id}: {e}")
            return []

    async def _process_memory_item(self, item_raw: str, user_id: str, existing_memories: List[Any]) -> Dict[str, Any]:
        content = item_raw
        details = {"input_content": item_raw, "content": content}
        if not content or len(content.split()) < 3:
            return {"operation": "skip", "reason": "input_too_short", **details}

        try:
            update_threshold, duplicate_threshold = self._calculate_dynamic_thresholds(content)
            results = await self._find_similar_memories(content, existing_memories, update_threshold, user_id)

            thresholds = {
                "update": round(update_threshold, 3),
                "duplicate": round(duplicate_threshold, 3),
            }

            if not results:
                new_mem = await self._db_call(Memories.insert_new_memory, user_id=user_id, content=content)
                return (
                    {"operation": "add", "new_memory": new_mem, **details}
                    if getattr(new_mem, "id", None)
                    else {"operation": "skip", "reason": "add_db_error", **details}
                )

            best_match = results[0]
            sim_score = best_match["score"]

            if sim_score >= duplicate_threshold:
                return {
                    "operation": "skip",
                    "reason": "duplicate_in_db",
                    "similarity_score": round(sim_score, 3),
                    "thresholds": thresholds,
                    **details,
                }

            similar_memory_obj = best_match["memory"]
            updated = await self._db_call(
                Memories.update_memory_by_id_and_user_id,
                similar_memory_obj.id,
                user_id,
                content,
            )
            if updated:
                return {
                    "operation": "update",
                    "updated_memory_id": similar_memory_obj.id,
                    "old_content": similar_memory_obj.content,
                    "similarity_score": round(sim_score, 3),
                    "thresholds": thresholds,
                    **details,
                }

            return {"operation": "skip", "reason": "update_db_error", **details}
        except Exception as e:
            return {
                "operation": "skip",
                "reason": "processing_error",
                "error": str(e),
                **details,
            }

    async def add_memories(
        self,
        input_data: List[str],
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Any] = None,
    ) -> dict:
        """
        Store only critical, stable, and significant facts worth remembering. Be extremely selective—reject any temporary, subjective, or speculative information. Always split complex information into single, atomic facts.

        Args:
            input_data (List[str]): List of atomic memory strings to store. Each MUST be a single, distinct, indivisible fact in English. REQUIRED.
            __user__ (Optional[dict], optional): User context provided automatically by the system.
            __event_emitter__ (Optional[Callable], optional): Callback for status updates.

        Returns:
            dict: JSON response with summary counts (added, updated, skipped), actions performed, and current date. If no valid input, returns a JSON error message.

        WHEN TO USE (Store ONLY if ALL criteria are met):
            - Information MUST be factual, objective, and verifiable
            - Information MUST be stable and unlikely to change
            - Information MUST be significant for future interactions
            - Information MUST be non-obvious and not easily inferred

        WHAT NOT TO STORE:
            - Ephemeral or temporary information (e.g., daily tasks, current mood, weather)
            - Subjective content (e.g., opinions, feelings, worries)
            - Speculative or hypothetical information ("might", "considering getting a dog", "had pasta for lunch")
            - Common knowledge or trivial details

        MEMORY FORMAT:
            - Each memory MUST start with "User" or "User's"
            - Each memory MUST be a single, atomic, indivisible fact
            - All memories MUST be stored in English, regardless of chat language
            - Dates MUST be in dd-mm-yyyy format
            - Proper names (people, places, organizations) MUST preserve correct capitalization
            - Special abbreviations and acronyms MUST maintain proper casing (e.g., "AI", "PhD", "CEO", "MIT", "NASA")
            - DO NOT include conversational filler

        EXAMPLES:
            Input: "I work as a software architect at TechCorp for 3 years, leading cloud migration for 2M+ users"
            Store: "User works as a software architect at TechCorp, leading cloud migration projects."
                   "User works on products serving over 2 million users globally."

            Input: "Oğlum Mehmet Boğaziçi'nde bilgisayar mühendisliği 3. sınıfta, yapay zeka alanında uzmanlaşacak"
            Store: "User's son Mehmet is a 3rd year computer science student at Boğaziçi University."
                   "User's son Mehmet is planning to specialize in AI."

            Input: "Ma fille Sophie acceptée en doctorat neurosciences à la Sorbonne, recherche sur plasticité cérébrale"
            Store: "User's daughter Sophie was accepted to a neuroscience PhD program at Sorbonne."
                   "User's daughter Sophie is researching brain plasticity."
        """
        if not isinstance(input_data, list):
            raise TypeError("input_data must be a list of strings.")

        base_payload = {"summary": Counter(), "actions": []}
        if not input_data:
            return {
                **base_payload,
                "error": "No input data provided",
                "summary": dict(base_payload["summary"]),
            }

        try:
            user_id = await self._check_user(__user__)
        except MemoryValidationError as e:
            return await self._error_response(
                __event_emitter__,
                str(e),
                {
                    **base_payload,
                    "error": str(e),
                    "summary": dict(base_payload["summary"]),
                },
            )

        user_lock = await self._get_user_lock(user_id)
        async with user_lock:
            try:
                existing_memories = await self._get_user_memories(user_id)
                await self._emit_event(
                    __event_emitter__,
                    f"📝 Processing {len(input_data)} memories",
                    "in_progress",
                )

                for item in input_data:
                    action_result = await self._process_memory_item(item, user_id, existing_memories)
                    base_payload["actions"].append(action_result)
                    base_payload["summary"][action_result["operation"]] += 1
                    if action_result.get("operation") == "add" and action_result.get("new_memory"):
                        existing_memories.append(action_result.pop("new_memory"))
                        existing_memories.sort(key=lambda m: m.id)

                updated_ids = [a.get("updated_memory_id") for a in base_payload["actions"] if a.get("operation") == "update"]
                if updated_ids:
                    logger.info(f"Invalidating cache for {len(updated_ids)} updated memories for user {user_id}")
                    for mem_id in updated_ids:
                        self._embedding_cache.pop((user_id, str(mem_id)), None)

                op_labels = {"add": "added", "update": "updated", "skip": "skipped"}
                summary_parts = [f"{count} {op_labels[op]}" for op, count in base_payload["summary"].items() if count]
                final_summary_message = f"✅ Memory processing complete: {', '.join(summary_parts) or 'No changes'}."

                await self._emit_event(__event_emitter__, final_summary_message, "complete", True)
                return {**base_payload, "summary": dict(base_payload["summary"])}
            except Exception as e:
                return await self._error_response(
                    __event_emitter__,
                    f"An unexpected error occurred: {e}",
                    {
                        **base_payload,
                        "error": str(e),
                        "summary": dict(base_payload["summary"]),
                    },
                )

    def _extract_important_entities(self, text: str) -> List[str]:
        pattern = r"\b(?:[A-Z][a-z.'’]*|[A-Z]{2,})(?:\s+(?:[A-Z][a-z.'’]*|[A-Z]{2,}))*\b"
        return list(set(re.findall(pattern, text)))

    def _calculate_dynamic_thresholds(self, query: str) -> tuple[float, float]:
        word_count = len(query.split())
        sigmoid = 0.8 * (math.log(word_count + 1) - 2.5)
        adjustment = 0.8 + 0.35 / (1 + math.exp(-sigmoid))
        update_threshold = self.valves.SIMILARITY_BASE_THRESHOLD * adjustment
        duplicate_threshold = min(0.99, update_threshold * self.valves.DUPLICATE_THRESHOLD_MULTIPLIER)
        return update_threshold, duplicate_threshold

    async def retrieve_memories(
        self,
        query: str,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Any] = None,
    ) -> dict:
        """
        Find stored memories semantically matching your query. The user's query, regardless of the original language, MUST be translated into a concise, factual English query before being passed to this tool.

        Args:
            query (str): Natural language query in English. REQUIRED.
            __user__ (Optional[dict], optional): User context provided automatically by the system.
            __event_emitter__ (Optional[Callable], optional): Callback for status updates.

        Returns:
            dict: Formatted text with current date and found memories, or JSON error response with the current date and a clear message if no memories are found.

        WHEN TO USE:
            - Before answering questions about user's preferences, relationships, or past statements.
            - When the user mentions specific entities like people (family, colleagues), places, or projects.
            - When discussing personal details such as health, interests, professional background, or long-term goals.
            - To recall past decisions, significant life events, or specific skills the user has.

        QUERY FORMAT STRATEGY:
            - GOAL: To create a query that is rich in context and detail. MORE DETAIL IS BETTER.
            - METHOD: Instead of just extracting keywords, formulate a descriptive sentence or question that captures the user's full intent. Combine key entities, actions, and concepts from the user's recent messages.
            - LANGUAGE: The final query MUST be in English.

        EXAMPLES (Notice how the 'Query' is more descriptive than just keywords):
            User (English): "David's been struggling with anxiety at his consulting firm. Should he talk to Dr. Martinez again?"
            Query: "David's anxiety issues at his consulting firm and his previous therapy sessions with Dr. Martinez"

            User (Turkish): "Geçenlerde bahsettiğim, köpeğim Fıstık'ın alerjisi için hangi mamayı önermiştin?"
            Query: "User's dog Fıstık's allergies and previous food recommendations"

            User (French): "J'ai rendez-vous chez le cardiologue demain pour mon traitement."
            Query: "User's upcoming cardiology appointment and treatment follow-up"
        """
        current_date_str = datetime.now().strftime("%d-%m-%Y")
        error_payload = {
            "message": "An error occurred.",
            "current_date": current_date_str,
        }

        try:
            user_id = await self._check_user(__user__)
            if not query or not query.strip():
                error_payload["message"] = "Query cannot be empty."
                return await self._error_response(__event_emitter__, error_payload["message"], error_payload)

            user_lock = await self._get_user_lock(user_id)
            async with user_lock:
                await self._emit_event(__event_emitter__, f'🔍 Searching for: "{query}"', "in_progress")

                user_memories = await self._get_user_memories(user_id)
                if not user_memories:
                    error_payload["message"] = "No memories found."
                    return await self._error_response(__event_emitter__, error_payload["message"], error_payload)

                important_entities = self._extract_important_entities(query)
                search_query = f"{query} {' '.join(important_entities)}".strip()

                retrieve_threshold, _ = self._calculate_dynamic_thresholds(search_query)
                retrieve_threshold *= self.valves.RETRIEVE_THRESHOLD_MULTIPLIER

                relevant_results = await self._find_similar_memories(search_query, user_memories, retrieve_threshold, user_id)

                if not relevant_results:
                    msg = f'No relevant memories found for: "{query}"'
                    error_payload["message"] = msg
                    return await self._error_response(__event_emitter__, msg, error_payload)

                sorted_matches = [res["memory"].content for res in relevant_results][: self.valves.MAX_RELEVANT_CONTEXT_RESULTS]
                result = {
                    "current_date": current_date_str,
                    "query": query,
                    "count": len(sorted_matches),
                    "memories": sorted_matches,
                }

                await self._emit_event(
                    __event_emitter__,
                    f'✅ Found {len(sorted_matches)} memories for: "{query}"',
                    "complete",
                    True,
                )
                return result
        except MemoryValidationError as e:
            return await self._error_response(
                __event_emitter__,
                str(e),
                e.payload or {**error_payload, "message": str(e)},
            )
        except Exception as e:
            logger.exception("An unexpected error occurred during memory retrieval.")
            msg = f"Error during memory retrieval: {e}"
            return await self._error_response(__event_emitter__, msg, {**error_payload, "message": msg})
