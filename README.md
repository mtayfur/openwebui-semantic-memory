# OpenWebUI Semantic Memory

A semantic memory tool for OpenWebUI that enables AI assistants to store, retrieve, and manage user-specific memories using vector embeddings and natural language processing.

## Features

- Semantic storage and retrieval of user facts
- Multilingual support (stores in English)
- Automatic deduplication and updating
- LRU caching and async operations for speed
- User privacy: local, isolated processing

## How It Works

Uses a configurable sentence transformer (default: `Alibaba-NLP/gte-multilingual-base`) to create embeddings for memories and queries. The model can be changed via the `SENTENCE_TRANSFORMER_MODEL` valve parameter.

- Extracts important facts from conversations
- Converts text to vector embeddings
- Prevents duplicates and updates similar memories
- Retrieves relevant memories based on meaning

## Installation

1. Upload `memory.py` in OpenWebUI (**Workspace → Tools**)
2. Adjust valve parameters if needed (defaults work well)
3. Enable the tool

> No restart required – tools are loaded dynamically

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_MEMORY` | `true` | Enable/disable memory features |
| `SIMILARITY_BASE_THRESHOLD` | `0.80` | Base similarity score for memory matching |
| `RETRIEVE_THRESHOLD_MULTIPLIER` | `0.7` | Lower = more inclusive retrieval |
| `MAX_RELEVANT_CONTEXT_RESULTS` | `20` | Maximum memories returned per query |
| `SENTENCE_TRANSFORMER_MODEL` | `Alibaba-NLP/gte-multilingual-base` | HuggingFace model name for SentenceTransformer |

**Tip:** Lower `RETRIEVE_THRESHOLD_MULTIPLIER` for more inclusive retrieval. You can set `SENTENCE_TRANSFORMER_MODEL` to any compatible HuggingFace model name.

## Usage

- **Stores**: Important facts (profession, family, preferences)
- **Retrieves**: Relevant memories for user queries
- **Updates**: Existing memories with new info

**Stored:** Personal facts, relationships, profession, preferences, important dates  
**Not stored:** Temporary info, opinions, daily tasks, common knowledge

## Troubleshooting

- Check `USE_MEMORY` is enabled
- Lower `RETRIEVE_THRESHOLD_MULTIPLIER` for more results
- Use descriptive queries

## Links

- [OpenWebUI Documentation](https://docs.openwebui.com/)
- [Sentence Transformers](https://www.sbert.net/)

*Built for the OpenWebUI community* 🚀
