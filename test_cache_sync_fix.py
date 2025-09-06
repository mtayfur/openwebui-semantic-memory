#!/usr/bin/env python3
"""
Test script to verify the cache synchronization fix for KeyError in _get_user_cache
"""
import asyncio
import pytest
from unittest.mock import patch
from neural_recall_v3 import Filter


@pytest.mark.asyncio
async def test_cache_sync_repair():
    """Test that the cache synchronization repair mechanism works correctly."""
    filter_instance = Filter()
    user_id = "test-user-id"
    
    # Initialize the caches
    await filter_instance._get_user_cache(user_id)
    
    # Verify both caches have the user_id
    assert user_id in Filter._embedding_cache
    assert user_id in Filter._cache_access_order
    
    # Simulate a desync condition - remove from access order but keep in embedding cache
    Filter._cache_access_order.pop(user_id, None)
    
    # Verify the desync
    assert user_id in Filter._embedding_cache
    assert user_id not in Filter._cache_access_order
    
    # Now call _get_user_cache again - it should repair the sync
    cache = await filter_instance._get_user_cache(user_id)
    
    # Verify the repair worked
    assert user_id in Filter._embedding_cache
    assert user_id in Filter._cache_access_order
    assert cache is not None
    
    # Clean up
    await Filter.force_cleanup()


@pytest.mark.asyncio
async def test_cache_no_keyerror_on_move_to_end():
    """Test that we don't get KeyError when calling move_to_end on a missing user_id."""
    filter_instance = Filter()
    user_id = "test-user-missing"
    
    # Clear the caches first
    await Filter.force_cleanup()
    
    # Manually add to embedding cache but not to access order (simulating desync)
    from modules.cache import LRUCache
    from modules.config import Config
    
    async with Filter._cache_lock:
        Filter._embedding_cache[user_id] = LRUCache(Config.CACHE_MAX_SIZE)
        # Intentionally don't add to _cache_access_order
    
    # This should not raise a KeyError
    try:
        cache = await filter_instance._get_user_cache(user_id)
        assert cache is not None
        assert user_id in Filter._cache_access_order  # Should be repaired
    except KeyError:
        pytest.fail("KeyError should not be raised when cache is out of sync")
    
    # Clean up
    await Filter.force_cleanup()


if __name__ == "__main__":
    asyncio.run(test_cache_sync_repair())
    asyncio.run(test_cache_no_keyerror_on_move_to_end())
    print("✅ All cache synchronization tests passed!")
