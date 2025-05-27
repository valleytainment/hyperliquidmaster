"""
Advanced Caching System for HyperLiquid Trading Bot

This module provides a high-performance caching system for market data
and API responses to reduce latency and API usage.
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, Callable, TypeVar, Generic, List, Tuple

T = TypeVar('T')

class CacheItem(Generic[T]):
    """Cache item with value, timestamp, and expiration."""
    
    def __init__(self, value: T, ttl: float = 60.0):
        """
        Initialize cache item.
        
        Args:
            value: Cached value
            ttl: Time to live in seconds
        """
        self.value = value
        self.timestamp = time.time()
        self.ttl = ttl
        self.access_count = 0
        self.last_access = self.timestamp
    
    def is_expired(self) -> bool:
        """
        Check if cache item is expired.
        
        Returns:
            True if expired, False otherwise
        """
        return time.time() > self.timestamp + self.ttl
    
    def access(self) -> T:
        """
        Access cache item and update access statistics.
        
        Returns:
            Cached value
        """
        self.access_count += 1
        self.last_access = time.time()
        return self.value
    
    def get_age(self) -> float:
        """
        Get age of cache item in seconds.
        
        Returns:
            Age in seconds
        """
        return time.time() - self.timestamp
    
    def get_time_to_expiry(self) -> float:
        """
        Get time to expiry in seconds.
        
        Returns:
            Time to expiry in seconds, negative if expired
        """
        return (self.timestamp + self.ttl) - time.time()
    
    def extend_ttl(self, additional_seconds: float = 60.0):
        """
        Extend TTL of cache item.
        
        Args:
            additional_seconds: Additional seconds to add to TTL
        """
        self.ttl += additional_seconds


class AdvancedCache:
    """
    Advanced caching system with TTL, memory management, and statistics.
    """
    
    def __init__(self, name: str = "default", logger=None):
        """
        Initialize cache.
        
        Args:
            name: Cache name for identification
            logger: Optional logger instance
        """
        # Setup logging
        self.logger = logger or logging.getLogger(f"Cache.{name}")
        
        # Cache name
        self.name = name
        
        # Cache storage
        self.cache: Dict[str, CacheItem] = {}
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Cache configuration
        self.default_ttl = 60.0  # seconds
        self.max_size = 1000  # items
        self.cleanup_interval = 300.0  # seconds
        
        # Cache lock for thread safety
        self.lock = threading.RLock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        self.logger.info(f"Cache '{name}' initialized")
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """
        Set cache item.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds, or None to use default
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            # Check if cache is full
            if len(self.cache) >= self.max_size and key not in self.cache:
                # Evict least recently used item
                self._evict_lru()
            
            # Set cache item
            self.cache[key] = CacheItem(value, ttl or self.default_ttl)
            
            return True
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get cache item.
        
        Args:
            key: Cache key
            default: Default value if key not found or expired
            
        Returns:
            Cached value or default
        """
        with self.lock:
            # Check if key exists
            if key in self.cache:
                cache_item = self.cache[key]
                
                # Check if expired
                if cache_item.is_expired():
                    # Remove expired item
                    del self.cache[key]
                    self.misses += 1
                    return default
                
                # Update statistics
                self.hits += 1
                
                # Return value
                return cache_item.access()
            
            # Key not found
            self.misses += 1
            return default
    
    def get_or_set(self, key: str, value_func: Callable[[], T], ttl: Optional[float] = None) -> T:
        """
        Get cache item or set if not exists.
        
        Args:
            key: Cache key
            value_func: Function to call to get value if not cached
            ttl: Time to live in seconds, or None to use default
            
        Returns:
            Cached value or new value
        """
        with self.lock:
            # Check if key exists and not expired
            if key in self.cache and not self.cache[key].is_expired():
                # Update statistics
                self.hits += 1
                
                # Return value
                return self.cache[key].access()
            
            # Key not found or expired, call value function
            value = value_func()
            
            # Set cache item
            self.set(key, value, ttl)
            
            # Update statistics
            self.misses += 1
            
            return value
    
    def delete(self, key: str) -> bool:
        """
        Delete cache item.
        
        Args:
            key: Cache key
            
        Returns:
            True if item was deleted, False if not found
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self):
        """Clear all cache items."""
        with self.lock:
            self.cache.clear()
            self.logger.info(f"Cache '{self.name}' cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                "name": self.name,
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": hit_rate,
                "default_ttl": self.default_ttl,
                "cleanup_interval": self.cleanup_interval
            }
    
    def get_keys(self) -> List[str]:
        """
        Get list of cache keys.
        
        Returns:
            List of cache keys
        """
        with self.lock:
            return list(self.cache.keys())
    
    def get_items(self) -> List[Tuple[str, Any, float]]:
        """
        Get list of cache items with expiry times.
        
        Returns:
            List of tuples (key, value, time_to_expiry)
        """
        with self.lock:
            return [(key, item.value, item.get_time_to_expiry()) for key, item in self.cache.items()]
    
    def _evict_lru(self):
        """Evict least recently used cache item."""
        if not self.cache:
            return
            
        # Find least recently used item
        lru_key = min(self.cache.items(), key=lambda x: x[1].last_access)[0]
        
        # Remove item
        del self.cache[lru_key]
        
        # Update statistics
        self.evictions += 1
        
        self.logger.debug(f"Evicted LRU cache item: {lru_key}")
    
    def _cleanup_expired(self):
        """Clean up expired cache items."""
        with self.lock:
            # Find expired items
            expired_keys = [key for key, item in self.cache.items() if item.is_expired()]
            
            # Remove expired items
            for key in expired_keys:
                del self.cache[key]
            
            if expired_keys:
                self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache items")
    
    def _cleanup_loop(self):
        """Cleanup loop for expired cache items."""
        while True:
            # Sleep for cleanup interval
            time.sleep(self.cleanup_interval)
            
            try:
                # Clean up expired items
                self._cleanup_expired()
                
                # Log cache statistics
                stats = self.get_stats()
                self.logger.debug(f"Cache '{self.name}' stats: size={stats['size']}, hit_rate={stats['hit_rate']:.2f}, evictions={stats['evictions']}")
                
            except Exception as e:
                self.logger.error(f"Error in cache cleanup: {e}")


class CacheManager:
    """
    Manager for multiple cache instances.
    """
    
    def __init__(self, logger=None):
        """
        Initialize cache manager.
        
        Args:
            logger: Optional logger instance
        """
        # Setup logging
        self.logger = logger or logging.getLogger("CacheManager")
        
        # Cache instances
        self.caches: Dict[str, AdvancedCache] = {}
        
        # Cache lock for thread safety
        self.lock = threading.RLock()
        
        self.logger.info("Cache manager initialized")
    
    def get_cache(self, name: str) -> AdvancedCache:
        """
        Get or create cache instance.
        
        Args:
            name: Cache name
            
        Returns:
            Cache instance
        """
        with self.lock:
            if name not in self.caches:
                self.caches[name] = AdvancedCache(name, self.logger)
                
            return self.caches[name]
    
    def clear_all(self):
        """Clear all cache instances."""
        with self.lock:
            for cache in self.caches.values():
                cache.clear()
                
            self.logger.info("All caches cleared")
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all cache instances.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            return {name: cache.get_stats() for name, cache in self.caches.items()}
