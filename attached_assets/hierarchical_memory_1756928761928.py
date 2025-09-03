"""
Hierarchical Memory Architecture
Multi-tier memory system with hot/warm/cold storage layers for optimal performance.
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta
import json
import pickle
import hashlib
import heapq
from enum import Enum
import os

logger = logging.getLogger(__name__)

class MemoryTier(Enum):
    """Memory tier classifications."""
    HOT = "hot"          # Frequently accessed, in-memory
    WARM = "warm"        # Moderately accessed, fast storage
    COLD = "cold"        # Rarely accessed, compressed storage

class MemoryBlock:
    """Represents a block of memory with access tracking and tier management."""
    
    def __init__(self, key: str, data: Any, tier: MemoryTier = MemoryTier.HOT):
        self.key = key
        self.data = data
        self.tier = tier
        self.access_count = 0
        self.last_accessed = datetime.now()
        self.created_at = datetime.now()
        self.size_bytes = self._calculate_size(data)
        self.access_pattern = []  # Track access times for pattern analysis
        self.importance_score = 1.0
        self.compression_ratio = 1.0
        self.serialized_data = None
        self.lock = threading.RLock()
    
    def _calculate_size(self, data: Any) -> int:
        """Calculate approximate size of data in bytes."""
        try:
            if isinstance(data, str):
                return len(data.encode('utf-8'))
            elif isinstance(data, (int, float)):
                return 8
            elif isinstance(data, (list, dict)):
                return len(pickle.dumps(data))
            else:
                return len(str(data).encode('utf-8'))
        except:
            return 1024  # Default size if calculation fails
    
    def access(self) -> Any:
        """Access the memory block and update tracking metrics."""
        with self.lock:
            self.access_count += 1
            current_time = datetime.now()
            self.last_accessed = current_time
            
            # Track access pattern (keep last 100 accesses)
            self.access_pattern.append(current_time.timestamp())
            if len(self.access_pattern) > 100:
                self.access_pattern.pop(0)
            
            # Update importance score based on access frequency
            self._update_importance_score()
            
            return self.data
    
    def _update_importance_score(self):
        """Update importance score based on access patterns."""
        current_time = datetime.now()
        age_days = (current_time - self.created_at).days + 1
        
        # Frequency factor
        frequency = self.access_count / age_days
        
        # Recency factor (exponential decay)
        hours_since_access = (current_time - self.last_accessed).total_seconds() / 3600
        recency = max(0.1, 1.0 / (1.0 + hours_since_access / 24))
        
        # Pattern regularity factor
        regularity = self._calculate_access_regularity()
        
        # Combined importance score
        self.importance_score = (
            0.4 * min(1.0, frequency * 10) +  # Frequency capped at 1.0
            0.4 * recency +                   # Recency factor
            0.2 * regularity                  # Access pattern regularity
        )
    
    def _calculate_access_regularity(self) -> float:
        """Calculate how regular the access pattern is."""
        if len(self.access_pattern) < 3:
            return 0.5  # Default for insufficient data
        
        # Calculate intervals between accesses
        intervals = []
        for i in range(1, len(self.access_pattern)):
            intervals.append(self.access_pattern[i] - self.access_pattern[i-1])
        
        if not intervals:
            return 0.5
        
        # Regularity based on standard deviation of intervals
        mean_interval = sum(intervals) / len(intervals)
        if mean_interval == 0:
            return 1.0
        
        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
        std_dev = variance ** 0.5
        coefficient_of_variation = std_dev / mean_interval if mean_interval > 0 else 1.0
        
        # Lower coefficient of variation = more regular = higher score
        return max(0.0, 1.0 - coefficient_of_variation)
    
    def serialize(self) -> bytes:
        """Serialize the data for cold storage."""
        if self.serialized_data is None:
            try:
                self.serialized_data = pickle.dumps(self.data)
                self.compression_ratio = len(self.serialized_data) / self.size_bytes
            except Exception as e:
                logger.warning(f"Failed to serialize data for key {self.key}: {e}")
                self.serialized_data = str(self.data).encode('utf-8')
        return self.serialized_data
    
    def deserialize(self) -> Any:
        """Deserialize data from cold storage."""
        if self.serialized_data is not None:
            try:
                return pickle.loads(self.serialized_data)
            except:
                return self.serialized_data.decode('utf-8')
        return self.data

class HierarchicalMemoryManager:
    """
    Advanced hierarchical memory management system with intelligent tier management.
    """
    
    def __init__(self, hot_limit_mb: int = 100, warm_limit_mb: int = 500, 
                 migration_threshold: float = 0.1):
        self.hot_memory = OrderedDict()      # LRU cache for hot tier
        self.warm_memory = {}                # Regular dict for warm tier
        self.cold_storage = {}               # File-based cold storage
        
        self.hot_limit_bytes = hot_limit_mb * 1024 * 1024
        self.warm_limit_bytes = warm_limit_mb * 1024 * 1024
        self.migration_threshold = migration_threshold
        
        self.tier_stats = {
            MemoryTier.HOT: {'size': 0, 'count': 0, 'hits': 0, 'misses': 0},
            MemoryTier.WARM: {'size': 0, 'count': 0, 'hits': 0, 'misses': 0},
            MemoryTier.COLD: {'size': 0, 'count': 0, 'hits': 0, 'misses': 0}
        }
        
        self.access_patterns = defaultdict(list)
        self.migration_queue = []
        self.optimization_thread = None
        self.optimization_running = False
        self.storage_path = "memory_storage"
        self.initialized = False
        
        # Performance metrics
        self.performance_metrics = {
            'cache_hit_rate': 0.0,
            'average_access_time': 0.0,
            'memory_efficiency': 0.0,
            'tier_distribution': {},
            'migration_frequency': 0
        }
        
        logger.info("Hierarchical Memory Manager initialized")
    
    def initialize(self):
        """Initialize the hierarchical memory system."""
        try:
            # Create storage directory for cold tier
            os.makedirs(self.storage_path, exist_ok=True)
            
            # Start optimization thread
            self._start_optimization_thread()
            
            self.initialized = True
            logger.info("✅ Hierarchical Memory Manager initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize hierarchical memory: {e}")
            return False
    
    def store(self, key: str, data: Any, importance: float = 1.0) -> bool:
        """Store data in the appropriate memory tier."""
        try:
            # Create memory block
            block = MemoryBlock(key, data)
            block.importance_score = importance
            
            # Determine initial tier based on importance and current usage
            if importance > 0.8 and self._get_tier_size(MemoryTier.HOT) < self.hot_limit_bytes:
                self._store_in_hot(key, block)
            elif importance > 0.5 and self._get_tier_size(MemoryTier.WARM) < self.warm_limit_bytes:
                self._store_in_warm(key, block)
            else:
                self._store_in_cold(key, block)
            
            # Trigger tier optimization if needed
            self._check_tier_limits()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store data for key {key}: {e}")
            return False
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data from any tier, promoting if accessed frequently."""
        start_time = time.time()
        
        try:
            # Check hot tier first
            if key in self.hot_memory:
                self.tier_stats[MemoryTier.HOT]['hits'] += 1
                block = self.hot_memory[key]
                data = block.access()
                # Move to end for LRU
                self.hot_memory.move_to_end(key)
                self._update_performance_metrics(time.time() - start_time, True)
                return data
            
            # Check warm tier
            if key in self.warm_memory:
                self.tier_stats[MemoryTier.WARM]['hits'] += 1
                block = self.warm_memory[key]
                data = block.access()
                
                # Consider promoting to hot tier
                if self._should_promote_to_hot(block):
                    self._promote_to_hot(key, block)
                
                self._update_performance_metrics(time.time() - start_time, True)
                return data
            
            # Check cold tier
            if key in self.cold_storage:
                self.tier_stats[MemoryTier.COLD]['hits'] += 1
                block = self.cold_storage[key]
                data = block.deserialize()
                block.data = data  # Cache deserialized data
                block.access()
                
                # Consider promoting to warm tier
                if self._should_promote_to_warm(block):
                    self._promote_to_warm(key, block)
                
                self._update_performance_metrics(time.time() - start_time, True)
                return data
            
            # Data not found
            self._update_performance_metrics(time.time() - start_time, False)
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve data for key {key}: {e}")
            self._update_performance_metrics(time.time() - start_time, False)
            return None
    
    def _store_in_hot(self, key: str, block: MemoryBlock):
        """Store block in hot tier."""
        block.tier = MemoryTier.HOT
        self.hot_memory[key] = block
        self.tier_stats[MemoryTier.HOT]['size'] += block.size_bytes
        self.tier_stats[MemoryTier.HOT]['count'] += 1
    
    def _store_in_warm(self, key: str, block: MemoryBlock):
        """Store block in warm tier."""
        block.tier = MemoryTier.WARM
        self.warm_memory[key] = block
        self.tier_stats[MemoryTier.WARM]['size'] += block.size_bytes
        self.tier_stats[MemoryTier.WARM]['count'] += 1
    
    def _store_in_cold(self, key: str, block: MemoryBlock):
        """Store block in cold tier."""
        block.tier = MemoryTier.COLD
        block.serialize()  # Serialize for storage
        self.cold_storage[key] = block
        self.tier_stats[MemoryTier.COLD]['size'] += len(block.serialized_data or b'')
        self.tier_stats[MemoryTier.COLD]['count'] += 1
        
        # Optionally write to disk for persistence
        self._persist_to_disk(key, block)
    
    def _persist_to_disk(self, key: str, block: MemoryBlock):
        """Persist cold storage block to disk."""
        try:
            file_path = os.path.join(self.storage_path, f"{hashlib.md5(key.encode()).hexdigest()}.dat")
            with open(file_path, 'wb') as f:
                f.write(block.serialize())
        except Exception as e:
            logger.warning(f"Failed to persist block {key} to disk: {e}")
    
    def _get_tier_size(self, tier: MemoryTier) -> int:
        """Get current size of a memory tier."""
        return self.tier_stats[tier]['size']
    
    def _should_promote_to_hot(self, block: MemoryBlock) -> bool:
        """Determine if a block should be promoted to hot tier."""
        return (block.importance_score > 0.7 and 
                block.access_count > 5 and
                self._get_tier_size(MemoryTier.HOT) < self.hot_limit_bytes * 0.8)
    
    def _should_promote_to_warm(self, block: MemoryBlock) -> bool:
        """Determine if a block should be promoted to warm tier."""
        return (block.importance_score > 0.4 and 
                block.access_count > 2 and
                self._get_tier_size(MemoryTier.WARM) < self.warm_limit_bytes * 0.8)
    
    def _promote_to_hot(self, key: str, block: MemoryBlock):
        """Promote a block from warm to hot tier."""
        if key in self.warm_memory:
            del self.warm_memory[key]
            self.tier_stats[MemoryTier.WARM]['size'] -= block.size_bytes
            self.tier_stats[MemoryTier.WARM]['count'] -= 1
        
        self._store_in_hot(key, block)
        logger.debug(f"Promoted {key} to hot tier")
    
    def _promote_to_warm(self, key: str, block: MemoryBlock):
        """Promote a block from cold to warm tier."""
        if key in self.cold_storage:
            del self.cold_storage[key]
            self.tier_stats[MemoryTier.COLD]['size'] -= len(block.serialized_data or b'')
            self.tier_stats[MemoryTier.COLD]['count'] -= 1
        
        self._store_in_warm(key, block)
        logger.debug(f"Promoted {key} to warm tier")
    
    def _check_tier_limits(self):
        """Check if tier limits are exceeded and trigger migrations."""
        # Check hot tier limit
        if self._get_tier_size(MemoryTier.HOT) > self.hot_limit_bytes:
            self._demote_from_hot()
        
        # Check warm tier limit
        if self._get_tier_size(MemoryTier.WARM) > self.warm_limit_bytes:
            self._demote_from_warm()
    
    def _demote_from_hot(self):
        """Demote least important blocks from hot to warm tier."""
        while (self._get_tier_size(MemoryTier.HOT) > self.hot_limit_bytes * 0.8 and 
               self.hot_memory):
            # Find least important block
            min_importance = float('inf')
            least_important_key = None
            
            for key, block in self.hot_memory.items():
                if block.importance_score < min_importance:
                    min_importance = block.importance_score
                    least_important_key = key
            
            if least_important_key:
                block = self.hot_memory[least_important_key]
                del self.hot_memory[least_important_key]
                self.tier_stats[MemoryTier.HOT]['size'] -= block.size_bytes
                self.tier_stats[MemoryTier.HOT]['count'] -= 1
                
                self._store_in_warm(least_important_key, block)
                logger.debug(f"Demoted {least_important_key} from hot to warm tier")
    
    def _demote_from_warm(self):
        """Demote least important blocks from warm to cold tier."""
        # Sort by importance score
        sorted_blocks = sorted(self.warm_memory.items(), 
                             key=lambda x: x[1].importance_score)
        
        demote_count = max(1, len(sorted_blocks) // 4)  # Demote 25% of blocks
        
        for i in range(min(demote_count, len(sorted_blocks))):
            key, block = sorted_blocks[i]
            del self.warm_memory[key]
            self.tier_stats[MemoryTier.WARM]['size'] -= block.size_bytes
            self.tier_stats[MemoryTier.WARM]['count'] -= 1
            
            self._store_in_cold(key, block)
            logger.debug(f"Demoted {key} from warm to cold tier")
    
    def _start_optimization_thread(self):
        """Start background thread for memory optimization."""
        if self.optimization_thread is None or not self.optimization_thread.is_alive():
            self.optimization_running = True
            self.optimization_thread = threading.Thread(target=self._optimization_worker)
            self.optimization_thread.daemon = True
            self.optimization_thread.start()
    
    def _optimization_worker(self):
        """Background worker for memory optimization."""
        while self.optimization_running:
            try:
                time.sleep(60)  # Run optimization every minute
                self._optimize_memory_tiers()
                self._update_tier_distribution()
            except Exception as e:
                logger.error(f"Error in memory optimization: {e}")
    
    def _optimize_memory_tiers(self):
        """Perform periodic memory tier optimization."""
        current_time = datetime.now()
        
        # Update importance scores for all blocks
        for memory_dict in [self.hot_memory, self.warm_memory, self.cold_storage]:
            for block in memory_dict.values():
                block._update_importance_score()
        
        # Check for promotion/demotion opportunities
        self._check_tier_limits()
        
        # Clean up old cold storage files
        self._cleanup_old_files()
    
    def _cleanup_old_files(self):
        """Clean up old cold storage files."""
        try:
            cutoff_time = datetime.now() - timedelta(days=30)
            for filename in os.listdir(self.storage_path):
                file_path = os.path.join(self.storage_path, filename)
                if os.path.isfile(file_path):
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_time < cutoff_time:
                        os.remove(file_path)
        except Exception as e:
            logger.warning(f"Error cleaning up old files: {e}")
    
    def _update_performance_metrics(self, access_time: float, hit: bool):
        """Update performance metrics."""
        # Update cache hit rate
        total_accesses = sum(sum(stats['hits'] + stats['misses'] for stats in tier.values()) 
                           for tier in [self.tier_stats])
        total_hits = sum(stats['hits'] for stats in self.tier_stats.values())
        
        if total_accesses > 0:
            self.performance_metrics['cache_hit_rate'] = total_hits / total_accesses
        
        # Update average access time (exponential moving average)
        alpha = 0.1
        current_avg = self.performance_metrics['average_access_time']
        self.performance_metrics['average_access_time'] = (
            alpha * access_time + (1 - alpha) * current_avg
        )
        
        # Update memory efficiency
        total_memory = sum(stats['size'] for stats in self.tier_stats.values())
        if total_memory > 0:
            hot_ratio = self.tier_stats[MemoryTier.HOT]['size'] / total_memory
            warm_ratio = self.tier_stats[MemoryTier.WARM]['size'] / total_memory
            cold_ratio = self.tier_stats[MemoryTier.COLD]['size'] / total_memory
            
            # Efficiency based on appropriate tier usage
            self.performance_metrics['memory_efficiency'] = (
                hot_ratio * 1.0 +   # Hot tier is most efficient
                warm_ratio * 0.7 +  # Warm tier is moderately efficient
                cold_ratio * 0.3    # Cold tier is least efficient but necessary
            )
    
    def _update_tier_distribution(self):
        """Update tier distribution metrics."""
        total_count = sum(stats['count'] for stats in self.tier_stats.values())
        if total_count > 0:
            self.performance_metrics['tier_distribution'] = {
                'hot_percentage': (self.tier_stats[MemoryTier.HOT]['count'] / total_count) * 100,
                'warm_percentage': (self.tier_stats[MemoryTier.WARM]['count'] / total_count) * 100,
                'cold_percentage': (self.tier_stats[MemoryTier.COLD]['count'] / total_count) * 100
            }
    
    def get_memory_insights(self) -> Dict[str, Any]:
        """Get comprehensive insights about memory system performance."""
        if not self.initialized:
            return {'error': 'Memory system not initialized'}
        
        total_memory = sum(stats['size'] for stats in self.tier_stats.values())
        total_blocks = sum(stats['count'] for stats in self.tier_stats.values())
        
        return {
            'tier_statistics': dict(self.tier_stats),
            'performance_metrics': self.performance_metrics,
            'memory_utilization': {
                'total_memory_bytes': total_memory,
                'total_blocks': total_blocks,
                'hot_utilization': self.tier_stats[MemoryTier.HOT]['size'] / self.hot_limit_bytes if self.hot_limit_bytes > 0 else 0,
                'warm_utilization': self.tier_stats[MemoryTier.WARM]['size'] / self.warm_limit_bytes if self.warm_limit_bytes > 0 else 0,
                'average_block_size': total_memory / max(1, total_blocks)
            },
            'optimization_status': {
                'optimization_thread_active': self.optimization_running,
                'last_optimization': datetime.now().isoformat(),
                'migrations_pending': len(self.migration_queue)
            }
        }
    
    def cleanup(self):
        """Clean up resources and stop optimization thread."""
        self.optimization_running = False
        if self.optimization_thread and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=5)
        
        logger.info("Hierarchical Memory Manager cleaned up")