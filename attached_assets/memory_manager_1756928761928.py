"""
Advanced Memory Management System for Sentinel AI
Handles pattern recognition, learning, and intelligent memory optimization.
"""

import logging
import pickle
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class MemoryBlock:
    """Represents a structured memory block."""
    id: str
    content: Any
    timestamp: datetime
    access_count: int = 0
    last_accessed: datetime = None
    importance_score: float = 0.0
    tags: List[str] = None
    relationships: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.relationships is None:
            self.relationships = []
        if self.last_accessed is None:
            self.last_accessed = self.timestamp

@dataclass
class LearningPattern:
    """Represents a learned pattern from interactions."""
    pattern_id: str
    pattern_type: str  # 'query', 'response', 'interaction', 'performance'
    pattern_data: Dict[str, Any]
    confidence: float
    occurrences: int
    first_seen: datetime
    last_seen: datetime
    effectiveness_score: float = 0.0

class AdvancedMemoryManager:
    """
    Advanced memory management system with pattern recognition,
    learning capabilities, and intelligent optimization.
    """
    
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.memory_blocks: Dict[str, MemoryBlock] = {}
        self.learned_patterns: Dict[str, LearningPattern] = {}
        self.access_history = deque(maxlen=10000)
        self.performance_history = deque(maxlen=1000)
        
        # Learning and optimization
        self.pattern_recognition_enabled = True
        self.auto_optimization_enabled = True
        self.learning_rate = 0.1
        
        # Threading for background operations
        self.optimization_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        self.initialized = False
        
    def initialize(self):
        """Initialize the memory management system."""
        if self.initialized:
            return
            
        logger.info("Initializing Advanced Memory Manager...")
        
        # Load existing memory blocks if available
        self._load_persistent_memory()
        
        # Start background optimization
        self._start_background_optimization()
        
        self.initialized = True
        logger.info(f"Memory Manager initialized with {len(self.memory_blocks)} blocks")
    
    def store_memory(self, memory_id: str, content: Any, tags: List[str] = None, 
                    importance: float = 0.5) -> bool:
        """Store a new memory block with intelligent organization."""
        try:
            with self.lock:
                # Create memory block
                memory_block = MemoryBlock(
                    id=memory_id,
                    content=content,
                    timestamp=datetime.now(),
                    importance_score=importance,
                    tags=tags or []
                )
                
                # Analyze content for automatic tagging
                auto_tags = self._analyze_content_for_tags(content)
                memory_block.tags.extend(auto_tags)
                
                # Find relationships with existing memories
                relationships = self._find_memory_relationships(memory_block)
                memory_block.relationships = relationships
                
                # Store the memory
                self.memory_blocks[memory_id] = memory_block
                
                # Record access
                self.access_history.append({
                    'type': 'store',
                    'memory_id': memory_id,
                    'timestamp': datetime.now(),
                    'importance': importance
                })
                
                # Check memory limits
                self._enforce_memory_limits()
                
                logger.debug(f"Stored memory block: {memory_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error storing memory {memory_id}: {e}")
            return False
    
    def retrieve_memory(self, memory_id: str = None, tags: List[str] = None, 
                       content_query: str = None, limit: int = 10) -> List[MemoryBlock]:
        """Retrieve memory blocks with intelligent search."""
        try:
            with self.lock:
                results = []
                
                if memory_id:
                    # Direct ID lookup
                    if memory_id in self.memory_blocks:
                        block = self.memory_blocks[memory_id]
                        self._update_access_stats(block)
                        results = [block]
                else:
                    # Search by tags or content
                    candidates = list(self.memory_blocks.values())
                    
                    if tags:
                        candidates = [
                            block for block in candidates
                            if any(tag in block.tags for tag in tags)
                        ]
                    
                    if content_query:
                        candidates = self._search_by_content(candidates, content_query)
                    
                    # Sort by relevance and importance
                    candidates.sort(
                        key=lambda x: (x.importance_score, x.access_count, -len(x.relationships)),
                        reverse=True
                    )
                    
                    results = candidates[:limit]
                    
                    # Update access stats
                    for block in results:
                        self._update_access_stats(block)
                
                # Record retrieval
                self.access_history.append({
                    'type': 'retrieve',
                    'query': content_query or str(tags),
                    'results_count': len(results),
                    'timestamp': datetime.now()
                })
                
                return results
                
        except Exception as e:
            logger.error(f"Error retrieving memory: {e}")
            return []
    
    def learn_from_interaction(self, interaction_data: Dict[str, Any]) -> bool:
        """Learn patterns from user interactions and system performance."""
        try:
            # Extract patterns from the interaction
            patterns = self._extract_interaction_patterns(interaction_data)
            
            for pattern in patterns:
                pattern_id = pattern['id']
                
                if pattern_id in self.learned_patterns:
                    # Update existing pattern
                    existing = self.learned_patterns[pattern_id]
                    existing.occurrences += 1
                    existing.last_seen = datetime.now()
                    existing.confidence = min(1.0, existing.confidence + self.learning_rate)
                else:
                    # Create new pattern
                    self.learned_patterns[pattern_id] = LearningPattern(
                        pattern_id=pattern_id,
                        pattern_type=pattern['type'],
                        pattern_data=pattern['data'],
                        confidence=0.3,  # Start with low confidence
                        occurrences=1,
                        first_seen=datetime.now(),
                        last_seen=datetime.now()
                    )
            
            # Update performance metrics
            self._update_performance_metrics(interaction_data)
            
            logger.debug(f"Learned {len(patterns)} patterns from interaction")
            return True
            
        except Exception as e:
            logger.error(f"Error learning from interaction: {e}")
            return False
    
    def get_memory_insights(self) -> Dict[str, Any]:
        """Get insights about memory usage and learned patterns."""
        try:
            with self.lock:
                total_blocks = len(self.memory_blocks)
                total_patterns = len(self.learned_patterns)
                
                # Calculate memory usage distribution
                tag_distribution = defaultdict(int)
                importance_distribution = {'high': 0, 'medium': 0, 'low': 0}
                
                for block in self.memory_blocks.values():
                    for tag in block.tags:
                        tag_distribution[tag] += 1
                    
                    if block.importance_score > 0.7:
                        importance_distribution['high'] += 1
                    elif block.importance_score > 0.4:
                        importance_distribution['medium'] += 1
                    else:
                        importance_distribution['low'] += 1
                
                # Top patterns by confidence
                top_patterns = sorted(
                    self.learned_patterns.values(),
                    key=lambda x: x.confidence * x.occurrences,
                    reverse=True
                )[:10]
                
                return {
                    'total_memory_blocks': total_blocks,
                    'total_patterns': total_patterns,
                    'tag_distribution': dict(tag_distribution),
                    'importance_distribution': importance_distribution,
                    'top_patterns': [
                        {
                            'id': p.pattern_id,
                            'type': p.pattern_type,
                            'confidence': p.confidence,
                            'occurrences': p.occurrences
                        } for p in top_patterns
                    ],
                    'memory_efficiency': self._calculate_memory_efficiency()
                }
                
        except Exception as e:
            logger.error(f"Error getting memory insights: {e}")
            return {}
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization and cleanup."""
        try:
            with self.lock:
                initial_count = len(self.memory_blocks)
                
                # Remove low-value memories
                removed_blocks = self._remove_low_value_memories()
                
                # Consolidate related memories
                consolidated = self._consolidate_related_memories()
                
                # Update importance scores
                updated_scores = self._update_importance_scores()
                
                optimization_result = {
                    'initial_blocks': initial_count,
                    'final_blocks': len(self.memory_blocks),
                    'removed_blocks': removed_blocks,
                    'consolidated_blocks': consolidated,
                    'updated_scores': updated_scores,
                    'efficiency_improvement': self._calculate_memory_efficiency()
                }
                
                logger.info(f"Memory optimization completed: {optimization_result}")
                return optimization_result
                
        except Exception as e:
            logger.error(f"Error during memory optimization: {e}")
            return {}
    
    def _analyze_content_for_tags(self, content: Any) -> List[str]:
        """Automatically generate tags for content."""
        tags = []
        
        if isinstance(content, str):
            content_lower = content.lower()
            
            # Domain-specific keywords
            if any(word in content_lower for word in ['math', 'equation', 'formula']):
                tags.append('mathematics')
            if any(word in content_lower for word in ['code', 'python', 'algorithm']):
                tags.append('programming')
            if any(word in content_lower for word in ['physics', 'quantum', 'energy']):
                tags.append('physics')
            if any(word in content_lower for word in ['ai', 'model', 'learning']):
                tags.append('artificial_intelligence')
        
        return tags
    
    def _find_memory_relationships(self, memory_block: MemoryBlock) -> List[str]:
        """Find relationships between memory blocks."""
        relationships = []
        
        for existing_id, existing_block in self.memory_blocks.items():
            # Check tag overlap
            tag_overlap = set(memory_block.tags) & set(existing_block.tags)
            if len(tag_overlap) >= 2:
                relationships.append(existing_id)
        
        return relationships[:5]  # Limit relationships
    
    def _search_by_content(self, candidates: List[MemoryBlock], query: str) -> List[MemoryBlock]:
        """Search memory blocks by content similarity."""
        query_lower = query.lower()
        scored_candidates = []
        
        for block in candidates:
            score = 0
            content_str = str(block.content).lower()
            
            # Simple keyword matching
            query_words = query_lower.split()
            for word in query_words:
                if word in content_str:
                    score += 1
                if word in block.tags:
                    score += 2
            
            if score > 0:
                scored_candidates.append((block, score))
        
        # Sort by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return [block for block, _ in scored_candidates]
    
    def _update_access_stats(self, memory_block: MemoryBlock):
        """Update access statistics for a memory block."""
        memory_block.access_count += 1
        memory_block.last_accessed = datetime.now()
        
        # Increase importance based on frequent access
        if memory_block.access_count > 10:
            memory_block.importance_score = min(1.0, memory_block.importance_score + 0.1)
    
    def _extract_interaction_patterns(self, interaction_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract learning patterns from interaction data."""
        patterns = []
        
        # Query patterns
        if 'query' in interaction_data:
            query = interaction_data['query']
            patterns.append({
                'id': f"query_pattern_{hash(query) % 10000}",
                'type': 'query',
                'data': {
                    'query_length': len(query),
                    'query_words': len(query.split()),
                    'query_complexity': self._assess_query_complexity(query)
                }
            })
        
        # Response patterns
        if 'response' in interaction_data:
            response = interaction_data['response']
            patterns.append({
                'id': f"response_pattern_{hash(response) % 10000}",
                'type': 'response',
                'data': {
                    'response_length': len(response),
                    'confidence': interaction_data.get('confidence', 0.5)
                }
            })
        
        return patterns
    
    def _assess_query_complexity(self, query: str) -> float:
        """Assess the complexity of a query."""
        words = query.split()
        complexity = 0.0
        
        # Length factor
        complexity += min(1.0, len(words) / 20)
        
        # Technical terms
        technical_terms = ['algorithm', 'quantum', 'machine learning', 'neural', 'optimization']
        for term in technical_terms:
            if term in query.lower():
                complexity += 0.2
        
        return min(1.0, complexity)
    
    def _update_performance_metrics(self, interaction_data: Dict[str, Any]):
        """Update system performance metrics."""
        metrics = {
            'timestamp': datetime.now(),
            'response_time': interaction_data.get('processing_time', 0),
            'confidence': interaction_data.get('confidence', 0),
            'user_satisfaction': interaction_data.get('user_satisfaction', 0.5)
        }
        
        self.performance_history.append(metrics)
    
    def _enforce_memory_limits(self):
        """Enforce memory usage limits."""
        # Simple implementation - remove oldest low-importance memories
        if len(self.memory_blocks) > 1000:  # Arbitrary limit
            candidates_for_removal = [
                (block_id, block) for block_id, block in self.memory_blocks.items()
                if block.importance_score < 0.3 and block.access_count < 3
            ]
            
            # Sort by age and remove oldest
            candidates_for_removal.sort(key=lambda x: x[1].timestamp)
            for block_id, _ in candidates_for_removal[:100]:
                del self.memory_blocks[block_id]
    
    def _remove_low_value_memories(self) -> int:
        """Remove memories with low value scores."""
        to_remove = []
        
        for block_id, block in self.memory_blocks.items():
            # Calculate value score
            age_days = (datetime.now() - block.timestamp).days
            value_score = (
                block.importance_score * 0.4 +
                min(1.0, block.access_count / 10) * 0.3 +
                max(0, 1 - age_days / 365) * 0.3
            )
            
            if value_score < 0.2 and age_days > 30:
                to_remove.append(block_id)
        
        for block_id in to_remove:
            del self.memory_blocks[block_id]
        
        return len(to_remove)
    
    def _consolidate_related_memories(self) -> int:
        """Consolidate related memory blocks."""
        # Simple implementation - group by tags
        consolidated = 0
        tag_groups = defaultdict(list)
        
        for block_id, block in self.memory_blocks.items():
            for tag in block.tags:
                tag_groups[tag].append((block_id, block))
        
        # Consolidate groups with many similar memories
        for tag, blocks in tag_groups.items():
            if len(blocks) > 20:  # Arbitrary threshold
                # Keep only the most important ones
                blocks.sort(key=lambda x: x[1].importance_score, reverse=True)
                to_keep = blocks[:10]
                to_remove = blocks[10:]
                
                for block_id, _ in to_remove:
                    if block_id in self.memory_blocks:
                        del self.memory_blocks[block_id]
                        consolidated += 1
        
        return consolidated
    
    def _update_importance_scores(self) -> int:
        """Update importance scores based on usage patterns."""
        updated = 0
        
        for block in self.memory_blocks.values():
            old_score = block.importance_score
            
            # Boost score for frequently accessed memories
            if block.access_count > 5:
                block.importance_score = min(1.0, block.importance_score + 0.1)
                updated += 1
            
            # Decay score for old, unused memories
            age_days = (datetime.now() - block.last_accessed).days
            if age_days > 7:
                block.importance_score = max(0.0, block.importance_score - 0.05)
                updated += 1
        
        return updated
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate overall memory efficiency score."""
        if not self.memory_blocks:
            return 1.0
        
        total_importance = sum(block.importance_score for block in self.memory_blocks.values())
        avg_importance = total_importance / len(self.memory_blocks)
        
        # Factor in access patterns
        total_access = sum(block.access_count for block in self.memory_blocks.values())
        avg_access = total_access / len(self.memory_blocks) if total_access > 0 else 0
        
        efficiency = (avg_importance * 0.6 + min(1.0, avg_access / 10) * 0.4)
        return round(efficiency, 3)
    
    def _start_background_optimization(self):
        """Start background optimization thread."""
        if self.auto_optimization_enabled and not self.optimization_thread:
            self.running = True
            self.optimization_thread = threading.Thread(target=self._background_optimization_loop)
            self.optimization_thread.daemon = True
            self.optimization_thread.start()
            logger.info("Background memory optimization started")
    
    def _background_optimization_loop(self):
        """Background optimization loop."""
        while self.running:
            try:
                time.sleep(300)  # Run every 5 minutes
                if len(self.memory_blocks) > 100:  # Only optimize if we have substantial memory
                    self.optimize_memory()
            except Exception as e:
                logger.error(f"Error in background optimization: {e}")
    
    def _load_persistent_memory(self):
        """Load persistent memory from storage."""
        try:
            # This would load from database or file system
            # For now, just initialize empty
            logger.info("Persistent memory loading not implemented yet")
        except Exception as e:
            logger.error(f"Error loading persistent memory: {e}")
    
    def shutdown(self):
        """Shutdown the memory manager."""
        self.running = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
        logger.info("Memory Manager shutdown completed")