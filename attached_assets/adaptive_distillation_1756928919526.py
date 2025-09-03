"""
Adaptive Knowledge Distillation Engine
Core system for intelligent knowledge compression and refinement.
"""

import logging
import time
import hashlib
import json
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)

class KnowledgePattern:
    """Represents a knowledge pattern with usage and importance metrics."""
    
    def __init__(self, content: str, pattern_type: str, confidence: float = 1.0):
        self.content = content
        self.pattern_type = pattern_type
        self.confidence = confidence
        self.usage_count = 0
        self.last_accessed = datetime.now()
        self.creation_time = datetime.now()
        self.importance_score = 0.0
        self.related_patterns = set()
        self.hash_id = hashlib.md5(content.encode()).hexdigest()[:16]
    
    def update_usage(self):
        """Update usage statistics when pattern is accessed."""
        self.usage_count += 1
        self.last_accessed = datetime.now()
        
    def calculate_importance(self, current_time: datetime) -> float:
        """Calculate dynamic importance score based on multiple factors."""
        # Recency factor (decreases over time)
        days_since_access = (current_time - self.last_accessed).days
        recency_factor = math.exp(-days_since_access / 30.0)  # 30-day half-life
        
        # Usage frequency factor
        age_days = max(1, (current_time - self.creation_time).days)
        usage_frequency = self.usage_count / age_days
        
        # Network connectivity factor (more connected = more important)
        connectivity_factor = min(1.0, len(self.related_patterns) / 10.0)
        
        # Combined importance score
        self.importance_score = (
            0.4 * recency_factor +
            0.4 * min(1.0, usage_frequency * 10) +
            0.2 * connectivity_factor
        ) * self.confidence
        
        return self.importance_score

class AdaptiveDistillationEngine:
    """
    Advanced knowledge distillation system that continuously refines
    the knowledge base by identifying and removing redundancies while
    preserving essential information patterns.
    """
    
    def __init__(self, compression_threshold: float = 0.7, min_importance: float = 0.1):
        self.patterns = {}  # hash_id -> KnowledgePattern
        self.pattern_clusters = defaultdict(set)  # pattern_type -> set of hash_ids
        self.redundancy_graph = defaultdict(set)  # hash_id -> set of similar hash_ids
        self.compression_threshold = compression_threshold
        self.min_importance = min_importance
        self.distillation_stats = {
            'patterns_processed': 0,
            'patterns_compressed': 0,
            'redundancies_removed': 0,
            'knowledge_efficiency_gain': 0.0,
            'last_distillation': None
        }
        self.initialized = False
        self.knowledge_base_size = 0
        self.compressed_size = 0
        
        logger.info("Adaptive Knowledge Distillation Engine initialized")
    
    def initialize(self):
        """Initialize the distillation engine."""
        try:
            self.initialized = True
            logger.info("✅ Adaptive Knowledge Distillation Engine initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize distillation engine: {e}")
            return False
    
    def add_knowledge_pattern(self, content: str, pattern_type: str, 
                            confidence: float = 1.0) -> str:
        """Add a new knowledge pattern to the system."""
        pattern = KnowledgePattern(content, pattern_type, confidence)
        
        # Check for existing similar patterns
        similar_patterns = self._find_similar_patterns(pattern)
        
        if similar_patterns:
            # Update existing pattern instead of creating duplicate
            best_match = max(similar_patterns, key=lambda p: p.confidence)
            best_match.update_usage()
            best_match.confidence = max(best_match.confidence, confidence)
            return best_match.hash_id
        
        # Add new pattern
        self.patterns[pattern.hash_id] = pattern
        self.pattern_clusters[pattern_type].add(pattern.hash_id)
        self.knowledge_base_size += len(content)
        
        return pattern.hash_id
    
    def _find_similar_patterns(self, new_pattern: KnowledgePattern, 
                             similarity_threshold: float = 0.85) -> List[KnowledgePattern]:
        """Find existing patterns similar to the new pattern."""
        similar = []
        
        # Check patterns of the same type first
        for pattern_id in self.pattern_clusters[new_pattern.pattern_type]:
            existing_pattern = self.patterns[pattern_id]
            similarity = self._calculate_similarity(new_pattern.content, 
                                                  existing_pattern.content)
            
            if similarity >= similarity_threshold:
                similar.append(existing_pattern)
                # Build redundancy graph
                self.redundancy_graph[new_pattern.hash_id].add(pattern_id)
                self.redundancy_graph[pattern_id].add(new_pattern.hash_id)
        
        return similar
    
    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate semantic similarity between two content strings."""
        # Simple similarity calculation (can be enhanced with embeddings)
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard_similarity = len(intersection) / len(union)
        
        # Length similarity factor
        length_ratio = min(len(content1), len(content2)) / max(len(content1), len(content2))
        
        return (jaccard_similarity * 0.7) + (length_ratio * 0.3)
    
    def perform_distillation(self, force: bool = False) -> Dict[str, Any]:
        """
        Perform knowledge distillation to compress and optimize the knowledge base.
        """
        if not self.initialized:
            return {'success': False, 'message': 'Engine not initialized'}
        
        start_time = time.time()
        current_time = datetime.now()
        
        # Update importance scores for all patterns
        self._update_importance_scores(current_time)
        
        # Identify redundant patterns
        redundant_patterns = self._identify_redundant_patterns()
        
        # Compress redundant patterns
        compression_results = self._compress_patterns(redundant_patterns)
        
        # Remove low-importance patterns
        pruning_results = self._prune_low_importance_patterns()
        
        # Update statistics
        processing_time = time.time() - start_time
        self._update_distillation_stats(compression_results, pruning_results, processing_time)
        
        # Calculate knowledge efficiency gain
        efficiency_gain = self._calculate_efficiency_gain()
        
        return {
            'success': True,
            'processing_time': processing_time,
            'patterns_before': len(self.patterns),
            'patterns_compressed': compression_results['compressed_count'],
            'patterns_pruned': pruning_results['pruned_count'],
            'redundancies_removed': compression_results['redundancies_removed'],
            'efficiency_gain': efficiency_gain,
            'knowledge_base_size_reduction': self._calculate_size_reduction(),
            'timestamp': current_time.isoformat()
        }
    
    def _update_importance_scores(self, current_time: datetime):
        """Update importance scores for all patterns."""
        for pattern in self.patterns.values():
            pattern.calculate_importance(current_time)
    
    def _identify_redundant_patterns(self) -> List[Set[str]]:
        """Identify clusters of redundant patterns."""
        redundant_clusters = []
        processed = set()
        
        for pattern_id, similar_ids in self.redundancy_graph.items():
            if pattern_id in processed:
                continue
            
            if len(similar_ids) > 0:
                cluster = {pattern_id} | similar_ids
                redundant_clusters.append(cluster)
                processed.update(cluster)
        
        return redundant_clusters
    
    def _compress_patterns(self, redundant_clusters: List[Set[str]]) -> Dict[str, Any]:
        """Compress redundant patterns into optimized representations."""
        compressed_count = 0
        redundancies_removed = 0
        
        for cluster in redundant_clusters:
            if len(cluster) <= 1:
                continue
            
            # Find the best representative pattern
            cluster_patterns = [self.patterns[pid] for pid in cluster if pid in self.patterns]
            if not cluster_patterns:
                continue
            
            # Choose pattern with highest importance and confidence
            best_pattern = max(cluster_patterns, 
                             key=lambda p: p.importance_score * p.confidence)
            
            # Merge usage statistics from other patterns
            total_usage = sum(p.usage_count for p in cluster_patterns)
            best_pattern.usage_count = total_usage
            
            # Update related patterns
            all_related = set()
            for pattern in cluster_patterns:
                all_related.update(pattern.related_patterns)
            best_pattern.related_patterns = all_related
            
            # Remove redundant patterns
            for pattern in cluster_patterns:
                if pattern.hash_id != best_pattern.hash_id:
                    self._remove_pattern(pattern.hash_id)
                    redundancies_removed += 1
            
            compressed_count += 1
        
        return {
            'compressed_count': compressed_count,
            'redundancies_removed': redundancies_removed
        }
    
    def _prune_low_importance_patterns(self) -> Dict[str, Any]:
        """Remove patterns with very low importance scores."""
        current_time = datetime.now()
        patterns_to_remove = []
        
        for pattern_id, pattern in self.patterns.items():
            # Skip recently created patterns
            if (current_time - pattern.creation_time).days < 7:
                continue
            
            if pattern.importance_score < self.min_importance:
                patterns_to_remove.append(pattern_id)
        
        # Remove low-importance patterns
        for pattern_id in patterns_to_remove:
            self._remove_pattern(pattern_id)
        
        return {'pruned_count': len(patterns_to_remove)}
    
    def _remove_pattern(self, pattern_id: str):
        """Remove a pattern from the system."""
        if pattern_id not in self.patterns:
            return
        
        pattern = self.patterns[pattern_id]
        
        # Update compressed size tracking
        self.compressed_size += len(pattern.content)
        
        # Remove from pattern clusters
        self.pattern_clusters[pattern.pattern_type].discard(pattern_id)
        
        # Remove from redundancy graph
        if pattern_id in self.redundancy_graph:
            for related_id in self.redundancy_graph[pattern_id]:
                self.redundancy_graph[related_id].discard(pattern_id)
            del self.redundancy_graph[pattern_id]
        
        # Remove the pattern
        del self.patterns[pattern_id]
    
    def _calculate_efficiency_gain(self) -> float:
        """Calculate the knowledge efficiency gain from distillation."""
        if self.knowledge_base_size == 0:
            return 0.0
        
        current_size = sum(len(p.content) for p in self.patterns.values())
        original_size = self.knowledge_base_size
        
        if original_size == 0:
            return 0.0
        
        efficiency_gain = (original_size - current_size) / original_size
        return max(0.0, efficiency_gain)
    
    def _calculate_size_reduction(self) -> float:
        """Calculate the percentage reduction in knowledge base size."""
        if self.knowledge_base_size == 0:
            return 0.0
        
        reduction = self.compressed_size / self.knowledge_base_size
        return min(1.0, reduction)
    
    def _update_distillation_stats(self, compression_results: Dict, 
                                 pruning_results: Dict, processing_time: float):
        """Update distillation statistics."""
        self.distillation_stats.update({
            'patterns_processed': len(self.patterns),
            'patterns_compressed': compression_results['compressed_count'],
            'redundancies_removed': compression_results['redundancies_removed'],
            'patterns_pruned': pruning_results['pruned_count'],
            'knowledge_efficiency_gain': self._calculate_efficiency_gain(),
            'last_distillation': datetime.now(),
            'last_processing_time': processing_time
        })
    
    def get_distillation_insights(self) -> Dict[str, Any]:
        """Get insights about the distillation process and current state."""
        if not self.initialized:
            return {'error': 'Engine not initialized'}
        
        current_time = datetime.now()
        
        # Pattern type distribution
        type_distribution = {}
        for pattern_type, pattern_ids in self.pattern_clusters.items():
            type_distribution[pattern_type] = len(pattern_ids)
        
        # Usage statistics
        usage_stats = {
            'total_patterns': len(self.patterns),
            'high_usage_patterns': len([p for p in self.patterns.values() if p.usage_count > 10]),
            'recent_patterns': len([p for p in self.patterns.values() 
                                  if (current_time - p.creation_time).days <= 7]),
            'average_importance': sum(p.importance_score for p in self.patterns.values()) / max(1, len(self.patterns))
        }
        
        return {
            'distillation_stats': self.distillation_stats,
            'pattern_type_distribution': type_distribution,
            'usage_statistics': usage_stats,
            'redundancy_clusters': len([cluster for cluster in self.redundancy_graph.values() if len(cluster) > 0]),
            'knowledge_base_size': self.knowledge_base_size,
            'compressed_size': self.compressed_size,
            'compression_ratio': self._calculate_size_reduction(),
            'efficiency_score': self._calculate_efficiency_gain()
        }
    
    def suggest_distillation_schedule(self) -> Dict[str, Any]:
        """Suggest optimal timing for next distillation based on system state."""
        if not self.distillation_stats['last_distillation']:
            return {
                'recommendation': 'immediate',
                'reason': 'Initial distillation has not been performed'
            }
        
        last_distillation = self.distillation_stats['last_distillation']
        days_since_last = (datetime.now() - last_distillation).days
        
        # Calculate distillation urgency based on multiple factors
        redundancy_ratio = len([cluster for cluster in self.redundancy_graph.values() 
                               if len(cluster) > 0]) / max(1, len(self.patterns))
        
        low_importance_ratio = len([p for p in self.patterns.values() 
                                   if p.importance_score < self.min_importance]) / max(1, len(self.patterns))
        
        urgency_score = (redundancy_ratio * 0.5) + (low_importance_ratio * 0.3) + (days_since_last / 30.0 * 0.2)
        
        if urgency_score > 0.7:
            recommendation = 'immediate'
            reason = 'High redundancy and low-importance pattern accumulation detected'
        elif urgency_score > 0.4:
            recommendation = 'within_24_hours'
            reason = 'Moderate optimization opportunities available'
        elif days_since_last > 7:
            recommendation = 'within_week'
            reason = 'Regular maintenance distillation recommended'
        else:
            recommendation = 'not_needed'
            reason = 'Knowledge base is well-optimized'
        
        return {
            'recommendation': recommendation,
            'reason': reason,
            'urgency_score': urgency_score,
            'days_since_last_distillation': days_since_last,
            'estimated_efficiency_gain': urgency_score * 0.3
        }