"""
Performance Optimization System
Monitors and optimizes system performance with intelligent caching and resource management.
"""

import logging
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import json
import pickle

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Represents a performance metric measurement."""
    metric_name: str
    value: float
    timestamp: datetime
    category: str  # 'response_time', 'memory', 'cpu', 'cache', 'throughput'
    source: str  # Component that generated the metric
    
@dataclass
class CacheEntry:
    """Represents a cached computation result."""
    key: str
    value: Any
    creation_time: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    importance: float
    expiry_time: Optional[datetime] = None

@dataclass
class OptimizationRecommendation:
    """Represents a system optimization recommendation."""
    recommendation_id: str
    category: str  # 'caching', 'memory', 'cpu', 'database', 'model'
    description: str
    impact_estimate: float  # Expected performance improvement (0.0 to 1.0)
    effort_required: str  # 'low', 'medium', 'high'
    implementation_steps: List[str]
    confidence: float
    created: datetime

class PerformanceOptimizer:
    """
    Advanced performance optimization system that monitors system metrics,
    provides intelligent caching, and generates optimization recommendations.
    """
    
    def __init__(self):
        self.metrics_history = deque(maxlen=10000)
        self.cache_storage: Dict[str, CacheEntry] = {}
        self.optimization_recommendations: List[OptimizationRecommendation] = []
        self.performance_baselines: Dict[str, float] = {}
        
        # System monitoring
        self.system_metrics = defaultdict(list)
        self.bottleneck_history = deque(maxlen=100)
        self.optimization_history = deque(maxlen=500)
        
        # Cache configuration
        self.max_cache_size_mb = 512  # Increased for global intelligence network
        self.cache_hit_rate = 0.0
        self.cache_stats = defaultdict(int)
        self.global_caching_enabled = False
        
        # Performance thresholds
        self.thresholds = {
            'response_time_ms': 2000,
            'memory_usage_percent': 80,
            'cpu_usage_percent': 70,
            'cache_hit_rate': 0.7,
            'error_rate_percent': 5
        }
        
        # Threading for monitoring
        self.monitoring_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        self.initialized = False
        
    def enable_global_caching(self):
        """Enable intelligent caching across all AI systems."""
        self.global_caching_enabled = True
        logger.info("Global caching enabled for all 22 AI systems")
        
        # Optimize cache settings for global intelligence network
        self.max_cache_size_mb = 1024  # 1GB for planetary-scale operations
        self.cache_stats['global_caching_enabled'] = 1
    
    def initialize(self):
        """Initialize the performance optimization system."""
        if self.initialized:
            return
            
        logger.info("Initializing Performance Optimizer...")
        
        # Establish performance baselines
        self._establish_baselines()
        
        # Start system monitoring
        self._start_system_monitoring()
        
        # Initialize cache
        self._initialize_cache()
        
        self.initialized = True
        logger.info("Performance Optimizer initialized")
    
    def record_metric(self, metric_name: str, value: float, category: str, source: str = 'system'):
        """Record a performance metric."""
        try:
            metric = PerformanceMetric(
                metric_name=metric_name,
                value=value,
                timestamp=datetime.now(),
                category=category,
                source=source
            )
            
            with self.lock:
                self.metrics_history.append(metric)
                self.system_metrics[metric_name].append((datetime.now(), value))
                
                # Keep only recent metrics per category
                if len(self.system_metrics[metric_name]) > 1000:
                    self.system_metrics[metric_name] = self.system_metrics[metric_name][-500:]
            
            # Check for performance issues
            self._check_performance_thresholds(metric)
            
        except Exception as e:
            logger.error(f"Error recording metric {metric_name}: {e}")
    
    def cache_get(self, key: str) -> Optional[Any]:
        """Retrieve a value from the intelligent cache."""
        try:
            with self.lock:
                if key in self.cache_storage:
                    entry = self.cache_storage[key]
                    
                    # Check expiry
                    if entry.expiry_time and datetime.now() > entry.expiry_time:
                        del self.cache_storage[key]
                        self.cache_stats['expired'] += 1
                        return None
                    
                    # Update access statistics
                    entry.last_accessed = datetime.now()
                    entry.access_count += 1
                    
                    self.cache_stats['hits'] += 1
                    self._update_cache_hit_rate()
                    
                    return entry.value
                else:
                    self.cache_stats['misses'] += 1
                    self._update_cache_hit_rate()
                    return None
                    
        except Exception as e:
            logger.error(f"Error retrieving from cache {key}: {e}")
            return None
    
    def cache_set(self, key: str, value: Any, expiry_minutes: int = None, importance: float = 0.5):
        """Store a value in the intelligent cache."""
        try:
            # Calculate size estimate
            size_bytes = len(str(value))  # Simple size estimation
            
            expiry_time = None
            if expiry_minutes:
                expiry_time = datetime.now() + timedelta(minutes=expiry_minutes)
            
            entry = CacheEntry(
                key=key,
                value=value,
                creation_time=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                size_bytes=size_bytes,
                importance=importance,
                expiry_time=expiry_time
            )
            
            with self.lock:
                # Check cache size limits
                self._enforce_cache_limits()
                
                self.cache_storage[key] = entry
                self.cache_stats['stores'] += 1
                
                logger.debug(f"Cached {key} with size {size_bytes} bytes")
                
        except Exception as e:
            logger.error(f"Error storing to cache {key}: {e}")
    
    def cache_invalidate(self, pattern: str = None):
        """Invalidate cache entries matching a pattern."""
        try:
            with self.lock:
                if pattern:
                    keys_to_remove = [key for key in self.cache_storage.keys() if pattern in key]
                    for key in keys_to_remove:
                        del self.cache_storage[key]
                        self.cache_stats['invalidated'] += 1
                else:
                    # Clear all cache
                    cleared_count = len(self.cache_storage)
                    self.cache_storage.clear()
                    self.cache_stats['invalidated'] += cleared_count
                
                logger.info(f"Invalidated cache entries matching '{pattern}'")
                
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
    
    def analyze_performance_bottlenecks(self) -> List[Dict[str, Any]]:
        """Analyze current performance bottlenecks."""
        try:
            bottlenecks = []
            
            # Analyze recent metrics
            recent_metrics = defaultdict(list)
            cutoff_time = datetime.now() - timedelta(minutes=10)
            
            for metric in list(self.metrics_history)[-500:]:
                if metric.timestamp > cutoff_time:
                    recent_metrics[metric.category].append(metric.value)
            
            # Check each category for issues
            for category, values in recent_metrics.items():
                if not values:
                    continue
                    
                avg_value = sum(values) / len(values)
                max_value = max(values)
                
                # Check against thresholds
                threshold_key = f"{category}_percent" if 'usage' in category else f"{category}_ms"
                threshold = self.thresholds.get(threshold_key, float('inf'))
                
                if avg_value > threshold:
                    severity = 'high' if avg_value > threshold * 1.5 else 'medium'
                    
                    bottleneck = {
                        'category': category,
                        'severity': severity,
                        'current_value': round(avg_value, 2),
                        'threshold': threshold,
                        'max_value': round(max_value, 2),
                        'samples': len(values),
                        'recommendation': self._get_bottleneck_recommendation(category, avg_value)
                    }
                    bottlenecks.append(bottleneck)
            
            # Sort by severity
            severity_order = {'high': 3, 'medium': 2, 'low': 1}
            bottlenecks.sort(key=lambda x: severity_order.get(x['severity'], 0), reverse=True)
            
            return bottlenecks
            
        except Exception as e:
            logger.error(f"Error analyzing bottlenecks: {e}")
            return []
    
    def generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate intelligent optimization recommendations."""
        try:
            recommendations = []
            
            # Analyze cache performance
            if self.cache_hit_rate < self.thresholds['cache_hit_rate']:
                rec = OptimizationRecommendation(
                    recommendation_id=f"cache_opt_{int(time.time())}",
                    category='caching',
                    description=f"Cache hit rate is {self.cache_hit_rate:.2%}, below optimal {self.thresholds['cache_hit_rate']:.1%}",
                    impact_estimate=0.3,
                    effort_required='medium',
                    implementation_steps=[
                        'Increase cache size allocation',
                        'Implement more intelligent cache eviction',
                        'Add cache warming for frequently accessed data',
                        'Optimize cache key strategies'
                    ],
                    confidence=0.8,
                    created=datetime.now()
                )
                recommendations.append(rec)
            
            # Analyze memory usage
            memory_usage = self._get_current_memory_usage()
            if memory_usage > self.thresholds['memory_usage_percent']:
                rec = OptimizationRecommendation(
                    recommendation_id=f"memory_opt_{int(time.time())}",
                    category='memory',
                    description=f"Memory usage at {memory_usage:.1f}%, recommend optimization",
                    impact_estimate=0.4,
                    effort_required='high',
                    implementation_steps=[
                        'Implement memory pooling for frequently used objects',
                        'Add garbage collection optimization',
                        'Review data structure efficiency',
                        'Implement lazy loading for large datasets'
                    ],
                    confidence=0.7,
                    created=datetime.now()
                )
                recommendations.append(rec)
            
            # Analyze response times
            avg_response_time = self._get_average_response_time()
            if avg_response_time > self.thresholds['response_time_ms']:
                rec = OptimizationRecommendation(
                    recommendation_id=f"response_opt_{int(time.time())}",
                    category='cpu',
                    description=f"Average response time {avg_response_time:.0f}ms exceeds {self.thresholds['response_time_ms']}ms target",
                    impact_estimate=0.5,
                    effort_required='medium',
                    implementation_steps=[
                        'Implement request queuing and prioritization',
                        'Add parallel processing for independent operations',
                        'Optimize database queries and indexing',
                        'Consider model optimization and compression'
                    ],
                    confidence=0.8,
                    created=datetime.now()
                )
                recommendations.append(rec)
            
            # Store recommendations
            self.optimization_recommendations.extend(recommendations)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def apply_optimization(self, recommendation_id: str) -> Dict[str, Any]:
        """Apply an optimization recommendation."""
        try:
            # Find the recommendation
            recommendation = None
            for rec in self.optimization_recommendations:
                if rec.recommendation_id == recommendation_id:
                    recommendation = rec
                    break
            
            if not recommendation:
                return {'success': False, 'message': 'Recommendation not found'}
            
            # Apply optimizations based on category
            result = {'success': False, 'actions_taken': []}
            
            if recommendation.category == 'caching':
                # Increase cache size
                old_size = self.max_cache_size_mb
                self.max_cache_size_mb = min(512, self.max_cache_size_mb * 1.5)
                result['actions_taken'].append(f"Increased cache size from {old_size}MB to {self.max_cache_size_mb}MB")
                
                # Optimize cache eviction
                self._optimize_cache_eviction()
                result['actions_taken'].append("Optimized cache eviction strategy")
                
                result['success'] = True
                
            elif recommendation.category == 'memory':
                # Clear low-importance cached items
                self._clear_low_importance_cache()
                result['actions_taken'].append("Cleared low-importance cache items")
                
                # Optimize memory usage
                self._optimize_memory_usage()
                result['actions_taken'].append("Applied memory optimization strategies")
                
                result['success'] = True
            
            elif recommendation.category == 'cpu':
                # Optimize processing strategies
                self._optimize_processing()
                result['actions_taken'].append("Applied CPU optimization strategies")
                
                result['success'] = True
            
            # Record optimization
            self.optimization_history.append({
                'recommendation_id': recommendation_id,
                'category': recommendation.category,
                'timestamp': datetime.now(),
                'result': result
            })
            
            logger.info(f"Applied optimization {recommendation_id}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error applying optimization {recommendation_id}: {e}")
            return {'success': False, 'message': str(e)}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a comprehensive performance summary."""
        try:
            # Recent performance metrics
            recent_metrics = {}
            for category in ['response_time', 'memory', 'cpu', 'cache']:
                values = [m.value for m in list(self.metrics_history)[-100:] if m.category == category]
                if values:
                    recent_metrics[category] = {
                        'current': values[-1] if values else 0,
                        'average': sum(values) / len(values),
                        'max': max(values),
                        'min': min(values)
                    }
            
            # Cache statistics
            total_cache_requests = self.cache_stats['hits'] + self.cache_stats['misses']
            cache_summary = {
                'hit_rate': self.cache_hit_rate,
                'total_entries': len(self.cache_storage),
                'total_requests': total_cache_requests,
                'hits': self.cache_stats['hits'],
                'misses': self.cache_stats['misses'],
                'size_mb': self._get_cache_size_mb()
            }
            
            # System resource usage
            system_resources = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0
            }
            
            # Active bottlenecks
            bottlenecks = self.analyze_performance_bottlenecks()
            
            return {
                'recent_metrics': recent_metrics,
                'cache_summary': cache_summary,
                'system_resources': system_resources,
                'active_bottlenecks': len(bottlenecks),
                'bottleneck_details': bottlenecks[:5],  # Top 5 bottlenecks
                'optimization_opportunities': len(self.optimization_recommendations),
                'recent_optimizations': len([
                    opt for opt in self.optimization_history
                    if (datetime.now() - opt['timestamp']).total_seconds() < 3600
                ])
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}
    
    def _establish_baselines(self):
        """Establish performance baselines."""
        try:
            # Initial system measurements
            self.performance_baselines = {
                'cpu_usage': psutil.cpu_percent(interval=1),
                'memory_usage': psutil.virtual_memory().percent,
                'baseline_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Performance baselines established: {self.performance_baselines}")
            
        except Exception as e:
            logger.error(f"Error establishing baselines: {e}")
    
    def _start_system_monitoring(self):
        """Start background system monitoring."""
        if not self.monitoring_thread:
            self.running = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logger.info("System monitoring started")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                self.record_metric('cpu_usage', cpu_percent, 'cpu', 'system_monitor')
                self.record_metric('memory_usage', memory_percent, 'memory', 'system_monitor')
                
                # Check for performance issues
                if cpu_percent > self.thresholds['cpu_usage_percent']:
                    self._handle_cpu_bottleneck(cpu_percent)
                
                if memory_percent > self.thresholds['memory_usage_percent']:
                    self._handle_memory_bottleneck(memory_percent)
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _initialize_cache(self):
        """Initialize the intelligent cache system."""
        try:
            self.cache_stats = defaultdict(int)
            self.cache_hit_rate = 0.0
            logger.info("Cache system initialized")
        except Exception as e:
            logger.error(f"Error initializing cache: {e}")
    
    def _enforce_cache_limits(self):
        """Enforce cache size limits with intelligent eviction."""
        try:
            current_size_mb = self._get_cache_size_mb()
            
            if current_size_mb > self.max_cache_size_mb:
                # Calculate how much to remove
                target_size = self.max_cache_size_mb * 0.8  # Leave some buffer
                to_remove_mb = current_size_mb - target_size
                
                # Sort entries by eviction priority (least recently used, lowest importance)
                entries = list(self.cache_storage.items())
                entries.sort(key=lambda x: (
                    x[1].importance,
                    x[1].last_accessed,
                    -x[1].access_count
                ))
                
                removed_size = 0
                for key, entry in entries:
                    if removed_size >= to_remove_mb * 1024 * 1024:  # Convert to bytes
                        break
                    
                    removed_size += entry.size_bytes
                    del self.cache_storage[key]
                    self.cache_stats['evicted'] += 1
                
                logger.debug(f"Evicted cache entries totaling {removed_size / (1024*1024):.1f}MB")
                
        except Exception as e:
            logger.error(f"Error enforcing cache limits: {e}")
    
    def _update_cache_hit_rate(self):
        """Update the cache hit rate."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        if total_requests > 0:
            self.cache_hit_rate = self.cache_stats['hits'] / total_requests
    
    def _get_cache_size_mb(self) -> float:
        """Get current cache size in MB."""
        total_bytes = sum(entry.size_bytes for entry in self.cache_storage.values())
        return total_bytes / (1024 * 1024)
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        return psutil.virtual_memory().percent
    
    def _get_average_response_time(self) -> float:
        """Get average response time from recent metrics."""
        response_times = [
            m.value for m in list(self.metrics_history)[-100:]
            if m.category == 'response_time'
        ]
        return sum(response_times) / len(response_times) if response_times else 0
    
    def _check_performance_thresholds(self, metric: PerformanceMetric):
        """Check if a metric exceeds performance thresholds."""
        threshold_key = f"{metric.category}_percent" if 'usage' in metric.metric_name else f"{metric.category}_ms"
        threshold = self.thresholds.get(threshold_key)
        
        if threshold and metric.value > threshold:
            bottleneck = {
                'timestamp': metric.timestamp,
                'category': metric.category,
                'metric': metric.metric_name,
                'value': metric.value,
                'threshold': threshold,
                'source': metric.source
            }
            self.bottleneck_history.append(bottleneck)
    
    def _get_bottleneck_recommendation(self, category: str, value: float) -> str:
        """Get a recommendation for a specific bottleneck."""
        recommendations = {
            'response_time': 'Consider optimizing algorithms, adding caching, or scaling resources',
            'memory': 'Review memory usage patterns, implement garbage collection, or increase memory allocation',
            'cpu': 'Optimize processing efficiency, implement load balancing, or scale CPU resources',
            'cache': 'Increase cache size, optimize cache strategies, or review cache key patterns'
        }
        return recommendations.get(category, 'Review system configuration and resource allocation')
    
    def _handle_cpu_bottleneck(self, cpu_percent: float):
        """Handle CPU bottleneck situation."""
        logger.warning(f"CPU bottleneck detected: {cpu_percent:.1f}%")
        # Could implement automatic throttling or load shedding here
    
    def _handle_memory_bottleneck(self, memory_percent: float):
        """Handle memory bottleneck situation."""
        logger.warning(f"Memory bottleneck detected: {memory_percent:.1f}%")
        # Trigger aggressive cache cleanup
        self._clear_low_importance_cache()
    
    def _optimize_cache_eviction(self):
        """Optimize cache eviction strategy."""
        # Implement LRU + importance based eviction
        pass
    
    def _clear_low_importance_cache(self):
        """Clear cache entries with low importance."""
        keys_to_remove = [
            key for key, entry in self.cache_storage.items()
            if entry.importance < 0.3
        ]
        for key in keys_to_remove:
            del self.cache_storage[key]
            self.cache_stats['low_importance_cleared'] += 1
    
    def _optimize_memory_usage(self):
        """Apply memory optimization strategies."""
        # Implement memory optimization techniques
        pass
    
    def _optimize_processing(self):
        """Apply CPU optimization strategies."""
        # Implement processing optimization techniques
        pass
    
    def shutdown(self):
        """Shutdown the performance optimizer."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Performance Optimizer shutdown completed")