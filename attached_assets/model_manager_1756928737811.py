"""
Dynamic model management system for loading, updating, and optimizing models.
"""

import logging
import time
import json
import statistics
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

from .ensemble_core import ModelInterface, ModelType, ModelResult

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    LOADING = "loading"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    UPDATING = "updating"

@dataclass
class ModelConfig:
    """Configuration for a model."""
    model_id: str
    model_type: ModelType
    model_class: str
    weight: float
    config: Dict[str, Any]
    requirements: List[str]
    version: str
    last_updated: float

class ModelManager:
    """
    Manages dynamic loading, updating, and optimization of models in the ensemble.
    """
    
    def __init__(self, ensemble_core=None):
        self.ensemble_core = ensemble_core
        self.model_configs: Dict[str, ModelConfig] = {}
        self.model_instances: Dict[str, ModelInterface] = {}
        self.model_status: Dict[str, ModelStatus] = {}
        self.model_performance: Dict[str, Dict[str, float]] = {}
        
        # Dynamic loading
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.auto_update_enabled = True
        self.performance_threshold = 0.6
        
        # Model registry
        self.available_models = {
            'echo': 'ensemble.ensemble_core.EchoModel',
            'analytics': 'ensemble.ensemble_core.AnalyticsModel',
            'reasoning': 'ensemble.ensemble_core.ReasoningModel',
            'pattern_matcher': 'ensemble.advanced_models.PatternMatcherModel',
            'knowledge_retriever': 'ensemble.advanced_models.KnowledgeRetrieverModel'
        }
        
        self.initialized = False
    
    def initialize(self):
        """Initialize the model manager."""
        if self.initialized:
            return
            
        logger.info("Initializing Model Manager...")
        
        # Register advanced models
        self._register_advanced_models()
        
        # Load default model configurations
        self._load_default_configs()
        
        # Start performance monitoring
        self._start_performance_monitoring()
        
        self.initialized = True
        logger.info("Model Manager initialized")
    
    def add_model_config(self, model_id: str, model_type: ModelType, 
                        model_class: str, weight: float = 1.0,
                        config: Dict[str, Any] = None) -> bool:
        """Add a new model configuration."""
        try:
            model_config = ModelConfig(
                model_id=model_id,
                model_type=model_type,
                model_class=model_class,
                weight=weight,
                config=config or {},
                requirements=[],
                version="1.0.0",
                last_updated=time.time()
            )
            
            self.model_configs[model_id] = model_config
            self.model_status[model_id] = ModelStatus.INACTIVE
            
            logger.info(f"Added model configuration: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding model config {model_id}: {e}")
            return False
    
    def load_model(self, model_id: str) -> bool:
        """Dynamically load a model."""
        if model_id not in self.model_configs:
            logger.error(f"Model config not found: {model_id}")
            return False
        
        if model_id in self.model_instances:
            logger.info(f"Model {model_id} already loaded")
            return True
        
        config = self.model_configs[model_id]
        self.model_status[model_id] = ModelStatus.LOADING
        
        try:
            # Dynamically import and instantiate model
            model_instance = self._instantiate_model(config)
            
            if model_instance:
                self.model_instances[model_id] = model_instance
                self.model_status[model_id] = ModelStatus.ACTIVE
                
                # Add to ensemble if available
                if self.ensemble_core:
                    self.ensemble_core.add_model(model_instance)
                
                logger.info(f"Successfully loaded model: {model_id}")
                return True
            else:
                self.model_status[model_id] = ModelStatus.ERROR
                return False
                
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            self.model_status[model_id] = ModelStatus.ERROR
            return False
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory."""
        if model_id not in self.model_instances:
            return True
        
        try:
            # Remove from ensemble
            if self.ensemble_core:
                self.ensemble_core.remove_model(model_id)
            
            # Remove from manager
            del self.model_instances[model_id]
            self.model_status[model_id] = ModelStatus.INACTIVE
            
            logger.info(f"Unloaded model: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading model {model_id}: {e}")
            return False
    
    def update_model_weight(self, model_id: str, new_weight: float) -> bool:
        """Update a model's weight dynamically."""
        if model_id in self.model_instances:
            self.model_instances[model_id].weight = new_weight
            
        if model_id in self.model_configs:
            self.model_configs[model_id].weight = new_weight
            self.model_configs[model_id].last_updated = time.time()
        
        logger.info(f"Updated weight for model {model_id}: {new_weight}")
        return True
    
    def auto_optimize_models(self) -> Dict[str, Any]:
        """Automatically optimize model weights and selection based on performance."""
        optimization_results = {
            'models_optimized': 0,
            'models_disabled': 0,
            'weight_adjustments': {},
            'recommendations': []
        }
        
        for model_id, performance in self.model_performance.items():
            if model_id not in self.model_instances:
                continue
            
            accuracy = performance.get('accuracy', 0.8)
            latency = performance.get('latency', 1.0)
            success_rate = performance.get('success_rate', 1.0)
            
            # Calculate performance score
            performance_score = (accuracy * 0.5 + success_rate * 0.3 + 
                               (1.0 - min(latency, 2.0) / 2.0) * 0.2)
            
            current_weight = self.model_instances[model_id].weight
            
            if performance_score < self.performance_threshold:
                # Reduce weight or disable poor performing models
                if current_weight > 0.1:
                    new_weight = max(0.1, current_weight * 0.8)
                    self.update_model_weight(model_id, new_weight)
                    optimization_results['weight_adjustments'][model_id] = {
                        'old': current_weight,
                        'new': new_weight,
                        'reason': 'poor_performance'
                    }
                else:
                    # Disable model
                    self.model_instances[model_id].enabled = False
                    optimization_results['models_disabled'] += 1
                    
            elif performance_score > 0.9:
                # Increase weight for high-performing models
                new_weight = min(2.0, current_weight * 1.1)
                if new_weight != current_weight:
                    self.update_model_weight(model_id, new_weight)
                    optimization_results['weight_adjustments'][model_id] = {
                        'old': current_weight,
                        'new': new_weight,
                        'reason': 'high_performance'
                    }
            
            optimization_results['models_optimized'] += 1
        
        # Generate recommendations for new models
        if len(self.model_instances) < 5:
            optimization_results['recommendations'].append({
                'type': 'add_model',
                'suggestion': 'Consider adding specialized models for better coverage'
            })
        
        return optimization_results
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get comprehensive model statistics."""
        stats = {
            'total_configs': len(self.model_configs),
            'loaded_models': len(self.model_instances),
            'active_models': len([m for m in self.model_instances.values() if m.enabled]),
            'model_status': {},
            'performance_summary': {},
            'available_models': list(self.available_models.keys())
        }
        
        # Model status summary
        for status in ModelStatus:
            count = len([s for s in self.model_status.values() if s == status])
            stats['model_status'][status.value] = count
        
        # Performance summary
        if self.model_performance:
            avg_accuracy = statistics.mean(
                p.get('accuracy', 0) for p in self.model_performance.values()
            )
            avg_latency = statistics.mean(
                p.get('latency', 0) for p in self.model_performance.values()
            )
            stats['performance_summary'] = {
                'avg_accuracy': avg_accuracy,
                'avg_latency': avg_latency,
                'models_tracked': len(self.model_performance)
            }
        
        return stats
    
    def _register_advanced_models(self):
        """Register advanced model classes."""
        # This would register additional specialized models
        pass
    
    def _load_default_configs(self):
        """Load default model configurations."""
        default_configs = [
            ('echo_model', ModelType.LANGUAGE_MODEL, 'ensemble.ensemble_core.EchoModel', 0.5),
            ('analytics_model', ModelType.ANALYTICS_MODULE, 'ensemble.ensemble_core.AnalyticsModel', 1.0),
            ('reasoning_model', ModelType.REASONING_ENGINE, 'ensemble.ensemble_core.ReasoningModel', 1.2)
        ]
        
        for model_id, model_type, model_class, weight in default_configs:
            self.add_model_config(model_id, model_type, model_class, weight)
    
    def _instantiate_model(self, config: ModelConfig) -> Optional[ModelInterface]:
        """Instantiate a model from its configuration."""
        try:
            # Import the model class
            module_path, class_name = config.model_class.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            model_class = getattr(module, class_name)
            
            # Create instance
            instance = model_class(
                model_id=config.model_id,
                model_type=config.model_type,
                weight=config.weight
            )
            
            return instance
            
        except Exception as e:
            logger.error(f"Error instantiating model {config.model_id}: {e}")
            return None
    
    def _start_performance_monitoring(self):
        """Start background performance monitoring."""
        def monitor_performance():
            while self.auto_update_enabled:
                try:
                    # Update performance metrics from ensemble
                    if self.ensemble_core and hasattr(self.ensemble_core, 'model_performance'):
                        self.model_performance.update(self.ensemble_core.model_performance)
                    
                    # Auto-optimize periodically
                    if len(self.model_performance) > 0:
                        self.auto_optimize_models()
                    
                    time.sleep(300)  # Check every 5 minutes
                    
                except Exception as e:
                    logger.error(f"Error in performance monitoring: {e}")
                    time.sleep(60)
        
        monitoring_thread = threading.Thread(target=monitor_performance, daemon=True)
        monitoring_thread.start()
