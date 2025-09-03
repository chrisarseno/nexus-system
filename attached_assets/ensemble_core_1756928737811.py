"""
Enhanced ensemble inference core for multi-model decision making.
"""

import logging
import time
import threading
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

logger = logging.getLogger(__name__)

class ModelType(Enum):
    LANGUAGE_MODEL = "language_model"
    REASONING_ENGINE = "reasoning_engine"
    ANALYTICS_MODULE = "analytics_module"
    PATTERN_MATCHER = "pattern_matcher"
    KNOWLEDGE_RETRIEVER = "knowledge_retriever"

class VotingStrategy(Enum):
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    CONFIDENCE_BASED = "confidence_based"
    CONSENSUS = "consensus"
    HYBRID = "hybrid"

@dataclass
class ModelResult:
    """Result from a single model."""
    model_id: str
    prediction: Any
    confidence: float
    processing_time: float
    metadata: Dict[str, Any] = None

@dataclass
class EnsembleResult:
    """Final ensemble result."""
    prediction: Any
    confidence: float
    voting_strategy: str
    model_results: List[ModelResult]
    processing_time: float
    consensus_score: float
    metadata: Dict[str, Any] = None

class ModelInterface:
    """Base interface for ensemble models."""
    
    def __init__(self, model_id: str, model_type: ModelType, weight: float = 1.0):
        self.model_id = model_id
        self.model_type = model_type
        self.weight = weight
        self.enabled = True
        self.performance_history = []
        
    def predict(self, input_data: Any) -> ModelResult:
        """Make a prediction. To be implemented by subclasses."""
        raise NotImplementedError
    
    def update_performance(self, accuracy: float, latency: float):
        """Update performance metrics."""
        self.performance_history.append({
            'accuracy': accuracy,
            'latency': latency,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

class EchoModel(ModelInterface):
    """Simple echo model for testing."""
    
    def predict(self, input_data: Any) -> ModelResult:
        start_time = time.time()
        result = f"Echo: {str(input_data)}"
        confidence = 0.8
        processing_time = time.time() - start_time
        
        return ModelResult(
            model_id=self.model_id,
            prediction=result,
            confidence=confidence,
            processing_time=processing_time,
            metadata={'model_type': self.model_type.value}
        )

class AnalyticsModel(ModelInterface):
    """Analytics-focused model."""
    
    def predict(self, input_data: Any) -> ModelResult:
        start_time = time.time()
        
        # Perform analytics
        if isinstance(input_data, str):
            if "?" in input_data:
                result = f"Question analysis: {len(input_data)} characters, likely seeking information"
                confidence = 0.9
            else:
                result = f"Text analysis: {len(input_data.split())} words, statement type"
                confidence = 0.7
        elif isinstance(input_data, (list, tuple)) and all(isinstance(x, (int, float)) for x in input_data):
            avg = sum(input_data) / len(input_data)
            result = f"Numerical analysis: Average={avg:.2f}, Range={max(input_data)-min(input_data)}"
            confidence = 0.95
        else:
            result = f"General analysis: {type(input_data).__name__} data structure"
            confidence = 0.6
            
        processing_time = time.time() - start_time
        
        return ModelResult(
            model_id=self.model_id,
            prediction=result,
            confidence=confidence,
            processing_time=processing_time,
            metadata={'analysis_type': 'statistical'}
        )

class ReasoningModel(ModelInterface):
    """Reasoning-focused model."""
    
    def predict(self, input_data: Any) -> ModelResult:
        start_time = time.time()
        
        # Perform reasoning
        if isinstance(input_data, str):
            input_lower = input_data.lower()
            if any(word in input_lower for word in ["calculate", "solve", "compute"]):
                # Look for mathematical operations
                import re
                numbers = re.findall(r'\d+', input_data)
                if len(numbers) >= 2:
                    nums = [int(n) for n in numbers[:2]]
                    if "+" in input_data or "plus" in input_data:
                        result = f"Mathematical reasoning: {nums[0]} + {nums[1]} = {sum(nums)}"
                        confidence = 0.95
                    elif "-" in input_data or "minus" in input_data:
                        result = f"Mathematical reasoning: {nums[0]} - {nums[1]} = {nums[0] - nums[1]}"
                        confidence = 0.95
                    else:
                        result = f"Mathematical reasoning: Operations on {nums[0]} and {nums[1]}"
                        confidence = 0.8
                else:
                    result = "Reasoning: Mathematical operation requested but numbers unclear"
                    confidence = 0.5
            elif "why" in input_lower:
                result = "Reasoning: Causal explanation requested - requires domain knowledge"
                confidence = 0.7
            elif "how" in input_lower:
                result = "Reasoning: Procedural explanation requested - requires step-by-step analysis"
                confidence = 0.7
            else:
                result = f"Reasoning: General logical analysis of statement"
                confidence = 0.6
        else:
            result = "Reasoning: Non-textual input requires specialized logical framework"
            confidence = 0.5
            
        processing_time = time.time() - start_time
        
        return ModelResult(
            model_id=self.model_id,
            prediction=result,
            confidence=confidence,
            processing_time=processing_time,
            metadata={'reasoning_type': 'logical'}
        )

class EnsembleCore:
    """
    Enhanced ensemble inference system that coordinates multiple models
    with advanced voting strategies and adaptive weighting.
    """
    
    def __init__(self, knowledge_base=None):
        self.models: List[ModelInterface] = []
        self.voting_strategy = VotingStrategy.WEIGHTED
        self.confidence_threshold = 0.5
        self.consensus_threshold = 0.7
        self.knowledge_base = knowledge_base
        self.initialized = False
        
        # Performance tracking
        self.ensemble_history = []
        self.model_performance = {}
        
        # Concurrency
        self.max_workers = 4
        self.timeout_seconds = 30
        
        # AI System Components (initialized later)
        self.quarantine_manager = None
        self.vector_store = None
        self.domain_manager = None
        self.policy_engine = None
        self.memory_manager = None
        self.self_learning_system = None
        self.graph_network = None
        self.performance_optimizer = None
        self.ethics_monitor = None
        self.creative_reasoning = None
        self.multimodal_processor = None
        self.experiment_framework = None
        self.autonomous_research = None
        self.federated_learning = None
        self.research_intelligence = None
        self.collaborative_intelligence = None
        self.worldwide_federation = None
        self.international_collaboration = None
        self.global_safety_standards = None
        self.planetary_intelligence = None
        
    def initialize(self):
        """Initialize the ensemble system with default models."""
        if self.initialized:
            return
            
        # Add default models
        self.add_model(EchoModel("echo_model", ModelType.LANGUAGE_MODEL, weight=0.5))
        self.add_model(AnalyticsModel("analytics_model", ModelType.ANALYTICS_MODULE, weight=1.0))
        self.add_model(ReasoningModel("reasoning_model", ModelType.REASONING_ENGINE, weight=1.2))
        
        self.initialized = True
        logger.info(f"Ensemble core initialized with {len(self.models)} models")
        
    def add_model(self, model: ModelInterface):
        """Add a model to the ensemble."""
        self.models.append(model)
        self.model_performance[model.model_id] = {
            'total_calls': 0,
            'avg_accuracy': 0.8,  # Default
            'avg_latency': 0.1,
            'success_rate': 1.0
        }
        logger.info(f"Added model {model.model_id} with weight {model.weight}")
        
    def remove_model(self, model_id: str) -> bool:
        """Remove a model from the ensemble."""
        for i, model in enumerate(self.models):
            if model.model_id == model_id:
                del self.models[i]
                if model_id in self.model_performance:
                    del self.model_performance[model_id]
                logger.info(f"Removed model {model_id}")
                return True
        return False
        
    def set_voting_strategy(self, strategy: VotingStrategy):
        """Set the voting strategy for ensemble decisions."""
        self.voting_strategy = strategy
        logger.info(f"Set voting strategy to {strategy.value}")
        
    def predict(self, input_data: Any, strategy: VotingStrategy = None) -> EnsembleResult:
        """
        Run ensemble prediction on input data with specified strategy.
        """
        if not self.initialized:
            self.initialize()
            
        if not self.models:
            return EnsembleResult(
                prediction="No models available",
                confidence=0.0,
                voting_strategy="none",
                model_results=[],
                processing_time=0.0,
                consensus_score=0.0,
                metadata={'error': 'No models available'}
            )
            
        start_time = time.time()
        strategy = strategy or self.voting_strategy
        
        # Get predictions from all models
        model_results = self._get_model_predictions(input_data)
        
        if not model_results:
            return EnsembleResult(
                prediction="All models failed",
                confidence=0.0,
                voting_strategy=strategy.value,
                model_results=[],
                processing_time=time.time() - start_time,
                consensus_score=0.0,
                metadata={'error': 'All models failed'}
            )
        
        # Apply voting strategy
        ensemble_prediction, ensemble_confidence, consensus_score = self._apply_voting_strategy(
            model_results, strategy
        )
        
        processing_time = time.time() - start_time
        
        # Create result
        result = EnsembleResult(
            prediction=ensemble_prediction,
            confidence=ensemble_confidence,
            voting_strategy=strategy.value,
            model_results=model_results,
            processing_time=processing_time,
            consensus_score=consensus_score,
            metadata={
                'active_models': len([m for m in self.models if m.enabled]),
                'successful_models': len(model_results),
                'input_type': type(input_data).__name__
            }
        )
        
        # Update performance tracking
        self._update_ensemble_history(result)
        
        return result
        
    def _get_model_predictions(self, input_data: Any) -> List[ModelResult]:
        """Get predictions from all enabled models."""
        model_results = []
        
        # Use threading for concurrent model execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all model tasks
            future_to_model = {}
            for model in self.models:
                if model.enabled:
                    future = executor.submit(self._safe_model_predict, model, input_data)
                    future_to_model[future] = model
            
            # Collect results with timeout
            for future in as_completed(future_to_model, timeout=self.timeout_seconds):
                model = future_to_model[future]
                try:
                    result = future.result()
                    if result:
                        model_results.append(result)
                        # Update model performance
                        self._update_model_performance(model.model_id, True, result.processing_time)
                except Exception as e:
                    logger.error(f"Model {model.model_id} failed: {e}")
                    self._update_model_performance(model.model_id, False, 0.0)
        
        return model_results
        
    def _safe_model_predict(self, model: ModelInterface, input_data: Any) -> Optional[ModelResult]:
        """Safely execute model prediction with error handling."""
        try:
            return model.predict(input_data)
        except Exception as e:
            logger.error(f"Error in model {model.model_id}: {e}")
            return None
            
    def _apply_voting_strategy(self, model_results: List[ModelResult], 
                             strategy: VotingStrategy) -> tuple:
        """Apply the specified voting strategy to model results."""
        
        if strategy == VotingStrategy.MAJORITY:
            return self._majority_vote(model_results)
        elif strategy == VotingStrategy.WEIGHTED:
            return self._weighted_vote(model_results)
        elif strategy == VotingStrategy.CONFIDENCE_BASED:
            return self._confidence_based_vote(model_results)
        elif strategy == VotingStrategy.CONSENSUS:
            return self._consensus_vote(model_results)
        elif strategy == VotingStrategy.HYBRID:
            return self._hybrid_vote(model_results)
        else:
            return self._weighted_vote(model_results)
    
    def _majority_vote(self, model_results: List[ModelResult]) -> tuple:
        """Simple majority voting."""
        if not model_results:
            return "No results", 0.0, 0.0
        
        # For simplicity, return the prediction from the highest confidence model
        best_result = max(model_results, key=lambda r: r.confidence)
        consensus_score = best_result.confidence
        
        return best_result.prediction, best_result.confidence, consensus_score
    
    def _weighted_vote(self, model_results: List[ModelResult]) -> tuple:
        """Weighted voting based on model weights."""
        if not model_results:
            return "No results", 0.0, 0.0
        
        # Get model weights
        weighted_predictions = []
        total_weight = 0.0
        
        for result in model_results:
            # Find the model weight
            model_weight = 1.0
            for model in self.models:
                if model.model_id == result.model_id:
                    model_weight = model.weight
                    break
            
            weight = model_weight * result.confidence
            weighted_predictions.append((result.prediction, weight))
            total_weight += weight
        
        if total_weight == 0:
            return model_results[0].prediction, 0.0, 0.0
        
        # For now, return the highest weighted prediction
        best_prediction = max(weighted_predictions, key=lambda x: x[1])
        ensemble_confidence = best_prediction[1] / total_weight
        
        # Calculate consensus score
        confidences = [r.confidence for r in model_results]
        consensus_score = 1.0 - statistics.stdev(confidences) if len(confidences) > 1 else 1.0
        
        return best_prediction[0], ensemble_confidence, consensus_score
    
    def _confidence_based_vote(self, model_results: List[ModelResult]) -> tuple:
        """Voting based purely on confidence scores."""
        if not model_results:
            return "No results", 0.0, 0.0
        
        # Return the highest confidence result
        best_result = max(model_results, key=lambda r: r.confidence)
        
        # Calculate consensus based on confidence agreement
        confidences = [r.confidence for r in model_results]
        consensus_score = 1.0 - statistics.stdev(confidences) if len(confidences) > 1 else 1.0
        
        return best_result.prediction, best_result.confidence, consensus_score
    
    def _consensus_vote(self, model_results: List[ModelResult]) -> tuple:
        """Consensus-based voting requiring agreement threshold."""
        if not model_results:
            return "No results", 0.0, 0.0
        
        # Calculate average confidence
        avg_confidence = sum(r.confidence for r in model_results) / len(model_results)
        
        # Check if consensus threshold is met
        if avg_confidence >= self.consensus_threshold:
            # Return weighted average of high-confidence results
            high_conf_results = [r for r in model_results if r.confidence >= self.confidence_threshold]
            if high_conf_results:
                best_result = max(high_conf_results, key=lambda r: r.confidence)
                return best_result.prediction, avg_confidence, avg_confidence
        
        # Fall back to weighted voting
        return self._weighted_vote(model_results)
    
    def _hybrid_vote(self, model_results: List[ModelResult]) -> tuple:
        """Hybrid strategy combining multiple voting methods."""
        if not model_results:
            return "No results", 0.0, 0.0
        
        # Try consensus first
        consensus_pred, consensus_conf, consensus_score = self._consensus_vote(model_results)
        
        if consensus_score >= self.consensus_threshold:
            return consensus_pred, consensus_conf, consensus_score
        
        # Fall back to confidence-based voting
        return self._confidence_based_vote(model_results)
    
    def _update_model_performance(self, model_id: str, success: bool, latency: float):
        """Update model performance tracking."""
        if model_id not in self.model_performance:
            self.model_performance[model_id] = {
                'total_calls': 0,
                'avg_accuracy': 0.8,
                'avg_latency': 0.1,
                'success_rate': 1.0
            }
        
        perf = self.model_performance[model_id]
        perf['total_calls'] += 1
        
        # Update success rate
        old_success_rate = perf['success_rate']
        new_success_rate = (old_success_rate * (perf['total_calls'] - 1) + (1.0 if success else 0.0)) / perf['total_calls']
        perf['success_rate'] = new_success_rate
        
        # Update latency
        if success:
            old_latency = perf['avg_latency']
            perf['avg_latency'] = (old_latency * (perf['total_calls'] - 1) + latency) / perf['total_calls']
    
    def _update_ensemble_history(self, result: EnsembleResult):
        """Update ensemble history for analysis."""
        self.ensemble_history.append(result)
        
        # Keep only recent history
        if len(self.ensemble_history) > 1000:
            self.ensemble_history = self.ensemble_history[-1000:]
