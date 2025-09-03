"""
Self-Learning System for Sentinel AI
Enables autonomous learning, model improvement, and adaptive optimization.
"""

import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import threading
import time
import math

logger = logging.getLogger(__name__)

@dataclass
class LearningMetric:
    """Represents a learning performance metric."""
    metric_name: str
    current_value: float
    target_value: float
    improvement_rate: float
    last_updated: datetime
    trend: str  # 'improving', 'stable', 'declining'

@dataclass
class AdaptationRule:
    """Represents an adaptation rule learned by the system."""
    rule_id: str
    condition: Dict[str, Any]
    action: Dict[str, Any]
    success_rate: float
    usage_count: int
    confidence: float
    created: datetime
    last_applied: datetime

@dataclass
class ModelPerformanceProfile:
    """Performance profile for individual models."""
    model_id: str
    accuracy_trend: List[float]
    response_times: List[float]
    confidence_scores: List[float]
    error_patterns: Dict[str, int]
    optimal_parameters: Dict[str, Any]
    specialization_domains: List[str]
    last_updated: datetime

class SelfLearningSystem:
    """
    Advanced self-learning system that enables autonomous improvement
    of AI models, optimization strategies, and system performance.
    """
    
    def __init__(self):
        self.learning_metrics: Dict[str, LearningMetric] = {}
        self.adaptation_rules: Dict[str, AdaptationRule] = {}
        self.model_profiles: Dict[str, ModelPerformanceProfile] = {}
        self.learning_history = deque(maxlen=10000)
        self.optimization_candidates = []
        
        # Learning parameters
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.7
        self.confidence_threshold = 0.8
        self.max_experiments = 5
        
        # Active learning experiments
        self.active_experiments = {}
        self.experiment_results = []
        
        # Training data storage
        self.training_data = []
        self.learned_patterns_count = 0
        
        # Threading for continuous learning
        self.learning_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        self.initialized = False
    
    def initialize(self):
        """Initialize the self-learning system."""
        if self.initialized:
            return
            
        logger.info("Initializing Self-Learning System...")
        
        try:
            # Initialize learning metrics
            self._initialize_learning_metrics()
            logger.info("Learning metrics initialized")
            
            # Load existing learning data
            self._load_learning_data()
            logger.info("Learning data loaded")
            
            # Start continuous learning thread
            self._start_continuous_learning()
            logger.info("Continuous learning thread started")
            
            self.initialized = True
            logger.info("Self-Learning System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Self-Learning System: {e}")
            # Set as initialized anyway to avoid blocking the application
            self.initialized = True
    
    def record_interaction_outcome(self, interaction_data: Dict[str, Any]) -> bool:
        """Record the outcome of an AI interaction for learning."""
        try:
            with self.lock:
                # Extract learning signals
                learning_signals = self._extract_learning_signals(interaction_data)
                
                # Update model performance profiles
                self._update_model_profiles(interaction_data)
                
                # Update learning metrics
                self._update_learning_metrics(learning_signals)
                
                # Check for adaptation opportunities
                adaptations = self._identify_adaptation_opportunities(interaction_data)
                
                # Apply immediate adaptations if confidence is high
                for adaptation in adaptations:
                    if adaptation['confidence'] > self.confidence_threshold:
                        self._apply_adaptation(adaptation)
                
                # Record in learning history
                self.learning_history.append({
                    'timestamp': datetime.now(),
                    'interaction_data': interaction_data,
                    'learning_signals': learning_signals,
                    'adaptations_applied': len(adaptations)
                })
                
                logger.debug(f"Recorded interaction outcome with {len(learning_signals)} signals")
                return True
                
        except Exception as e:
            logger.error(f"Error recording interaction outcome: {e}")
            return False
    
    def optimize_model_parameters(self, model_id: str, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize parameters for a specific model based on performance data."""
        try:
            if model_id not in self.model_profiles:
                self._create_model_profile(model_id)
            
            profile = self.model_profiles[model_id]
            
            # Analyze current performance
            current_performance = self._analyze_model_performance(profile, performance_data)
            
            # Generate optimization candidates
            optimization_candidates = self._generate_optimization_candidates(profile, current_performance)
            
            # Select best optimization
            best_optimization = self._select_best_optimization(optimization_candidates)
            
            if best_optimization:
                # Apply optimization
                optimization_result = self._apply_model_optimization(model_id, best_optimization)
                
                # Update profile with new parameters
                profile.optimal_parameters.update(best_optimization['parameters'])
                profile.last_updated = datetime.now()
                
                logger.info(f"Optimized model {model_id}: {optimization_result}")
                return optimization_result
            else:
                logger.info(f"No beneficial optimizations found for model {model_id}")
                return {'status': 'no_optimization_needed'}
                
        except Exception as e:
            logger.error(f"Error optimizing model {model_id}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def learn_from_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """Learn from explicit user feedback."""
        try:
            # Extract feedback signals
            feedback_type = feedback_data.get('type', 'general')
            feedback_score = feedback_data.get('score', 0.5)
            feedback_text = feedback_data.get('text', '')
            context = feedback_data.get('context', {})
            
            # Create learning rule from feedback
            if feedback_score < 0.3:  # Negative feedback
                # Learn what to avoid
                avoid_rule = self._create_avoidance_rule(feedback_data)
                if avoid_rule:
                    self.adaptation_rules[avoid_rule.rule_id] = avoid_rule
            elif feedback_score > 0.7:  # Positive feedback
                # Reinforce successful patterns
                reinforce_rule = self._create_reinforcement_rule(feedback_data)
                if reinforce_rule:
                    self.adaptation_rules[reinforce_rule.rule_id] = reinforce_rule
            
            # Update learning metrics based on feedback
            metric_updates = self._process_feedback_metrics(feedback_data)
            for metric_name, update in metric_updates.items():
                if metric_name in self.learning_metrics:
                    self._update_metric(metric_name, update)
            
            logger.info(f"Learned from {feedback_type} feedback with score {feedback_score}")
            return True
            
        except Exception as e:
            logger.error(f"Error learning from feedback: {e}")
            return False
    
    def conduct_learning_experiment(self, experiment_config: Dict[str, Any]) -> str:
        """Conduct a controlled learning experiment."""
        try:
            experiment_id = f"exp_{int(time.time())}"
            
            experiment = {
                'id': experiment_id,
                'config': experiment_config,
                'start_time': datetime.now(),
                'status': 'running',
                'results': [],
                'baseline_metrics': self._capture_baseline_metrics()
            }
            
            # Validate experiment feasibility
            if not self._validate_experiment(experiment_config):
                logger.warning(f"Experiment {experiment_id} failed validation")
                return None
            
            # Start experiment
            self.active_experiments[experiment_id] = experiment
            
            # Schedule experiment evaluation
            self._schedule_experiment_evaluation(experiment_id)
            
            logger.info(f"Started learning experiment: {experiment_id}")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Error conducting learning experiment: {e}")
            return None
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about the learning progress and discoveries."""
        try:
            with self.lock:
                # Calculate learning progress
                learning_progress = self._calculate_learning_progress()
                
                # Get top performing adaptations
                top_adaptations = self._get_top_adaptations()
                
                # Analyze model improvements
                model_improvements = self._analyze_model_improvements()
                
                # Get experimental results
                experiment_summary = self._summarize_experiments()
                
                return {
                    'learning_progress': learning_progress,
                    'total_adaptations': len(self.adaptation_rules),
                    'active_experiments': len(self.active_experiments),
                    'model_profiles': len(self.model_profiles),
                    'top_adaptations': top_adaptations,
                    'model_improvements': model_improvements,
                    'experiment_summary': experiment_summary,
                    'learning_efficiency': self._calculate_learning_efficiency()
                }
                
        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            return {}
    
    def _initialize_learning_metrics(self):
        """Initialize key learning metrics."""
        base_metrics = [
            ('accuracy_improvement', 0.0, 0.95, 'improving'),
            ('response_time_optimization', 1.0, 0.5, 'improving'),
            ('confidence_calibration', 0.5, 0.85, 'stable'),
            ('user_satisfaction', 0.6, 0.9, 'improving'),
            ('model_stability', 0.8, 0.95, 'stable')
        ]
        
        for name, current, target, trend in base_metrics:
            self.learning_metrics[name] = LearningMetric(
                metric_name=name,
                current_value=current,
                target_value=target,
                improvement_rate=0.01,
                last_updated=datetime.now(),
                trend=trend
            )
    
    def _extract_learning_signals(self, interaction_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract learning signals from interaction data."""
        signals = []
        
        # Performance signals
        if 'response_time' in interaction_data:
            signals.append({
                'type': 'performance',
                'metric': 'response_time',
                'value': interaction_data['response_time'],
                'context': 'query_processing'
            })
        
        # Confidence signals
        if 'confidence' in interaction_data:
            signals.append({
                'type': 'confidence',
                'metric': 'prediction_confidence',
                'value': interaction_data['confidence'],
                'context': 'model_output'
            })
        
        # Quality signals
        if 'scoring_results' in interaction_data:
            for dimension, score in interaction_data['scoring_results'].items():
                signals.append({
                    'type': 'quality',
                    'metric': f'quality_{dimension}',
                    'value': score,
                    'context': 'response_evaluation'
                })
        
        return signals
    
    def _update_model_profiles(self, interaction_data: Dict[str, Any]):
        """Update performance profiles for models."""
        model_results = interaction_data.get('model_results', [])
        
        for result in model_results:
            model_id = result.get('model_id')
            if not model_id:
                continue
                
            if model_id not in self.model_profiles:
                self._create_model_profile(model_id)
            
            profile = self.model_profiles[model_id]
            
            # Update performance trends
            if 'accuracy' in result:
                profile.accuracy_trend.append(result['accuracy'])
                if len(profile.accuracy_trend) > 100:
                    profile.accuracy_trend.pop(0)
            
            if 'response_time' in result:
                profile.response_times.append(result['response_time'])
                if len(profile.response_times) > 100:
                    profile.response_times.pop(0)
            
            if 'confidence' in result:
                profile.confidence_scores.append(result['confidence'])
                if len(profile.confidence_scores) > 100:
                    profile.confidence_scores.pop(0)
            
            # Track error patterns
            if 'error_type' in result:
                error_type = result['error_type']
                profile.error_patterns[error_type] = profile.error_patterns.get(error_type, 0) + 1
            
            profile.last_updated = datetime.now()
    
    def _create_model_profile(self, model_id: str):
        """Create a new model performance profile."""
        self.model_profiles[model_id] = ModelPerformanceProfile(
            model_id=model_id,
            accuracy_trend=[],
            response_times=[],
            confidence_scores=[],
            error_patterns={},
            optimal_parameters={},
            specialization_domains=[],
            last_updated=datetime.now()
        )
    
    def _identify_adaptation_opportunities(self, interaction_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify opportunities for system adaptation."""
        opportunities = []
        
        # Low confidence adaptation
        confidence = interaction_data.get('confidence', 1.0)
        if confidence < 0.5:
            opportunities.append({
                'type': 'confidence_boost',
                'confidence': 0.7,
                'action': 'increase_ensemble_diversity',
                'expected_improvement': 0.2
            })
        
        # Slow response adaptation
        response_time = interaction_data.get('response_time', 0)
        if response_time > 10:  # seconds
            opportunities.append({
                'type': 'performance_optimization',
                'confidence': 0.8,
                'action': 'optimize_model_weights',
                'expected_improvement': 0.3
            })
        
        # Quality improvement adaptation
        scoring_results = interaction_data.get('scoring_results', {})
        low_quality_dimensions = [dim for dim, score in scoring_results.items() if score < 0.6]
        if low_quality_dimensions:
            opportunities.append({
                'type': 'quality_improvement',
                'confidence': 0.6,
                'action': f'focus_on_{low_quality_dimensions[0]}',
                'expected_improvement': 0.25
            })
        
        return opportunities
    
    def _apply_adaptation(self, adaptation: Dict[str, Any]):
        """Apply an adaptation to the system."""
        try:
            adaptation_type = adaptation['type']
            action = adaptation['action']
            
            # Create adaptation rule
            rule = AdaptationRule(
                rule_id=f"{adaptation_type}_{int(time.time())}",
                condition={'type': adaptation_type},
                action={'action': action},
                success_rate=0.5,  # Start neutral
                usage_count=1,
                confidence=adaptation['confidence'],
                created=datetime.now(),
                last_applied=datetime.now()
            )
            
            self.adaptation_rules[rule.rule_id] = rule
            
            logger.debug(f"Applied adaptation: {adaptation_type} -> {action}")
            
        except Exception as e:
            logger.error(f"Error applying adaptation: {e}")
    
    def _create_avoidance_rule(self, feedback_data: Dict[str, Any]) -> Optional[AdaptationRule]:
        """Create a rule to avoid patterns that led to negative feedback."""
        try:
            context = feedback_data.get('context', {})
            
            rule = AdaptationRule(
                rule_id=f"avoid_{int(time.time())}",
                condition=context,
                action={'action': 'avoid_pattern', 'pattern': context},
                success_rate=0.3,
                usage_count=0,
                confidence=0.7,
                created=datetime.now(),
                last_applied=datetime.now()
            )
            
            return rule
            
        except Exception as e:
            logger.error(f"Error creating avoidance rule: {e}")
            return None
    
    def _create_reinforcement_rule(self, feedback_data: Dict[str, Any]) -> Optional[AdaptationRule]:
        """Create a rule to reinforce patterns that led to positive feedback."""
        try:
            context = feedback_data.get('context', {})
            
            rule = AdaptationRule(
                rule_id=f"reinforce_{int(time.time())}",
                condition=context,
                action={'action': 'reinforce_pattern', 'pattern': context},
                success_rate=0.8,
                usage_count=0,
                confidence=0.8,
                created=datetime.now(),
                last_applied=datetime.now()
            )
            
            return rule
            
        except Exception as e:
            logger.error(f"Error creating reinforcement rule: {e}")
            return None
    
    def _calculate_learning_progress(self) -> Dict[str, float]:
        """Calculate overall learning progress."""
        progress = {}
        
        for metric_name, metric in self.learning_metrics.items():
            if metric.target_value != metric.current_value:
                progress_pct = min(1.0, abs(metric.current_value - metric.target_value) / 
                                 abs(metric.target_value) if metric.target_value != 0 else 1.0)
                progress[metric_name] = round(1.0 - progress_pct, 3)
            else:
                progress[metric_name] = 1.0
        
        return progress
    
    def _calculate_learning_efficiency(self) -> float:
        """Calculate the efficiency of the learning process."""
        if not self.learning_history:
            return 0.0
        
        # Simple efficiency based on adaptation success rate
        successful_adaptations = sum(1 for rule in self.adaptation_rules.values() 
                                   if rule.success_rate > 0.6)
        total_adaptations = len(self.adaptation_rules)
        
        if total_adaptations == 0:
            return 0.0
        
        efficiency = successful_adaptations / total_adaptations
        return round(efficiency, 3)
    
    def _start_continuous_learning(self):
        """Start the continuous learning thread."""
        if not self.learning_thread:
            self.running = True
            self.learning_thread = threading.Thread(target=self._continuous_learning_loop)
            self.learning_thread.daemon = True
            self.learning_thread.start()
            logger.info("Continuous learning started")
    
    def _continuous_learning_loop(self):
        """Main continuous learning loop."""
        while self.running:
            try:
                time.sleep(60)  # Run every minute
                
                # Evaluate active experiments
                self._evaluate_active_experiments()
                
                # Update learning metrics
                self._update_learning_trends()
                
                # Prune ineffective rules
                self._prune_ineffective_rules()
                
            except Exception as e:
                logger.error(f"Error in continuous learning loop: {e}")
    
    def _evaluate_active_experiments(self):
        """Evaluate active learning experiments."""
        completed_experiments = []
        
        for exp_id, experiment in self.active_experiments.items():
            duration = (datetime.now() - experiment['start_time']).total_seconds()
            
            if duration > 3600:  # 1 hour experiments
                # Evaluate experiment results
                results = self._evaluate_experiment_results(experiment)
                experiment['results'] = results
                experiment['status'] = 'completed'
                
                # Store results
                self.experiment_results.append(experiment)
                completed_experiments.append(exp_id)
                
                logger.info(f"Completed experiment {exp_id}: {results}")
        
        # Remove completed experiments
        for exp_id in completed_experiments:
            del self.active_experiments[exp_id]
    
    def _load_learning_data(self):
        """Load existing learning data."""
        # Placeholder for loading from persistent storage
        logger.info("Learning data loading not implemented yet")
    
    def shutdown(self):
        """Shutdown the self-learning system."""
        self.running = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        logger.info("Self-Learning System shutdown completed")
    
    def add_training_data(self, content: str, training_type: str, source_file: str) -> Dict[str, Any]:
        """Add new training data and process it."""
        try:
            training_entry = {
                'content': content,
                'training_type': training_type,
                'source_file': source_file,
                'timestamp': datetime.now(),
                'processed': False
            }
            
            self.training_data.append(training_entry)
            
            # Process the content based on type
            if training_type == 'knowledge':
                patterns = self._extract_knowledge_patterns(content)
                self.learned_patterns_count += len(patterns)
            elif training_type == 'conversation':
                patterns = self._extract_conversation_patterns(content)
                self.learned_patterns_count += len(patterns)
            elif training_type == 'task':
                patterns = self._extract_task_patterns(content)
                self.learned_patterns_count += len(patterns)
            
            # Mark as processed
            training_entry['processed'] = True
            
            # Generate training ID
            training_id = f"train_{int(time.time())}_{len(self.training_data)}"
            
            logger.info(f"Training data processed: {training_type} from {source_file}")
            
            return {
                'training_id': training_id,
                'patterns_extracted': len(patterns) if 'patterns' in locals() else 0,
                'estimated_time': '1-2 minutes',
                'status': 'processing'
            }
            
        except Exception as e:
            logger.error(f"Error adding training data: {e}")
            return {'error': str(e), 'training_id': None}
    
    def _extract_knowledge_patterns(self, content: str) -> List[Dict[str, Any]]:
        """Extract knowledge patterns from content."""
        patterns = []
        sentences = content.split('.')
        for sentence in sentences[:10]:  # Limit processing
            if len(sentence.strip()) > 10:
                patterns.append({
                    'type': 'knowledge',
                    'content': sentence.strip(),
                    'confidence': 0.8
                })
        return patterns
    
    def _extract_conversation_patterns(self, content: str) -> List[Dict[str, Any]]:
        """Extract conversation patterns from content."""
        patterns = []
        lines = content.split('\n')
        for line in lines[:10]:  # Limit processing
            if ':' in line and len(line.strip()) > 5:
                patterns.append({
                    'type': 'conversation',
                    'content': line.strip(),
                    'confidence': 0.7
                })
        return patterns
    
    def _extract_task_patterns(self, content: str) -> List[Dict[str, Any]]:
        """Extract task-specific patterns from content."""
        patterns = []
        words = content.split()
        if len(words) > 20:
            patterns.append({
                'type': 'task',
                'content': ' '.join(words[:50]),  # First 50 words
                'confidence': 0.6
            })
        return patterns