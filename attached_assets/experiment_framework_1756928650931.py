"""
Advanced Experimentation Framework
Provides systematic A/B testing, hypothesis validation, and scientific methodology automation.
"""

import logging
import json
import time
import random
import statistics
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
import uuid
import math

logger = logging.getLogger(__name__)

class ExperimentType(Enum):
    AB_TEST = "ab_test"
    MULTIVARIATE = "multivariate"
    SPLIT_TEST = "split_test"
    FACTORIAL = "factorial"
    SEQUENTIAL = "sequential"
    BAYESIAN = "bayesian"

class ExperimentStatus(Enum):
    DESIGNED = "designed"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StatisticalSignificance(Enum):
    NOT_SIGNIFICANT = "not_significant"
    MARGINALLY_SIGNIFICANT = "marginally_significant"
    SIGNIFICANT = "significant"
    HIGHLY_SIGNIFICANT = "highly_significant"

@dataclass
class ExperimentVariant:
    """Represents a variant in an experiment."""
    variant_id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    allocation_weight: float = 1.0
    is_control: bool = False

@dataclass
class ExperimentMetric:
    """Defines a metric to measure in experiments."""
    metric_id: str
    name: str
    description: str
    metric_type: str  # 'conversion', 'continuous', 'count', 'rate'
    primary: bool = False
    direction: str = "increase"  # 'increase', 'decrease', 'neutral'
    minimum_detectable_effect: float = 0.05
    baseline_value: Optional[float] = None

@dataclass
class ExperimentDesign:
    """Complete experiment design specification."""
    experiment_id: str
    name: str
    description: str
    experiment_type: ExperimentType
    hypothesis: str
    variants: List[ExperimentVariant]
    metrics: List[ExperimentMetric]
    target_sample_size: int
    confidence_level: float = 0.95
    power: float = 0.8
    duration_days: int = 14
    traffic_allocation: float = 1.0
    randomization_unit: str = "user"
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ExperimentResult:
    """Results from an experiment."""
    experiment_id: str
    variant_id: str
    metric_id: str
    sample_size: int
    mean_value: float
    std_dev: float
    confidence_interval: Tuple[float, float]
    statistical_significance: StatisticalSignificance
    p_value: float
    effect_size: float
    practical_significance: bool
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ExperimentReport:
    """Comprehensive experiment report."""
    experiment_id: str
    experiment_name: str
    status: ExperimentStatus
    start_date: datetime
    end_date: Optional[datetime]
    duration_actual: Optional[timedelta]
    total_participants: int
    results: List[ExperimentResult]
    conclusions: List[str]
    recommendations: List[str]
    statistical_power_achieved: float
    effect_sizes: Dict[str, float]
    generated_at: datetime = field(default_factory=datetime.now)

class ExperimentFramework:
    """
    Advanced experimentation framework providing systematic A/B testing,
    hypothesis validation, and scientific methodology automation.
    """
    
    def __init__(self):
        self.active_experiments: Dict[str, ExperimentDesign] = {}
        self.experiment_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.experiment_results: Dict[str, List[ExperimentResult]] = defaultdict(list)
        self.experiment_reports: Dict[str, ExperimentReport] = {}
        
        # Statistical configurations
        self.statistical_config = {
            'confidence_levels': [0.90, 0.95, 0.99],
            'power_thresholds': [0.8, 0.9],
            'effect_size_thresholds': {
                'small': 0.2,
                'medium': 0.5,
                'large': 0.8
            },
            'minimum_sample_size': 100,
            'maximum_duration_days': 90
        }
        
        # Experiment tracking
        self.participant_assignments: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.experiment_history = deque(maxlen=1000)
        self.experimentation_stats = defaultdict(int)
        
        # Bayesian updating for adaptive experiments
        self.bayesian_priors: Dict[str, Dict[str, Any]] = {}
        self.adaptive_thresholds = {
            'early_stopping_probability': 0.95,
            'futility_threshold': 0.1,
            'superiority_threshold': 0.95
        }
        
        # Background processing
        self.monitoring_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        self.initialized = False
    
    def initialize(self):
        """Initialize the experimentation framework."""
        if self.initialized:
            return
            
        logger.info("Initializing Experimentation Framework...")
        
        # Load statistical libraries and configurations
        self._initialize_statistical_tools()
        
        # Start experiment monitoring
        self._start_experiment_monitoring()
        
        self.initialized = True
        logger.info("Experimentation Framework initialized")
    
    def design_experiment(self, experiment_spec: Dict[str, Any]) -> ExperimentDesign:
        """Design a new experiment with proper statistical planning."""
        try:
            # Validate experiment specification
            self._validate_experiment_spec(experiment_spec)
            
            experiment_id = experiment_spec.get('experiment_id') or f"exp_{int(time.time())}"
            
            # Create variants
            variants = []
            for variant_spec in experiment_spec['variants']:
                variant = ExperimentVariant(
                    variant_id=variant_spec['variant_id'],
                    name=variant_spec['name'],
                    description=variant_spec['description'],
                    parameters=variant_spec['parameters'],
                    allocation_weight=variant_spec.get('allocation_weight', 1.0),
                    is_control=variant_spec.get('is_control', False)
                )
                variants.append(variant)
            
            # Create metrics
            metrics = []
            for metric_spec in experiment_spec['metrics']:
                metric = ExperimentMetric(
                    metric_id=metric_spec['metric_id'],
                    name=metric_spec['name'],
                    description=metric_spec['description'],
                    metric_type=metric_spec['metric_type'],
                    primary=metric_spec.get('primary', False),
                    direction=metric_spec.get('direction', 'increase'),
                    minimum_detectable_effect=metric_spec.get('minimum_detectable_effect', 0.05),
                    baseline_value=metric_spec.get('baseline_value')
                )
                metrics.append(metric)
            
            # Calculate sample size requirements
            target_sample_size = self._calculate_sample_size(
                metrics, 
                experiment_spec.get('confidence_level', 0.95),
                experiment_spec.get('power', 0.8)
            )
            
            # Create experiment design
            design = ExperimentDesign(
                experiment_id=experiment_id,
                name=experiment_spec['name'],
                description=experiment_spec['description'],
                experiment_type=ExperimentType(experiment_spec.get('experiment_type', 'ab_test')),
                hypothesis=experiment_spec['hypothesis'],
                variants=variants,
                metrics=metrics,
                target_sample_size=target_sample_size,
                confidence_level=experiment_spec.get('confidence_level', 0.95),
                power=experiment_spec.get('power', 0.8),
                duration_days=experiment_spec.get('duration_days', 14),
                traffic_allocation=experiment_spec.get('traffic_allocation', 1.0),
                randomization_unit=experiment_spec.get('randomization_unit', 'user')
            )
            
            logger.info(f"Designed experiment: {experiment_id}")
            return design
            
        except Exception as e:
            logger.error(f"Error designing experiment: {e}")
            raise
    
    def start_experiment(self, design: ExperimentDesign) -> bool:
        """Start running an experiment."""
        try:
            with self.lock:
                # Validate experiment can be started
                if design.experiment_id in self.active_experiments:
                    raise ValueError(f"Experiment {design.experiment_id} is already running")
                
                # Initialize experiment data structures
                self.active_experiments[design.experiment_id] = design
                self.experiment_data[design.experiment_id] = []
                self.participant_assignments[design.experiment_id] = {}
                
                # Initialize Bayesian priors if needed
                if design.experiment_type == ExperimentType.BAYESIAN:
                    self._initialize_bayesian_priors(design)
                
                # Record experiment start
                self.experiment_history.append({
                    'experiment_id': design.experiment_id,
                    'action': 'started',
                    'timestamp': datetime.now(),
                    'design': asdict(design)
                })
                
                self.experimentation_stats['experiments_started'] += 1
                
                logger.info(f"Started experiment: {design.experiment_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error starting experiment: {e}")
            return False
    
    def assign_participant(self, experiment_id: str, participant_id: str, 
                         context: Dict[str, Any] = None) -> Optional[str]:
        """Assign a participant to an experiment variant."""
        try:
            if experiment_id not in self.active_experiments:
                return None
            
            design = self.active_experiments[experiment_id]
            
            # Check if participant already assigned
            if participant_id in self.participant_assignments[experiment_id]:
                return self.participant_assignments[experiment_id][participant_id]
            
            # Apply traffic allocation
            if random.random() > design.traffic_allocation:
                return None
            
            # Assign to variant based on weights
            variant = self._weighted_random_assignment(design.variants, participant_id)
            
            # Store assignment
            self.participant_assignments[experiment_id][participant_id] = variant.variant_id
            
            # Record assignment
            self.experiment_data[experiment_id].append({
                'participant_id': participant_id,
                'variant_id': variant.variant_id,
                'assignment_time': datetime.now(),
                'context': context or {}
            })
            
            return variant.variant_id
            
        except Exception as e:
            logger.error(f"Error assigning participant: {e}")
            return None
    
    def record_metric(self, experiment_id: str, participant_id: str, 
                     metric_id: str, value: float, timestamp: datetime = None) -> bool:
        """Record a metric value for a participant in an experiment."""
        try:
            if experiment_id not in self.active_experiments:
                return False
            
            if participant_id not in self.participant_assignments[experiment_id]:
                return False
            
            variant_id = self.participant_assignments[experiment_id][participant_id]
            
            # Record metric data
            metric_record = {
                'experiment_id': experiment_id,
                'participant_id': participant_id,
                'variant_id': variant_id,
                'metric_id': metric_id,
                'value': value,
                'timestamp': timestamp or datetime.now()
            }
            
            self.experiment_data[experiment_id].append(metric_record)
            
            # Check for early stopping conditions
            if self._should_check_early_stopping(experiment_id):
                self._evaluate_early_stopping(experiment_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording metric: {e}")
            return False
    
    def analyze_experiment(self, experiment_id: str, 
                         interim: bool = False) -> List[ExperimentResult]:
        """Analyze experiment results with statistical testing."""
        try:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            design = self.active_experiments[experiment_id]
            data = self.experiment_data[experiment_id]
            
            results = []
            
            # Analyze each metric
            for metric in design.metrics:
                metric_results = self._analyze_metric(experiment_id, metric, data, interim)
                results.extend(metric_results)
            
            # Store results
            self.experiment_results[experiment_id] = results
            
            logger.info(f"Analyzed experiment {experiment_id}: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing experiment: {e}")
            return []
    
    def generate_experiment_report(self, experiment_id: str) -> ExperimentReport:
        """Generate a comprehensive experiment report."""
        try:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            design = self.active_experiments[experiment_id]
            results = self.experiment_results.get(experiment_id, [])
            
            # Calculate basic statistics
            total_participants = len(set(
                record['participant_id'] 
                for record in self.experiment_data[experiment_id]
                if 'participant_id' in record
            ))
            
            # Determine experiment status
            status = self._determine_experiment_status(experiment_id)
            
            # Generate conclusions and recommendations
            conclusions = self._generate_conclusions(design, results)
            recommendations = self._generate_recommendations(design, results)
            
            # Calculate achieved statistical power
            achieved_power = self._calculate_achieved_power(design, results)
            
            # Calculate effect sizes
            effect_sizes = self._calculate_effect_sizes(results)
            
            report = ExperimentReport(
                experiment_id=experiment_id,
                experiment_name=design.name,
                status=status,
                start_date=design.created_at,
                end_date=datetime.now() if status == ExperimentStatus.COMPLETED else None,
                duration_actual=datetime.now() - design.created_at,
                total_participants=total_participants,
                results=results,
                conclusions=conclusions,
                recommendations=recommendations,
                statistical_power_achieved=achieved_power,
                effect_sizes=effect_sizes
            )
            
            self.experiment_reports[experiment_id] = report
            
            logger.info(f"Generated report for experiment: {experiment_id}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating experiment report: {e}")
            raise
    
    def stop_experiment(self, experiment_id: str, reason: str = "manual") -> bool:
        """Stop a running experiment."""
        try:
            with self.lock:
                if experiment_id not in self.active_experiments:
                    return False
                
                # Analyze final results
                final_results = self.analyze_experiment(experiment_id, interim=False)
                
                # Generate final report
                final_report = self.generate_experiment_report(experiment_id)
                
                # Mark experiment as completed
                self.experiment_history.append({
                    'experiment_id': experiment_id,
                    'action': 'stopped',
                    'reason': reason,
                    'timestamp': datetime.now(),
                    'final_results': len(final_results)
                })
                
                # Remove from active experiments
                del self.active_experiments[experiment_id]
                
                self.experimentation_stats['experiments_completed'] += 1
                
                logger.info(f"Stopped experiment {experiment_id}: {reason}")
                return True
                
        except Exception as e:
            logger.error(f"Error stopping experiment: {e}")
            return False
    
    def get_experimentation_analytics(self) -> Dict[str, Any]:
        """Get comprehensive experimentation analytics."""
        try:
            with self.lock:
                active_count = len(self.active_experiments)
                total_completed = self.experimentation_stats.get('experiments_completed', 0)
                
                # Calculate success rates
                successful_experiments = sum(
                    1 for report in self.experiment_reports.values()
                    if any(result.statistical_significance != StatisticalSignificance.NOT_SIGNIFICANT 
                          for result in report.results)
                )
                
                success_rate = successful_experiments / max(total_completed, 1)
                
                # Recent activity
                recent_activity = self._calculate_recent_experimentation_activity()
                
                # Statistical power analysis
                power_analysis = self._analyze_statistical_power_trends()
                
                return {
                    'experiment_summary': {
                        'active_experiments': active_count,
                        'completed_experiments': total_completed,
                        'success_rate': round(success_rate, 3),
                        'total_participants': sum(
                            len(self.participant_assignments[exp_id]) 
                            for exp_id in self.participant_assignments
                        )
                    },
                    'recent_activity': recent_activity,
                    'power_analysis': power_analysis,
                    'statistical_configuration': self.statistical_config,
                    'experimentation_stats': dict(self.experimentation_stats),
                    'system_health': {
                        'monitoring_active': self.running,
                        'frameworks_loaded': self.initialized,
                        'data_integrity': self._check_data_integrity()
                    }
                }
                
        except Exception as e:
            logger.error(f"Error generating experimentation analytics: {e}")
            return {}
    
    def _calculate_sample_size(self, metrics: List[ExperimentMetric], 
                             confidence_level: float, power: float) -> int:
        """Calculate required sample size for experiment."""
        try:
            primary_metrics = [m for m in metrics if m.primary]
            if not primary_metrics:
                primary_metrics = metrics[:1]  # Use first metric if none marked primary
            
            max_sample_size = 0
            
            for metric in primary_metrics:
                # Simplified sample size calculation
                # In practice, this would use proper statistical formulas
                alpha = 1 - confidence_level
                beta = 1 - power
                effect_size = metric.minimum_detectable_effect
                
                # Cohen's formula approximation
                sample_size_per_group = max(
                    int((16 * (1.96 + 0.84) ** 2) / (effect_size ** 2)),
                    self.statistical_config['minimum_sample_size']
                )
                
                total_sample_size = sample_size_per_group * 2  # Assuming 2 groups
                max_sample_size = max(max_sample_size, total_sample_size)
            
            return max_sample_size
            
        except Exception as e:
            logger.error(f"Error calculating sample size: {e}")
            return self.statistical_config['minimum_sample_size']
    
    def _weighted_random_assignment(self, variants: List[ExperimentVariant], 
                                  participant_id: str) -> ExperimentVariant:
        """Assign participant to variant using weighted randomization."""
        # Use participant_id for consistent assignment
        random.seed(hash(participant_id))
        
        total_weight = sum(v.allocation_weight for v in variants)
        r = random.random() * total_weight
        
        current_weight = 0
        for variant in variants:
            current_weight += variant.allocation_weight
            if r <= current_weight:
                return variant
        
        return variants[-1]  # Fallback
    
    def _analyze_metric(self, experiment_id: str, metric: ExperimentMetric, 
                       data: List[Dict[str, Any]], interim: bool) -> List[ExperimentResult]:
        """Analyze a specific metric across experiment variants."""
        results = []
        
        try:
            # Group data by variant
            variant_data = defaultdict(list)
            for record in data:
                if record.get('metric_id') == metric.metric_id:
                    variant_data[record['variant_id']].append(record['value'])
            
            # Need at least 2 variants to compare
            if len(variant_data) < 2:
                return results
            
            variants = list(variant_data.keys())
            
            # Perform pairwise comparisons
            for i, variant_a in enumerate(variants):
                for variant_b in variants[i+1:]:
                    result = self._statistical_test(
                        variant_data[variant_a], 
                        variant_data[variant_b],
                        experiment_id, variant_a, variant_b, metric, interim
                    )
                    if result:
                        results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing metric {metric.metric_id}: {e}")
            return results
    
    def _statistical_test(self, data_a: List[float], data_b: List[float],
                         experiment_id: str, variant_a: str, variant_b: str,
                         metric: ExperimentMetric, interim: bool) -> Optional[ExperimentResult]:
        """Perform statistical test between two variants."""
        try:
            if len(data_a) < 10 or len(data_b) < 10:
                return None  # Insufficient data
            
            # Calculate basic statistics
            mean_a = statistics.mean(data_a)
            mean_b = statistics.mean(data_b)
            std_a = statistics.stdev(data_a) if len(data_a) > 1 else 0
            std_b = statistics.stdev(data_b) if len(data_b) > 1 else 0
            
            # Simplified t-test (in practice, would use proper statistical libraries)
            n_a, n_b = len(data_a), len(data_b)
            pooled_std = math.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
            
            if pooled_std == 0:
                return None
            
            t_stat = (mean_a - mean_b) / (pooled_std * math.sqrt(1/n_a + 1/n_b))
            
            # Simplified p-value calculation (approximation)
            p_value = 2 * (1 - abs(t_stat) / (abs(t_stat) + math.sqrt(n_a + n_b - 2)))
            p_value = max(0.001, min(0.999, p_value))  # Clamp p-value
            
            # Determine significance
            if p_value < 0.001:
                significance = StatisticalSignificance.HIGHLY_SIGNIFICANT
            elif p_value < 0.05:
                significance = StatisticalSignificance.SIGNIFICANT
            elif p_value < 0.1:
                significance = StatisticalSignificance.MARGINALLY_SIGNIFICANT
            else:
                significance = StatisticalSignificance.NOT_SIGNIFICANT
            
            # Calculate effect size (Cohen's d)
            effect_size = abs(mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
            
            # Calculate confidence interval (simplified)
            margin_error = 1.96 * pooled_std * math.sqrt(1/n_a + 1/n_b)
            ci_lower = (mean_a - mean_b) - margin_error
            ci_upper = (mean_a - mean_b) + margin_error
            
            # Practical significance
            practical_significant = effect_size > metric.minimum_detectable_effect
            
            return ExperimentResult(
                experiment_id=experiment_id,
                variant_id=f"{variant_a}_vs_{variant_b}",
                metric_id=metric.metric_id,
                sample_size=n_a + n_b,
                mean_value=mean_a - mean_b,  # Difference in means
                std_dev=pooled_std,
                confidence_interval=(ci_lower, ci_upper),
                statistical_significance=significance,
                p_value=p_value,
                effect_size=effect_size,
                practical_significance=practical_significant
            )
            
        except Exception as e:
            logger.error(f"Error in statistical test: {e}")
            return None
    
    def _start_experiment_monitoring(self):
        """Start background experiment monitoring."""
        if not self.monitoring_thread:
            self.running = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logger.info("Experiment monitoring started")
    
    def _monitoring_loop(self):
        """Main monitoring loop for experiments."""
        while self.running:
            try:
                time.sleep(3600)  # Check every hour
                
                # Check experiment completion conditions
                self._check_experiment_completion()
                
                # Update experiment statistics
                self._update_experiment_statistics()
                
                # Check for data quality issues
                self._check_data_quality()
                
            except Exception as e:
                logger.error(f"Error in experiment monitoring loop: {e}")
    
    def shutdown(self):
        """Shutdown the experimentation framework."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Experimentation Framework shutdown completed")