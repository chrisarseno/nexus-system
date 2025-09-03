"""
Advanced Neural Architecture Search Engine
Optimizes neural network configurations for breakthrough performance through evolutionary algorithms.
"""

import logging
import time
import threading
import random
import json
import math
import statistics
from typing import Dict, List, Any, Set, Tuple, Optional, Union
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
import copy
import numpy as np
from dataclasses import dataclass, asdict
import pickle
import hashlib

logger = logging.getLogger(__name__)

class LayerType(Enum):
    """Types of neural network layers."""
    DENSE = "dense"
    CONV2D = "conv2d"
    LSTM = "lstm"
    GRU = "gru"
    ATTENTION = "attention"
    TRANSFORMER = "transformer"
    RESIDUAL = "residual"
    DROPOUT = "dropout"
    BATCH_NORM = "batch_norm"
    ACTIVATION = "activation"

class ActivationType(Enum):
    """Types of activation functions."""
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    GELU = "gelu"
    SWISH = "swish"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"

class OptimizerType(Enum):
    """Types of optimizers."""
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    ADAMW = "adamw"

@dataclass
class LayerConfig:
    """Configuration for a neural network layer."""
    layer_type: LayerType
    units: Optional[int] = None
    kernel_size: Optional[Tuple[int, int]] = None
    stride: Optional[Tuple[int, int]] = None
    padding: str = "same"
    activation: Optional[ActivationType] = None
    dropout_rate: Optional[float] = None
    use_bias: bool = True
    regularization: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayerConfig':
        """Create from dictionary representation."""
        # Convert enum values
        if 'layer_type' in data:
            data['layer_type'] = LayerType(data['layer_type'])
        if 'activation' in data and data['activation']:
            data['activation'] = ActivationType(data['activation'])
        return cls(**data)

@dataclass
class ArchitectureConfig:
    """Complete neural architecture configuration."""
    layers: List[LayerConfig]
    optimizer: OptimizerType = OptimizerType.ADAM
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping: bool = True
    patience: int = 10
    validation_split: float = 0.2
    architecture_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate unique architecture ID if not provided."""
        if self.architecture_id is None:
            # Create hash based on architecture configuration
            config_str = json.dumps(self.to_dict(), sort_keys=True)
            self.architecture_id = hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'layers': [layer.to_dict() for layer in self.layers],
            'optimizer': self.optimizer.value,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'validation_split': self.validation_split,
            'architecture_id': self.architecture_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArchitectureConfig':
        """Create from dictionary representation."""
        layers = [LayerConfig.from_dict(layer_data) for layer_data in data['layers']]
        data['layers'] = layers
        if 'optimizer' in data:
            data['optimizer'] = OptimizerType(data['optimizer'])
        return cls(**data)

@dataclass
class PerformanceMetrics:
    """Performance metrics for architecture evaluation."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    loss: float = float('inf')
    training_time: float = 0.0
    inference_time: float = 0.0
    memory_usage: float = 0.0
    model_size: float = 0.0
    flops: float = 0.0
    energy_consumption: float = 0.0
    
    def composite_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate composite performance score."""
        if weights is None:
            weights = {
                'accuracy': 0.25,
                'f1_score': 0.20,
                'training_time': -0.15,  # Negative weight (lower is better)
                'inference_time': -0.10,
                'memory_usage': -0.10,
                'model_size': -0.10,
                'energy_consumption': -0.10
            }
        
        score = 0.0
        for metric, weight in weights.items():
            value = getattr(self, metric, 0.0)
            
            # Normalize time and size metrics (invert since lower is better)
            if metric in ['training_time', 'inference_time', 'memory_usage', 'model_size', 'energy_consumption']:
                # Use reciprocal for metrics where lower is better
                normalized_value = 1.0 / (1.0 + value) if value > 0 else 1.0
            else:
                normalized_value = min(1.0, value)  # Cap at 1.0
            
            score += weight * normalized_value
        
        return max(0.0, score)  # Ensure non-negative score

class ArchitecturePopulation:
    """Manages population of neural architectures for evolutionary search."""
    
    def __init__(self, population_size: int = 50, elite_size: int = 10):
        self.population_size = population_size
        self.elite_size = elite_size
        self.architectures: Dict[str, ArchitectureConfig] = {}
        self.performance: Dict[str, PerformanceMetrics] = {}
        self.generation = 0
        self.diversity_threshold = 0.7
        
    def add_architecture(self, config: ArchitectureConfig, metrics: Optional[PerformanceMetrics] = None):
        """Add architecture to population."""
        if config.architecture_id:
            self.architectures[config.architecture_id] = config
            if metrics:
                self.performance[config.architecture_id] = metrics
    
    def get_elite(self) -> List[ArchitectureConfig]:
        """Get top performing architectures."""
        if not self.performance:
            return list(self.architectures.values())[:self.elite_size]
        
        # Sort by composite score
        sorted_ids = sorted(
            self.performance.keys(),
            key=lambda x: self.performance[x].composite_score(),
            reverse=True
        )
        
        return [self.architectures[arch_id] for arch_id in sorted_ids[:self.elite_size]]
    
    def calculate_diversity(self, config1: ArchitectureConfig, config2: ArchitectureConfig) -> float:
        """Calculate diversity score between two architectures."""
        # Simple diversity based on layer differences
        layers1 = [layer.layer_type.value for layer in config1.layers]
        layers2 = [layer.layer_type.value for layer in config2.layers]
        
        # Jaccard similarity
        set1, set2 = set(layers1), set(layers2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        jaccard = intersection / union if union > 0 else 0
        return 1.0 - jaccard  # Diversity is inverse of similarity
    
    def maintain_diversity(self):
        """Ensure population maintains diversity."""
        if len(self.architectures) <= self.elite_size:
            return
        
        diverse_configs = []
        remaining_configs = list(self.architectures.values())
        
        # Start with best performer
        if self.performance:
            best_id = max(self.performance.keys(), key=lambda x: self.performance[x].composite_score())
            diverse_configs.append(self.architectures[best_id])
            remaining_configs = [c for c in remaining_configs if c.architecture_id != best_id]
        else:
            diverse_configs.append(remaining_configs.pop(0))
        
        # Add diverse architectures
        while len(diverse_configs) < self.population_size and remaining_configs:
            best_candidate = None
            best_diversity = -1
            
            for candidate in remaining_configs:
                min_diversity = min(
                    self.calculate_diversity(candidate, existing)
                    for existing in diverse_configs
                )
                
                if min_diversity > best_diversity:
                    best_diversity = min_diversity
                    best_candidate = candidate
            
            if best_candidate and best_diversity >= self.diversity_threshold:
                diverse_configs.append(best_candidate)
                remaining_configs.remove(best_candidate)
            else:
                # If no diverse enough candidate, add random one
                if remaining_configs:
                    diverse_configs.append(remaining_configs.pop(0))
        
        # Update population
        self.architectures = {config.architecture_id: config for config in diverse_configs}

class AdvancedNASEngine:
    """
    Advanced Neural Architecture Search Engine using evolutionary algorithms 
    and reinforcement learning for optimal neural network configuration.
    """
    
    def __init__(self, population_size: int = 50, elite_ratio: float = 0.2,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.7):
        self.population_size = population_size
        self.elite_size = int(population_size * elite_ratio)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Core components
        self.population = ArchitecturePopulation(population_size, self.elite_size)
        self.search_space = self._define_search_space()
        self.evaluation_history: Dict[str, PerformanceMetrics] = {}
        
        # Search state
        self.current_generation = 0
        self.max_generations = 100
        self.convergence_threshold = 0.001
        self.best_architecture = None
        self.best_performance = None
        
        # Performance tracking
        self.generation_stats = []
        self.search_progress = []
        self.pareto_front = []
        
        # Configuration
        self.early_stopping_patience = 10
        self.early_stopping_counter = 0
        self.target_metrics = ['accuracy', 'f1_score']
        
        # Background processing
        self.search_thread = None
        self.search_running = False
        self.initialized = False
        
        logger.info("Advanced Neural Architecture Search Engine initialized")
    
    def _define_search_space(self) -> Dict[str, Any]:
        """Define the neural architecture search space."""
        return {
            'layer_types': list(LayerType),
            'activations': list(ActivationType),
            'optimizers': list(OptimizerType),
            'units_range': (16, 1024),
            'kernel_sizes': [(1, 1), (3, 3), (5, 5), (7, 7)],
            'dropout_rates': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'learning_rates': [0.0001, 0.001, 0.01, 0.1],
            'batch_sizes': [16, 32, 64, 128],
            'max_layers': 20,
            'min_layers': 3
        }
    
    def initialize(self) -> bool:
        """Initialize the NAS engine."""
        try:
            # Generate initial population
            self._generate_initial_population()
            
            # Start background search if not already running
            if not self.search_running:
                self._start_search_thread()
            
            self.initialized = True
            logger.info("✅ Advanced NAS Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize NAS engine: {e}")
            return False
    
    def _generate_initial_population(self):
        """Generate initial population of neural architectures."""
        logger.info("Generating initial architecture population...")
        
        for i in range(self.population_size):
            config = self._generate_random_architecture()
            self.population.add_architecture(config)
        
        logger.info(f"Generated {len(self.population.architectures)} initial architectures")
    
    def _generate_random_architecture(self) -> ArchitectureConfig:
        """Generate a random neural architecture."""
        num_layers = random.randint(
            self.search_space['min_layers'],
            self.search_space['max_layers']
        )
        
        layers = []
        for i in range(num_layers):
            layer_type = random.choice(self.search_space['layer_types'])
            
            # Generate layer configuration based on type
            layer_config = self._generate_layer_config(layer_type, i == 0, i == num_layers - 1)
            layers.append(layer_config)
        
        # Add final output layer if not present
        if layers and layers[-1].layer_type != LayerType.DENSE:
            output_layer = LayerConfig(
                layer_type=LayerType.DENSE,
                units=1,  # Binary classification
                activation=ActivationType.SIGMOID
            )
            layers.append(output_layer)
        
        return ArchitectureConfig(
            layers=layers,
            optimizer=random.choice(self.search_space['optimizers']),
            learning_rate=random.choice(self.search_space['learning_rates']),
            batch_size=random.choice(self.search_space['batch_sizes'])
        )
    
    def _generate_layer_config(self, layer_type: LayerType, 
                             is_first: bool, is_last: bool) -> LayerConfig:
        """Generate configuration for a specific layer type."""
        config = LayerConfig(layer_type=layer_type)
        
        if layer_type in [LayerType.DENSE, LayerType.LSTM, LayerType.GRU]:
            min_units, max_units = self.search_space['units_range']
            config.units = random.randint(min_units, max_units)
            
        elif layer_type == LayerType.CONV2D:
            config.kernel_size = random.choice(self.search_space['kernel_sizes'])
            config.units = random.randint(32, 256)  # Number of filters
            
        elif layer_type == LayerType.DROPOUT:
            config.dropout_rate = random.choice(self.search_space['dropout_rates'])
            
        # Add activation for most layers (except dropout, batch norm)
        if layer_type not in [LayerType.DROPOUT, LayerType.BATCH_NORM]:
            config.activation = random.choice(self.search_space['activations'])
        
        return config
    
    def evaluate_architecture(self, config: ArchitectureConfig) -> PerformanceMetrics:
        """Evaluate performance of a neural architecture."""
        # Simulate architecture evaluation (in real implementation, this would train the model)
        try:
            # Simulate training time based on complexity
            complexity = self._calculate_complexity(config)
            training_time = complexity * random.uniform(0.8, 1.2)  # Add noise
            
            # Simulate performance metrics
            base_accuracy = 0.7 + (0.2 * random.random())  # 70-90% base accuracy
            
            # Penalize overly complex architectures
            complexity_penalty = min(0.1, complexity / 100)
            accuracy = max(0.5, base_accuracy - complexity_penalty)
            
            # Generate correlated metrics
            precision = accuracy + random.uniform(-0.05, 0.05)
            recall = accuracy + random.uniform(-0.05, 0.05)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics = PerformanceMetrics(
                accuracy=min(1.0, max(0.0, accuracy)),
                precision=min(1.0, max(0.0, precision)),
                recall=min(1.0, max(0.0, recall)),
                f1_score=min(1.0, max(0.0, f1_score)),
                loss=random.uniform(0.1, 1.0),
                training_time=training_time,
                inference_time=complexity * 0.1,
                memory_usage=complexity * 2,
                model_size=complexity * 5,
                flops=complexity * 1000,
                energy_consumption=complexity * 0.5
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating architecture: {e}")
            return PerformanceMetrics()  # Return default metrics
    
    def _calculate_complexity(self, config: ArchitectureConfig) -> float:
        """Calculate architecture complexity score."""
        complexity = 0.0
        
        for layer in config.layers:
            if layer.layer_type == LayerType.DENSE:
                complexity += (layer.units or 0) * 0.01
            elif layer.layer_type == LayerType.CONV2D:
                kernel_size = layer.kernel_size or (3, 3)
                complexity += (layer.units or 0) * kernel_size[0] * kernel_size[1] * 0.001
            elif layer.layer_type in [LayerType.LSTM, LayerType.GRU]:
                complexity += (layer.units or 0) * 0.02
            elif layer.layer_type == LayerType.ATTENTION:
                complexity += (layer.units or 0) * 0.03
            elif layer.layer_type == LayerType.TRANSFORMER:
                complexity += (layer.units or 0) * 0.05
        
        return complexity
    
    def evolve_generation(self):
        """Evolve population to next generation using genetic algorithms."""
        logger.info(f"Evolving generation {self.current_generation}")
        
        # Evaluate current population
        self._evaluate_population()
        
        # Selection: Get elite architectures
        elite = self.population.get_elite()
        
        # Generate new population
        new_population = ArchitecturePopulation(self.population_size, self.elite_size)
        
        # Keep elite (elitism)
        for arch in elite:
            if arch.architecture_id:
                metrics = self.evaluation_history.get(arch.architecture_id)
                new_population.add_architecture(arch, metrics)
        
        # Generate offspring through crossover and mutation
        while len(new_population.architectures) < self.population_size:
            # Select parents
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = copy.deepcopy(random.choice([parent1, parent2]))
            
            # Mutation
            if random.random() < self.mutation_rate:
                child = self._mutate(child)
            
            # Ensure unique architecture ID
            child.__post_init__()
            
            new_population.add_architecture(child)
        
        # Maintain diversity
        new_population.maintain_diversity()
        
        # Update population
        self.population = new_population
        self.current_generation += 1
        
        # Update statistics
        self._update_generation_stats()
        
        # Check for convergence
        self._check_convergence()
    
    def _evaluate_population(self):
        """Evaluate all architectures in current population."""
        for arch_id, config in self.population.architectures.items():
            if arch_id not in self.evaluation_history:
                metrics = self.evaluate_architecture(config)
                self.evaluation_history[arch_id] = metrics
                self.population.performance[arch_id] = metrics
                
                # Update best architecture
                if (self.best_performance is None or 
                    metrics.composite_score() > self.best_performance.composite_score()):
                    self.best_architecture = config
                    self.best_performance = metrics
    
    def _tournament_selection(self, tournament_size: int = 3) -> ArchitectureConfig:
        """Select architecture using tournament selection."""
        candidates = random.sample(
            list(self.population.architectures.values()),
            min(tournament_size, len(self.population.architectures))
        )
        
        best_candidate = None
        best_score = -1
        
        for candidate in candidates:
            if candidate.architecture_id:
                metrics = self.evaluation_history.get(candidate.architecture_id)
                if metrics:
                    score = metrics.composite_score()
                    if score > best_score:
                        best_score = score
                        best_candidate = candidate
        
        return best_candidate or candidates[0]
    
    def _crossover(self, parent1: ArchitectureConfig, parent2: ArchitectureConfig) -> ArchitectureConfig:
        """Create offspring through crossover of two parent architectures."""
        # Layer crossover: randomly combine layers from both parents
        layers1, layers2 = parent1.layers, parent2.layers
        min_len = min(len(layers1), len(layers2))
        max_len = max(len(layers1), len(layers2))
        
        # Random crossover point
        crossover_point = random.randint(1, min_len - 1) if min_len > 1 else 0
        
        # Combine layers
        child_layers = []
        child_layers.extend(layers1[:crossover_point])
        child_layers.extend(layers2[crossover_point:])
        
        # Randomly add remaining layers if lengths differ
        if len(child_layers) < max_len:
            remaining_layers = (layers1 if len(layers1) > len(layers2) else layers2)[len(child_layers):]
            child_layers.extend(remaining_layers[:random.randint(0, len(remaining_layers))])
        
        # Create child configuration
        child = ArchitectureConfig(
            layers=child_layers,
            optimizer=random.choice([parent1.optimizer, parent2.optimizer]),
            learning_rate=random.choice([parent1.learning_rate, parent2.learning_rate]),
            batch_size=random.choice([parent1.batch_size, parent2.batch_size]),
            epochs=random.choice([parent1.epochs, parent2.epochs])
        )
        
        return child
    
    def _mutate(self, config: ArchitectureConfig) -> ArchitectureConfig:
        """Mutate architecture configuration."""
        mutated = copy.deepcopy(config)
        
        # Mutate layers
        if mutated.layers and random.random() < 0.3:
            # Add layer
            if len(mutated.layers) < self.search_space['max_layers']:
                new_layer = self._generate_layer_config(
                    random.choice(self.search_space['layer_types']),
                    False, False
                )
                insert_pos = random.randint(0, len(mutated.layers))
                mutated.layers.insert(insert_pos, new_layer)
        
        if mutated.layers and random.random() < 0.2:
            # Remove layer
            if len(mutated.layers) > self.search_space['min_layers']:
                remove_pos = random.randint(0, len(mutated.layers) - 1)
                mutated.layers.pop(remove_pos)
        
        if mutated.layers and random.random() < 0.5:
            # Modify existing layer
            layer_idx = random.randint(0, len(mutated.layers) - 1)
            layer = mutated.layers[layer_idx]
            
            if layer.layer_type in [LayerType.DENSE, LayerType.LSTM, LayerType.GRU]:
                if layer.units:
                    min_units, max_units = self.search_space['units_range']
                    layer.units = random.randint(min_units, max_units)
            
            if layer.activation:
                layer.activation = random.choice(self.search_space['activations'])
        
        # Mutate hyperparameters
        if random.random() < 0.3:
            mutated.optimizer = random.choice(self.search_space['optimizers'])
        
        if random.random() < 0.3:
            mutated.learning_rate = random.choice(self.search_space['learning_rates'])
        
        if random.random() < 0.2:
            mutated.batch_size = random.choice(self.search_space['batch_sizes'])
        
        return mutated
    
    def _update_generation_stats(self):
        """Update statistics for current generation."""
        if not self.population.performance:
            return
        
        scores = [metrics.composite_score() for metrics in self.population.performance.values()]
        
        stats = {
            'generation': self.current_generation,
            'population_size': len(self.population.architectures),
            'best_score': max(scores) if scores else 0,
            'avg_score': statistics.mean(scores) if scores else 0,
            'std_score': statistics.stdev(scores) if len(scores) > 1 else 0,
            'diversity': self._calculate_population_diversity()
        }
        
        self.generation_stats.append(stats)
        logger.info(f"Generation {self.current_generation}: Best={stats['best_score']:.4f}, Avg={stats['avg_score']:.4f}")
    
    def _calculate_population_diversity(self) -> float:
        """Calculate diversity of current population."""
        if len(self.population.architectures) < 2:
            return 0.0
        
        configs = list(self.population.architectures.values())
        diversities = []
        
        for i in range(len(configs)):
            for j in range(i + 1, len(configs)):
                diversity = self.population.calculate_diversity(configs[i], configs[j])
                diversities.append(diversity)
        
        return statistics.mean(diversities) if diversities else 0.0
    
    def _check_convergence(self):
        """Check if search has converged."""
        if len(self.generation_stats) < 2:
            return
        
        # Check improvement in last few generations
        recent_scores = [stats['best_score'] for stats in self.generation_stats[-5:]]
        
        if len(recent_scores) >= 3:
            improvement = max(recent_scores) - min(recent_scores)
            
            if improvement < self.convergence_threshold:
                self.early_stopping_counter += 1
            else:
                self.early_stopping_counter = 0
            
            if self.early_stopping_counter >= self.early_stopping_patience:
                logger.info(f"Search converged after {self.current_generation} generations")
                self.search_running = False
    
    def _start_search_thread(self):
        """Start background search thread."""
        if self.search_thread is None or not self.search_thread.is_alive():
            self.search_running = True
            self.search_thread = threading.Thread(target=self._search_worker)
            self.search_thread.daemon = True
            self.search_thread.start()
    
    def _search_worker(self):
        """Background worker for continuous architecture search."""
        while (self.search_running and 
               self.current_generation < self.max_generations):
            try:
                self.evolve_generation()
                time.sleep(1)  # Brief pause between generations
                
            except Exception as e:
                logger.error(f"Error in NAS evolution: {e}")
                time.sleep(5)  # Wait before retrying
    
    def get_nas_insights(self) -> Dict[str, Any]:
        """Get comprehensive insights about neural architecture search."""
        if not self.initialized:
            return {'error': 'NAS engine not initialized'}
        
        # Best architecture details
        best_arch_info = None
        if self.best_architecture:
            best_arch_info = {
                'architecture_id': self.best_architecture.architecture_id,
                'layer_count': len(self.best_architecture.layers),
                'layer_types': [layer.layer_type.value for layer in self.best_architecture.layers],
                'optimizer': self.best_architecture.optimizer.value,
                'learning_rate': self.best_architecture.learning_rate,
                'batch_size': self.best_architecture.batch_size,
                'performance': asdict(self.best_performance) if self.best_performance else None
            }
        
        # Search progress
        search_stats = {
            'current_generation': self.current_generation,
            'max_generations': self.max_generations,
            'population_size': len(self.population.architectures),
            'evaluated_architectures': len(self.evaluation_history),
            'search_running': self.search_running,
            'convergence_counter': self.early_stopping_counter
        }
        
        # Performance trends
        recent_stats = self.generation_stats[-10:] if len(self.generation_stats) >= 10 else self.generation_stats
        
        return {
            'search_statistics': search_stats,
            'best_architecture': best_arch_info,
            'generation_history': recent_stats,
            'population_diversity': self._calculate_population_diversity(),
            'search_space_coverage': len(self.evaluation_history) / (self.population_size * max(1, self.current_generation)),
            'performance_improvement': (
                self.generation_stats[-1]['best_score'] - self.generation_stats[0]['best_score']
                if len(self.generation_stats) >= 2 else 0.0
            )
        }
    
    def get_architecture_recommendations(self, task_type: str = "classification") -> List[Dict[str, Any]]:
        """Get top architecture recommendations for specific task."""
        if not self.evaluation_history:
            return []
        
        # Sort architectures by performance
        sorted_archs = sorted(
            self.evaluation_history.items(),
            key=lambda x: x[1].composite_score(),
            reverse=True
        )
        
        recommendations = []
        for arch_id, metrics in sorted_archs[:5]:  # Top 5
            config = self.population.architectures.get(arch_id)
            if config:
                recommendations.append({
                    'architecture_id': arch_id,
                    'performance_score': metrics.composite_score(),
                    'accuracy': metrics.accuracy,
                    'f1_score': metrics.f1_score,
                    'training_time': metrics.training_time,
                    'model_size': metrics.model_size,
                    'layer_count': len(config.layers),
                    'complexity': self._calculate_complexity(config),
                    'configuration': config.to_dict()
                })
        
        return recommendations
    
    def cleanup(self):
        """Clean up resources and stop background threads."""
        self.search_running = False
        if self.search_thread and self.search_thread.is_alive():
            self.search_thread.join(timeout=5)
        
        logger.info("Advanced NAS Engine cleaned up")