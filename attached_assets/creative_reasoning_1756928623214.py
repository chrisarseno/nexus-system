"""
Creative Reasoning Engine
Provides advanced problem-solving, innovative thinking, and creative synthesis capabilities.
"""

import logging
import json
import time
import random
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import itertools
import math

logger = logging.getLogger(__name__)

class ReasoningType(Enum):
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    LATERAL = "lateral"
    SYSTEMATIC = "systematic"
    INTUITIVE = "intuitive"
    DIVERGENT = "divergent"
    CONVERGENT = "convergent"

class CreativeMethod(Enum):
    BRAINSTORMING = "brainstorming"
    ANALOGICAL = "analogical"
    COMBINATORIAL = "combinatorial"
    CONSTRAINT_REMOVAL = "constraint_removal"
    PERSPECTIVE_SHIFT = "perspective_shift"
    REVERSE_THINKING = "reverse_thinking"
    PATTERN_BREAKING = "pattern_breaking"

@dataclass
class CreativeIdea:
    """Represents a creative idea or solution."""
    idea_id: str
    description: str
    reasoning_type: ReasoningType
    method_used: CreativeMethod
    originality_score: float
    feasibility_score: float
    impact_score: float
    confidence: float
    supporting_evidence: List[str]
    related_concepts: List[str]
    generated_at: datetime
    refined_versions: List[str] = None

@dataclass
class ProblemContext:
    """Context for a problem requiring creative solution."""
    problem_id: str
    description: str
    domain: str
    constraints: List[str]
    objectives: List[str]
    available_resources: List[str]
    success_criteria: List[str]
    deadline: Optional[datetime] = None
    priority: str = "medium"

@dataclass
class CreativeSynthesis:
    """Results from cross-domain knowledge synthesis."""
    synthesis_id: str
    source_domains: List[str]
    combined_concepts: List[str]
    novel_insights: List[str]
    potential_applications: List[str]
    confidence_score: float
    created_at: datetime

class CreativeReasoningEngine:
    """
    Advanced creative reasoning system that provides innovative problem-solving,
    lateral thinking, and cross-domain synthesis capabilities.
    """
    
    def __init__(self):
        self.creative_ideas = {}
        self.problem_solutions = {}
        self.synthesis_results = {}
        self.reasoning_patterns = defaultdict(list)
        
        # Creative thinking frameworks
        self.thinking_frameworks = self._initialize_frameworks()
        self.analogical_patterns = self._load_analogical_patterns()
        self.creative_catalysts = self._load_creative_catalysts()
        
        # Innovation metrics
        self.innovation_stats = defaultdict(int)
        self.creativity_metrics = defaultdict(float)
        self.solution_history = deque(maxlen=1000)
        
        # Learning and adaptation
        self.successful_patterns = defaultdict(list)
        self.creative_preferences = {}
        self.domain_expertise = defaultdict(float)
        
        # Threading for background creativity
        self.background_creativity = True
        self.creativity_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        self.initialized = False
    
    def initialize(self):
        """Initialize the creative reasoning engine."""
        if self.initialized:
            return
            
        logger.info("Initializing Creative Reasoning Engine...")
        
        # Load creative knowledge bases
        self._load_creative_knowledge()
        
        # Initialize reasoning models
        self._initialize_reasoning_models()
        
        # Start background creativity processes
        if self.background_creativity:
            self._start_background_creativity()
        
        self.initialized = True
        logger.info("Creative Reasoning Engine initialized")
    
    def solve_creative_problem(self, problem: ProblemContext, 
                             reasoning_types: List[ReasoningType] = None) -> List[CreativeIdea]:
        """Generate creative solutions for a given problem."""
        try:
            reasoning_types = reasoning_types or list(ReasoningType)
            solutions = []
            
            logger.info(f"Generating creative solutions for problem: {problem.problem_id}")
            
            # Apply multiple reasoning approaches
            for reasoning_type in reasoning_types:
                type_solutions = self._apply_reasoning_type(problem, reasoning_type)
                solutions.extend(type_solutions)
            
            # Cross-pollinate solutions
            hybrid_solutions = self._create_hybrid_solutions(solutions, problem)
            solutions.extend(hybrid_solutions)
            
            # Evaluate and rank solutions
            ranked_solutions = self._evaluate_solutions(solutions, problem)
            
            # Store successful patterns
            self._learn_from_solutions(ranked_solutions, problem)
            
            # Update innovation metrics
            self.innovation_stats['problems_solved'] += 1
            self.innovation_stats['solutions_generated'] += len(ranked_solutions)
            
            return ranked_solutions[:10]  # Return top 10 solutions
            
        except Exception as e:
            logger.error(f"Error solving creative problem: {e}")
            return []
    
    def generate_analogical_insights(self, source_domain: str, target_domain: str,
                                   concept: str) -> List[Dict[str, Any]]:
        """Generate insights by drawing analogies between domains."""
        try:
            insights = []
            
            # Find analogical patterns
            source_patterns = self._extract_domain_patterns(source_domain, concept)
            target_context = self._get_domain_context(target_domain)
            
            for pattern in source_patterns:
                # Map pattern to target domain
                mapped_insight = self._map_analogical_pattern(pattern, target_context)
                
                if mapped_insight and mapped_insight['relevance_score'] > 0.6:
                    insights.append({
                        'insight_id': f"analogy_{int(time.time())}_{random.randint(1000, 9999)}",
                        'source_pattern': pattern,
                        'target_application': mapped_insight,
                        'analogy_strength': mapped_insight['relevance_score'],
                        'potential_value': self._assess_insight_value(mapped_insight, target_domain),
                        'implementation_steps': self._generate_implementation_steps(mapped_insight),
                        'created_at': datetime.now().isoformat()
                    })
            
            # Sort by potential value
            insights.sort(key=lambda x: x['potential_value'], reverse=True)
            
            self.innovation_stats['analogical_insights'] += len(insights)
            return insights[:5]  # Return top 5 insights
            
        except Exception as e:
            logger.error(f"Error generating analogical insights: {e}")
            return []
    
    def perform_creative_synthesis(self, domains: List[str], 
                                 focus_concept: str = None) -> CreativeSynthesis:
        """Synthesize knowledge across multiple domains to create novel insights."""
        try:
            # Extract key concepts from each domain
            domain_concepts = {}
            for domain in domains:
                domain_concepts[domain] = self._extract_domain_concepts(domain, focus_concept)
            
            # Find intersection patterns
            intersection_patterns = self._find_intersection_patterns(domain_concepts)
            
            # Generate novel combinations
            novel_combinations = self._generate_novel_combinations(domain_concepts)
            
            # Create synthesis insights
            synthesis_insights = self._create_synthesis_insights(
                intersection_patterns, novel_combinations, domains
            )
            
            # Evaluate potential applications
            potential_applications = self._evaluate_synthesis_applications(
                synthesis_insights, domains
            )
            
            synthesis = CreativeSynthesis(
                synthesis_id=f"synthesis_{int(time.time())}",
                source_domains=domains,
                combined_concepts=list(novel_combinations),
                novel_insights=synthesis_insights,
                potential_applications=potential_applications,
                confidence_score=self._calculate_synthesis_confidence(synthesis_insights),
                created_at=datetime.now()
            )
            
            self.synthesis_results[synthesis.synthesis_id] = synthesis
            self.innovation_stats['cross_domain_syntheses'] += 1
            
            logger.info(f"Created creative synthesis across domains: {', '.join(domains)}")
            return synthesis
            
        except Exception as e:
            logger.error(f"Error performing creative synthesis: {e}")
            return None
    
    def generate_innovative_approaches(self, challenge: str, 
                                     current_approaches: List[str] = None) -> List[Dict[str, Any]]:
        """Generate innovative approaches to overcome challenges or improve processes."""
        try:
            current_approaches = current_approaches or []
            innovative_approaches = []
            
            # Apply creative methods
            for method in CreativeMethod:
                approach = self._apply_creative_method(method, challenge, current_approaches)
                if approach:
                    innovative_approaches.append(approach)
            
            # Generate constraint-free thinking
            unconstrained_approaches = self._generate_unconstrained_thinking(challenge)
            innovative_approaches.extend(unconstrained_approaches)
            
            # Apply perspective shifting
            perspective_approaches = self._apply_perspective_shifting(challenge)
            innovative_approaches.extend(perspective_approaches)
            
            # Evaluate innovation potential
            for approach in innovative_approaches:
                approach['innovation_score'] = self._calculate_innovation_score(approach, current_approaches)
                approach['implementation_difficulty'] = self._assess_implementation_difficulty(approach)
                approach['potential_impact'] = self._assess_potential_impact(approach, challenge)
            
            # Sort by innovation potential
            innovative_approaches.sort(
                key=lambda x: x['innovation_score'] * x['potential_impact'] / x['implementation_difficulty'],
                reverse=True
            )
            
            self.innovation_stats['innovative_approaches'] += len(innovative_approaches)
            return innovative_approaches[:8]  # Return top 8 approaches
            
        except Exception as e:
            logger.error(f"Error generating innovative approaches: {e}")
            return []
    
    def explore_creative_possibilities(self, starting_concept: str, 
                                     exploration_depth: int = 3) -> Dict[str, Any]:
        """Explore creative possibilities branching from a starting concept."""
        try:
            exploration_tree = {
                'root_concept': starting_concept,
                'branches': [],
                'depth': exploration_depth,
                'total_possibilities': 0,
                'novel_connections': [],
                'created_at': datetime.now().isoformat()
            }
            
            # Generate exploration branches
            current_concepts = [starting_concept]
            
            for depth in range(exploration_depth):
                next_level_concepts = []
                
                for concept in current_concepts:
                    # Generate creative extensions
                    extensions = self._generate_creative_extensions(concept, depth)
                    
                    # Add to exploration tree
                    for extension in extensions:
                        branch = {
                            'concept': extension['concept'],
                            'connection_type': extension['connection_type'],
                            'creativity_score': extension['creativity_score'],
                            'depth': depth + 1,
                            'parent': concept,
                            'potential_applications': extension.get('applications', [])
                        }
                        exploration_tree['branches'].append(branch)
                        next_level_concepts.append(extension['concept'])
                
                current_concepts = next_level_concepts[:20]  # Limit branching
            
            # Find novel connections
            exploration_tree['novel_connections'] = self._find_novel_connections(
                exploration_tree['branches']
            )
            
            exploration_tree['total_possibilities'] = len(exploration_tree['branches'])
            
            # Generate summary insights
            exploration_tree['summary_insights'] = self._generate_exploration_insights(
                exploration_tree
            )
            
            self.innovation_stats['creative_explorations'] += 1
            return exploration_tree
            
        except Exception as e:
            logger.error(f"Error exploring creative possibilities: {e}")
            return {}
    
    def get_creativity_analytics(self) -> Dict[str, Any]:
        """Get comprehensive creativity and innovation analytics."""
        try:
            with self.lock:
                # Calculate creativity metrics
                total_ideas = len(self.creative_ideas)
                total_solutions = len(self.problem_solutions)
                total_syntheses = len(self.synthesis_results)
                
                # Innovation rates
                innovation_rate = self.innovation_stats.get('solutions_generated', 0) / max(
                    self.innovation_stats.get('problems_solved', 1), 1
                )
                
                # Success patterns
                successful_patterns = {
                    method.value: len(self.successful_patterns[method.value])
                    for method in CreativeMethod
                }
                
                # Recent activity
                recent_activity = self._calculate_recent_activity()
                
                # Creativity trends
                creativity_trends = self._analyze_creativity_trends()
                
                return {
                    'total_metrics': {
                        'creative_ideas': total_ideas,
                        'problem_solutions': total_solutions,
                        'cross_domain_syntheses': total_syntheses,
                        'innovation_rate': round(innovation_rate, 2)
                    },
                    'method_effectiveness': successful_patterns,
                    'recent_activity': recent_activity,
                    'creativity_trends': creativity_trends,
                    'domain_expertise': dict(self.domain_expertise),
                    'innovation_stats': dict(self.innovation_stats),
                    'system_health': {
                        'background_creativity_active': self.running,
                        'reasoning_frameworks_loaded': len(self.thinking_frameworks),
                        'analogical_patterns_available': len(self.analogical_patterns)
                    }
                }
                
        except Exception as e:
            logger.error(f"Error generating creativity analytics: {e}")
            return {}
    
    def _apply_reasoning_type(self, problem: ProblemContext, 
                            reasoning_type: ReasoningType) -> List[CreativeIdea]:
        """Apply a specific reasoning type to generate solutions."""
        solutions = []
        
        if reasoning_type == ReasoningType.ANALYTICAL:
            solutions = self._analytical_reasoning(problem)
        elif reasoning_type == ReasoningType.CREATIVE:
            solutions = self._creative_reasoning(problem)
        elif reasoning_type == ReasoningType.LATERAL:
            solutions = self._lateral_thinking(problem)
        elif reasoning_type == ReasoningType.SYSTEMATIC:
            solutions = self._systematic_reasoning(problem)
        elif reasoning_type == ReasoningType.INTUITIVE:
            solutions = self._intuitive_reasoning(problem)
        elif reasoning_type == ReasoningType.DIVERGENT:
            solutions = self._divergent_thinking(problem)
        elif reasoning_type == ReasoningType.CONVERGENT:
            solutions = self._convergent_thinking(problem)
        
        return solutions
    
    def _analytical_reasoning(self, problem: ProblemContext) -> List[CreativeIdea]:
        """Apply analytical reasoning to break down and solve the problem."""
        solutions = []
        
        # Decompose problem into components
        components = self._decompose_problem(problem)
        
        # Analyze each component
        for component in components:
            component_solutions = self._solve_component_analytically(component, problem)
            solutions.extend(component_solutions)
        
        return solutions
    
    def _creative_reasoning(self, problem: ProblemContext) -> List[CreativeIdea]:
        """Apply creative reasoning using various creative methods."""
        solutions = []
        
        for method in CreativeMethod:
            method_solutions = self._apply_creative_method_to_problem(method, problem)
            solutions.extend(method_solutions)
        
        return solutions
    
    def _lateral_thinking(self, problem: ProblemContext) -> List[CreativeIdea]:
        """Apply lateral thinking to find unexpected solutions."""
        solutions = []
        
        # Random word association
        random_concepts = self._get_random_concepts(5)
        for concept in random_concepts:
            lateral_solution = self._connect_concept_to_problem(concept, problem)
            if lateral_solution:
                solutions.append(lateral_solution)
        
        # Assumption reversal
        assumptions = self._identify_assumptions(problem)
        for assumption in assumptions:
            reversed_solution = self._reverse_assumption_solution(assumption, problem)
            if reversed_solution:
                solutions.append(reversed_solution)
        
        return solutions
    
    def _initialize_frameworks(self) -> Dict[str, Any]:
        """Initialize creative thinking frameworks."""
        return {
            'scamper': {
                'substitute': 'What can be substituted?',
                'combine': 'What can be combined?',
                'adapt': 'What can be adapted?',
                'modify': 'What can be modified?',
                'put_to_other_uses': 'What other uses are there?',
                'eliminate': 'What can be eliminated?',
                'reverse': 'What can be reversed or rearranged?'
            },
            'six_thinking_hats': {
                'white': 'Facts and information',
                'red': 'Emotions and feelings',
                'black': 'Critical thinking',
                'yellow': 'Positive thinking',
                'green': 'Creative thinking',
                'blue': 'Process thinking'
            },
            'design_thinking': {
                'empathize': 'Understanding user needs',
                'define': 'Defining the problem',
                'ideate': 'Generating ideas',
                'prototype': 'Building solutions',
                'test': 'Testing and refining'
            }
        }
    
    def _start_background_creativity(self):
        """Start background creativity processes."""
        if not self.creativity_thread:
            self.running = True
            self.creativity_thread = threading.Thread(target=self._creativity_loop)
            self.creativity_thread.daemon = True
            self.creativity_thread.start()
            logger.info("Background creativity processes started")
    
    def _creativity_loop(self):
        """Main creativity loop for background processing."""
        while self.running:
            try:
                time.sleep(300)  # Run every 5 minutes
                
                # Generate random creative connections
                self._generate_background_connections()
                
                # Explore emerging patterns
                self._explore_emerging_patterns()
                
                # Update creativity metrics
                self._update_creativity_metrics()
                
            except Exception as e:
                logger.error(f"Error in creativity loop: {e}")
    
    def shutdown(self):
        """Shutdown the creative reasoning engine."""
        self.running = False
        if self.creativity_thread:
            self.creativity_thread.join(timeout=5)
        logger.info("Creative Reasoning Engine shutdown completed")