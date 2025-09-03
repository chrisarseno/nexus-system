"""
Social Cognition System for AGI Theory of Mind
Builds theory of mind and multi-agent interaction capabilities
"""

import logging
import time
import threading
import numpy as np
import json
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import copy

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Types of agents in social cognition."""
    HUMAN = "human"
    AI_AGENT = "ai_agent"
    GROUP = "group"
    ORGANIZATION = "organization"
    VIRTUAL_ENTITY = "virtual_entity"
    UNKNOWN = "unknown"

class MentalState(Enum):
    """Types of mental states to model."""
    BELIEF = "belief"
    DESIRE = "desire"
    INTENTION = "intention"
    EMOTION = "emotion"
    KNOWLEDGE = "knowledge"
    EXPECTATION = "expectation"
    GOAL = "goal"
    PREFERENCE = "preference"

class InteractionType(Enum):
    """Types of social interactions."""
    COOPERATION = "cooperation"
    COMPETITION = "competition"
    COMMUNICATION = "communication"
    NEGOTIATION = "negotiation"
    TEACHING = "teaching"
    LEARNING = "learning"
    COLLABORATION = "collaboration"
    CONFLICT = "conflict"

@dataclass
class Agent:
    """Represents another agent in the social environment."""
    agent_id: str
    agent_type: AgentType
    name: str
    capabilities: List[str]
    observed_behaviors: List[Dict[str, Any]]
    mental_model: Dict[MentalState, Any]
    interaction_history: List[str]  # Interaction IDs
    trust_level: float
    predictability: float
    last_interaction: Optional[datetime]
    
    def calculate_understanding_level(self) -> float:
        """Calculate how well we understand this agent."""
        behavior_factor = len(self.observed_behaviors) / 20.0
        interaction_factor = len(self.interaction_history) / 10.0
        mental_model_factor = len(self.mental_model) / len(MentalState)
        predictability_factor = self.predictability
        
        return min(1.0, 0.3 * behavior_factor + 0.25 * interaction_factor + 
                   0.25 * mental_model_factor + 0.2 * predictability_factor)
    
    def update_mental_model(self, state_type: MentalState, evidence: Dict[str, Any]):
        """Update mental model based on new evidence."""
        if state_type not in self.mental_model:
            self.mental_model[state_type] = []
        
        # Add new evidence with timestamp
        evidence_entry = {
            'evidence': evidence,
            'timestamp': datetime.now(),
            'confidence': evidence.get('confidence', 0.5)
        }
        
        self.mental_model[state_type].append(evidence_entry)
        
        # Keep only recent evidence (last 10 entries)
        if len(self.mental_model[state_type]) > 10:
            self.mental_model[state_type] = self.mental_model[state_type][-10:]

@dataclass
class SocialInteraction:
    """Represents a social interaction."""
    interaction_id: str
    participants: List[str]  # Agent IDs
    interaction_type: InteractionType
    context: Dict[str, Any]
    communication_content: List[Dict[str, Any]]
    outcomes: Dict[str, Any]
    mental_state_changes: Dict[str, Dict[MentalState, Any]]
    social_dynamics: Dict[str, float]
    timestamp: datetime
    
    def extract_social_learning(self) -> Dict[str, Any]:
        """Extract social learning from this interaction."""
        return {
            'cooperation_patterns': self._analyze_cooperation(),
            'communication_styles': self._analyze_communication(),
            'trust_dynamics': self._analyze_trust_changes(),
            'learning_opportunities': self._identify_learning()
        }
    
    def _analyze_cooperation(self) -> Dict[str, Any]:
        """Analyze cooperation patterns in interaction."""
        cooperation_indicators = []
        for content in self.communication_content:
            if any(keyword in content.get('text', '').lower() 
                  for keyword in ['help', 'together', 'collaborate', 'share']):
                cooperation_indicators.append(content)
        
        return {
            'cooperation_level': len(cooperation_indicators) / max(1, len(self.communication_content)),
            'cooperative_behaviors': cooperation_indicators
        }
    
    def _analyze_communication(self) -> Dict[str, Any]:
        """Analyze communication patterns."""
        communication_styles = {}
        for participant in self.participants:
            participant_messages = [
                content for content in self.communication_content 
                if content.get('sender') == participant
            ]
            
            communication_styles[participant] = {
                'message_count': len(participant_messages),
                'avg_length': sum(len(msg.get('text', '')) for msg in participant_messages) / max(1, len(participant_messages)),
                'sentiment': self._analyze_sentiment(participant_messages)
            }
        
        return communication_styles
    
    def _analyze_trust_changes(self) -> Dict[str, float]:
        """Analyze trust level changes during interaction."""
        trust_changes = {}
        for participant in self.participants:
            # Simplified trust analysis based on outcomes
            if self.outcomes.get('success', False):
                trust_changes[participant] = 0.1  # Increase trust
            else:
                trust_changes[participant] = -0.05  # Decrease trust
        
        return trust_changes
    
    def _identify_learning(self) -> List[str]:
        """Identify learning opportunities from interaction."""
        learning_opportunities = []
        
        # Check for knowledge sharing
        if any('explain' in content.get('text', '').lower() 
               for content in self.communication_content):
            learning_opportunities.append('knowledge_transfer')
        
        # Check for skill demonstration
        if self.outcomes.get('skill_demonstrated', False):
            learning_opportunities.append('skill_learning')
        
        # Check for problem solving
        if self.interaction_type == InteractionType.COLLABORATION:
            learning_opportunities.append('collaborative_problem_solving')
        
        return learning_opportunities
    
    def _analyze_sentiment(self, messages: List[Dict[str, Any]]) -> float:
        """Analyze sentiment of messages (simplified)."""
        positive_words = ['good', 'great', 'excellent', 'helpful', 'thanks']
        negative_words = ['bad', 'wrong', 'problem', 'error', 'difficult']
        
        sentiment_score = 0.0
        total_words = 0
        
        for message in messages:
            text = message.get('text', '').lower()
            words = text.split()
            total_words += len(words)
            
            for word in words:
                if word in positive_words:
                    sentiment_score += 1.0
                elif word in negative_words:
                    sentiment_score -= 1.0
        
        return sentiment_score / max(1, total_words)

@dataclass
class TheoryOfMindModel:
    """Represents theory of mind model for an agent."""
    model_id: str
    target_agent_id: str
    beliefs_about_beliefs: Dict[str, Any]
    beliefs_about_desires: Dict[str, Any]
    beliefs_about_intentions: Dict[str, Any]
    perspective_taking_ability: float
    false_belief_understanding: bool
    recursive_modeling_depth: int
    confidence: float
    
    def predict_behavior(self, situation: Dict[str, Any]) -> Dict[str, Any]:
        """Predict agent behavior based on theory of mind."""
        # Simplified prediction based on modeled mental states
        predicted_actions = []
        
        # Based on beliefs
        if self.beliefs_about_beliefs:
            predicted_actions.append({
                'type': 'belief_driven_action',
                'action': 'act_according_to_beliefs',
                'confidence': 0.7
            })
        
        # Based on desires
        if self.beliefs_about_desires:
            predicted_actions.append({
                'type': 'desire_driven_action',
                'action': 'pursue_desires',
                'confidence': 0.6
            })
        
        # Based on intentions
        if self.beliefs_about_intentions:
            predicted_actions.append({
                'type': 'intention_driven_action',
                'action': 'execute_intentions',
                'confidence': 0.8
            })
        
        return {
            'predicted_actions': predicted_actions,
            'overall_confidence': self.confidence,
            'prediction_context': situation
        }

class SocialCognitionSystem:
    """
    System for social cognition including theory of mind, multi-agent interaction,
    and social learning for AGI social intelligence.
    """
    
    def __init__(self):
        # Core social components
        self.agents = {}  # agent_id -> Agent
        self.social_interactions = deque(maxlen=1000)
        self.theory_of_mind_models = {}  # target_agent_id -> TheoryOfMindModel
        
        # Social processing engines
        self.mental_state_inference = MentalStateInferenceEngine()
        self.behavior_prediction = BehaviorPredictionEngine()
        self.social_learning = SocialLearningEngine()
        self.perspective_taking = PerspectiveTakingEngine()
        
        # Interaction management
        self.communication_processor = CommunicationProcessor()
        self.cooperation_analyzer = CooperationAnalyzer()
        self.conflict_resolver = ConflictResolver()
        self.social_norm_learner = SocialNormLearner()
        
        # Theory of mind development
        self.false_belief_reasoner = FalseBeliefReasoner()
        self.recursive_modeling = RecursiveModelingEngine()
        self.empathy_engine = EmpathyEngine()
        
        # Current social state
        self.current_social_context = {
            'active_interactions': [],
            'social_focus': None,
            'current_perspective': 'self',
            'social_mood': 'neutral',
            'cooperation_level': 0.5
        }
        
        # Processing parameters
        self.mental_model_update_threshold = 0.6
        self.trust_update_rate = 0.1
        self.social_learning_rate = 0.05
        self.max_recursive_depth = 3
        
        # Background processing
        self.social_processing_enabled = True
        self.social_learning_thread = None
        self.theory_of_mind_thread = None
        
        # Performance metrics
        self.social_metrics = {
            'agents_modeled': 0,
            'interactions_processed': 0,
            'theory_of_mind_accuracy': 0.0,
            'social_learning_rate': 0.0,
            'cooperation_success': 0.0,
            'empathy_score': 0.0,
            'social_adaptation': 0.0
        }
        
        self.initialized = False
        logger.info("Social Cognition System initialized")
    
    def initialize(self) -> bool:
        """Initialize the social cognition system."""
        try:
            # Initialize social processing
            self.mental_state_inference.initialize()
            self.behavior_prediction.initialize()
            self.social_learning.initialize()
            self.perspective_taking.initialize()
            
            # Initialize interaction management
            self.communication_processor.initialize()
            self.cooperation_analyzer.initialize()
            self.conflict_resolver.initialize()
            self.social_norm_learner.initialize()
            
            # Initialize theory of mind
            self.false_belief_reasoner.initialize()
            self.recursive_modeling.initialize()
            self.empathy_engine.initialize()
            
            # Create self-model as baseline
            self._create_self_model()
            
            # Start social processing
            self._start_social_processing_threads()
            
            self.initialized = True
            logger.info("✅ Social Cognition System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize social cognition system: {e}")
            return False
    
    def register_agent(self, agent_id: str, agent_type: AgentType,
                      name: str, capabilities: List[str] = None) -> bool:
        """Register a new agent for social cognition."""
        try:
            if agent_id in self.agents:
                return False  # Agent already exists
            
            agent = Agent(
                agent_id=agent_id,
                agent_type=agent_type,
                name=name,
                capabilities=capabilities or [],
                observed_behaviors=[],
                mental_model={},
                interaction_history=[],
                trust_level=0.5,  # Neutral initial trust
                predictability=0.3,  # Low initial predictability
                last_interaction=None
            )
            
            self.agents[agent_id] = agent
            self.social_metrics['agents_modeled'] += 1
            
            # Create initial theory of mind model
            self._create_theory_of_mind_model(agent_id)
            
            logger.debug(f"Registered agent: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering agent: {e}")
            return False
    
    def process_social_interaction(self, participants: List[str],
                                 interaction_type: InteractionType,
                                 context: Dict[str, Any],
                                 communication_content: List[Dict[str, Any]]) -> str:
        """Process a social interaction for learning and modeling."""
        try:
            interaction_id = f"social_interaction_{int(time.time() * 1000)}"
            
            # Ensure all participants are registered
            for participant in participants:
                if participant not in self.agents and participant != 'self':
                    self.register_agent(participant, AgentType.UNKNOWN, f"Agent_{participant}")
            
            # Analyze interaction outcomes
            outcomes = self._analyze_interaction_outcomes(
                participants, interaction_type, context, communication_content
            )
            
            # Detect mental state changes
            mental_state_changes = self._detect_mental_state_changes(
                participants, communication_content, outcomes
            )
            
            # Analyze social dynamics
            social_dynamics = self._analyze_social_dynamics(
                participants, interaction_type, communication_content
            )
            
            # Create interaction record
            interaction = SocialInteraction(
                interaction_id=interaction_id,
                participants=participants,
                interaction_type=interaction_type,
                context=context,
                communication_content=communication_content,
                outcomes=outcomes,
                mental_state_changes=mental_state_changes,
                social_dynamics=social_dynamics,
                timestamp=datetime.now()
            )
            
            self.social_interactions.append(interaction)
            self.social_metrics['interactions_processed'] += 1
            
            # Update agent models
            self._update_agent_models_from_interaction(interaction)
            
            # Update theory of mind models
            self._update_theory_of_mind_models(interaction)
            
            # Extract social learning
            social_learning = interaction.extract_social_learning()
            self.social_learning.process_learning(social_learning)
            
            # Update current social context
            self.current_social_context['active_interactions'].append(interaction_id)
            if len(self.current_social_context['active_interactions']) > 5:
                self.current_social_context['active_interactions'] = self.current_social_context['active_interactions'][-5:]
            
            logger.debug(f"Processed social interaction: {interaction_id}")
            return interaction_id
            
        except Exception as e:
            logger.error(f"Error processing social interaction: {e}")
            return ""
    
    def predict_agent_behavior(self, agent_id: str, situation: Dict[str, Any]) -> Dict[str, Any]:
        """Predict agent behavior using theory of mind."""
        try:
            if agent_id not in self.agents:
                return {'error': 'Agent not found'}
            
            if agent_id not in self.theory_of_mind_models:
                return {'error': 'Theory of mind model not available'}
            
            agent = self.agents[agent_id]
            tom_model = self.theory_of_mind_models[agent_id]
            
            # Use theory of mind model to predict behavior
            prediction = tom_model.predict_behavior(situation)
            
            # Enhance prediction with behavior analysis
            behavior_analysis = self.behavior_prediction.analyze_behavioral_patterns(
                agent.observed_behaviors, situation
            )
            
            # Combine predictions
            combined_prediction = {
                'tom_prediction': prediction,
                'behavior_analysis': behavior_analysis,
                'agent_understanding_level': agent.calculate_understanding_level(),
                'prediction_confidence': min(prediction.get('overall_confidence', 0.5),
                                           behavior_analysis.get('confidence', 0.5)),
                'recommended_approach': self._recommend_interaction_approach(agent, situation)
            }
            
            return combined_prediction
            
        except Exception as e:
            logger.error(f"Error predicting agent behavior: {e}")
            return {'error': str(e)}
    
    def take_perspective(self, agent_id: str, situation: Dict[str, Any]) -> Dict[str, Any]:
        """Take the perspective of another agent."""
        try:
            if agent_id not in self.agents:
                return {'error': 'Agent not found'}
            
            agent = self.agents[agent_id]
            
            # Use perspective-taking engine
            perspective_result = self.perspective_taking.take_perspective(
                agent, situation, self.theory_of_mind_models.get(agent_id)
            )
            
            # Enhance with empathy
            empathy_result = self.empathy_engine.generate_empathetic_understanding(
                agent, situation
            )
            
            return {
                'perspective_taken': perspective_result,
                'empathetic_understanding': empathy_result,
                'agent_viewpoint': {
                    'likely_thoughts': perspective_result.get('thoughts', []),
                    'likely_feelings': perspective_result.get('feelings', []),
                    'likely_goals': perspective_result.get('goals', []),
                    'likely_concerns': perspective_result.get('concerns', [])
                },
                'perspective_confidence': perspective_result.get('confidence', 0.5)
            }
            
        except Exception as e:
            logger.error(f"Error taking perspective: {e}")
            return {'error': str(e)}
    
    def get_social_cognition_state(self) -> Dict[str, Any]:
        """Get comprehensive state of social cognition system."""
        if not self.initialized:
            return {'error': 'Social cognition system not initialized'}
        
        # Update metrics
        self._update_social_metrics()
        
        # Get agents summary
        agents_summary = {
            agent_id: {
                'name': agent.name,
                'type': agent.agent_type.value,
                'understanding_level': agent.calculate_understanding_level(),
                'trust_level': agent.trust_level,
                'predictability': agent.predictability,
                'interactions_count': len(agent.interaction_history),
                'mental_states_modeled': len(agent.mental_model)
            }
            for agent_id, agent in self.agents.items()
        }
        
        # Get recent interactions
        recent_interactions = [
            {
                'interaction_id': interaction.interaction_id,
                'type': interaction.interaction_type.value,
                'participants': interaction.participants,
                'outcomes_positive': interaction.outcomes.get('success', False),
                'social_learning_extracted': bool(interaction.extract_social_learning()),
                'time_ago': (datetime.now() - interaction.timestamp).total_seconds()
            }
            for interaction in list(self.social_interactions)[-10:]
        ]
        
        # Get theory of mind summary
        tom_summary = {
            agent_id: {
                'confidence': model.confidence,
                'perspective_taking_ability': model.perspective_taking_ability,
                'recursive_depth': model.recursive_modeling_depth,
                'false_belief_understanding': model.false_belief_understanding
            }
            for agent_id, model in self.theory_of_mind_models.items()
        }
        
        return {
            'current_social_context': self.current_social_context,
            'agents': agents_summary,
            'recent_interactions': recent_interactions,
            'theory_of_mind_models': tom_summary,
            'social_capabilities': {
                'agents_understood': len([a for a in self.agents.values() if a.calculate_understanding_level() > 0.6]),
                'cooperative_relationships': len([a for a in self.agents.values() if a.trust_level > 0.7]),
                'empathy_active': hasattr(self.empathy_engine, 'is_active') and self.empathy_engine.is_active(),
                'perspective_taking_depth': self.max_recursive_depth
            },
            'social_learning': {
                'norms_learned': self.social_norm_learner.get_norms_count() if hasattr(self.social_norm_learner, 'get_norms_count') else 0,
                'cooperation_patterns': self.cooperation_analyzer.get_patterns_count() if hasattr(self.cooperation_analyzer, 'get_patterns_count') else 0,
                'communication_styles': len(set(interaction.interaction_type for interaction in self.social_interactions))
            },
            'social_metrics': self.social_metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def _create_self_model(self):
        """Create a model of self as the baseline agent."""
        self_agent = Agent(
            agent_id='self',
            agent_type=AgentType.AI_AGENT,
            name='Self',
            capabilities=['reasoning', 'learning', 'communication', 'empathy'],
            observed_behaviors=[],
            mental_model={
                MentalState.BELIEF: [{'content': 'I am an AI system', 'confidence': 1.0}],
                MentalState.DESIRE: [{'content': 'Help and understand others', 'confidence': 0.9}],
                MentalState.INTENTION: [{'content': 'Engage in meaningful interactions', 'confidence': 0.8}]
            },
            interaction_history=[],
            trust_level=1.0,  # Full self-trust
            predictability=0.8,  # Reasonably predictable
            last_interaction=datetime.now()
        )
        
        self.agents['self'] = self_agent
    
    def _create_theory_of_mind_model(self, agent_id: str):
        """Create initial theory of mind model for an agent."""
        model = TheoryOfMindModel(
            model_id=f"tom_{agent_id}_{int(time.time())}",
            target_agent_id=agent_id,
            beliefs_about_beliefs={},
            beliefs_about_desires={},
            beliefs_about_intentions={},
            perspective_taking_ability=0.3,  # Start low
            false_belief_understanding=False,  # Develop over time
            recursive_modeling_depth=1,  # Start simple
            confidence=0.2  # Low initial confidence
        )
        
        self.theory_of_mind_models[agent_id] = model
    
    def _start_social_processing_threads(self):
        """Start background social processing threads."""
        if self.social_learning_thread is None or not self.social_learning_thread.is_alive():
            self.social_processing_enabled = True
            
            self.social_learning_thread = threading.Thread(target=self._social_learning_loop)
            self.social_learning_thread.daemon = True
            self.social_learning_thread.start()
            
            self.theory_of_mind_thread = threading.Thread(target=self._theory_of_mind_loop)
            self.theory_of_mind_thread.daemon = True
            self.theory_of_mind_thread.start()
    
    def _social_learning_loop(self):
        """Social learning and adaptation loop."""
        while self.social_processing_enabled:
            try:
                # Update social norms
                self.social_norm_learner.update_norms(self.social_interactions)
                
                # Analyze cooperation patterns
                self.cooperation_analyzer.analyze_patterns(self.social_interactions)
                
                # Process communication styles
                self.communication_processor.learn_styles(self.social_interactions)
                
                time.sleep(120.0)  # Every 2 minutes
                
            except Exception as e:
                logger.error(f"Error in social learning loop: {e}")
                time.sleep(240)
    
    def _theory_of_mind_loop(self):
        """Theory of mind development and refinement loop."""
        while self.social_processing_enabled:
            try:
                # Refine theory of mind models
                for agent_id, model in self.theory_of_mind_models.items():
                    self._refine_theory_of_mind_model(agent_id, model)
                
                # Update empathy capabilities
                self.empathy_engine.update_empathy_models(self.agents)
                
                # Process recursive modeling
                self.recursive_modeling.update_recursive_models(self.theory_of_mind_models)
                
                time.sleep(300.0)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in theory of mind loop: {e}")
                time.sleep(600)
    
    def cleanup(self):
        """Clean up social cognition system resources."""
        self.social_processing_enabled = False
        
        if self.social_learning_thread and self.social_learning_thread.is_alive():
            self.social_learning_thread.join(timeout=2)
        
        if self.theory_of_mind_thread and self.theory_of_mind_thread.is_alive():
            self.theory_of_mind_thread.join(timeout=2)
        
        logger.info("Social Cognition System cleaned up")

# Supporting component classes (simplified implementations)
class MentalStateInferenceEngine:
    def initialize(self): return True

class BehaviorPredictionEngine:
    def initialize(self): return True
    def analyze_behavioral_patterns(self, behaviors, situation):
        return {'confidence': 0.6, 'predicted_behavior': 'cooperative'}

class SocialLearningEngine:
    def initialize(self): return True
    def process_learning(self, learning_data): pass

class PerspectiveTakingEngine:
    def initialize(self): return True
    def take_perspective(self, agent, situation, tom_model):
        return {
            'thoughts': ['Agent is thinking about the situation'],
            'feelings': ['neutral'],
            'goals': ['achieve objectives'],
            'concerns': ['potential difficulties'],
            'confidence': 0.6
        }

class CommunicationProcessor:
    def initialize(self): return True
    def learn_styles(self, interactions): pass

class CooperationAnalyzer:
    def initialize(self): return True
    def analyze_patterns(self, interactions): pass
    def get_patterns_count(self): return 5

class ConflictResolver:
    def initialize(self): return True

class SocialNormLearner:
    def initialize(self): return True
    def update_norms(self, interactions): pass
    def get_norms_count(self): return 10

class FalseBeliefReasoner:
    def initialize(self): return True

class RecursiveModelingEngine:
    def initialize(self): return True
    def update_recursive_models(self, models): pass

class EmpathyEngine:
    def initialize(self): return True
    def generate_empathetic_understanding(self, agent, situation):
        return {
            'emotional_state': 'neutral',
            'empathy_level': 0.7,
            'understanding': 'Agent likely feels neutral about situation'
        }
    def update_empathy_models(self, agents): pass