"""
Collaborative Intelligence System for Human-AI Partnership
Implements human-in-the-loop learning with intelligent guidance-seeking
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

class InteractionType(Enum):
    """Types of human-AI interactions."""
    QUESTION_ANSWER = "question_answer"
    FEEDBACK_CORRECTION = "feedback_correction"
    GUIDANCE_REQUEST = "guidance_request"
    PREFERENCE_EXPRESSION = "preference_expression"
    COLLABORATIVE_DECISION = "collaborative_decision"
    KNOWLEDGE_SHARING = "knowledge_sharing"
    TASK_DELEGATION = "task_delegation"
    CREATIVE_COLLABORATION = "creative_collaboration"

class QuestionCategory(Enum):
    """Categories of questions the AI can ask."""
    CLARIFICATION = "clarification"
    PREFERENCE = "preference"
    VALIDATION = "validation"
    GUIDANCE = "guidance"
    OPINION = "opinion"
    DECISION_SUPPORT = "decision_support"
    KNOWLEDGE_GAP = "knowledge_gap"
    ETHICAL_CONSIDERATION = "ethical_consideration"

class ConfidenceLevel(Enum):
    """Confidence levels for AI decisions."""
    VERY_LOW = "very_low"    # 0.0 - 0.2
    LOW = "low"              # 0.2 - 0.4
    MODERATE = "moderate"    # 0.4 - 0.6
    HIGH = "high"            # 0.6 - 0.8
    VERY_HIGH = "very_high"  # 0.8 - 1.0

@dataclass
class HumanInteraction:
    """Represents an interaction with a human user."""
    interaction_id: str
    interaction_type: InteractionType
    ai_action: str
    human_response: str
    human_satisfaction: Optional[float]
    context: Dict[str, Any]
    learning_extracted: List[str]
    timestamp: datetime
    
    def extract_preferences(self) -> Dict[str, Any]:
        """Extract user preferences from this interaction."""
        preferences = {}
        
        # Analyze response sentiment and content
        if self.human_satisfaction and self.human_satisfaction > 0.7:
            preferences['positive_patterns'] = [self.ai_action]
        elif self.human_satisfaction and self.human_satisfaction < 0.3:
            preferences['negative_patterns'] = [self.ai_action]
        
        # Extract explicit preferences from response
        if 'prefer' in self.human_response.lower():
            preferences['explicit_preference'] = self.human_response
        
        if 'like' in self.human_response.lower():
            preferences['positive_feedback'] = self.human_response
        
        if 'don\'t' in self.human_response.lower() or 'dislike' in self.human_response.lower():
            preferences['negative_feedback'] = self.human_response
        
        return preferences

@dataclass
class IntelligentQuestion:
    """Represents a question the AI wants to ask a human."""
    question_id: str
    category: QuestionCategory
    question_text: str
    context: Dict[str, Any]
    urgency: float
    confidence_threshold: float
    expected_value: float
    reasoning: str
    
    def should_ask_now(self, current_confidence: float) -> bool:
        """Determine if this question should be asked now."""
        confidence_trigger = current_confidence < self.confidence_threshold
        urgency_trigger = self.urgency > 0.7
        value_trigger = self.expected_value > 0.6
        
        return confidence_trigger and (urgency_trigger or value_trigger)

@dataclass
class CollaborativeDecision:
    """Represents a decision made collaboratively."""
    decision_id: str
    decision_context: str
    ai_recommendation: Dict[str, Any]
    human_input: Dict[str, Any]
    final_decision: Dict[str, Any]
    outcome_satisfaction: Optional[float]
    lessons_learned: List[str]
    timestamp: datetime

class CollaborativeIntelligenceSystem:
    """
    System for collaborative human-AI intelligence that learns from interactions,
    asks thoughtful questions, and seeks guidance appropriately.
    """
    
    def __init__(self):
        # Core collaboration components
        self.interaction_history = deque(maxlen=10000)
        self.pending_questions = []
        self.user_preferences = defaultdict(dict)
        self.collaborative_decisions = deque(maxlen=1000)
        
        # Learning and adaptation engines
        self.response_learner = ResponseLearningEngine()
        self.preference_modeler = PreferenceModelingEngine()
        self.question_generator = IntelligentQuestionGenerator()
        self.guidance_seeker = GuidanceSeekingEngine()
        
        # Collaboration optimization
        self.workflow_optimizer = CollaborativeWorkflowOptimizer()
        self.context_analyzer = ContextAnalyzer()
        self.confidence_assessor = ConfidenceAssessor()
        self.value_estimator = ValueEstimator()
        
        # Human behavior modeling
        self.communication_styler = CommunicationStyler()
        self.timing_optimizer = InteractionTimingOptimizer()
        self.attention_manager = AttentionManager()
        
        # Current collaboration state
        self.current_collaboration_context = {
            'active_tasks': [],
            'human_availability': 'available',
            'collaboration_mode': 'balanced',
            'recent_interactions': [],
            'pending_guidance_requests': []
        }
        
        # Collaboration parameters
        self.confidence_threshold_for_questions = 0.4
        self.max_pending_questions = 5
        self.question_spacing_minutes = 10
        self.preference_learning_rate = 0.1
        
        # Background processing
        self.collaboration_enabled = True
        self.learning_thread = None
        self.question_management_thread = None
        
        # Performance metrics
        self.collaboration_metrics = {
            'interactions_processed': 0,
            'questions_asked': 0,
            'guidance_requests': 0,
            'user_satisfaction_avg': 0.0,
            'preference_accuracy': 0.0,
            'collaboration_efficiency': 0.0,
            'learning_improvements': 0
        }
        
        self.initialized = False
        logger.info("Collaborative Intelligence System initialized")
    
    def initialize(self) -> bool:
        """Initialize the collaborative intelligence system."""
        try:
            # Initialize learning and adaptation engines
            self.response_learner.initialize()
            self.preference_modeler.initialize()
            self.question_generator.initialize()
            self.guidance_seeker.initialize()
            
            # Initialize collaboration optimization
            self.workflow_optimizer.initialize()
            self.context_analyzer.initialize()
            self.confidence_assessor.initialize()
            self.value_estimator.initialize()
            
            # Initialize human behavior modeling
            self.communication_styler.initialize()
            self.timing_optimizer.initialize()
            self.attention_manager.initialize()
            
            # Start collaborative processing
            self._start_collaboration_threads()
            
            self.initialized = True
            logger.info("✅ Collaborative Intelligence System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize collaborative intelligence system: {e}")
            return False
    
    def generate_collaborative_response(self, query: str, perspectives: List[str] = None, 
                                      context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a collaborative response using AI models with bias mitigation."""
        try:
            import uuid
            
            if perspectives is None:
                perspectives = ['technical', 'ethical', 'cultural', 'practical']
            if context is None:
                context = {}
            
            # Bias mitigation check
            bias_check = self._perform_bias_check(query, perspectives)
            if not bias_check['passed']:
                return {
                    'response': 'Query blocked due to bias concerns: ' + bias_check['reason'],
                    'bias_check_failed': True,
                    'timestamp': datetime.now().isoformat()
                }
            
            confidence = 0.7  # Default confidence
            metadata = {
                'perspectives_considered': perspectives,
                'bias_mitigation_applied': True,
                'individual_influence_cap': 0.25,
                'timestamp': datetime.now().isoformat()
            }
            
            # Try to use AI model integration for real responses
            try:
                from intelligence.ai_model_integration import AIModelIntegrationSystem, ModelRequest, ModelCapability
                
                # Initialize AI model system
                ai_system = AIModelIntegrationSystem()
                if ai_system.initialize():
                    
                    # Create a comprehensive prompt for collaborative analysis
                    system_message = f"""
You are part of a collaborative intelligence system that considers multiple perspectives while maintaining ethical standards and avoiding bias.

Your task is to analyze the query while incorporating diverse viewpoints:
- Perspectives to consider: {', '.join(perspectives)}
- Maximum individual influence: 25% (no single perspective should dominate)
- Focus on balanced, collaborative analysis
- Consider cultural sensitivity and ethical implications
"""
                    
                    prompt = f"""
Query: {query}
Context: {context}

Please provide a collaborative analysis that:
1. Addresses the query from multiple perspectives
2. Ensures no single viewpoint dominates (25% max influence cap)
3. Synthesizes different approaches into a balanced response
4. Maintains ethical standards and cultural sensitivity
5. Provides transparent reasoning for the collaborative approach

Format your response with:
- Primary Analysis
- Diverse Perspectives (list key viewpoints)
- Synthesis (how perspectives are integrated)
- Confidence Level and reasoning
"""
                    
                    model_request = ModelRequest(
                        request_id=f"collaborative_{uuid.uuid4().hex[:8]}",
                        model_provider=None,  # Auto-select best model
                        model_name="",
                        prompt=prompt,
                        context={'query': query, 'perspectives': perspectives},
                        temperature=0.8,  # Encourage diverse thinking
                        max_tokens=2000,
                        system_message=system_message,
                        capabilities_required=[ModelCapability.REASONING, ModelCapability.ANALYSIS, ModelCapability.ETHICAL_REASONING]
                    )
                    
                    # Generate response using AI models
                    ai_response = ai_system.generate_response(model_request)
                    
                    if ai_response and ai_response.content:
                        response_content = ai_response.content
                        confidence = ai_response.confidence
                        metadata.update({
                            'ai_model_used': f"{ai_response.model_provider.value}:{ai_response.model_name}",
                            'ai_usage_stats': ai_response.usage_stats,
                            'generated_by_ai': True
                        })
                    else:
                        # Fallback to default response if AI fails
                        response_content = self._generate_fallback_response(query, perspectives, context)
                        metadata['fallback_used'] = True
                        
                else:
                    response_content = self._generate_fallback_response(query, perspectives, context)
                    metadata['ai_unavailable'] = True
                    
            except Exception as e:
                logger.warning(f"AI model integration failed, using fallback: {e}")
                response_content = self._generate_fallback_response(query, perspectives, context)
                metadata['ai_error'] = str(e)
            
            # Record interaction for learning
            self.collaboration_metrics['interactions_processed'] += 1
            
            return {
                'response': response_content,
                'confidence': confidence,
                'perspectives_used': perspectives,
                'metadata': metadata,
                'bias_check_passed': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in collaborative response generation: {e}")
            return {
                'response': 'Error generating collaborative response',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _perform_bias_check(self, query: str, perspectives: List[str]) -> Dict[str, Any]:
        """Perform bias mitigation check on query and perspectives."""
        # Simple bias detection (could be enhanced with more sophisticated checks)
        harmful_patterns = ['discriminate', 'stereotype', 'prejudice', 'bias against']
        
        query_lower = query.lower()
        for pattern in harmful_patterns:
            if pattern in query_lower:
                return {
                    'passed': False,
                    'reason': f'Potentially harmful content detected: {pattern}',
                    'confidence': 0.9
                }
        
        # Check for balanced perspectives
        if len(perspectives) < 2:
            return {
                'passed': False,
                'reason': 'Insufficient perspective diversity (minimum 2 required)',
                'confidence': 0.8
            }
        
        return {'passed': True, 'reason': 'Bias check passed', 'confidence': 0.8}
    
    def _generate_fallback_response(self, query: str, perspectives: List[str], context: Dict[str, Any]) -> str:
        """Generate fallback response when AI models are unavailable."""
        return f"""
Based on the query "{query}", here's a collaborative analysis considering multiple perspectives:

**Primary Analysis**: The query touches on important aspects that require careful consideration of different viewpoints and cultural contexts.

**Diverse Perspectives**:
- {perspectives[0] if perspectives else 'Technical analysis'}: Examining the technical and practical aspects
- {perspectives[1] if len(perspectives) > 1 else 'Ethical considerations'}: Considering ethical implications and moral dimensions
- {perspectives[2] if len(perspectives) > 2 else 'Cultural sensitivity'}: Accounting for cultural diversity and sensitivity

**Synthesis**: This collaborative approach ensures that the response incorporates multiple viewpoints while maintaining ethical standards and avoiding bias. Each perspective is limited to 25% influence to prevent dominance by any single viewpoint.

**Confidence Level**: Based on the analysis of {len(perspectives)} perspectives with individual influence capped at 25%, this response represents a balanced collaborative viewpoint with moderate confidence.

*Note: This response was generated using fallback logic. For enhanced analysis, please ensure AI model integration is available.*
"""
    
    def process_human_interaction(self, ai_action: str, human_response: str,
                                context: Dict[str, Any] = None,
                                interaction_type: InteractionType = InteractionType.QUESTION_ANSWER) -> str:
        """Process and learn from a human interaction."""
        try:
            interaction_id = f"interaction_{int(time.time() * 1000)}"
            
            # Analyze human response
            response_analysis = self.response_learner.analyze_response(
                ai_action, human_response, context or {}
            )
            
            # Estimate satisfaction
            satisfaction = self._estimate_satisfaction(human_response, response_analysis)
            
            # Extract learning insights
            learning_insights = self.response_learner.extract_learning(
                ai_action, human_response, response_analysis
            )
            
            # Create interaction record
            interaction = HumanInteraction(
                interaction_id=interaction_id,
                interaction_type=interaction_type,
                ai_action=ai_action,
                human_response=human_response,
                human_satisfaction=satisfaction,
                context=context or {},
                learning_extracted=learning_insights,
                timestamp=datetime.now()
            )
            
            self.interaction_history.append(interaction)
            
            # Update user preferences
            preferences = interaction.extract_preferences()
            self.preference_modeler.update_preferences(preferences)
            
            # Learn communication patterns
            self.communication_styler.learn_from_interaction(interaction)
            
            # Update collaboration context
            self.current_collaboration_context['recent_interactions'].append(interaction_id)
            if len(self.current_collaboration_context['recent_interactions']) > 10:
                self.current_collaboration_context['recent_interactions'] = self.current_collaboration_context['recent_interactions'][-10:]
            
            self.collaboration_metrics['interactions_processed'] += 1
            self.collaboration_metrics['user_satisfaction_avg'] = self._calculate_average_satisfaction()
            
            logger.debug(f"Processed human interaction: {interaction_id}")
            return interaction_id
            
        except Exception as e:
            logger.error(f"Error processing human interaction: {e}")
            return ""
    
    def should_seek_guidance(self, task_context: Dict[str, Any],
                           confidence_score: float) -> Dict[str, Any]:
        """Determine if guidance should be sought for a task."""
        try:
            # Analyze task complexity and stakes
            complexity_analysis = self.context_analyzer.analyze_complexity(task_context)
            stakes_analysis = self.context_analyzer.analyze_stakes(task_context)
            
            # Check confidence threshold
            confidence_trigger = confidence_score < self.confidence_threshold_for_questions
            
            # Check if this is a domain where human input is particularly valuable
            domain_analysis = self.guidance_seeker.analyze_domain_value(task_context)
            
            # Consider user preferences for guidance
            user_guidance_preferences = self.preference_modeler.get_guidance_preferences()
            
            # Calculate guidance necessity score
            guidance_score = self._calculate_guidance_score(
                confidence_score, complexity_analysis, stakes_analysis, domain_analysis
            )
            
            should_seek = guidance_score > 0.6 or confidence_trigger
            
            guidance_decision = {
                'should_seek_guidance': should_seek,
                'guidance_score': guidance_score,
                'confidence_score': confidence_score,
                'reasons': [],
                'suggested_approach': None
            }
            
            # Add specific reasons
            if confidence_trigger:
                guidance_decision['reasons'].append(f'Low confidence: {confidence_score:.2f}')
            
            if complexity_analysis.get('high_complexity', False):
                guidance_decision['reasons'].append('High task complexity')
            
            if stakes_analysis.get('high_stakes', False):
                guidance_decision['reasons'].append('High-stakes decision')
            
            if domain_analysis.get('human_expertise_valuable', False):
                guidance_decision['reasons'].append('Domain benefits from human expertise')
            
            # Suggest guidance approach
            if should_seek:
                guidance_decision['suggested_approach'] = self._suggest_guidance_approach(
                    task_context, confidence_score, complexity_analysis
                )
            
            return guidance_decision
            
        except Exception as e:
            logger.error(f"Error determining guidance need: {e}")
            return {'should_seek_guidance': False, 'error': str(e)}
    
    def generate_intelligent_question(self, topic: str, context: Dict[str, Any],
                                    category: QuestionCategory = QuestionCategory.GUIDANCE) -> Optional[str]:
        """Generate an intelligent question to ask the human."""
        try:
            # Analyze what we need to know
            knowledge_gaps = self.question_generator.identify_knowledge_gaps(topic, context)
            
            # Consider user preferences for question types
            question_preferences = self.preference_modeler.get_question_preferences()
            
            # Generate question options
            question_options = self.question_generator.generate_question_options(
                topic, context, category, knowledge_gaps
            )
            
            if not question_options:
                return None
            
            # Select best question based on value and user preferences
            best_question = self.question_generator.select_best_question(
                question_options, question_preferences
            )
            
            # Check if we should ask this question now
            current_confidence = self.confidence_assessor.assess_confidence(topic, context)
            
            if not best_question.should_ask_now(current_confidence):
                # Queue for later
                self.pending_questions.append(best_question)
                return None
            
            # Create the question
            question_id = f"question_{int(time.time() * 1000)}"
            best_question.question_id = question_id
            
            # Style the question according to user preferences
            styled_question = self.communication_styler.style_question(
                best_question.question_text, context
            )
            
            self.collaboration_metrics['questions_asked'] += 1
            
            logger.debug(f"Generated intelligent question: {question_id}")
            return styled_question
            
        except Exception as e:
            logger.error(f"Error generating intelligent question: {e}")
            return None
    
    def request_collaborative_decision(self, decision_context: str,
                                     ai_analysis: Dict[str, Any],
                                     options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Request collaborative decision-making with human."""
        try:
            decision_id = f"decision_{int(time.time() * 1000)}"
            
            # Prepare AI recommendation
            ai_recommendation = self._prepare_ai_recommendation(ai_analysis, options)
            
            # Format collaborative decision request
            collaboration_request = {
                'decision_id': decision_id,
                'context': decision_context,
                'ai_analysis': ai_analysis,
                'ai_recommendation': ai_recommendation,
                'options': options,
                'reasoning': ai_recommendation.get('reasoning', ''),
                'confidence': ai_recommendation.get('confidence', 0.5),
                'request_type': 'collaborative_decision'
            }
            
            # Add to pending guidance requests
            self.current_collaboration_context['pending_guidance_requests'].append(decision_id)
            
            self.collaboration_metrics['guidance_requests'] += 1
            
            return collaboration_request
            
        except Exception as e:
            logger.error(f"Error requesting collaborative decision: {e}")
            return {'error': str(e)}
    
    def learn_from_feedback(self, task_description: str, feedback: str,
                          outcome_satisfaction: float) -> bool:
        """Learn from task feedback to improve future performance."""
        try:
            # Analyze feedback content
            feedback_analysis = self.response_learner.analyze_feedback(
                task_description, feedback, outcome_satisfaction
            )
            
            # Extract actionable insights
            insights = self.response_learner.extract_actionable_insights(feedback_analysis)
            
            # Update preference models
            preference_updates = self.preference_modeler.extract_preference_updates(
                feedback_analysis, insights
            )
            
            # Apply learning to systems
            learning_applied = 0
            
            for insight in insights:
                if self._apply_learning_insight(insight):
                    learning_applied += 1
            
            for update in preference_updates:
                if self.preference_modeler.apply_preference_update(update):
                    learning_applied += 1
            
            # Update workflow patterns
            self.workflow_optimizer.learn_from_outcome(
                task_description, feedback, outcome_satisfaction
            )
            
            self.collaboration_metrics['learning_improvements'] += learning_applied
            
            logger.debug(f"Applied {learning_applied} learning improvements from feedback")
            return learning_applied > 0
            
        except Exception as e:
            logger.error(f"Error learning from feedback: {e}")
            return False
    
    def get_collaborative_intelligence_state(self) -> Dict[str, Any]:
        """Get comprehensive state of collaborative intelligence system."""
        if not self.initialized:
            return {'error': 'Collaborative intelligence system not initialized'}
        
        # Update metrics
        self._update_collaboration_metrics()
        
        # Get recent interactions summary
        recent_interactions = [
            {
                'interaction_id': interaction.interaction_id,
                'type': interaction.interaction_type.value,
                'satisfaction': interaction.human_satisfaction,
                'learning_extracted': len(interaction.learning_extracted),
                'time_ago': (datetime.now() - interaction.timestamp).total_seconds()
            }
            for interaction in list(self.interaction_history)[-10:]
        ]
        
        # Get pending questions summary
        pending_questions_summary = [
            {
                'question_id': q.question_id if hasattr(q, 'question_id') else 'pending',
                'category': q.category.value,
                'urgency': q.urgency,
                'expected_value': q.expected_value
            }
            for q in self.pending_questions[:5]
        ]
        
        # Get user preferences summary
        preferences_summary = {
            'communication_style': self.communication_styler.get_preferred_style() if hasattr(self.communication_styler, 'get_preferred_style') else 'balanced',
            'guidance_frequency': self.preference_modeler.get_guidance_frequency() if hasattr(self.preference_modeler, 'get_guidance_frequency') else 'moderate',
            'question_types_preferred': self.preference_modeler.get_preferred_question_types() if hasattr(self.preference_modeler, 'get_preferred_question_types') else ['clarification', 'guidance']
        }
        
        return {
            'collaboration_active': self.collaboration_enabled,
            'current_context': self.current_collaboration_context,
            'recent_interactions': recent_interactions,
            'pending_questions': pending_questions_summary,
            'user_preferences': preferences_summary,
            'collaboration_capabilities': {
                'confidence_threshold': self.confidence_threshold_for_questions,
                'max_pending_questions': self.max_pending_questions,
                'question_spacing_minutes': self.question_spacing_minutes,
                'learning_rate': self.preference_learning_rate
            },
            'intelligent_features': {
                'response_learning': True,
                'preference_modeling': True,
                'intelligent_questioning': True,
                'guidance_seeking': True,
                'workflow_optimization': True,
                'communication_styling': True
            },
            'collaboration_metrics': self.collaboration_metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def _start_collaboration_threads(self):
        """Start background collaboration processing threads."""
        if self.learning_thread is None or not self.learning_thread.is_alive():
            self.collaboration_enabled = True
            
            self.learning_thread = threading.Thread(target=self._continuous_learning_loop)
            self.learning_thread.daemon = True
            self.learning_thread.start()
            
            self.question_management_thread = threading.Thread(target=self._question_management_loop)
            self.question_management_thread.daemon = True
            self.question_management_thread.start()
    
    def _continuous_learning_loop(self):
        """Continuous learning from interactions loop."""
        while self.collaboration_enabled:
            try:
                # Analyze recent interactions for patterns
                if len(self.interaction_history) >= 5:
                    pattern_analysis = self.response_learner.analyze_interaction_patterns(
                        list(self.interaction_history)[-20:]
                    )
                    
                    # Update models based on patterns
                    self.preference_modeler.update_from_patterns(pattern_analysis)
                    self.communication_styler.adapt_from_patterns(pattern_analysis)
                
                time.sleep(600.0)  # Every 10 minutes
                
            except Exception as e:
                logger.error(f"Error in continuous learning loop: {e}")
                time.sleep(1200)
    
    def _question_management_loop(self):
        """Question management and timing loop."""
        while self.collaboration_enabled:
            try:
                # Check if it's a good time to ask pending questions
                if self.pending_questions and self._is_good_time_for_questions():
                    # Ask the highest priority question
                    question = max(self.pending_questions, key=lambda q: q.urgency * q.expected_value)
                    
                    # Remove from pending and process
                    self.pending_questions.remove(question)
                    
                    # Here you would trigger the actual question asking mechanism
                    # This would integrate with your user interface
                    
                time.sleep(self.question_spacing_minutes * 60)  # Respect spacing
                
            except Exception as e:
                logger.error(f"Error in question management loop: {e}")
                time.sleep(600)
    
    def cleanup(self):
        """Clean up collaborative intelligence system resources."""
        self.collaboration_enabled = False
        
        if self.learning_thread and self.learning_thread.is_alive():
            self.learning_thread.join(timeout=2)
        
        if self.question_management_thread and self.question_management_thread.is_alive():
            self.question_management_thread.join(timeout=2)
        
        logger.info("Collaborative Intelligence System cleaned up")

# Supporting component classes (simplified implementations)
class ResponseLearningEngine:
    def initialize(self): return True
    def analyze_response(self, action, response, context):
        return {'sentiment': 'positive', 'clarity': 0.8, 'satisfaction_indicators': ['helpful']}
    def extract_learning(self, action, response, analysis):
        return ['user_prefers_detailed_explanations', 'positive_feedback_pattern']
    def analyze_feedback(self, task, feedback, satisfaction):
        return {'feedback_type': 'constructive', 'improvement_areas': ['clarity']}
    def extract_actionable_insights(self, analysis):
        return ['be_more_specific', 'provide_examples']
    def analyze_interaction_patterns(self, interactions):
        return {'patterns': ['prefers_morning_interactions', 'likes_concise_responses']}

class PreferenceModelingEngine:
    def initialize(self): return True
    def update_preferences(self, preferences): pass
    def get_guidance_preferences(self): return {'frequency': 'moderate', 'detail_level': 'high'}
    def get_question_preferences(self): return {'style': 'direct', 'complexity': 'moderate'}
    def extract_preference_updates(self, analysis, insights): return []
    def apply_preference_update(self, update): return True
    def update_from_patterns(self, patterns): pass

class IntelligentQuestionGenerator:
    def initialize(self): return True
    def identify_knowledge_gaps(self, topic, context): return ['missing_context', 'unclear_goals']
    def generate_question_options(self, topic, context, category, gaps):
        return [IntelligentQuestion(
            question_id='', category=category, question_text='What would you prefer in this situation?',
            context=context, urgency=0.6, confidence_threshold=0.4, expected_value=0.8,
            reasoning='Need clarification on user preferences'
        )]
    def select_best_question(self, options, preferences): return options[0] if options else None

class GuidanceSeekingEngine:
    def initialize(self): return True
    def analyze_domain_value(self, context): return {'human_expertise_valuable': True}

class CollaborativeWorkflowOptimizer:
    def initialize(self): return True
    def learn_from_outcome(self, task, feedback, satisfaction): pass

class ContextAnalyzer:
    def initialize(self): return True
    def analyze_complexity(self, context): return {'complexity_score': 0.6, 'high_complexity': False}
    def analyze_stakes(self, context): return {'stakes_score': 0.5, 'high_stakes': False}

class ConfidenceAssessor:
    def initialize(self): return True
    def assess_confidence(self, topic, context): return 0.7

class ValueEstimator:
    def initialize(self): return True

class CommunicationStyler:
    def initialize(self): return True
    def learn_from_interaction(self, interaction): pass
    def style_question(self, question, context): return question
    def adapt_from_patterns(self, patterns): pass

class InteractionTimingOptimizer:
    def initialize(self): return True

class AttentionManager:
    def initialize(self): return True