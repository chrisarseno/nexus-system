"""
Enhanced Collaborative Intelligence System
Provides interfaces for expert domain collaboration and collective intelligence.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
import uuid

logger = logging.getLogger(__name__)

class CollaborationType(Enum):
    PEER_REVIEW = "peer_review"
    JOINT_RESEARCH = "joint_research"
    KNOWLEDGE_SHARING = "knowledge_sharing"
    PROBLEM_SOLVING = "problem_solving"
    CONSENSUS_BUILDING = "consensus_building"
    EXPERT_CONSULTATION = "expert_consultation"

class ExpertiseLevel(Enum):
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    WORLD_CLASS = "world_class"

class CollaborationStatus(Enum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    REVIEW = "review"
    COMPLETED = "completed"
    ARCHIVED = "archived"

@dataclass
class ExpertProfile:
    """Represents an expert collaborator profile."""
    expert_id: str
    name: str
    domains: List[str]
    expertise_levels: Dict[str, ExpertiseLevel]
    specializations: List[str]
    credentials: List[str]
    collaboration_history: List[str]
    reputation_score: float
    availability: Dict[str, Any]
    communication_preferences: Dict[str, str]
    research_interests: List[str]
    publications: List[str]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class CollaborationSession:
    """Represents a collaborative session."""
    session_id: str
    title: str
    description: str
    collaboration_type: CollaborationType
    participants: List[str]
    facilitator_id: Optional[str]
    objectives: List[str]
    timeline: Dict[str, datetime]
    resources: List[str]
    status: CollaborationStatus
    outcomes: List[str]
    deliverables: List[str]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class KnowledgeContribution:
    """Represents a knowledge contribution from an expert."""
    contribution_id: str
    contributor_id: str
    session_id: str
    content_type: str  # 'insight', 'data', 'analysis', 'methodology', 'critique'
    content: Dict[str, Any]
    confidence_level: float
    evidence_provided: List[str]
    peer_validations: List[Dict[str, Any]]
    impact_score: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ConsensusItem:
    """Represents an item being evaluated for consensus."""
    consensus_id: str
    session_id: str
    statement: str
    evidence: List[str]
    expert_positions: Dict[str, Dict[str, Any]]
    consensus_level: float
    agreement_threshold: float
    dissenting_opinions: List[str]
    resolution_status: str
    final_decision: Optional[str] = None

class CollaborativeIntelligenceSystem:
    """
    Advanced collaborative intelligence system enabling expert domain collaboration
    and collective intelligence emergence.
    """
    
    def __init__(self):
        self.expert_profiles: Dict[str, ExpertProfile] = {}
        self.collaboration_sessions: Dict[str, CollaborationSession] = {}
        self.knowledge_contributions: Dict[str, KnowledgeContribution] = {}
        self.consensus_items: Dict[str, ConsensusItem] = {}
        
        # Collaboration frameworks
        self.collaboration_protocols = {}
        self.facilitation_agents = {}
        self.consensus_mechanisms = {}
        
        # Expert matching and recommendation
        self.expertise_mapping = defaultdict(list)
        self.collaboration_networks = defaultdict(set)
        self.recommendation_engine = {}
        
        # Quality and validation systems
        self.peer_review_system = {}
        self.contribution_validation = {}
        self.reputation_system = {}
        
        # Collaboration analytics
        self.collaboration_stats = defaultdict(int)
        self.effectiveness_metrics = defaultdict(float)
        self.knowledge_flow_tracking = deque(maxlen=1000)
        
        # Background collaboration processes
        self.collaboration_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        self.initialized = False
    
    def initialize(self):
        """Initialize the collaborative intelligence system."""
        if self.initialized:
            return
            
        logger.info("Initializing Collaborative Intelligence System...")
        
        # Initialize collaboration protocols
        self._initialize_collaboration_protocols()
        
        # Initialize facilitation agents
        self._initialize_facilitation_agents()
        
        # Initialize consensus mechanisms
        self._initialize_consensus_mechanisms()
        
        # Start collaboration processes
        self._start_collaboration_processes()
        
        self.initialized = True
        logger.info("Collaborative Intelligence System initialized")
    
    def register_expert(self, expert_info: Dict[str, Any]) -> ExpertProfile:
        """Register a new expert in the collaboration system."""
        try:
            expert_id = expert_info.get('expert_id') or f"expert_{int(time.time())}"
            
            # Process expertise levels
            expertise_levels = {}
            for domain, level in expert_info.get('expertise_levels', {}).items():
                if isinstance(level, str):
                    expertise_levels[domain] = ExpertiseLevel(level)
                else:
                    expertise_levels[domain] = level
            
            expert_profile = ExpertProfile(
                expert_id=expert_id,
                name=expert_info['name'],
                domains=expert_info.get('domains', []),
                expertise_levels=expertise_levels,
                specializations=expert_info.get('specializations', []),
                credentials=expert_info.get('credentials', []),
                collaboration_history=[],
                reputation_score=expert_info.get('initial_reputation', 0.8),
                availability=expert_info.get('availability', {}),
                communication_preferences=expert_info.get('communication_preferences', {}),
                research_interests=expert_info.get('research_interests', []),
                publications=expert_info.get('publications', [])
            )
            
            self.expert_profiles[expert_id] = expert_profile
            
            # Update expertise mapping
            for domain in expert_profile.domains:
                self.expertise_mapping[domain].append(expert_id)
            
            self.collaboration_stats['experts_registered'] += 1
            logger.info(f"Registered expert: {expert_id}")
            
            return expert_profile
            
        except Exception as e:
            logger.error(f"Error registering expert: {e}")
            raise
    
    def create_collaboration_session(self, session_spec: Dict[str, Any]) -> CollaborationSession:
        """Create a new collaboration session."""
        try:
            session_id = session_spec.get('session_id') or f"session_{int(time.time())}"
            
            # Recommend experts for the session
            recommended_experts = self._recommend_experts_for_session(session_spec)
            
            # Initialize participants with recommended experts
            participants = session_spec.get('participants', [])
            participants.extend([exp['expert_id'] for exp in recommended_experts[:5]])  # Add top 5 recommendations
            participants = list(set(participants))  # Remove duplicates
            
            collaboration_session = CollaborationSession(
                session_id=session_id,
                title=session_spec['title'],
                description=session_spec['description'],
                collaboration_type=CollaborationType(session_spec.get('collaboration_type', 'joint_research')),
                participants=participants,
                facilitator_id=session_spec.get('facilitator_id'),
                objectives=session_spec.get('objectives', []),
                timeline=self._create_collaboration_timeline(session_spec),
                resources=session_spec.get('resources', []),
                status=CollaborationStatus.PROPOSED,
                outcomes=[],
                deliverables=session_spec.get('expected_deliverables', [])
            )
            
            self.collaboration_sessions[session_id] = collaboration_session
            
            # Notify participants
            self._notify_collaboration_participants(collaboration_session)
            
            self.collaboration_stats['sessions_created'] += 1
            logger.info(f"Created collaboration session: {session_id}")
            
            return collaboration_session
            
        except Exception as e:
            logger.error(f"Error creating collaboration session: {e}")
            raise
    
    def facilitate_collaboration(self, session_id: str) -> Dict[str, Any]:
        """Actively facilitate a collaboration session."""
        try:
            if session_id not in self.collaboration_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.collaboration_sessions[session_id]
            
            # Initialize facilitation
            facilitation_state = self._initialize_facilitation_state(session)
            
            # Guide discussion and knowledge sharing
            discussion_guidance = self._guide_collaborative_discussion(session)
            
            # Facilitate knowledge integration
            knowledge_integration = self._facilitate_knowledge_integration(session_id)
            
            # Monitor collaboration dynamics
            dynamics_analysis = self._analyze_collaboration_dynamics(session)
            
            # Generate recommendations for improvement
            improvement_recommendations = self._generate_collaboration_improvements(
                session, dynamics_analysis
            )
            
            # Update session status
            session.status = CollaborationStatus.ACTIVE
            
            facilitation_result = {
                'session_id': session_id,
                'facilitation_state': facilitation_state,
                'discussion_guidance': discussion_guidance,
                'knowledge_integration': knowledge_integration,
                'dynamics_analysis': dynamics_analysis,
                'improvement_recommendations': improvement_recommendations,
                'next_steps': self._suggest_next_steps(session, dynamics_analysis)
            }
            
            self.collaboration_stats['sessions_facilitated'] += 1
            logger.info(f"Facilitated collaboration session: {session_id}")
            
            return facilitation_result
            
        except Exception as e:
            logger.error(f"Error facilitating collaboration: {e}")
            return {}
    
    def contribute_knowledge(self, contribution: Dict[str, Any]) -> KnowledgeContribution:
        """Process a knowledge contribution from an expert."""
        try:
            contribution_id = f"contrib_{int(time.time())}"
            
            # Validate contributor expertise
            contributor_validation = self._validate_contributor_expertise(
                contribution['contributor_id'], contribution['session_id']
            )
            
            # Process contribution content
            processed_content = self._process_contribution_content(contribution['content'])
            
            # Assess contribution quality
            quality_assessment = self._assess_contribution_quality(
                processed_content, contribution['contributor_id']
            )
            
            knowledge_contribution = KnowledgeContribution(
                contribution_id=contribution_id,
                contributor_id=contribution['contributor_id'],
                session_id=contribution['session_id'],
                content_type=contribution['content_type'],
                content=processed_content,
                confidence_level=contribution.get('confidence_level', 0.8),
                evidence_provided=contribution.get('evidence', []),
                peer_validations=[],
                impact_score=quality_assessment['impact_score']
            )
            
            self.knowledge_contributions[contribution_id] = knowledge_contribution
            
            # Trigger peer validation process
            self._initiate_peer_validation(knowledge_contribution)
            
            # Update collaboration network
            self._update_collaboration_network(contribution['contributor_id'], contribution['session_id'])
            
            self.collaboration_stats['knowledge_contributions'] += 1
            logger.info(f"Processed knowledge contribution: {contribution_id}")
            
            return knowledge_contribution
            
        except Exception as e:
            logger.error(f"Error processing knowledge contribution: {e}")
            raise
    
    def build_consensus(self, session_id: str, statements: List[str]) -> List[ConsensusItem]:
        """Build consensus on key statements or decisions."""
        try:
            if session_id not in self.collaboration_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.collaboration_sessions[session_id]
            consensus_items = []
            
            for statement in statements:
                consensus_id = f"consensus_{session_id}_{int(time.time())}"
                
                # Gather expert positions
                expert_positions = self._gather_expert_positions(session, statement)
                
                # Calculate consensus level
                consensus_level = self._calculate_consensus_level(expert_positions)
                
                # Identify dissenting opinions
                dissenting_opinions = self._identify_dissenting_opinions(expert_positions)
                
                # Determine resolution status
                resolution_status = self._determine_resolution_status(
                    consensus_level, dissenting_opinions
                )
                
                consensus_item = ConsensusItem(
                    consensus_id=consensus_id,
                    session_id=session_id,
                    statement=statement,
                    evidence=self._gather_supporting_evidence(statement, session_id),
                    expert_positions=expert_positions,
                    consensus_level=consensus_level,
                    agreement_threshold=0.75,  # 75% agreement threshold
                    dissenting_opinions=dissenting_opinions,
                    resolution_status=resolution_status
                )
                
                # Facilitate consensus building if needed
                if consensus_level < consensus_item.agreement_threshold:
                    consensus_facilitation = self._facilitate_consensus_building(consensus_item)
                    consensus_item.consensus_level = consensus_facilitation['updated_consensus']
                    consensus_item.resolution_status = consensus_facilitation['resolution_status']
                
                self.consensus_items[consensus_id] = consensus_item
                consensus_items.append(consensus_item)
            
            self.collaboration_stats['consensus_items_processed'] += len(consensus_items)
            logger.info(f"Built consensus on {len(consensus_items)} items for session {session_id}")
            
            return consensus_items
            
        except Exception as e:
            logger.error(f"Error building consensus: {e}")
            return []
    
    def generate_collective_insights(self, session_id: str) -> Dict[str, Any]:
        """Generate collective insights from collaboration."""
        try:
            if session_id not in self.collaboration_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            # Gather all contributions from the session
            session_contributions = [
                contrib for contrib in self.knowledge_contributions.values()
                if contrib.session_id == session_id
            ]
            
            # Analyze contribution patterns
            pattern_analysis = self._analyze_contribution_patterns(session_contributions)
            
            # Synthesize knowledge across contributions
            knowledge_synthesis = self._synthesize_collective_knowledge(session_contributions)
            
            # Identify emergent insights
            emergent_insights = self._identify_emergent_insights(
                session_contributions, pattern_analysis
            )
            
            # Generate novel hypotheses from collective intelligence
            collective_hypotheses = self._generate_collective_hypotheses(
                knowledge_synthesis, emergent_insights
            )
            
            # Assess collective intelligence quality
            collective_quality = self._assess_collective_intelligence_quality(
                session_contributions, knowledge_synthesis
            )
            
            collective_insights = {
                'session_id': session_id,
                'contribution_count': len(session_contributions),
                'pattern_analysis': pattern_analysis,
                'knowledge_synthesis': knowledge_synthesis,
                'emergent_insights': emergent_insights,
                'collective_hypotheses': collective_hypotheses,
                'collective_quality_score': collective_quality,
                'recommendations': self._generate_insight_recommendations(
                    emergent_insights, collective_hypotheses
                ),
                'generated_at': datetime.now().isoformat()
            }
            
            self.collaboration_stats['collective_insights_generated'] += 1
            logger.info(f"Generated collective insights for session: {session_id}")
            
            return collective_insights
            
        except Exception as e:
            logger.error(f"Error generating collective insights: {e}")
            return {}
    
    def get_collaboration_analytics(self) -> Dict[str, Any]:
        """Get comprehensive collaboration analytics."""
        try:
            with self.lock:
                active_sessions = len([
                    s for s in self.collaboration_sessions.values()
                    if s.status == CollaborationStatus.ACTIVE
                ])
                
                total_experts = len(self.expert_profiles)
                total_contributions = len(self.knowledge_contributions)
                total_consensus_items = len(self.consensus_items)
                
                # Calculate collaboration effectiveness
                effectiveness_metrics = self._calculate_collaboration_effectiveness()
                
                # Analyze expert network
                network_analysis = self._analyze_expert_network()
                
                # Calculate knowledge flow metrics
                knowledge_flow = self._calculate_knowledge_flow_metrics()
                
                return {
                    'collaboration_summary': {
                        'active_sessions': active_sessions,
                        'total_experts': total_experts,
                        'knowledge_contributions': total_contributions,
                        'consensus_items': total_consensus_items,
                        'expertise_domains': len(self.expertise_mapping)
                    },
                    'effectiveness_metrics': effectiveness_metrics,
                    'expert_network': network_analysis,
                    'knowledge_flow': knowledge_flow,
                    'collaboration_statistics': dict(self.collaboration_stats),
                    'quality_metrics': dict(self.effectiveness_metrics),
                    'system_health': {
                        'collaboration_processes_active': self.running,
                        'facilitation_agents_available': len(self.facilitation_agents),
                        'consensus_mechanisms_enabled': len(self.consensus_mechanisms)
                    }
                }
                
        except Exception as e:
            logger.error(f"Error generating collaboration analytics: {e}")
            return {}
    
    def _recommend_experts_for_session(self, session_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend experts for a collaboration session."""
        recommendations = []
        
        required_domains = session_spec.get('domains', [])
        collaboration_type = session_spec.get('collaboration_type', 'joint_research')
        
        # Find experts in relevant domains
        candidate_experts = []
        for domain in required_domains:
            domain_experts = self.expertise_mapping.get(domain, [])
            candidate_experts.extend(domain_experts)
        
        # Score experts based on relevance
        for expert_id in set(candidate_experts):
            if expert_id in self.expert_profiles:
                expert = self.expert_profiles[expert_id]
                score = self._calculate_expert_relevance_score(expert, session_spec)
                
                recommendations.append({
                    'expert_id': expert_id,
                    'expert_name': expert.name,
                    'relevance_score': score,
                    'domains': expert.domains,
                    'reputation_score': expert.reputation_score
                })
        
        # Sort by relevance score
        recommendations.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return recommendations
    
    def _calculate_expert_relevance_score(self, expert: ExpertProfile, 
                                        session_spec: Dict[str, Any]) -> float:
        """Calculate how relevant an expert is for a session."""
        score = 0.0
        
        # Domain expertise match
        required_domains = set(session_spec.get('domains', []))
        expert_domains = set(expert.domains)
        domain_overlap = len(required_domains.intersection(expert_domains))
        if required_domains:
            score += 0.4 * (domain_overlap / len(required_domains))
        
        # Reputation score
        score += 0.3 * expert.reputation_score
        
        # Collaboration history relevance
        collaboration_type = session_spec.get('collaboration_type', 'joint_research')
        if collaboration_type in [session['type'] for session in expert.collaboration_history[-5:]]:
            score += 0.2
        
        # Availability
        if expert.availability.get('available', True):
            score += 0.1
        
        return min(1.0, score)
    
    def _start_collaboration_processes(self):
        """Start background collaboration processes."""
        if not self.collaboration_thread:
            self.running = True
            self.collaboration_thread = threading.Thread(target=self._collaboration_loop)
            self.collaboration_thread.daemon = True
            self.collaboration_thread.start()
            logger.info("Collaboration processes started")
    
    def _collaboration_loop(self):
        """Main collaboration loop for facilitation and management."""
        while self.running:
            try:
                time.sleep(1800)  # Run every 30 minutes
                
                # Facilitate active sessions
                self._facilitate_active_sessions()
                
                # Update expert recommendations
                self._update_expert_recommendations()
                
                # Process peer validations
                self._process_pending_peer_validations()
                
                # Update collaboration networks
                self._update_collaboration_networks()
                
            except Exception as e:
                logger.error(f"Error in collaboration loop: {e}")
    
    def shutdown(self):
        """Shutdown the collaborative intelligence system."""
        self.running = False
        if self.collaboration_thread:
            self.collaboration_thread.join(timeout=5)
        logger.info("Collaborative Intelligence System shutdown completed")