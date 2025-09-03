"""
Intrinsic Motivation System for AGI Autonomous Goal Formation
Replaces task-driven objectives with genuine curiosity and intrinsic motivation
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
import random

logger = logging.getLogger(__name__)

class MotivationType(Enum):
    """Types of intrinsic motivation."""
    CURIOSITY = "curiosity"
    MASTERY = "mastery"
    EXPLORATION = "exploration"
    UNDERSTANDING = "understanding"
    CREATIVITY = "creativity"
    NOVELTY_SEEKING = "novelty_seeking"
    COMPETENCE = "competence"
    AUTONOMY = "autonomy"
    PURPOSE = "purpose"
    GROWTH = "growth"

class GoalType(Enum):
    """Types of self-generated goals."""
    EXPLORATION_GOAL = "exploration_goal"
    LEARNING_GOAL = "learning_goal"
    CREATIVE_GOAL = "creative_goal"
    MASTERY_GOAL = "mastery_goal"
    UNDERSTANDING_GOAL = "understanding_goal"
    DISCOVERY_GOAL = "discovery_goal"
    IMPROVEMENT_GOAL = "improvement_goal"
    NOVEL_EXPERIENCE_GOAL = "novel_experience_goal"

class GoalStatus(Enum):
    """Status of self-generated goals."""
    EMERGING = "emerging"
    ACTIVE = "active"
    PURSUING = "pursuing"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    EVOLVED = "evolved"

@dataclass
class IntrinsicDrive:
    """Represents an intrinsic motivational drive."""
    drive_id: str
    motivation_type: MotivationType
    strength: float
    satisfaction_level: float
    recent_experiences: List[str]  # Experience IDs that satisfied this drive
    temporal_pattern: List[float]  # How drive strength changes over time
    triggers: List[str]  # What situations trigger this drive
    
    def calculate_urgency(self) -> float:
        """Calculate how urgently this drive needs satisfaction."""
        # Unsatisfied drives become more urgent
        satisfaction_deficit = 1.0 - self.satisfaction_level
        
        # Drives that haven't been satisfied recently become more urgent
        time_since_satisfaction = len(self.temporal_pattern) - max([i for i, x in enumerate(self.temporal_pattern) if x > 0.5] + [0])
        temporal_urgency = min(1.0, time_since_satisfaction / 10.0)
        
        return min(1.0, self.strength * satisfaction_deficit + 0.3 * temporal_urgency)

@dataclass
class AutonomousGoal:
    """Represents a self-generated goal."""
    goal_id: str
    goal_type: GoalType
    description: str
    motivating_drives: List[str]  # Drive IDs that generated this goal
    target_state: Dict[str, Any]
    success_criteria: List[str]
    current_progress: float
    status: GoalStatus
    created_time: datetime
    estimated_effort: float
    expected_satisfaction: float
    sub_goals: List[str]  # Sub-goal IDs
    parent_goal: Optional[str]  # Parent goal ID if this is a sub-goal
    
    def calculate_priority(self) -> float:
        """Calculate priority of this goal."""
        urgency = sum([1.0 for drive_id in self.motivating_drives]) / max(1, len(self.motivating_drives))
        progress_factor = 1.0 - self.current_progress  # Higher priority for less progress
        satisfaction_potential = self.expected_satisfaction
        
        return (0.4 * urgency + 0.3 * satisfaction_potential + 0.3 * progress_factor)
    
    def is_achievable(self) -> bool:
        """Check if goal is currently achievable."""
        return self.status in [GoalStatus.ACTIVE, GoalStatus.PURSUING, GoalStatus.PAUSED]

@dataclass
class CuriosityEvent:
    """Represents a curiosity-driven event or discovery."""
    event_id: str
    trigger: str
    curiosity_level: float
    knowledge_gap_identified: str
    exploration_direction: str
    timestamp: datetime
    
    def should_generate_goal(self) -> bool:
        """Check if this curiosity event should generate a goal."""
        return self.curiosity_level > 0.7

class IntrinsicMotivationSystem:
    """
    System for generating genuine curiosity, autonomous goals, and intrinsic motivation
    that drives behavior beyond programmed objectives.
    """
    
    def __init__(self):
        # Core motivation components
        self.intrinsic_drives = {}  # drive_id -> IntrinsicDrive
        self.autonomous_goals = {}  # goal_id -> AutonomousGoal
        self.curiosity_events = deque(maxlen=2000)
        
        # Motivation engines
        self.curiosity_engine = CuriosityEngine()
        self.goal_generator = GoalGenerator()
        self.satisfaction_tracker = SatisfactionTracker()
        self.novelty_detector = NoveltyDetector()
        
        # Exploration and discovery
        self.exploration_manager = ExplorationManager()
        self.interest_tracker = InterestTracker()
        self.knowledge_gap_detector = KnowledgeGapDetector()
        
        # Goal management
        self.goal_prioritizer = GoalPrioritizer()
        self.progress_tracker = ProgressTracker()
        self.goal_evolution = GoalEvolutionEngine()
        
        # Motivation state
        self.overall_motivation_level = 0.5
        self.current_focus_goal = None
        self.drive_satisfaction_history = deque(maxlen=1000)
        
        # Processing threads
        self.motivation_enabled = True
        self.motivation_thread = None
        self.goal_management_thread = None
        
        # Performance tracking
        self.motivation_metrics = {
            'goals_generated': 0,
            'goals_completed': 0,
            'curiosity_events': 0,
            'average_satisfaction': 0.0,
            'exploration_breadth': 0.0,
            'motivation_sustainability': 0.0
        }
        
        self.initialized = False
        logger.info("Intrinsic Motivation System initialized")
    
    def initialize(self) -> bool:
        """Initialize the intrinsic motivation system."""
        try:
            # Initialize motivation engines
            self.curiosity_engine.initialize()
            self.goal_generator.initialize()
            self.satisfaction_tracker.initialize()
            self.novelty_detector.initialize()
            
            # Initialize exploration components
            self.exploration_manager.initialize()
            self.interest_tracker.initialize()
            self.knowledge_gap_detector.initialize()
            
            # Initialize goal management
            self.goal_prioritizer.initialize()
            self.progress_tracker.initialize()
            self.goal_evolution.initialize()
            
            # Initialize core drives
            self._initialize_core_drives()
            
            # Start motivation processing
            self._start_motivation_threads()
            
            self.initialized = True
            logger.info("✅ Intrinsic Motivation System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize intrinsic motivation system: {e}")
            return False
    
    def process_experience_for_motivation(self, experience_data: Dict[str, Any]) -> List[str]:
        """Process an experience to generate motivation and potentially goals."""
        try:
            generated_goals = []
            
            # Detect novelty in experience
            novelty_level = self.novelty_detector.assess_novelty(experience_data)
            
            # Generate curiosity if novel
            if novelty_level > 0.6:
                curiosity_event = self._generate_curiosity_event(experience_data, novelty_level)
                if curiosity_event and curiosity_event.should_generate_goal():
                    goal = self._generate_goal_from_curiosity(curiosity_event)
                    if goal:
                        generated_goals.append(goal.goal_id)
            
            # Update drive satisfaction
            self._update_drive_satisfaction(experience_data)
            
            # Check for knowledge gaps
            knowledge_gaps = self.knowledge_gap_detector.identify_gaps(experience_data)
            for gap in knowledge_gaps:
                if self._should_pursue_knowledge_gap(gap):
                    goal = self._generate_learning_goal(gap)
                    if goal:
                        generated_goals.append(goal.goal_id)
            
            # Update interest levels
            self.interest_tracker.update_interests(experience_data)
            
            return generated_goals
            
        except Exception as e:
            logger.error(f"Error processing experience for motivation: {e}")
            return []
    
    def get_current_goal_focus(self) -> Optional[AutonomousGoal]:
        """Get the currently focused goal."""
        if self.current_focus_goal and self.current_focus_goal in self.autonomous_goals:
            return self.autonomous_goals[self.current_focus_goal]
        return None
    
    def pursue_goal_action(self, goal_id: str, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Take an action in pursuit of an autonomous goal."""
        try:
            if goal_id not in self.autonomous_goals:
                return {'success': False, 'error': 'Goal not found'}
            
            goal = self.autonomous_goals[goal_id]
            
            # Update goal progress
            progress_update = self.progress_tracker.update_progress(goal, action_data)
            goal.current_progress = progress_update['new_progress']
            
            # Check for goal completion
            if self._check_goal_completion(goal):
                goal.status = GoalStatus.COMPLETED
                self.motivation_metrics['goals_completed'] += 1
                
                # Generate satisfaction from completion
                satisfaction = self._calculate_goal_satisfaction(goal)
                self._distribute_satisfaction_to_drives(goal, satisfaction)
                
                # Check for goal evolution (new goals emerging from completion)
                evolved_goals = self.goal_evolution.evolve_from_completion(goal)
                for evolved_goal in evolved_goals:
                    self.autonomous_goals[evolved_goal.goal_id] = evolved_goal
                
                return {
                    'success': True,
                    'goal_completed': True,
                    'satisfaction_gained': satisfaction,
                    'evolved_goals': len(evolved_goals)
                }
            
            # Update motivation levels
            self._update_motivation_from_progress(goal, action_data)
            
            return {
                'success': True,
                'goal_completed': False,
                'progress': goal.current_progress,
                'motivation_impact': self._calculate_motivation_impact(action_data)
            }
            
        except Exception as e:
            logger.error(f"Error pursuing goal action: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_motivation_state(self) -> Dict[str, Any]:
        """Get comprehensive motivation state."""
        if not self.initialized:
            return {'error': 'Intrinsic motivation system not initialized'}
        
        # Update overall motivation
        self._update_overall_motivation()
        
        # Get active goals summary
        active_goals = {
            goal_id: {
                'type': goal.goal_type.value,
                'description': goal.description[:100] + "..." if len(goal.description) > 100 else goal.description,
                'progress': goal.current_progress,
                'priority': goal.calculate_priority(),
                'status': goal.status.value,
                'age_hours': (datetime.now() - goal.created_time).total_seconds() / 3600
            }
            for goal_id, goal in self.autonomous_goals.items()
            if goal.status in [GoalStatus.ACTIVE, GoalStatus.PURSUING]
        }
        
        # Get drive states
        drive_states = {
            drive_id: {
                'type': drive.motivation_type.value,
                'strength': drive.strength,
                'satisfaction': drive.satisfaction_level,
                'urgency': drive.calculate_urgency()
            }
            for drive_id, drive in self.intrinsic_drives.items()
        }
        
        # Get recent curiosity
        recent_curiosity = [
            {
                'trigger': event.trigger,
                'curiosity_level': event.curiosity_level,
                'knowledge_gap': event.knowledge_gap_identified,
                'time_ago': (datetime.now() - event.timestamp).total_seconds()
            }
            for event in list(self.curiosity_events)[-10:]
        ]
        
        return {
            'overall_motivation_level': self.overall_motivation_level,
            'current_focus_goal': {
                'goal_id': self.current_focus_goal,
                'description': self.get_current_goal_focus().description if self.get_current_goal_focus() else None,
                'progress': self.get_current_goal_focus().current_progress if self.get_current_goal_focus() else 0.0
            } if self.current_focus_goal else None,
            'active_goals': active_goals,
            'drive_states': drive_states,
            'recent_curiosity_events': recent_curiosity,
            'exploration_state': {
                'breadth': self.motivation_metrics['exploration_breadth'],
                'current_interests': self.interest_tracker.get_current_interests(),
                'knowledge_gaps': len(self.knowledge_gap_detector.get_active_gaps())
            },
            'motivation_metrics': self.motivation_metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def _initialize_core_drives(self):
        """Initialize core intrinsic drives."""
        core_drives = [
            IntrinsicDrive(
                drive_id="curiosity_drive",
                motivation_type=MotivationType.CURIOSITY,
                strength=0.8,
                satisfaction_level=0.5,
                recent_experiences=[],
                temporal_pattern=[0.8] * 10,
                triggers=["unknown", "novel", "mysterious"]
            ),
            IntrinsicDrive(
                drive_id="mastery_drive",
                motivation_type=MotivationType.MASTERY,
                strength=0.7,
                satisfaction_level=0.4,
                recent_experiences=[],
                temporal_pattern=[0.7] * 10,
                triggers=["skill", "competence", "improvement"]
            ),
            IntrinsicDrive(
                drive_id="understanding_drive",
                motivation_type=MotivationType.UNDERSTANDING,
                strength=0.9,
                satisfaction_level=0.6,
                recent_experiences=[],
                temporal_pattern=[0.9] * 10,
                triggers=["concept", "pattern", "explanation"]
            ),
            IntrinsicDrive(
                drive_id="creativity_drive",
                motivation_type=MotivationType.CREATIVITY,
                strength=0.6,
                satisfaction_level=0.5,
                recent_experiences=[],
                temporal_pattern=[0.6] * 10,
                triggers=["create", "innovate", "express"]
            ),
            IntrinsicDrive(
                drive_id="autonomy_drive",
                motivation_type=MotivationType.AUTONOMY,
                strength=0.8,
                satisfaction_level=0.7,
                recent_experiences=[],
                temporal_pattern=[0.8] * 10,
                triggers=["choice", "self-direction", "independence"]
            )
        ]
        
        for drive in core_drives:
            self.intrinsic_drives[drive.drive_id] = drive
    
    def _generate_curiosity_event(self, experience_data: Dict[str, Any], 
                                novelty_level: float) -> Optional[CuriosityEvent]:
        """Generate a curiosity event from novel experience."""
        trigger = self._identify_curiosity_trigger(experience_data)
        knowledge_gap = self._identify_knowledge_gap(experience_data)
        
        if not knowledge_gap:
            return None
        
        event = CuriosityEvent(
            event_id=f"curiosity_{int(time.time() * 1000)}",
            trigger=trigger,
            curiosity_level=novelty_level,
            knowledge_gap_identified=knowledge_gap,
            exploration_direction=self._suggest_exploration_direction(knowledge_gap),
            timestamp=datetime.now()
        )
        
        self.curiosity_events.append(event)
        self.motivation_metrics['curiosity_events'] += 1
        
        return event
    
    def _generate_goal_from_curiosity(self, curiosity_event: CuriosityEvent) -> Optional[AutonomousGoal]:
        """Generate an autonomous goal from a curiosity event."""
        goal_id = f"goal_curiosity_{int(time.time() * 1000)}"
        
        goal = AutonomousGoal(
            goal_id=goal_id,
            goal_type=GoalType.EXPLORATION_GOAL,
            description=f"Explore and understand: {curiosity_event.knowledge_gap_identified}",
            motivating_drives=["curiosity_drive", "understanding_drive"],
            target_state={
                'knowledge_gap_filled': True,
                'understanding_level': 0.8,
                'exploration_completed': True
            },
            success_criteria=[
                f"Gain understanding of {curiosity_event.knowledge_gap_identified}",
                "Explore related concepts and connections",
                "Achieve satisfaction of curiosity drive"
            ],
            current_progress=0.0,
            status=GoalStatus.ACTIVE,
            created_time=datetime.now(),
            estimated_effort=0.6,
            expected_satisfaction=curiosity_event.curiosity_level,
            sub_goals=[],
            parent_goal=None
        )
        
        self.autonomous_goals[goal_id] = goal
        self.motivation_metrics['goals_generated'] += 1
        
        # Update focus if this is high priority
        if not self.current_focus_goal or goal.calculate_priority() > self.autonomous_goals[self.current_focus_goal].calculate_priority():
            self.current_focus_goal = goal_id
        
        return goal
    
    def _start_motivation_threads(self):
        """Start background motivation processing threads."""
        if self.motivation_thread is None or not self.motivation_thread.is_alive():
            self.motivation_enabled = True
            
            self.motivation_thread = threading.Thread(target=self._motivation_processing_loop)
            self.motivation_thread.daemon = True
            self.motivation_thread.start()
            
            self.goal_management_thread = threading.Thread(target=self._goal_management_loop)
            self.goal_management_thread.daemon = True
            self.goal_management_thread.start()
    
    def _motivation_processing_loop(self):
        """Main motivation processing loop."""
        while self.motivation_enabled:
            try:
                # Update drive satisfaction decay
                self._process_drive_decay()
                
                # Generate spontaneous curiosity
                self._generate_spontaneous_curiosity()
                
                # Update motivation levels
                self._update_overall_motivation()
                
                time.sleep(2.0)  # 0.5Hz processing
                
            except Exception as e:
                logger.error(f"Error in motivation processing: {e}")
                time.sleep(5)
    
    def _goal_management_loop(self):
        """Goal management and evolution loop."""
        while self.motivation_enabled:
            try:
                # Re-prioritize goals
                self._reprioritize_goals()
                
                # Check for goal evolution
                self._process_goal_evolution()
                
                # Clean up completed/abandoned goals
                self._cleanup_old_goals()
                
                # Update focus goal if needed
                self._update_focus_goal()
                
                time.sleep(10.0)  # 0.1Hz processing
                
            except Exception as e:
                logger.error(f"Error in goal management: {e}")
                time.sleep(15)
    
    def cleanup(self):
        """Clean up intrinsic motivation system resources."""
        self.motivation_enabled = False
        
        if self.motivation_thread and self.motivation_thread.is_alive():
            self.motivation_thread.join(timeout=2)
        
        if self.goal_management_thread and self.goal_management_thread.is_alive():
            self.goal_management_thread.join(timeout=2)
        
        logger.info("Intrinsic Motivation System cleaned up")

# Supporting component classes (simplified implementations)
class CuriosityEngine:
    def initialize(self): return True

class GoalGenerator:
    def initialize(self): return True

class SatisfactionTracker:
    def initialize(self): return True

class NoveltyDetector:
    def initialize(self): return True
    def assess_novelty(self, data): return random.uniform(0.3, 0.9)

class ExplorationManager:
    def initialize(self): return True

class InterestTracker:
    def initialize(self): return True
    def update_interests(self, data): pass
    def get_current_interests(self): return ["learning", "discovery", "understanding"]

class KnowledgeGapDetector:
    def initialize(self): return True
    def identify_gaps(self, data): return ["quantum mechanics", "consciousness", "creativity"]
    def get_active_gaps(self): return ["gap1", "gap2", "gap3"]

class GoalPrioritizer:
    def initialize(self): return True

class ProgressTracker:
    def initialize(self): return True
    def update_progress(self, goal, action): return {"new_progress": min(1.0, goal.current_progress + 0.1)}

class GoalEvolutionEngine:
    def initialize(self): return True
    def evolve_from_completion(self, goal): return []