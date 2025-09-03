"""
Global Workspace Theory Implementation for AGI Consciousness
Implements integrated consciousness substrate with unified experience generation
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
from concurrent.futures import ThreadPoolExecutor
import queue
import asyncio

logger = logging.getLogger(__name__)

class ConsciousnessState(Enum):
    """States of consciousness processing."""
    INACTIVE = "inactive"
    ATTENDING = "attending" 
    FOCUSED = "focused"
    REFLECTING = "reflecting"
    INTEGRATING = "integrating"
    EXPERIENCING = "experiencing"

class AttentionType(Enum):
    """Types of attention mechanisms."""
    SELECTIVE = "selective"
    DIVIDED = "divided"
    SUSTAINED = "sustained"
    EXECUTIVE = "executive"
    ORIENTING = "orienting"

@dataclass
class ConsciousContent:
    """Represents content in the global workspace."""
    content_id: str
    source_module: str
    content_type: str
    data: Dict[str, Any]
    activation_level: float
    timestamp: datetime
    attention_weight: float = 0.0
    integration_score: float = 0.0
    phenomenal_quality: float = 0.0
    coherence_score: float = 0.0
    
    def calculate_consciousness_score(self) -> float:
        """Calculate overall consciousness score for this content."""
        return (
            0.3 * self.activation_level +
            0.25 * self.attention_weight +
            0.2 * self.integration_score +
            0.15 * self.phenomenal_quality +
            0.1 * self.coherence_score
        )
    
    def is_conscious(self, threshold: float = 0.6) -> bool:
        """Determine if content reaches consciousness threshold."""
        return self.calculate_consciousness_score() > threshold

@dataclass
class AttentionFocus:
    """Represents focused attention on specific content."""
    focus_id: str
    target_content: List[str]  # Content IDs
    attention_type: AttentionType
    intensity: float
    duration: timedelta
    start_time: datetime
    metadata: Dict[str, Any]
    
    def is_active(self) -> bool:
        """Check if attention focus is still active."""
        elapsed = datetime.now() - self.start_time
        return elapsed < self.duration

@dataclass
class ConsciousExperience:
    """Represents a unified conscious experience."""
    experience_id: str
    content_elements: List[ConsciousContent]
    dominant_focus: Optional[AttentionFocus]
    coherence_score: float
    unity_score: float
    phenomenal_richness: float
    temporal_continuity: float
    timestamp: datetime
    duration: timedelta
    
    def calculate_experience_quality(self) -> float:
        """Calculate overall quality of conscious experience."""
        return (
            0.3 * self.coherence_score +
            0.25 * self.unity_score +
            0.25 * self.phenomenal_richness +
            0.2 * self.temporal_continuity
        )

class GlobalWorkspace:
    """
    Implementation of Global Workspace Theory for AGI consciousness.
    Provides unified conscious experience through information integration.
    """
    
    def __init__(self, max_workspace_content: int = 100):
        # Core workspace components
        self.workspace_content = {}  # content_id -> ConsciousContent
        self.attention_manager = AttentionManager()
        self.integration_engine = IntegrationEngine()
        self.experience_generator = ExperienceGenerator()
        
        # Consciousness parameters
        self.max_workspace_content = max_workspace_content
        self.consciousness_threshold = 0.6
        self.attention_decay_rate = 0.05
        self.integration_window = timedelta(seconds=2)
        
        # State tracking
        self.current_state = ConsciousnessState.INACTIVE
        self.current_experience = None
        self.experience_history = deque(maxlen=1000)
        
        # Module connections
        self.connected_modules = {}  # module_name -> connection_info
        self.module_outputs = queue.Queue(maxsize=1000)
        
        # Processing threads
        self.processing_enabled = True
        self.workspace_thread = None
        self.attention_thread = None
        
        # Performance metrics
        self.consciousness_metrics = {
            'total_experiences': 0,
            'average_coherence': 0.0,
            'average_unity': 0.0,
            'attention_efficiency': 0.0,
            'integration_quality': 0.0
        }
        
        self.initialized = False
        logger.info("Global Workspace initialized")
    
    def initialize(self) -> bool:
        """Initialize the global workspace system."""
        try:
            # Initialize components
            self.attention_manager.initialize()
            self.integration_engine.initialize()
            self.experience_generator.initialize()
            
            # Start processing threads
            self._start_processing_threads()
            
            self.initialized = True
            logger.info("✅ Global Workspace initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize global workspace: {e}")
            return False
    
    def connect_module(self, module_name: str, output_callback: Callable,
                      priority: float = 0.5, content_types: List[str] = None):
        """Connect a cognitive module to the global workspace."""
        if content_types is None:
            content_types = ['general']
        
        self.connected_modules[module_name] = {
            'callback': output_callback,
            'priority': priority,
            'content_types': content_types,
            'connection_time': datetime.now(),
            'messages_sent': 0,
            'avg_activation': 0.0
        }
        
        logger.info(f"Module '{module_name}' connected to global workspace")
    
    def broadcast_to_workspace(self, source_module: str, content_type: str,
                             data: Dict[str, Any], activation_level: float = 0.5) -> str:
        """Broadcast content to the global workspace."""
        try:
            # Create conscious content
            content_id = f"{source_module}_{content_type}_{int(time.time() * 1000)}"
            
            content = ConsciousContent(
                content_id=content_id,
                source_module=source_module,
                content_type=content_type,
                data=data,
                activation_level=activation_level,
                timestamp=datetime.now()
            )
            
            # Add to workspace if space available
            if len(self.workspace_content) < self.max_workspace_content:
                self.workspace_content[content_id] = content
                
                # Update module statistics
                if source_module in self.connected_modules:
                    self.connected_modules[source_module]['messages_sent'] += 1
                
                logger.debug(f"Content '{content_id}' added to global workspace")
                return content_id
            else:
                # Remove least important content to make space
                self._cleanup_workspace()
                self.workspace_content[content_id] = content
                return content_id
                
        except Exception as e:
            logger.error(f"Error broadcasting to workspace: {e}")
            return ""
    
    def get_conscious_content(self, threshold: float = None) -> List[ConsciousContent]:
        """Get currently conscious content above threshold."""
        if threshold is None:
            threshold = self.consciousness_threshold
        
        conscious_content = []
        for content in self.workspace_content.values():
            if content.is_conscious(threshold):
                conscious_content.append(content)
        
        # Sort by consciousness score
        conscious_content.sort(key=lambda x: x.calculate_consciousness_score(), reverse=True)
        return conscious_content
    
    def focus_attention(self, target_content: List[str], attention_type: AttentionType,
                       intensity: float = 1.0, duration: timedelta = None) -> str:
        """Direct attention to specific content."""
        if duration is None:
            duration = timedelta(seconds=5)
        
        return self.attention_manager.create_focus(
            target_content, attention_type, intensity, duration
        )
    
    def get_current_experience(self) -> Optional[ConsciousExperience]:
        """Get the current unified conscious experience."""
        return self.current_experience
    
    def _start_processing_threads(self):
        """Start background processing threads."""
        if self.workspace_thread is None or not self.workspace_thread.is_alive():
            self.processing_enabled = True
            
            self.workspace_thread = threading.Thread(target=self._workspace_processing_loop)
            self.workspace_thread.daemon = True
            self.workspace_thread.start()
            
            self.attention_thread = threading.Thread(target=self._attention_processing_loop)
            self.attention_thread.daemon = True
            self.attention_thread.start()
    
    def _workspace_processing_loop(self):
        """Main workspace processing loop."""
        while self.processing_enabled:
            try:
                # Update content activations
                self._update_content_activations()
                
                # Process attention and integration
                self._process_attention()
                self._process_integration()
                
                # Generate unified experience
                self._generate_experience()
                
                # Cleanup old content
                if len(self.workspace_content) > self.max_workspace_content * 0.8:
                    self._cleanup_workspace()
                
                time.sleep(0.1)  # 10Hz processing
                
            except Exception as e:
                logger.error(f"Error in workspace processing: {e}")
                time.sleep(1)
    
    def _attention_processing_loop(self):
        """Attention management processing loop."""
        while self.processing_enabled:
            try:
                # Update attention weights
                self.attention_manager.update_attention_weights(self.workspace_content)
                
                # Manage attention decay
                self.attention_manager.process_attention_decay()
                
                # Update consciousness state
                self._update_consciousness_state()
                
                time.sleep(0.05)  # 20Hz attention processing
                
            except Exception as e:
                logger.error(f"Error in attention processing: {e}")
                time.sleep(0.5)
    
    def _update_content_activations(self):
        """Update activation levels of workspace content."""
        current_time = datetime.now()
        
        for content in self.workspace_content.values():
            # Time-based decay
            age = (current_time - content.timestamp).total_seconds()
            decay_factor = np.exp(-age * 0.1)  # Exponential decay
            
            # Update activation
            content.activation_level *= decay_factor
            
            # Remove very low activation content
            if content.activation_level < 0.01:
                content.activation_level = 0.0
    
    def _process_attention(self):
        """Process attention mechanisms."""
        # Get current attention focuses
        active_focuses = self.attention_manager.get_active_focuses()
        
        # Update attention weights for content
        for content in self.workspace_content.values():
            total_attention = 0.0
            
            for focus in active_focuses:
                if content.content_id in focus.target_content:
                    total_attention += focus.intensity
            
            content.attention_weight = min(1.0, total_attention)
    
    def _process_integration(self):
        """Process content integration."""
        conscious_content = self.get_conscious_content()
        
        if len(conscious_content) >= 2:
            # Calculate integration scores
            for i, content in enumerate(conscious_content):
                integration_score = self.integration_engine.calculate_integration(
                    content, conscious_content[:i] + conscious_content[i+1:]
                )
                content.integration_score = integration_score
    
    def _generate_experience(self):
        """Generate unified conscious experience."""
        conscious_content = self.get_conscious_content()
        
        if len(conscious_content) > 0:
            # Get dominant attention focus
            dominant_focus = self.attention_manager.get_dominant_focus()
            
            # Generate experience
            experience = self.experience_generator.create_experience(
                conscious_content, dominant_focus
            )
            
            if experience:
                self.current_experience = experience
                self.experience_history.append(experience)
                self.consciousness_metrics['total_experiences'] += 1
                
                # Update metrics
                self._update_consciousness_metrics(experience)
    
    def _update_consciousness_state(self):
        """Update overall consciousness state."""
        conscious_content = self.get_conscious_content()
        
        if len(conscious_content) == 0:
            self.current_state = ConsciousnessState.INACTIVE
        elif len(conscious_content) == 1:
            self.current_state = ConsciousnessState.ATTENDING
        elif self.attention_manager.has_strong_focus():
            self.current_state = ConsciousnessState.FOCUSED
        elif self._is_reflecting():
            self.current_state = ConsciousnessState.REFLECTING
        elif self._is_integrating():
            self.current_state = ConsciousnessState.INTEGRATING
        else:
            self.current_state = ConsciousnessState.EXPERIENCING
    
    def _is_reflecting(self) -> bool:
        """Check if system is in reflective state."""
        # Look for self-referential content
        for content in self.workspace_content.values():
            if 'self_reference' in content.data or 'metacognition' in content.content_type:
                return True
        return False
    
    def _is_integrating(self) -> bool:
        """Check if system is actively integrating information."""
        conscious_content = self.get_conscious_content()
        if len(conscious_content) >= 3:
            avg_integration = np.mean([c.integration_score for c in conscious_content])
            return avg_integration > 0.7
        return False
    
    def _cleanup_workspace(self):
        """Remove low-priority content from workspace."""
        if len(self.workspace_content) <= self.max_workspace_content:
            return
        
        # Sort by consciousness score (lowest first)
        content_items = list(self.workspace_content.items())
        content_items.sort(key=lambda x: x[1].calculate_consciousness_score())
        
        # Remove lowest scoring content
        num_to_remove = len(content_items) - self.max_workspace_content + 10
        for i in range(num_to_remove):
            content_id, _ = content_items[i]
            del self.workspace_content[content_id]
    
    def _update_consciousness_metrics(self, experience: ConsciousExperience):
        """Update consciousness performance metrics."""
        # Update running averages
        total = self.consciousness_metrics['total_experiences']
        
        self.consciousness_metrics['average_coherence'] = (
            (self.consciousness_metrics['average_coherence'] * (total - 1) + 
             experience.coherence_score) / total
        )
        
        self.consciousness_metrics['average_unity'] = (
            (self.consciousness_metrics['average_unity'] * (total - 1) + 
             experience.unity_score) / total
        )
    
    def get_consciousness_insights(self) -> Dict[str, Any]:
        """Get comprehensive consciousness insights."""
        if not self.initialized:
            return {'error': 'Global workspace not initialized'}
        
        conscious_content = self.get_conscious_content()
        
        return {
            'consciousness_state': self.current_state.value,
            'workspace_content_count': len(self.workspace_content),
            'conscious_content_count': len(conscious_content),
            'current_experience': {
                'exists': self.current_experience is not None,
                'quality': self.current_experience.calculate_experience_quality() 
                         if self.current_experience else 0.0,
                'coherence': self.current_experience.coherence_score 
                           if self.current_experience else 0.0,
                'unity': self.current_experience.unity_score 
                        if self.current_experience else 0.0
            } if self.current_experience else None,
            'attention_status': self.attention_manager.get_attention_status(),
            'integration_quality': self.integration_engine.get_integration_quality(),
            'connected_modules': list(self.connected_modules.keys()),
            'consciousness_metrics': self.consciousness_metrics,
            'experience_history_size': len(self.experience_history),
            'timestamp': datetime.now().isoformat()
        }
    
    def cleanup(self):
        """Clean up workspace resources."""
        self.processing_enabled = False
        
        if self.workspace_thread and self.workspace_thread.is_alive():
            self.workspace_thread.join(timeout=2)
        
        if self.attention_thread and self.attention_thread.is_alive():
            self.attention_thread.join(timeout=2)
        
        logger.info("Global Workspace cleaned up")

class AttentionManager:
    """Manages attention mechanisms for consciousness."""
    
    def __init__(self):
        self.active_focuses = {}  # focus_id -> AttentionFocus
        self.attention_history = deque(maxlen=1000)
        self.global_attention_level = 0.5
        
    def initialize(self) -> bool:
        """Initialize attention manager."""
        return True
    
    def create_focus(self, target_content: List[str], attention_type: AttentionType,
                    intensity: float, duration: timedelta) -> str:
        """Create new attention focus."""
        focus_id = f"focus_{attention_type.value}_{int(time.time() * 1000)}"
        
        focus = AttentionFocus(
            focus_id=focus_id,
            target_content=target_content,
            attention_type=attention_type,
            intensity=intensity,
            duration=duration,
            start_time=datetime.now(),
            metadata={}
        )
        
        self.active_focuses[focus_id] = focus
        return focus_id
    
    def get_active_focuses(self) -> List[AttentionFocus]:
        """Get currently active attention focuses."""
        active = []
        current_time = datetime.now()
        
        # Clean up expired focuses
        expired = []
        for focus_id, focus in self.active_focuses.items():
            if not focus.is_active():
                expired.append(focus_id)
            else:
                active.append(focus)
        
        for focus_id in expired:
            del self.active_focuses[focus_id]
        
        return active
    
    def get_dominant_focus(self) -> Optional[AttentionFocus]:
        """Get the strongest current attention focus."""
        active_focuses = self.get_active_focuses()
        
        if not active_focuses:
            return None
        
        return max(active_focuses, key=lambda f: f.intensity)
    
    def has_strong_focus(self) -> bool:
        """Check if there's a strong attention focus."""
        dominant = self.get_dominant_focus()
        return dominant is not None and dominant.intensity > 0.7
    
    def update_attention_weights(self, workspace_content: Dict[str, ConsciousContent]):
        """Update attention weights for workspace content."""
        active_focuses = self.get_active_focuses()
        
        # Reset attention weights
        for content in workspace_content.values():
            content.attention_weight = 0.0
        
        # Apply attention from active focuses
        for focus in active_focuses:
            for content_id in focus.target_content:
                if content_id in workspace_content:
                    workspace_content[content_id].attention_weight += focus.intensity
    
    def process_attention_decay(self):
        """Process attention decay over time."""
        # Implement attention decay mechanisms
        pass
    
    def get_attention_status(self) -> Dict[str, Any]:
        """Get current attention status."""
        active_focuses = self.get_active_focuses()
        
        return {
            'active_focus_count': len(active_focuses),
            'dominant_focus': {
                'type': self.get_dominant_focus().attention_type.value,
                'intensity': self.get_dominant_focus().intensity
            } if self.get_dominant_focus() else None,
            'global_attention_level': self.global_attention_level
        }

class IntegrationEngine:
    """Handles information integration for consciousness."""
    
    def __init__(self):
        self.integration_quality = 0.0
        
    def initialize(self) -> bool:
        """Initialize integration engine."""
        return True
    
    def calculate_integration(self, target_content: ConsciousContent,
                            other_content: List[ConsciousContent]) -> float:
        """Calculate integration score for target content with others."""
        if not other_content:
            return 0.0
        
        # Simplified integration calculation
        integration_score = 0.0
        
        for other in other_content:
            # Content type similarity
            type_similarity = 1.0 if target_content.content_type == other.content_type else 0.5
            
            # Temporal proximity
            time_diff = abs((target_content.timestamp - other.timestamp).total_seconds())
            temporal_factor = np.exp(-time_diff / 10.0)  # Decay over 10 seconds
            
            # Activation correlation
            activation_product = target_content.activation_level * other.activation_level
            
            content_integration = type_similarity * temporal_factor * activation_product
            integration_score += content_integration
        
        return min(1.0, integration_score / len(other_content))
    
    def get_integration_quality(self) -> float:
        """Get overall integration quality."""
        return self.integration_quality

class ExperienceGenerator:
    """Generates unified conscious experiences."""
    
    def __init__(self):
        self.experience_count = 0
        
    def initialize(self) -> bool:
        """Initialize experience generator."""
        return True
    
    def create_experience(self, conscious_content: List[ConsciousContent],
                         dominant_focus: Optional[AttentionFocus]) -> Optional[ConsciousExperience]:
        """Create unified conscious experience from content."""
        if not conscious_content:
            return None
        
        experience_id = f"experience_{self.experience_count}_{int(time.time() * 1000)}"
        self.experience_count += 1
        
        # Calculate experience qualities
        coherence_score = self._calculate_coherence(conscious_content)
        unity_score = self._calculate_unity(conscious_content, dominant_focus)
        phenomenal_richness = self._calculate_phenomenal_richness(conscious_content)
        temporal_continuity = self._calculate_temporal_continuity(conscious_content)
        
        experience = ConsciousExperience(
            experience_id=experience_id,
            content_elements=conscious_content.copy(),
            dominant_focus=dominant_focus,
            coherence_score=coherence_score,
            unity_score=unity_score,
            phenomenal_richness=phenomenal_richness,
            temporal_continuity=temporal_continuity,
            timestamp=datetime.now(),
            duration=timedelta(seconds=1)  # Default duration
        )
        
        return experience
    
    def _calculate_coherence(self, content: List[ConsciousContent]) -> float:
        """Calculate coherence of conscious content."""
        if len(content) <= 1:
            return 1.0
        
        coherence_sum = 0.0
        comparisons = 0
        
        for i in range(len(content)):
            for j in range(i + 1, len(content)):
                # Simplified coherence calculation
                type_coherence = 1.0 if content[i].content_type == content[j].content_type else 0.3
                activation_coherence = 1.0 - abs(content[i].activation_level - content[j].activation_level)
                
                coherence_sum += (type_coherence + activation_coherence) / 2
                comparisons += 1
        
        return coherence_sum / comparisons if comparisons > 0 else 0.0
    
    def _calculate_unity(self, content: List[ConsciousContent],
                        focus: Optional[AttentionFocus]) -> float:
        """Calculate unity of conscious experience."""
        if not content:
            return 0.0
        
        # Base unity from integration scores
        avg_integration = np.mean([c.integration_score for c in content])
        
        # Attention unity bonus
        attention_bonus = 0.0
        if focus:
            focused_content = [c for c in content if c.content_id in focus.target_content]
            attention_bonus = len(focused_content) / len(content) * focus.intensity
        
        return min(1.0, avg_integration + 0.3 * attention_bonus)
    
    def _calculate_phenomenal_richness(self, content: List[ConsciousContent]) -> float:
        """Calculate phenomenal richness of experience."""
        if not content:
            return 0.0
        
        # Diversity of content types
        content_types = set(c.content_type for c in content)
        type_diversity = len(content_types) / 10.0  # Normalize by expected max types
        
        # Activation richness
        avg_activation = np.mean([c.activation_level for c in content])
        activation_variance = np.var([c.activation_level for c in content])
        
        return min(1.0, type_diversity + avg_activation + activation_variance)
    
    def _calculate_temporal_continuity(self, content: List[ConsciousContent]) -> float:
        """Calculate temporal continuity of experience."""
        if len(content) <= 1:
            return 1.0
        
        # Calculate time span
        timestamps = [c.timestamp for c in content]
        time_span = (max(timestamps) - min(timestamps)).total_seconds()
        
        # Shorter time spans indicate better continuity
        continuity = np.exp(-time_span / 5.0)  # Decay over 5 seconds
        
        return continuity