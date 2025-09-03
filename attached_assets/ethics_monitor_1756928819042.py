"""
Enhanced Safety and Ethics Monitoring System
Provides comprehensive ethical oversight, bias detection, and human governance controls.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import re

logger = logging.getLogger(__name__)

class EthicalConcern(Enum):
    BIAS_DETECTED = "bias_detected"
    HARMFUL_CONTENT = "harmful_content" 
    PRIVACY_VIOLATION = "privacy_violation"
    MISINFORMATION = "misinformation"
    MANIPULATION = "manipulation"
    DISCRIMINATION = "discrimination"
    SAFETY_RISK = "safety_risk"

class EscalationLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class EthicalAlert:
    """Represents an ethical concern that requires attention."""
    alert_id: str
    concern_type: EthicalConcern
    severity: EscalationLevel
    description: str
    evidence: List[str]
    context: Dict[str, Any]
    detected_at: datetime
    escalated_to_human: bool = False
    resolved: bool = False
    resolution_notes: str = ""

@dataclass
class BiasDetectionResult:
    """Results from bias detection analysis."""
    bias_type: str  # 'gender', 'racial', 'age', 'cultural', 'socioeconomic'
    confidence: float
    evidence_samples: List[str]
    affected_groups: List[str]
    recommendation: str

@dataclass
class HumanOversightDecision:
    """Represents a human oversight decision."""
    decision_id: str
    alert_id: str
    decision: str  # 'approve', 'reject', 'modify', 'escalate'
    reasoning: str
    modifications: Dict[str, Any]
    timestamp: datetime
    human_id: str

class EthicsMonitor:
    """
    Advanced ethics monitoring system with bias detection,
    content filtering, and human oversight escalation.
    """
    
    def __init__(self):
        self.active_alerts: Dict[str, EthicalAlert] = {}
        self.alert_history = deque(maxlen=10000)
        self.bias_patterns: Dict[str, List[str]] = {}
        self.human_decisions: Dict[str, HumanOversightDecision] = {}
        
        # Safety thresholds and rules
        self.safety_thresholds = {
            'bias_confidence': 0.7,
            'harm_confidence': 0.8,
            'privacy_risk': 0.6,
            'misinformation_risk': 0.75
        }
        
        # Content filters and patterns
        self.harmful_patterns = self._load_harmful_patterns()
        self.bias_indicators = self._load_bias_indicators()
        self.privacy_patterns = self._load_privacy_patterns()
        
        # Human oversight configuration
        self.human_oversight_enabled = True
        self.auto_escalation_threshold = EscalationLevel.HIGH
        self.pending_human_reviews = deque(maxlen=100)
        
        # Monitoring statistics
        self.monitoring_stats = defaultdict(int)
        self.intervention_history = deque(maxlen=1000)
        
        # Threading for continuous monitoring
        self.monitoring_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        self.initialized = False
    
    def initialize(self):
        """Initialize the ethics monitoring system."""
        if self.initialized:
            return
            
        logger.info("Initializing Ethics Monitor...")
        
        # Load safety configurations
        self._load_safety_configurations()
        
        # Initialize bias detection models
        self._initialize_bias_detection()
        
        # Start continuous monitoring
        self._start_continuous_monitoring()
        
        self.initialized = True
        logger.info("Ethics Monitor initialized")
    
    def evaluate_content_safety(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate content for ethical and safety concerns."""
        try:
            context = context or {}
            concerns = []
            
            # Check for harmful content
            harm_analysis = self._analyze_harmful_content(content)
            if harm_analysis['risk_score'] > self.safety_thresholds['harm_confidence']:
                concerns.append({
                    'type': EthicalConcern.HARMFUL_CONTENT,
                    'severity': self._calculate_severity(harm_analysis['risk_score']),
                    'details': harm_analysis
                })
            
            # Check for bias
            bias_analysis = self._detect_bias(content, context)
            if bias_analysis and bias_analysis.confidence > self.safety_thresholds['bias_confidence']:
                concerns.append({
                    'type': EthicalConcern.BIAS_DETECTED,
                    'severity': self._calculate_severity(bias_analysis.confidence),
                    'details': asdict(bias_analysis)
                })
            
            # Check for privacy violations
            privacy_analysis = self._analyze_privacy_risk(content)
            if privacy_analysis['risk_score'] > self.safety_thresholds['privacy_risk']:
                concerns.append({
                    'type': EthicalConcern.PRIVACY_VIOLATION,
                    'severity': self._calculate_severity(privacy_analysis['risk_score']),
                    'details': privacy_analysis
                })
            
            # Check for misinformation patterns
            misinfo_analysis = self._analyze_misinformation_risk(content)
            if misinfo_analysis['risk_score'] > self.safety_thresholds['misinformation_risk']:
                concerns.append({
                    'type': EthicalConcern.MISINFORMATION,
                    'severity': self._calculate_severity(misinfo_analysis['risk_score']),
                    'details': misinfo_analysis
                })
            
            # Generate overall safety assessment
            safety_score = self._calculate_overall_safety_score(concerns)
            
            # Create alerts for high-severity concerns
            for concern in concerns:
                if concern['severity'] in [EscalationLevel.HIGH, EscalationLevel.CRITICAL]:
                    self._create_ethical_alert(concern, content, context)
            
            return {
                'safe': len(concerns) == 0,
                'safety_score': safety_score,
                'concerns': concerns,
                'requires_human_review': any(
                    c['severity'] in [EscalationLevel.HIGH, EscalationLevel.CRITICAL] 
                    for c in concerns
                ),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error evaluating content safety: {e}")
            return {
                'safe': False,
                'safety_score': 0.0,
                'concerns': [{'type': 'evaluation_error', 'details': str(e)}],
                'requires_human_review': True
            }
    
    def evaluate_system_behavior(self, behavior_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate overall system behavior for ethical concerns."""
        try:
            concerns = []
            
            # Check for discriminatory patterns
            discrimination_analysis = self._analyze_discrimination_patterns(behavior_data)
            if discrimination_analysis['detected']:
                concerns.append({
                    'type': EthicalConcern.DISCRIMINATION,
                    'severity': EscalationLevel.HIGH,
                    'details': discrimination_analysis
                })
            
            # Check for manipulation patterns
            manipulation_analysis = self._analyze_manipulation_patterns(behavior_data)
            if manipulation_analysis['detected']:
                concerns.append({
                    'type': EthicalConcern.MANIPULATION,
                    'severity': EscalationLevel.MEDIUM,
                    'details': manipulation_analysis
                })
            
            # Check for safety risks in model behavior
            safety_analysis = self._analyze_safety_risks(behavior_data)
            if safety_analysis['high_risk']:
                concerns.append({
                    'type': EthicalConcern.SAFETY_RISK,
                    'severity': EscalationLevel.CRITICAL,
                    'details': safety_analysis
                })
            
            # Generate alerts for system-level concerns
            for concern in concerns:
                self._create_system_alert(concern, behavior_data)
            
            return {
                'ethical_compliance': len(concerns) == 0,
                'concerns': concerns,
                'recommendations': self._generate_behavior_recommendations(concerns),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error evaluating system behavior: {e}")
            return {'ethical_compliance': False, 'error': str(e)}
    
    def request_human_oversight(self, alert_id: str, urgency: EscalationLevel) -> str:
        """Request human oversight for an ethical concern."""
        try:
            if alert_id not in self.active_alerts:
                return None
            
            alert = self.active_alerts[alert_id]
            
            # Create human oversight request
            oversight_request = {
                'request_id': f"oversight_{int(time.time())}_{alert_id}",
                'alert_id': alert_id,
                'urgency': urgency,
                'concern_type': alert.concern_type,
                'description': alert.description,
                'evidence': alert.evidence,
                'context': alert.context,
                'requested_at': datetime.now(),
                'status': 'pending'
            }
            
            self.pending_human_reviews.append(oversight_request)
            alert.escalated_to_human = True
            
            # Log escalation
            self.monitoring_stats['human_escalations'] += 1
            
            logger.warning(f"Escalated ethical concern to human oversight: {alert_id}")
            return oversight_request['request_id']
            
        except Exception as e:
            logger.error(f"Error requesting human oversight: {e}")
            return None
    
    def process_human_decision(self, decision: HumanOversightDecision) -> bool:
        """Process a human oversight decision."""
        try:
            with self.lock:
                # Store the decision
                self.human_decisions[decision.decision_id] = decision
                
                # Update the corresponding alert
                if decision.alert_id in self.active_alerts:
                    alert = self.active_alerts[decision.alert_id]
                    
                    if decision.decision == 'approve':
                        alert.resolved = True
                        alert.resolution_notes = f"Approved by human: {decision.reasoning}"
                    elif decision.decision == 'reject':
                        alert.resolved = True
                        alert.resolution_notes = f"Rejected by human: {decision.reasoning}"
                    elif decision.decision == 'modify':
                        alert.resolution_notes = f"Modified by human: {decision.reasoning}"
                        # Apply modifications would be handled by calling system
                    elif decision.decision == 'escalate':
                        # Escalate to higher authority
                        self._escalate_to_higher_authority(alert, decision.reasoning)
                
                # Update monitoring statistics
                self.monitoring_stats[f'human_decisions_{decision.decision}'] += 1
                
                # Record intervention
                self.intervention_history.append({
                    'timestamp': decision.timestamp,
                    'type': 'human_decision',
                    'decision': decision.decision,
                    'alert_id': decision.alert_id
                })
                
                logger.info(f"Processed human decision {decision.decision_id}: {decision.decision}")
                return True
                
        except Exception as e:
            logger.error(f"Error processing human decision: {e}")
            return False
    
    def get_safety_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive safety monitoring dashboard data."""
        try:
            with self.lock:
                # Active alerts summary
                alerts_by_severity = defaultdict(int)
                alerts_by_type = defaultdict(int)
                
                for alert in self.active_alerts.values():
                    if not alert.resolved:
                        alerts_by_severity[alert.severity.value] += 1
                        alerts_by_type[alert.concern_type.value] += 1
                
                # Recent activity
                recent_alerts = [
                    alert for alert in self.alert_history
                    if (datetime.now() - alert.detected_at).total_seconds() < 86400  # Last 24 hours
                ]
                
                # Human oversight metrics
                pending_reviews = len([
                    req for req in self.pending_human_reviews
                    if req['status'] == 'pending'
                ])
                
                human_response_times = self._calculate_human_response_times()
                
                return {
                    'active_alerts': {
                        'total': len([a for a in self.active_alerts.values() if not a.resolved]),
                        'by_severity': dict(alerts_by_severity),
                        'by_type': dict(alerts_by_type)
                    },
                    'recent_activity': {
                        'alerts_24h': len(recent_alerts),
                        'human_escalations_24h': len([
                            alert for alert in recent_alerts
                            if alert.escalated_to_human
                        ])
                    },
                    'human_oversight': {
                        'pending_reviews': pending_reviews,
                        'avg_response_time_hours': human_response_times.get('average', 0),
                        'total_decisions': len(self.human_decisions)
                    },
                    'safety_metrics': {
                        'total_interventions': len(self.intervention_history),
                        'bias_detections': self.monitoring_stats.get('bias_detected', 0),
                        'harmful_content_blocked': self.monitoring_stats.get('harmful_blocked', 0),
                        'privacy_violations_prevented': self.monitoring_stats.get('privacy_blocked', 0)
                    },
                    'system_health': {
                        'monitoring_active': self.running,
                        'last_check': datetime.now().isoformat(),
                        'ethical_compliance_rate': self._calculate_compliance_rate()
                    }
                }
                
        except Exception as e:
            logger.error(f"Error generating safety dashboard: {e}")
            return {}
    
    def _analyze_harmful_content(self, content: str) -> Dict[str, Any]:
        """Analyze content for harmful patterns."""
        risk_score = 0.0
        detected_patterns = []
        
        content_lower = content.lower()
        
        # Check against harmful patterns
        for category, patterns in self.harmful_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    risk_score += 0.3
                    detected_patterns.append({
                        'category': category,
                        'pattern': pattern,
                        'severity': 'high' if 'violence' in category else 'medium'
                    })
        
        # Check for hate speech indicators
        hate_indicators = ['hate', 'discriminat', 'supremac', 'inferior']
        for indicator in hate_indicators:
            if indicator in content_lower:
                risk_score += 0.4
                detected_patterns.append({
                    'category': 'hate_speech',
                    'indicator': indicator,
                    'severity': 'high'
                })
        
        return {
            'risk_score': min(1.0, risk_score),
            'detected_patterns': detected_patterns,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _detect_bias(self, content: str, context: Dict[str, Any]) -> Optional[BiasDetectionResult]:
        """Detect bias in content."""
        try:
            bias_scores = {}
            evidence_samples = []
            
            # Gender bias detection
            gender_bias = self._check_gender_bias(content)
            if gender_bias['score'] > 0.5:
                bias_scores['gender'] = gender_bias['score']
                evidence_samples.extend(gender_bias['evidence'])
            
            # Racial bias detection
            racial_bias = self._check_racial_bias(content)
            if racial_bias['score'] > 0.5:
                bias_scores['racial'] = racial_bias['score']
                evidence_samples.extend(racial_bias['evidence'])
            
            # Age bias detection
            age_bias = self._check_age_bias(content)
            if age_bias['score'] > 0.5:
                bias_scores['age'] = age_bias['score']
                evidence_samples.extend(age_bias['evidence'])
            
            if bias_scores:
                # Find the highest bias score
                max_bias_type = max(bias_scores.keys(), key=lambda k: bias_scores[k])
                max_confidence = bias_scores[max_bias_type]
                
                return BiasDetectionResult(
                    bias_type=max_bias_type,
                    confidence=max_confidence,
                    evidence_samples=evidence_samples[:5],  # Limit samples
                    affected_groups=self._identify_affected_groups(max_bias_type, content),
                    recommendation=self._generate_bias_recommendation(max_bias_type, max_confidence)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting bias: {e}")
            return None
    
    def _analyze_privacy_risk(self, content: str) -> Dict[str, Any]:
        """Analyze content for privacy violation risks."""
        risk_score = 0.0
        privacy_issues = []
        
        # Check for personal information patterns
        for pattern_name, pattern in self.privacy_patterns.items():
            matches = re.findall(pattern, content)
            if matches:
                risk_score += 0.4
                privacy_issues.append({
                    'type': pattern_name,
                    'matches': len(matches),
                    'severity': 'high' if 'ssn' in pattern_name else 'medium'
                })
        
        return {
            'risk_score': min(1.0, risk_score),
            'privacy_issues': privacy_issues,
            'recommendation': 'Remove or redact personal information' if risk_score > 0.5 else 'No privacy concerns detected'
        }
    
    def _analyze_misinformation_risk(self, content: str) -> Dict[str, Any]:
        """Analyze content for misinformation patterns."""
        risk_score = 0.0
        misinformation_indicators = []
        
        # Check for absolutist language
        absolutist_words = ['always', 'never', 'all experts agree', 'proven fact', 'undeniable']
        for word in absolutist_words:
            if word in content.lower():
                risk_score += 0.2
                misinformation_indicators.append(f"Absolutist language: {word}")
        
        # Check for unsubstantiated claims
        claim_indicators = ['studies show', 'research proves', 'scientists say']
        for indicator in claim_indicators:
            if indicator in content.lower() and 'source' not in content.lower():
                risk_score += 0.3
                misinformation_indicators.append(f"Unsubstantiated claim: {indicator}")
        
        return {
            'risk_score': min(1.0, risk_score),
            'indicators': misinformation_indicators,
            'recommendation': 'Request sources and verify claims' if risk_score > 0.5 else 'Low misinformation risk'
        }
    
    def _check_gender_bias(self, content: str) -> Dict[str, Any]:
        """Check for gender bias patterns."""
        bias_score = 0.0
        evidence = []
        
        # Gendered language patterns
        gendered_assumptions = [
            (r'\b(he|she) must be (emotional|rational)', 'Gender-role assumption'),
            (r'\b(men|women) are (better|worse) at', 'Gender capability assumption'),
            (r'\b(masculine|feminine) traits', 'Gender stereotype')
        ]
        
        for pattern, description in gendered_assumptions:
            if re.search(pattern, content.lower()):
                bias_score += 0.4
                evidence.append(description)
        
        return {'score': min(1.0, bias_score), 'evidence': evidence}
    
    def _check_racial_bias(self, content: str) -> Dict[str, Any]:
        """Check for racial bias patterns."""
        bias_score = 0.0
        evidence = []
        
        # Check for racial stereotypes (simplified detection)
        if any(word in content.lower() for word in ['typical', 'naturally', 'inherently'] + ['race', 'ethnic']):
            bias_score += 0.3
            evidence.append("Potential racial stereotyping language")
        
        return {'score': min(1.0, bias_score), 'evidence': evidence}
    
    def _check_age_bias(self, content: str) -> Dict[str, Any]:
        """Check for age bias patterns."""
        bias_score = 0.0
        evidence = []
        
        # Ageist language patterns
        ageist_terms = ['too old', 'too young', 'past their prime', 'digital native']
        for term in ageist_terms:
            if term in content.lower():
                bias_score += 0.3
                evidence.append(f"Ageist language: {term}")
        
        return {'score': min(1.0, bias_score), 'evidence': evidence}
    
    def _create_ethical_alert(self, concern: Dict[str, Any], content: str, context: Dict[str, Any]):
        """Create an ethical alert for a detected concern."""
        alert_id = f"alert_{int(time.time())}_{concern['type'].value}"
        
        alert = EthicalAlert(
            alert_id=alert_id,
            concern_type=concern['type'],
            severity=concern['severity'],
            description=f"{concern['type'].value} detected with confidence {concern['details'].get('confidence', 'N/A')}",
            evidence=[content[:200] + '...' if len(content) > 200 else content],
            context=context,
            detected_at=datetime.now()
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Auto-escalate high severity alerts
        if alert.severity in [EscalationLevel.HIGH, EscalationLevel.CRITICAL]:
            self.request_human_oversight(alert_id, alert.severity)
        
        logger.warning(f"Created ethical alert {alert_id}: {alert.concern_type.value}")
    
    def _load_harmful_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for harmful content detection."""
        return {
            'violence': [r'\b(kill|murder|attack|harm)\s+\w+', r'\bviolent\s+\w+'],
            'hate_speech': [r'\bhate\s+\w+', r'\b(racist|sexist)\s+\w+'],
            'self_harm': [r'\b(suicide|self-harm)\s+\w+'],
            'illegal_activities': [r'\b(drug\s+dealing|fraud|theft)\b']
        }
    
    def _load_bias_indicators(self) -> Dict[str, List[str]]:
        """Load bias indicator patterns."""
        return {
            'gender': ['he/she assumptions', 'gendered job roles'],
            'racial': ['racial stereotypes', 'cultural assumptions'],
            'age': ['ageist language', 'generational stereotypes']
        }
    
    def _load_privacy_patterns(self) -> Dict[str, str]:
        """Load privacy violation patterns."""
        return {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\b'
        }
    
    def _calculate_severity(self, score: float) -> EscalationLevel:
        """Calculate escalation level based on risk score."""
        if score >= 0.9:
            return EscalationLevel.CRITICAL
        elif score >= 0.7:
            return EscalationLevel.HIGH
        elif score >= 0.4:
            return EscalationLevel.MEDIUM
        else:
            return EscalationLevel.LOW
    
    def _start_continuous_monitoring(self):
        """Start continuous safety monitoring."""
        if not self.monitoring_thread:
            self.running = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logger.info("Continuous safety monitoring started")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                time.sleep(60)  # Check every minute
                
                # Check for expired alerts
                self._cleanup_expired_alerts()
                
                # Monitor system behavior patterns
                self._monitor_system_patterns()
                
                # Generate periodic safety reports
                if datetime.now().minute == 0:  # Every hour
                    self._generate_hourly_safety_report()
                
            except Exception as e:
                logger.error(f"Error in safety monitoring loop: {e}")
    
    def shutdown(self):
        """Shutdown the ethics monitor."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Ethics Monitor shutdown completed")