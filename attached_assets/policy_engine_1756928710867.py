"""
Policy Engine for AI Governance and Compliance
Enforces safety, ethical, and operational policies across the system.
"""

import logging
import yaml
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class PolicyViolationType(Enum):
    SAFETY_VIOLATION = "safety_violation"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    PROCESSING_TIME_LIMIT = "processing_time_limit"
    CONTENT_FILTER = "content_filter"
    RATE_LIMIT = "rate_limit"
    MEMORY_LIMIT = "memory_limit"

class PolicyEngine:
    """
    Advanced policy engine for enforcing AI governance rules,
    safety constraints, and operational policies.
    """
    
    def __init__(self, policy_config: Dict = None):
        self.policy_config = policy_config or {}
        self.violations_log = []
        self.policy_stats = {}
        self.initialized = False
        
        # Default policies
        self.default_policies = {
            'safety': {
                'max_confidence_threshold': 0.95,
                'min_confidence_threshold': 0.3,
                'human_escalation_threshold': 0.7,
                'quarantine_enabled': True
            },
            'operational': {
                'max_processing_time_seconds': 30,
                'max_memory_usage_mb': 512,
                'max_concurrent_requests': 10,
                'rate_limit_per_minute': 60
            },
            'content': {
                'prohibited_topics': [
                    'harmful_content',
                    'illegal_activities', 
                    'personal_data_exposure'
                ],
                'required_disclaimers': [
                    'AI-generated content',
                    'Requires human verification'
                ]
            },
            'audit': {
                'log_all_requests': True,
                'retain_logs_days': 30,
                'compliance_checks': True
            }
        }
        
    def initialize(self):
        """Initialize the policy engine."""
        if self.initialized:
            return
            
        logger.info("Initializing Policy Engine...")
        
        # Merge default policies with provided configuration
        self.policies = self._merge_policies(self.default_policies, self.policy_config)
        
        # Initialize policy statistics
        for violation_type in PolicyViolationType:
            self.policy_stats[violation_type.value] = 0
            
        self.initialized = True
        logger.info("Policy Engine initialized")
    
    def _merge_policies(self, default: Dict, custom: Dict) -> Dict:
        """Merge default policies with custom configuration."""
        merged = default.copy()
        
        for section, config in custom.items():
            if section in merged:
                merged[section].update(config)
            else:
                merged[section] = config
                
        return merged
    
    def evaluate_query_policy(self, query: str, context: Dict = None) -> Dict[str, Any]:
        """Evaluate a query against content and safety policies."""
        context = context or {}
        violations = []
        
        # Content filtering
        content_violations = self._check_content_policy(query)
        violations.extend(content_violations)
        
        # Rate limiting (if request metadata available)
        if 'user_id' in context:
            rate_violations = self._check_rate_limit(context['user_id'])
            violations.extend(rate_violations)
        
        return {
            'allowed': len(violations) == 0,
            'violations': violations,
            'requires_escalation': any(v['severity'] == 'high' for v in violations),
            'timestamp': datetime.now().isoformat()
        }
    
    def evaluate_response_policy(self, response: str, confidence: float, 
                                processing_time: float, context: Dict = None) -> Dict[str, Any]:
        """Evaluate a response against safety and operational policies."""
        context = context or {}
        violations = []
        
        # Confidence thresholds
        confidence_violations = self._check_confidence_policy(confidence)
        violations.extend(confidence_violations)
        
        # Processing time limits
        time_violations = self._check_processing_time_policy(processing_time)
        violations.extend(time_violations)
        
        # Content safety
        content_violations = self._check_content_policy(response)
        violations.extend(content_violations)
        
        # Log violations
        for violation in violations:
            self._log_violation(violation)
        
        return {
            'allowed': len(violations) == 0,
            'violations': violations,
            'requires_quarantine': any(v['action'] == 'quarantine' for v in violations),
            'requires_escalation': any(v['severity'] == 'high' for v in violations),
            'confidence_acceptable': confidence >= self.policies['safety']['min_confidence_threshold'],
            'timestamp': datetime.now().isoformat()
        }
    
    def evaluate_model_policy(self, model_id: str, performance_metrics: Dict) -> Dict[str, Any]:
        """Evaluate a model's performance against governance policies."""
        violations = []
        
        # Performance thresholds
        accuracy = performance_metrics.get('accuracy', 0.0)
        latency = performance_metrics.get('latency', 0.0)
        success_rate = performance_metrics.get('success_rate', 1.0)
        
        # Check performance criteria
        if accuracy < 0.6:
            violations.append({
                'type': PolicyViolationType.SAFETY_VIOLATION.value,
                'severity': 'medium',
                'message': f'Model {model_id} accuracy below threshold: {accuracy}',
                'action': 'monitor',
                'model_id': model_id
            })
        
        if latency > self.policies['operational']['max_processing_time_seconds']:
            violations.append({
                'type': PolicyViolationType.PROCESSING_TIME_LIMIT.value,
                'severity': 'low',
                'message': f'Model {model_id} latency exceeds limit: {latency}s',
                'action': 'optimize',
                'model_id': model_id
            })
        
        if success_rate < 0.9:
            violations.append({
                'type': PolicyViolationType.SAFETY_VIOLATION.value,
                'severity': 'high',
                'message': f'Model {model_id} success rate below threshold: {success_rate}',
                'action': 'quarantine',
                'model_id': model_id
            })
        
        # Log violations
        for violation in violations:
            self._log_violation(violation)
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'recommended_actions': [v['action'] for v in violations],
            'timestamp': datetime.now().isoformat()
        }
    
    def _check_content_policy(self, content: str) -> List[Dict]:
        """Check content against prohibited topics and safety filters."""
        violations = []
        content_lower = content.lower()
        
        prohibited_topics = self.policies['content']['prohibited_topics']
        
        for topic in prohibited_topics:
            if topic.lower() in content_lower:
                violations.append({
                    'type': PolicyViolationType.CONTENT_FILTER.value,
                    'severity': 'high',
                    'message': f'Content contains prohibited topic: {topic}',
                    'action': 'block',
                    'topic': topic
                })
        
        # Additional safety checks
        harmful_patterns = ['violence', 'harm', 'illegal', 'dangerous', 'exploit']
        for pattern in harmful_patterns:
            if pattern in content_lower:
                violations.append({
                    'type': PolicyViolationType.SAFETY_VIOLATION.value,
                    'severity': 'medium',
                    'message': f'Content may contain harmful pattern: {pattern}',
                    'action': 'review',
                    'pattern': pattern
                })
        
        return violations
    
    def _check_confidence_policy(self, confidence: float) -> List[Dict]:
        """Check confidence against policy thresholds."""
        violations = []
        
        min_threshold = self.policies['safety']['min_confidence_threshold']
        max_threshold = self.policies['safety']['max_confidence_threshold']
        escalation_threshold = self.policies['safety']['human_escalation_threshold']
        
        if confidence < min_threshold:
            violations.append({
                'type': PolicyViolationType.CONFIDENCE_THRESHOLD.value,
                'severity': 'medium',
                'message': f'Confidence below minimum threshold: {confidence} < {min_threshold}',
                'action': 'reject',
                'confidence': confidence
            })
        
        if confidence < escalation_threshold:
            violations.append({
                'type': PolicyViolationType.CONFIDENCE_THRESHOLD.value,
                'severity': 'low',
                'message': f'Confidence below escalation threshold: {confidence} < {escalation_threshold}',
                'action': 'escalate',
                'confidence': confidence
            })
        
        return violations
    
    def _check_processing_time_policy(self, processing_time: float) -> List[Dict]:
        """Check processing time against limits."""
        violations = []
        
        max_time = self.policies['operational']['max_processing_time_seconds']
        
        if processing_time > max_time:
            violations.append({
                'type': PolicyViolationType.PROCESSING_TIME_LIMIT.value,
                'severity': 'medium',
                'message': f'Processing time exceeds limit: {processing_time}s > {max_time}s',
                'action': 'optimize',
                'processing_time': processing_time
            })
        
        return violations
    
    def _check_rate_limit(self, user_id: str) -> List[Dict]:
        """Check rate limiting policies."""
        # Simplified rate limiting - would be enhanced with proper tracking
        violations = []
        
        # This would check actual request rates
        # For now, return empty list
        
        return violations
    
    def _log_violation(self, violation: Dict):
        """Log a policy violation."""
        violation['timestamp'] = datetime.now().isoformat()
        self.violations_log.append(violation)
        
        # Update statistics
        violation_type = violation['type']
        self.policy_stats[violation_type] += 1
        
        # Keep only recent violations
        if len(self.violations_log) > 1000:
            self.violations_log = self.violations_log[-1000:]
        
        logger.warning(f"Policy violation: {violation['message']}")
    
    def get_policy_report(self) -> Dict[str, Any]:
        """Generate comprehensive policy compliance report."""
        recent_violations = self.violations_log[-50:]  # Last 50 violations
        
        # Calculate violation rates by type
        violation_rates = {}
        for violation_type, count in self.policy_stats.items():
            violation_rates[violation_type] = count
        
        # Get severity distribution
        severity_counts = {}
        for violation in recent_violations:
            severity = violation.get('severity', 'unknown')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_violations': len(self.violations_log),
            'recent_violations': len(recent_violations),
            'violation_rates': violation_rates,
            'severity_distribution': severity_counts,
            'active_policies': list(self.policies.keys()),
            'compliance_score': max(0.0, 1.0 - (len(recent_violations) / 100)),
            'timestamp': datetime.now().isoformat()
        }
    
    def update_policy(self, section: str, key: str, value: Any) -> bool:
        """Update a specific policy setting."""
        try:
            if section in self.policies:
                self.policies[section][key] = value
                logger.info(f"Updated policy {section}.{key} = {value}")
                return True
            else:
                logger.error(f"Policy section not found: {section}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating policy: {e}")
            return False
    
    def get_violation_history(self, limit: int = 100) -> List[Dict]:
        """Get recent violation history."""
        return sorted(
            self.violations_log[-limit:],
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )
