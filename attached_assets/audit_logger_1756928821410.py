"""
Comprehensive Security Audit Logger
Provides detailed logging and monitoring for all security-related events.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from flask import request, session
from app import db
from models import SecurityAuditLog, EthicalIncident

logger = logging.getLogger(__name__)

class SecurityAuditLogger:
    """Advanced security audit logging system for comprehensive monitoring."""
    
    def __init__(self):
        self.initialized = False
        self.threat_patterns = [
            'sql injection', 'xss', 'csrf', 'directory traversal',
            'command injection', 'brute force', 'unauthorized access'
        ]
        
    def initialize(self):
        """Initialize the security audit logger."""
        if self.initialized:
            return
            
        logger.info("Initializing Security Audit Logger...")
        self.initialized = True
        logger.info("Security Audit Logger initialized")
    
    def log_security_event(self, event_type: str, resource: str = None, 
                          action: str = None, success: bool = True,
                          security_level: str = 'normal', 
                          additional_data: Dict[str, Any] = None):
        """Log a comprehensive security event."""
        try:
            # Extract request information safely
            ip_address = None
            user_agent = None
            request_data = {}
            
            if request:
                ip_address = request.remote_addr
                user_agent = request.headers.get('User-Agent', '')[:500]  # Limit length
                request_data = {
                    'method': request.method,
                    'path': request.path,
                    'args': dict(request.args) if request.args else {},
                    'headers': dict(request.headers) if hasattr(request, 'headers') else {}
                }
            
            # Detect threat indicators
            threat_indicators = self._detect_threats(request_data, user_agent)
            
            # Create audit log entry
            audit_entry = SecurityAuditLog(
                event_type=event_type,
                user_id=session.get('user_id') if session else None,
                ip_address=ip_address,
                user_agent=user_agent,
                resource_accessed=resource,
                action_performed=action,
                security_level=security_level,
                success=success,
                failure_reason=additional_data.get('failure_reason') if additional_data else None,
                request_data=request_data,
                response_data=additional_data.get('response_data') if additional_data else {},
                session_id=session.get('session_id') if session else str(uuid.uuid4()),
                threat_indicators=threat_indicators,
                model_metadata=additional_data or {}
            )
            
            db.session.add(audit_entry)
            db.session.commit()
            
            # Escalate if high-risk event
            if security_level in ['elevated', 'critical'] or threat_indicators:
                self._escalate_security_event(audit_entry)
                
        except Exception as e:
            logger.error(f"Error logging security event: {e}")
    
    def log_ethical_incident(self, incident_type: str, severity: str,
                           description: str, evidence: Dict = None,
                           context: Dict = None):
        """Log an ethical incident or safety violation."""
        try:
            incident = EthicalIncident(
                incident_id=str(uuid.uuid4()),
                incident_type=incident_type,
                severity=severity,
                description=description,
                evidence=evidence or {},
                context=context or {},
                escalated_to_human=(severity in ['high', 'critical'])
            )
            
            db.session.add(incident)
            db.session.commit()
            
            logger.warning(f"Ethical incident logged: {incident_type} - {severity}")
            
            return incident.incident_id
            
        except Exception as e:
            logger.error(f"Error logging ethical incident: {e}")
            return None
    
    def _detect_threats(self, request_data: Dict, user_agent: str) -> Dict[str, Any]:
        """Detect potential security threats in request data."""
        threats = {}
        
        try:
            # Check for suspicious patterns
            suspicious_content = str(request_data).lower()
            for pattern in self.threat_patterns:
                if pattern in suspicious_content:
                    threats[pattern] = True
            
            # Check user agent for suspicious patterns
            if user_agent:
                ua_lower = user_agent.lower()
                if any(bot in ua_lower for bot in ['bot', 'crawler', 'spider', 'scan']):
                    threats['automated_access'] = True
                    
                if len(user_agent) > 1000:  # Unusually long user agent
                    threats['suspicious_user_agent'] = True
            
            # Check for rapid requests (basic rate limiting detection)
            # This would be enhanced with actual rate limiting logic
            
        except Exception as e:
            logger.error(f"Error detecting threats: {e}")
            
        return threats
    
    def _escalate_security_event(self, audit_entry: SecurityAuditLog):
        """Escalate high-risk security events for immediate attention."""
        try:
            logger.critical(f"Security escalation: {audit_entry.event_type} - "
                          f"Level: {audit_entry.security_level}")
            
            # In a production environment, this would:
            # - Send alerts to security team
            # - Trigger automated responses
            # - Update security dashboards
            # - Potentially block suspicious IPs
            
        except Exception as e:
            logger.error(f"Error escalating security event: {e}")
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics."""
        try:
            # Count events by type and severity
            total_events = SecurityAuditLog.query.count()
            failed_events = SecurityAuditLog.query.filter_by(success=False).count()
            high_risk_events = SecurityAuditLog.query.filter(
                SecurityAuditLog.security_level.in_(['elevated', 'critical'])
            ).count()
            
            # Count ethical incidents
            total_incidents = EthicalIncident.query.count()
            unresolved_incidents = EthicalIncident.query.filter_by(resolved=False).count()
            critical_incidents = EthicalIncident.query.filter_by(severity='critical').count()
            
            return {
                'total_security_events': total_events,
                'failed_events': failed_events,
                'high_risk_events': high_risk_events,
                'total_ethical_incidents': total_incidents,
                'unresolved_incidents': unresolved_incidents,
                'critical_incidents': critical_incidents,
                'security_score': max(0, 100 - (failed_events * 2) - (high_risk_events * 5)),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting security metrics: {e}")
            return {}

# Global instance
security_audit_logger = SecurityAuditLogger()