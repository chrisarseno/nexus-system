"""
Quarantine Manager for Model Safety and Integrity
Isolates models that show integrity issues or performance problems.
"""

import logging
from typing import Dict, Set, List, Any
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class QuarantineReason(Enum):
    INTEGRITY_ISSUE = "integrity_issue"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SAFETY_VIOLATION = "safety_violation"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    EXTERNAL_ALERT = "external_alert"

class QuarantineStatus(Enum):
    ACTIVE = "active"
    QUARANTINED = "quarantined"
    MUTED = "muted"
    RESTORED = "restored"

class QuarantineManager:
    """Controls model availability and isolation for integrity issues."""

    def __init__(self, models=None):
        self.models = models or []
        self.quarantined: Dict[str, Dict] = {}
        self.muted: Dict[str, Dict] = {}
        self.quarantine_history: List[Dict] = []
        self.auto_restore_enabled = True
        self.quarantine_duration = timedelta(hours=1)  # Default quarantine period

    def is_active(self, model_id: str) -> bool:
        """Check if a model is active (not quarantined or muted)."""
        return model_id not in self.quarantined and model_id not in self.muted

    def quarantine_model(self, model_id: str, reason: QuarantineReason, 
                        description: str = "", duration: timedelta = None):
        """Quarantine a model for integrity or safety issues."""
        if duration is None:
            duration = self.quarantine_duration
            
        quarantine_entry = {
            "model_id": model_id,
            "reason": reason,
            "description": description,
            "quarantined_at": datetime.now(),
            "restore_at": datetime.now() + duration,
            "status": QuarantineStatus.QUARANTINED
        }
        
        self.quarantined[model_id] = quarantine_entry
        self.quarantine_history.append(quarantine_entry.copy())
        
        logger.warning(f"[QUARANTINE] {model_id} quarantined: {reason.value} - {description}")
        return True

    def mute_model(self, model_id: str, reason: str = "Performance issue", 
                   duration: timedelta = None):
        """Temporarily mute a model for performance issues."""
        if duration is None:
            duration = timedelta(minutes=30)  # Shorter mute period
            
        mute_entry = {
            "model_id": model_id,
            "reason": reason,
            "muted_at": datetime.now(),
            "restore_at": datetime.now() + duration,
            "status": QuarantineStatus.MUTED
        }
        
        self.muted[model_id] = mute_entry
        self.quarantine_history.append(mute_entry.copy())
        
        logger.info(f"[MUTED] {model_id} muted: {reason}")
        return True

    def restore_model(self, model_id: str, force: bool = False):
        """Restore a model to active status."""
        restored = False
        
        if model_id in self.quarantined:
            if force or datetime.now() >= self.quarantined[model_id]["restore_at"]:
                del self.quarantined[model_id]
                restored = True
                logger.info(f"[RESTORED] {model_id} restored from quarantine")
                
        if model_id in self.muted:
            if force or datetime.now() >= self.muted[model_id]["restore_at"]:
                del self.muted[model_id]
                restored = True
                logger.info(f"[RESTORED] {model_id} restored from mute")
        
        if restored:
            self.quarantine_history.append({
                "model_id": model_id,
                "status": QuarantineStatus.RESTORED,
                "restored_at": datetime.now(),
                "forced": force
            })
            
        return restored

    def get_active_models(self):
        """Get list of active (non-quarantined, non-muted) models."""
        return [m for m in self.models if self.is_active(getattr(m, 'model_id', getattr(m, 'name', str(m))))]

    def auto_restore_check(self):
        """Check for models ready for automatic restoration."""
        if not self.auto_restore_enabled:
            return
            
        current_time = datetime.now()
        to_restore = []
        
        # Check quarantined models
        for model_id, entry in self.quarantined.items():
            if current_time >= entry["restore_at"]:
                to_restore.append(model_id)
                
        # Check muted models
        for model_id, entry in self.muted.items():
            if current_time >= entry["restore_at"]:
                to_restore.append(model_id)
                
        # Restore eligible models
        for model_id in to_restore:
            self.restore_model(model_id)
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get current quarantine status report."""
        return {
            'total_quarantined': len(self.quarantined),
            'total_muted': len(self.muted),
            'quarantined_models': list(self.quarantined.keys()),
            'muted_models': list(self.muted.keys()),
            'recent_events': self.quarantine_history[-10:],  # Last 10 events
            'auto_restore_enabled': self.auto_restore_enabled
        }
    
    def get_quarantine_history(self, limit: int = 50) -> List[Dict]:
        """Get quarantine history with limit."""
        return self.quarantine_history[-limit:] if limit else self.quarantine_history

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        self.auto_restore_check()  # Update status first
        
        active_models = []
        for model in self.models:
            model_id = getattr(model, 'model_id', getattr(model, 'name', str(model)))
            if self.is_active(model_id):
                active_models.append(model_id)
        
        return {
            "active_models": active_models,
            "quarantined_models": list(self.quarantined.keys()),
            "muted_models": list(self.muted.keys()),
            "quarantine_details": self.quarantined,
            "mute_details": self.muted,
            "total_models": len(self.models),
            "active_count": len(active_models),
            "quarantined_count": len(self.quarantined),
            "muted_count": len(self.muted)
        }

    def get_quarantine_history(self, limit: int = 50) -> List[Dict]:
        """Get recent quarantine history."""
        return sorted(self.quarantine_history, 
                     key=lambda x: x.get('quarantined_at', x.get('muted_at', x.get('restored_at', datetime.min))), 
                     reverse=True)[:limit]

    def analyze_model_reliability(self, model_id: str) -> Dict[str, Any]:
        """Analyze a model's reliability based on quarantine history."""
        model_history = [h for h in self.quarantine_history if h['model_id'] == model_id]
        
        quarantine_count = len([h for h in model_history if h.get('reason')])
        mute_count = len([h for h in model_history if 'muted_at' in h])
        
        # Calculate reliability score (higher is better)
        total_incidents = quarantine_count + mute_count
        if total_incidents == 0:
            reliability_score = 1.0
        else:
            # Score decreases with more incidents
            reliability_score = max(0.0, 1.0 - (total_incidents * 0.1))
        
        return {
            "model_id": model_id,
            "quarantine_incidents": quarantine_count,
            "mute_incidents": mute_count,
            "total_incidents": total_incidents,
            "reliability_score": reliability_score,
            "current_status": "quarantined" if model_id in self.quarantined else 
                            "muted" if model_id in self.muted else "active"
        }
