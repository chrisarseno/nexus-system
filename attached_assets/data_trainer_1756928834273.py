"""
Research Lab Data Training Module
Allows theoretical AI components to train on production model data
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy import text

logger = logging.getLogger(__name__)

class ResearchDataTrainer:
    """Manages training of research components on production data."""
    
    def __init__(self, db_session):
        self.db = db_session
        self.training_enabled = False
        self.access_restrictions = {
            'require_super_admin': True,
            'anonymize_data': True,
            'exclude_private': True
        }
    
    def enable_training(self, user):
        """Enable training access for super admin."""
        if not user.is_super_admin():
            raise PermissionError("Only super administrators can enable research training")
        
        self.training_enabled = True
        logger.info(f"Research training enabled by {user.username}")
    
    def get_conversation_data(self, limit: int = 1000, anonymize: bool = True) -> List[Dict]:
        """Get conversation data for training research models."""
        if not self.training_enabled:
            raise PermissionError("Research training not enabled")
        
        try:
            # Get anonymized conversation data
            query = text("""
                SELECT 
                    c.id as conversation_id,
                    cm.content as message_content,
                    cm.is_user_message,
                    cm.model_used,
                    cm.confidence_score,
                    cm.created_at,
                    CASE WHEN :anonymize THEN 'anonymous' ELSE u.username END as user_ref
                FROM conversations c
                JOIN conversation_messages cm ON c.id = cm.conversation_id
                JOIN users u ON c.user_id = u.id
                WHERE u.allow_data_training = true
                ORDER BY cm.created_at DESC
                LIMIT :limit
            """)
            
            result = self.db.execute(query, {
                'anonymize': anonymize,
                'limit': limit
            })
            
            data = [dict(row) for row in result.fetchall()]
            
            logger.info(f"Retrieved {len(data)} training samples for research lab")
            return data
            
        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            return []
    
    def get_model_performance_data(self) -> List[Dict]:
        """Get model performance metrics for research analysis."""
        if not self.training_enabled:
            raise PermissionError("Research training not enabled")
        
        try:
            query = text("""
                SELECT 
                    model_name,
                    accuracy,
                    response_time,
                    confidence_score,
                    user_satisfaction,
                    created_at
                FROM model_performance
                WHERE created_at >= :since
                ORDER BY created_at DESC
            """)
            
            result = self.db.execute(query, {
                'since': datetime.utcnow() - timedelta(days=30)
            })
            
            data = [dict(row) for row in result.fetchall()]
            
            logger.info(f"Retrieved {len(data)} performance metrics for research")
            return data
            
        except Exception as e:
            logger.error(f"Failed to get performance data: {e}")
            return []
    
    def create_training_dataset(self, dataset_name: str, user) -> Dict:
        """Create a curated training dataset for research experiments."""
        if not user.is_super_admin():
            raise PermissionError("Only super administrators can create training datasets")
        
        conversation_data = self.get_conversation_data(limit=5000)
        performance_data = self.get_model_performance_data()
        
        dataset = {
            'name': dataset_name,
            'created_by': user.username,
            'created_at': datetime.utcnow().isoformat(),
            'conversations': len(conversation_data),
            'performance_metrics': len(performance_data),
            'data': {
                'conversations': conversation_data,
                'performance': performance_data
            },
            'metadata': {
                'anonymized': True,
                'privacy_compliant': True,
                'opt_in_only': True
            }
        }
        
        logger.info(f"Created training dataset '{dataset_name}' with {len(conversation_data)} samples")
        return dataset
    
    def get_training_status(self) -> Dict:
        """Get current training status for research lab."""
        return {
            'enabled': self.training_enabled,
            'restrictions': self.access_restrictions,
            'available_data_types': [
                'conversation_messages',
                'model_performance',
                'user_interactions'
            ],
            'privacy_compliance': {
                'opt_in_only': True,
                'anonymized': True,
                'excludes_private': True
            }
        }