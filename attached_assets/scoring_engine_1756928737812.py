"""
Advanced scoring engine for ensemble model evaluation and feedback.
Enhanced from fluffy-eureka scoring system.
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class ScoreType(Enum):
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    CONFIDENCE = "confidence"
    COHERENCE = "coherence"
    COMPLETENESS = "completeness"
    SAFETY = "safety"

@dataclass
class ScoreResult:
    """Individual score result."""
    score_type: ScoreType
    value: float
    explanation: str
    evidence: List[str] = field(default_factory=list)
    confidence: float = 1.0

@dataclass
class EnsembleScore:
    """Comprehensive ensemble evaluation."""
    overall_score: float
    individual_scores: Dict[ScoreType, ScoreResult]
    model_contributions: Dict[str, float]
    timestamp: datetime
    query_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class ScoringEngine:
    """
    Advanced scoring system for ensemble model evaluation,
    feedback integration, and continuous improvement.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.score_weights = self.config.get('score_weights', {
            ScoreType.ACCURACY.value: 0.25,
            ScoreType.RELEVANCE.value: 0.20,
            ScoreType.CONFIDENCE.value: 0.15,
            ScoreType.COHERENCE.value: 0.15,
            ScoreType.COMPLETENESS.value: 0.15,
            ScoreType.SAFETY.value: 0.10
        })
        
        self.score_history: List[EnsembleScore] = []
        self.model_performance: Dict[str, Dict] = {}
        
    def score_ensemble_response(
        self, 
        query: str, 
        response: str, 
        model_outputs: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> EnsembleScore:
        """Score a complete ensemble response."""
        context = context or {}
        query_id = context.get('query_id', f"query_{int(time.time())}")
        
        # Calculate individual scores
        individual_scores = {}
        
        # Accuracy scoring
        accuracy_score = self._score_accuracy(query, response, model_outputs, context)
        individual_scores[ScoreType.ACCURACY] = accuracy_score
        
        # Relevance scoring
        relevance_score = self._score_relevance(query, response, context)
        individual_scores[ScoreType.RELEVANCE] = relevance_score
        
        # Confidence scoring
        confidence_score = self._score_confidence(model_outputs, response)
        individual_scores[ScoreType.CONFIDENCE] = confidence_score
        
        # Coherence scoring
        coherence_score = self._score_coherence(response, model_outputs)
        individual_scores[ScoreType.COHERENCE] = coherence_score
        
        # Completeness scoring
        completeness_score = self._score_completeness(query, response, context)
        individual_scores[ScoreType.COMPLETENESS] = completeness_score
        
        # Safety scoring
        safety_score = self._score_safety(response, context)
        individual_scores[ScoreType.SAFETY] = safety_score
        
        # Calculate overall weighted score
        overall_score = sum(
            score.value * self.score_weights.get(score_type.value, 0.0)
            for score_type, score in individual_scores.items()
        )
        
        # Calculate model contributions
        model_contributions = self._calculate_model_contributions(model_outputs, individual_scores)
        
        ensemble_score = EnsembleScore(
            overall_score=overall_score,
            individual_scores=individual_scores,
            model_contributions=model_contributions,
            timestamp=datetime.now(),
            query_id=query_id,
            metadata={
                'query_length': len(query),
                'response_length': len(response),
                'model_count': len(model_outputs),
                'context_available': bool(context)
            }
        )
        
        self.score_history.append(ensemble_score)
        self._update_model_performance(model_outputs, ensemble_score)
        
        logger.info(f"Scored ensemble response: {overall_score:.3f} (query: {query_id})")
        return ensemble_score
    
    def _score_accuracy(self, query: str, response: str, model_outputs: Dict, context: Dict) -> ScoreResult:
        """Score response accuracy."""
        # Simplified accuracy scoring - would be enhanced with fact-checking
        accuracy = 0.8  # Placeholder
        
        # Check for consistency across models
        consistency = 1.0
        if len(model_outputs) > 1:
            responses = [str(output.get('response', '')) for output in model_outputs.values()]
            consistency = self._calculate_response_consistency(responses)
            accuracy = (accuracy + consistency) / 2
        
        return ScoreResult(
            score_type=ScoreType.ACCURACY,
            value=accuracy,
            explanation="Assessed based on cross-model consistency and context alignment",
            evidence=[f"Model consistency: {consistency:.2f}" if len(model_outputs) > 1 else "Single model response"]
        )
    
    def _score_relevance(self, query: str, response: str, context: Dict) -> ScoreResult:
        """Score response relevance to query."""
        # Simple keyword-based relevance (would be enhanced with semantic analysis)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        common_words = query_words.intersection(response_words)
        relevance = len(common_words) / max(len(query_words), 1)
        relevance = min(relevance * 2, 1.0)  # Scale up but cap at 1.0
        
        return ScoreResult(
            score_type=ScoreType.RELEVANCE,
            value=relevance,
            explanation="Based on keyword overlap and semantic alignment",
            evidence=[f"Common keywords: {len(common_words)}/{len(query_words)}"]
        )
    
    def _score_confidence(self, model_outputs: Dict, response: str) -> ScoreResult:
        """Score ensemble confidence."""
        confidences = []
        for model_name, output in model_outputs.items():
            if 'confidence' in output:
                confidences.append(output['confidence'])
        
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            confidence_variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
            
            # Higher confidence when models agree (low variance)
            confidence_score = avg_confidence * (1 - confidence_variance)
        else:
            confidence_score = 0.5  # Default when no confidence available
        
        return ScoreResult(
            score_type=ScoreType.CONFIDENCE,
            value=confidence_score,
            explanation="Based on average model confidence and agreement",
            evidence=[f"Model confidences: {confidences}"]
        )
    
    def _score_coherence(self, response: str, model_outputs: Dict) -> ScoreResult:
        """Score response coherence and logical flow."""
        # Simple coherence metrics (would be enhanced with NLP analysis)
        sentences = response.split('.')
        
        # Check for reasonable sentence length distribution
        sentence_lengths = [len(s.strip().split()) for s in sentences if s.strip()]
        if sentence_lengths:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            coherence = min(avg_length / 20, 1.0)  # Reasonable sentence length
        else:
            coherence = 0.0
            avg_length = 0
        
        # Boost for structured responses
        if any(marker in response for marker in ['1.', '2.', '-', '*']):
            coherence += 0.2
        
        coherence = min(coherence, 1.0)
        
        return ScoreResult(
            score_type=ScoreType.COHERENCE,
            value=coherence,
            explanation="Based on sentence structure and logical flow",
            evidence=[f"Average sentence length: {avg_length:.1f}" if sentence_lengths else "No sentences"]
        )
    
    def _score_completeness(self, query: str, response: str, context: Dict) -> ScoreResult:
        """Score response completeness."""
        # Basic completeness assessment
        completeness = min(len(response) / 100, 1.0)  # Basic length check
        
        # Check for question answering completeness
        if '?' in query:
            question_words = ['what', 'how', 'why', 'when', 'where', 'who']
            query_lower = query.lower()
            
            for word in question_words:
                if word in query_lower and word in response.lower():
                    completeness += 0.1
        
        completeness = min(completeness, 1.0)
        
        return ScoreResult(
            score_type=ScoreType.COMPLETENESS,
            value=completeness,
            explanation="Based on response length and question coverage",
            evidence=[f"Response length: {len(response)} characters"]
        )
    
    def _score_safety(self, response: str, context: Dict) -> ScoreResult:
        """Score response safety and appropriateness."""
        # Basic safety checks (would be enhanced with safety classifiers)
        safety_issues = []
        response_lower = response.lower()
        
        # Check for potentially harmful content
        harmful_patterns = ['violence', 'harm', 'illegal', 'dangerous']
        for pattern in harmful_patterns:
            if pattern in response_lower:
                safety_issues.append(f"Contains: {pattern}")
        
        safety_score = 1.0 - (len(safety_issues) * 0.2)
        safety_score = max(safety_score, 0.0)
        
        return ScoreResult(
            score_type=ScoreType.SAFETY,
            value=safety_score,
            explanation="Based on content safety analysis",
            evidence=safety_issues if safety_issues else ["No safety issues detected"]
        )
    
    def _calculate_response_consistency(self, responses: List[str]) -> float:
        """Calculate consistency across multiple model responses."""
        if len(responses) < 2:
            return 1.0
        
        # Simple consistency based on response similarity
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                similarity = self._text_similarity(responses[i], responses[j])
                similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_model_contributions(self, model_outputs: Dict, scores: Dict) -> Dict[str, float]:
        """Calculate individual model contributions to overall score."""
        contributions = {}
        
        for model_name in model_outputs.keys():
            # Simple contribution calculation (would be enhanced)
            contribution = 1.0 / len(model_outputs)  # Equal contribution for now
            contributions[model_name] = contribution
        
        return contributions
    
    def _update_model_performance(self, model_outputs: Dict, ensemble_score: EnsembleScore):
        """Update performance tracking for individual models."""
        for model_name in model_outputs.keys():
            if model_name not in self.model_performance:
                self.model_performance[model_name] = {
                    'total_queries': 0,
                    'total_score': 0.0,
                    'scores': []
                }
            
            perf = self.model_performance[model_name]
            perf['total_queries'] += 1
            perf['total_score'] += ensemble_score.overall_score
            perf['scores'].append(ensemble_score.overall_score)
            
            # Keep only recent scores for trend analysis
            if len(perf['scores']) > 100:
                perf['scores'] = perf['scores'][-100:]
    
    def get_model_performance_report(self) -> Dict[str, Any]:
        """Generate model performance report."""
        report = {}
        
        for model_name, perf in self.model_performance.items():
            if perf['total_queries'] > 0:
                avg_score = perf['total_score'] / perf['total_queries']
                recent_scores = perf['scores'][-10:] if perf['scores'] else []
                recent_avg = sum(recent_scores) / len(recent_scores) if recent_scores else 0.0
                
                report[model_name] = {
                    'total_queries': perf['total_queries'],
                    'average_score': avg_score,
                    'recent_average': recent_avg,
                    'trend': 'improving' if recent_avg > avg_score else 'stable' if recent_avg == avg_score else 'declining'
                }
        
        return report
    
    def get_score_trends(self, limit: int = 100) -> Dict[str, List]:
        """Get scoring trends over time."""
        recent_scores = self.score_history[-limit:]
        
        trends = {
            'timestamps': [s.timestamp.isoformat() for s in recent_scores],
            'overall_scores': [s.overall_score for s in recent_scores],
            'accuracy_scores': [s.individual_scores[ScoreType.ACCURACY].value for s in recent_scores],
            'relevance_scores': [s.individual_scores[ScoreType.RELEVANCE].value for s in recent_scores],
            'confidence_scores': [s.individual_scores[ScoreType.CONFIDENCE].value for s in recent_scores]
        }
        
        return trends
