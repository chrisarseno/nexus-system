"""
Multi-Modal Knowledge Integration System
Advanced cross-modal understanding and unified knowledge representation for enhanced AI capabilities.
"""

import logging
import time
import threading
import numpy as np
import json
import pickle
import hashlib
import math
import statistics
from typing import Dict, List, Any, Set, Tuple, Optional, Union, Callable
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import copy
import heapq
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import re

logger = logging.getLogger(__name__)

class ModalityType(Enum):
    """Types of modalities for knowledge representation."""
    TEXT = "text"
    VISUAL = "visual"
    AUDIO = "audio"
    STRUCTURED = "structured"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    GRAPH = "graph"
    NUMERICAL = "numerical"

class IntegrationType(Enum):
    """Types of multi-modal integration strategies."""
    EARLY_FUSION = "early_fusion"
    LATE_FUSION = "late_fusion"
    INTERMEDIATE_FUSION = "intermediate_fusion"
    ATTENTION_FUSION = "attention_fusion"
    CROSS_MODAL_ATTENTION = "cross_modal_attention"
    HIERARCHICAL_FUSION = "hierarchical_fusion"

class ReasoningMode(Enum):
    """Modes of cross-modal reasoning."""
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    ABSTRACT = "abstract"
    COMPOSITIONAL = "compositional"

@dataclass
class ModalRepresentation:
    """Represents knowledge in a specific modality."""
    modality: ModalityType
    content: Any
    embedding: np.ndarray
    metadata: Dict[str, Any]
    confidence: float = 1.0
    quality_score: float = 1.0
    timestamp: Optional[datetime] = None
    source_id: str = ""
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def calculate_relevance(self, query_embedding: np.ndarray) -> float:
        """Calculate relevance to a query embedding."""
        if self.embedding is None or query_embedding is None:
            return 0.0
        
        # Cosine similarity as base relevance
        base_relevance = cosine_similarity([self.embedding], [query_embedding])[0][0]
        
        # Adjust for confidence and quality
        adjusted_relevance = base_relevance * self.confidence * self.quality_score
        
        return min(1.0, max(0.0, adjusted_relevance))

@dataclass
class CrossModalLink:
    """Represents a link between different modal representations."""
    source_modality: ModalityType
    target_modality: ModalityType
    source_id: str
    target_id: str
    link_strength: float
    link_type: str
    semantic_similarity: float = 0.0
    temporal_correlation: float = 0.0
    spatial_correlation: float = 0.0
    creation_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.creation_time is None:
            self.creation_time = datetime.now()
    
    def calculate_overall_strength(self) -> float:
        """Calculate overall link strength from multiple factors."""
        return (
            0.4 * self.link_strength +
            0.3 * self.semantic_similarity +
            0.2 * self.temporal_correlation +
            0.1 * self.spatial_correlation
        )

class ModalityProcessor:
    """Base class for processing different modalities."""
    
    def __init__(self, modality: ModalityType, embedding_dim: int = 512):
        self.modality = modality
        self.embedding_dim = embedding_dim
        self.processing_cache = {}
        self.cache_size_limit = 1000
        
    def process(self, content: Any, metadata: Optional[Dict[str, Any]] = None) -> ModalRepresentation:
        """Process content into modal representation."""
        raise NotImplementedError("Subclasses must implement process method")
    
    def extract_features(self, content: Any) -> np.ndarray:
        """Extract features from content."""
        raise NotImplementedError("Subclasses must implement extract_features method")
    
    def calculate_quality(self, content: Any, embedding: np.ndarray) -> float:
        """Calculate quality score for the representation."""
        # Default quality calculation
        if embedding is None or len(embedding) == 0:
            return 0.0
        
        # Basic quality metrics
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm == 0:
            return 0.0
        
        # Quality based on embedding distribution and content characteristics
        embedding_variance = np.var(embedding)
        content_length = len(str(content)) if content else 0
        
        quality = min(1.0, (float(embedding_variance) * 0.5 + min(1.0, content_length / 1000) * 0.5))
        return quality

class TextModalityProcessor(ModalityProcessor):
    """Processor for text modality."""
    
    def __init__(self, embedding_dim: int = 512):
        super().__init__(ModalityType.TEXT, embedding_dim)
        self.vectorizer = TfidfVectorizer(max_features=embedding_dim)
        self.is_fitted = False
        
    def process(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ModalRepresentation:
        """Process text content into modal representation."""
        start_time = time.time()
        
        if metadata is None:
            metadata = {}
        
        # Check cache
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if content_hash in self.processing_cache:
            cached_result = self.processing_cache[content_hash]
            return cached_result
        
        # Extract features
        embedding = self.extract_features(content)
        
        # Calculate quality
        quality = self.calculate_quality(content, embedding)
        
        # Calculate confidence based on text characteristics
        confidence = self._calculate_text_confidence(content)
        
        representation = ModalRepresentation(
            modality=ModalityType.TEXT,
            content=content,
            embedding=embedding,
            metadata={**metadata, 'length': len(content), 'word_count': len(content.split())},
            confidence=confidence,
            quality_score=quality,
            processing_time=time.time() - start_time
        )
        
        # Cache result
        if len(self.processing_cache) < self.cache_size_limit:
            self.processing_cache[content_hash] = representation
        
        return representation
    
    def extract_features(self, content: str) -> np.ndarray:
        """Extract TF-IDF features from text."""
        try:
            if not self.is_fitted:
                # Fit vectorizer with sample text if not already fitted
                sample_texts = [content, "sample text for fitting", "another sample"]
                self.vectorizer.fit(sample_texts)
                self.is_fitted = True
            
            # Transform text to vector
            tfidf_matrix = self.vectorizer.transform([content])
            tfidf_vector = tfidf_matrix.toarray()[0] if hasattr(tfidf_matrix, 'toarray') else np.array(tfidf_matrix)[0]
            
            # Ensure correct dimension
            if len(tfidf_vector) < self.embedding_dim:
                padded = np.zeros(self.embedding_dim)
                padded[:len(tfidf_vector)] = tfidf_vector
                tfidf_vector = padded
            elif len(tfidf_vector) > self.embedding_dim:
                tfidf_vector = tfidf_vector[:self.embedding_dim]
            
            # Normalize
            norm = np.linalg.norm(tfidf_vector)
            if norm > 0:
                tfidf_vector = tfidf_vector / norm
            
            return tfidf_vector
            
        except Exception as e:
            logger.warning(f"Text feature extraction failed: {e}")
            return np.random.normal(0, 0.1, self.embedding_dim)
    
    def _calculate_text_confidence(self, content: str) -> float:
        """Calculate confidence score for text content."""
        if not content or len(content.strip()) == 0:
            return 0.0
        
        # Length factor
        length_score = min(1.0, len(content) / 500)  # Optimal around 500 chars
        
        # Vocabulary richness
        words = content.lower().split()
        unique_words = set(words)
        vocab_richness = len(unique_words) / max(1, len(words))
        
        # Sentence structure (basic heuristic)
        sentences = re.split(r'[.!?]+', content)
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        structure_score = min(1.0, float(avg_sentence_length) / 20)  # Optimal around 20 words per sentence
        
        confidence = (0.4 * length_score + 0.4 * vocab_richness + 0.2 * structure_score)
        return min(1.0, max(0.1, confidence))

class StructuredModalityProcessor(ModalityProcessor):
    """Processor for structured data modality."""
    
    def __init__(self, embedding_dim: int = 512):
        super().__init__(ModalityType.STRUCTURED, embedding_dim)
        
    def process(self, content: Union[Dict, List], metadata: Optional[Dict[str, Any]] = None) -> ModalRepresentation:
        """Process structured data into modal representation."""
        start_time = time.time()
        
        if metadata is None:
            metadata = {}
        
        # Extract features
        embedding = self.extract_features(content)
        
        # Calculate quality
        quality = self.calculate_quality(content, embedding)
        
        # Calculate confidence
        confidence = self._calculate_structure_confidence(content)
        
        representation = ModalRepresentation(
            modality=ModalityType.STRUCTURED,
            content=content,
            embedding=embedding,
            metadata={**metadata, 'size': self._get_structure_size(content), 'type': type(content).__name__},
            confidence=confidence,
            quality_score=quality,
            processing_time=time.time() - start_time
        )
        
        return representation
    
    def extract_features(self, content: Union[Dict, List]) -> np.ndarray:
        """Extract features from structured data."""
        try:
            # Convert structured data to feature vector
            if isinstance(content, dict):
                features = self._extract_dict_features(content)
            elif isinstance(content, list):
                features = self._extract_list_features(content)
            else:
                # Fallback: convert to string and use simple encoding
                content_str = str(content)
                features = np.array([hash(content_str) % 1000 for _ in range(self.embedding_dim)])
                features = features / np.linalg.norm(features)
            
            return features
            
        except Exception as e:
            logger.warning(f"Structured data feature extraction failed: {e}")
            return np.random.normal(0, 0.1, self.embedding_dim)
    
    def _extract_dict_features(self, data: Dict) -> np.ndarray:
        """Extract features from dictionary data."""
        features = np.zeros(self.embedding_dim)
        
        # Extract key-value pair features
        for i, (key, value) in enumerate(data.items()):
            if i >= self.embedding_dim:
                break
            
            # Simple hashing-based feature
            key_hash = hash(str(key)) % 1000
            value_hash = hash(str(value)) % 1000
            features[i] = (key_hash + value_hash) / 2000.0
        
        # Add structural features
        if len(data) > 0:
            features[min(len(features) - 1, len(data))] = len(data) / 100.0  # Size feature
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def _extract_list_features(self, data: List) -> np.ndarray:
        """Extract features from list data."""
        features = np.zeros(self.embedding_dim)
        
        # Extract element features
        for i, item in enumerate(data):
            if i >= self.embedding_dim:
                break
            
            item_hash = hash(str(item)) % 1000
            features[i] = item_hash / 1000.0
        
        # Add structural features
        if len(data) > 0:
            features[min(len(features) - 1, len(data))] = len(data) / 100.0  # Size feature
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def _calculate_structure_confidence(self, content: Union[Dict, List]) -> float:
        """Calculate confidence for structured data."""
        if not content:
            return 0.0
        
        if isinstance(content, dict):
            # Dictionary confidence based on key-value completeness
            non_empty_values = sum(1 for v in content.values() if v is not None and v != "")
            confidence = non_empty_values / max(1, len(content))
        elif isinstance(content, list):
            # List confidence based on element completeness
            non_empty_items = sum(1 for item in content if item is not None and item != "")
            confidence = non_empty_items / max(1, len(content))
        else:
            confidence = 0.5  # Default for other types
        
        return min(1.0, max(0.1, confidence))
    
    def _get_structure_size(self, content: Union[Dict, List]) -> int:
        """Get size of structured data."""
        if isinstance(content, (dict, list)):
            return len(content)
        return 1

class CrossModalReasoner:
    """Advanced cross-modal reasoning engine."""
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.reasoning_cache = {}
        self.cache_size_limit = 1000
        
        # Reasoning strategies
        self.reasoning_strategies = {
            ReasoningMode.ANALOGICAL: self._analogical_reasoning,
            ReasoningMode.CAUSAL: self._causal_reasoning,
            ReasoningMode.SPATIAL: self._spatial_reasoning,
            ReasoningMode.TEMPORAL: self._temporal_reasoning,
            ReasoningMode.ABSTRACT: self._abstract_reasoning,
            ReasoningMode.COMPOSITIONAL: self._compositional_reasoning
        }
    
    def reason_across_modalities(self, 
                                representations: List[ModalRepresentation],
                                query: str,
                                reasoning_mode: ReasoningMode = ReasoningMode.ANALOGICAL) -> Dict[str, Any]:
        """Perform cross-modal reasoning."""
        try:
            # Check cache
            cache_key = hashlib.md5(f"{query}{reasoning_mode.value}{len(representations)}".encode()).hexdigest()
            if cache_key in self.reasoning_cache:
                return self.reasoning_cache[cache_key]
            
            start_time = time.time()
            
            # Group representations by modality
            modal_groups = defaultdict(list)
            for rep in representations:
                modal_groups[rep.modality].append(rep)
            
            # Apply reasoning strategy
            reasoning_strategy = self.reasoning_strategies.get(reasoning_mode, self._analogical_reasoning)
            reasoning_result = reasoning_strategy(modal_groups, query)
            
            # Add metadata
            reasoning_result.update({
                'reasoning_mode': reasoning_mode.value,
                'processing_time': time.time() - start_time,
                'modalities_involved': list(modal_groups.keys()),
                'total_representations': len(representations)
            })
            
            # Cache result
            if len(self.reasoning_cache) < self.cache_size_limit:
                self.reasoning_cache[cache_key] = reasoning_result
            
            return reasoning_result
            
        except Exception as e:
            logger.error(f"Cross-modal reasoning failed: {e}")
            return {'error': str(e), 'reasoning_mode': reasoning_mode.value}
    
    def _analogical_reasoning(self, modal_groups: Dict[ModalityType, List[ModalRepresentation]], query: str) -> Dict[str, Any]:
        """Perform analogical reasoning across modalities."""
        analogies = []
        
        # Find analogies between different modalities
        modality_types = list(modal_groups.keys())
        
        for i in range(len(modality_types)):
            for j in range(i + 1, len(modality_types)):
                mod1, mod2 = modality_types[i], modality_types[j]
                
                # Find best matching representations
                best_analogy = self._find_best_analogy(modal_groups[mod1], modal_groups[mod2])
                if best_analogy:
                    analogies.append(best_analogy)
        
        # Sort analogies by strength
        analogies.sort(key=lambda x: x['strength'], reverse=True)
        
        return {
            'reasoning_type': 'analogical',
            'analogies': analogies[:5],  # Top 5 analogies
            'confidence': np.mean([a['strength'] for a in analogies]) if analogies else 0.0
        }
    
    def _find_best_analogy(self, group1: List[ModalRepresentation], group2: List[ModalRepresentation]) -> Optional[Dict[str, Any]]:
        """Find the best analogy between two modal groups."""
        best_similarity = 0.0
        best_analogy = None
        
        for rep1 in group1:
            for rep2 in group2:
                if rep1.embedding is not None and rep2.embedding is not None:
                    similarity = cosine_similarity([rep1.embedding], [rep2.embedding])[0][0]
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_analogy = {
                            'modality1': rep1.modality.value,
                            'modality2': rep2.modality.value,
                            'content1': str(rep1.content)[:100] + "..." if len(str(rep1.content)) > 100 else str(rep1.content),
                            'content2': str(rep2.content)[:100] + "..." if len(str(rep2.content)) > 100 else str(rep2.content),
                            'strength': similarity,
                            'confidence_factor': rep1.confidence * rep2.confidence
                        }
        
        return best_analogy
    
    def _causal_reasoning(self, modal_groups: Dict[ModalityType, List[ModalRepresentation]], query: str) -> Dict[str, Any]:
        """Perform causal reasoning across modalities."""
        causal_chains = []
        
        # Look for temporal sequences that might indicate causation
        temporal_reps = []
        for modality, reps in modal_groups.items():
            for rep in reps:
                temporal_reps.append((rep.timestamp, rep))
        
        # Sort by timestamp
        temporal_reps.sort(key=lambda x: x[0])
        
        # Identify potential causal relationships
        for i in range(len(temporal_reps) - 1):
            time1, rep1 = temporal_reps[i]
            time2, rep2 = temporal_reps[i + 1]
            
            time_diff = (time2 - time1).total_seconds()
            
            # If representations are close in time and from different modalities
            if time_diff < 3600 and rep1.modality != rep2.modality:  # Within 1 hour
                if rep1.embedding is not None and rep2.embedding is not None:
                    similarity = cosine_similarity([rep1.embedding], [rep2.embedding])[0][0]
                    
                    if similarity > 0.5:  # Threshold for potential causation
                        causal_chains.append({
                            'cause_modality': rep1.modality.value,
                            'effect_modality': rep2.modality.value,
                            'time_difference': time_diff,
                            'semantic_similarity': similarity,
                            'causal_strength': similarity * (1.0 / (1.0 + time_diff / 3600))  # Decay with time
                        })
        
        return {
            'reasoning_type': 'causal',
            'causal_chains': causal_chains,
            'confidence': np.mean([c['causal_strength'] for c in causal_chains]) if causal_chains else 0.0
        }
    
    def _spatial_reasoning(self, modal_groups: Dict[ModalityType, List[ModalRepresentation]], query: str) -> Dict[str, Any]:
        """Perform spatial reasoning across modalities."""
        spatial_relationships = []
        
        # Look for spatial information in metadata
        spatial_reps = []
        for modality, reps in modal_groups.items():
            for rep in reps:
                if any(key in rep.metadata for key in ['location', 'position', 'coordinates', 'spatial']):
                    spatial_reps.append(rep)
        
        # Analyze spatial relationships
        for i in range(len(spatial_reps)):
            for j in range(i + 1, len(spatial_reps)):
                rep1, rep2 = spatial_reps[i], spatial_reps[j]
                
                relationship = self._analyze_spatial_relationship(rep1, rep2)
                if relationship:
                    spatial_relationships.append(relationship)
        
        return {
            'reasoning_type': 'spatial',
            'spatial_relationships': spatial_relationships,
            'confidence': np.mean([r['confidence'] for r in spatial_relationships]) if spatial_relationships else 0.0
        }
    
    def _analyze_spatial_relationship(self, rep1: ModalRepresentation, rep2: ModalRepresentation) -> Optional[Dict[str, Any]]:
        """Analyze spatial relationship between two representations."""
        # Simple spatial analysis based on metadata
        spatial_keys = ['location', 'position', 'coordinates', 'spatial']
        
        spatial1 = None
        spatial2 = None
        
        for key in spatial_keys:
            if key in rep1.metadata:
                spatial1 = rep1.metadata[key]
                break
        
        for key in spatial_keys:
            if key in rep2.metadata:
                spatial2 = rep2.metadata[key]
                break
        
        if spatial1 and spatial2:
            # Simple string comparison for spatial relationship
            if isinstance(spatial1, str) and isinstance(spatial2, str):
                common_terms = len(set(spatial1.lower().split()) & set(spatial2.lower().split()))
                confidence = min(1.0, common_terms / 5.0)  # Normalize
                
                return {
                    'modality1': rep1.modality.value,
                    'modality2': rep2.modality.value,
                    'spatial1': spatial1,
                    'spatial2': spatial2,
                    'relationship_type': 'co_located' if confidence > 0.5 else 'related',
                    'confidence': confidence
                }
        
        return None
    
    def _temporal_reasoning(self, modal_groups: Dict[ModalityType, List[ModalRepresentation]], query: str) -> Dict[str, Any]:
        """Perform temporal reasoning across modalities."""
        temporal_patterns = []
        
        # Analyze temporal patterns
        all_reps = []
        for reps in modal_groups.values():
            all_reps.extend(reps)
        
        # Sort by timestamp
        all_reps.sort(key=lambda x: x.timestamp)
        
        # Look for temporal patterns
        time_windows = self._identify_time_windows(all_reps)
        
        for window in time_windows:
            pattern = self._analyze_temporal_window(window)
            if pattern:
                temporal_patterns.append(pattern)
        
        return {
            'reasoning_type': 'temporal',
            'temporal_patterns': temporal_patterns,
            'confidence': np.mean([p['confidence'] for p in temporal_patterns]) if temporal_patterns else 0.0
        }
    
    def _identify_time_windows(self, representations: List[ModalRepresentation], window_size: timedelta = timedelta(hours=1)) -> List[List[ModalRepresentation]]:
        """Identify time windows for temporal analysis."""
        if not representations:
            return []
        
        windows = []
        current_window = [representations[0]]
        window_start = representations[0].timestamp
        
        for rep in representations[1:]:
            if rep.timestamp - window_start <= window_size:
                current_window.append(rep)
            else:
                if len(current_window) > 1:
                    windows.append(current_window)
                current_window = [rep]
                window_start = rep.timestamp
        
        # Add the last window
        if len(current_window) > 1:
            windows.append(current_window)
        
        return windows
    
    def _analyze_temporal_window(self, window: List[ModalRepresentation]) -> Optional[Dict[str, Any]]:
        """Analyze temporal patterns within a window."""
        if len(window) < 2:
            return None
        
        modalities_in_window = set(rep.modality for rep in window)
        
        if len(modalities_in_window) > 1:
            # Multi-modal temporal pattern
            return {
                'window_start': window[0].timestamp.isoformat(),
                'window_end': window[-1].timestamp.isoformat(),
                'modalities': [mod.value for mod in modalities_in_window],
                'sequence_length': len(window),
                'pattern_type': 'multi_modal_sequence',
                'confidence': min(1.0, len(modalities_in_window) / 4.0)  # More modalities = higher confidence
            }
        
        return None
    
    def _abstract_reasoning(self, modal_groups: Dict[ModalityType, List[ModalRepresentation]], query: str) -> Dict[str, Any]:
        """Perform abstract reasoning across modalities."""
        abstract_concepts = []
        
        # Extract abstract concepts from each modality
        for modality, reps in modal_groups.items():
            concepts = self._extract_abstract_concepts(reps, modality)
            abstract_concepts.extend(concepts)
        
        # Find connections between abstract concepts
        concept_connections = self._find_concept_connections(abstract_concepts)
        
        return {
            'reasoning_type': 'abstract',
            'abstract_concepts': abstract_concepts,
            'concept_connections': concept_connections,
            'confidence': np.mean([c['strength'] for c in concept_connections]) if concept_connections else 0.0
        }
    
    def _extract_abstract_concepts(self, representations: List[ModalRepresentation], modality: ModalityType) -> List[Dict[str, Any]]:
        """Extract abstract concepts from representations."""
        concepts = []
        
        for rep in representations:
            if modality == ModalityType.TEXT:
                # Extract concepts from text using simple keyword extraction
                text = str(rep.content).lower()
                concept_keywords = ['concept', 'idea', 'theory', 'principle', 'notion', 'abstract']
                
                for keyword in concept_keywords:
                    if keyword in text:
                        concepts.append({
                            'concept_type': keyword,
                            'modality': modality.value,
                            'confidence': rep.confidence,
                            'context': text[:200]  # First 200 chars for context
                        })
            
            elif modality == ModalityType.STRUCTURED:
                # Extract concepts from structured data
                if isinstance(rep.content, dict):
                    for key, value in rep.content.items():
                        if isinstance(key, str) and any(term in key.lower() for term in ['type', 'category', 'class', 'concept']):
                            concepts.append({
                                'concept_type': key,
                                'concept_value': str(value),
                                'modality': modality.value,
                                'confidence': rep.confidence
                            })
        
        return concepts
    
    def _find_concept_connections(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find connections between abstract concepts."""
        connections = []
        
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                concept1, concept2 = concepts[i], concepts[j]
                
                # Calculate connection strength
                strength = self._calculate_concept_similarity(concept1, concept2)
                
                if strength > 0.3:  # Threshold for meaningful connection
                    connections.append({
                        'concept1': concept1,
                        'concept2': concept2,
                        'strength': strength,
                        'connection_type': 'semantic_similarity'
                    })
        
        return connections
    
    def _calculate_concept_similarity(self, concept1: Dict[str, Any], concept2: Dict[str, Any]) -> float:
        """Calculate similarity between two concepts."""
        # Simple string-based similarity
        text1 = str(concept1.get('concept_type', '') + ' ' + concept1.get('concept_value', ''))
        text2 = str(concept2.get('concept_type', '') + ' ' + concept2.get('concept_value', ''))
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Adjust for confidence
        conf_factor = (concept1.get('confidence', 1.0) + concept2.get('confidence', 1.0)) / 2.0
        
        return jaccard_similarity * conf_factor
    
    def _compositional_reasoning(self, modal_groups: Dict[ModalityType, List[ModalRepresentation]], query: str) -> Dict[str, Any]:
        """Perform compositional reasoning across modalities."""
        compositions = []
        
        # Create compositions by combining representations from different modalities
        modality_types = list(modal_groups.keys())
        
        for i in range(len(modality_types)):
            for j in range(i + 1, len(modality_types)):
                mod1, mod2 = modality_types[i], modality_types[j]
                
                # Create compositions between modalities
                for rep1 in modal_groups[mod1]:
                    for rep2 in modal_groups[mod2]:
                        composition = self._create_composition(rep1, rep2)
                        if composition:
                            compositions.append(composition)
        
        # Sort by composition strength
        compositions.sort(key=lambda x: x['strength'], reverse=True)
        
        return {
            'reasoning_type': 'compositional',
            'compositions': compositions[:10],  # Top 10 compositions
            'confidence': np.mean([c['strength'] for c in compositions]) if compositions else 0.0
        }
    
    def _create_composition(self, rep1: ModalRepresentation, rep2: ModalRepresentation) -> Optional[Dict[str, Any]]:
        """Create a composition from two representations."""
        if rep1.embedding is None or rep2.embedding is None:
            return None
        
        # Calculate composition strength
        semantic_similarity = cosine_similarity([rep1.embedding], [rep2.embedding])[0][0]
        
        # Combine confidence scores
        combined_confidence = (rep1.confidence + rep2.confidence) / 2.0
        
        # Calculate temporal proximity
        time_diff = abs((rep1.timestamp - rep2.timestamp).total_seconds())
        temporal_factor = 1.0 / (1.0 + time_diff / 3600)  # Decay over hours
        
        # Overall composition strength
        strength = semantic_similarity * combined_confidence * temporal_factor
        
        if strength > 0.2:  # Threshold for meaningful composition
            return {
                'modality1': rep1.modality.value,
                'modality2': rep2.modality.value,
                'content1_preview': str(rep1.content)[:50] + "..." if len(str(rep1.content)) > 50 else str(rep1.content),
                'content2_preview': str(rep2.content)[:50] + "..." if len(str(rep2.content)) > 50 else str(rep2.content),
                'strength': strength,
                'semantic_similarity': semantic_similarity,
                'temporal_proximity': temporal_factor,
                'combined_confidence': combined_confidence
            }
        
        return None

class MultiModalKnowledgeIntegrator:
    """
    Advanced multi-modal knowledge integration system with cross-modal understanding,
    unified knowledge representation, and intelligent reasoning capabilities.
    """
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        
        # Core components
        self.modality_processors = {
            ModalityType.TEXT: TextModalityProcessor(embedding_dim),
            ModalityType.STRUCTURED: StructuredModalityProcessor(embedding_dim)
        }
        
        self.cross_modal_reasoner = CrossModalReasoner(embedding_dim)
        
        # Storage
        self.representations: Dict[str, ModalRepresentation] = {}
        self.cross_modal_links: List[CrossModalLink] = []
        
        # Advanced features
        self.unified_embeddings = {}
        self.integration_strategies = {}
        self.reasoning_history = deque(maxlen=100)
        
        # Performance optimization
        self.processing_cache = {}
        self.link_cache = {}
        self.cache_size_limit = 5000
        
        # Real-time processing
        self.processing_queue = deque()
        self.batch_processing = True
        self.processing_thread = None
        self.processing_running = False
        
        # Analytics
        self.integration_analytics = {
            'total_representations': 0,
            'modalities_processed': set(),
            'cross_modal_links': 0,
            'reasoning_sessions': 0,
            'average_processing_time': 0.0
        }
        
        self.initialized = False
        logger.info("Multi-Modal Knowledge Integrator initialized")
    
    def initialize(self) -> bool:
        """Initialize the multi-modal integration system."""
        try:
            # Start background processing thread
            self._start_processing_thread()
            
            # Initialize integration strategies
            self._initialize_integration_strategies()
            
            self.initialized = True
            logger.info("✅ Multi-Modal Knowledge Integration system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize multi-modal integration: {e}")
            return False
    
    def add_knowledge(self, content: Any, modality: ModalityType, 
                     metadata: Optional[Dict[str, Any]] = None, content_id: Optional[str] = None) -> str:
        """Add knowledge in specified modality to the integration system."""
        try:
            if content_id is None:
                content_id = hashlib.sha256(str(content).encode()).hexdigest()[:16]
            
            if metadata is None:
                metadata = {}
            
            # Process content using appropriate modality processor
            if modality in self.modality_processors:
                processor = self.modality_processors[modality]
                representation = processor.process(content, metadata)
                representation.source_id = content_id
                
                # Store representation
                self.representations[content_id] = representation
                
                # Update analytics
                self.integration_analytics['total_representations'] += 1
                self.integration_analytics['modalities_processed'].add(modality)
                
                # Queue for cross-modal linking
                if self.batch_processing:
                    self.processing_queue.append(('link', content_id))
                else:
                    self._create_cross_modal_links(content_id)
                
                logger.info(f"Added {modality.value} knowledge: {content_id}")
                return content_id
            else:
                logger.warning(f"No processor available for modality: {modality}")
                return ""
                
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return ""
    
    def integrate_knowledge(self, content_ids: List[str], 
                          integration_type: IntegrationType = IntegrationType.ATTENTION_FUSION) -> Dict[str, Any]:
        """Integrate knowledge from multiple representations."""
        try:
            # Get representations
            representations = []
            for content_id in content_ids:
                if content_id in self.representations:
                    representations.append(self.representations[content_id])
            
            if not representations:
                return {'error': 'No valid representations found'}
            
            start_time = time.time()
            
            # Apply integration strategy
            integration_result = self._apply_integration_strategy(representations, integration_type)
            
            # Create unified embedding
            unified_embedding = self._create_unified_embedding(representations, integration_type)
            
            # Store unified representation
            unified_id = hashlib.sha256(str(content_ids).encode()).hexdigest()[:16]
            self.unified_embeddings[unified_id] = {
                'embedding': unified_embedding,
                'source_ids': content_ids,
                'integration_type': integration_type.value,
                'creation_time': datetime.now(),
                'quality_score': integration_result.get('quality_score', 0.0)
            }
            
            processing_time = time.time() - start_time
            
            result = {
                'success': True,
                'unified_id': unified_id,
                'integration_type': integration_type.value,
                'source_representations': len(representations),
                'modalities_integrated': list(set(rep.modality.value for rep in representations)),
                'unified_embedding_dim': len(unified_embedding),
                'quality_score': integration_result.get('quality_score', 0.0),
                'processing_time': processing_time,
                **integration_result
            }
            
            logger.info(f"Integrated knowledge from {len(representations)} representations")
            return result
            
        except Exception as e:
            logger.error(f"Error integrating knowledge: {e}")
            return {'error': str(e)}
    
    def reason_across_modalities(self, query: str, content_ids: Optional[List[str]] = None,
                               reasoning_mode: ReasoningMode = ReasoningMode.ANALOGICAL) -> Dict[str, Any]:
        """Perform cross-modal reasoning."""
        try:
            # Get representations for reasoning
            if content_ids:
                representations = [self.representations[cid] for cid in content_ids if cid in self.representations]
            else:
                # Use all representations
                representations = list(self.representations.values())
            
            if not representations:
                return {'error': 'No representations available for reasoning'}
            
            # Perform reasoning
            reasoning_result = self.cross_modal_reasoner.reason_across_modalities(
                representations, query, reasoning_mode
            )
            
            # Store reasoning in history
            self.reasoning_history.append({
                'query': query,
                'reasoning_mode': reasoning_mode.value,
                'representations_count': len(representations),
                'result': reasoning_result,
                'timestamp': datetime.now()
            })
            
            # Update analytics
            self.integration_analytics['reasoning_sessions'] += 1
            
            return reasoning_result
            
        except Exception as e:
            logger.error(f"Error in cross-modal reasoning: {e}")
            return {'error': str(e)}
    
    def _apply_integration_strategy(self, representations: List[ModalRepresentation], 
                                  integration_type: IntegrationType) -> Dict[str, Any]:
        """Apply specific integration strategy."""
        if integration_type == IntegrationType.EARLY_FUSION:
            return self._early_fusion(representations)
        elif integration_type == IntegrationType.LATE_FUSION:
            return self._late_fusion(representations)
        elif integration_type == IntegrationType.ATTENTION_FUSION:
            return self._attention_fusion(representations)
        elif integration_type == IntegrationType.HIERARCHICAL_FUSION:
            return self._hierarchical_fusion(representations)
        else:
            return self._early_fusion(representations)  # Default
    
    def _early_fusion(self, representations: List[ModalRepresentation]) -> Dict[str, Any]:
        """Early fusion integration strategy."""
        # Concatenate embeddings
        embeddings = [rep.embedding for rep in representations if rep.embedding is not None]
        
        if not embeddings:
            return {'quality_score': 0.0, 'fusion_type': 'early'}
        
        # Concatenate and normalize
        fused_embedding = np.concatenate(embeddings)
        norm = np.linalg.norm(fused_embedding)
        if norm > 0:
            fused_embedding = fused_embedding / norm
        
        # Calculate quality based on individual representation qualities
        quality_score = np.mean([rep.quality_score for rep in representations])
        
        return {
            'fused_embedding': fused_embedding,
            'quality_score': quality_score,
            'fusion_type': 'early',
            'modalities_fused': [rep.modality.value for rep in representations]
        }
    
    def _late_fusion(self, representations: List[ModalRepresentation]) -> Dict[str, Any]:
        """Late fusion integration strategy."""
        # Average embeddings
        embeddings = [rep.embedding for rep in representations if rep.embedding is not None]
        
        if not embeddings:
            return {'quality_score': 0.0, 'fusion_type': 'late'}
        
        # Average and normalize
        fused_embedding = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(fused_embedding)
        if norm > 0:
            fused_embedding = fused_embedding / norm
        
        # Quality based on consensus
        pairwise_similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                pairwise_similarities.append(sim)
        
        quality_score = np.mean(pairwise_similarities) if pairwise_similarities else 0.0
        
        return {
            'fused_embedding': fused_embedding,
            'quality_score': quality_score,
            'fusion_type': 'late',
            'consensus_score': quality_score,
            'modalities_fused': [rep.modality.value for rep in representations]
        }
    
    def _attention_fusion(self, representations: List[ModalRepresentation]) -> Dict[str, Any]:
        """Attention-based fusion strategy."""
        embeddings = [rep.embedding for rep in representations if rep.embedding is not None]
        
        if not embeddings:
            return {'quality_score': 0.0, 'fusion_type': 'attention'}
        
        # Calculate attention weights based on quality and confidence
        attention_weights = []
        for rep in representations:
            if rep.embedding is not None:
                weight = rep.quality_score * rep.confidence
                attention_weights.append(weight)
        
        # Normalize attention weights
        attention_weights = np.array(attention_weights)
        attention_weights = attention_weights / np.sum(attention_weights)
        
        # Weighted average
        fused_embedding = np.zeros_like(embeddings[0])
        for i, embedding in enumerate(embeddings):
            fused_embedding += attention_weights[i] * embedding
        
        # Normalize
        norm = np.linalg.norm(fused_embedding)
        if norm > 0:
            fused_embedding = fused_embedding / norm
        
        quality_score = np.sum(attention_weights * np.array([rep.quality_score for rep in representations if rep.embedding is not None]))
        
        return {
            'fused_embedding': fused_embedding,
            'quality_score': quality_score,
            'fusion_type': 'attention',
            'attention_weights': attention_weights.tolist(),
            'modalities_fused': [rep.modality.value for rep in representations if rep.embedding is not None]
        }
    
    def _hierarchical_fusion(self, representations: List[ModalRepresentation]) -> Dict[str, Any]:
        """Hierarchical fusion strategy."""
        # Group by modality type
        modality_groups = defaultdict(list)
        for rep in representations:
            if rep.embedding is not None:
                modality_groups[rep.modality].append(rep)
        
        # Fuse within each modality first
        modality_embeddings = []
        modality_qualities = []
        
        for modality, reps in modality_groups.items():
            if len(reps) == 1:
                modality_embeddings.append(reps[0].embedding)
                modality_qualities.append(reps[0].quality_score)
            else:
                # Average within modality
                embeddings = [rep.embedding for rep in reps]
                avg_embedding = np.mean(embeddings, axis=0)
                modality_embeddings.append(avg_embedding)
                modality_qualities.append(np.mean([rep.quality_score for rep in reps]))
        
        if not modality_embeddings:
            return {'quality_score': 0.0, 'fusion_type': 'hierarchical'}
        
        # Fuse across modalities with quality-based weighting
        quality_weights = np.array(modality_qualities)
        quality_weights = quality_weights / np.sum(quality_weights)
        
        fused_embedding = np.zeros_like(modality_embeddings[0])
        for i, embedding in enumerate(modality_embeddings):
            fused_embedding += quality_weights[i] * embedding
        
        # Normalize
        norm = np.linalg.norm(fused_embedding)
        if norm > 0:
            fused_embedding = fused_embedding / norm
        
        quality_score = np.mean(modality_qualities)
        
        return {
            'fused_embedding': fused_embedding,
            'quality_score': quality_score,
            'fusion_type': 'hierarchical',
            'modality_weights': quality_weights.tolist(),
            'modalities_fused': list(modality_groups.keys())
        }
    
    def _create_unified_embedding(self, representations: List[ModalRepresentation], 
                                integration_type: IntegrationType) -> np.ndarray:
        """Create unified embedding from representations."""
        integration_result = self._apply_integration_strategy(representations, integration_type)
        return integration_result.get('fused_embedding', np.zeros(self.embedding_dim))
    
    def _create_cross_modal_links(self, content_id: str):
        """Create cross-modal links for a representation."""
        if content_id not in self.representations:
            return
        
        new_rep = self.representations[content_id]
        
        # Find links with existing representations
        for existing_id, existing_rep in self.representations.items():
            if existing_id != content_id and existing_rep.modality != new_rep.modality:
                # Calculate link strength
                link_strength = self._calculate_link_strength(new_rep, existing_rep)
                
                if link_strength > 0.3:  # Threshold for meaningful link
                    link = CrossModalLink(
                        source_modality=new_rep.modality,
                        target_modality=existing_rep.modality,
                        source_id=content_id,
                        target_id=existing_id,
                        link_strength=link_strength,
                        link_type='semantic_similarity',
                        semantic_similarity=link_strength
                    )
                    
                    self.cross_modal_links.append(link)
                    self.integration_analytics['cross_modal_links'] += 1
    
    def _calculate_link_strength(self, rep1: ModalRepresentation, rep2: ModalRepresentation) -> float:
        """Calculate strength of cross-modal link."""
        if rep1.embedding is None or rep2.embedding is None:
            return 0.0
        
        # Semantic similarity
        semantic_sim = cosine_similarity([rep1.embedding], [rep2.embedding])[0][0]
        
        # Confidence factor
        confidence_factor = (rep1.confidence + rep2.confidence) / 2.0
        
        # Temporal proximity
        time_diff = abs((rep1.timestamp - rep2.timestamp).total_seconds())
        temporal_factor = 1.0 / (1.0 + time_diff / 3600)  # Decay over hours
        
        # Combined strength
        link_strength = semantic_sim * confidence_factor * temporal_factor
        
        return link_strength
    
    def _start_processing_thread(self):
        """Start background processing thread."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_running = True
            self.processing_thread = threading.Thread(target=self._processing_worker)
            self.processing_thread.daemon = True
            self.processing_thread.start()
    
    def _processing_worker(self):
        """Background worker for processing cross-modal links."""
        while self.processing_running:
            try:
                # Process queued operations
                while self.processing_queue:
                    operation, data = self.processing_queue.popleft()
                    
                    if operation == 'link':
                        self._create_cross_modal_links(data)
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in processing worker: {e}")
                time.sleep(30)
    
    def _initialize_integration_strategies(self):
        """Initialize integration strategies."""
        self.integration_strategies = {
            IntegrationType.EARLY_FUSION: "Concatenate embeddings from all modalities",
            IntegrationType.LATE_FUSION: "Average embeddings with consensus scoring", 
            IntegrationType.ATTENTION_FUSION: "Quality and confidence-weighted fusion",
            IntegrationType.HIERARCHICAL_FUSION: "Within-modality then cross-modality fusion"
        }
    
    def get_integration_insights(self) -> Dict[str, Any]:
        """Get comprehensive multi-modal integration insights."""
        if not self.initialized:
            return {'error': 'Multi-modal integration system not initialized'}
        
        # Modality distribution
        modality_counts = defaultdict(int)
        for rep in self.representations.values():
            modality_counts[rep.modality.value] += 1
        
        # Link analysis
        link_types = defaultdict(int)
        for link in self.cross_modal_links:
            link_types[f"{link.source_modality.value}->{link.target_modality.value}"] += 1
        
        # Quality analysis
        quality_scores = [rep.quality_score for rep in self.representations.values()]
        confidence_scores = [rep.confidence for rep in self.representations.values()]
        
        # Processing performance
        processing_times = [rep.processing_time for rep in self.representations.values()]
        
        return {
            'system_status': {
                'initialized': self.initialized,
                'processing_running': self.processing_running,
                'cache_utilization': len(self.processing_cache) / self.cache_size_limit
            },
            'knowledge_statistics': {
                'total_representations': len(self.representations),
                'modality_distribution': dict(modality_counts),
                'cross_modal_links': len(self.cross_modal_links),
                'unified_embeddings': len(self.unified_embeddings)
            },
            'quality_metrics': {
                'average_quality': statistics.mean(quality_scores) if quality_scores else 0.0,
                'average_confidence': statistics.mean(confidence_scores) if confidence_scores else 0.0,
                'quality_std': statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0.0
            },
            'performance_metrics': {
                'average_processing_time': statistics.mean(processing_times) if processing_times else 0.0,
                'processing_queue_size': len(self.processing_queue),
                'reasoning_sessions': len(self.reasoning_history)
            },
            'integration_analytics': self.integration_analytics,
            'link_analysis': {
                'link_types': dict(link_types),
                'average_link_strength': statistics.mean([link.calculate_overall_strength() for link in self.cross_modal_links]) if self.cross_modal_links else 0.0
            },
            'available_modalities': list(self.modality_processors.keys()),
            'integration_strategies': list(self.integration_strategies.keys())
        }
    
    def cleanup(self):
        """Clean up resources and stop background threads."""
        self.processing_running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
        
        logger.info("Multi-Modal Knowledge Integration system cleaned up")