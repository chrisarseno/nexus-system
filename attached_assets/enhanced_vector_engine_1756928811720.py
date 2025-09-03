"""
Enhanced RAG Vector Engine
Revolutionary retrieval-augmented generation with advanced vector operations and semantic search.
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
from sklearn.cluster import KMeans
import re

logger = logging.getLogger(__name__)

class VectorSpace(Enum):
    """Types of vector spaces for different data modalities."""
    SEMANTIC = "semantic"
    SYNTACTIC = "syntactic"
    MULTIMODAL = "multimodal"
    KNOWLEDGE = "knowledge"
    TEMPORAL = "temporal"
    HIERARCHICAL = "hierarchical"

class ChunkingStrategy(Enum):
    """Strategies for text chunking."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC_BOUNDARY = "semantic_boundary"
    SLIDING_WINDOW = "sliding_window"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"
    CONTEXT_AWARE = "context_aware"

class RetrievalMethod(Enum):
    """Retrieval methods for vector search."""
    COSINE_SIMILARITY = "cosine_similarity"
    EUCLIDEAN_DISTANCE = "euclidean_distance"
    DOT_PRODUCT = "dot_product"
    HYBRID_SEARCH = "hybrid_search"
    SEMANTIC_FUSION = "semantic_fusion"
    CONTEXTUAL_RANKING = "contextual_ranking"

@dataclass
class VectorDocument:
    """Represents a document with vector embeddings and metadata."""
    doc_id: str
    content: str
    vector_embedding: np.ndarray
    metadata: Dict[str, Any]
    chunk_index: int = 0
    parent_doc_id: Optional[str] = None
    creation_time: datetime = None
    last_accessed: datetime = None
    access_count: int = 0
    relevance_score: float = 0.0
    semantic_cluster: Optional[int] = None
    
    def __post_init__(self):
        if self.creation_time is None:
            self.creation_time = datetime.now()
        if self.last_accessed is None:
            self.last_accessed = datetime.now()
    
    def update_access(self, relevance_boost: float = 0.0):
        """Update access statistics and relevance."""
        self.last_accessed = datetime.now()
        self.access_count += 1
        self.relevance_score = min(1.0, self.relevance_score + relevance_boost)
    
    def calculate_importance(self) -> float:
        """Calculate document importance based on multiple factors."""
        # Recency factor
        hours_since_access = (datetime.now() - self.last_accessed).total_seconds() / 3600
        recency_score = 1.0 / (1.0 + hours_since_access / 24)
        
        # Frequency factor
        frequency_score = min(1.0, self.access_count / 100.0)
        
        # Content quality factor (length-based heuristic)
        quality_score = min(1.0, len(self.content) / 1000.0)
        
        # Combine factors
        importance = (
            0.3 * recency_score + 
            0.3 * frequency_score + 
            0.2 * self.relevance_score + 
            0.2 * quality_score
        )
        
        return min(1.0, max(0.0, importance))

@dataclass
class SearchResult:
    """Represents a search result with scoring information."""
    document: VectorDocument
    similarity_score: float
    rank: int
    explanation: str = ""
    fusion_scores: Dict[str, float] = None
    
    def __post_init__(self):
        if self.fusion_scores is None:
            self.fusion_scores = {}

class SemanticChunker:
    """Advanced semantic chunking for optimal context preservation."""
    
    def __init__(self, max_chunk_size: int = 1000, overlap_size: int = 200):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.sentence_patterns = [
            r'[.!?]+\s+',
            r'[.!?]+$',
            r'\n\s*\n',
            r';\s+',
            r':\s+'
        ]
        
    def chunk_text(self, text: str, strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC_BOUNDARY) -> List[str]:
        """Chunk text using specified strategy."""
        if strategy == ChunkingStrategy.FIXED_SIZE:
            return self._fixed_size_chunks(text)
        elif strategy == ChunkingStrategy.SEMANTIC_BOUNDARY:
            return self._semantic_boundary_chunks(text)
        elif strategy == ChunkingStrategy.SLIDING_WINDOW:
            return self._sliding_window_chunks(text)
        elif strategy == ChunkingStrategy.HIERARCHICAL:
            return self._hierarchical_chunks(text)
        elif strategy == ChunkingStrategy.ADAPTIVE:
            return self._adaptive_chunks(text)
        elif strategy == ChunkingStrategy.CONTEXT_AWARE:
            return self._context_aware_chunks(text)
        else:
            return self._semantic_boundary_chunks(text)
    
    def _fixed_size_chunks(self, text: str) -> List[str]:
        """Create fixed-size chunks with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.max_chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end >= len(text):
                break
            
            start = end - self.overlap_size
        
        return chunks
    
    def _semantic_boundary_chunks(self, text: str) -> List[str]:
        """Create chunks at semantic boundaries."""
        chunks = []
        current_chunk = ""
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        for sentence in sentences:
            # Check if adding this sentence would exceed max size
            if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + sentence
            else:
                current_chunk += sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _sliding_window_chunks(self, text: str) -> List[str]:
        """Create overlapping sliding window chunks."""
        chunks = []
        window_size = self.max_chunk_size
        step_size = window_size - self.overlap_size
        
        for i in range(0, len(text), step_size):
            chunk = text[i:i + window_size]
            if len(chunk.strip()) > 0:
                chunks.append(chunk)
                
            if i + window_size >= len(text):
                break
        
        return chunks
    
    def _hierarchical_chunks(self, text: str) -> List[str]:
        """Create hierarchical chunks based on document structure."""
        chunks = []
        
        # Split by major structural elements
        sections = re.split(r'\n\s*\n\s*\n', text)
        
        for section in sections:
            if len(section) <= self.max_chunk_size:
                chunks.append(section.strip())
            else:
                # Further split large sections
                sub_chunks = self._semantic_boundary_chunks(section)
                chunks.extend(sub_chunks)
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _adaptive_chunks(self, text: str) -> List[str]:
        """Create adaptive chunks based on content complexity."""
        chunks = []
        
        # Analyze text complexity
        complexity_score = self._calculate_complexity(text)
        
        # Adjust chunk size based on complexity
        if complexity_score > 0.8:
            adjusted_size = int(self.max_chunk_size * 0.7)  # Smaller chunks for complex text
        elif complexity_score < 0.3:
            adjusted_size = int(self.max_chunk_size * 1.3)  # Larger chunks for simple text
        else:
            adjusted_size = self.max_chunk_size
        
        # Use semantic chunking with adjusted size
        old_max_size = self.max_chunk_size
        self.max_chunk_size = adjusted_size
        chunks = self._semantic_boundary_chunks(text)
        self.max_chunk_size = old_max_size
        
        return chunks
    
    def _context_aware_chunks(self, text: str) -> List[str]:
        """Create context-aware chunks that preserve meaning."""
        chunks = []
        
        # Identify important phrases and entities
        important_phrases = self._identify_key_phrases(text)
        
        sentences = self._split_sentences(text)
        current_chunk = ""
        
        for sentence in sentences:
            # Check if sentence contains important phrases
            has_important_phrase = any(phrase in sentence for phrase in important_phrases)
            
            # Be more conservative about splitting if important phrases are involved
            size_threshold = self.max_chunk_size * 0.8 if has_important_phrase else self.max_chunk_size
            
            if len(current_chunk) + len(sentence) > size_threshold and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Preserve context with larger overlap for important content
                overlap_size = self.overlap_size * 2 if has_important_phrase else self.overlap_size
                overlap_text = self._get_overlap_text(current_chunk, overlap_size)
                current_chunk = overlap_text + sentence
            else:
                current_chunk += sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using multiple patterns."""
        sentences = [text]
        
        for pattern in self.sentence_patterns:
            new_sentences = []
            for sentence in sentences:
                parts = re.split(pattern, sentence)
                new_sentences.extend(parts)
            sentences = new_sentences
        
        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _get_overlap_text(self, text: str, overlap_size: Optional[int] = None) -> str:
        """Get overlap text from the end of current chunk."""
        if overlap_size is None:
            overlap_size = self.overlap_size
            
        if len(text) <= overlap_size:
            return text
        
        # Try to find a good breaking point within overlap
        overlap_text = text[-overlap_size:]
        
        # Find the last sentence boundary in overlap
        for pattern in self.sentence_patterns:
            matches = list(re.finditer(pattern, overlap_text))
            if matches:
                last_match = matches[-1]
                return overlap_text[last_match.end():]
        
        return overlap_text
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score."""
        # Simple complexity metrics
        avg_word_length = np.mean([len(word) for word in text.split()])
        avg_sentence_length = len(text.split()) / max(1, len(self._split_sentences(text)))
        unique_word_ratio = len(set(text.lower().split())) / max(1, len(text.split()))
        
        # Normalize and combine
        complexity = (
            min(1.0, avg_word_length / 10.0) * 0.3 +
            min(1.0, avg_sentence_length / 30.0) * 0.4 +
            unique_word_ratio * 0.3
        )
        
        return complexity
    
    def _identify_key_phrases(self, text: str) -> List[str]:
        """Identify key phrases in text."""
        # Simple key phrase identification
        words = text.split()
        
        # Find repeated phrases
        phrase_counts = defaultdict(int)
        
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            phrase_counts[bigram] += 1
            
            if i < len(words) - 2:
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                phrase_counts[trigram] += 1
        
        # Return phrases that appear multiple times
        key_phrases = [phrase for phrase, count in phrase_counts.items() if count > 1]
        return key_phrases[:10]  # Limit to top 10

class VectorIndex:
    """High-performance vector index with multiple search strategies."""
    
    def __init__(self, dimension: int, index_type: str = "flat"):
        self.dimension = dimension
        self.index_type = index_type
        self.vectors = []
        self.doc_ids = []
        self.metadata = []
        
        # Performance optimizations
        self.cluster_centers = None
        self.cluster_assignments = None
        self.pca_model = None
        self.use_clustering = True
        self.use_pca = False
        self.n_clusters = 100
        
        # Statistics
        self.total_searches = 0
        self.total_search_time = 0.0
        self.cache_hits = 0
        self.search_cache = {}
        self.cache_size_limit = 1000
        
    def add_vectors(self, vectors: np.ndarray, doc_ids: List[str], metadata: List[Dict[str, Any]]):
        """Add vectors to the index."""
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)
        
        self.vectors.extend(vectors)
        self.doc_ids.extend(doc_ids)
        self.metadata.extend(metadata)
        
        # Rebuild optimizations if we have enough vectors
        if len(self.vectors) >= 100 and len(self.vectors) % 100 == 0:
            self._rebuild_optimizations()
    
    def _rebuild_optimizations(self):
        """Rebuild clustering and PCA optimizations."""
        vectors_array = np.array(self.vectors)
        
        if self.use_clustering and len(self.vectors) >= self.n_clusters:
            try:
                # K-means clustering for faster search
                kmeans = KMeans(n_clusters=min(self.n_clusters, len(self.vectors) // 2), random_state=42)
                self.cluster_assignments = kmeans.fit_predict(vectors_array)
                self.cluster_centers = kmeans.cluster_centers_
                
                logger.info(f"Built {len(self.cluster_centers)} clusters for vector index")
            except Exception as e:
                logger.warning(f"Clustering failed: {e}")
                self.use_clustering = False
        
        if self.use_pca and vectors_array.shape[1] > 50:
            try:
                # PCA for dimensionality reduction
                self.pca_model = PCA(n_components=min(50, vectors_array.shape[1] // 2))
                self.pca_model.fit(vectors_array)
                
                logger.info(f"Built PCA model reducing from {vectors_array.shape[1]} to {self.pca_model.n_components_} dimensions")
            except Exception as e:
                logger.warning(f"PCA failed: {e}")
                self.use_pca = False
    
    def search(self, query_vector: np.ndarray, k: int = 10, method: RetrievalMethod = RetrievalMethod.COSINE_SIMILARITY) -> List[Tuple[int, float]]:
        """Search for similar vectors."""
        start_time = time.time()
        
        # Check cache
        cache_key = hashlib.md5(f"{query_vector.tobytes()}{k}{method.value}".encode()).hexdigest()
        if cache_key in self.search_cache:
            self.cache_hits += 1
            return self.search_cache[cache_key]
        
        if not self.vectors:
            return []
        
        vectors_array = np.array(self.vectors)
        
        if method == RetrievalMethod.COSINE_SIMILARITY:
            similarities = cosine_similarity([query_vector], vectors_array)[0]
            indices = np.argsort(similarities)[::-1][:k]
            results = [(idx, similarities[idx]) for idx in indices]
            
        elif method == RetrievalMethod.EUCLIDEAN_DISTANCE:
            distances = np.linalg.norm(vectors_array - query_vector, axis=1)
            indices = np.argsort(distances)[:k]
            # Convert distance to similarity (invert and normalize)
            max_dist = np.max(distances) if len(distances) > 0 else 1.0
            results = [(idx, 1.0 - distances[idx] / max_dist) for idx in indices]
            
        elif method == RetrievalMethod.DOT_PRODUCT:
            dot_products = np.dot(vectors_array, query_vector)
            indices = np.argsort(dot_products)[::-1][:k]
            results = [(idx, dot_products[idx]) for idx in indices]
            
        elif method == RetrievalMethod.HYBRID_SEARCH:
            results = self._hybrid_search(query_vector, vectors_array, k)
            
        else:
            # Default to cosine similarity
            similarities = cosine_similarity([query_vector], vectors_array)[0]
            indices = np.argsort(similarities)[::-1][:k]
            results = [(idx, similarities[idx]) for idx in indices]
        
        # Cache results
        if len(self.search_cache) < self.cache_size_limit:
            self.search_cache[cache_key] = results
        
        # Update statistics
        self.total_searches += 1
        self.total_search_time += time.time() - start_time
        
        return results
    
    def _hybrid_search(self, query_vector: np.ndarray, vectors_array: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """Perform hybrid search combining multiple similarity measures."""
        # Cosine similarity
        cosine_sims = cosine_similarity([query_vector], vectors_array)[0]
        
        # Euclidean distance (converted to similarity)
        euclidean_dists = np.linalg.norm(vectors_array - query_vector, axis=1)
        max_dist = np.max(euclidean_dists) if len(euclidean_dists) > 0 else 1.0
        euclidean_sims = 1.0 - euclidean_dists / max_dist
        
        # Dot product (normalized)
        dot_products = np.dot(vectors_array, query_vector)
        max_dot = np.max(dot_products) if len(dot_products) > 0 else 1.0
        min_dot = np.min(dot_products) if len(dot_products) > 0 else 0.0
        dot_sims = (dot_products - min_dot) / (max_dot - min_dot) if max_dot > min_dot else np.zeros_like(dot_products)
        
        # Combine similarities with weights
        combined_scores = (
            0.5 * cosine_sims + 
            0.3 * euclidean_sims + 
            0.2 * dot_sims
        )
        
        indices = np.argsort(combined_scores)[::-1][:k]
        return [(idx, combined_scores[idx]) for idx in indices]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index performance statistics."""
        avg_search_time = self.total_search_time / max(1, self.total_searches)
        cache_hit_rate = self.cache_hits / max(1, self.total_searches)
        
        return {
            'total_vectors': len(self.vectors),
            'dimension': self.dimension,
            'total_searches': self.total_searches,
            'average_search_time_ms': avg_search_time * 1000,
            'cache_hit_rate': cache_hit_rate,
            'clustering_enabled': self.use_clustering,
            'pca_enabled': self.use_pca,
            'n_clusters': len(self.cluster_centers) if self.cluster_centers is not None else 0
        }

class EnhancedRAGEngine:
    """
    Enhanced Retrieval-Augmented Generation Engine with advanced vector operations,
    semantic search optimization, and intelligent context management.
    """
    
    def __init__(self, embedding_dimension: int = 384, max_context_length: int = 4000):
        self.embedding_dimension = embedding_dimension
        self.max_context_length = max_context_length
        
        # Core components
        self.vector_index = VectorIndex(embedding_dimension)
        self.semantic_chunker = SemanticChunker()
        self.documents: Dict[str, VectorDocument] = {}
        
        # Multiple vector spaces for different modalities
        self.vector_spaces = {
            VectorSpace.SEMANTIC: VectorIndex(embedding_dimension),
            VectorSpace.SYNTACTIC: VectorIndex(embedding_dimension),
            VectorSpace.KNOWLEDGE: VectorIndex(embedding_dimension),
            VectorSpace.TEMPORAL: VectorIndex(embedding_dimension)
        }
        
        # Advanced features
        self.semantic_clusters = {}
        self.context_memory = deque(maxlen=1000)
        self.query_history = deque(maxlen=100)
        self.relevance_feedback = {}
        
        # Performance optimization
        self.embedding_cache = {}
        self.search_cache = {}
        self.cache_size_limit = 10000
        
        # Real-time update management
        self.update_queue = deque()
        self.incremental_updates = True
        self.last_index_update = datetime.now()
        self.update_frequency = timedelta(minutes=5)
        
        # Analytics and monitoring
        self.search_analytics = {
            'total_searches': 0,
            'average_response_time': 0.0,
            'cache_hit_rate': 0.0,
            'user_satisfaction': 0.0
        }
        
        # Background processing
        self.update_thread = None
        self.update_running = False
        self.initialized = False
        
        # Simple embedding simulation (in production, use real embeddings)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=embedding_dimension)
        
        logger.info("Enhanced RAG Vector Engine initialized")
    
    def initialize(self) -> bool:
        """Initialize the enhanced RAG engine."""
        try:
            # Start background update thread
            self._start_update_thread()
            
            self.initialized = True
            logger.info("✅ Enhanced RAG Vector Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced RAG engine: {e}")
            return False
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None, 
                    doc_id: str = None, chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC_BOUNDARY) -> List[str]:
        """Add a document to the RAG system with advanced chunking."""
        try:
            if doc_id is None:
                doc_id = hashlib.sha256(content.encode()).hexdigest()[:16]
            
            if metadata is None:
                metadata = {}
            
            # Chunk the document
            chunks = self.semantic_chunker.chunk_text(content, chunking_strategy)
            chunk_ids = []
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                
                # Generate embeddings
                embedding = self._generate_embedding(chunk)
                
                # Create vector document
                vector_doc = VectorDocument(
                    doc_id=chunk_id,
                    content=chunk,
                    vector_embedding=embedding,
                    metadata={**metadata, 'chunk_index': i, 'total_chunks': len(chunks)},
                    chunk_index=i,
                    parent_doc_id=doc_id
                )
                
                # Store document
                self.documents[chunk_id] = vector_doc
                
                # Add to multiple vector spaces
                self._add_to_vector_spaces(vector_doc)
                
                # Queue for incremental update
                if self.incremental_updates:
                    self.update_queue.append(('add', vector_doc))
                
                chunk_ids.append(chunk_id)
            
            logger.info(f"Added document {doc_id} with {len(chunks)} chunks")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return []
    
    def search(self, query: str, k: int = 10, vector_space: VectorSpace = VectorSpace.SEMANTIC,
              retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID_SEARCH,
              context_aware: bool = True) -> List[SearchResult]:
        """Perform enhanced vector search with multiple strategies."""
        try:
            start_time = time.time()
            
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Store query in history for context awareness
            self.query_history.append({
                'query': query,
                'timestamp': datetime.now(),
                'embedding': query_embedding
            })
            
            # Perform search in specified vector space
            if vector_space in self.vector_spaces:
                raw_results = self.vector_spaces[vector_space].search(
                    query_embedding, k * 2, retrieval_method  # Get more for reranking
                )
            else:
                raw_results = self.vector_index.search(query_embedding, k * 2, retrieval_method)
            
            # Convert to SearchResult objects
            search_results = []
            for rank, (doc_idx, similarity) in enumerate(raw_results):
                if doc_idx < len(self.documents):
                    doc_id = list(self.documents.keys())[doc_idx]
                    if doc_id in self.documents:
                        document = self.documents[doc_id]
                        
                        result = SearchResult(
                            document=document,
                            similarity_score=similarity,
                            rank=rank + 1,
                            explanation=f"Found via {retrieval_method.value} in {vector_space.value} space"
                        )
                        
                        search_results.append(result)
            
            # Apply advanced ranking and filtering
            if context_aware:
                search_results = self._apply_contextual_ranking(search_results, query)
            
            # Apply semantic fusion if using hybrid search
            if retrieval_method == RetrievalMethod.SEMANTIC_FUSION:
                search_results = self._apply_semantic_fusion(search_results, query_embedding)
            
            # Update document access statistics
            for result in search_results[:k]:
                result.document.update_access(0.1)  # Small relevance boost
            
            # Limit to requested number of results
            final_results = search_results[:k]
            
            # Update analytics
            search_time = time.time() - start_time
            self._update_search_analytics(search_time, len(final_results))
            
            logger.info(f"Enhanced search completed: {len(final_results)} results in {search_time:.3f}s")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in enhanced search: {e}")
            return []
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text (simplified implementation)."""
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        try:
            # Simple TF-IDF based embedding (in production, use transformer models)
            if hasattr(self.tfidf_vectorizer, 'vocabulary_') and self.tfidf_vectorizer.vocabulary_:
                # Vectorizer is fitted, use it
                embedding = self.tfidf_vectorizer.transform([text]).toarray()[0]
            else:
                # Need to fit vectorizer with some text
                corpus = [text] + [doc.content for doc in list(self.documents.values())[:100]]
                self.tfidf_vectorizer.fit(corpus)
                embedding = self.tfidf_vectorizer.transform([text]).toarray()[0]
            
            # Ensure correct dimension
            if len(embedding) < self.embedding_dimension:
                # Pad with zeros
                padded = np.zeros(self.embedding_dimension)
                padded[:len(embedding)] = embedding
                embedding = padded
            elif len(embedding) > self.embedding_dimension:
                # Truncate
                embedding = embedding[:self.embedding_dimension]
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            # Cache the result
            if len(self.embedding_cache) < self.cache_size_limit:
                self.embedding_cache[text_hash] = embedding
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            # Return random normalized vector as fallback
            embedding = np.random.normal(0, 1, self.embedding_dimension)
            return embedding / np.linalg.norm(embedding)
    
    def _add_to_vector_spaces(self, vector_doc: VectorDocument):
        """Add document to appropriate vector spaces."""
        doc_metadata = [vector_doc.metadata]
        
        # Add to semantic space (default)
        self.vector_spaces[VectorSpace.SEMANTIC].add_vectors(
            vector_doc.vector_embedding.reshape(1, -1),
            [vector_doc.doc_id],
            doc_metadata
        )
        
        # Add to knowledge space if it contains structured information
        if any(keyword in vector_doc.content.lower() for keyword in ['definition', 'concept', 'theory', 'principle']):
            self.vector_spaces[VectorSpace.KNOWLEDGE].add_vectors(
                vector_doc.vector_embedding.reshape(1, -1),
                [vector_doc.doc_id],
                doc_metadata
            )
        
        # Add to temporal space if it contains time-related information
        if any(keyword in vector_doc.content.lower() for keyword in ['date', 'time', 'year', 'month', 'day']):
            self.vector_spaces[VectorSpace.TEMPORAL].add_vectors(
                vector_doc.vector_embedding.reshape(1, -1),
                [vector_doc.doc_id],
                doc_metadata
            )
    
    def _apply_contextual_ranking(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Apply contextual ranking based on query history and document relevance."""
        if not self.query_history:
            return results
        
        # Get recent query context
        recent_queries = list(self.query_history)[-5:]  # Last 5 queries
        
        for result in results:
            # Calculate context relevance
            context_boost = 0.0
            
            # Boost based on recent query similarity
            query_embedding = self._generate_embedding(query)
            for past_query in recent_queries:
                if 'embedding' in past_query:
                    past_similarity = cosine_similarity(
                        [query_embedding],
                        [past_query['embedding']]
                    )[0][0]
                    context_boost += past_similarity * 0.1  # Small boost
            
            # Boost based on document importance
            importance_boost = result.document.calculate_importance() * 0.2
            
            # Apply boosts
            result.similarity_score += context_boost + importance_boost
            result.fusion_scores['context'] = context_boost
            result.fusion_scores['importance'] = importance_boost
        
        # Re-sort by updated scores
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
    
    def _apply_semantic_fusion(self, results: List[SearchResult], query_embedding: np.ndarray) -> List[SearchResult]:
        """Apply semantic fusion combining multiple similarity measures."""
        for result in results:
            doc_embedding = result.document.vector_embedding
            
            # Calculate multiple similarity measures
            cosine_sim = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            
            # Euclidean similarity
            euclidean_dist = np.linalg.norm(query_embedding - doc_embedding)
            euclidean_sim = 1.0 / (1.0 + euclidean_dist)
            
            # Dot product similarity
            dot_sim = np.dot(query_embedding, doc_embedding)
            
            # Combine with weights
            fused_score = (
                0.5 * cosine_sim +
                0.3 * euclidean_sim +
                0.2 * dot_sim
            )
            
            # Update result
            result.similarity_score = fused_score
            result.fusion_scores.update({
                'cosine': cosine_sim,
                'euclidean': euclidean_sim,
                'dot_product': dot_sim,
                'fused': fused_score
            })
        
        # Re-sort by fused scores
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1
            result.explanation = f"Semantic fusion ranking (fused score: {result.similarity_score:.3f})"
        
        return results
    
    def generate_context(self, search_results: List[SearchResult], query: str) -> str:
        """Generate optimized context from search results."""
        if not search_results:
            return ""
        
        context_parts = []
        total_length = 0
        
        # Sort by relevance and importance
        sorted_results = sorted(
            search_results,
            key=lambda x: x.similarity_score * x.document.calculate_importance(),
            reverse=True
        )
        
        for result in sorted_results:
            content = result.document.content.strip()
            
            # Check if adding this content would exceed limit
            if total_length + len(content) > self.max_context_length:
                # Try to include partial content
                remaining_space = self.max_context_length - total_length
                if remaining_space > 100:  # Only include if meaningful space left
                    # Find a good breaking point
                    truncated = content[:remaining_space]
                    last_sentence = truncated.rfind('.')
                    if last_sentence > remaining_space * 0.7:
                        content = truncated[:last_sentence + 1]
                    else:
                        content = truncated + "..."
                    
                    context_parts.append(content)
                break
            
            context_parts.append(content)
            total_length += len(content)
        
        # Join with separators
        context = "\n\n---\n\n".join(context_parts)
        
        # Store in context memory for future reference
        self.context_memory.append({
            'query': query,
            'context': context,
            'timestamp': datetime.now(),
            'result_count': len(search_results)
        })
        
        return context
    
    def _start_update_thread(self):
        """Start background thread for incremental updates."""
        if self.update_thread is None or not self.update_thread.is_alive():
            self.update_running = True
            self.update_thread = threading.Thread(target=self._update_worker)
            self.update_thread.daemon = True
            self.update_thread.start()
    
    def _update_worker(self):
        """Background worker for processing incremental updates."""
        while self.update_running:
            try:
                current_time = datetime.now()
                
                # Process queued updates
                if self.update_queue and current_time - self.last_index_update >= self.update_frequency:
                    self._process_update_queue()
                    self.last_index_update = current_time
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in update worker: {e}")
                time.sleep(60)
    
    def _process_update_queue(self):
        """Process queued incremental updates."""
        if not self.update_queue:
            return
        
        logger.info(f"Processing {len(self.update_queue)} incremental updates...")
        
        updates_processed = 0
        while self.update_queue and updates_processed < 100:  # Limit batch size
            try:
                operation, data = self.update_queue.popleft()
                
                if operation == 'add' and isinstance(data, VectorDocument):
                    # Add to main index
                    self.vector_index.add_vectors(
                        data.vector_embedding.reshape(1, -1),
                        [data.doc_id],
                        [data.metadata]
                    )
                
                updates_processed += 1
                
            except Exception as e:
                logger.error(f"Error processing update: {e}")
        
        logger.info(f"Processed {updates_processed} incremental updates")
    
    def _update_search_analytics(self, search_time: float, result_count: int):
        """Update search analytics."""
        self.search_analytics['total_searches'] += 1
        
        # Update average response time
        total_time = (self.search_analytics['average_response_time'] * 
                     (self.search_analytics['total_searches'] - 1) + search_time)
        self.search_analytics['average_response_time'] = total_time / self.search_analytics['total_searches']
        
        # Update cache hit rate
        total_cache_hits = sum(space.cache_hits for space in self.vector_spaces.values())
        total_searches = sum(space.total_searches for space in self.vector_spaces.values())
        self.search_analytics['cache_hit_rate'] = total_cache_hits / max(1, total_searches)
    
    def get_rag_insights(self) -> Dict[str, Any]:
        """Get comprehensive RAG system insights."""
        if not self.initialized:
            return {'error': 'Enhanced RAG engine not initialized'}
        
        # Vector space statistics
        vector_space_stats = {}
        for space_name, space_index in self.vector_spaces.items():
            vector_space_stats[space_name.value] = space_index.get_statistics()
        
        # Document statistics
        doc_stats = {
            'total_documents': len(self.documents),
            'total_chunks': len([doc for doc in self.documents.values() if doc.chunk_index >= 0]),
            'average_chunk_size': statistics.mean([len(doc.content) for doc in self.documents.values()]) if self.documents else 0,
            'most_accessed_docs': sorted(
                self.documents.items(),
                key=lambda x: x[1].access_count,
                reverse=True
            )[:5]
        }
        
        # Query analytics
        query_stats = {
            'total_queries': len(self.query_history),
            'average_results_per_query': statistics.mean([
                ctx.get('result_count', 0) for ctx in self.context_memory
            ]) if self.context_memory else 0,
            'context_cache_size': len(self.context_memory)
        }
        
        return {
            'search_analytics': self.search_analytics,
            'vector_space_statistics': vector_space_stats,
            'document_statistics': doc_stats,
            'query_analytics': query_stats,
            'cache_statistics': {
                'embedding_cache_size': len(self.embedding_cache),
                'search_cache_size': len(self.search_cache),
                'update_queue_size': len(self.update_queue)
            },
            'system_configuration': {
                'embedding_dimension': self.embedding_dimension,
                'max_context_length': self.max_context_length,
                'incremental_updates': self.incremental_updates,
                'cache_size_limit': self.cache_size_limit
            }
        }
    
    def cleanup(self):
        """Clean up resources and stop background threads."""
        self.update_running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5)
        
        logger.info("Enhanced RAG Vector Engine cleaned up")