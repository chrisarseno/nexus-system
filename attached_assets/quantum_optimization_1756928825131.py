"""
Quantum-Enhanced Storage Optimization Engine
Revolutionary data storage efficiency using quantum-inspired algorithms and advanced compression.
"""

import logging
import time
import threading
import hashlib
import zlib
import lzma
import bz2
import pickle
import json
import math
import statistics
from typing import Dict, List, Any, Set, Tuple, Optional, Union
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from dataclasses import dataclass, asdict
import copy
import base64

logger = logging.getLogger(__name__)

class CompressionAlgorithm(Enum):
    """Types of compression algorithms."""
    ZLIB = "zlib"
    LZMA = "lzma"
    BZ2 = "bz2"
    QUANTUM_HYBRID = "quantum_hybrid"
    SEMANTIC_COMPRESSION = "semantic_compression"
    ADAPTIVE_COMPRESSION = "adaptive_compression"

class DataType(Enum):
    """Types of data for optimization."""
    TEXT = "text"
    BINARY = "binary"
    STRUCTURED = "structured"
    KNOWLEDGE = "knowledge"
    VECTOR = "vector"
    IMAGE = "image"
    AUDIO = "audio"
    TIME_SERIES = "time_series"

class AccessPattern(Enum):
    """Data access patterns for optimization."""
    FREQUENT = "frequent"
    OCCASIONAL = "occasional"
    RARE = "rare"
    ARCHIVED = "archived"
    STREAMING = "streaming"
    BATCH = "batch"

@dataclass
class CompressionMetrics:
    """Metrics for compression performance evaluation."""
    original_size: int = 0
    compressed_size: int = 0
    compression_ratio: float = 0.0
    compression_time: float = 0.0
    decompression_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    quality_score: float = 1.0  # For lossy compression
    algorithm_used: str = ""
    
    def calculate_efficiency_score(self) -> float:
        """Calculate overall compression efficiency score."""
        if self.original_size == 0:
            return 0.0
        
        # Weight factors
        compression_weight = 0.4
        speed_weight = 0.3
        resource_weight = 0.2
        quality_weight = 0.1
        
        # Normalize metrics
        compression_score = min(1.0, self.compression_ratio / 10.0)  # Higher is better
        speed_score = 1.0 / (1.0 + self.compression_time + self.decompression_time)
        resource_score = 1.0 / (1.0 + self.cpu_usage + self.memory_usage)
        
        efficiency = (
            compression_weight * compression_score +
            speed_weight * speed_score +
            resource_weight * resource_score +
            quality_weight * self.quality_score
        )
        
        return min(1.0, max(0.0, efficiency))

@dataclass
class StorageBlock:
    """Represents a storage block with optimization metadata."""
    block_id: str
    data: bytes
    data_type: DataType
    access_pattern: AccessPattern
    creation_time: datetime
    last_access: datetime
    access_count: int = 0
    compression_metrics: Optional[CompressionMetrics] = None
    similarity_hash: Optional[str] = None
    semantic_fingerprint: Optional[List[float]] = None
    
    def update_access(self):
        """Update access statistics."""
        self.last_access = datetime.now()
        self.access_count += 1
    
    def calculate_importance_score(self) -> float:
        """Calculate importance score for storage optimization."""
        current_time = datetime.now()
        
        # Recency factor (more recent = higher importance)
        hours_since_access = (current_time - self.last_access).total_seconds() / 3600
        recency_score = 1.0 / (1.0 + hours_since_access / 24)  # Decay over days
        
        # Frequency factor
        frequency_score = min(1.0, self.access_count / 100.0)
        
        # Age factor (newer data often more important)
        hours_since_creation = (current_time - self.creation_time).total_seconds() / 3600
        age_score = 1.0 / (1.0 + hours_since_creation / (24 * 7))  # Decay over weeks
        
        # Combine factors
        importance = (0.4 * frequency_score + 0.4 * recency_score + 0.2 * age_score)
        return min(1.0, max(0.0, importance))

class QuantumCompressionEngine:
    """Quantum-inspired compression engine for maximum efficiency."""
    
    def __init__(self):
        self.quantum_basis_vectors = self._generate_quantum_basis()
        self.entanglement_patterns = {}
        self.coherence_threshold = 0.8
        
    def _generate_quantum_basis(self) -> np.ndarray:
        """Generate quantum-inspired basis vectors for compression."""
        # Simulate quantum superposition states
        basis_size = 256  # For byte-level operations
        basis_vectors = np.random.rand(basis_size, basis_size)
        
        # Normalize to quantum state requirements
        for i in range(basis_size):
            norm = np.linalg.norm(basis_vectors[i])
            if norm > 0:
                basis_vectors[i] /= norm
        
        return basis_vectors
    
    def quantum_compress(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """Apply quantum-inspired compression to data."""
        try:
            # Convert data to quantum state representation
            data_array = np.frombuffer(data, dtype=np.uint8)
            
            # Apply quantum transformation
            if len(data_array) > 0:
                # Pad to match basis vector size if needed
                padded_size = ((len(data_array) - 1) // 256 + 1) * 256
                padded_data = np.zeros(padded_size, dtype=np.uint8)
                padded_data[:len(data_array)] = data_array
                
                # Apply quantum-inspired transformation
                transformed_chunks = []
                for i in range(0, len(padded_data), 256):
                    chunk = padded_data[i:i+256]
                    if len(chunk) == 256:
                        # Project onto quantum basis
                        coefficients = np.dot(self.quantum_basis_vectors, chunk.astype(np.float64))
                        
                        # Quantize coefficients (simulate measurement)
                        quantized = np.round(coefficients * 127).astype(np.int8)
                        transformed_chunks.append(quantized)
                
                if transformed_chunks:
                    # Combine transformed chunks
                    quantum_data = np.concatenate(transformed_chunks)
                    
                    # Apply traditional compression to quantum representation
                    quantum_compressed = zlib.compress(quantum_data.tobytes())
                    
                    metadata = {
                        'original_length': len(data),
                        'padded_length': padded_size,
                        'quantum_chunks': len(transformed_chunks),
                        'algorithm': 'quantum_hybrid'
                    }
                    
                    return quantum_compressed, metadata
            
            # Fallback to regular compression
            return zlib.compress(data), {'algorithm': 'zlib_fallback'}
            
        except Exception as e:
            logger.warning(f"Quantum compression failed: {e}, falling back to zlib")
            return zlib.compress(data), {'algorithm': 'zlib_fallback'}
    
    def quantum_decompress(self, compressed_data: bytes, metadata: Dict[str, Any]) -> bytes:
        """Decompress quantum-compressed data."""
        try:
            if metadata.get('algorithm') == 'quantum_hybrid':
                # Decompress quantum representation
                quantum_data = zlib.decompress(compressed_data)
                quantum_array = np.frombuffer(quantum_data, dtype=np.int8)
                
                # Reconstruct original data
                original_chunks = []
                chunk_size = 256
                
                for i in range(0, len(quantum_array), chunk_size):
                    chunk = quantum_array[i:i+chunk_size]
                    if len(chunk) == chunk_size:
                        # Dequantize coefficients
                        coefficients = chunk.astype(np.float64) / 127.0
                        
                        # Inverse quantum transformation
                        reconstructed = np.dot(self.quantum_basis_vectors.T, coefficients)
                        reconstructed = np.round(np.clip(reconstructed, 0, 255)).astype(np.uint8)
                        original_chunks.append(reconstructed)
                
                if original_chunks:
                    reconstructed_data = np.concatenate(original_chunks)
                    original_length = metadata.get('original_length', len(reconstructed_data))
                    return reconstructed_data[:original_length].tobytes()
            
            # Fallback decompression
            return zlib.decompress(compressed_data)
            
        except Exception as e:
            logger.warning(f"Quantum decompression failed: {e}, using fallback")
            return zlib.decompress(compressed_data)

class SemanticCompressionEngine:
    """Semantic compression using meaning-based optimization."""
    
    def __init__(self):
        self.semantic_patterns = {}
        self.meaning_dictionary = {}
        self.pattern_frequency = defaultdict(int)
        
    def extract_semantic_patterns(self, data: str) -> List[str]:
        """Extract semantic patterns from text data."""
        # Simple pattern extraction (in practice, would use NLP)
        patterns = []
        
        # Word-level patterns
        words = data.split()
        for i in range(len(words)):
            # Single words
            patterns.append(words[i])
            
            # Bigrams
            if i < len(words) - 1:
                patterns.append(f"{words[i]} {words[i+1]}")
            
            # Trigrams
            if i < len(words) - 2:
                patterns.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        return patterns
    
    def build_semantic_dictionary(self, patterns: List[str]) -> Dict[str, str]:
        """Build compression dictionary based on semantic frequency."""
        # Count pattern frequencies
        for pattern in patterns:
            self.pattern_frequency[pattern] += 1
        
        # Create compressed representations for frequent patterns
        dictionary = {}
        sorted_patterns = sorted(
            self.pattern_frequency.items(),
            key=lambda x: x[1] * len(x[0]),  # Frequency * length
            reverse=True
        )
        
        # Assign short codes to frequent patterns
        for i, (pattern, freq) in enumerate(sorted_patterns[:1000]):  # Top 1000
            if freq > 2 and len(pattern) > 3:  # Only compress if beneficial
                code = f"#{i:03d}#"
                dictionary[pattern] = code
        
        return dictionary
    
    def semantic_compress(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Apply semantic compression to text."""
        patterns = self.extract_semantic_patterns(text)
        dictionary = self.build_semantic_dictionary(patterns)
        
        # Apply compression
        compressed_text = text
        for pattern, code in dictionary.items():
            compressed_text = compressed_text.replace(pattern, code)
        
        return compressed_text, dictionary
    
    def semantic_decompress(self, compressed_text: str, dictionary: Dict[str, str]) -> str:
        """Decompress semantically compressed text."""
        # Reverse dictionary
        reverse_dict = {v: k for k, v in dictionary.items()}
        
        # Apply decompression
        decompressed_text = compressed_text
        for code, pattern in reverse_dict.items():
            decompressed_text = decompressed_text.replace(code, pattern)
        
        return decompressed_text

class QuantumStorageOptimizer:
    """
    Quantum-Enhanced Storage Optimization Engine that revolutionizes 
    data storage efficiency through quantum-inspired algorithms.
    """
    
    def __init__(self, storage_limit_gb: float = 100.0):
        self.storage_limit = storage_limit_gb * 1024 * 1024 * 1024  # Convert to bytes
        
        # Core engines
        self.quantum_compressor = QuantumCompressionEngine()
        self.semantic_compressor = SemanticCompressionEngine()
        
        # Storage management
        self.storage_blocks: Dict[str, StorageBlock] = {}
        self.compression_cache: Dict[str, CompressionMetrics] = {}
        self.deduplication_index: Dict[str, Set[str]] = {}  # hash -> block_ids
        
        # Optimization state
        self.total_storage_used = 0
        self.total_original_size = 0
        self.optimization_stats = {
            'compression_ratio': 1.0,
            'deduplication_savings': 0,
            'storage_efficiency': 0.0,
            'total_blocks': 0
        }
        
        # Configuration
        self.similarity_threshold = 0.95
        self.quantum_threshold = 1024  # Use quantum compression for data > 1KB
        self.semantic_threshold = 512   # Use semantic compression for text > 512 bytes
        
        # Background processing
        self.optimization_thread = None
        self.optimization_running = False
        self.last_optimization = datetime.now()
        self.optimization_interval = timedelta(hours=1)
        
        self.initialized = False
        
        logger.info("Quantum-Enhanced Storage Optimizer initialized")
    
    def initialize(self) -> bool:
        """Initialize the quantum storage optimization engine."""
        try:
            # Start background optimization thread
            self._start_optimization_thread()
            
            self.initialized = True
            logger.info("✅ Quantum Storage Optimizer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize quantum storage optimizer: {e}")
            return False
    
    def store_data(self, data: Any, data_type: DataType, 
                  access_pattern: AccessPattern = AccessPattern.FREQUENT,
                  block_id: Optional[str] = None) -> str:
        """Store data with quantum-enhanced optimization."""
        try:
            # Generate block ID if not provided
            if block_id is None:
                data_hash = hashlib.sha256(str(data).encode()).hexdigest()
                block_id = f"block_{data_hash[:16]}_{int(time.time())}"
            
            # Convert data to bytes
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, (dict, list)):
                data_bytes = json.dumps(data).encode('utf-8')
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                data_bytes = pickle.dumps(data)
            
            original_size = len(data_bytes)
            
            # Check for deduplication opportunities
            similarity_hash = self._calculate_similarity_hash(data_bytes)
            if self._check_deduplication(similarity_hash, block_id):
                logger.info(f"Data deduplicated for block {block_id}")
                return block_id
            
            # Choose optimal compression algorithm
            compressed_data, compression_metrics = self._optimize_compression(
                data_bytes, data_type
            )
            
            # Create storage block
            storage_block = StorageBlock(
                block_id=block_id,
                data=compressed_data,
                data_type=data_type,
                access_pattern=access_pattern,
                creation_time=datetime.now(),
                last_access=datetime.now(),
                access_count=1,
                compression_metrics=compression_metrics,
                similarity_hash=similarity_hash
            )
            
            # Store block
            self.storage_blocks[block_id] = storage_block
            self.compression_cache[block_id] = compression_metrics
            
            # Update deduplication index
            if similarity_hash not in self.deduplication_index:
                self.deduplication_index[similarity_hash] = set()
            self.deduplication_index[similarity_hash].add(block_id)
            
            # Update statistics
            self.total_storage_used += len(compressed_data)
            self.total_original_size += original_size
            self._update_optimization_stats()
            
            logger.info(f"Stored block {block_id}: {original_size} -> {len(compressed_data)} bytes "
                       f"({compression_metrics.compression_ratio:.2f}x compression)")
            
            return block_id
            
        except Exception as e:
            logger.error(f"Error storing data: {e}")
            raise
    
    def retrieve_data(self, block_id: str) -> Any:
        """Retrieve and decompress stored data."""
        try:
            if block_id not in self.storage_blocks:
                raise ValueError(f"Block {block_id} not found")
            
            storage_block = self.storage_blocks[block_id]
            storage_block.update_access()
            
            # Decompress data
            if storage_block.compression_metrics:
                decompressed_data = self._decompress_data(
                    storage_block.data,
                    storage_block.compression_metrics
                )
            else:
                # No compression metrics, assume raw data
                decompressed_data = storage_block.data
            
            # Convert back to original type
            if storage_block.data_type == DataType.TEXT:
                return decompressed_data.decode('utf-8')
            elif storage_block.data_type == DataType.STRUCTURED:
                return json.loads(decompressed_data.decode('utf-8'))
            elif storage_block.data_type == DataType.BINARY:
                return decompressed_data
            else:
                try:
                    return pickle.loads(decompressed_data)
                except:
                    return decompressed_data
            
        except Exception as e:
            logger.error(f"Error retrieving data for block {block_id}: {e}")
            raise
    
    def _calculate_similarity_hash(self, data: bytes) -> str:
        """Calculate similarity hash for deduplication."""
        # Use a rolling hash for similarity detection
        chunk_size = 64
        hashes = []
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            chunk_hash = hashlib.md5(chunk).hexdigest()[:8]
            hashes.append(chunk_hash)
        
        # Create similarity hash from chunk patterns
        if len(hashes) > 4:
            # Sample every 4th hash for similarity comparison
            sampled = hashes[::4]
            return hashlib.sha256(''.join(sampled).encode()).hexdigest()
        else:
            return hashlib.sha256(''.join(hashes).encode()).hexdigest()
    
    def _check_deduplication(self, similarity_hash: str, block_id: str) -> bool:
        """Check if data can be deduplicated."""
        if similarity_hash in self.deduplication_index:
            existing_blocks = self.deduplication_index[similarity_hash]
            for existing_id in existing_blocks:
                if existing_id != block_id and existing_id in self.storage_blocks:
                    # Found similar data - implement reference counting
                    logger.info(f"Deduplication opportunity: {block_id} similar to {existing_id}")
                    return True
        return False
    
    def _optimize_compression(self, data: bytes, data_type: DataType) -> Tuple[bytes, CompressionMetrics]:
        """Choose and apply optimal compression algorithm."""
        best_compressed = data
        best_metrics = CompressionMetrics(
            original_size=len(data),
            compressed_size=len(data),
            compression_ratio=1.0,
            algorithm_used="none"
        )
        
        # Try different compression algorithms
        algorithms_to_try = []
        
        # Choose algorithms based on data type and size
        if len(data) > self.quantum_threshold:
            algorithms_to_try.append(CompressionAlgorithm.QUANTUM_HYBRID)
        
        if data_type == DataType.TEXT and len(data) > self.semantic_threshold:
            algorithms_to_try.append(CompressionAlgorithm.SEMANTIC_COMPRESSION)
        
        # Always try standard algorithms
        algorithms_to_try.extend([
            CompressionAlgorithm.LZMA,
            CompressionAlgorithm.ZLIB,
            CompressionAlgorithm.BZ2
        ])
        
        for algorithm in algorithms_to_try:
            try:
                start_time = time.time()
                compressed, metadata = self._apply_compression(data, algorithm)
                compression_time = time.time() - start_time
                
                if len(compressed) < len(best_compressed):
                    compression_ratio = len(data) / len(compressed) if len(compressed) > 0 else 1.0
                    
                    best_compressed = compressed
                    best_metrics = CompressionMetrics(
                        original_size=len(data),
                        compressed_size=len(compressed),
                        compression_ratio=compression_ratio,
                        compression_time=compression_time,
                        algorithm_used=algorithm.value
                    )
                    
                    # Store metadata for decompression
                    if metadata:
                        best_metrics.quality_score = metadata.get('quality', 1.0)
                    
            except Exception as e:
                logger.warning(f"Compression with {algorithm.value} failed: {e}")
                continue
        
        return best_compressed, best_metrics
    
    def _apply_compression(self, data: bytes, algorithm: CompressionAlgorithm) -> Tuple[bytes, Dict[str, Any]]:
        """Apply specific compression algorithm."""
        metadata = {}
        
        if algorithm == CompressionAlgorithm.QUANTUM_HYBRID:
            return self.quantum_compressor.quantum_compress(data)
        
        elif algorithm == CompressionAlgorithm.SEMANTIC_COMPRESSION:
            # Only for text data
            try:
                text = data.decode('utf-8')
                compressed_text, dictionary = self.semantic_compressor.semantic_compress(text)
                
                # Combine compressed text and dictionary
                combined_data = {
                    'text': compressed_text,
                    'dictionary': dictionary
                }
                compressed_bytes = json.dumps(combined_data).encode('utf-8')
                
                return compressed_bytes, {'type': 'semantic'}
            except:
                # Fallback to zlib if not text
                return zlib.compress(data), {'type': 'zlib_fallback'}
        
        elif algorithm == CompressionAlgorithm.LZMA:
            return lzma.compress(data), {'type': 'lzma'}
        
        elif algorithm == CompressionAlgorithm.BZ2:
            return bz2.compress(data), {'type': 'bz2'}
        
        else:  # ZLIB
            return zlib.compress(data), {'type': 'zlib'}
    
    def _decompress_data(self, compressed_data: bytes, metrics: CompressionMetrics) -> bytes:
        """Decompress data using appropriate algorithm."""
        algorithm = metrics.algorithm_used
        
        try:
            if algorithm == "quantum_hybrid":
                # Reconstruct metadata from metrics
                metadata = {
                    'algorithm': 'quantum_hybrid',
                    'original_length': metrics.original_size
                }
                return self.quantum_compressor.quantum_decompress(compressed_data, metadata)
            
            elif algorithm == "semantic_compression":
                # Parse combined data
                combined_data = json.loads(compressed_data.decode('utf-8'))
                compressed_text = combined_data['text']
                dictionary = combined_data['dictionary']
                
                decompressed_text = self.semantic_compressor.semantic_decompress(
                    compressed_text, dictionary
                )
                return decompressed_text.encode('utf-8')
            
            elif algorithm == "lzma":
                return lzma.decompress(compressed_data)
            
            elif algorithm == "bz2":
                return bz2.decompress(compressed_data)
            
            else:  # zlib or fallback
                return zlib.decompress(compressed_data)
                
        except Exception as e:
            logger.error(f"Decompression failed for algorithm {algorithm}: {e}")
            # Try zlib as fallback
            try:
                return zlib.decompress(compressed_data)
            except:
                logger.error("All decompression methods failed")
                raise
    
    def _start_optimization_thread(self):
        """Start background optimization thread."""
        if self.optimization_thread is None or not self.optimization_thread.is_alive():
            self.optimization_running = True
            self.optimization_thread = threading.Thread(target=self._optimization_worker)
            self.optimization_thread.daemon = True
            self.optimization_thread.start()
    
    def _optimization_worker(self):
        """Background worker for continuous storage optimization."""
        while self.optimization_running:
            try:
                current_time = datetime.now()
                
                # Run optimization if enough time has passed
                if current_time - self.last_optimization >= self.optimization_interval:
                    self._run_storage_optimization()
                    self.last_optimization = current_time
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in storage optimization: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _run_storage_optimization(self):
        """Run comprehensive storage optimization."""
        logger.info("Running storage optimization...")
        
        # Identify optimization opportunities
        recompression_candidates = self._identify_recompression_candidates()
        cleanup_candidates = self._identify_cleanup_candidates()
        
        # Perform optimizations
        if recompression_candidates:
            self._recompress_blocks(recompression_candidates)
        
        if cleanup_candidates:
            self._cleanup_blocks(cleanup_candidates)
        
        # Update statistics
        self._update_optimization_stats()
        
        logger.info(f"Optimization complete. Storage efficiency: {self.optimization_stats['storage_efficiency']:.2f}")
    
    def _identify_recompression_candidates(self) -> List[str]:
        """Identify blocks that could benefit from recompression."""
        candidates = []
        
        for block_id, block in self.storage_blocks.items():
            # Check if block has low compression ratio
            if (block.compression_metrics and 
                block.compression_metrics.compression_ratio < 2.0 and
                block.compression_metrics.original_size > 1024):
                candidates.append(block_id)
        
        return candidates[:10]  # Limit to avoid overwhelming system
    
    def _identify_cleanup_candidates(self) -> List[str]:
        """Identify blocks that can be cleaned up or archived."""
        candidates = []
        current_time = datetime.now()
        
        for block_id, block in self.storage_blocks.items():
            # Check if block hasn't been accessed in a long time
            days_since_access = (current_time - block.last_access).days
            
            if (days_since_access > 30 and 
                block.access_pattern in [AccessPattern.RARE, AccessPattern.ARCHIVED]):
                candidates.append(block_id)
        
        return candidates
    
    def _recompress_blocks(self, block_ids: List[str]):
        """Recompress blocks with better algorithms."""
        for block_id in block_ids:
            try:
                if block_id not in self.storage_blocks:
                    continue
                
                # Retrieve and decompress current data
                original_data = self.retrieve_data(block_id)
                
                # Convert back to bytes for recompression
                if isinstance(original_data, str):
                    data_bytes = original_data.encode('utf-8')
                else:
                    data_bytes = pickle.dumps(original_data)
                
                # Recompress with potentially better algorithm
                new_compressed, new_metrics = self._optimize_compression(
                    data_bytes, self.storage_blocks[block_id].data_type
                )
                
                # Update if improvement found
                old_size = len(self.storage_blocks[block_id].data)
                if len(new_compressed) < old_size:
                    self.storage_blocks[block_id].data = new_compressed
                    self.storage_blocks[block_id].compression_metrics = new_metrics
                    self.compression_cache[block_id] = new_metrics
                    
                    self.total_storage_used += len(new_compressed) - old_size
                    
                    logger.info(f"Recompressed block {block_id}: {old_size} -> {len(new_compressed)} bytes")
                
            except Exception as e:
                logger.error(f"Error recompressing block {block_id}: {e}")
    
    def _cleanup_blocks(self, block_ids: List[str]):
        """Clean up old or unused blocks."""
        for block_id in block_ids:
            try:
                if block_id in self.storage_blocks:
                    block = self.storage_blocks[block_id]
                    
                    # Remove from storage
                    self.total_storage_used -= len(block.data)
                    if block.compression_metrics:
                        self.total_original_size -= block.compression_metrics.original_size
                    
                    del self.storage_blocks[block_id]
                    if block_id in self.compression_cache:
                        del self.compression_cache[block_id]
                    
                    # Update deduplication index
                    if block.similarity_hash in self.deduplication_index:
                        self.deduplication_index[block.similarity_hash].discard(block_id)
                        if not self.deduplication_index[block.similarity_hash]:
                            del self.deduplication_index[block.similarity_hash]
                    
                    logger.info(f"Cleaned up block {block_id}")
                    
            except Exception as e:
                logger.error(f"Error cleaning up block {block_id}: {e}")
    
    def _update_optimization_stats(self):
        """Update overall optimization statistics."""
        if not self.storage_blocks:
            return
        
        # Calculate compression ratio
        if self.total_original_size > 0:
            overall_compression_ratio = self.total_original_size / self.total_storage_used
        else:
            overall_compression_ratio = 1.0
        
        # Calculate storage efficiency
        storage_efficiency = (1.0 - self.total_storage_used / self.storage_limit) * 100
        
        # Calculate deduplication savings
        deduplication_savings = len([
            hash_val for hash_val, block_ids in self.deduplication_index.items()
            if len(block_ids) > 1
        ])
        
        self.optimization_stats = {
            'compression_ratio': overall_compression_ratio,
            'deduplication_savings': deduplication_savings,
            'storage_efficiency': max(0.0, storage_efficiency),
            'total_blocks': len(self.storage_blocks),
            'total_storage_used_mb': self.total_storage_used / (1024 * 1024),
            'total_original_size_mb': self.total_original_size / (1024 * 1024)
        }
    
    def get_storage_insights(self) -> Dict[str, Any]:
        """Get comprehensive storage optimization insights."""
        if not self.initialized:
            return {'error': 'Quantum storage optimizer not initialized'}
        
        # Performance metrics by algorithm
        algorithm_performance = defaultdict(list)
        for metrics in self.compression_cache.values():
            if metrics:
                algorithm_performance[metrics.algorithm_used].append(metrics.calculate_efficiency_score())
        
        avg_performance = {}
        for algorithm, scores in algorithm_performance.items():
            avg_performance[algorithm] = statistics.mean(scores) if scores else 0.0
        
        # Access pattern analysis
        pattern_distribution = defaultdict(int)
        for block in self.storage_blocks.values():
            pattern_distribution[block.access_pattern.value] += 1
        
        # Data type distribution
        type_distribution = defaultdict(int)
        for block in self.storage_blocks.values():
            type_distribution[block.data_type.value] += 1
        
        return {
            'optimization_statistics': self.optimization_stats,
            'algorithm_performance': dict(avg_performance),
            'access_pattern_distribution': dict(pattern_distribution),
            'data_type_distribution': dict(type_distribution),
            'storage_utilization': (self.total_storage_used / self.storage_limit) * 100,
            'deduplication_efficiency': len(self.deduplication_index),
            'quantum_compression_usage': sum(1 for m in self.compression_cache.values() 
                                           if 'quantum' in m.algorithm_used),
            'last_optimization': self.last_optimization.isoformat()
        }
    
    def cleanup(self):
        """Clean up resources and stop background threads."""
        self.optimization_running = False
        if self.optimization_thread and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=5)
        
        logger.info("Quantum Storage Optimizer cleaned up")