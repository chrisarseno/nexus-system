"""
Advanced Vector Storage and Retrieval System
High-performance vector operations with semantic search capabilities.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import sqlite3
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Advanced vector storage system with semantic search,
    similarity matching, and efficient retrieval.
    """
    
    def __init__(self, db_path: str = "vector_store.db"):
        self.db_path = db_path
        self.connection = None
        self.vector_cache = {}
        self.index_cache = {}
        self.initialized = False
        
    def initialize(self):
        """Initialize the vector storage system."""
        if self.initialized:
            return
            
        logger.info("Initializing Vector Store...")
        
        try:
            self._setup_database()
            self._load_vector_cache()
            
            self.initialized = True
            logger.info("Vector Store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vector Store: {e}")
            raise
    
    def _setup_database(self):
        """Set up the SQLite database for vector storage."""
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # Create tables
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vector_id TEXT UNIQUE NOT NULL,
                domain_id TEXT NOT NULL,
                content TEXT NOT NULL,
                vector_data BLOB NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS vector_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain_id TEXT NOT NULL,
                index_type TEXT NOT NULL,
                index_data BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for performance
        self.connection.execute('CREATE INDEX IF NOT EXISTS idx_vectors_domain ON vectors(domain_id)')
        self.connection.execute('CREATE INDEX IF NOT EXISTS idx_vectors_vector_id ON vectors(vector_id)')
        
        self.connection.commit()
        
    def _load_vector_cache(self):
        """Load frequently accessed vectors into cache."""
        cursor = self.connection.cursor()
        cursor.execute('SELECT domain_id, COUNT(*) as count FROM vectors GROUP BY domain_id')
        
        for domain_id, count in cursor.fetchall():
            logger.info(f"Domain {domain_id}: {count} vectors")
            
        cursor.close()
    
    def store_vector(self, vector_id: str, domain_id: str, content: str, 
                    vector: np.ndarray, metadata: Dict = None) -> bool:
        """Store a vector with associated content and metadata."""
        try:
            vector_blob = vector.tobytes()
            metadata_json = json.dumps(metadata or {})
            
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO vectors 
                (vector_id, domain_id, content, vector_data, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (vector_id, domain_id, content, vector_blob, metadata_json, datetime.now()))
            
            self.connection.commit()
            cursor.close()
            
            # Update cache
            if domain_id not in self.vector_cache:
                self.vector_cache[domain_id] = {}
            self.vector_cache[domain_id][vector_id] = {
                'content': content,
                'vector': vector,
                'metadata': metadata or {}
            }
            
            logger.debug(f"Stored vector {vector_id} in domain {domain_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing vector {vector_id}: {e}")
            return False
    
    def retrieve_vector(self, vector_id: str, domain_id: str = None) -> Optional[Dict]:
        """Retrieve a specific vector by ID."""
        try:
            # Check cache first
            if domain_id and domain_id in self.vector_cache:
                if vector_id in self.vector_cache[domain_id]:
                    return self.vector_cache[domain_id][vector_id]
            
            # Query database
            cursor = self.connection.cursor()
            if domain_id:
                cursor.execute('''
                    SELECT domain_id, content, vector_data, metadata 
                    FROM vectors WHERE vector_id = ? AND domain_id = ?
                ''', (vector_id, domain_id))
            else:
                cursor.execute('''
                    SELECT domain_id, content, vector_data, metadata 
                    FROM vectors WHERE vector_id = ?
                ''', (vector_id,))
            
            row = cursor.fetchone()
            cursor.close()
            
            if row:
                domain_id, content, vector_blob, metadata_json = row
                vector = np.frombuffer(vector_blob, dtype=np.float64)
                metadata = json.loads(metadata_json)
                
                result = {
                    'domain_id': domain_id,
                    'content': content,
                    'vector': vector,
                    'metadata': metadata
                }
                
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving vector {vector_id}: {e}")
            return None
    
    def semantic_search(self, query_vector: np.ndarray, domain_id: str = None, 
                       limit: int = 10, similarity_threshold: float = 0.7) -> List[Dict]:
        """Perform semantic search using vector similarity."""
        try:
            results = []
            
            # Query database for vectors
            cursor = self.connection.cursor()
            if domain_id:
                cursor.execute('''
                    SELECT vector_id, domain_id, content, vector_data, metadata 
                    FROM vectors WHERE domain_id = ?
                ''', (domain_id,))
            else:
                cursor.execute('''
                    SELECT vector_id, domain_id, content, vector_data, metadata 
                    FROM vectors
                ''')
            
            for row in cursor.fetchall():
                vector_id, domain_id, content, vector_blob, metadata_json = row
                
                # Calculate similarity
                stored_vector = np.frombuffer(vector_blob, dtype=np.float64)
                similarity = self._cosine_similarity(query_vector, stored_vector)
                
                if similarity >= similarity_threshold:
                    results.append({
                        'vector_id': vector_id,
                        'domain_id': domain_id,
                        'content': content,
                        'similarity': similarity,
                        'metadata': json.loads(metadata_json)
                    })
            
            cursor.close()
            
            # Sort by similarity and return top results
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def find_similar_vectors(self, vector_id: str, domain_id: str = None, 
                           limit: int = 5) -> List[Dict]:
        """Find vectors similar to a specific vector."""
        try:
            # Get the reference vector
            ref_vector_data = self.retrieve_vector(vector_id, domain_id)
            if not ref_vector_data:
                return []
            
            query_vector = ref_vector_data['vector']
            
            # Perform semantic search excluding the reference vector
            results = self.semantic_search(query_vector, domain_id, limit + 1)
            
            # Remove the reference vector from results
            filtered_results = [r for r in results if r['vector_id'] != vector_id]
            
            return filtered_results[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar vectors: {e}")
            return []
    
    def get_domain_vectors(self, domain_id: str, limit: int = None) -> List[Dict]:
        """Get all vectors for a specific domain."""
        try:
            cursor = self.connection.cursor()
            
            if limit:
                cursor.execute('''
                    SELECT vector_id, content, metadata 
                    FROM vectors WHERE domain_id = ? LIMIT ?
                ''', (domain_id, limit))
            else:
                cursor.execute('''
                    SELECT vector_id, content, metadata 
                    FROM vectors WHERE domain_id = ?
                ''', (domain_id,))
            
            results = []
            for row in cursor.fetchall():
                vector_id, content, metadata_json = row
                results.append({
                    'vector_id': vector_id,
                    'content': content,
                    'metadata': json.loads(metadata_json)
                })
            
            cursor.close()
            return results
            
        except Exception as e:
            logger.error(f"Error getting domain vectors: {e}")
            return []
    
    def delete_vector(self, vector_id: str, domain_id: str = None) -> bool:
        """Delete a vector from storage."""
        try:
            cursor = self.connection.cursor()
            
            if domain_id:
                cursor.execute('DELETE FROM vectors WHERE vector_id = ? AND domain_id = ?', 
                             (vector_id, domain_id))
            else:
                cursor.execute('DELETE FROM vectors WHERE vector_id = ?', (vector_id,))
            
            deleted_count = cursor.rowcount
            self.connection.commit()
            cursor.close()
            
            # Update cache
            if domain_id and domain_id in self.vector_cache:
                if vector_id in self.vector_cache[domain_id]:
                    del self.vector_cache[domain_id][vector_id]
            
            logger.info(f"Deleted {deleted_count} vector(s) with ID {vector_id}")
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting vector {vector_id}: {e}")
            return False
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics."""
        try:
            cursor = self.connection.cursor()
            
            # Total vectors
            cursor.execute('SELECT COUNT(*) FROM vectors')
            total_vectors = cursor.fetchone()[0]
            
            # Vectors by domain
            cursor.execute('SELECT domain_id, COUNT(*) FROM vectors GROUP BY domain_id')
            domain_counts = dict(cursor.fetchall())
            
            # Database size
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            cursor.close()
            
            return {
                'total_vectors': total_vectors,
                'domain_counts': domain_counts,
                'database_size_bytes': db_size,
                'cache_domains': len(self.vector_cache),
                'initialized': self.initialized
            }
            
        except Exception as e:
            logger.error(f"Error getting storage statistics: {e}")
            return {}
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            # Ensure vectors have the same length
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def cleanup(self):
        """Clean up resources."""
        if self.connection:
            self.connection.close()
            logger.info("Vector store connection closed")
