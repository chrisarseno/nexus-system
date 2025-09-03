"""
Graph-Based Knowledge Network System
Creates and manages relationships between knowledge domains with inference capabilities.
"""

import logging
import json
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import threading
import time
import pickle

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeNode:
    """Represents a node in the knowledge graph."""
    node_id: str
    name: str
    description: str
    domain: str
    concepts: List[str]
    importance: float
    creation_time: datetime
    last_accessed: datetime
    access_count: int = 0
    
@dataclass
class KnowledgeEdge:
    """Represents a relationship between knowledge nodes."""
    source_id: str
    target_id: str
    relationship_type: str  # 'depends_on', 'enables', 'contradicts', 'supports', 'extends'
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    evidence: List[str]
    creation_time: datetime
    
@dataclass
class InferenceChain:
    """Represents a chain of reasoning through the knowledge graph."""
    chain_id: str
    start_node: str
    end_node: str
    path: List[str]
    reasoning_steps: List[Dict[str, Any]]
    confidence: float
    strength: float
    inference_type: str  # 'deductive', 'inductive', 'abductive'

class GraphKnowledgeNetwork:
    """
    Advanced graph-based knowledge network that maps relationships
    between concepts and enables sophisticated reasoning chains.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[str, KnowledgeEdge] = {}
        self.inference_cache: Dict[str, InferenceChain] = {}
        self.reasoning_patterns: Dict[str, List[str]] = {}
        
        # Performance tracking
        self.query_history = deque(maxlen=1000)
        self.inference_stats = defaultdict(int)
        
        # Threading for background operations
        self.optimization_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        self.initialized = False
    
    def initialize(self):
        """Initialize the knowledge graph network."""
        if self.initialized:
            return
            
        logger.info("Initializing Graph Knowledge Network...")
        
        # Build initial knowledge graph
        self._build_initial_graph()
        
        # Load reasoning patterns
        self._load_reasoning_patterns()
        
        # Start background optimization
        self._start_background_optimization()
        
        self.initialized = True
        logger.info(f"Graph Knowledge Network initialized with {len(self.nodes)} nodes and {len(self.edges)} edges")
    
    def add_knowledge_node(self, node_id: str, name: str, description: str, 
                          domain: str, concepts: List[str], importance: float = 0.5) -> bool:
        """Add a new knowledge node to the graph."""
        try:
            with self.lock:
                node = KnowledgeNode(
                    node_id=node_id,
                    name=name,
                    description=description,
                    domain=domain,
                    concepts=concepts,
                    importance=importance,
                    creation_time=datetime.now(),
                    last_accessed=datetime.now()
                )
                
                self.nodes[node_id] = node
                self.graph.add_node(node_id, **asdict(node))
                
                # Auto-discover relationships
                relationships = self._discover_relationships(node)
                for rel in relationships:
                    self.add_knowledge_edge(
                        rel['source'], rel['target'], rel['type'], 
                        rel['strength'], rel['confidence']
                    )
                
                logger.debug(f"Added knowledge node: {node_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error adding knowledge node {node_id}: {e}")
            return False
    
    def add_knowledge_edge(self, source_id: str, target_id: str, relationship_type: str,
                          strength: float, confidence: float, evidence: List[str] = None) -> bool:
        """Add a relationship edge between knowledge nodes."""
        try:
            with self.lock:
                if source_id not in self.nodes or target_id not in self.nodes:
                    logger.warning(f"Cannot add edge: nodes {source_id} or {target_id} don't exist")
                    return False
                
                edge_id = f"{source_id}->{target_id}_{relationship_type}"
                
                edge = KnowledgeEdge(
                    source_id=source_id,
                    target_id=target_id,
                    relationship_type=relationship_type,
                    strength=strength,
                    confidence=confidence,
                    evidence=evidence or [],
                    creation_time=datetime.now()
                )
                
                self.edges[edge_id] = edge
                self.graph.add_edge(
                    source_id, target_id,
                    relationship=relationship_type,
                    strength=strength,
                    confidence=confidence,
                    edge_id=edge_id
                )
                
                logger.debug(f"Added knowledge edge: {edge_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error adding knowledge edge: {e}")
            return False
    
    def find_inference_chain(self, start_concept: str, target_concept: str, 
                           max_depth: int = 5, inference_type: str = 'deductive') -> Optional[InferenceChain]:
        """Find a reasoning chain between two concepts."""
        try:
            # Check cache first
            cache_key = f"{start_concept}->{target_concept}_{inference_type}"
            if cache_key in self.inference_cache:
                cached = self.inference_cache[cache_key]
                cached.confidence *= 0.95  # Slight decay for cached results
                return cached
            
            # Find source and target nodes
            start_nodes = self._find_nodes_by_concept(start_concept)
            target_nodes = self._find_nodes_by_concept(target_concept)
            
            if not start_nodes or not target_nodes:
                return None
            
            best_chain = None
            best_score = 0.0
            
            # Try all combinations of start and target nodes
            for start_node in start_nodes[:3]:  # Limit to prevent explosion
                for target_node in target_nodes[:3]:
                    try:
                        # Find shortest weighted path
                        path = nx.shortest_path(
                            self.graph, start_node, target_node,
                            weight=lambda u, v, d: 1.0 / (d.get('strength', 0.1) * d.get('confidence', 0.1))
                        )
                        
                        if len(path) <= max_depth + 1:
                            # Build reasoning chain
                            chain = self._build_reasoning_chain(
                                path, start_concept, target_concept, inference_type
                            )
                            
                            if chain and chain.confidence > best_score:
                                best_chain = chain
                                best_score = chain.confidence
                                
                    except nx.NetworkXNoPath:
                        continue
                    except Exception as e:
                        logger.debug(f"Error finding path from {start_node} to {target_node}: {e}")
                        continue
            
            # Cache the result
            if best_chain:
                self.inference_cache[cache_key] = best_chain
                self.inference_stats[inference_type] += 1
            
            return best_chain
            
        except Exception as e:
            logger.error(f"Error finding inference chain: {e}")
            return None
    
    def explore_related_concepts(self, concept: str, depth: int = 2, 
                               relationship_types: List[str] = None) -> Dict[str, Any]:
        """Explore concepts related to a given concept."""
        try:
            nodes = self._find_nodes_by_concept(concept)
            if not nodes:
                return {'related_concepts': [], 'reasoning_paths': []}
            
            related_concepts = set()
            reasoning_paths = []
            
            for node_id in nodes:
                # BFS to find related concepts
                visited = set()
                queue = deque([(node_id, 0, [])])
                
                while queue:
                    current_node, current_depth, path = queue.popleft()
                    
                    if current_depth >= depth or current_node in visited:
                        continue
                        
                    visited.add(current_node)
                    
                    # Get neighbors
                    for neighbor in self.graph.neighbors(current_node):
                        edge_data = self.graph[current_node][neighbor]
                        relationship = edge_data.get('relationship', 'unknown')
                        
                        # Filter by relationship type if specified
                        if relationship_types and relationship not in relationship_types:
                            continue
                        
                        # Add to related concepts
                        if neighbor in self.nodes:
                            related_node = self.nodes[neighbor]
                            for related_concept in related_node.concepts:
                                related_concepts.add(related_concept)
                        
                        # Build reasoning path
                        new_path = path + [(current_node, neighbor, relationship)]
                        reasoning_paths.append({
                            'path': new_path,
                            'strength': edge_data.get('strength', 0.5),
                            'confidence': edge_data.get('confidence', 0.5)
                        })
                        
                        queue.append((neighbor, current_depth + 1, new_path))
            
            # Sort by relevance
            reasoning_paths.sort(key=lambda x: x['strength'] * x['confidence'], reverse=True)
            
            return {
                'related_concepts': list(related_concepts),
                'reasoning_paths': reasoning_paths[:20],  # Limit results
                'exploration_depth': depth,
                'total_paths_found': len(reasoning_paths)
            }
            
        except Exception as e:
            logger.error(f"Error exploring related concepts: {e}")
            return {'related_concepts': [], 'reasoning_paths': []}
    
    def analyze_concept_importance(self, concept: str) -> Dict[str, Any]:
        """Analyze the importance and centrality of a concept."""
        try:
            nodes = self._find_nodes_by_concept(concept)
            if not nodes:
                return {'importance': 0.0, 'centrality': 0.0, 'connections': 0}
            
            total_importance = 0.0
            total_centrality = 0.0
            total_connections = 0
            
            # Calculate centrality measures
            centrality_measures = {
                'betweenness': nx.betweenness_centrality(self.graph),
                'closeness': nx.closeness_centrality(self.graph),
                'pagerank': nx.pagerank(self.graph)
            }
            
            for node_id in nodes:
                node = self.nodes[node_id]
                total_importance += node.importance
                
                # Average centrality measures
                node_centrality = sum([
                    centrality_measures['betweenness'].get(node_id, 0),
                    centrality_measures['closeness'].get(node_id, 0),
                    centrality_measures['pagerank'].get(node_id, 0)
                ]) / 3
                
                total_centrality += node_centrality
                total_connections += self.graph.degree(node_id)
            
            avg_importance = total_importance / len(nodes)
            avg_centrality = total_centrality / len(nodes)
            avg_connections = total_connections / len(nodes)
            
            return {
                'importance': round(avg_importance, 3),
                'centrality': round(avg_centrality, 3),
                'connections': avg_connections,
                'nodes_found': len(nodes),
                'influence_score': round(avg_importance * avg_centrality * (1 + avg_connections/10), 3)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing concept importance: {e}")
            return {'importance': 0.0, 'centrality': 0.0, 'connections': 0}
    
    def get_network_insights(self) -> Dict[str, Any]:
        """Get insights about the knowledge network structure."""
        try:
            with self.lock:
                # Basic statistics
                node_count = len(self.nodes)
                edge_count = len(self.edges)
                
                # Graph properties
                if node_count > 1:
                    density = nx.density(self.graph)
                    avg_clustering = nx.average_clustering(self.graph)
                    
                    # Connected components
                    components = list(nx.weakly_connected_components(self.graph))
                    largest_component = max(components, key=len) if components else set()
                else:
                    density = 0.0
                    avg_clustering = 0.0
                    largest_component = set()
                
                # Domain distribution
                domain_distribution = defaultdict(int)
                for node in self.nodes.values():
                    domain_distribution[node.domain] += 1
                
                # Relationship type distribution
                relationship_distribution = defaultdict(int)
                for edge in self.edges.values():
                    relationship_distribution[edge.relationship_type] += 1
                
                # Recent inference activity
                recent_inferences = len([
                    record for record in self.query_history
                    if (datetime.now() - record.get('timestamp', datetime.min)).total_seconds() < 3600
                ])
                
                return {
                    'network_size': {
                        'nodes': node_count,
                        'edges': edge_count,
                        'largest_component': len(largest_component)
                    },
                    'network_properties': {
                        'density': round(density, 3),
                        'clustering_coefficient': round(avg_clustering, 3),
                        'connected_components': len(components) if node_count > 1 else 0
                    },
                    'domain_distribution': dict(domain_distribution),
                    'relationship_distribution': dict(relationship_distribution),
                    'inference_activity': {
                        'cached_chains': len(self.inference_cache),
                        'recent_queries': recent_inferences,
                        'inference_stats': dict(self.inference_stats)
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting network insights: {e}")
            return {}
    
    def _build_initial_graph(self):
        """Build the initial knowledge graph from domain data."""
        try:
            # Core mathematical concepts
            math_concepts = [
                ('calculus', 'Mathematical analysis of continuous change', ['integration', 'differentiation', 'limits']),
                ('algebra', 'Study of mathematical symbols and rules', ['equations', 'polynomials', 'variables']),
                ('geometry', 'Study of shapes, sizes, and spatial relationships', ['triangles', 'circles', 'coordinate_systems']),
                ('statistics', 'Collection, analysis, and interpretation of data', ['probability', 'distributions', 'inference'])
            ]
            
            for concept_id, description, concepts in math_concepts:
                self.add_knowledge_node(
                    node_id=f"math_{concept_id}",
                    name=concept_id.title(),
                    description=description,
                    domain='mathematics',
                    concepts=concepts,
                    importance=0.8
                )
            
            # Physics concepts
            physics_concepts = [
                ('mechanics', 'Study of motion and forces', ['newton_laws', 'energy', 'momentum']),
                ('thermodynamics', 'Study of heat and temperature', ['entropy', 'heat_transfer', 'phase_transitions']),
                ('electromagnetism', 'Study of electric and magnetic fields', ['maxwell_equations', 'electromagnetic_waves']),
                ('quantum_mechanics', 'Physics of atomic and subatomic particles', ['wave_function', 'uncertainty_principle'])
            ]
            
            for concept_id, description, concepts in physics_concepts:
                self.add_knowledge_node(
                    node_id=f"physics_{concept_id}",
                    name=concept_id.title(),
                    description=description,
                    domain='physics',
                    concepts=concepts,
                    importance=0.8
                )
            
            # Computer science concepts
            cs_concepts = [
                ('algorithms', 'Step-by-step procedures for calculations', ['sorting', 'searching', 'optimization']),
                ('data_structures', 'Ways of organizing and storing data', ['arrays', 'trees', 'graphs']),
                ('machine_learning', 'Algorithms that improve through experience', ['neural_networks', 'regression', 'classification']),
                ('databases', 'Organized collections of structured information', ['sql', 'nosql', 'transactions'])
            ]
            
            for concept_id, description, concepts in cs_concepts:
                self.add_knowledge_node(
                    node_id=f"cs_{concept_id}",
                    name=concept_id.title(),
                    description=description,
                    domain='computer_science',
                    concepts=concepts,
                    importance=0.7
                )
            
            # Add some foundational relationships
            relationships = [
                ('math_calculus', 'physics_mechanics', 'enables', 0.9, 0.8),
                ('math_algebra', 'cs_algorithms', 'enables', 0.7, 0.7),
                ('cs_data_structures', 'cs_algorithms', 'supports', 0.8, 0.9),
                ('physics_quantum_mechanics', 'cs_machine_learning', 'extends', 0.6, 0.6),
                ('math_statistics', 'cs_machine_learning', 'enables', 0.9, 0.9)
            ]
            
            for source, target, rel_type, strength, confidence in relationships:
                self.add_knowledge_edge(source, target, rel_type, strength, confidence)
            
            logger.info("Initial knowledge graph built successfully")
            
        except Exception as e:
            logger.error(f"Error building initial graph: {e}")
    
    def _discover_relationships(self, node: KnowledgeNode) -> List[Dict[str, Any]]:
        """Automatically discover relationships for a new node."""
        relationships = []
        
        try:
            for existing_id, existing_node in self.nodes.items():
                if existing_id == node.node_id:
                    continue
                
                # Check concept overlap
                concept_overlap = set(node.concepts) & set(existing_node.concepts)
                if concept_overlap:
                    strength = len(concept_overlap) / max(len(node.concepts), len(existing_node.concepts))
                    relationships.append({
                        'source': node.node_id,
                        'target': existing_id,
                        'type': 'relates_to',
                        'strength': min(0.8, strength),
                        'confidence': 0.6
                    })
                
                # Check domain relationships
                if node.domain == existing_node.domain:
                    relationships.append({
                        'source': node.node_id,
                        'target': existing_id,
                        'type': 'same_domain',
                        'strength': 0.5,
                        'confidence': 0.7
                    })
                elif self._are_related_domains(node.domain, existing_node.domain):
                    relationships.append({
                        'source': node.node_id,
                        'target': existing_id,
                        'type': 'cross_domain',
                        'strength': 0.4,
                        'confidence': 0.5
                    })
        
        except Exception as e:
            logger.error(f"Error discovering relationships: {e}")
        
        return relationships[:5]  # Limit to prevent too many connections
    
    def _find_nodes_by_concept(self, concept: str) -> List[str]:
        """Find nodes that contain a specific concept."""
        matching_nodes = []
        concept_lower = concept.lower()
        
        for node_id, node in self.nodes.items():
            # Check if concept appears in node concepts
            for node_concept in node.concepts:
                if concept_lower in node_concept.lower() or node_concept.lower() in concept_lower:
                    matching_nodes.append(node_id)
                    break
            
            # Also check name and description
            if (concept_lower in node.name.lower() or 
                concept_lower in node.description.lower()):
                if node_id not in matching_nodes:
                    matching_nodes.append(node_id)
        
        return matching_nodes
    
    def _build_reasoning_chain(self, path: List[str], start_concept: str, 
                             target_concept: str, inference_type: str) -> InferenceChain:
        """Build a reasoning chain from a graph path."""
        try:
            reasoning_steps = []
            total_confidence = 1.0
            total_strength = 1.0
            
            for i in range(len(path) - 1):
                source_node = path[i]
                target_node = path[i + 1]
                
                if self.graph.has_edge(source_node, target_node):
                    edge_data = self.graph[source_node][target_node]
                    
                    step = {
                        'from': source_node,
                        'to': target_node,
                        'relationship': edge_data.get('relationship', 'unknown'),
                        'strength': edge_data.get('strength', 0.5),
                        'confidence': edge_data.get('confidence', 0.5),
                        'reasoning': f"From {self.nodes[source_node].name} to {self.nodes[target_node].name}"
                    }
                    
                    reasoning_steps.append(step)
                    total_confidence *= step['confidence']
                    total_strength *= step['strength']
            
            # Adjust confidence based on path length
            path_length_penalty = 0.9 ** (len(path) - 2)
            final_confidence = total_confidence * path_length_penalty
            
            chain = InferenceChain(
                chain_id=f"chain_{int(time.time())}",
                start_node=path[0],
                end_node=path[-1],
                path=path,
                reasoning_steps=reasoning_steps,
                confidence=round(final_confidence, 3),
                strength=round(total_strength, 3),
                inference_type=inference_type
            )
            
            return chain
            
        except Exception as e:
            logger.error(f"Error building reasoning chain: {e}")
            return None
    
    def _are_related_domains(self, domain1: str, domain2: str) -> bool:
        """Check if two domains are related."""
        related_pairs = [
            ('mathematics', 'physics'),
            ('mathematics', 'computer_science'),
            ('physics', 'chemistry'),
            ('computer_science', 'artificial_intelligence'),
            ('biology', 'chemistry'),
            ('psychology', 'philosophy')
        ]
        
        return (domain1, domain2) in related_pairs or (domain2, domain1) in related_pairs
    
    def _load_reasoning_patterns(self):
        """Load common reasoning patterns."""
        self.reasoning_patterns = {
            'deductive': [
                'general_to_specific',
                'rule_application',
                'logical_inference'
            ],
            'inductive': [
                'pattern_recognition',
                'generalization',
                'statistical_inference'
            ],
            'abductive': [
                'best_explanation',
                'hypothesis_formation',
                'inference_to_best_explanation'
            ]
        }
    
    def _start_background_optimization(self):
        """Start background optimization thread."""
        if not self.optimization_thread:
            self.running = True
            self.optimization_thread = threading.Thread(target=self._optimization_loop)
            self.optimization_thread.daemon = True
            self.optimization_thread.start()
            logger.info("Graph optimization started")
    
    def _optimization_loop(self):
        """Background optimization loop."""
        while self.running:
            try:
                time.sleep(300)  # Run every 5 minutes
                
                # Clear old cache entries
                if len(self.inference_cache) > 1000:
                    # Remove oldest 20% of cache entries
                    cache_items = list(self.inference_cache.items())
                    cache_items.sort(key=lambda x: x[1].confidence)
                    
                    remove_count = len(cache_items) // 5
                    for i in range(remove_count):
                        del self.inference_cache[cache_items[i][0]]
                
                # Update node access times
                self._update_node_statistics()
                
            except Exception as e:
                logger.error(f"Error in graph optimization loop: {e}")
    
    def _update_node_statistics(self):
        """Update node access statistics."""
        try:
            # Update access counts based on recent queries
            for record in list(self.query_history)[-100:]:  # Last 100 queries
                if 'nodes_accessed' in record:
                    for node_id in record['nodes_accessed']:
                        if node_id in self.nodes:
                            self.nodes[node_id].access_count += 1
                            self.nodes[node_id].last_accessed = datetime.now()
        except Exception as e:
            logger.error(f"Error updating node statistics: {e}")
    
    def shutdown(self):
        """Shutdown the graph network."""
        self.running = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
        logger.info("Graph Knowledge Network shutdown completed")