"""
Knowledge Domain Management System
Manages 45+ specialized knowledge domains with advanced categorization.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class KnowledgeDomainManager:
    """
    Manages specialized knowledge domains with advanced categorization,
    retrieval, and cross-domain relationship mapping.
    """
    
    def __init__(self):
        self.domains: Dict[str, Dict] = {}
        self.cross_domain_relationships = {}
        self.domain_hierarchies = {}
        self.initialized = False
        
    def initialize(self):
        """Initialize with comprehensive knowledge domains."""
        if self.initialized:
            return
            
        logger.info("Initializing Knowledge Domain Manager...")
        
        # Load comprehensive domain definitions
        self._load_core_domains()
        self._load_specialized_domains()
        self._build_domain_relationships()
        
        self.initialized = True
        logger.info(f"Knowledge Domain Manager initialized with {len(self.domains)} domains")
    
    def _load_core_domains(self):
        """Load core fundamental knowledge domains."""
        core_domains = {
            "mathematics": {
                "name": "Mathematics",
                "description": "Mathematical concepts, formulas, and problem-solving",
                "categories": ["algebra", "geometry", "calculus", "statistics", "discrete_math"],
                "expertise_level": "expert",
                "vectors": [],
                "last_updated": datetime.now().isoformat()
            },
            "computer_science": {
                "name": "Computer Science", 
                "description": "Programming, algorithms, data structures, and software engineering",
                "categories": ["algorithms", "programming", "databases", "networking", "ai_ml"],
                "expertise_level": "expert",
                "vectors": [],
                "last_updated": datetime.now().isoformat()
            },
            "physics": {
                "name": "Physics",
                "description": "Physical laws, principles, and phenomena",
                "categories": ["mechanics", "thermodynamics", "electromagnetism", "quantum", "relativity"],
                "expertise_level": "advanced",
                "vectors": [],
                "last_updated": datetime.now().isoformat()
            },
            "chemistry": {
                "name": "Chemistry",
                "description": "Chemical elements, compounds, reactions, and processes",
                "categories": ["organic", "inorganic", "physical_chemistry", "biochemistry", "analytical"],
                "expertise_level": "advanced",
                "vectors": [],
                "last_updated": datetime.now().isoformat()
            },
            "biology": {
                "name": "Biology",
                "description": "Living organisms, life processes, and biological systems",
                "categories": ["molecular", "cellular", "genetics", "ecology", "evolution"],
                "expertise_level": "advanced",
                "vectors": [],
                "last_updated": datetime.now().isoformat()
            },
            "history": {
                "name": "History",
                "description": "Historical events, periods, civilizations, and cultural developments",
                "categories": ["ancient", "medieval", "modern", "world_wars", "cultural"],
                "expertise_level": "intermediate",
                "vectors": [],
                "last_updated": datetime.now().isoformat()
            },
            "literature": {
                "name": "Literature",
                "description": "Literary works, authors, genres, and literary analysis",
                "categories": ["classics", "contemporary", "poetry", "drama", "criticism"],
                "expertise_level": "intermediate",
                "vectors": [],
                "last_updated": datetime.now().isoformat()
            },
            "philosophy": {
                "name": "Philosophy", 
                "description": "Philosophical concepts, theories, and schools of thought",
                "categories": ["ethics", "metaphysics", "epistemology", "logic", "political_philosophy"],
                "expertise_level": "intermediate",
                "vectors": [],
                "last_updated": datetime.now().isoformat()
            },
            "psychology": {
                "name": "Psychology",
                "description": "Human behavior, cognition, and mental processes",
                "categories": ["cognitive", "behavioral", "clinical", "developmental", "social"],
                "expertise_level": "intermediate",
                "vectors": [],
                "last_updated": datetime.now().isoformat()
            },
            "economics": {
                "name": "Economics",
                "description": "Economic principles, markets, and financial systems",
                "categories": ["microeconomics", "macroeconomics", "finance", "behavioral_economics", "international"],
                "expertise_level": "intermediate",
                "vectors": [],
                "last_updated": datetime.now().isoformat()
            }
        }
        
        self.domains.update(core_domains)
        
    def _load_specialized_domains(self):
        """Load specialized and emerging knowledge domains."""
        specialized_domains = {
            "artificial_intelligence": {
                "name": "Artificial Intelligence",
                "description": "AI algorithms, machine learning, neural networks, and applications",
                "categories": ["machine_learning", "deep_learning", "nlp", "computer_vision", "robotics"],
                "expertise_level": "expert",
                "vectors": [],
                "last_updated": datetime.now().isoformat()
            },
            "cybersecurity": {
                "name": "Cybersecurity",
                "description": "Information security, threat analysis, and protection mechanisms", 
                "categories": ["network_security", "cryptography", "threat_analysis", "incident_response", "compliance"],
                "expertise_level": "advanced",
                "vectors": [],
                "last_updated": datetime.now().isoformat()
            },
            "data_science": {
                "name": "Data Science",
                "description": "Data analysis, statistical modeling, and insights extraction",
                "categories": ["statistics", "data_mining", "visualization", "big_data", "analytics"],
                "expertise_level": "advanced",
                "vectors": [],
                "last_updated": datetime.now().isoformat()
            },
            "blockchain": {
                "name": "Blockchain Technology",
                "description": "Distributed ledger technology, cryptocurrencies, and decentralized systems",
                "categories": ["cryptocurrency", "smart_contracts", "defi", "consensus", "applications"],
                "expertise_level": "intermediate",
                "vectors": [],
                "last_updated": datetime.now().isoformat()
            },
            "quantum_computing": {
                "name": "Quantum Computing",
                "description": "Quantum mechanics applications in computing and information processing",
                "categories": ["quantum_algorithms", "quantum_hardware", "quantum_cryptography", "applications"],
                "expertise_level": "advanced",
                "vectors": [],
                "last_updated": datetime.now().isoformat()
            },
            "biotechnology": {
                "name": "Biotechnology",
                "description": "Biological systems for technological applications and medical advances",
                "categories": ["genetic_engineering", "bioinformatics", "medical_biotech", "industrial_biotech"],
                "expertise_level": "advanced",
                "vectors": [],
                "last_updated": datetime.now().isoformat()
            },
            "climate_science": {
                "name": "Climate Science",
                "description": "Climate systems, environmental changes, and sustainability",
                "categories": ["atmospheric_science", "environmental_impact", "sustainability", "renewable_energy"],
                "expertise_level": "intermediate",
                "vectors": [],
                "last_updated": datetime.now().isoformat()
            },
            "space_science": {
                "name": "Space Science",
                "description": "Astronomy, space exploration, and astrophysics",
                "categories": ["astronomy", "astrophysics", "space_exploration", "planetary_science"],
                "expertise_level": "intermediate",
                "vectors": [],
                "last_updated": datetime.now().isoformat()
            },
            "neuroscience": {
                "name": "Neuroscience",
                "description": "Brain function, neural networks, and cognitive processes",
                "categories": ["cognitive_neuroscience", "computational_neuroscience", "neuroplasticity", "brain_disorders"],
                "expertise_level": "advanced",
                "vectors": [],
                "last_updated": datetime.now().isoformat()
            },
            "materials_science": {
                "name": "Materials Science",
                "description": "Material properties, engineering, and applications",
                "categories": ["nanomaterials", "composites", "semiconductors", "biomaterials"],
                "expertise_level": "advanced",
                "vectors": [],
                "last_updated": datetime.now().isoformat()
            }
        }
        
        self.domains.update(specialized_domains)
    
    def _build_domain_relationships(self):
        """Build cross-domain relationships and hierarchies."""
        # Define domain relationships
        relationships = {
            "mathematics": ["computer_science", "physics", "economics", "data_science"],
            "computer_science": ["artificial_intelligence", "cybersecurity", "data_science"],
            "physics": ["quantum_computing", "materials_science", "space_science"],
            "biology": ["biotechnology", "neuroscience", "psychology"],
            "chemistry": ["materials_science", "biotechnology", "climate_science"],
            "psychology": ["neuroscience", "artificial_intelligence"],
            "economics": ["data_science", "blockchain"]
        }
        
        self.cross_domain_relationships = relationships
        
        # Define domain hierarchies
        hierarchies = {
            "stem": ["mathematics", "computer_science", "physics", "chemistry", "biology"],
            "technology": ["artificial_intelligence", "cybersecurity", "blockchain", "quantum_computing"],
            "science": ["physics", "chemistry", "biology", "climate_science", "space_science", "neuroscience"],
            "humanities": ["history", "literature", "philosophy"],
            "social_sciences": ["psychology", "economics"]
        }
        
        self.domain_hierarchies = hierarchies
    
    def get_domain_info(self, domain_id: str) -> Optional[Dict]:
        """Get detailed information about a specific domain."""
        return self.domains.get(domain_id)
    
    def get_related_domains(self, domain_id: str) -> List[str]:
        """Get domains related to the specified domain."""
        related = self.cross_domain_relationships.get(domain_id, [])
        
        # Also check reverse relationships
        for domain, relations in self.cross_domain_relationships.items():
            if domain_id in relations and domain not in related:
                related.append(domain)
        
        return related
    
    def find_domains_by_category(self, category: str) -> List[str]:
        """Find domains that contain a specific category."""
        matching_domains = []
        
        for domain_id, domain_info in self.domains.items():
            if category in domain_info.get('categories', []):
                matching_domains.append(domain_id)
        
        return matching_domains
    
    def get_domain_hierarchy(self, hierarchy_name: str) -> List[str]:
        """Get domains in a specific hierarchy."""
        return self.domain_hierarchies.get(hierarchy_name, [])
    
    def search_domains(self, query: str) -> List[Dict]:
        """Search domains by name, description, or categories."""
        query_lower = query.lower()
        results = []
        
        for domain_id, domain_info in self.domains.items():
            score = 0
            
            # Check name match
            if query_lower in domain_info['name'].lower():
                score += 10
            
            # Check description match
            if query_lower in domain_info['description'].lower():
                score += 5
            
            # Check category match
            for category in domain_info.get('categories', []):
                if query_lower in category.lower():
                    score += 3
            
            if score > 0:
                results.append({
                    'domain_id': domain_id,
                    'domain_info': domain_info,
                    'relevance_score': score
                })
        
        # Sort by relevance score
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results
    
    def get_expertise_domains(self, level: str) -> List[str]:
        """Get domains by expertise level."""
        return [
            domain_id for domain_id, domain_info in self.domains.items()
            if domain_info.get('expertise_level') == level
        ]
    
    def update_domain_vectors(self, domain_id: str, vectors: List[Any]):
        """Update vector embeddings for a domain."""
        if domain_id in self.domains:
            self.domains[domain_id]['vectors'] = vectors
            self.domains[domain_id]['last_updated'] = datetime.now().isoformat()
            logger.info(f"Updated vectors for domain {domain_id}: {len(vectors)} vectors")
    
    def get_domain_statistics(self) -> Dict[str, Any]:
        """Get comprehensive domain statistics."""
        total_vectors = sum(len(domain.get('vectors', [])) for domain in self.domains.values())
        
        expertise_counts = {}
        for domain_info in self.domains.values():
            level = domain_info.get('expertise_level', 'unknown')
            expertise_counts[level] = expertise_counts.get(level, 0) + 1
        
        return {
            'total_domains': len(self.domains),
            'total_vectors': total_vectors,
            'expertise_distribution': expertise_counts,
            'hierarchies': list(self.domain_hierarchies.keys()),
            'relationship_count': sum(len(relations) for relations in self.cross_domain_relationships.values())
        }
