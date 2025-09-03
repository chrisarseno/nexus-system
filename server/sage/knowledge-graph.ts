import { IStorage } from '../storage';
import { LocalAIService } from './local-ai-service';

export interface KnowledgeNode {
  id: string;
  type: 'fact' | 'concept' | 'relationship' | 'hypothesis' | 'rule';
  content: string;
  confidence: number;
  sources: string[];
  createdAt: Date;
  lastUpdated: Date;
  metadata: {
    domain: string;
    tags: string[];
    verificationLevel: 'unverified' | 'peer-reviewed' | 'multi-source' | 'consensus';
    importance: number; // 0-1 scale
  };
}

export interface KnowledgeEdge {
  id: string;
  fromNodeId: string;
  toNodeId: string;
  relationshipType: 'supports' | 'contradicts' | 'implies' | 'explains' | 'causes' | 'correlates' | 'similar' | 'part_of' | 'depends_on';
  strength: number; // 0-1 scale
  evidence: string[];
  confidence: number;
  bidirectional: boolean;
}

export interface Contradiction {
  id: string;
  nodeIds: string[];
  contradictionType: 'logical' | 'empirical' | 'semantic' | 'temporal';
  severity: 'minor' | 'moderate' | 'major' | 'critical';
  description: string;
  detectedAt: Date;
  resolution: 'pending' | 'resolved' | 'accepted_ambiguity' | 'requires_human_review';
  proposedResolution?: string;
  metadata: {
    autoDetected: boolean;
    reviewPriority: number;
    domain: string;
  };
}

export interface KnowledgeCluster {
  id: string;
  name: string;
  nodeIds: string[];
  coherenceScore: number;
  centralConcepts: string[];
  domain: string;
  lastAnalyzed: Date;
}

export interface GraphMetrics {
  totalNodes: number;
  totalEdges: number;
  totalContradictions: number;
  unresolvedContradictions: number;
  avgConfidence: number;
  knowledgeDensity: number;
  domainCoverage: string[];
  lastAnalysis: Date;
}

/**
 * NEXUS Knowledge Graph Engine
 * Manages knowledge representation, relationship detection, and contradiction analysis
 */
export class KnowledgeGraphEngine {
  private storage: IStorage;
  private localAI: LocalAIService;
  private nodes: Map<string, KnowledgeNode> = new Map();
  private edges: Map<string, KnowledgeEdge> = new Map();
  private contradictions: Map<string, Contradiction> = new Map();
  private clusters: Map<string, KnowledgeCluster> = new Map();

  constructor(storage: IStorage, localAI: LocalAIService) {
    this.storage = storage;
    this.localAI = localAI;
  }

  /**
   * Add a knowledge fact and analyze its relationships
   */
  async addKnowledgeNode(content: string, type: KnowledgeNode['type'], metadata: Partial<KnowledgeNode['metadata']>, sources: string[] = []): Promise<KnowledgeNode> {
    const node: KnowledgeNode = {
      id: `node_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type,
      content,
      confidence: 0.8, // Default confidence
      sources,
      createdAt: new Date(),
      lastUpdated: new Date(),
      metadata: {
        domain: metadata.domain || 'general',
        tags: metadata.tags || [],
        verificationLevel: metadata.verificationLevel || 'unverified',
        importance: metadata.importance || 0.5,
      }
    };

    // Store the node
    this.nodes.set(node.id, node);

    // Analyze relationships with existing nodes
    await this.analyzeRelationships(node);

    // Check for contradictions
    await this.detectContradictions(node);

    // Update clusters
    await this.updateClusters(node);

    // Log activity
    await this.storage.addActivity({
      type: 'knowledge' as const,
      message: `New knowledge added: ${content.substring(0, 50)}...`,
      moduleId: 'knowledge_graph'
    });

    console.log(`📚 Knowledge node added: ${node.type} - ${content.substring(0, 100)}`);
    return node;
  }

  /**
   * Analyze relationships between a new node and existing knowledge
   */
  private async analyzeRelationships(newNode: KnowledgeNode): Promise<void> {
    console.log(`🔍 Analyzing relationships for: ${newNode.content.substring(0, 50)}...`);

    const existingNodes = Array.from(this.nodes.values()).filter(n => n.id !== newNode.id);
    const relationships: KnowledgeEdge[] = [];

    // Batch analyze relationships with AI
    for (const existingNode of existingNodes.slice(0, 20)) { // Limit to recent nodes for performance
      try {
        const analysisPrompt = `Analyze the relationship between these two knowledge statements:

Statement A: "${newNode.content}"
Statement B: "${existingNode.content}"

Determine:
1. Relationship type (supports, contradicts, implies, explains, causes, correlates, similar, part_of, depends_on, none)
2. Relationship strength (0.0-1.0)
3. Confidence in this assessment (0.0-1.0)
4. Brief evidence/reasoning

Respond in JSON format: {"relationship": "type", "strength": 0.0, "confidence": 0.0, "evidence": "brief explanation"}`;

        const response = await this.localAI.generateResponse(analysisPrompt, 'analysis', 0.3, 300);
        const analysis = this.parseRelationshipAnalysis(response.content);

        if (analysis && analysis.relationship !== 'none' && analysis.confidence > 0.4) {
          const edge: KnowledgeEdge = {
            id: `edge_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            fromNodeId: newNode.id,
            toNodeId: existingNode.id,
            relationshipType: analysis.relationship,
            strength: analysis.strength,
            evidence: [analysis.evidence],
            confidence: analysis.confidence,
            bidirectional: ['similar', 'correlates'].includes(analysis.relationship)
          };

          this.edges.set(edge.id, edge);
          relationships.push(edge);
        }
      } catch (error) {
        console.warn(`Failed to analyze relationship with node ${existingNode.id}:`, error);
      }
    }

    console.log(`🔗 Found ${relationships.length} relationships for new node`);
  }

  /**
   * Parse AI response for relationship analysis
   */
  private parseRelationshipAnalysis(response: string): any {
    try {
      // Try to extract JSON from response
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]);
      }

      // Fallback parsing
      const relationshipMatch = response.match(/relationship[^\w]*(\w+)/i);
      const strengthMatch = response.match(/strength[^\d]*([\d.]+)/i);
      const confidenceMatch = response.match(/confidence[^\d]*([\d.]+)/i);

      if (relationshipMatch) {
        return {
          relationship: relationshipMatch[1],
          strength: strengthMatch ? parseFloat(strengthMatch[1]) : 0.5,
          confidence: confidenceMatch ? parseFloat(confidenceMatch[1]) : 0.5,
          evidence: 'Parsed from analysis'
        };
      }
    } catch (error) {
      console.warn('Failed to parse relationship analysis:', error);
    }

    return null;
  }

  /**
   * Detect contradictions involving a node
   */
  private async detectContradictions(node: KnowledgeNode): Promise<void> {
    console.log(`⚠️ Checking for contradictions with: ${node.content.substring(0, 50)}...`);

    // Find nodes that contradict this one
    const contradictingEdges = Array.from(this.edges.values()).filter(
      edge => (edge.fromNodeId === node.id || edge.toNodeId === node.id) && 
              edge.relationshipType === 'contradicts' && 
              edge.confidence > 0.6
    );

    for (const edge of contradictingEdges) {
      const otherNodeId = edge.fromNodeId === node.id ? edge.toNodeId : edge.fromNodeId;
      const otherNode = this.nodes.get(otherNodeId);
      
      if (otherNode) {
        const contradiction: Contradiction = {
          id: `contradiction_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          nodeIds: [node.id, otherNode.id],
          contradictionType: this.determineContradictionType(node, otherNode),
          severity: this.assessContradictionSeverity(node, otherNode, edge),
          description: `Contradiction detected between "${node.content.substring(0, 100)}" and "${otherNode.content.substring(0, 100)}"`,
          detectedAt: new Date(),
          resolution: 'pending',
          metadata: {
            autoDetected: true,
            reviewPriority: this.calculateReviewPriority(node, otherNode, edge),
            domain: node.metadata.domain
          }
        };

        this.contradictions.set(contradiction.id, contradiction);

        // Log high-severity contradictions
        if (['major', 'critical'].includes(contradiction.severity)) {
          await this.storage.addActivity({
            type: 'safety' as const,
            message: `${contradiction.severity.toUpperCase()} contradiction detected in ${node.metadata.domain}`,
            moduleId: 'knowledge_graph'
          });

          console.log(`🚨 ${contradiction.severity.toUpperCase()} contradiction detected between nodes`);
        }
      }
    }
  }

  /**
   * Determine the type of contradiction
   */
  private determineContradictionType(node1: KnowledgeNode, node2: KnowledgeNode): Contradiction['contradictionType'] {
    // Simple heuristics for contradiction type
    if (node1.type === 'rule' || node2.type === 'rule') return 'logical';
    if (node1.metadata.domain !== node2.metadata.domain) return 'semantic';
    if (node1.content.includes('always') || node1.content.includes('never') || 
        node2.content.includes('always') || node2.content.includes('never')) return 'logical';
    return 'empirical';
  }

  /**
   * Assess the severity of a contradiction
   */
  private assessContradictionSeverity(node1: KnowledgeNode, node2: KnowledgeNode, edge: KnowledgeEdge): Contradiction['severity'] {
    let severityScore = 0;

    // Factor in confidence and importance
    severityScore += (node1.confidence + node2.confidence) * 0.3;
    severityScore += (node1.metadata.importance + node2.metadata.importance) * 0.3;
    severityScore += edge.strength * 0.4;

    if (severityScore > 0.8) return 'critical';
    if (severityScore > 0.6) return 'major';
    if (severityScore > 0.4) return 'moderate';
    return 'minor';
  }

  /**
   * Calculate review priority for contradictions
   */
  private calculateReviewPriority(node1: KnowledgeNode, node2: KnowledgeNode, edge: KnowledgeEdge): number {
    let priority = 0;

    // Higher priority for high-confidence, important nodes
    priority += (node1.confidence + node2.confidence) * 25;
    priority += (node1.metadata.importance + node2.metadata.importance) * 25;
    priority += edge.strength * 30;

    // Boost priority for safety-critical domains
    if (['safety', 'ethics', 'security'].includes(node1.metadata.domain) ||
        ['safety', 'ethics', 'security'].includes(node2.metadata.domain)) {
      priority += 20;
    }

    return Math.min(100, Math.max(1, Math.round(priority)));
  }

  /**
   * Update knowledge clusters
   */
  private async updateClusters(node: KnowledgeNode): Promise<void> {
    // Find or create clusters for the node
    const relatedClusters = Array.from(this.clusters.values()).filter(
      cluster => cluster.domain === node.metadata.domain
    );

    if (relatedClusters.length === 0) {
      // Create new cluster
      const cluster: KnowledgeCluster = {
        id: `cluster_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        name: `${node.metadata.domain} Knowledge`,
        nodeIds: [node.id],
        coherenceScore: 1.0,
        centralConcepts: node.metadata.tags,
        domain: node.metadata.domain,
        lastAnalyzed: new Date()
      };

      this.clusters.set(cluster.id, cluster);
    } else {
      // Add to existing cluster and update coherence
      const mainCluster = relatedClusters[0];
      mainCluster.nodeIds.push(node.id);
      mainCluster.lastAnalyzed = new Date();
      mainCluster.coherenceScore = await this.calculateClusterCoherence(mainCluster);
    }
  }

  /**
   * Calculate cluster coherence score
   */
  private async calculateClusterCoherence(cluster: KnowledgeCluster): Promise<number> {
    const clusterNodes = cluster.nodeIds.map(id => this.nodes.get(id)).filter(Boolean) as KnowledgeNode[];
    if (clusterNodes.length <= 1) return 1.0;

    const clusterEdges = Array.from(this.edges.values()).filter(
      edge => cluster.nodeIds.includes(edge.fromNodeId) && cluster.nodeIds.includes(edge.toNodeId)
    );

    // Calculate coherence based on internal connections and contradictions
    const totalPossibleEdges = clusterNodes.length * (clusterNodes.length - 1) / 2;
    const actualEdges = clusterEdges.length;
    const supportiveEdges = clusterEdges.filter(e => ['supports', 'implies', 'explains'].includes(e.relationshipType)).length;
    const contradictoryEdges = clusterEdges.filter(e => e.relationshipType === 'contradicts').length;

    const connectivity = totalPossibleEdges > 0 ? actualEdges / totalPossibleEdges : 0;
    const support = actualEdges > 0 ? supportiveEdges / actualEdges : 0;
    const contradiction = actualEdges > 0 ? contradictoryEdges / actualEdges : 0;

    return Math.max(0, Math.min(1, connectivity * 0.4 + support * 0.4 - contradiction * 0.2));
  }

  /**
   * Get knowledge graph metrics
   */
  getGraphMetrics(): GraphMetrics {
    const contradictions = Array.from(this.contradictions.values());
    const nodes = Array.from(this.nodes.values());
    const edges = Array.from(this.edges.values());

    return {
      totalNodes: nodes.length,
      totalEdges: edges.length,
      totalContradictions: contradictions.length,
      unresolvedContradictions: contradictions.filter(c => c.resolution === 'pending').length,
      avgConfidence: nodes.length > 0 ? nodes.reduce((sum, n) => sum + n.confidence, 0) / nodes.length : 0,
      knowledgeDensity: nodes.length > 0 ? edges.length / nodes.length : 0,
      domainCoverage: [...new Set(nodes.map(n => n.metadata.domain))],
      lastAnalysis: new Date()
    };
  }

  /**
   * Get contradictions by severity
   */
  getContradictionsBySeverity(severity?: Contradiction['severity']): Contradiction[] {
    const contradictions = Array.from(this.contradictions.values());
    
    if (severity) {
      return contradictions.filter(c => c.severity === severity);
    }
    
    return contradictions.sort((a, b) => {
      const severityOrder = { 'critical': 4, 'major': 3, 'moderate': 2, 'minor': 1 };
      return severityOrder[b.severity] - severityOrder[a.severity];
    });
  }

  /**
   * Get knowledge nodes by domain
   */
  getNodesByDomain(domain: string): KnowledgeNode[] {
    return Array.from(this.nodes.values()).filter(n => n.metadata.domain === domain);
  }

  /**
   * Get related nodes for a given node
   */
  getRelatedNodes(nodeId: string, relationshipTypes?: string[]): { node: KnowledgeNode; relationship: KnowledgeEdge }[] {
    const relatedEdges = Array.from(this.edges.values()).filter(edge => {
      const isRelated = edge.fromNodeId === nodeId || edge.toNodeId === nodeId;
      const typeMatch = !relationshipTypes || relationshipTypes.includes(edge.relationshipType);
      return isRelated && typeMatch;
    });

    return relatedEdges.map(edge => {
      const relatedNodeId = edge.fromNodeId === nodeId ? edge.toNodeId : edge.fromNodeId;
      const node = this.nodes.get(relatedNodeId);
      return { node: node!, relationship: edge };
    }).filter(item => item.node);
  }

  /**
   * Resolve a contradiction
   */
  async resolveContradiction(
    contradictionId: string, 
    resolution: Contradiction['resolution'],
    proposedResolution?: string
  ): Promise<boolean> {
    const contradiction = this.contradictions.get(contradictionId);
    if (!contradiction) return false;

    contradiction.resolution = resolution;
    contradiction.proposedResolution = proposedResolution;

    await this.storage.addActivity({
      type: 'knowledge' as const,
      message: `Contradiction ${resolution}: ${contradiction.description.substring(0, 100)}...`,
      moduleId: 'knowledge_graph'
    });

    console.log(`✅ Contradiction ${resolution}: ${contradiction.id}`);
    return true;
  }

  /**
   * Export graph data for visualization
   */
  exportGraphData(): {
    nodes: any[];
    edges: any[];
    clusters: any[];
    contradictions: any[];
    metrics: GraphMetrics;
  } {
    const nodes = Array.from(this.nodes.values()).map(node => ({
      id: node.id,
      label: node.content.substring(0, 50) + (node.content.length > 50 ? '...' : ''),
      type: node.type,
      confidence: node.confidence,
      domain: node.metadata.domain,
      importance: node.metadata.importance,
      verificationLevel: node.metadata.verificationLevel
    }));

    const edges = Array.from(this.edges.values()).map(edge => ({
      id: edge.id,
      from: edge.fromNodeId,
      to: edge.toNodeId,
      label: edge.relationshipType,
      strength: edge.strength,
      confidence: edge.confidence
    }));

    return {
      nodes,
      edges,
      clusters: Array.from(this.clusters.values()),
      contradictions: Array.from(this.contradictions.values()),
      metrics: this.getGraphMetrics()
    };
  }
}