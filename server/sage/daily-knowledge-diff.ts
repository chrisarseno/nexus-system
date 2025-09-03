import { IStorage } from '../storage';
import { KnowledgeGraphEngine } from './knowledge-graph';
import { LocalAIService } from './local-ai-service';

export interface KnowledgeDiff {
  date: string;
  id: string;
  changes: {
    added: number;
    modified: number;
    removed: number;
    contradictions: number;
  };
  domains: string[];
  keyInsights: string[];
  beliefScoreChanges: {
    nodeId: string;
    content: string;
    previousScore: number;
    newScore: number;
    reason: string;
  }[];
  newContradictions: {
    id: string;
    description: string;
    severity: string;
    involvedNodes: string[];
  }[];
  resolvedContradictions: {
    id: string;
    description: string;
    resolution: string;
  }[];
  summary: string;
  totalNodes: number;
  totalEdges: number;
  avgConfidence: number;
}

export interface KnowledgeSnapshot {
  date: string;
  totalNodes: number;
  totalEdges: number;
  totalContradictions: number;
  avgConfidence: number;
  domainDistribution: Record<string, number>;
  nodeConfidences: Record<string, number>;
  contradictionIds: string[];
  lastModified: Date;
}

/**
 * Daily Knowledge Diff Generator
 * Tracks changes in the knowledge graph over time and generates meaningful diffs
 */
export class DailyKnowledgeDiffEngine {
  private storage: IStorage;
  private knowledgeGraph: KnowledgeGraphEngine;
  private localAI: LocalAIService;
  private snapshots: Map<string, KnowledgeSnapshot> = new Map();

  constructor(storage: IStorage, knowledgeGraph: KnowledgeGraphEngine, localAI: LocalAIService) {
    this.storage = storage;
    this.knowledgeGraph = knowledgeGraph;
    this.localAI = localAI;
  }

  /**
   * Generate a daily knowledge diff
   */
  async generateDailyDiff(date?: string): Promise<KnowledgeDiff> {
    const today = date || new Date().toISOString().split('T')[0];
    const yesterday = new Date(new Date(today).getTime() - 24 * 60 * 60 * 1000)
      .toISOString().split('T')[0];

    console.log(`📊 Generating knowledge diff for ${today}`);

    // Get current and previous snapshots
    const currentSnapshot = await this.createSnapshot(today);
    const previousSnapshot = this.snapshots.get(yesterday) || await this.createEmptySnapshot(yesterday);

    // Store current snapshot
    this.snapshots.set(today, currentSnapshot);

    // Calculate changes
    const changes = {
      added: Math.max(0, currentSnapshot.totalNodes - previousSnapshot.totalNodes),
      modified: 0, // Will calculate based on confidence changes
      removed: Math.max(0, previousSnapshot.totalNodes - currentSnapshot.totalNodes),
      contradictions: Math.max(0, currentSnapshot.totalContradictions - previousSnapshot.totalContradictions)
    };

    // Calculate belief score changes
    const beliefScoreChanges = this.calculateBeliefScoreChanges(
      previousSnapshot.nodeConfidences,
      currentSnapshot.nodeConfidences
    );
    changes.modified = beliefScoreChanges.length;

    // Get new and resolved contradictions
    const newContradictions = await this.getNewContradictions(
      previousSnapshot.contradictionIds,
      currentSnapshot.contradictionIds
    );

    const resolvedContradictions = await this.getResolvedContradictions(
      previousSnapshot.contradictionIds,
      currentSnapshot.contradictionIds
    );

    // Get key domains affected
    const domains = Object.keys(currentSnapshot.domainDistribution);

    // Generate key insights using AI
    const keyInsights = await this.generateKeyInsights(changes, domains, beliefScoreChanges, newContradictions);

    // Generate summary
    const summary = await this.generateSummary(changes, domains, keyInsights, newContradictions);

    const diff: KnowledgeDiff = {
      date: today,
      id: `diff_${today}_${Date.now()}`,
      changes,
      domains,
      keyInsights,
      beliefScoreChanges,
      newContradictions,
      resolvedContradictions,
      summary,
      totalNodes: currentSnapshot.totalNodes,
      totalEdges: currentSnapshot.totalEdges,
      avgConfidence: currentSnapshot.avgConfidence
    };

    // Log the diff
    await this.storage.addActivity({
      type: 'knowledge' as const,
      message: `Daily diff: +${changes.added} facts, ${changes.contradictions} new contradictions`,
      moduleId: 'knowledge_diff'
    });

    console.log(`📈 Daily knowledge diff completed: +${changes.added} nodes, ${newContradictions.length} new contradictions`);
    
    return diff;
  }

  /**
   * Create a knowledge snapshot for a given date
   */
  private async createSnapshot(date: string): Promise<KnowledgeSnapshot> {
    const graphData = this.knowledgeGraph.exportGraphData();
    const metrics = graphData.metrics;

    // Calculate domain distribution
    const domainDistribution: Record<string, number> = {};
    graphData.nodes.forEach(node => {
      domainDistribution[node.domain] = (domainDistribution[node.domain] || 0) + 1;
    });

    // Extract node confidences
    const nodeConfidences: Record<string, number> = {};
    graphData.nodes.forEach(node => {
      nodeConfidences[node.id] = node.confidence;
    });

    // Extract contradiction IDs
    const contradictionIds = graphData.contradictions
      .filter(c => c.resolution === 'pending')
      .map(c => c.id);

    return {
      date,
      totalNodes: metrics.totalNodes,
      totalEdges: metrics.totalEdges,
      totalContradictions: metrics.unresolvedContradictions,
      avgConfidence: metrics.avgConfidence,
      domainDistribution,
      nodeConfidences,
      contradictionIds,
      lastModified: new Date()
    };
  }

  /**
   * Create an empty snapshot for comparison
   */
  private async createEmptySnapshot(date: string): Promise<KnowledgeSnapshot> {
    return {
      date,
      totalNodes: 0,
      totalEdges: 0,
      totalContradictions: 0,
      avgConfidence: 0,
      domainDistribution: {},
      nodeConfidences: {},
      contradictionIds: [],
      lastModified: new Date()
    };
  }

  /**
   * Calculate belief score changes between snapshots
   */
  private calculateBeliefScoreChanges(
    previousConfidences: Record<string, number>,
    currentConfidences: Record<string, number>
  ): KnowledgeDiff['beliefScoreChanges'] {
    const changes: KnowledgeDiff['beliefScoreChanges'] = [];

    for (const [nodeId, currentScore] of Object.entries(currentConfidences)) {
      const previousScore = previousConfidences[nodeId];
      
      if (previousScore !== undefined && Math.abs(currentScore - previousScore) > 0.1) {
        const graphData = this.knowledgeGraph.exportGraphData();
        const node = graphData.nodes.find(n => n.id === nodeId);
        
        if (node) {
          changes.push({
            nodeId,
            content: node.label,
            previousScore,
            newScore: currentScore,
            reason: currentScore > previousScore ? 
              'Increased confidence due to supporting evidence' : 
              'Decreased confidence due to conflicting information'
          });
        }
      }
    }

    return changes.sort((a, b) => Math.abs(b.newScore - b.previousScore) - Math.abs(a.newScore - a.previousScore));
  }

  /**
   * Get newly detected contradictions
   */
  private async getNewContradictions(
    previousIds: string[],
    currentIds: string[]
  ): Promise<KnowledgeDiff['newContradictions']> {
    const newIds = currentIds.filter(id => !previousIds.includes(id));
    const graphData = this.knowledgeGraph.exportGraphData();
    
    return newIds.map(id => {
      const contradiction = graphData.contradictions.find(c => c.id === id);
      return contradiction ? {
        id: contradiction.id,
        description: contradiction.description,
        severity: contradiction.severity,
        involvedNodes: contradiction.nodeIds
      } : {
        id,
        description: 'Unknown contradiction',
        severity: 'minor',
        involvedNodes: []
      };
    });
  }

  /**
   * Get resolved contradictions
   */
  private async getResolvedContradictions(
    previousIds: string[],
    currentIds: string[]
  ): Promise<KnowledgeDiff['resolvedContradictions']> {
    const resolvedIds = previousIds.filter(id => !currentIds.includes(id));
    const graphData = this.knowledgeGraph.exportGraphData();
    
    return resolvedIds.map(id => {
      const contradiction = graphData.contradictions.find(c => c.id === id);
      return contradiction ? {
        id: contradiction.id,
        description: contradiction.description,
        resolution: contradiction.resolution || 'resolved'
      } : {
        id,
        description: 'Previously detected contradiction',
        resolution: 'resolved'
      };
    });
  }

  /**
   * Generate key insights using AI
   */
  private async generateKeyInsights(
    changes: KnowledgeDiff['changes'],
    domains: string[],
    beliefChanges: KnowledgeDiff['beliefScoreChanges'],
    contradictions: KnowledgeDiff['newContradictions']
  ): Promise<string[]> {
    try {
      const prompt = `Analyze these knowledge changes and identify key insights:

Changes: ${changes.added} added, ${changes.modified} modified, ${changes.removed} removed, ${changes.contradictions} new contradictions
Affected domains: ${domains.join(', ')}
Top belief changes: ${beliefChanges.slice(0, 3).map(b => `${b.content.substring(0, 50)} (${b.previousScore.toFixed(2)} → ${b.newScore.toFixed(2)})`).join('; ')}
New contradictions: ${contradictions.slice(0, 2).map(c => c.description.substring(0, 80)).join('; ')}

Provide 3-5 key insights about what these changes reveal about the knowledge evolution. Focus on patterns, trends, and implications.

Format as a JSON array of insight strings.`;

      const response = await this.localAI.generateResponse(prompt, 'analysis', 0.2, 500);
      
      try {
        const insights = JSON.parse(response.content);
        return Array.isArray(insights) ? insights.slice(0, 5) : [
          `Added ${changes.added} new knowledge nodes across ${domains.length} domains`,
          `Detected ${changes.contradictions} new contradictions requiring review`,
          `${beliefChanges.length} existing facts had significant confidence changes`
        ];
      } catch {
        return [
          `Added ${changes.added} new knowledge nodes`,
          `Modified ${changes.modified} existing facts`,
          `Detected ${changes.contradictions} new contradictions`
        ];
      }
    } catch (error) {
      console.warn('Failed to generate AI insights:', error);
      return [
        `Knowledge base grew by ${changes.added} facts`,
        `${domains.length} domains were affected by changes`,
        `System detected ${changes.contradictions} new conflicts`
      ];
    }
  }

  /**
   * Generate summary using AI
   */
  private async generateSummary(
    changes: KnowledgeDiff['changes'],
    domains: string[],
    insights: string[],
    contradictions: KnowledgeDiff['newContradictions']
  ): Promise<string> {
    try {
      const prompt = `Create a concise summary of today's knowledge changes:

Changes: +${changes.added} new, ~${changes.modified} modified, -${changes.removed} removed
Key insights: ${insights.join('; ')}
New contradictions: ${contradictions.length}
Affected domains: ${domains.join(', ')}

Write a 2-3 sentence summary highlighting the most significant developments.`;

      const response = await this.localAI.generateResponse(prompt, 'synthesis', 0.3, 200);
      return response.content.trim() || `Added ${changes.added} new knowledge facts across ${domains.length} domains, with ${contradictions.length} new contradictions detected.`;
    } catch (error) {
      console.warn('Failed to generate AI summary:', error);
      return `Knowledge base updated with ${changes.added} new facts and ${contradictions.length} new contradictions across ${domains.length} domains.`;
    }
  }

  /**
   * Get historical diffs for a date range
   */
  async getHistoricalDiffs(startDate: string, endDate: string): Promise<KnowledgeDiff[]> {
    const diffs: KnowledgeDiff[] = [];
    const start = new Date(startDate);
    const end = new Date(endDate);
    
    for (let date = new Date(start); date <= end; date.setDate(date.getDate() + 1)) {
      const dateStr = date.toISOString().split('T')[0];
      try {
        const diff = await this.generateDailyDiff(dateStr);
        diffs.push(diff);
      } catch (error) {
        console.warn(`Failed to generate diff for ${dateStr}:`, error);
      }
    }
    
    return diffs;
  }

  /**
   * Get knowledge growth metrics
   */
  getGrowthMetrics(): {
    totalSnapshots: number;
    dateRange: { start: string; end: string } | null;
    avgDailyGrowth: number;
    domains: string[];
  } {
    const dates = Array.from(this.snapshots.keys()).sort();
    
    if (dates.length === 0) {
      return {
        totalSnapshots: 0,
        dateRange: null,
        avgDailyGrowth: 0,
        domains: []
      };
    }
    
    const firstSnapshot = this.snapshots.get(dates[0])!;
    const lastSnapshot = this.snapshots.get(dates[dates.length - 1])!;
    
    const daysDiff = dates.length > 1 ? dates.length - 1 : 1;
    const totalGrowth = lastSnapshot.totalNodes - firstSnapshot.totalNodes;
    
    const allDomains = new Set<string>();
    this.snapshots.forEach(snapshot => {
      Object.keys(snapshot.domainDistribution).forEach(domain => allDomains.add(domain));
    });
    
    return {
      totalSnapshots: this.snapshots.size,
      dateRange: { start: dates[0], end: dates[dates.length - 1] },
      avgDailyGrowth: totalGrowth / daysDiff,
      domains: Array.from(allDomains)
    };
  }
}