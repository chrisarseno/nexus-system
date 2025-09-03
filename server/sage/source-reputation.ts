import { IStorage } from '../storage';
import { LocalAIService } from './local-ai-service';

export interface SourceRecord {
  id: string;
  name: string;
  type: 'academic' | 'news' | 'government' | 'expert' | 'community' | 'ai_generated' | 'unknown';
  domain: string;
  firstSeen: Date;
  lastSeen: Date;
  totalCitations: number;
  correctPredictions: number;
  incorrectPredictions: number;
  contradictions: number;
  verificationAttempts: number;
  metadata: {
    url?: string;
    author?: string;
    publication?: string;
    impact_factor?: number;
    peer_reviewed?: boolean;
    language?: string;
    region?: string;
  };
}

export interface SourceReputation {
  sourceId: string;
  reputationScore: number; // 0-1 scale
  confidenceLevel: 'very_low' | 'low' | 'medium' | 'high' | 'very_high';
  trackRecord: {
    accuracy: number;
    reliability: number;
    consistency: number;
    timeliness: number;
    bias_score: number;
  };
  flags: string[];
  lastUpdated: Date;
  details: {
    totalFactsContributed: number;
    verificationSuccessRate: number;
    contradictionRate: number;
    expertEndorsements: number;
    communityTrust: number;
  };
}

export interface FactVerification {
  factId: string;
  sourceId: string;
  verificationResult: 'verified' | 'disputed' | 'false' | 'unverifiable';
  confidence: number;
  verificationMethod: 'cross_reference' | 'expert_review' | 'ai_analysis' | 'consensus' | 'experiment';
  timestamp: Date;
  verifierId?: string;
  notes?: string;
}

export interface MultiSourceConsensus {
  factContent: string;
  sources: {
    sourceId: string;
    sourceName: string;
    confidence: number;
    position: 'supports' | 'contradicts' | 'neutral' | 'unclear';
    reputation: number;
  }[];
  consensusScore: number;
  majorityPosition: 'supports' | 'contradicts' | 'split' | 'insufficient_data';
  weightedConsensus: number;
  highReputationConsensus: number;
  conflictingSourcesCount: number;
  recommendedAction: 'accept' | 'reject' | 'investigate' | 'mark_disputed';
}

/**
 * Source Reputation Engine for NEXUS
 * Tracks and evaluates the reliability of information sources
 */
export class SourceReputationEngine {
  private storage: IStorage;
  private localAI: LocalAIService;
  private sources: Map<string, SourceRecord> = new Map();
  private reputations: Map<string, SourceReputation> = new Map();
  private verifications: Map<string, FactVerification[]> = new Map();

  constructor(storage: IStorage, localAI: LocalAIService) {
    this.storage = storage;
    this.localAI = localAI;
  }

  /**
   * Add or update a source
   */
  async addSource(name: string, type: SourceRecord['type'], metadata: Partial<SourceRecord['metadata']> = {}): Promise<SourceRecord> {
    const existingSource = Array.from(this.sources.values()).find(s => s.name === name);
    
    if (existingSource) {
      existingSource.lastSeen = new Date();
      existingSource.totalCitations++;
      return existingSource;
    }

    const source: SourceRecord = {
      id: `source_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name,
      type,
      domain: this.inferDomain(name, metadata.url),
      firstSeen: new Date(),
      lastSeen: new Date(),
      totalCitations: 1,
      correctPredictions: 0,
      incorrectPredictions: 0,
      contradictions: 0,
      verificationAttempts: 0,
      metadata
    };

    this.sources.set(source.id, source);

    // Initialize reputation
    await this.initializeReputation(source);

    console.log(`📚 New source registered: ${name} (${type})`);
    return source;
  }

  /**
   * Initialize reputation for a new source
   */
  private async initializeReputation(source: SourceRecord): Promise<void> {
    // Calculate initial reputation based on source type and metadata
    let initialScore = 0.5; // Neutral starting point

    // Adjust based on source type
    switch (source.type) {
      case 'academic':
        initialScore = 0.8;
        break;
      case 'government':
        initialScore = 0.7;
        break;
      case 'expert':
        initialScore = 0.6;
        break;
      case 'news':
        initialScore = 0.5;
        break;
      case 'community':
        initialScore = 0.4;
        break;
      case 'ai_generated':
        initialScore = 0.3;
        break;
      case 'unknown':
        initialScore = 0.2;
        break;
    }

    // Adjust for peer review
    if (source.metadata.peer_reviewed) {
      initialScore += 0.1;
    }

    // Adjust for impact factor
    if (source.metadata.impact_factor && source.metadata.impact_factor > 1) {
      initialScore += Math.min(0.1, source.metadata.impact_factor / 50);
    }

    const reputation: SourceReputation = {
      sourceId: source.id,
      reputationScore: Math.min(1, initialScore),
      confidenceLevel: this.calculateConfidenceLevel(Math.min(1, initialScore)),
      trackRecord: {
        accuracy: initialScore,
        reliability: initialScore,
        consistency: initialScore,
        timeliness: 0.5,
        bias_score: 0.5
      },
      flags: [],
      lastUpdated: new Date(),
      details: {
        totalFactsContributed: 0,
        verificationSuccessRate: 0,
        contradictionRate: 0,
        expertEndorsements: 0,
        communityTrust: initialScore
      }
    };

    this.reputations.set(source.id, reputation);
  }

  /**
   * Verify a fact against multiple sources
   */
  async verifyFactWithSources(
    factContent: string, 
    claimedSources: string[],
    verificationMethod: FactVerification['verificationMethod'] = 'ai_analysis'
  ): Promise<MultiSourceConsensus> {
    console.log(`🔍 Multi-source verification: "${factContent.substring(0, 50)}..."`);

    const sourceAnalyses: MultiSourceConsensus['sources'] = [];

    // Analyze each source
    for (const sourceName of claimedSources) {
      let source = Array.from(this.sources.values()).find(s => s.name === sourceName);
      
      if (!source) {
        source = await this.addSource(sourceName, 'unknown');
      }

      const reputation = this.reputations.get(source.id);
      if (!reputation) continue;

      // AI analysis of source credibility for this fact
      const analysis = await this.analyzeSourceForFact(source, factContent);
      
      sourceAnalyses.push({
        sourceId: source.id,
        sourceName: source.name,
        confidence: analysis.confidence,
        position: analysis.position,
        reputation: reputation.reputationScore
      });

      // Record verification attempt
      source.verificationAttempts++;
      
      const verification: FactVerification = {
        factId: `fact_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        sourceId: source.id,
        verificationResult: analysis.position === 'supports' ? 'verified' : 
                          analysis.position === 'contradicts' ? 'disputed' : 'unverifiable',
        confidence: analysis.confidence,
        verificationMethod,
        timestamp: new Date()
      };

      if (!this.verifications.has(source.id)) {
        this.verifications.set(source.id, []);
      }
      this.verifications.get(source.id)!.push(verification);
    }

    // Calculate consensus
    const consensus = this.calculateMultiSourceConsensus(factContent, sourceAnalyses);
    
    // Update source reputations based on consensus
    await this.updateReputationsFromConsensus(sourceAnalyses, consensus);

    console.log(`📊 Multi-source consensus: ${consensus.consensusScore.toFixed(2)} (${consensus.majorityPosition})`);
    
    return consensus;
  }

  /**
   * Analyze how a source relates to a specific fact
   */
  private async analyzeSourceForFact(
    source: SourceRecord, 
    factContent: string
  ): Promise<{ confidence: number; position: MultiSourceConsensus['sources'][0]['position'] }> {
    try {
      const prompt = `Analyze how reliable this source would be for the following fact:

Source: ${source.name} (${source.type})
Domain: ${source.domain}
Metadata: ${JSON.stringify(source.metadata)}
Fact: "${factContent}"

Consider:
1. Is this source likely to have expertise in the fact's domain?
2. What is their track record for similar claims?
3. Are there any obvious biases or conflicts of interest?
4. How does the source type relate to the fact's verifiability?

Respond with JSON: {"confidence": 0.0-1.0, "position": "supports|contradicts|neutral|unclear", "reasoning": "brief explanation"}`;

      const response = await this.localAI.generateResponse(prompt, 'analysis', 0.2, 300);
      const analysis = this.parseSourceAnalysis(response.content);
      
      return {
        confidence: analysis?.confidence || 0.3,
        position: analysis?.position || 'unclear'
      };
    } catch (error) {
      console.warn(`Failed to analyze source ${source.name}:`, error);
      return { confidence: 0.3, position: 'unclear' };
    }
  }

  /**
   * Parse AI analysis of source reliability
   */
  private parseSourceAnalysis(response: string): { confidence: number; position: string; reasoning: string } | null {
    try {
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]);
      }
    } catch (error) {
      console.warn('Failed to parse source analysis:', error);
    }
    return null;
  }

  /**
   * Calculate multi-source consensus
   */
  private calculateMultiSourceConsensus(
    factContent: string,
    sources: MultiSourceConsensus['sources']
  ): MultiSourceConsensus {
    if (sources.length === 0) {
      return {
        factContent,
        sources: [],
        consensusScore: 0,
        majorityPosition: 'insufficient_data',
        weightedConsensus: 0,
        highReputationConsensus: 0,
        conflictingSourcesCount: 0,
        recommendedAction: 'investigate'
      };
    }

    // Count positions
    const positions = {
      supports: sources.filter(s => s.position === 'supports'),
      contradicts: sources.filter(s => s.position === 'contradicts'),
      neutral: sources.filter(s => s.position === 'neutral'),
      unclear: sources.filter(s => s.position === 'unclear')
    };

    // Calculate weighted consensus (reputation * confidence)
    const weightedSupport = positions.supports.reduce((sum, s) => sum + (s.reputation * s.confidence), 0);
    const weightedContradict = positions.contradicts.reduce((sum, s) => sum + (s.reputation * s.confidence), 0);
    const totalWeighted = weightedSupport + weightedContradict;
    
    const weightedConsensus = totalWeighted > 0 ? weightedSupport / totalWeighted : 0.5;

    // High reputation sources (>0.7) consensus
    const highRepSources = sources.filter(s => s.reputation > 0.7);
    const highRepSupport = highRepSources.filter(s => s.position === 'supports').length;
    const highRepContradict = highRepSources.filter(s => s.position === 'contradicts').length;
    const highRepTotal = highRepSupport + highRepContradict;
    
    const highReputationConsensus = highRepTotal > 0 ? highRepSupport / highRepTotal : 0.5;

    // Determine majority position
    let majorityPosition: MultiSourceConsensus['majorityPosition'];
    if (positions.supports.length > positions.contradicts.length * 2) {
      majorityPosition = 'supports';
    } else if (positions.contradicts.length > positions.supports.length * 2) {
      majorityPosition = 'contradicts';
    } else if (Math.abs(positions.supports.length - positions.contradicts.length) <= 1) {
      majorityPosition = 'split';
    } else {
      majorityPosition = 'insufficient_data';
    }

    // Calculate overall consensus score
    const consensusScore = (weightedConsensus + highReputationConsensus + 
                           (positions.supports.length / sources.length)) / 3;

    // Count conflicting sources
    const conflictingSourcesCount = Math.min(positions.supports.length, positions.contradicts.length);

    // Determine recommended action
    let recommendedAction: MultiSourceConsensus['recommendedAction'];
    if (consensusScore > 0.8 && conflictingSourcesCount === 0) {
      recommendedAction = 'accept';
    } else if (consensusScore < 0.2 && conflictingSourcesCount === 0) {
      recommendedAction = 'reject';
    } else if (conflictingSourcesCount > 0) {
      recommendedAction = 'mark_disputed';
    } else {
      recommendedAction = 'investigate';
    }

    return {
      factContent,
      sources,
      consensusScore,
      majorityPosition,
      weightedConsensus,
      highReputationConsensus,
      conflictingSourcesCount,
      recommendedAction
    };
  }

  /**
   * Update source reputations based on consensus results
   */
  private async updateReputationsFromConsensus(
    sources: MultiSourceConsensus['sources'],
    consensus: MultiSourceConsensus
  ): Promise<void> {
    for (const sourceData of sources) {
      const reputation = this.reputations.get(sourceData.sourceId);
      const source = this.sources.get(sourceData.sourceId);
      
      if (!reputation || !source) continue;

      // Update based on alignment with consensus
      const alignsWithMajority = 
        (consensus.majorityPosition === 'supports' && sourceData.position === 'supports') ||
        (consensus.majorityPosition === 'contradicts' && sourceData.position === 'contradicts');

      if (alignsWithMajority) {
        source.correctPredictions++;
        reputation.trackRecord.accuracy += 0.01;
        reputation.details.verificationSuccessRate = 
          source.correctPredictions / (source.correctPredictions + source.incorrectPredictions + 1);
      } else if (consensus.majorityPosition !== 'split' && consensus.majorityPosition !== 'insufficient_data') {
        source.incorrectPredictions++;
        reputation.trackRecord.accuracy = Math.max(0, reputation.trackRecord.accuracy - 0.01);
      }

      // Update contradiction count
      if (consensus.conflictingSourcesCount > 0) {
        source.contradictions++;
        reputation.details.contradictionRate = source.contradictions / source.totalCitations;
      }

      // Recalculate reputation score
      reputation.reputationScore = Math.max(0, Math.min(1, 
        (reputation.trackRecord.accuracy + 
         reputation.trackRecord.reliability + 
         reputation.trackRecord.consistency - 
         reputation.details.contradictionRate) / 3
      ));

      reputation.confidenceLevel = this.calculateConfidenceLevel(reputation.reputationScore);
      reputation.lastUpdated = new Date();

      // Add flags for problematic sources
      if (reputation.details.contradictionRate > 0.3) {
        if (!reputation.flags.includes('high_contradiction_rate')) {
          reputation.flags.push('high_contradiction_rate');
        }
      }

      if (reputation.details.verificationSuccessRate < 0.3) {
        if (!reputation.flags.includes('low_accuracy')) {
          reputation.flags.push('low_accuracy');
        }
      }
    }
  }

  /**
   * Calculate confidence level from reputation score
   */
  private calculateConfidenceLevel(score: number): SourceReputation['confidenceLevel'] {
    if (score >= 0.9) return 'very_high';
    if (score >= 0.7) return 'high';
    if (score >= 0.5) return 'medium';
    if (score >= 0.3) return 'low';
    return 'very_low';
  }

  /**
   * Infer domain from source name or URL
   */
  private inferDomain(name: string, url?: string): string {
    const text = `${name} ${url || ''}`.toLowerCase();
    
    if (text.includes('science') || text.includes('research') || text.includes('journal')) return 'science';
    if (text.includes('news') || text.includes('media') || text.includes('press')) return 'journalism';
    if (text.includes('gov') || text.includes('government') || text.includes('official')) return 'government';
    if (text.includes('edu') || text.includes('university') || text.includes('academic')) return 'academic';
    if (text.includes('medical') || text.includes('health') || text.includes('medicine')) return 'healthcare';
    if (text.includes('tech') || text.includes('technology') || text.includes('software')) return 'technology';
    if (text.includes('wikipedia') || text.includes('wiki')) return 'encyclopedia';
    
    return 'general';
  }

  /**
   * Get source reputation by ID
   */
  getSourceReputation(sourceId: string): SourceReputation | undefined {
    return this.reputations.get(sourceId);
  }

  /**
   * Get top sources by reputation
   */
  getTopSources(limit: number = 10, domain?: string): { source: SourceRecord; reputation: SourceReputation }[] {
    const results: { source: SourceRecord; reputation: SourceReputation }[] = [];

    for (const [sourceId, reputation] of this.reputations) {
      const source = this.sources.get(sourceId);
      if (source && (!domain || source.domain === domain)) {
        results.push({ source, reputation });
      }
    }

    return results
      .sort((a, b) => b.reputation.reputationScore - a.reputation.reputationScore)
      .slice(0, limit);
  }

  /**
   * Get sources with flags (problematic sources)
   */
  getFlaggedSources(): { source: SourceRecord; reputation: SourceReputation }[] {
    const results: { source: SourceRecord; reputation: SourceReputation }[] = [];

    for (const [sourceId, reputation] of this.reputations) {
      if (reputation.flags.length > 0) {
        const source = this.sources.get(sourceId);
        if (source) {
          results.push({ source, reputation });
        }
      }
    }

    return results.sort((a, b) => b.reputation.details.contradictionRate - a.reputation.details.contradictionRate);
  }

  /**
   * Get reputation statistics
   */
  getReputationMetrics(): {
    totalSources: number;
    averageReputation: number;
    highReputationSources: number;
    flaggedSources: number;
    domainDistribution: Record<string, number>;
    typeDistribution: Record<string, number>;
  } {
    const sources = Array.from(this.sources.values());
    const reputations = Array.from(this.reputations.values());

    const domainDistribution: Record<string, number> = {};
    const typeDistribution: Record<string, number> = {};

    sources.forEach(source => {
      domainDistribution[source.domain] = (domainDistribution[source.domain] || 0) + 1;
      typeDistribution[source.type] = (typeDistribution[source.type] || 0) + 1;
    });

    return {
      totalSources: sources.length,
      averageReputation: reputations.length > 0 ? 
        reputations.reduce((sum, r) => sum + r.reputationScore, 0) / reputations.length : 0,
      highReputationSources: reputations.filter(r => r.reputationScore > 0.7).length,
      flaggedSources: reputations.filter(r => r.flags.length > 0).length,
      domainDistribution,
      typeDistribution
    };
  }
}