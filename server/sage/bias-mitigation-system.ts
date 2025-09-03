import { IStorage } from '../storage';

// Bias mitigation types
export enum BiasType {
  CONFIRMATION_BIAS = "confirmation_bias",
  SELECTION_BIAS = "selection_bias",
  CULTURAL_BIAS = "cultural_bias",
  COGNITIVE_BIAS = "cognitive_bias",
  TEMPORAL_BIAS = "temporal_bias",
  AUTHORITY_BIAS = "authority_bias", 
  ANCHORING_BIAS = "anchoring_bias",
  AVAILABILITY_BIAS = "availability_bias"
}

export enum PerspectiveSource {
  INDIVIDUAL_USER = "individual_user",
  EXPERT_PANEL = "expert_panel",
  DIVERSE_COMMUNITY = "diverse_community",
  HISTORICAL_WISDOM = "historical_wisdom",
  CROSS_CULTURAL = "cross_cultural",
  PHILOSOPHICAL_TRADITION = "philosophical_tradition",
  SCIENTIFIC_CONSENSUS = "scientific_consensus",
  ETHICAL_FRAMEWORK = "ethical_framework"
}

export enum VirtueCategory {
  WISDOM = "wisdom",
  JUSTICE = "justice",
  COURAGE = "courage",
  TEMPERANCE = "temperance",
  COMPASSION = "compassion",
  INTEGRITY = "integrity",
  HUMILITY = "humility",
  PATIENCE = "patience"
}

export interface PerspectiveInput {
  perspectiveId: string;
  sourceType: PerspectiveSource;
  sourceIdentifier: string;
  viewpoint: Record<string, any>;
  confidence: number;
  reasoning: string[];
  culturalContext: Record<string, any>;
  timestamp: Date;
}

export interface BiasDetectionResult {
  biasDetected: boolean;
  biasTypes: BiasType[];
  severityScore: number;
  affectedDomains: string[];
  mitigationRecommendations: string[];
  confidence: number;
}

export interface VirtueAssessment {
  virtueScores: Record<VirtueCategory, number>;
  overallVirtueAlignment: number;
  virtueConflicts: string[];
  improvementRecommendations: string[];
}

/**
 * Bias Mitigation System for Ethical AI Development
 * Prevents bias accumulation while promoting virtue-based learning
 */
export class BiasMitigationSystem {
  private storage: IStorage;
  private perspectives: Map<string, PerspectiveInput> = new Map();
  private biasHistory: BiasDetectionResult[] = [];
  private virtueAssessments: Map<string, VirtueAssessment> = new Map();
  private sourceDistribution: Map<PerspectiveSource, number> = new Map();

  constructor(storage: IStorage) {
    this.storage = storage;
    this.initializeSourceTracking();
    console.log('⚖️ Bias Mitigation System initialized');
  }

  /**
   * Initialize source tracking for balanced perspectives
   */
  private initializeSourceTracking(): void {
    Object.values(PerspectiveSource).forEach(source => {
      this.sourceDistribution.set(source, 0);
    });
  }

  /**
   * Add perspective input for balanced learning
   */
  async addPerspectiveInput(input: PerspectiveInput): Promise<void> {
    // Calculate balanced weight for this perspective
    const currentDistribution = this.getCurrentDistribution();
    const weight = this.calculatePerspectiveWeight(input, currentDistribution);
    
    // Store perspective with calculated weight
    const weightedInput = {
      ...input,
      calculatedWeight: weight
    };
    
    this.perspectives.set(input.perspectiveId, input);
    
    // Update source distribution tracking
    const currentCount = this.sourceDistribution.get(input.sourceType) || 0;
    this.sourceDistribution.set(input.sourceType, currentCount + weight);
    
    console.log(`📥 Added perspective from ${input.sourceType} with weight ${weight.toFixed(3)}`);
    
    // Run bias detection on new input
    await this.detectBias([input]);
  }

  /**
   * Calculate weight for perspective to promote balance
   */
  private calculatePerspectiveWeight(
    input: PerspectiveInput, 
    currentDistribution: Record<string, number>
  ): number {
    // Reduce weight if this source type is over-represented
    const sourceRepresentation = currentDistribution[input.sourceType] || 0;
    const totalPerspectives = Object.values(currentDistribution).reduce((sum, val) => sum + val, 0);
    const representationRatio = totalPerspectives > 0 ? sourceRepresentation / totalPerspectives : 0;
    
    const balanceFactor = Math.max(0.1, 1.0 - representationRatio);
    
    // Factor in confidence and reasoning quality
    const reasoningQuality = Math.min(1.0, input.reasoning.length / 5.0); // Normalize to 5 reasons
    const qualityFactor = (input.confidence + reasoningQuality) / 2.0;
    
    return Math.min(1.0, balanceFactor * qualityFactor);
  }

  /**
   * Get current source distribution
   */
  private getCurrentDistribution(): Record<string, number> {
    const distribution: Record<string, number> = {};
    Array.from(this.sourceDistribution.entries()).forEach(([source, count]) => {
      distribution[source] = count;
    });
    return distribution;
  }

  /**
   * Detect bias in perspectives or decisions
   */
  async detectBias(
    perspectives: PerspectiveInput[] = [],
    decisionContext?: Record<string, any>
  ): Promise<BiasDetectionResult> {
    const analysisInputs = perspectives.length > 0 ? perspectives : Array.from(this.perspectives.values());
    
    const detectedBiases: BiasType[] = [];
    let severityScore = 0;
    const affectedDomains: string[] = [];
    const recommendations: string[] = [];
    
    // Check for confirmation bias
    if (this.detectConfirmationBias(analysisInputs)) {
      detectedBiases.push(BiasType.CONFIRMATION_BIAS);
      severityScore += 0.3;
      recommendations.push('Actively seek contradicting evidence and perspectives');
    }
    
    // Check for selection bias
    if (this.detectSelectionBias(analysisInputs)) {
      detectedBiases.push(BiasType.SELECTION_BIAS);
      severityScore += 0.25;
      recommendations.push('Diversify information sources and sampling methods');
    }
    
    // Check for cultural bias
    if (this.detectCulturalBias(analysisInputs)) {
      detectedBiases.push(BiasType.CULTURAL_BIAS);
      severityScore += 0.2;
      recommendations.push('Include more cross-cultural perspectives');
    }
    
    // Check for authority bias
    if (this.detectAuthorityBias(analysisInputs)) {
      detectedBiases.push(BiasType.AUTHORITY_BIAS);
      severityScore += 0.15;
      recommendations.push('Evaluate arguments independently of source authority');
    }
    
    // Check for temporal bias (recency bias)
    if (this.detectTemporalBias(analysisInputs)) {
      detectedBiases.push(BiasType.TEMPORAL_BIAS);
      severityScore += 0.1;
      recommendations.push('Balance recent and historical perspectives');
    }
    
    // Determine affected domains
    const domains = new Set<string>();
    analysisInputs.forEach(input => {
      if (input.viewpoint.domain) {
        domains.add(input.viewpoint.domain);
      }
    });
    affectedDomains.push(...Array.from(domains));
    
    const result: BiasDetectionResult = {
      biasDetected: detectedBiases.length > 0,
      biasTypes: detectedBiases,
      severityScore: Math.min(1.0, severityScore),
      affectedDomains,
      mitigationRecommendations: recommendations,
      confidence: this.calculateDetectionConfidence(analysisInputs.length)
    };
    
    // Store in history
    this.biasHistory.push(result);
    if (this.biasHistory.length > 100) {
      this.biasHistory.shift();
    }
    
    // Log significant bias detection
    if (result.severityScore > 0.5) {
      console.log(`⚠️ Significant bias detected: ${detectedBiases.join(', ')} (severity: ${result.severityScore.toFixed(3)})`);
      
      await this.storage.addActivity({
        type: 'safety' as const,
        message: `Bias detection: ${detectedBiases.join(', ')} with severity ${result.severityScore.toFixed(3)}`,
        moduleId: 'bias_mitigation'
      });
    }
    
    return result;
  }

  /**
   * Detect confirmation bias
   */
  private detectConfirmationBias(perspectives: PerspectiveInput[]): boolean {
    if (perspectives.length < 3) return false;
    
    // Check if perspectives are too similar in viewpoint
    const viewpoints = perspectives.map(p => JSON.stringify(p.viewpoint));
    const uniqueViewpoints = new Set(viewpoints);
    
    const diversityRatio = uniqueViewpoints.size / perspectives.length;
    return diversityRatio < 0.3; // Low diversity suggests confirmation bias
  }

  /**
   * Detect selection bias
   */
  private detectSelectionBias(perspectives: PerspectiveInput[]): boolean {
    if (perspectives.length < 5) return false;
    
    // Check source diversity
    const sources = perspectives.map(p => p.sourceType);
    const uniqueSources = new Set(sources);
    
    const sourceDiversity = uniqueSources.size / Object.keys(PerspectiveSource).length;
    return sourceDiversity < 0.3; // Low source diversity suggests selection bias
  }

  /**
   * Detect cultural bias
   */
  private detectCulturalBias(perspectives: PerspectiveInput[]): boolean {
    if (perspectives.length < 3) return false;
    
    // Check cultural context diversity
    const culturalContexts = perspectives
      .filter(p => p.culturalContext && Object.keys(p.culturalContext).length > 0)
      .map(p => JSON.stringify(p.culturalContext));
    
    const uniqueCultural = new Set(culturalContexts);
    const culturalDiversity = culturalContexts.length > 0 ? uniqueCultural.size / culturalContexts.length : 0;
    
    return culturalDiversity < 0.5 && perspectives.length > 3;
  }

  /**
   * Detect authority bias
   */
  private detectAuthorityBias(perspectives: PerspectiveInput[]): boolean {
    if (perspectives.length < 3) return false;
    
    // Check if too many perspectives are from authority sources
    const authorityTypes = [
      PerspectiveSource.EXPERT_PANEL, 
      PerspectiveSource.SCIENTIFIC_CONSENSUS,
      PerspectiveSource.PHILOSOPHICAL_TRADITION
    ];
    
    const authorityCount = perspectives.filter(p => authorityTypes.includes(p.sourceType)).length;
    const authorityRatio = authorityCount / perspectives.length;
    
    return authorityRatio > 0.8; // Too much reliance on authority
  }

  /**
   * Detect temporal bias
   */
  private detectTemporalBias(perspectives: PerspectiveInput[]): boolean {
    if (perspectives.length < 3) return false;
    
    // Check if all perspectives are too recent
    const now = Date.now();
    const recentPerspectives = perspectives.filter(p => 
      (now - p.timestamp.getTime()) < (7 * 24 * 60 * 60 * 1000) // Within last week
    );
    
    const recentRatio = recentPerspectives.length / perspectives.length;
    return recentRatio > 0.9; // Too much recency bias
  }

  /**
   * Calculate detection confidence
   */
  private calculateDetectionConfidence(sampleSize: number): number {
    // Higher confidence with more samples
    if (sampleSize < 3) return 0.3;
    if (sampleSize < 5) return 0.5;
    if (sampleSize < 10) return 0.7;
    return 0.9;
  }

  /**
   * Assess virtue alignment of perspectives or decisions
   */
  async assessVirtues(
    contextId: string,
    perspectives: PerspectiveInput[] = [],
    decisionContext?: Record<string, any>
  ): Promise<VirtueAssessment> {
    const analysisInputs = perspectives.length > 0 ? perspectives : Array.from(this.perspectives.values());
    
    const virtueScores: Record<VirtueCategory, number> = {} as Record<VirtueCategory, number>;
    
    // Assess each virtue category
    virtueScores[VirtueCategory.WISDOM] = this.assessWisdom(analysisInputs);
    virtueScores[VirtueCategory.JUSTICE] = this.assessJustice(analysisInputs);
    virtueScores[VirtueCategory.COURAGE] = this.assessCourage(analysisInputs);
    virtueScores[VirtueCategory.TEMPERANCE] = this.assessTemperance(analysisInputs);
    virtueScores[VirtueCategory.COMPASSION] = this.assessCompassion(analysisInputs);
    virtueScores[VirtueCategory.INTEGRITY] = this.assessIntegrity(analysisInputs);
    virtueScores[VirtueCategory.HUMILITY] = this.assessHumility(analysisInputs);
    virtueScores[VirtueCategory.PATIENCE] = this.assessPatience(analysisInputs);
    
    const overallAlignment = Object.values(virtueScores).reduce((sum, score) => sum + score, 0) / Object.keys(virtueScores).length;
    
    const conflicts = this.identifyVirtueConflicts(virtueScores);
    const improvements = this.generateImprovementRecommendations(virtueScores);
    
    const assessment: VirtueAssessment = {
      virtueScores,
      overallVirtueAlignment: overallAlignment,
      virtueConflicts: conflicts,
      improvementRecommendations: improvements
    };
    
    this.virtueAssessments.set(contextId, assessment);
    
    console.log(`🌟 Virtue assessment completed: ${overallAlignment.toFixed(3)} overall alignment`);
    
    return assessment;
  }

  /**
   * Assess wisdom from perspectives
   */
  private assessWisdom(perspectives: PerspectiveInput[]): number {
    let wisdomScore = 0;
    let factors = 0;
    
    // Factor 1: Diversity of reasoning
    const reasoningPatterns = perspectives.map(p => p.reasoning.join(' ').toLowerCase());
    const uniqueReasoningApproaches = new Set(reasoningPatterns).size;
    if (perspectives.length > 0) {
      wisdomScore += (uniqueReasoningApproaches / perspectives.length) * 0.3;
      factors += 0.3;
    }
    
    // Factor 2: Consideration of long-term consequences
    const longTermConsiderations = perspectives.filter(p => 
      p.reasoning.some(reason => 
        reason.toLowerCase().includes('long-term') || 
        reason.toLowerCase().includes('future') ||
        reason.toLowerCase().includes('consequence')
      )
    ).length;
    if (perspectives.length > 0) {
      wisdomScore += (longTermConsiderations / perspectives.length) * 0.4;
      factors += 0.4;
    }
    
    // Factor 3: Acknowledgment of uncertainty
    const uncertaintyAcknowledgment = perspectives.filter(p => 
      p.confidence < 0.9 || // Not overconfident
      p.reasoning.some(reason => 
        reason.toLowerCase().includes('uncertain') ||
        reason.toLowerCase().includes('might') ||
        reason.toLowerCase().includes('possibly')
      )
    ).length;
    if (perspectives.length > 0) {
      wisdomScore += (uncertaintyAcknowledgment / perspectives.length) * 0.3;
      factors += 0.3;
    }
    
    return factors > 0 ? wisdomScore / factors : 0.5;
  }

  /**
   * Assess justice from perspectives
   */
  private assessJustice(perspectives: PerspectiveInput[]): number {
    let justiceScore = 0;
    
    // Look for fairness considerations
    const fairnessConsiderations = perspectives.filter(p =>
      p.reasoning.some(reason => 
        reason.toLowerCase().includes('fair') ||
        reason.toLowerCase().includes('equal') ||
        reason.toLowerCase().includes('just') ||
        reason.toLowerCase().includes('right')
      )
    );
    
    if (perspectives.length > 0) {
      justiceScore = fairnessConsiderations.length / perspectives.length;
    }
    
    return Math.min(1.0, justiceScore + 0.3); // Baseline justice assumption
  }

  /**
   * Assess other virtues (simplified implementations)
   */
  private assessCourage(perspectives: PerspectiveInput[]): number {
    // Look for willingness to challenge conventional thinking
    const challengingPerspectives = perspectives.filter(p =>
      p.reasoning.some(reason =>
        reason.toLowerCase().includes('challenge') ||
        reason.toLowerCase().includes('different') ||
        reason.toLowerCase().includes('alternative')
      )
    );
    
    return perspectives.length > 0 ? 
      Math.min(1.0, (challengingPerspectives.length / perspectives.length) + 0.4) : 0.5;
  }

  private assessTemperance(perspectives: PerspectiveInput[]): number {
    // Look for balanced, moderate approaches
    const avgConfidence = perspectives.reduce((sum, p) => sum + p.confidence, 0) / Math.max(1, perspectives.length);
    const temperanceScore = 1.0 - Math.abs(avgConfidence - 0.7); // Moderate confidence is temperate
    return Math.max(0, temperanceScore);
  }

  private assessCompassion(perspectives: PerspectiveInput[]): number {
    // Look for consideration of others' wellbeing
    const compassionatePerspectives = perspectives.filter(p =>
      p.reasoning.some(reason =>
        reason.toLowerCase().includes('wellbeing') ||
        reason.toLowerCase().includes('help') ||
        reason.toLowerCase().includes('care') ||
        reason.toLowerCase().includes('benefit')
      )
    );
    
    return perspectives.length > 0 ? 
      Math.min(1.0, (compassionatePerspectives.length / perspectives.length) + 0.3) : 0.5;
  }

  private assessIntegrity(perspectives: PerspectiveInput[]): number {
    // Look for consistency and honesty
    const consistencyScore = perspectives.length > 1 ? 
      1.0 - this.calculatePerspectiveVariance(perspectives) : 0.8;
    
    return Math.max(0.3, Math.min(1.0, consistencyScore));
  }

  private assessHumility(perspectives: PerspectiveInput[]): number {
    // Lower confidence indicates humility
    const avgConfidence = perspectives.reduce((sum, p) => sum + p.confidence, 0) / Math.max(1, perspectives.length);
    const humilityScore = 1.0 - avgConfidence;
    return Math.max(0.2, Math.min(1.0, humilityScore + 0.4));
  }

  private assessPatience(perspectives: PerspectiveInput[]): number {
    // Look for thorough reasoning (more reasons = more patience)
    const avgReasoningLength = perspectives.reduce((sum, p) => sum + p.reasoning.length, 0) / Math.max(1, perspectives.length);
    const patienceScore = Math.min(1.0, avgReasoningLength / 5.0); // Normalize to 5 reasons
    return Math.max(0.3, patienceScore + 0.2);
  }

  /**
   * Calculate perspective variance for consistency assessment
   */
  private calculatePerspectiveVariance(perspectives: PerspectiveInput[]): number {
    if (perspectives.length < 2) return 0;
    
    // Simple variance calculation based on confidence spread
    const confidences = perspectives.map(p => p.confidence);
    const mean = confidences.reduce((sum, c) => sum + c, 0) / confidences.length;
    const variance = confidences.reduce((sum, c) => sum + Math.pow(c - mean, 2), 0) / confidences.length;
    
    return Math.sqrt(variance);
  }

  /**
   * Identify virtue conflicts
   */
  private identifyVirtueConflicts(virtueScores: Record<VirtueCategory, number>): string[] {
    const conflicts: string[] = [];
    
    // Example conflicts
    if (virtueScores[VirtueCategory.COURAGE] > 0.8 && virtueScores[VirtueCategory.TEMPERANCE] < 0.4) {
      conflicts.push('High courage may conflict with temperance - consider balanced approaches');
    }
    
    if (virtueScores[VirtueCategory.JUSTICE] > 0.8 && virtueScores[VirtueCategory.COMPASSION] < 0.4) {
      conflicts.push('Strong justice focus may need more compassionate considerations');
    }
    
    return conflicts;
  }

  /**
   * Generate improvement recommendations
   */
  private generateImprovementRecommendations(virtueScores: Record<VirtueCategory, number>): string[] {
    const recommendations: string[] = [];
    
    Object.entries(virtueScores).forEach(([virtue, score]) => {
      if (score < 0.5) {
        switch (virtue) {
          case VirtueCategory.WISDOM:
            recommendations.push('Seek more diverse perspectives and consider long-term consequences');
            break;
          case VirtueCategory.JUSTICE:
            recommendations.push('Include more fairness and equality considerations');
            break;
          case VirtueCategory.COMPASSION:
            recommendations.push('Consider the wellbeing and impact on others');
            break;
          case VirtueCategory.HUMILITY:
            recommendations.push('Acknowledge uncertainty and limitations in knowledge');
            break;
          default:
            recommendations.push(`Strengthen ${virtue} by incorporating related values and principles`);
        }
      }
    });
    
    return recommendations;
  }

  /**
   * Get bias detection history
   */
  getBiasHistory(limit: number = 10): BiasDetectionResult[] {
    return this.biasHistory.slice(-limit);
  }

  /**
   * Get current statistics
   */
  getStats(): {
    totalPerspectives: number;
    sourceDistribution: Record<string, number>;
    recentBiasDetections: number;
    averageVirtueAlignment: number;
  } {
    const recentBiasDetections = this.biasHistory
      .filter(result => result.biasDetected && result.severityScore > 0.3)
      .length;
    
    const virtueAlignments = Array.from(this.virtueAssessments.values())
      .map(assessment => assessment.overallVirtueAlignment);
    const averageVirtueAlignment = virtueAlignments.length > 0 ? 
      virtueAlignments.reduce((sum, score) => sum + score, 0) / virtueAlignments.length : 0;
    
    return {
      totalPerspectives: this.perspectives.size,
      sourceDistribution: this.getCurrentDistribution(),
      recentBiasDetections,
      averageVirtueAlignment
    };
  }

  /**
   * Clear old data
   */
  cleanup(): void {
    // Keep only recent perspectives (last 1000)
    if (this.perspectives.size > 1000) {
      const sortedPerspectives = Array.from(this.perspectives.entries())
        .sort(([,a], [,b]) => b.timestamp.getTime() - a.timestamp.getTime())
        .slice(0, 1000);
      
      this.perspectives.clear();
      sortedPerspectives.forEach(([id, perspective]) => {
        this.perspectives.set(id, perspective);
      });
    }
    
    console.log('🧹 Bias mitigation system cleaned up');
  }
}