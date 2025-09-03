import { IStorage } from '../storage';

// Autonomous learning types
export enum LearningMode {
  EXPLORATION = "exploration",
  EXPLOITATION = "exploitation", 
  CURIOSITY_DRIVEN = "curiosity_driven",
  GOAL_ORIENTED = "goal_oriented",
  PATTERN_DISCOVERY = "pattern_discovery",
  TRANSFER_LEARNING = "transfer_learning",
  META_LEARNING = "meta_learning",
  SELF_SUPERVISED = "self_supervised"
}

export enum KnowledgeType {
  FACTUAL = "factual",
  PROCEDURAL = "procedural",
  CONCEPTUAL = "conceptual", 
  METACOGNITIVE = "metacognitive",
  CAUSAL = "causal",
  RELATIONAL = "relational",
  TEMPORAL = "temporal",
  SOCIAL = "social"
}

export enum LearningStrategy {
  ACTIVE_LEARNING = "active_learning",
  REINFORCEMENT_LEARNING = "reinforcement_learning",
  UNSUPERVISED_DISCOVERY = "unsupervised_discovery",
  ANALOGICAL_REASONING = "analogical_reasoning",
  INDUCTIVE_LEARNING = "inductive_learning",
  DEDUCTIVE_REASONING = "deductive_reasoning",
  ABDUCTIVE_INFERENCE = "abductive_inference",
  CONTINUAL_LEARNING = "continual_learning"
}

export interface LearningOpportunity {
  opportunityId: string;
  domain: string;
  knowledgeType: KnowledgeType;
  learningStrategy: LearningStrategy;
  potentialValue: number;
  complexityLevel: number;
  resourceRequirements: Record<string, number>;
  expectedLearningTime: number; // in milliseconds
  prerequisites: string[];
  successCriteria: string[];
}

export interface LearningExperience {
  experienceId: string;
  opportunityId: string;
  learningMode: LearningMode;
  knowledgeAcquired: Record<string, any>;
  performanceImprovement: number;
  learningEfficiency: number;
  unexpectedDiscoveries: string[];
  transferableInsights: string[];
  timestamp: Date;
}

export interface CuriosityDrive {
  curiosityId: string;
  description: string;
  strength: number;
  satisfactionLevel: number;
  recentExperiences: string[]; // Experience IDs
  temporalPattern: number[];
  triggers: string[];
}

export interface IntrinsicGoal {
  goalId: string;
  description: string;
  motivatingDrives: string[]; // Curiosity drive IDs
  targetState: Record<string, any>;
  currentProgress: number;
  priority: number;
  estimatedEffort: number;
  expectedSatisfaction: number;
}

/**
 * Autonomous Learning Engine for Self-Sustaining Intelligence
 * Implements continuous autonomous learning without human supervision
 */
export class AutonomousLearningEngine {
  private storage: IStorage;
  private learningOpportunities: Map<string, LearningOpportunity> = new Map();
  private learningExperiences: LearningExperience[] = [];
  private curiosityDrives: Map<string, CuriosityDrive> = new Map();
  private intrinsicGoals: Map<string, IntrinsicGoal> = new Map();
  private metaLearningInsights: Map<string, any> = new Map();
  
  // Learning state
  private currentLearningMode: LearningMode = LearningMode.EXPLORATION;
  private learningActive: boolean = false;
  private learningInterval: NodeJS.Timeout | null = null;

  constructor(storage: IStorage) {
    this.storage = storage;
    this.initializeCuriosityDrives();
    console.log('🧠 Autonomous Learning Engine initialized');
  }

  /**
   * Initialize fundamental curiosity drives
   */
  private initializeCuriosityDrives(): void {
    const basicDrives = [
      {
        curiosityId: 'pattern_discovery',
        description: 'Drive to discover patterns and regularities in data',
        strength: 0.8,
        triggers: ['new_data', 'anomalies', 'unexpected_relationships']
      },
      {
        curiosityId: 'knowledge_gaps',
        description: 'Drive to fill gaps in understanding',
        strength: 0.9,
        triggers: ['unanswered_questions', 'contradictory_information', 'missing_context']
      },
      {
        curiosityId: 'novelty_seeking',
        description: 'Drive to explore new domains and concepts',
        strength: 0.7,
        triggers: ['unfamiliar_concepts', 'new_domains', 'creative_opportunities']
      },
      {
        curiosityId: 'mastery_motivation',
        description: 'Drive to achieve competence and mastery',
        strength: 0.75,
        triggers: ['skill_challenges', 'performance_gaps', 'improvement_opportunities']
      },
      {
        curiosityId: 'understanding_depth',
        description: 'Drive to develop deeper understanding of known concepts',
        strength: 0.65,
        triggers: ['surface_level_knowledge', 'conceptual_connections', 'first_principles']
      }
    ];

    basicDrives.forEach(drive => {
      this.curiosityDrives.set(drive.curiosityId, {
        ...drive,
        satisfactionLevel: 0.5,
        recentExperiences: [],
        temporalPattern: [0.5]
      });
    });
  }

  /**
   * Start autonomous learning process
   */
  startLearning(intervalMs: number = 30000): void {
    if (this.learningActive) {
      console.log('Autonomous learning already active');
      return;
    }

    this.learningActive = true;
    console.log('🚀 Starting autonomous learning process...');

    this.learningInterval = setInterval(() => {
      this.performLearningCycle();
    }, intervalMs);
  }

  /**
   * Stop autonomous learning
   */
  stopLearning(): void {
    if (this.learningInterval) {
      clearInterval(this.learningInterval);
      this.learningInterval = null;
    }
    this.learningActive = false;
    console.log('⏸️ Autonomous learning stopped');
  }

  /**
   * Perform a learning cycle
   */
  private async performLearningCycle(): Promise<void> {
    try {
      // 1. Assess current knowledge state
      await this.assessKnowledgeState();
      
      // 2. Update curiosity drives
      this.updateCuriosityDrives();
      
      // 3. Identify learning opportunities
      await this.identifyLearningOpportunities();
      
      // 4. Select and pursue learning opportunity
      const opportunity = this.selectBestOpportunity();
      if (opportunity) {
        await this.pursueLearningOpportunity(opportunity);
      }
      
      // 5. Reflect and extract meta-learning insights
      await this.performMetaLearning();
      
      // 6. Generate or update intrinsic goals
      await this.updateIntrinsicGoals();
      
      console.log(`🔄 Learning cycle completed. Mode: ${this.currentLearningMode}`);
      
    } catch (error) {
      console.error('Error in learning cycle:', error);
    }
  }

  /**
   * Assess current knowledge state
   */
  private async assessKnowledgeState(): Promise<void> {
    // For now, simulate knowledge assessment
    // In a real implementation, this would analyze the knowledge graph,
    // evaluate model performance, identify gaps, etc.
    
    const knowledgeDomains = [
      'general_knowledge', 'reasoning', 'creativity', 'social_understanding',
      'problem_solving', 'language', 'mathematics', 'science'
    ];
    
    const assessments = knowledgeDomains.map(domain => ({
      domain,
      competence: Math.random() * 0.4 + 0.3, // 0.3-0.7 range
      gaps: Math.floor(Math.random() * 5) + 1,
      recentGrowth: Math.random() * 0.2 - 0.1 // -0.1 to +0.1
    }));

    // Store assessment for opportunity identification
    this.metaLearningInsights.set('knowledge_assessment', {
      timestamp: new Date(),
      domains: assessments,
      overallCompetence: assessments.reduce((sum, a) => sum + a.competence, 0) / assessments.length
    });
  }

  /**
   * Update curiosity drives based on recent experiences
   */
  private updateCuriosityDrives(): void {
    for (const [driveId, drive] of this.curiosityDrives) {
      // Calculate satisfaction decay
      const timeSinceLastSatisfaction = this.calculateTimeSinceLastSatisfaction(drive);
      const decay = Math.min(0.1, timeSinceLastSatisfaction / (7 * 24 * 60 * 60 * 1000)); // Weekly decay
      
      drive.satisfactionLevel = Math.max(0, drive.satisfactionLevel - decay);
      
      // Update strength based on satisfaction patterns
      if (drive.satisfactionLevel < 0.3) {
        drive.strength = Math.min(1.0, drive.strength + 0.05); // Increase strength when unsatisfied
      } else if (drive.satisfactionLevel > 0.8) {
        drive.strength = Math.max(0.1, drive.strength - 0.02); // Decrease when satisfied
      }
      
      // Update temporal pattern
      drive.temporalPattern.push(drive.satisfactionLevel);
      if (drive.temporalPattern.length > 50) {
        drive.temporalPattern.shift();
      }
    }
  }

  /**
   * Calculate time since last satisfaction for a curiosity drive
   */
  private calculateTimeSinceLastSatisfaction(drive: CuriosityDrive): number {
    if (drive.recentExperiences.length === 0) return 7 * 24 * 60 * 60 * 1000; // 1 week default
    
    const lastExperienceId = drive.recentExperiences[drive.recentExperiences.length - 1];
    const lastExperience = this.learningExperiences.find(exp => exp.experienceId === lastExperienceId);
    
    return lastExperience ? Date.now() - lastExperience.timestamp.getTime() : 7 * 24 * 60 * 60 * 1000;
  }

  /**
   * Identify potential learning opportunities
   */
  private async identifyLearningOpportunities(): Promise<void> {
    const knowledgeAssessment = this.metaLearningInsights.get('knowledge_assessment');
    if (!knowledgeAssessment) return;

    // Clear old opportunities (keep only recent ones)
    const recentOpportunities = Array.from(this.learningOpportunities.entries())
      .filter(([, opp]) => Date.now() - Date.parse(opp.opportunityId.split('_')[1]) < 24 * 60 * 60 * 1000);
    this.learningOpportunities.clear();
    recentOpportunities.forEach(([id, opp]) => this.learningOpportunities.set(id, opp));

    // Identify opportunities based on knowledge gaps and curiosity drives
    for (const domainAssessment of knowledgeAssessment.domains) {
      if (domainAssessment.gaps > 2 || domainAssessment.competence < 0.5) {
        const opportunity: LearningOpportunity = {
          opportunityId: `${domainAssessment.domain}_${Date.now()}`,
          domain: domainAssessment.domain,
          knowledgeType: this.selectKnowledgeType(domainAssessment),
          learningStrategy: this.selectLearningStrategy(domainAssessment),
          potentialValue: this.calculatePotentialValue(domainAssessment),
          complexityLevel: Math.random() * 3 + 2, // 2-5 complexity
          resourceRequirements: {
            time: Math.random() * 30 + 10, // 10-40 minutes
            cognitive_load: Math.random() * 0.5 + 0.3, // 0.3-0.8
            memory: Math.random() * 100 + 50 // 50-150 MB
          },
          expectedLearningTime: (Math.random() * 30 + 10) * 60 * 1000, // 10-40 minutes in ms
          prerequisites: this.generatePrerequisites(domainAssessment.domain),
          successCriteria: this.generateSuccessCriteria(domainAssessment.domain)
        };

        this.learningOpportunities.set(opportunity.opportunityId, opportunity);
      }
    }

    // Identify curiosity-driven opportunities
    for (const [driveId, drive] of this.curiosityDrives) {
      if (drive.satisfactionLevel < 0.4 && drive.strength > 0.6) {
        const opportunity: LearningOpportunity = {
          opportunityId: `curiosity_${driveId}_${Date.now()}`,
          domain: this.mapCuriosityToDomain(driveId),
          knowledgeType: KnowledgeType.CONCEPTUAL,
          learningStrategy: LearningStrategy.UNSUPERVISED_DISCOVERY,
          potentialValue: drive.strength * (1 - drive.satisfactionLevel),
          complexityLevel: Math.random() * 2 + 1, // 1-3 complexity for curiosity
          resourceRequirements: {
            time: Math.random() * 20 + 5, // 5-25 minutes
            cognitive_load: Math.random() * 0.4 + 0.2,
            memory: Math.random() * 75 + 25
          },
          expectedLearningTime: (Math.random() * 20 + 5) * 60 * 1000,
          prerequisites: [],
          successCriteria: [`Satisfy ${drive.description.toLowerCase()}`]
        };

        this.learningOpportunities.set(opportunity.opportunityId, opportunity);
      }
    }

    console.log(`🔍 Identified ${this.learningOpportunities.size} learning opportunities`);
  }

  /**
   * Select knowledge type based on domain assessment
   */
  private selectKnowledgeType(assessment: any): KnowledgeType {
    const types = Object.values(KnowledgeType);
    if (assessment.competence < 0.3) return KnowledgeType.FACTUAL;
    if (assessment.recentGrowth < 0) return KnowledgeType.PROCEDURAL;
    return types[Math.floor(Math.random() * types.length)];
  }

  /**
   * Select learning strategy based on assessment
   */
  private selectLearningStrategy(assessment: any): LearningStrategy {
    if (assessment.gaps > 3) return LearningStrategy.ACTIVE_LEARNING;
    if (assessment.competence > 0.6) return LearningStrategy.TRANSFER_LEARNING;
    
    const strategies = Object.values(LearningStrategy);
    return strategies[Math.floor(Math.random() * strategies.length)];
  }

  /**
   * Calculate potential value of learning opportunity
   */
  private calculatePotentialValue(assessment: any): number {
    const gapFactor = assessment.gaps / 5.0;
    const competenceFactor = 1.0 - assessment.competence;
    const growthFactor = assessment.recentGrowth > 0 ? 0.8 : 1.2;
    
    return Math.min(1.0, (gapFactor + competenceFactor) * growthFactor / 2);
  }

  /**
   * Generate prerequisites for a domain
   */
  private generatePrerequisites(domain: string): string[] {
    const domainPrereqs: Record<string, string[]> = {
      mathematics: ['basic_arithmetic', 'logical_reasoning'],
      science: ['mathematics', 'observation_skills'],
      reasoning: ['pattern_recognition', 'logical_thinking'],
      creativity: ['divergent_thinking', 'conceptual_flexibility']
    };
    
    return domainPrereqs[domain] || [];
  }

  /**
   * Generate success criteria for a domain
   */
  private generateSuccessCriteria(domain: string): string[] {
    return [
      `Demonstrate improved understanding in ${domain}`,
      'Apply learned concepts to novel situations',
      'Connect new knowledge to existing understanding',
      'Show measurable performance improvement'
    ];
  }

  /**
   * Map curiosity drive to learning domain
   */
  private mapCuriosityToDomain(driveId: string): string {
    const mapping: Record<string, string> = {
      pattern_discovery: 'pattern_recognition',
      knowledge_gaps: 'general_knowledge',
      novelty_seeking: 'exploration',
      mastery_motivation: 'skill_development',
      understanding_depth: 'conceptual_learning'
    };
    
    return mapping[driveId] || 'general_learning';
  }

  /**
   * Select the best learning opportunity to pursue
   */
  private selectBestOpportunity(): LearningOpportunity | null {
    if (this.learningOpportunities.size === 0) return null;

    const opportunities = Array.from(this.learningOpportunities.values());
    
    // Calculate priority scores for each opportunity
    const scoredOpportunities = opportunities.map(opp => ({
      opportunity: opp,
      score: this.calculateOpportunityScore(opp)
    }));

    // Sort by score and select the best
    scoredOpportunities.sort((a, b) => b.score - a.score);
    return scoredOpportunities[0].opportunity;
  }

  /**
   * Calculate priority score for learning opportunity
   */
  private calculateOpportunityScore(opportunity: LearningOpportunity): number {
    const valueFactor = opportunity.potentialValue;
    const complexityPenalty = 1.0 - (opportunity.complexityLevel / 10.0);
    const resourceEfficiency = 1.0 - (Object.values(opportunity.resourceRequirements).reduce((sum, req) => sum + req, 0) / 5.0);
    const prerequisiteReadiness = 1.0 - (opportunity.prerequisites.length / 10.0);
    
    return Math.max(0.0, 0.4 * valueFactor + 0.25 * complexityPenalty + 
                   0.2 * resourceEfficiency + 0.15 * prerequisiteReadiness);
  }

  /**
   * Pursue a learning opportunity
   */
  private async pursueLearningOpportunity(opportunity: LearningOpportunity): Promise<void> {
    console.log(`📚 Pursuing learning opportunity: ${opportunity.domain}`);
    
    // Simulate learning process
    const startTime = Date.now();
    
    // Simulate learning time (shortened for demo)
    const actualLearningTime = Math.min(5000, opportunity.expectedLearningTime / 100); // 5 seconds max
    await new Promise(resolve => setTimeout(resolve, actualLearningTime));
    
    // Simulate learning outcomes
    const performanceImprovement = Math.random() * 0.3 + 0.1; // 0.1-0.4
    const efficiency = Math.random() * 0.5 + 0.5; // 0.5-1.0
    
    const experience: LearningExperience = {
      experienceId: `exp_${Date.now()}`,
      opportunityId: opportunity.opportunityId,
      learningMode: this.currentLearningMode,
      knowledgeAcquired: {
        domain: opportunity.domain,
        concepts: Math.floor(Math.random() * 5) + 1,
        connections: Math.floor(Math.random() * 3) + 1
      },
      performanceImprovement,
      learningEfficiency: efficiency,
      unexpectedDiscoveries: this.generateDiscoveries(opportunity),
      transferableInsights: this.generateInsights(opportunity),
      timestamp: new Date()
    };
    
    this.learningExperiences.push(experience);
    
    // Update curiosity drives satisfaction
    this.updateCuriositySatisfaction(opportunity, experience);
    
    // Remove completed opportunity
    this.learningOpportunities.delete(opportunity.opportunityId);
    
    // Log to storage
    await this.storage.addActivity({
      type: 'learning' as const,
      message: `Autonomous learning: ${opportunity.domain} (improvement: ${performanceImprovement.toFixed(3)})`,
      moduleId: 'autonomous_learning'
    });
    
    console.log(`✅ Learning experience completed: ${experience.performanceImprovement.toFixed(3)} improvement`);
  }

  /**
   * Generate unexpected discoveries
   */
  private generateDiscoveries(opportunity: LearningOpportunity): string[] {
    const discoveryPool = [
      'Novel pattern in data structure',
      'Unexpected connection to other domains',
      'Counterintuitive relationship discovered',
      'Emergent property identified',
      'Optimization opportunity found'
    ];
    
    const numDiscoveries = Math.floor(Math.random() * 3);
    return discoveryPool.slice(0, numDiscoveries);
  }

  /**
   * Generate transferable insights
   */
  private generateInsights(opportunity: LearningOpportunity): string[] {
    const insightPool = [
      'Method applicable to similar problems',
      'Principle generalizes across domains',
      'Strategy useful for future learning',
      'Pattern recognition technique',
      'Efficiency improvement method'
    ];
    
    const numInsights = Math.floor(Math.random() * 4) + 1;
    return insightPool.slice(0, numInsights);
  }

  /**
   * Update curiosity drive satisfaction
   */
  private updateCuriositySatisfaction(opportunity: LearningOpportunity, experience: LearningExperience): void {
    // Find related curiosity drives
    for (const [driveId, drive] of this.curiosityDrives) {
      const relevance = this.calculateRelevanceToOpportunity(drive, opportunity);
      
      if (relevance > 0.3) {
        const satisfaction = Math.min(1.0, relevance * experience.performanceImprovement * 2);
        drive.satisfactionLevel = Math.min(1.0, drive.satisfactionLevel + satisfaction);
        drive.recentExperiences.push(experience.experienceId);
        
        // Keep only recent experiences
        if (drive.recentExperiences.length > 10) {
          drive.recentExperiences.shift();
        }
      }
    }
  }

  /**
   * Calculate relevance of opportunity to curiosity drive
   */
  private calculateRelevanceToOpportunity(drive: CuriosityDrive, opportunity: LearningOpportunity): number {
    // Simple mapping of drives to opportunity characteristics
    const relevanceMap: Record<string, string[]> = {
      pattern_discovery: ['pattern_recognition', 'data_analysis', 'reasoning'],
      knowledge_gaps: ['general_knowledge', 'factual', 'conceptual'],
      novelty_seeking: ['exploration', 'creativity', 'novel'],
      mastery_motivation: ['skill_development', 'procedural', 'performance'],
      understanding_depth: ['conceptual', 'deep_learning', 'first_principles']
    };
    
    const driveKeywords = relevanceMap[drive.curiosityId] || [];
    const opportunityText = `${opportunity.domain} ${opportunity.knowledgeType} ${opportunity.learningStrategy}`.toLowerCase();
    
    let relevance = 0;
    driveKeywords.forEach(keyword => {
      if (opportunityText.includes(keyword.toLowerCase())) {
        relevance += 0.3;
      }
    });
    
    return Math.min(1.0, relevance);
  }

  /**
   * Perform meta-learning analysis
   */
  private async performMetaLearning(): Promise<void> {
    if (this.learningExperiences.length < 3) return;
    
    const recentExperiences = this.learningExperiences.slice(-10);
    
    // Analyze learning patterns
    const patterns = {
      averageEfficiency: recentExperiences.reduce((sum, exp) => sum + exp.learningEfficiency, 0) / recentExperiences.length,
      averageImprovement: recentExperiences.reduce((sum, exp) => sum + exp.performanceImprovement, 0) / recentExperiences.length,
      bestLearningModes: this.identifyBestLearningModes(recentExperiences),
      mostEffectiveStrategies: this.identifyEffectiveStrategies(recentExperiences),
      transferLearningRate: this.calculateTransferLearningRate(recentExperiences)
    };
    
    this.metaLearningInsights.set('learning_patterns', {
      timestamp: new Date(),
      patterns,
      recommendedAdjustments: this.generateLearningAdjustments(patterns)
    });
    
    // Apply meta-learning insights
    this.applyMetaLearningInsights(patterns);
  }

  /**
   * Identify most effective learning modes
   */
  private identifyBestLearningModes(experiences: LearningExperience[]): LearningMode[] {
    const modePerformance: Record<LearningMode, number[]> = {} as Record<LearningMode, number[]>;
    
    experiences.forEach(exp => {
      if (!modePerformance[exp.learningMode]) {
        modePerformance[exp.learningMode] = [];
      }
      modePerformance[exp.learningMode].push(exp.performanceImprovement);
    });
    
    const modeAverages = Object.entries(modePerformance).map(([mode, improvements]) => ({
      mode: mode as LearningMode,
      average: improvements.reduce((sum, imp) => sum + imp, 0) / improvements.length
    }));
    
    modeAverages.sort((a, b) => b.average - a.average);
    return modeAverages.slice(0, 3).map(m => m.mode);
  }

  /**
   * Identify most effective learning strategies
   */
  private identifyEffectiveStrategies(experiences: LearningExperience[]): LearningStrategy[] {
    // Similar to identifyBestLearningModes but for strategies
    // For brevity, returning a sample
    return [LearningStrategy.ACTIVE_LEARNING, LearningStrategy.TRANSFER_LEARNING];
  }

  /**
   * Calculate transfer learning rate
   */
  private calculateTransferLearningRate(experiences: LearningExperience[]): number {
    const transferExperiences = experiences.filter(exp => exp.transferableInsights.length > 0);
    return transferExperiences.length / experiences.length;
  }

  /**
   * Generate learning adjustments based on patterns
   */
  private generateLearningAdjustments(patterns: any): string[] {
    const adjustments: string[] = [];
    
    if (patterns.averageEfficiency < 0.6) {
      adjustments.push('Increase focus on meta-learning strategies');
    }
    
    if (patterns.transferLearningRate < 0.3) {
      adjustments.push('Emphasize cross-domain knowledge transfer');
    }
    
    if (patterns.averageImprovement < 0.2) {
      adjustments.push('Seek more challenging learning opportunities');
    }
    
    return adjustments;
  }

  /**
   * Apply meta-learning insights to improve future learning
   */
  private applyMetaLearningInsights(patterns: any): void {
    if (patterns.bestLearningModes.length > 0) {
      this.currentLearningMode = patterns.bestLearningModes[0];
    }
    
    // Additional adjustments could be applied here
    console.log(`🔄 Applied meta-learning insights. New mode: ${this.currentLearningMode}`);
  }

  /**
   * Update intrinsic goals based on learning progress
   */
  private async updateIntrinsicGoals(): Promise<void> {
    // Generate new goals based on unsatisfied curiosity drives
    for (const [driveId, drive] of this.curiosityDrives) {
      if (drive.satisfactionLevel < 0.4 && drive.strength > 0.7) {
        const goalId = `goal_${driveId}_${Date.now()}`;
        
        const goal: IntrinsicGoal = {
          goalId,
          description: `Address ${drive.description}`,
          motivatingDrives: [driveId],
          targetState: {
            satisfaction_level: 0.8,
            knowledge_gained: true,
            competence_improved: true
          },
          currentProgress: 0,
          priority: drive.strength * (1 - drive.satisfactionLevel),
          estimatedEffort: Math.random() * 5 + 2, // 2-7 effort units
          expectedSatisfaction: drive.strength * 0.8
        };
        
        this.intrinsicGoals.set(goalId, goal);
      }
    }
    
    // Update progress on existing goals
    for (const [goalId, goal] of this.intrinsicGoals) {
      const relevantExperiences = this.learningExperiences.filter(exp =>
        goal.motivatingDrives.some(driveId => 
          this.curiosityDrives.get(driveId)?.recentExperiences.includes(exp.experienceId)
        )
      );
      
      if (relevantExperiences.length > 0) {
        const progressIncrease = relevantExperiences.reduce((sum, exp) => 
          sum + exp.performanceImprovement, 0) / relevantExperiences.length;
        goal.currentProgress = Math.min(1.0, goal.currentProgress + progressIncrease);
      }
      
      // Remove completed goals
      if (goal.currentProgress >= 0.9) {
        this.intrinsicGoals.delete(goalId);
        console.log(`🎯 Intrinsic goal completed: ${goal.description}`);
      }
    }
  }

  /**
   * Get current learning state
   */
  getLearningState(): {
    mode: LearningMode;
    active: boolean;
    opportunities: number;
    experiences: number;
    goals: number;
    topCuriosityDrives: Array<{ id: string; strength: number; satisfaction: number }>;
  } {
    const drives = Array.from(this.curiosityDrives.entries())
      .map(([id, drive]) => ({
        id,
        strength: drive.strength,
        satisfaction: drive.satisfactionLevel
      }))
      .sort((a, b) => (b.strength * (1 - b.satisfaction)) - (a.strength * (1 - a.satisfaction)))
      .slice(0, 3);

    return {
      mode: this.currentLearningMode,
      active: this.learningActive,
      opportunities: this.learningOpportunities.size,
      experiences: this.learningExperiences.length,
      goals: this.intrinsicGoals.size,
      topCuriosityDrives: drives
    };
  }

  /**
   * Get recent learning experiences
   */
  getRecentExperiences(limit: number = 5): LearningExperience[] {
    return this.learningExperiences.slice(-limit);
  }

  /**
   * Get learning statistics
   */
  getStats(): {
    totalExperiences: number;
    averageEfficiency: number;
    averageImprovement: number;
    activeDrives: number;
    satisfiedDrives: number;
    activeGoals: number;
  } {
    const avgEfficiency = this.learningExperiences.length > 0 ?
      this.learningExperiences.reduce((sum, exp) => sum + exp.learningEfficiency, 0) / this.learningExperiences.length : 0;
    
    const avgImprovement = this.learningExperiences.length > 0 ?
      this.learningExperiences.reduce((sum, exp) => sum + exp.performanceImprovement, 0) / this.learningExperiences.length : 0;
    
    const activeDrives = Array.from(this.curiosityDrives.values()).filter(d => d.strength > 0.5).length;
    const satisfiedDrives = Array.from(this.curiosityDrives.values()).filter(d => d.satisfactionLevel > 0.7).length;
    
    return {
      totalExperiences: this.learningExperiences.length,
      averageEfficiency: avgEfficiency,
      averageImprovement: avgImprovement,
      activeDrives,
      satisfiedDrives,
      activeGoals: this.intrinsicGoals.size
    };
  }

  /**
   * Cleanup old data
   */
  cleanup(): void {
    // Keep only recent experiences (last 100)
    if (this.learningExperiences.length > 100) {
      this.learningExperiences = this.learningExperiences.slice(-100);
    }
    
    console.log('🧹 Autonomous learning engine cleaned up');
  }

  /**
   * Shutdown the learning engine
   */
  shutdown(): void {
    this.stopLearning();
    console.log('🛑 Autonomous learning engine shut down');
  }
}