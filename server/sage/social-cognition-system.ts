import { IStorage } from '../storage';

// Social cognition types
export enum AgentType {
  HUMAN = "human",
  AI_AGENT = "ai_agent", 
  GROUP = "group",
  ORGANIZATION = "organization",
  VIRTUAL_ENTITY = "virtual_entity",
  UNKNOWN = "unknown"
}

export enum MentalState {
  BELIEF = "belief",
  DESIRE = "desire",
  INTENTION = "intention",
  EMOTION = "emotion",
  KNOWLEDGE = "knowledge",
  EXPECTATION = "expectation",
  GOAL = "goal",
  PREFERENCE = "preference"
}

export enum InteractionType {
  COOPERATION = "cooperation",
  COMPETITION = "competition",
  COMMUNICATION = "communication",
  NEGOTIATION = "negotiation", 
  TEACHING = "teaching",
  LEARNING = "learning",
  COLLABORATION = "collaboration",
  CONFLICT = "conflict"
}

export interface Agent {
  agentId: string;
  agentType: AgentType;
  name: string;
  capabilities: string[];
  observedBehaviors: Array<Record<string, any>>;
  mentalModel: Record<MentalState, any[]>;
  interactionHistory: string[]; // Interaction IDs
  trustLevel: number;
  predictability: number;
  lastInteraction?: Date;
}

export interface SocialInteraction {
  interactionId: string;
  participants: string[]; // Agent IDs
  interactionType: InteractionType;
  context: Record<string, any>;
  communicationContent: Array<Record<string, any>>;
  outcomes: Record<string, any>;
  mentalStateChanges: Record<string, Record<MentalState, any>>;
  socialDynamics: Record<string, number>;
  timestamp: Date;
}

export interface TheoryOfMindModel {
  agentId: string;
  beliefs: Array<{ belief: string; confidence: number; evidence: string[] }>;
  desires: Array<{ desire: string; strength: number; indicators: string[] }>;
  intentions: Array<{ intention: string; likelihood: number; timeframe: string }>;
  emotions: Array<{ emotion: string; intensity: number; triggers: string[] }>;
  predictions: Array<{ prediction: string; confidence: number; timeframe: string }>;
}

export interface SocialContext {
  contextId: string;
  participants: string[];
  relationships: Record<string, Record<string, any>>; // agent_id -> agent_id -> relationship
  groupDynamics: Record<string, number>;
  culturalFactors: Record<string, any>;
  situationalFactors: Record<string, any>;
  timestamp: Date;
}

/**
 * Social Cognition System for Theory of Mind and Multi-Agent Interaction
 * Builds understanding of other agents' mental states and social dynamics
 */
export class SocialCognitionSystem {
  private storage: IStorage;
  private agents: Map<string, Agent> = new Map();
  private interactions: SocialInteraction[] = [];
  private theoryOfMindModels: Map<string, TheoryOfMindModel> = new Map();
  private socialContexts: Map<string, SocialContext> = new Map();
  private relationshipModels: Map<string, Record<string, any>> = new Map();

  constructor(storage: IStorage) {
    this.storage = storage;
    console.log('🤝 Social Cognition System initialized');
  }

  /**
   * Register a new agent in the social environment
   */
  async registerAgent(
    agentId: string,
    agentType: AgentType,
    name: string,
    capabilities: string[] = []
  ): Promise<void> {
    const agent: Agent = {
      agentId,
      agentType,
      name,
      capabilities,
      observedBehaviors: [],
      mentalModel: {} as Record<MentalState, any[]>,
      interactionHistory: [],
      trustLevel: 0.5, // Neutral starting trust
      predictability: 0.5, // Neutral starting predictability
      lastInteraction: new Date()
    };

    // Initialize mental model structure
    Object.values(MentalState).forEach(state => {
      agent.mentalModel[state] = [];
    });

    this.agents.set(agentId, agent);
    
    // Initialize Theory of Mind model
    this.initializeTheoryOfMind(agentId);

    console.log(`👤 Registered agent: ${name} (${agentType})`);
    
    await this.storage.addActivity({
      type: 'social' as const,
      message: `New agent registered: ${name}`,
      moduleId: 'social_cognition'
    });
  }

  /**
   * Initialize Theory of Mind model for an agent
   */
  private initializeTheoryOfMind(agentId: string): void {
    const tomModel: TheoryOfMindModel = {
      agentId,
      beliefs: [],
      desires: [],
      intentions: [],
      emotions: [],
      predictions: []
    };

    this.theoryOfMindModels.set(agentId, tomModel);
  }

  /**
   * Record a social interaction
   */
  async recordInteraction(
    participants: string[],
    interactionType: InteractionType,
    context: Record<string, any> = {},
    communicationContent: Array<Record<string, any>> = []
  ): Promise<string> {
    const interactionId = `interaction_${Date.now()}`;
    
    const interaction: SocialInteraction = {
      interactionId,
      participants,
      interactionType,
      context,
      communicationContent,
      outcomes: {},
      mentalStateChanges: {},
      socialDynamics: {},
      timestamp: new Date()
    };

    this.interactions.push(interaction);

    // Update agents' interaction histories
    participants.forEach(agentId => {
      const agent = this.agents.get(agentId);
      if (agent) {
        agent.interactionHistory.push(interactionId);
        agent.lastInteraction = new Date();
        
        // Keep only recent interactions (last 100)
        if (agent.interactionHistory.length > 100) {
          agent.interactionHistory.shift();
        }
      }
    });

    // Analyze interaction and update mental models
    await this.analyzeInteraction(interaction);
    
    // Update Theory of Mind models
    await this.updateTheoryOfMindModels(interaction);
    
    // Update social context
    await this.updateSocialContext(interaction);

    console.log(`🔄 Recorded ${interactionType} interaction with ${participants.length} participants`);
    
    return interactionId;
  }

  /**
   * Analyze social interaction for insights
   */
  private async analyzeInteraction(interaction: SocialInteraction): Promise<void> {
    const { participants, interactionType, communicationContent } = interaction;

    // Analyze cooperation vs competition
    const cooperationScore = this.assessCooperationLevel(communicationContent);
    const conflictLevel = this.assessConflictLevel(communicationContent);
    
    interaction.socialDynamics = {
      cooperation: cooperationScore,
      conflict: conflictLevel,
      communication_quality: this.assessCommunicationQuality(communicationContent),
      power_balance: this.assessPowerBalance(participants, communicationContent),
      trust_changes: this.assessTrustChanges(participants, communicationContent)
    };

    // Update agent relationships based on interaction
    await this.updateAgentRelationships(participants, interaction);
    
    // Record behavioral observations
    await this.recordBehavioralObservations(interaction);
  }

  /**
   * Assess cooperation level in interaction
   */
  private assessCooperationLevel(content: Array<Record<string, any>>): number {
    let cooperationScore = 0.5; // Neutral baseline
    
    const cooperativeKeywords = [
      'help', 'collaborate', 'together', 'share', 'support',
      'agree', 'coordinate', 'assist', 'cooperate', 'team'
    ];
    
    const totalContent = content.length;
    if (totalContent === 0) return cooperationScore;
    
    let cooperativeCount = 0;
    content.forEach(msg => {
      const text = JSON.stringify(msg).toLowerCase();
      cooperativeKeywords.forEach(keyword => {
        if (text.includes(keyword)) {
          cooperativeCount++;
        }
      });
    });
    
    const cooperationRatio = cooperativeCount / (totalContent * cooperativeKeywords.length);
    return Math.min(1.0, 0.5 + cooperationRatio * 2);
  }

  /**
   * Assess conflict level in interaction
   */
  private assessConflictLevel(content: Array<Record<string, any>>): number {
    const conflictKeywords = [
      'disagree', 'conflict', 'oppose', 'refuse', 'deny',
      'argue', 'dispute', 'reject', 'contradict', 'challenge'
    ];
    
    const totalContent = content.length;
    if (totalContent === 0) return 0;
    
    let conflictCount = 0;
    content.forEach(msg => {
      const text = JSON.stringify(msg).toLowerCase();
      conflictKeywords.forEach(keyword => {
        if (text.includes(keyword)) {
          conflictCount++;
        }
      });
    });
    
    return Math.min(1.0, conflictCount / (totalContent * 0.5));
  }

  /**
   * Assess communication quality
   */
  private assessCommunicationQuality(content: Array<Record<string, any>>): number {
    if (content.length === 0) return 0;
    
    let qualityScore = 0;
    const factors = {
      clarity: 0,
      responsiveness: 0,
      depth: 0,
      empathy: 0
    };
    
    content.forEach(msg => {
      const text = JSON.stringify(msg);
      
      // Clarity - longer, well-structured messages
      factors.clarity += Math.min(1, text.length / 200);
      
      // Responsiveness - messages that reference previous content
      if (text.includes('you said') || text.includes('regarding') || text.includes('about')) {
        factors.responsiveness += 0.5;
      }
      
      // Depth - questions, explanations, examples
      if (text.includes('?') || text.includes('because') || text.includes('example')) {
        factors.depth += 0.3;
      }
      
      // Empathy - understanding, acknowledgment
      if (text.includes('understand') || text.includes('feel') || text.includes('appreciate')) {
        factors.empathy += 0.4;
      }
    });
    
    qualityScore = Object.values(factors).reduce((sum, score) => sum + score, 0) / (content.length * 4);
    return Math.min(1.0, qualityScore);
  }

  /**
   * Assess power balance in interaction
   */
  private assessPowerBalance(participants: string[], content: Array<Record<string, any>>): number {
    // Simplified power assessment based on communication patterns
    const speakingTime: Record<string, number> = {};
    
    participants.forEach(id => {
      speakingTime[id] = 0;
    });
    
    content.forEach(msg => {
      const speaker = msg.speaker || msg.from;
      if (speaker && speakingTime.hasOwnProperty(speaker)) {
        speakingTime[speaker] += 1;
      }
    });
    
    const totalSpeaking = Object.values(speakingTime).reduce((sum, time) => sum + time, 0);
    if (totalSpeaking === 0) return 0.5; // Neutral if no clear communication
    
    // Calculate balance (1.0 = perfectly balanced, 0.0 = highly imbalanced)
    const idealShare = 1 / participants.length;
    const deviations = Object.values(speakingTime).map(time => 
      Math.abs((time / totalSpeaking) - idealShare)
    );
    const avgDeviation = deviations.reduce((sum, dev) => sum + dev, 0) / deviations.length;
    
    return Math.max(0, 1 - (avgDeviation * participants.length));
  }

  /**
   * Assess trust changes during interaction
   */
  private assessTrustChanges(participants: string[], content: Array<Record<string, any>>): number {
    const trustBuilders = ['promise', 'commit', 'guarantee', 'reliable', 'honest', 'transparent'];
    const trustBreakers = ['lie', 'deceive', 'unreliable', 'broken', 'failed', 'dishonest'];
    
    let trustScore = 0;
    
    content.forEach(msg => {
      const text = JSON.stringify(msg).toLowerCase();
      
      trustBuilders.forEach(word => {
        if (text.includes(word)) trustScore += 0.1;
      });
      
      trustBreakers.forEach(word => {
        if (text.includes(word)) trustScore -= 0.2;
      });
    });
    
    return Math.max(-1, Math.min(1, trustScore));
  }

  /**
   * Update agent relationships based on interaction
   */
  private async updateAgentRelationships(participants: string[], interaction: SocialInteraction): Promise<void> {
    const dynamics = interaction.socialDynamics;
    
    // Update pairwise relationships
    for (let i = 0; i < participants.length; i++) {
      for (let j = i + 1; j < participants.length; j++) {
        const agentA = participants[i];
        const agentB = participants[j];
        
        await this.updateBilateralRelationship(agentA, agentB, dynamics, interaction.interactionType);
      }
    }
  }

  /**
   * Update bilateral relationship between two agents
   */
  private async updateBilateralRelationship(
    agentA: string, 
    agentB: string, 
    dynamics: Record<string, number>,
    interactionType: InteractionType
  ): Promise<void> {
    const relationshipKey = `${agentA}_${agentB}`;
    let relationship = this.relationshipModels.get(relationshipKey) || {
      trust: 0.5,
      cooperation: 0.5,
      communication_quality: 0.5,
      interaction_frequency: 0,
      last_updated: new Date()
    };
    
    // Update relationship metrics
    const learningRate = 0.1;
    relationship.trust += learningRate * (dynamics.trust_changes || 0);
    relationship.cooperation += learningRate * ((dynamics.cooperation || 0.5) - relationship.cooperation);
    relationship.communication_quality += learningRate * ((dynamics.communication_quality || 0.5) - relationship.communication_quality);
    relationship.interaction_frequency += 1;
    relationship.last_updated = new Date();
    
    // Clamp values to valid ranges
    relationship.trust = Math.max(0, Math.min(1, relationship.trust));
    relationship.cooperation = Math.max(0, Math.min(1, relationship.cooperation));
    relationship.communication_quality = Math.max(0, Math.min(1, relationship.communication_quality));
    
    this.relationshipModels.set(relationshipKey, relationship);
    
    // Update agent trust levels
    const agentAObj = this.agents.get(agentA);
    const agentBObj = this.agents.get(agentB);
    
    if (agentAObj) {
      agentAObj.trustLevel = Math.max(0, Math.min(1, 
        agentAObj.trustLevel + learningRate * (dynamics.trust_changes || 0)
      ));
    }
    
    if (agentBObj) {
      agentBObj.trustLevel = Math.max(0, Math.min(1, 
        agentBObj.trustLevel + learningRate * (dynamics.trust_changes || 0)
      ));
    }
  }

  /**
   * Record behavioral observations from interaction
   */
  private async recordBehavioralObservations(interaction: SocialInteraction): Promise<void> {
    const { participants, interactionType, communicationContent, socialDynamics } = interaction;
    
    participants.forEach(agentId => {
      const agent = this.agents.get(agentId);
      if (!agent) return;
      
      // Create behavioral observation
      const observation = {
        interaction_id: interaction.interactionId,
        interaction_type: interactionType,
        behavior_patterns: this.extractBehaviorPatterns(agentId, communicationContent),
        social_responses: this.extractSocialResponses(agentId, socialDynamics),
        timestamp: interaction.timestamp
      };
      
      agent.observedBehaviors.push(observation);
      
      // Keep only recent observations (last 50)
      if (agent.observedBehaviors.length > 50) {
        agent.observedBehaviors.shift();
      }
      
      // Update predictability based on behavioral consistency
      this.updatePredictability(agent);
    });
  }

  /**
   * Extract behavior patterns for specific agent
   */
  private extractBehaviorPatterns(agentId: string, content: Array<Record<string, any>>): string[] {
    const patterns: string[] = [];
    
    // Filter content from this agent
    const agentMessages = content.filter(msg => msg.speaker === agentId || msg.from === agentId);
    
    if (agentMessages.length === 0) return patterns;
    
    // Analyze communication patterns
    const avgMessageLength = agentMessages.reduce((sum, msg) => 
      sum + JSON.stringify(msg).length, 0) / agentMessages.length;
    
    if (avgMessageLength > 200) patterns.push('verbose_communicator');
    if (avgMessageLength < 50) patterns.push('concise_communicator');
    
    // Analyze response timing (if available)
    const hasQuickResponses = agentMessages.some(msg => msg.quick_response === true);
    if (hasQuickResponses) patterns.push('quick_responder');
    
    // Analyze question patterns
    const questionCount = agentMessages.filter(msg => 
      JSON.stringify(msg).includes('?')).length;
    if (questionCount > agentMessages.length * 0.3) patterns.push('inquisitive');
    
    return patterns;
  }

  /**
   * Extract social responses for specific agent
   */
  private extractSocialResponses(agentId: string, dynamics: Record<string, number>): Record<string, number> {
    // Simplified extraction - in real implementation would be more sophisticated
    return {
      cooperation_tendency: dynamics.cooperation || 0.5,
      conflict_avoidance: 1 - (dynamics.conflict || 0),
      communication_engagement: dynamics.communication_quality || 0.5
    };
  }

  /**
   * Update agent predictability based on behavioral consistency
   */
  private updatePredictability(agent: Agent): void {
    if (agent.observedBehaviors.length < 3) return;
    
    const recentBehaviors = agent.observedBehaviors.slice(-10);
    const patterns = recentBehaviors.flatMap(obs => obs.behavior_patterns);
    
    // Calculate pattern consistency
    const patternCounts: Record<string, number> = {};
    patterns.forEach(pattern => {
      patternCounts[pattern] = (patternCounts[pattern] || 0) + 1;
    });
    
    const uniquePatterns = Object.keys(patternCounts).length;
    const totalPatterns = patterns.length;
    
    if (totalPatterns > 0) {
      const consistency = 1 - (uniquePatterns / totalPatterns);
      const learningRate = 0.1;
      agent.predictability += learningRate * (consistency - agent.predictability);
      agent.predictability = Math.max(0, Math.min(1, agent.predictability));
    }
  }

  /**
   * Update Theory of Mind models based on interaction
   */
  private async updateTheoryOfMindModels(interaction: SocialInteraction): Promise<void> {
    const { participants, communicationContent, socialDynamics } = interaction;
    
    participants.forEach(agentId => {
      const tomModel = this.theoryOfMindModels.get(agentId);
      if (!tomModel) return;
      
      // Infer beliefs from communication
      this.inferBeliefsFromCommunication(tomModel, communicationContent, agentId);
      
      // Infer desires and intentions
      this.inferDesiresAndIntentions(tomModel, communicationContent, agentId);
      
      // Infer emotional states
      this.inferEmotionalStates(tomModel, communicationContent, socialDynamics, agentId);
      
      // Generate predictions about future behavior
      this.generateBehaviorPredictions(tomModel, interaction);
    });
  }

  /**
   * Infer beliefs from communication patterns
   */
  private inferBeliefsFromCommunication(
    tomModel: TheoryOfMindModel, 
    content: Array<Record<string, any>>, 
    agentId: string
  ): void {
    const agentMessages = content.filter(msg => msg.speaker === agentId || msg.from === agentId);
    
    agentMessages.forEach(msg => {
      const text = JSON.stringify(msg).toLowerCase();
      
      // Look for belief indicators
      const beliefIndicators = [
        { pattern: /i think/, belief: 'has opinion formation tendency' },
        { pattern: /i believe/, belief: 'expresses strong convictions' },
        { pattern: /in my view/, belief: 'values personal perspective' },
        { pattern: /it seems/, belief: 'uses tentative reasoning' },
        { pattern: /clearly/, belief: 'expresses confidence in judgments' }
      ];
      
      beliefIndicators.forEach(indicator => {
        if (indicator.pattern.test(text)) {
          const existingBelief = tomModel.beliefs.find(b => b.belief === indicator.belief);
          if (existingBelief) {
            existingBelief.confidence = Math.min(1.0, existingBelief.confidence + 0.1);
          } else {
            tomModel.beliefs.push({
              belief: indicator.belief,
              confidence: 0.6,
              evidence: [msg.content || text]
            });
          }
        }
      });
    });
    
    // Keep only recent beliefs (last 20)
    if (tomModel.beliefs.length > 20) {
      tomModel.beliefs.sort((a, b) => b.confidence - a.confidence);
      tomModel.beliefs = tomModel.beliefs.slice(0, 20);
    }
  }

  /**
   * Infer desires and intentions
   */
  private inferDesiresAndIntentions(
    tomModel: TheoryOfMindModel, 
    content: Array<Record<string, any>>, 
    agentId: string
  ): void {
    const agentMessages = content.filter(msg => msg.speaker === agentId || msg.from === agentId);
    
    agentMessages.forEach(msg => {
      const text = JSON.stringify(msg).toLowerCase();
      
      // Desire indicators
      const desirePatterns = [
        { pattern: /i want/, desire: 'direct want expression', strength: 0.8 },
        { pattern: /i need/, desire: 'strong need expression', strength: 0.9 },
        { pattern: /i hope/, desire: 'hopeful desire', strength: 0.6 },
        { pattern: /i wish/, desire: 'wishful thinking', strength: 0.5 },
        { pattern: /i prefer/, desire: 'preference expression', strength: 0.7 }
      ];
      
      desirePatterns.forEach(pattern => {
        if (pattern.pattern.test(text)) {
          const existing = tomModel.desires.find(d => d.desire === pattern.desire);
          if (existing) {
            existing.strength = Math.min(1.0, existing.strength + 0.1);
          } else {
            tomModel.desires.push({
              desire: pattern.desire,
              strength: pattern.strength,
              indicators: [text]
            });
          }
        }
      });
      
      // Intention indicators
      const intentionPatterns = [
        { pattern: /i will/, intention: 'stated future action', likelihood: 0.8 },
        { pattern: /i plan/, intention: 'planned action', likelihood: 0.9 },
        { pattern: /i intend/, intention: 'intended action', likelihood: 0.85 },
        { pattern: /going to/, intention: 'future commitment', likelihood: 0.7 }
      ];
      
      intentionPatterns.forEach(pattern => {
        if (pattern.pattern.test(text)) {
          tomModel.intentions.push({
            intention: pattern.intention,
            likelihood: pattern.likelihood,
            timeframe: 'short_term' // Could be more sophisticated
          });
        }
      });
    });
    
    // Keep only recent desires and intentions
    if (tomModel.desires.length > 15) {
      tomModel.desires.sort((a, b) => b.strength - a.strength);
      tomModel.desires = tomModel.desires.slice(0, 15);
    }
    
    if (tomModel.intentions.length > 15) {
      tomModel.intentions.sort((a, b) => b.likelihood - a.likelihood);
      tomModel.intentions = tomModel.intentions.slice(0, 15);
    }
  }

  /**
   * Infer emotional states from communication and dynamics
   */
  private inferEmotionalStates(
    tomModel: TheoryOfMindModel, 
    content: Array<Record<string, any>>, 
    dynamics: Record<string, number>,
    agentId: string
  ): void {
    const agentMessages = content.filter(msg => msg.speaker === agentId || msg.from === agentId);
    
    // Emotion inference from text
    const emotionPatterns = [
      { pattern: /happy|glad|pleased|excited/, emotion: 'positive', intensity: 0.7 },
      { pattern: /sad|disappointed|upset/, emotion: 'negative', intensity: 0.6 },
      { pattern: /angry|frustrated|annoyed/, emotion: 'anger', intensity: 0.8 },
      { pattern: /worried|concerned|anxious/, emotion: 'anxiety', intensity: 0.6 },
      { pattern: /surprised|amazed|shocked/, emotion: 'surprise', intensity: 0.7 }
    ];
    
    agentMessages.forEach(msg => {
      const text = JSON.stringify(msg).toLowerCase();
      
      emotionPatterns.forEach(pattern => {
        if (pattern.pattern.test(text)) {
          tomModel.emotions.push({
            emotion: pattern.emotion,
            intensity: pattern.intensity,
            triggers: ['communication_content']
          });
        }
      });
    });
    
    // Emotion inference from social dynamics
    if (dynamics.cooperation > 0.7) {
      tomModel.emotions.push({
        emotion: 'cooperative_satisfaction',
        intensity: dynamics.cooperation,
        triggers: ['high_cooperation']
      });
    }
    
    if (dynamics.conflict > 0.6) {
      tomModel.emotions.push({
        emotion: 'conflict_stress',
        intensity: dynamics.conflict,
        triggers: ['social_conflict']
      });
    }
    
    // Keep only recent emotions (last 10)
    if (tomModel.emotions.length > 10) {
      tomModel.emotions.sort((a, b) => b.intensity - a.intensity);
      tomModel.emotions = tomModel.emotions.slice(0, 10);
    }
  }

  /**
   * Generate behavior predictions
   */
  private generateBehaviorPredictions(tomModel: TheoryOfMindModel, interaction: SocialInteraction): void {
    // Predict based on observed patterns
    const agent = this.agents.get(tomModel.agentId);
    if (!agent) return;
    
    const recentBehaviors = agent.observedBehaviors.slice(-5);
    const cooperationLevel = interaction.socialDynamics.cooperation || 0.5;
    const trustLevel = agent.trustLevel;
    
    // Generate predictions
    const predictions = [];
    
    if (cooperationLevel > 0.7 && trustLevel > 0.6) {
      predictions.push({
        prediction: 'likely_to_cooperate_in_future',
        confidence: 0.8,
        timeframe: 'short_term'
      });
    }
    
    if (agent.predictability > 0.7) {
      predictions.push({
        prediction: 'behavior_patterns_will_remain_consistent',
        confidence: agent.predictability,
        timeframe: 'medium_term'
      });
    }
    
    if (recentBehaviors.some(b => b.behavior_patterns.includes('quick_responder'))) {
      predictions.push({
        prediction: 'will_respond_quickly_to_communication',
        confidence: 0.7,
        timeframe: 'immediate'
      });
    }
    
    tomModel.predictions = predictions;
  }

  /**
   * Update social context based on interaction
   */
  private async updateSocialContext(interaction: SocialInteraction): Promise<void> {
    const contextId = `context_${interaction.participants.join('_')}`;
    let context = this.socialContexts.get(contextId);
    
    if (!context) {
      context = {
        contextId,
        participants: interaction.participants,
        relationships: {},
        groupDynamics: {},
        culturalFactors: {},
        situationalFactors: {},
        timestamp: new Date()
      };
    }
    
    // Update group dynamics
    context.groupDynamics = {
      ...context.groupDynamics,
      cooperation_level: interaction.socialDynamics.cooperation || 0.5,
      communication_quality: interaction.socialDynamics.communication_quality || 0.5,
      conflict_level: interaction.socialDynamics.conflict || 0,
      last_interaction: interaction.timestamp
    };
    
    // Update relationships matrix
    for (let i = 0; i < interaction.participants.length; i++) {
      const agentA = interaction.participants[i];
      if (!context.relationships[agentA]) {
        context.relationships[agentA] = {};
      }
      
      for (let j = 0; j < interaction.participants.length; j++) {
        if (i !== j) {
          const agentB = interaction.participants[j];
          const relationshipKey = `${agentA}_${agentB}`;
          const relationship = this.relationshipModels.get(relationshipKey);
          
          if (relationship) {
            context.relationships[agentA][agentB] = {
              trust: relationship.trust,
              cooperation: relationship.cooperation,
              communication_quality: relationship.communication_quality
            };
          }
        }
      }
    }
    
    context.timestamp = new Date();
    this.socialContexts.set(contextId, context);
  }

  /**
   * Get agent information and mental model
   */
  getAgent(agentId: string): Agent | null {
    return this.agents.get(agentId) || null;
  }

  /**
   * Get Theory of Mind model for an agent
   */
  getTheoryOfMindModel(agentId: string): TheoryOfMindModel | null {
    return this.theoryOfMindModels.get(agentId) || null;
  }

  /**
   * Get social context for a group of agents
   */
  getSocialContext(participants: string[]): SocialContext | null {
    const contextId = `context_${participants.sort().join('_')}`;
    return this.socialContexts.get(contextId) || null;
  }

  /**
   * Get relationship between two agents
   */
  getRelationship(agentA: string, agentB: string): Record<string, any> | null {
    const relationshipKey = `${agentA}_${agentB}`;
    const reverseKey = `${agentB}_${agentA}`;
    
    return this.relationshipModels.get(relationshipKey) || 
           this.relationshipModels.get(reverseKey) || 
           null;
  }

  /**
   * Get recent interactions
   */
  getRecentInteractions(limit: number = 10): SocialInteraction[] {
    return this.interactions.slice(-limit);
  }

  /**
   * Get system statistics
   */
  getStats(): {
    totalAgents: number;
    totalInteractions: number;
    averageTrustLevel: number;
    averagePredictability: number;
    activeRelationships: number;
    socialContexts: number;
  } {
    const agents = Array.from(this.agents.values());
    
    const avgTrust = agents.length > 0 ?
      agents.reduce((sum, agent) => sum + agent.trustLevel, 0) / agents.length : 0;
    
    const avgPredictability = agents.length > 0 ?
      agents.reduce((sum, agent) => sum + agent.predictability, 0) / agents.length : 0;
    
    return {
      totalAgents: this.agents.size,
      totalInteractions: this.interactions.length,
      averageTrustLevel: avgTrust,
      averagePredictability: avgPredictability,
      activeRelationships: this.relationshipModels.size,
      socialContexts: this.socialContexts.size
    };
  }

  /**
   * Cleanup old data
   */
  cleanup(): void {
    // Keep only recent interactions (last 1000)
    if (this.interactions.length > 1000) {
      this.interactions = this.interactions.slice(-1000);
    }
    
    console.log('🧹 Social cognition system cleaned up');
  }
}