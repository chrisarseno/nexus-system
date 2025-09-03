/**
 * Local SAGE System - Complete self-contained ensemble intelligence
 * No external API dependencies - runs entirely on local infrastructure
 */

import { LocalAIService } from './local-ai-service';
import { IStorage } from '../storage';

export interface LocalTask {
  id: string;
  description: string;
  type: 'research' | 'analysis' | 'verification' | 'synthesis' | 'creative';
  priority: 'low' | 'medium' | 'high' | 'critical';
  estimatedComputeTime: number;
  parentGoalId?: string;
}

export interface LocalExecutionResult {
  taskId: string;
  content: string;
  sources: string[];
  confidence: number;
  computeCost: number;
  modelUsed: string;
  executionTime: number;
  metadata: Record<string, any>;
}

export interface LocalVerificationResult {
  passed: boolean;
  confidence: number;
  issues: string[];
  believabilityScore: number;
  needsHumanReview: boolean;
  verificationModel: string;
}

export interface LocalKnowledgeFact {
  id: string;
  content: string;
  sources: string[];
  believabilityScore: number;
  lastVerified: Date;
  taskId: string;
  provenance: string[];
  version: number;
  localVerification: boolean;
}

/**
 * LOCAL PLANNER: Uses local AI for goal decomposition
 */
export class LocalTaskPlanner {
  private localAI: LocalAIService;

  constructor(localAI: LocalAIService) {
    this.localAI = localAI;
  }

  async decompose(goal: string, context: any = {}): Promise<LocalTask[]> {
    try {
      console.log(`📋 Local planning for goal: ${goal}`);
      
      const response = await this.localAI.planTasks(goal);
      
      // Parse the response and create structured tasks
      const taskDescriptions = this.extractTasks(response.content);
      
      return taskDescriptions.map((desc, index) => ({
        id: `task_${Date.now()}_${index}`,
        description: desc,
        type: this.inferTaskType(desc),
        priority: this.inferPriority(desc, goal),
        estimatedComputeTime: this.estimateComputeTime(desc),
        parentGoalId: goal
      }));

    } catch (error) {
      console.error('Local planning failed:', error);
      // Fallback to simple task breakdown
      return [{
        id: `task_${Date.now()}_fallback`,
        description: `Research and analyze: ${goal}`,
        type: 'analysis',
        priority: 'medium',
        estimatedComputeTime: 30000
      }];
    }
  }

  private extractTasks(planContent: string): string[] {
    // Extract task descriptions from AI response
    const lines = planContent.split('\n').filter(line => 
      line.trim() && 
      (line.includes('•') || line.includes('-') || line.includes('1.') || line.includes('2.'))
    );
    
    return lines.map(line => 
      line.replace(/^[•\-\d\.\s]+/, '').trim()
    ).filter(task => task.length > 10);
  }

  private inferTaskType(description: string): LocalTask['type'] {
    const desc = description.toLowerCase();
    if (desc.includes('research') || desc.includes('find') || desc.includes('gather')) return 'research';
    if (desc.includes('analyze') || desc.includes('examine') || desc.includes('evaluate')) return 'analysis';
    if (desc.includes('verify') || desc.includes('check') || desc.includes('validate')) return 'verification';
    if (desc.includes('create') || desc.includes('generate') || desc.includes('design')) return 'creative';
    return 'synthesis';
  }

  private inferPriority(description: string, goal: string): LocalTask['priority'] {
    const desc = description.toLowerCase();
    if (desc.includes('critical') || desc.includes('urgent') || desc.includes('immediate')) return 'critical';
    if (desc.includes('important') || desc.includes('key') || desc.includes('essential')) return 'high';
    if (desc.includes('optional') || desc.includes('nice to have')) return 'low';
    return 'medium';
  }

  private estimateComputeTime(description: string): number {
    // Estimate in milliseconds based on task complexity
    const words = description.split(' ').length;
    const baseTime = 10000; // 10 seconds base
    const complexityMultiplier = Math.min(3, words / 10);
    
    return Math.round(baseTime * complexityMultiplier);
  }
}

/**
 * LOCAL ROUTER: Selects optimal local models for each task
 */
export class LocalTaskRouter {
  private localAI: LocalAIService;
  private routingHistory: Map<string, number[]> = new Map();

  constructor(localAI: LocalAIService) {
    this.localAI = localAI;
  }

  selectModel(task: LocalTask, computeBudget: number): {
    modelType: string;
    confidence: number;
    estimatedCost: number;
  } {
    const availableModels = this.localAI.getAvailableModels();
    
    // Filter models by capability
    const capableModels = availableModels.filter(model => 
      model.capabilities.includes(task.type) || 
      model.specialized === this.getModelSpecialization(task.type)
    );

    if (capableModels.length === 0) {
      // Fallback to reasoning model
      return {
        modelType: 'reasoning',
        confidence: 0.6,
        estimatedCost: task.estimatedComputeTime * 0.0001
      };
    }

    // Select based on performance history and task requirements
    const bestModel = capableModels.reduce((best, current) => {
      const bestScore = this.getModelScore(best.specialized, task.type);
      const currentScore = this.getModelScore(current.specialized, task.type);
      return currentScore > bestScore ? current : best;
    });

    return {
      modelType: bestModel.specialized,
      confidence: 0.8,
      estimatedCost: task.estimatedComputeTime * 0.0001
    };
  }

  private getModelSpecialization(taskType: string): string {
    const mapping = {
      'research': 'reasoning',
      'analysis': 'analysis', 
      'verification': 'verification',
      'creative': 'creative',
      'synthesis': 'reasoning'
    };
    return mapping[taskType] || 'reasoning';
  }

  private getModelScore(modelType: string, taskType: string): number {
    const history = this.routingHistory.get(`${modelType}_${taskType}`) || [0.7];
    return history.reduce((sum, score) => sum + score) / history.length;
  }

  updatePerformance(modelType: string, taskType: string, performance: number) {
    const key = `${modelType}_${taskType}`;
    const history = this.routingHistory.get(key) || [];
    history.push(performance);
    
    if (history.length > 10) {
      history.shift();
    }
    
    this.routingHistory.set(key, history);
  }
}

/**
 * LOCAL EXECUTOR: Runs tasks using selected local models
 */
export class LocalTaskExecutor {
  private localAI: LocalAIService;

  constructor(localAI: LocalAIService) {
    this.localAI = localAI;
  }

  async execute(task: LocalTask, selectedModel: any): Promise<LocalExecutionResult> {
    const startTime = Date.now();

    try {
      console.log(`⚡ Executing task ${task.id} with ${selectedModel.modelType} model`);
      
      let result: any;

      switch (task.type) {
        case 'analysis':
          result = await this.localAI.analyzeTask(task.description, {});
          break;
        case 'creative':
          result = await this.localAI.generateCreative(task.description);
          break;
        case 'verification':
          result = await this.localAI.verifyResult(task.description, task.description);
          break;
        default:
          result = await this.localAI.generateResponse(task.description, selectedModel.modelType);
      }

      return {
        taskId: task.id,
        content: result.content,
        sources: ['local_ai_processing'],
        confidence: result.confidence,
        computeCost: result.cost,
        modelUsed: result.model,
        executionTime: Date.now() - startTime,
        metadata: {
          modelType: selectedModel.modelType,
          tokens: result.tokensGenerated,
          localProcessing: true
        }
      };

    } catch (error) {
      console.error(`Local execution failed for task ${task.id}:`, error);
      throw error;
    }
  }
}

/**
 * LOCAL VERIFIER: Multi-model verification using different local models
 */
export class LocalTaskVerifier {
  private localAI: LocalAIService;

  constructor(localAI: LocalAIService) {
    this.localAI = localAI;
  }

  async verify(result: LocalExecutionResult, task: LocalTask): Promise<LocalVerificationResult> {
    try {
      console.log(`✅ Verifying task ${task.id} with adversarial model`);
      
      // Use a different model for verification to avoid same-model bias
      const verificationResponse = await this.localAI.verifyResult(result.content, task.description);
      
      // Extract verification details
      const believabilityScore = this.extractBelievabilityScore(verificationResponse.content);
      const issues = this.extractIssues(verificationResponse.content);
      
      const passed = believabilityScore > 0.7 && issues.length < 2;
      const needsHumanReview = believabilityScore < 0.8 || issues.length > 0;

      return {
        passed,
        confidence: verificationResponse.confidence,
        issues,
        believabilityScore,
        needsHumanReview,
        verificationModel: verificationResponse.model
      };

    } catch (error) {
      console.error('Local verification failed:', error);
      return {
        passed: false,
        confidence: 0.5,
        issues: ['Verification process failed'],
        believabilityScore: 0.5,
        needsHumanReview: true,
        verificationModel: 'fallback'
      };
    }
  }

  private extractBelievabilityScore(content: string): number {
    // Extract believability score from verification response
    const match = content.match(/believability[:\s]*([0-9.]+)/i);
    if (match) {
      return Math.min(1.0, Math.max(0.0, parseFloat(match[1])));
    }
    
    // Fallback: estimate based on content indicators
    if (content.includes('high confidence') || content.includes('reliable')) return 0.9;
    if (content.includes('moderate') || content.includes('likely')) return 0.7;
    if (content.includes('uncertain') || content.includes('unclear')) return 0.5;
    
    return 0.6;
  }

  private extractIssues(content: string): string[] {
    const issues: string[] = [];
    
    if (content.includes('inconsistent')) issues.push('Logical inconsistency detected');
    if (content.includes('unsupported')) issues.push('Insufficient evidence');
    if (content.includes('bias')) issues.push('Potential bias identified');
    if (content.includes('incomplete')) issues.push('Incomplete analysis');
    
    return issues;
  }
}

/**
 * LOCAL KNOWLEDGE BASE: Stores verified facts with local tracking
 */
export class LocalKnowledgeBase {
  private storage: IStorage;
  private facts: Map<string, LocalKnowledgeFact> = new Map();

  constructor(storage: IStorage) {
    this.storage = storage;
  }

  async storeFact(
    result: LocalExecutionResult,
    verification: LocalVerificationResult,
    task: LocalTask
  ): Promise<LocalKnowledgeFact> {
    const fact: LocalKnowledgeFact = {
      id: `fact_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      content: result.content,
      sources: result.sources,
      believabilityScore: verification.believabilityScore,
      lastVerified: new Date(),
      taskId: task.id,
      provenance: [
        `Task: ${task.description}`,
        `Local Model: ${result.modelUsed}`,
        `Verification: ${verification.verificationModel}`,
        `Confidence: ${result.confidence}`
      ],
      version: 1,
      localVerification: true
    };

    this.facts.set(fact.id, fact);

    // Store in activity feed
    await this.storage.addActivity({
      type: 'knowledge',
      message: `Local fact verified: ${fact.content.substring(0, 100)}...`,
      moduleId: 'local_knowledge'
    });

    return fact;
  }

  async getFacts(query?: string): Promise<LocalKnowledgeFact[]> {
    const allFacts = Array.from(this.facts.values());
    
    if (!query) {
      return allFacts.sort((a, b) => b.believabilityScore - a.believabilityScore);
    }

    return allFacts.filter(fact => 
      fact.content.toLowerCase().includes(query.toLowerCase())
    ).sort((a, b) => b.believabilityScore - a.believabilityScore);
  }
}

/**
 * LOCAL SAGE SYSTEM: Complete local ensemble intelligence
 */
export class LocalSAGESystem {
  private localAI: LocalAIService;
  private planner: LocalTaskPlanner;
  private router: LocalTaskRouter;
  private executor: LocalTaskExecutor;
  private verifier: LocalTaskVerifier;
  private knowledgeBase: LocalKnowledgeBase;
  private storage: IStorage;

  constructor(storage: IStorage) {
    this.storage = storage;
    this.localAI = new LocalAIService();
    this.planner = new LocalTaskPlanner(this.localAI);
    this.router = new LocalTaskRouter(this.localAI);
    this.executor = new LocalTaskExecutor(this.localAI);
    this.verifier = new LocalTaskVerifier(this.localAI);
    this.knowledgeBase = new LocalKnowledgeBase(storage);
  }

  async executeGoal(goal: string, context: any = {}, computeBudget: number = 60000): Promise<{
    tasks: LocalTask[];
    results: LocalExecutionResult[];
    verifiedFacts: LocalKnowledgeFact[];
    needsHumanReview: string[];
    totalCost: number;
    systemMetrics: any;
  }> {
    console.log(`🎯 Local SAGE Learning Loop: Processing goal - ${goal}`);

    try {
      // 1. PLAN: Decompose goal using local AI
      const tasks = await this.planner.decompose(goal, context);
      console.log(`📋 Planned ${tasks.length} tasks locally`);

      const results: LocalExecutionResult[] = [];
      const verifiedFacts: LocalKnowledgeFact[] = [];
      const needsHumanReview: string[] = [];
      let totalCost = 0;

      // 2. EXECUTE & VERIFY each task locally
      for (const task of tasks) {
        try {
          // 3. ROUTE: Select optimal local model
          const selectedModel = this.router.selectModel(task, computeBudget - totalCost);
          console.log(`🔀 Routing task ${task.id} to ${selectedModel.modelType} model`);

          // 4. EXECUTE: Run task with local AI
          const result = await this.executor.execute(task, selectedModel);
          results.push(result);
          totalCost += result.computeCost;

          // 5. VERIFY: Check with different local model
          const verification = await this.verifier.verify(result, task);

          // 6. STORE: Save verified facts
          if (verification.passed) {
            const fact = await this.knowledgeBase.storeFact(result, verification, task);
            verifiedFacts.push(fact);
            console.log(`✅ Local fact verified and stored: ${fact.id}`);
          } else {
            console.log(`❌ Local verification failed for task ${task.id}`);
          }

          // 7. HUMAN REVIEW: Flag for review
          if (verification.needsHumanReview) {
            needsHumanReview.push(task.id);
          }

          // 8. LEARN: Update routing performance
          this.router.updatePerformance(selectedModel.modelType, task.type, verification.confidence);

        } catch (error) {
          console.error(`Local task ${task.id} failed:`, error);
          continue; // Graceful degradation
        }
      }

      // 9. REFLECT: Add system activity
      await this.storage.addActivity({
        type: 'knowledge',
        message: `Local SAGE completed: ${verifiedFacts.length} facts verified, $${totalCost.toFixed(6)} compute cost`,
        moduleId: 'local_sage'
      });

      const systemMetrics = this.localAI.getSystemMetrics();

      return {
        tasks,
        results,
        verifiedFacts,
        needsHumanReview,
        totalCost,
        systemMetrics
      };

    } catch (error) {
      console.error('Local SAGE Learning Loop failed:', error);
      throw error;
    }
  }

  async getKnowledgeBase() {
    return this.knowledgeBase;
  }

  async getLocalMetrics() {
    return this.localAI.getSystemMetrics();
  }

  getTotalCost(): number {
    return this.localAI.getTotalCost();
  }

  getHourlyCostRate(): number {
    return this.localAI.getHourlyCostRate();
  }
}