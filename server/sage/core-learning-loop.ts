import { OpenAIService } from "../ai/openai-service";
import { IStorage } from "../storage";

// Core Learning Loop Implementation following the SAGE blueprint

export interface Task {
  id: string;
  description: string;
  type: "research" | "analysis" | "verification" | "synthesis" | "creative";
  priority: "low" | "medium" | "high" | "critical";
  requiredCapabilities: string[];
  estimatedCost: number;
  parentGoalId?: string;
}

export interface ExecutionResult {
  taskId: string;
  content: string;
  sources: string[];
  confidence: number;
  cost: number;
  modelUsed: string;
  executionTime: number;
  metadata: Record<string, any>;
}

export interface VerificationResult {
  passed: boolean;
  confidence: number;
  issues: string[];
  sources: string[];
  believabilityScore: number;
  needsHumanReview: boolean;
}

export interface KnowledgeFact {
  id: string;
  content: string;
  sources: string[];
  believabilityScore: number;
  lastVerified: Date;
  taskId: string;
  provenance: string[];
  version: number;
}

/**
 * PLANNER: Decomposes user goals into executable tasks
 */
export class TaskPlanner {
  private openai: OpenAIService;

  constructor(openai: OpenAIService) {
    this.openai = openai;
  }

  async decompose(goal: string, context: any = {}): Promise<Task[]> {
    const prompt = `Break down this goal into specific, executable tasks:

Goal: ${goal}
Context: ${JSON.stringify(context, null, 2)}

Create a task breakdown that follows these principles:
1. Each task should be specific and actionable
2. Tasks should build on each other logically
3. Include verification and validation steps
4. Estimate computational cost (low/medium/high)
5. Identify required capabilities

Return a JSON array of tasks with this structure:
{
  "tasks": [
    {
      "description": "specific task description",
      "type": "research|analysis|verification|synthesis|creative",
      "priority": "low|medium|high|critical",
      "requiredCapabilities": ["capability1", "capability2"],
      "estimatedCost": 1-10
    }
  ]
}`;

    try {
      const response = await this.openai.processConversation([
        { role: "user", content: prompt }
      ]);

      const parsed = JSON.parse(response.response);
      
      return parsed.tasks.map((task: any, index: number) => ({
        id: `task_${Date.now()}_${index}`,
        description: task.description,
        type: task.type || "analysis",
        priority: task.priority || "medium",
        requiredCapabilities: task.requiredCapabilities || [],
        estimatedCost: task.estimatedCost || 3,
        parentGoalId: goal
      }));

    } catch (error) {
      console.error("Task planning failed:", error);
      // Fallback to basic task breakdown
      return [{
        id: `task_${Date.now()}_fallback`,
        description: `Analyze and research: ${goal}`,
        type: "analysis",
        priority: "medium",
        requiredCapabilities: ["reasoning", "research"],
        estimatedCost: 5
      }];
    }
  }
}

/**
 * ROUTER: Selects best model/tool for each task (cost-aware)
 */
export class TaskRouter {
  private routingPolicies: Map<string, any> = new Map();
  private performanceHistory: Map<string, number[]> = new Map();

  constructor() {
    // Initialize with baseline policies
    this.initializeBasePolicies();
  }

  private initializeBasePolicies() {
    this.routingPolicies.set("research", {
      preferredModels: ["gpt-4o", "claude"],
      costThreshold: 7,
      accuracyWeight: 0.8,
      speedWeight: 0.2
    });
    
    this.routingPolicies.set("analysis", {
      preferredModels: ["gpt-4o"],
      costThreshold: 5,
      accuracyWeight: 0.9,
      speedWeight: 0.1
    });
    
    this.routingPolicies.set("creative", {
      preferredModels: ["gpt-4o"],
      costThreshold: 8,
      accuracyWeight: 0.6,
      speedWeight: 0.4
    });
  }

  selectTool(task: Task, costBudget: number): {
    tool: string;
    model: string;
    confidence: number;
  } {
    const policy = this.routingPolicies.get(task.type) || this.routingPolicies.get("analysis");
    
    // Cost-aware selection
    if (task.estimatedCost > costBudget) {
      return {
        tool: "lightweight_analysis",
        model: "gpt-4o-mini",
        confidence: 0.7
      };
    }

    // High-priority tasks get premium models
    if (task.priority === "critical" || task.priority === "high") {
      return {
        tool: "openai_service",
        model: "gpt-4o",
        confidence: 0.95
      };
    }

    // Standard routing based on performance history
    const historicalPerformance = this.performanceHistory.get(task.type) || [0.8];
    const avgPerformance = historicalPerformance.reduce((a, b) => a + b) / historicalPerformance.length;

    return {
      tool: "openai_service",
      model: policy.preferredModels[0],
      confidence: avgPerformance
    };
  }

  updatePerformance(taskType: string, performance: number) {
    const history = this.performanceHistory.get(taskType) || [];
    history.push(performance);
    
    // Keep only last 20 entries
    if (history.length > 20) {
      history.shift();
    }
    
    this.performanceHistory.set(taskType, history);
  }
}

/**
 * EXECUTOR: Runs tasks using selected tools
 */
export class TaskExecutor {
  private openai: OpenAIService;

  constructor(openai: OpenAIService) {
    this.openai = openai;
  }

  async execute(task: Task, selectedTool: any): Promise<ExecutionResult> {
    const startTime = Date.now();

    try {
      let result: any;

      switch (selectedTool.tool) {
        case "openai_service":
          result = await this.executeWithOpenAI(task, selectedTool.model);
          break;
        case "lightweight_analysis":
          result = await this.executeLightweightAnalysis(task);
          break;
        default:
          result = await this.executeWithOpenAI(task, "gpt-4o");
      }

      return {
        taskId: task.id,
        content: result.content,
        sources: result.sources || [],
        confidence: result.confidence || 0.8,
        cost: result.cost || 0.01,
        modelUsed: selectedTool.model,
        executionTime: Date.now() - startTime,
        metadata: result.metadata || {}
      };

    } catch (error) {
      console.error(`Task execution failed for ${task.id}:`, error);
      throw error;
    }
  }

  private async executeWithOpenAI(task: Task, model: string) {
    const prompt = `Execute this task with high quality and cite sources:

Task: ${task.description}
Type: ${task.type}
Required capabilities: ${task.requiredCapabilities.join(", ")}

Provide:
1. Detailed analysis or result
2. Source citations where applicable
3. Confidence level in your response
4. Any caveats or limitations

Format as JSON:
{
  "content": "detailed result",
  "sources": ["source1", "source2"],
  "confidence": 0.85,
  "caveats": ["limitation1", "limitation2"]
}`;

    const response = await this.openai.processConversation([
      { role: "user", content: prompt }
    ]);

    try {
      const parsed = JSON.parse(response.response);
      return {
        content: parsed.content,
        sources: parsed.sources || [],
        confidence: parsed.confidence || 0.8,
        cost: response.cost,
        metadata: {
          caveats: parsed.caveats || [],
          sentiment: response.sentiment
        }
      };
    } catch (parseError) {
      return {
        content: response.response,
        sources: [],
        confidence: 0.7,
        cost: response.cost,
        metadata: { parseError: true }
      };
    }
  }

  private async executeLightweightAnalysis(task: Task) {
    // Simplified analysis for cost-conscious execution
    return {
      content: `Lightweight analysis of: ${task.description}`,
      sources: [],
      confidence: 0.6,
      cost: 0.001,
      metadata: { mode: "lightweight" }
    };
  }
}

/**
 * VERIFIER: Multi-source verification with confidence scoring
 */
export class TaskVerifier {
  private openai: OpenAIService;

  constructor(openai: OpenAIService) {
    this.openai = openai;
  }

  async verify(result: ExecutionResult, task: Task): Promise<VerificationResult> {
    const verificationPrompt = `Verify this result with skeptical analysis:

Task: ${task.description}
Result: ${result.content}
Sources: ${result.sources.join(", ")}
Model Confidence: ${result.confidence}

Perform verification checks:
1. Fact accuracy and logical consistency
2. Source quality and independence
3. Identify any potential biases or gaps
4. Rate believability (0-1 scale)
5. Determine if human review is needed

Return JSON:
{
  "passed": true/false,
  "confidence": 0.85,
  "issues": ["issue1", "issue2"],
  "believabilityScore": 0.9,
  "needsHumanReview": false,
  "reasoning": "detailed reasoning"
}`;

    try {
      const response = await this.openai.processConversation([
        {
          role: "system",
          content: "You are a skeptical fact-checker. Be rigorous and demand high standards of evidence."
        },
        { role: "user", content: verificationPrompt }
      ]);

      const parsed = JSON.parse(response.response);

      return {
        passed: parsed.passed && parsed.believabilityScore > 0.7,
        confidence: parsed.confidence || 0.7,
        issues: parsed.issues || [],
        sources: result.sources,
        believabilityScore: parsed.believabilityScore || 0.7,
        needsHumanReview: parsed.needsHumanReview || parsed.believabilityScore < 0.8
      };

    } catch (error) {
      console.error("Verification failed:", error);
      return {
        passed: false,
        confidence: 0.5,
        issues: ["Verification process failed"],
        sources: result.sources,
        believabilityScore: 0.5,
        needsHumanReview: true
      };
    }
  }
}

/**
 * KNOWLEDGE BASE: Stores verified facts with provenance
 */
export class KnowledgeBase {
  private storage: IStorage;
  private facts: Map<string, KnowledgeFact> = new Map();

  constructor(storage: IStorage) {
    this.storage = storage;
  }

  async storeFact(
    result: ExecutionResult,
    verification: VerificationResult,
    task: Task
  ): Promise<KnowledgeFact> {
    const fact: KnowledgeFact = {
      id: `fact_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      content: result.content,
      sources: verification.sources,
      believabilityScore: verification.believabilityScore,
      lastVerified: new Date(),
      taskId: task.id,
      provenance: [
        `Task: ${task.description}`,
        `Model: ${result.modelUsed}`,
        `Confidence: ${result.confidence}`,
        `Verification: ${verification.confidence}`
      ],
      version: 1
    };

    this.facts.set(fact.id, fact);

    // Store in activity feed
    await this.storage.addActivity({
      type: "knowledge",
      message: `New fact stored: ${fact.content.substring(0, 100)}...`,
      moduleId: "knowledge_base"
    });

    return fact;
  }

  async getFacts(query?: string): Promise<KnowledgeFact[]> {
    const allFacts = Array.from(this.facts.values());
    
    if (!query) {
      return allFacts.sort((a, b) => b.believabilityScore - a.believabilityScore);
    }

    // Simple keyword search
    return allFacts.filter(fact => 
      fact.content.toLowerCase().includes(query.toLowerCase())
    ).sort((a, b) => b.believabilityScore - a.believabilityScore);
  }

  async getFactById(id: string): Promise<KnowledgeFact | null> {
    return this.facts.get(id) || null;
  }

  async updateBelievability(factId: string, newScore: number, reason: string) {
    const fact = this.facts.get(factId);
    if (fact) {
      fact.believabilityScore = newScore;
      fact.lastVerified = new Date();
      fact.provenance.push(`Believability updated: ${newScore} (${reason})`);
      fact.version++;
      
      this.facts.set(factId, fact);
    }
  }
}

/**
 * SAGE CORE LEARNING LOOP: Orchestrates the entire process
 */
export class SAGELearningLoop {
  private planner: TaskPlanner;
  private router: TaskRouter;
  private executor: TaskExecutor;
  private verifier: TaskVerifier;
  private knowledgeBase: KnowledgeBase;
  private storage: IStorage;

  constructor(openai: OpenAIService, storage: IStorage) {
    this.planner = new TaskPlanner(openai);
    this.router = new TaskRouter();
    this.executor = new TaskExecutor(openai);
    this.verifier = new TaskVerifier(openai);
    this.knowledgeBase = new KnowledgeBase(storage);
    this.storage = storage;
  }

  async executeGoal(goal: string, context: any = {}, costBudget: number = 10): Promise<{
    tasks: Task[];
    results: ExecutionResult[];
    verifiedFacts: KnowledgeFact[];
    needsHumanReview: string[];
    totalCost: number;
  }> {
    console.log(`🎯 SAGE Learning Loop: Processing goal - ${goal}`);

    try {
      // 1. PLAN: Decompose goal into tasks
      const tasks = await this.planner.decompose(goal, context);
      console.log(`📋 Planned ${tasks.length} tasks`);

      const results: ExecutionResult[] = [];
      const verifiedFacts: KnowledgeFact[] = [];
      const needsHumanReview: string[] = [];
      let totalCost = 0;

      // 2. EXECUTE & VERIFY each task
      for (const task of tasks) {
        try {
          // 3. ROUTE: Select best tool for task
          const selectedTool = this.router.selectTool(task, costBudget - totalCost);
          console.log(`🔀 Routing task ${task.id} to ${selectedTool.tool}`);

          // 4. EXECUTE: Run the task
          const result = await this.executor.execute(task, selectedTool);
          results.push(result);
          totalCost += result.cost;

          // 5. VERIFY: Check result quality
          const verification = await this.verifier.verify(result, task);

          // 6. STORE: Save verified facts
          if (verification.passed) {
            const fact = await this.knowledgeBase.storeFact(result, verification, task);
            verifiedFacts.push(fact);
            console.log(`✅ Fact verified and stored: ${fact.id}`);
          } else {
            console.log(`❌ Verification failed for task ${task.id}`);
          }

          // 7. HUMAN REVIEW: Flag for human attention
          if (verification.needsHumanReview) {
            needsHumanReview.push(task.id);
          }

          // 8. LEARN: Update routing performance
          this.router.updatePerformance(task.type, verification.confidence);

        } catch (error) {
          console.error(`Task ${task.id} failed:`, error);
          continue; // Graceful degradation
        }
      }

      // 9. REFLECT: Add reflection activity
      await this.storage.addActivity({
        type: "knowledge",
        message: `SAGE Goal completed: ${verifiedFacts.length} facts verified, $${totalCost.toFixed(4)} cost`,
        moduleId: "sage_core"
      });

      return {
        tasks,
        results,
        verifiedFacts,
        needsHumanReview,
        totalCost
      };

    } catch (error) {
      console.error("SAGE Learning Loop failed:", error);
      throw error;
    }
  }

  async getKnowledgeBase() {
    return this.knowledgeBase;
  }

  async getPerformanceMetrics() {
    return {
      routingHistory: this.router,
      totalFacts: (await this.knowledgeBase.getFacts()).length,
      averageBelievability: (await this.knowledgeBase.getFacts())
        .reduce((sum, fact) => sum + fact.believabilityScore, 0) / 
        (await this.knowledgeBase.getFacts()).length
    };
  }
}