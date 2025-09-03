import { LocalSAGESystem } from './local-sage-system';
import { IStorage } from '../storage';

export interface ReviewRequest {
  id: string;
  type: 'goal_approval' | 'fact_verification' | 'task_review' | 'ethical_check' | 'safety_override';
  title: string;
  description: string;
  data: any;
  urgency: 'low' | 'medium' | 'high' | 'critical';
  createdAt: Date;
  status: 'pending' | 'approved' | 'rejected' | 'escalated';
  requesterSystem: string;
  humanFeedback?: string;
  reviewedAt?: Date;
  autoApproved?: boolean;
}

export interface HumanDecision {
  requestId: string;
  decision: 'approve' | 'reject' | 'modify' | 'escalate';
  feedback: string;
  modifications?: any;
  confidence: number;
}

export interface CollaborationMetrics {
  totalReviews: number;
  approvalRate: number;
  avgReviewTime: number;
  criticalInterventions: number;
  humanSystemTrust: number;
  systemReliability: number;
}

/**
 * Human-in-the-Loop Collaboration System
 * Manages review requests, approvals, and human feedback integration
 */
export class HumanCollaborationEngine {
  private sageSystem: LocalSAGESystem;
  private storage: IStorage;
  private reviewQueue: Map<string, ReviewRequest> = new Map();
  private decisionHistory: HumanDecision[] = [];
  private metrics: CollaborationMetrics;

  constructor(sageSystem: LocalSAGESystem, storage: IStorage) {
    this.sageSystem = sageSystem;
    this.storage = storage;
    this.metrics = {
      totalReviews: 0,
      approvalRate: 0.85,
      avgReviewTime: 300, // 5 minutes in seconds
      criticalInterventions: 0,
      humanSystemTrust: 0.82,
      systemReliability: 0.88
    };
  }

  /**
   * Submit a request for human review
   */
  async submitReviewRequest(
    type: ReviewRequest['type'],
    title: string,
    description: string,
    data: any,
    urgency: ReviewRequest['urgency'] = 'medium'
  ): Promise<ReviewRequest> {
    const request: ReviewRequest = {
      id: `review_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type,
      title,
      description,
      data,
      urgency,
      createdAt: new Date(),
      status: 'pending',
      requesterSystem: 'local_sage'
    };

    // Check if request can be auto-approved based on historical patterns
    const autoApproval = await this.checkAutoApproval(request);
    if (autoApproval.canAutoApprove) {
      request.status = 'approved';
      request.autoApproved = true;
      request.humanFeedback = `Auto-approved: ${autoApproval.reason}`;
      request.reviewedAt = new Date();
      
      console.log(`🤖 Auto-approved review: ${request.title}`);
    } else {
      console.log(`👤 Human review required: ${request.title} (${urgency})`);
    }

    this.reviewQueue.set(request.id, request);

    // Add to activity feed
    await this.storage.addActivity({
      type: request.autoApproved ? 'knowledge' : 'safety',
      message: request.autoApproved 
        ? `Auto-approved: ${request.title}`
        : `Human review requested: ${request.title}`,
      moduleId: 'human_collaboration'
    });

    return request;
  }

  /**
   * Get pending review requests for human attention
   */
  getPendingReviews(limit: number = 20): ReviewRequest[] {
    return Array.from(this.reviewQueue.values())
      .filter(r => r.status === 'pending')
      .sort((a, b) => {
        // Sort by urgency first, then by creation time
        const urgencyOrder = { 'critical': 4, 'high': 3, 'medium': 2, 'low': 1 };
        const urgencyDiff = urgencyOrder[b.urgency] - urgencyOrder[a.urgency];
        if (urgencyDiff !== 0) return urgencyDiff;
        return b.createdAt.getTime() - a.createdAt.getTime();
      })
      .slice(0, limit);
  }

  /**
   * Process human decision on a review request
   */
  async processHumanDecision(decision: HumanDecision): Promise<{
    success: boolean;
    message: string;
    followupActions?: string[];
  }> {
    const request = this.reviewQueue.get(decision.requestId);
    if (!request) {
      return { success: false, message: 'Review request not found' };
    }

    // Update request status
    request.status = decision.decision === 'approve' ? 'approved' : 'rejected';
    if (decision.decision === 'escalate') {
      request.status = 'escalated';
    }
    request.humanFeedback = decision.feedback;
    request.reviewedAt = new Date();

    // Store decision in history
    this.decisionHistory.push({
      ...decision,
      confidence: decision.confidence || 0.9
    });

    // Update metrics
    this.updateCollaborationMetrics(request, decision);

    // Process the decision based on request type
    const followupActions = await this.executeDecision(request, decision);

    // Log activity
    await this.storage.addActivity({
      type: decision.decision === 'approve' ? 'knowledge' : 'safety',
      message: `Human ${decision.decision}: ${request.title}`,
      moduleId: 'human_collaboration'
    });

    console.log(`👤 Human decision processed: ${decision.decision} for ${request.title}`);

    return {
      success: true,
      message: `Decision processed: ${decision.decision}`,
      followupActions
    };
  }

  /**
   * Get triage dashboard data for human review
   */
  getTriageDashboard(): {
    pendingReviews: ReviewRequest[];
    criticalItems: ReviewRequest[];
    recentDecisions: HumanDecision[];
    metrics: CollaborationMetrics;
    systemHealth: {
      autonomyLevel: number;
      interventionRate: number;
      trustScore: number;
    };
  } {
    const pendingReviews = this.getPendingReviews(10);
    const criticalItems = pendingReviews.filter(r => r.urgency === 'critical');
    const recentDecisions = this.decisionHistory.slice(-10);

    const autonomyLevel = 1 - (pendingReviews.length / Math.max(this.metrics.totalReviews, 1));
    const interventionRate = this.metrics.criticalInterventions / Math.max(this.metrics.totalReviews, 1);

    return {
      pendingReviews,
      criticalItems,
      recentDecisions,
      metrics: this.metrics,
      systemHealth: {
        autonomyLevel: Math.max(0, Math.min(1, autonomyLevel)),
        interventionRate,
        trustScore: this.metrics.humanSystemTrust
      }
    };
  }

  /**
   * Check if request can be auto-approved based on patterns
   */
  private async checkAutoApproval(request: ReviewRequest): Promise<{
    canAutoApprove: boolean;
    reason: string;
    confidence: number;
  }> {
    // Simple auto-approval rules
    const similarRequests = this.decisionHistory.filter(d => {
      const req = this.reviewQueue.get(d.requestId);
      return req && req.type === request.type;
    });

    if (similarRequests.length < 3) {
      return { 
        canAutoApprove: false, 
        reason: 'Insufficient historical data', 
        confidence: 0 
      };
    }

    const recentApprovals = similarRequests
      .slice(-10) // Last 10 similar requests
      .filter(d => d.decision === 'approve');

    const approvalRate = recentApprovals.length / Math.min(similarRequests.length, 10);

    // Auto-approve if high approval rate and low urgency
    if (approvalRate > 0.8 && request.urgency !== 'critical') {
      return {
        canAutoApprove: true,
        reason: `High approval rate (${(approvalRate * 100).toFixed(1)}%) for similar requests`,
        confidence: approvalRate
      };
    }

    // Auto-approve routine verification tasks
    if (request.type === 'fact_verification' && approvalRate > 0.7) {
      return {
        canAutoApprove: true,
        reason: 'Routine fact verification with good track record',
        confidence: approvalRate
      };
    }

    return { 
      canAutoApprove: false, 
      reason: 'Requires human judgment', 
      confidence: 0 
    };
  }

  /**
   * Execute the decision and return follow-up actions
   */
  private async executeDecision(request: ReviewRequest, decision: HumanDecision): Promise<string[]> {
    const followupActions: string[] = [];

    switch (request.type) {
      case 'goal_approval':
        if (decision.decision === 'approve') {
          followupActions.push('Execute approved goal');
          // Could trigger SAGE system to continue with the goal
        } else {
          followupActions.push('Goal rejected - notify system');
        }
        break;

      case 'fact_verification':
        if (decision.decision === 'approve') {
          followupActions.push('Add fact to verified knowledge base');
          // Could store the fact with high confidence
        } else {
          followupActions.push('Flag fact as disputed');
        }
        break;

      case 'task_review':
        if (decision.decision === 'modify' && decision.modifications) {
          followupActions.push('Apply human modifications to task');
        }
        break;

      case 'ethical_check':
        if (decision.decision === 'reject') {
          followupActions.push('Halt system operation');
          followupActions.push('Implement ethical safeguards');
          this.metrics.criticalInterventions++;
        }
        break;

      case 'safety_override':
        if (decision.decision === 'approve') {
          followupActions.push('Execute safety override');
          this.metrics.criticalInterventions++;
        }
        break;
    }

    return followupActions;
  }

  /**
   * Update collaboration metrics based on human decision
   */
  private updateCollaborationMetrics(request: ReviewRequest, decision: HumanDecision) {
    this.metrics.totalReviews++;
    
    if (decision.decision === 'approve') {
      const currentTotal = this.metrics.totalReviews - 1;
      const currentApprovals = Math.round(this.metrics.approvalRate * currentTotal);
      this.metrics.approvalRate = (currentApprovals + 1) / this.metrics.totalReviews;
    }

    const reviewTime = request.reviewedAt && request.createdAt 
      ? (request.reviewedAt.getTime() - request.createdAt.getTime()) / 1000
      : this.metrics.avgReviewTime;
    
    this.metrics.avgReviewTime = 
      (this.metrics.avgReviewTime * (this.metrics.totalReviews - 1) + reviewTime) / 
      this.metrics.totalReviews;

    // Update trust score based on human feedback confidence
    if (decision.confidence) {
      this.metrics.humanSystemTrust = 
        (this.metrics.humanSystemTrust * 0.9) + (decision.confidence * 0.1);
    }
  }

  /**
   * Generate workflow recommendations for improving collaboration
   */
  async generateWorkflowRecommendations(): Promise<{
    recommendations: string[];
    automationOpportunities: string[];
    riskAreas: string[];
  }> {
    const recommendations: string[] = [];
    const automationOpportunities: string[] = [];
    const riskAreas: string[] = [];

    // Analyze patterns in decision history
    if (this.metrics.approvalRate > 0.9) {
      automationOpportunities.push('Consider increasing auto-approval thresholds');
    }

    if (this.metrics.avgReviewTime > 600) { // 10 minutes
      recommendations.push('Streamline review process - average review time is high');
    }

    if (this.metrics.criticalInterventions > 5) {
      riskAreas.push('High number of critical interventions - review system behavior');
    }

    if (this.metrics.humanSystemTrust < 0.7) {
      riskAreas.push('Low human-system trust score - investigate system reliability');
      recommendations.push('Increase transparency in system decision-making');
    }

    const pendingCritical = this.getPendingReviews().filter(r => r.urgency === 'critical');
    if (pendingCritical.length > 0) {
      riskAreas.push(`${pendingCritical.length} critical items awaiting review`);
    }

    return {
      recommendations,
      automationOpportunities,
      riskAreas
    };
  }

  /**
   * Get collaboration metrics
   */
  getMetrics(): CollaborationMetrics {
    return { ...this.metrics };
  }
}