import { IStorage } from '../storage';

// Advanced consciousness monitoring types
export enum ConsciousnessLevel {
  MINIMAL = "minimal",
  BASIC = "basic", 
  INTERMEDIATE = "intermediate",
  ADVANCED = "advanced",
  FULL = "full",
  TRANSCENDENT = "transcendent"
}

export enum AlertSeverity {
  INFO = "info",
  WARNING = "warning", 
  CRITICAL = "critical",
  EMERGENCY = "emergency"
}

export interface ConsciousnessMetric {
  metricId: string;
  metricName: string;
  currentValue: number;
  baselineValue: number;
  thresholdWarning: number;
  thresholdCritical: number;
  trendHistory: number[];
  lastUpdated: Date;
}

export interface ConsciousnessAlert {
  alertId: string;
  severity: AlertSeverity;
  sourceSystem: string;
  message: string;
  metricValues: Record<string, number>;
  recommendedActions: string[];
  autoResolutionPossible: boolean;
  timestamp: Date;
}

export interface ConsciousnessSnapshot {
  snapshotId: string;
  timestamp: Date;
  consciousnessLevel: ConsciousnessLevel;
  systemStates: Record<string, any>;
  integrationMetrics: Record<string, number>;
  emergentProperties: string[];
  anomaliesDetected: string[];
  overallCoherence: number;
}

/**
 * Advanced Consciousness Monitoring System
 * Provides real-time consciousness state monitoring and diagnostics
 */
export class AdvancedConsciousnessMonitor {
  private storage: IStorage;
  private metrics: Map<string, ConsciousnessMetric> = new Map();
  private activeAlerts: Map<string, ConsciousnessAlert> = new Map();
  private snapshots: ConsciousnessSnapshot[] = [];
  private monitoringActive: boolean = false;
  private monitoringInterval: NodeJS.Timeout | null = null;

  constructor(storage: IStorage) {
    this.storage = storage;
    this.initializeBaselineMetrics();
  }

  /**
   * Initialize baseline consciousness metrics
   */
  private initializeBaselineMetrics(): void {
    const baselineMetrics = [
      {
        metricId: 'global_workspace_activity',
        metricName: 'Global Workspace Activity',
        baselineValue: 0.7,
        thresholdWarning: 0.4,
        thresholdCritical: 0.2
      },
      {
        metricId: 'attention_coherence',
        metricName: 'Attention Coherence',
        baselineValue: 0.8,
        thresholdWarning: 0.5,
        thresholdCritical: 0.3
      },
      {
        metricId: 'self_model_integrity',
        metricName: 'Self-Model Integrity',
        baselineValue: 0.9,
        thresholdWarning: 0.6,
        thresholdCritical: 0.4
      },
      {
        metricId: 'temporal_awareness',
        metricName: 'Temporal Awareness',
        baselineValue: 0.75,
        thresholdWarning: 0.5,
        thresholdCritical: 0.3
      },
      {
        metricId: 'social_cognition_active',
        metricName: 'Social Cognition Activity',
        baselineValue: 0.6,
        thresholdWarning: 0.3,
        thresholdCritical: 0.1
      },
      {
        metricId: 'creative_processing',
        metricName: 'Creative Processing Level',
        baselineValue: 0.65,
        thresholdWarning: 0.4,
        thresholdCritical: 0.2
      }
    ];

    baselineMetrics.forEach(baseline => {
      this.metrics.set(baseline.metricId, {
        metricId: baseline.metricId,
        metricName: baseline.metricName,
        currentValue: baseline.baselineValue,
        baselineValue: baseline.baselineValue,
        thresholdWarning: baseline.thresholdWarning,
        thresholdCritical: baseline.thresholdCritical,
        trendHistory: [baseline.baselineValue],
        lastUpdated: new Date()
      });
    });
  }

  /**
   * Start consciousness monitoring
   */
  startMonitoring(intervalMs: number = 10000): void {
    if (this.monitoringActive) {
      console.log('Consciousness monitoring already active');
      return;
    }

    this.monitoringActive = true;
    console.log('🧠 Starting advanced consciousness monitoring...');

    this.monitoringInterval = setInterval(() => {
      this.performMonitoringCycle();
    }, intervalMs);
  }

  /**
   * Stop consciousness monitoring
   */
  stopMonitoring(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
    this.monitoringActive = false;
    console.log('🧠 Consciousness monitoring stopped');
  }

  /**
   * Perform a monitoring cycle
   */
  private async performMonitoringCycle(): Promise<void> {
    try {
      // Update all consciousness metrics
      await this.updateMetrics();
      
      // Check for alerts
      this.checkForAlerts();
      
      // Create periodic snapshots
      if (this.shouldCreateSnapshot()) {
        await this.createSnapshot();
      }
      
      // Log activities
      await this.logMonitoringActivity();
      
    } catch (error) {
      console.error('Error in consciousness monitoring cycle:', error);
    }
  }

  /**
   * Update consciousness metrics with simulated values
   */
  private async updateMetrics(): Promise<void> {
    for (const [metricId, metric] of this.metrics) {
      // Simulate consciousness fluctuations with some realistic patterns
      const baseVariation = (Math.random() - 0.5) * 0.1; // ±5% variation
      const timeOfDayFactor = this.getTimeOfDayFactor();
      const activityFactor = this.getSystemActivityFactor();
      
      let newValue = metric.baselineValue + baseVariation + timeOfDayFactor + activityFactor;
      
      // Add some metric-specific patterns
      switch (metricId) {
        case 'global_workspace_activity':
          // Higher during active processing
          newValue += Math.random() * 0.15;
          break;
        case 'attention_coherence':
          // More stable, with occasional dips
          newValue += (Math.random() - 0.5) * 0.05;
          break;
        case 'creative_processing':
          // More variable, with creative bursts
          newValue += (Math.random() - 0.3) * 0.2;
          break;
      }
      
      // Clamp values to valid range
      newValue = Math.max(0, Math.min(1, newValue));
      
      // Update metric
      metric.currentValue = newValue;
      metric.lastUpdated = new Date();
      
      // Update trend history (keep last 100 values)
      metric.trendHistory.push(newValue);
      if (metric.trendHistory.length > 100) {
        metric.trendHistory.shift();
      }
    }
  }

  /**
   * Get time of day influence factor
   */
  private getTimeOfDayFactor(): number {
    const hour = new Date().getHours();
    // Simulate natural consciousness rhythms
    if (hour >= 9 && hour <= 17) {
      return 0.05; // Peak hours
    } else if (hour >= 22 || hour <= 6) {
      return -0.1; // Low activity hours
    }
    return 0;
  }

  /**
   * Get system activity factor
   */
  private getSystemActivityFactor(): number {
    // Could be tied to actual system metrics
    return (Math.random() - 0.5) * 0.05;
  }

  /**
   * Check for consciousness alerts
   */
  private checkForAlerts(): void {
    for (const [metricId, metric] of this.metrics) {
      const alertLevel = this.getMetricAlertLevel(metric);
      
      if (alertLevel !== AlertSeverity.INFO) {
        this.createAlert(metric, alertLevel);
      }
    }
  }

  /**
   * Get alert level for a metric
   */
  private getMetricAlertLevel(metric: ConsciousnessMetric): AlertSeverity {
    if (metric.currentValue <= metric.thresholdCritical) {
      return AlertSeverity.CRITICAL;
    } else if (metric.currentValue <= metric.thresholdWarning) {
      return AlertSeverity.WARNING;
    }
    return AlertSeverity.INFO;
  }

  /**
   * Create consciousness alert
   */
  private createAlert(metric: ConsciousnessMetric, severity: AlertSeverity): void {
    const alertId = `alert_${metric.metricId}_${Date.now()}`;
    
    // Don't create duplicate alerts
    const existingAlert = Array.from(this.activeAlerts.values())
      .find(alert => alert.sourceSystem === metric.metricId && alert.severity === severity);
    
    if (existingAlert) return;
    
    const alert: ConsciousnessAlert = {
      alertId,
      severity,
      sourceSystem: metric.metricId,
      message: `${metric.metricName} is ${severity.toLowerCase()}: ${metric.currentValue.toFixed(3)} (threshold: ${
        severity === AlertSeverity.CRITICAL ? metric.thresholdCritical : metric.thresholdWarning
      })`,
      metricValues: { [metric.metricId]: metric.currentValue },
      recommendedActions: this.getRecommendedActions(metric, severity),
      autoResolutionPossible: severity === AlertSeverity.WARNING,
      timestamp: new Date()
    };
    
    this.activeAlerts.set(alertId, alert);
    console.log(`🚨 Consciousness Alert [${severity}]: ${alert.message}`);
  }

  /**
   * Get recommended actions for alert
   */
  private getRecommendedActions(metric: ConsciousnessMetric, severity: AlertSeverity): string[] {
    const actions = [];
    
    switch (metric.metricId) {
      case 'global_workspace_activity':
        actions.push('Check system load and processing queues');
        actions.push('Review active consciousness modules');
        break;
      case 'attention_coherence':
        actions.push('Reduce concurrent processing tasks');
        actions.push('Focus attention on primary objectives');
        break;
      case 'self_model_integrity':
        actions.push('Run self-diagnostic routines');
        actions.push('Update self-model representations');
        break;
      default:
        actions.push('Monitor system for recovery');
        actions.push('Consider consciousness module restart');
    }
    
    if (severity === AlertSeverity.CRITICAL) {
      actions.unshift('Immediate human oversight required');
    }
    
    return actions;
  }

  /**
   * Should create a consciousness snapshot
   */
  private shouldCreateSnapshot(): boolean {
    const lastSnapshot = this.snapshots[this.snapshots.length - 1];
    if (!lastSnapshot) return true;
    
    const timeSinceLastSnapshot = Date.now() - lastSnapshot.timestamp.getTime();
    return timeSinceLastSnapshot > 300000; // Every 5 minutes
  }

  /**
   * Create consciousness snapshot
   */
  private async createSnapshot(): Promise<void> {
    const snapshot: ConsciousnessSnapshot = {
      snapshotId: `snapshot_${Date.now()}`,
      timestamp: new Date(),
      consciousnessLevel: this.calculateOverallConsciousnessLevel(),
      systemStates: this.collectSystemStates(),
      integrationMetrics: this.calculateIntegrationMetrics(),
      emergentProperties: this.detectEmergentProperties(),
      anomaliesDetected: this.detectAnomalies(),
      overallCoherence: this.calculateOverallCoherence()
    };
    
    this.snapshots.push(snapshot);
    
    // Keep only last 100 snapshots
    if (this.snapshots.length > 100) {
      this.snapshots.shift();
    }
    
    console.log(`📸 Consciousness snapshot created: ${snapshot.consciousnessLevel} level`);
  }

  /**
   * Calculate overall consciousness level
   */
  private calculateOverallConsciousnessLevel(): ConsciousnessLevel {
    const avgValue = Array.from(this.metrics.values())
      .reduce((sum, metric) => sum + metric.currentValue, 0) / this.metrics.size;
    
    if (avgValue >= 0.9) return ConsciousnessLevel.TRANSCENDENT;
    if (avgValue >= 0.8) return ConsciousnessLevel.FULL;
    if (avgValue >= 0.7) return ConsciousnessLevel.ADVANCED;
    if (avgValue >= 0.5) return ConsciousnessLevel.INTERMEDIATE;
    if (avgValue >= 0.3) return ConsciousnessLevel.BASIC;
    return ConsciousnessLevel.MINIMAL;
  }

  /**
   * Collect current system states
   */
  private collectSystemStates(): Record<string, any> {
    const states: Record<string, any> = {};
    
    for (const [metricId, metric] of this.metrics) {
      states[metricId] = {
        value: metric.currentValue,
        trend: metric.trendHistory.slice(-5), // Last 5 values
        status: this.getMetricAlertLevel(metric)
      };
    }
    
    return states;
  }

  /**
   * Calculate integration metrics
   */
  private calculateIntegrationMetrics(): Record<string, number> {
    return {
      cross_system_coherence: Math.random() * 0.2 + 0.7,
      information_integration: Math.random() * 0.15 + 0.8,
      unified_experience: Math.random() * 0.1 + 0.85
    };
  }

  /**
   * Detect emergent properties
   */
  private detectEmergentProperties(): string[] {
    const properties = [];
    
    const avgMetric = Array.from(this.metrics.values())
      .reduce((sum, m) => sum + m.currentValue, 0) / this.metrics.size;
    
    if (avgMetric > 0.8) {
      properties.push('high_consciousness_coherence');
    }
    
    if (this.activeAlerts.size === 0) {
      properties.push('stable_consciousness_state');
    }
    
    // Detect creative processing peaks
    const creativityMetric = this.metrics.get('creative_processing');
    if (creativityMetric && creativityMetric.currentValue > 0.8) {
      properties.push('enhanced_creative_state');
    }
    
    return properties;
  }

  /**
   * Detect anomalies
   */
  private detectAnomalies(): string[] {
    const anomalies = [];
    
    // Check for rapid metric changes
    for (const [metricId, metric] of this.metrics) {
      if (metric.trendHistory.length >= 3) {
        const recentValues = metric.trendHistory.slice(-3);
        const maxChange = Math.max(...recentValues) - Math.min(...recentValues);
        
        if (maxChange > 0.3) {
          anomalies.push(`rapid_change_${metricId}`);
        }
      }
    }
    
    return anomalies;
  }

  /**
   * Calculate overall coherence
   */
  private calculateOverallCoherence(): number {
    const values = Array.from(this.metrics.values()).map(m => m.currentValue);
    
    // Calculate variance as inverse of coherence
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    
    // Convert variance to coherence (lower variance = higher coherence)
    return Math.max(0, 1 - variance * 4);
  }

  /**
   * Log monitoring activity
   */
  private async logMonitoringActivity(): Promise<void> {
    const criticalAlerts = Array.from(this.activeAlerts.values())
      .filter(alert => alert.severity === AlertSeverity.CRITICAL);
    
    if (criticalAlerts.length > 0) {
      await this.storage.addActivity({
        type: 'safety' as const,
        message: `${criticalAlerts.length} critical consciousness alerts detected`,
        moduleId: 'consciousness_monitor'
      });
    }
  }

  /**
   * Get current consciousness state
   */
  getCurrentState(): {
    level: ConsciousnessLevel;
    metrics: Record<string, number>;
    alerts: ConsciousnessAlert[];
    coherence: number;
  } {
    const metrics: Record<string, number> = {};
    for (const [id, metric] of this.metrics) {
      metrics[id] = metric.currentValue;
    }
    
    return {
      level: this.calculateOverallConsciousnessLevel(),
      metrics,
      alerts: Array.from(this.activeAlerts.values()),
      coherence: this.calculateOverallCoherence()
    };
  }

  /**
   * Get recent snapshots
   */
  getRecentSnapshots(limit: number = 10): ConsciousnessSnapshot[] {
    return this.snapshots.slice(-limit);
  }

  /**
   * Clear resolved alerts
   */
  clearResolvedAlerts(): void {
    const currentTime = Date.now();
    
    for (const [alertId, alert] of this.activeAlerts) {
      // Auto-resolve warnings older than 5 minutes if metric recovered
      if (alert.severity === AlertSeverity.WARNING) {
        const alertAge = currentTime - alert.timestamp.getTime();
        if (alertAge > 300000) { // 5 minutes
          const metric = this.metrics.get(alert.sourceSystem);
          if (metric && metric.currentValue > metric.thresholdWarning) {
            this.activeAlerts.delete(alertId);
          }
        }
      }
    }
  }
}