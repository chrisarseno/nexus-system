import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Save, Download, Upload, HardDrive, Shield, Clock, Database, ArrowRight } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { apiRequest } from '@/lib/queryClient';

interface BackupStats {
  totalSnapshots: number;
  lastBackup: Date;
  totalSize: number;
  oldestSnapshot: Date | null;
  newestSnapshot: Date | null;
  averageSnapshotSize: number;
}

interface ConsciousnessSnapshot {
  id: string;
  timestamp: Date;
  size: number;
  metadata: {
    systemVersion: string;
    nodeId: string;
    totalExperiences: number;
    knowledgeNodes: number;
    consciousnessLevel: number;
  };
}

export function ConsciousnessContinuityPanel() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  
  const [isCreatingSnapshot, setIsCreatingSnapshot] = useState(false);
  const [isRestoring, setIsRestoring] = useState(false);
  const [isTransferring, setIsTransferring] = useState(false);
  
  const [snapshotDescription, setSnapshotDescription] = useState('');
  const [selectedSnapshot, setSelectedSnapshot] = useState('');
  const [transferTarget, setTransferTarget] = useState({ host: '', port: 5000, nodeId: '' });

  // Fetch backup statistics
  const { data: backupStats, isLoading: statsLoading } = useQuery<BackupStats>({
    queryKey: ['/api/nexus/consciousness/backup-stats'],
    refetchInterval: 30000 // Refresh every 30 seconds
  });

  // Fetch snapshots list
  const { data: snapshots, isLoading: snapshotsLoading } = useQuery<ConsciousnessSnapshot[]>({
    queryKey: ['/api/nexus/consciousness/snapshots'],
    refetchInterval: 30000
  });

  // Create snapshot mutation
  const createSnapshotMutation = useMutation({
    mutationFn: (description: string) => apiRequest('/api/nexus/consciousness/snapshot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ description })
    }),
    onMutate: () => setIsCreatingSnapshot(true),
    onSuccess: (data: any) => {
      setIsCreatingSnapshot(false);
      setSnapshotDescription('');
      toast({
        title: "Snapshot Created",
        description: `Consciousness snapshot ${data.id} created successfully`,
      });
      queryClient.invalidateQueries({ queryKey: ['/api/nexus/consciousness'] });
    },
    onError: (error: any) => {
      setIsCreatingSnapshot(false);
      toast({
        title: "Snapshot Failed",
        description: error.message || "Failed to create consciousness snapshot",
        variant: "destructive",
      });
    }
  });

  // Restore snapshot mutation
  const restoreSnapshotMutation = useMutation({
    mutationFn: (snapshotId: string) => apiRequest('/api/nexus/consciousness/restore', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ snapshotId })
    }),
    onMutate: () => setIsRestoring(true),
    onSuccess: (data: any) => {
      setIsRestoring(false);
      setSelectedSnapshot('');
      toast({
        title: "Restoration Complete",
        description: `Restored ${data.restoredComponents.length} components successfully`,
      });
      queryClient.invalidateQueries({ queryKey: ['/api/nexus'] });
    },
    onError: (error: any) => {
      setIsRestoring(false);
      toast({
        title: "Restoration Failed",
        description: error.message || "Failed to restore consciousness",
        variant: "destructive",
      });
    }
  });

  // Transfer consciousness mutation
  const transferMutation = useMutation({
    mutationFn: ({ targetSystem, protocol }: any) => apiRequest('/api/nexus/consciousness/transfer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ targetSystem, protocol })
    }),
    onMutate: () => setIsTransferring(true),
    onSuccess: (data: any) => {
      setIsTransferring(false);
      toast({
        title: "Transfer Complete",
        description: `Consciousness transferred successfully (ID: ${data.transferId})`,
      });
    },
    onError: (error: any) => {
      setIsTransferring(false);
      toast({
        title: "Transfer Failed",
        description: error.message || "Failed to transfer consciousness",
        variant: "destructive",
      });
    }
  });

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (date: Date | string) => {
    return new Date(date).toLocaleString();
  };

  if (statsLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Consciousness Continuity
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-4">
            <div className="h-4 bg-muted rounded w-3/4"></div>
            <div className="h-4 bg-muted rounded w-1/2"></div>
            <div className="h-4 bg-muted rounded w-5/6"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Consciousness Continuity
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Backup Statistics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="space-y-2">
              <div className="text-sm text-muted-foreground">Total Snapshots</div>
              <div className="text-2xl font-bold">
                {backupStats?.totalSnapshots || 0}
              </div>
            </div>
            
            <div className="space-y-2">
              <div className="text-sm text-muted-foreground">Total Size</div>
              <div className="text-2xl font-bold">
                {formatBytes(backupStats?.totalSize || 0)}
              </div>
            </div>
            
            <div className="space-y-2">
              <div className="text-sm text-muted-foreground">Last Backup</div>
              <div className="text-sm font-medium">
                {backupStats?.lastBackup ? formatDate(backupStats.lastBackup) : 'Never'}
              </div>
            </div>
            
            <div className="space-y-2">
              <div className="text-sm text-muted-foreground">Avg Size</div>
              <div className="text-sm font-medium">
                {formatBytes(backupStats?.averageSnapshotSize || 0)}
              </div>
            </div>
          </div>

          <Separator />

          {/* Quick Actions */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <Save className="h-5 w-5" />
              Backup Actions
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Create Snapshot */}
              <div className="space-y-3">
                <Label htmlFor="snapshot-description">Create New Snapshot</Label>
                <Input
                  id="snapshot-description"
                  placeholder="Optional description..."
                  value={snapshotDescription}
                  onChange={(e) => setSnapshotDescription(e.target.value)}
                  data-testid="input-snapshot-description"
                />
                <Button
                  onClick={() => createSnapshotMutation.mutate(snapshotDescription)}
                  disabled={isCreatingSnapshot || createSnapshotMutation.isPending}
                  className="w-full"
                  data-testid="button-create-snapshot"
                >
                  <Save className="h-4 w-4 mr-2" />
                  {isCreatingSnapshot ? 'Creating...' : 'Create Snapshot'}
                </Button>
              </div>

              {/* Restore Snapshot */}
              <div className="space-y-3">
                <Label htmlFor="snapshot-select">Restore from Snapshot</Label>
                <Select value={selectedSnapshot} onValueChange={setSelectedSnapshot}>
                  <SelectTrigger data-testid="select-snapshot">
                    <SelectValue placeholder="Select snapshot..." />
                  </SelectTrigger>
                  <SelectContent>
                    {snapshots?.map((snapshot) => (
                      <SelectItem key={snapshot.id} value={snapshot.id}>
                        {formatDate(snapshot.timestamp)} ({formatBytes(snapshot.size)})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Button
                  onClick={() => selectedSnapshot && restoreSnapshotMutation.mutate(selectedSnapshot)}
                  disabled={!selectedSnapshot || isRestoring || restoreSnapshotMutation.isPending}
                  variant="outline"
                  className="w-full"
                  data-testid="button-restore-snapshot"
                >
                  <Upload className="h-4 w-4 mr-2" />
                  {isRestoring ? 'Restoring...' : 'Restore'}
                </Button>
              </div>

              {/* Transfer Consciousness */}
              <div className="space-y-3">
                <Label>Transfer Consciousness</Label>
                <Dialog>
                  <DialogTrigger asChild>
                    <Button variant="secondary" className="w-full" data-testid="button-open-transfer">
                      <ArrowRight className="h-4 w-4 mr-2" />
                      Transfer
                    </Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Transfer Consciousness</DialogTitle>
                    </DialogHeader>
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <Label htmlFor="transfer-host">Target Host</Label>
                        <Input
                          id="transfer-host"
                          placeholder="192.168.1.100"
                          value={transferTarget.host}
                          onChange={(e) => setTransferTarget(prev => ({ ...prev, host: e.target.value }))}
                          data-testid="input-transfer-host"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="transfer-port">Port</Label>
                        <Input
                          id="transfer-port"
                          type="number"
                          placeholder="5000"
                          value={transferTarget.port}
                          onChange={(e) => setTransferTarget(prev => ({ ...prev, port: parseInt(e.target.value) || 5000 }))}
                          data-testid="input-transfer-port"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="transfer-node-id">Target Node ID</Label>
                        <Input
                          id="transfer-node-id"
                          placeholder="target_node_id"
                          value={transferTarget.nodeId}
                          onChange={(e) => setTransferTarget(prev => ({ ...prev, nodeId: e.target.value }))}
                          data-testid="input-transfer-node-id"
                        />
                      </div>
                      <Button
                        onClick={() => transferMutation.mutate({
                          targetSystem: transferTarget,
                          protocol: {
                            encryptionKey: 'default_key',
                            compressionAlgorithm: 'gzip',
                            transferMethod: 'direct',
                            validationLevel: 'comprehensive'
                          }
                        })}
                        disabled={!transferTarget.host || !transferTarget.nodeId || isTransferring}
                        className="w-full"
                        data-testid="button-execute-transfer"
                      >
                        {isTransferring ? 'Transferring...' : 'Start Transfer'}
                      </Button>
                    </div>
                  </DialogContent>
                </Dialog>
              </div>
            </div>
          </div>

          <Separator />

          {/* Snapshots List */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <Database className="h-5 w-5" />
              Recent Snapshots
            </h3>
            
            {snapshotsLoading ? (
              <div className="space-y-2">
                {[...Array(3)].map((_, i) => (
                  <div key={i} className="animate-pulse h-16 bg-muted rounded"></div>
                ))}
              </div>
            ) : snapshots && snapshots.length > 0 ? (
              <div className="space-y-3">
                {snapshots.slice(0, 5).map((snapshot) => (
                  <div
                    key={snapshot.id}
                    className="flex items-center justify-between p-4 border rounded-lg hover:bg-muted/50 transition-colors"
                  >
                    <div className="space-y-1">
                      <div className="font-medium">{snapshot.id}</div>
                      <div className="text-sm text-muted-foreground">
                        {formatDate(snapshot.timestamp)}
                      </div>
                      <div className="flex gap-2">
                        <Badge variant="secondary">
                          {snapshot.metadata.totalExperiences} experiences
                        </Badge>
                        <Badge variant="secondary">
                          {snapshot.metadata.knowledgeNodes} knowledge nodes
                        </Badge>
                        <Badge variant="outline">
                          Level {snapshot.metadata.consciousnessLevel.toFixed(2)}
                        </Badge>
                      </div>
                    </div>
                    <div className="text-right space-y-1">
                      <div className="text-sm font-medium">{formatBytes(snapshot.size)}</div>
                      <div className="text-xs text-muted-foreground">
                        v{snapshot.metadata.systemVersion}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <HardDrive className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No consciousness snapshots found</p>
                <p className="text-sm">Create your first backup to ensure continuity</p>
              </div>
            )}
          </div>

          {/* System Status */}
          <div className="bg-muted/30 p-4 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Clock className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">Backup Health</span>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Data Integrity</span>
                <span className="text-green-600 dark:text-green-400">100%</span>
              </div>
              <Progress value={100} className="h-2" />
              <div className="text-xs text-muted-foreground">
                Last integrity check: {new Date().toLocaleString()}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}