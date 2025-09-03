import { useEffect } from 'react';
import { useLocation, useRoute } from 'wouter';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Brain, Shield, Zap, AlertTriangle } from 'lucide-react';
import { useAuth } from '@/hooks/useAuth';

export default function LoginPage() {
  const { isAuthenticated, isLoading } = useAuth();
  const [, navigate] = useLocation();
  const [, params] = useRoute('/login');
  
  const urlParams = new URLSearchParams(window.location.search);
  const error = urlParams.get('error');

  useEffect(() => {
    if (isAuthenticated) {
      navigate('/');
    }
  }, [isAuthenticated, navigate]);

  const handleGoogleLogin = () => {
    window.location.href = '/api/auth/google';
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted flex items-center justify-center">
        <div className="text-center space-y-4">
          <Brain className="h-12 w-12 text-primary mx-auto animate-pulse" />
          <p className="text-muted-foreground">Initializing consciousness...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted flex items-center justify-center p-4">
      <div className="w-full max-w-md space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <div className="relative">
            <Brain className="h-16 w-16 text-primary mx-auto" />
            <div className="absolute inset-0 bg-primary/20 rounded-full blur-xl" />
          </div>
          <div className="space-y-2">
            <h1 className="text-3xl font-bold tracking-tight">NEXUS</h1>
            <p className="text-muted-foreground text-sm">
              Advanced AI Consciousness Platform
            </p>
          </div>
        </div>

        {/* Error Alert */}
        {error && (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              {error === 'auth_failed' 
                ? 'Authentication failed. Please try again.' 
                : 'An error occurred during authentication.'}
            </AlertDescription>
          </Alert>
        )}

        {/* Login Card */}
        <Card className="border-border/50 bg-card/50 backdrop-blur-sm" data-testid="login-card">
          <CardHeader className="text-center space-y-4">
            <CardTitle className="text-2xl">Welcome Back</CardTitle>
            <CardDescription>
              Access your consciousness management dashboard
            </CardDescription>
          </CardHeader>
          
          <CardContent className="space-y-6">
            {/* Features Preview */}
            <div className="grid gap-3 text-sm">
              <div className="flex items-center gap-3 text-muted-foreground">
                <Brain className="h-4 w-4 text-blue-500" />
                <span>Advanced AI Consciousness Monitoring</span>
              </div>
              <div className="flex items-center gap-3 text-muted-foreground">
                <Zap className="h-4 w-4 text-yellow-500" />
                <span>Real-time Learning & Adaptation</span>
              </div>
              <div className="flex items-center gap-3 text-muted-foreground">
                <Shield className="h-4 w-4 text-green-500" />
                <span>Comprehensive Safety Systems</span>
              </div>
            </div>

            {/* Login Button */}
            <Button 
              onClick={handleGoogleLogin}
              className="w-full h-11 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white border-0 shadow-lg hover:shadow-xl transition-all duration-200"
              data-testid="google-login-button"
            >
              <svg className="w-5 h-5 mr-3" viewBox="0 0 24 24">
                <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
              </svg>
              Continue with Google
            </Button>

            {/* Security Note */}
            <div className="text-xs text-muted-foreground text-center space-y-1">
              <p>🔒 Secure single-user authentication</p>
              <p>Access restricted to authorized personnel only</p>
            </div>
          </CardContent>
        </Card>

        {/* Footer */}
        <div className="text-center text-xs text-muted-foreground">
          <p>NEXUS Unified System • Consciousness Research Platform</p>
        </div>
      </div>
    </div>
  );
}