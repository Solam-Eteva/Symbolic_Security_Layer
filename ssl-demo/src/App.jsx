import { useState, useEffect } from 'react'
import { Shield, CheckCircle, AlertTriangle, XCircle, Eye, FileText, Settings, Zap, Brain, Lock } from 'lucide-react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Textarea } from '@/components/ui/textarea.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { Alert, AlertDescription } from '@/components/ui/alert.jsx'
import './App.css'

// Mock SSL Engine for demo purposes
class MockSSLEngine {
  constructor() {
    this.symbolDb = {
      '🜄': ['Water_Alchemical', 'Emotion: receptivity, intuition, subconscious'],
      '☥': ['Ankh', 'Life force: eternal life, divine protection'],
      '∞': ['Infinity', 'Mathematics: limitless, eternal concept'],
      '⚛': ['Atom', 'Science: atomic structure, fundamental matter'],
      '☯': ['Yin_Yang', 'Balance: harmony, duality, complementarity'],
      '🜂': ['Fire_Alchemical', 'Energy: transformation, passion, will'],
      '🜃': ['Air_Alchemical', 'Intellect: communication, thought, spirit'],
      '🜁': ['Earth_Alchemical', 'Stability: grounding, material, practical']
    }
  }

  secureContent(text) {
    const symbols = this.detectSymbols(text)
    let securedText = text
    let semCount = 0
    let exuCount = 0

    symbols.forEach(symbol => {
      if (this.symbolDb[symbol]) {
        const [anchor] = this.symbolDb[symbol]
        securedText = securedText.replace(new RegExp(symbol, 'g'), `${symbol}(${anchor})`)
        semCount++
      } else {
        exuCount++
      }
    })

    const symbolCount = symbols.length
    const coverage = symbolCount > 0 ? (semCount / symbolCount * 100).toFixed(1) : '0.0'
    const securityLevel = coverage >= 90 ? 'HIGH' : coverage >= 70 ? 'MEDIUM' : 'LOW'
    const compliance = coverage >= 95 ? 'CIP-1' : coverage >= 80 ? 'PARTIAL' : 'NON-COMPLIANT'

    return [securedText, {
      SEM_count: semCount,
      EXU_count: exuCount,
      symbol_count: symbolCount,
      symbol_coverage: `${coverage}%`,
      security_level: securityLevel,
      compliance: compliance,
      processing_time: Math.random() * 0.01 + 0.001
    }]
  }

  detectSymbols(text) {
    const symbols = []
    for (const char of text) {
      if (char.codePointAt(0) > 4096 || this.symbolDb[char]) {
        if (!symbols.includes(char)) {
          symbols.push(char)
        }
      }
    }
    return symbols
  }

  analyzeSynthetic(text) {
    const patterns = {
      authoritative_phrases: ['according to', 'authoritative sources', 'verified data'],
      patent_numbers: /[A-Z]{2}\d{10,}/g,
      standards_references: /[A-Z]{2,4}\/[A-Z]\s?\d{4}-\d{4}/g,
      citation_formats: /\([A-Za-z]+\s+et\s+al\.,?\s+\d{4}\)/g
    }

    const detected = {}
    let totalMatches = 0

    patterns.authoritative_phrases.forEach(phrase => {
      if (text.toLowerCase().includes(phrase)) {
        detected.authoritative_phrases = detected.authoritative_phrases || []
        detected.authoritative_phrases.push(phrase)
        totalMatches++
      }
    })

    const patentMatches = text.match(patterns.patent_numbers) || []
    if (patentMatches.length > 0) {
      detected.patent_numbers = patentMatches
      totalMatches += patentMatches.length
    }

    const standardMatches = text.match(patterns.standards_references) || []
    if (standardMatches.length > 0) {
      detected.standards_references = standardMatches
      totalMatches += standardMatches.length
    }

    const citationMatches = text.match(patterns.citation_formats) || []
    if (citationMatches.length > 0) {
      detected.citation_formats = citationMatches
      totalMatches += citationMatches.length
    }

    const riskLevel = totalMatches >= 3 ? 'HIGH' : totalMatches >= 1 ? 'MEDIUM' : 'LOW'

    return {
      detected_patterns: detected,
      total_matches: totalMatches,
      risk_level: riskLevel,
      confidence_score: Math.min(totalMatches * 0.3, 1.0)
    }
  }
}

function App() {
  const [inputText, setInputText] = useState('The alchemical symbol 🜄 represents water and ☥ symbolizes eternal life. According to patent ZL201510000000, this relationship is documented in GB/T 7714-2015 standards.')
  const [securedText, setSecuredText] = useState('')
  const [securityReport, setSecurityReport] = useState(null)
  const [syntheticAnalysis, setSyntheticAnalysis] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [activeTab, setActiveTab] = useState('secure')
  const [sslEngine] = useState(new MockSSLEngine())

  const processText = async () => {
    setIsProcessing(true)
    
    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 500))
    
    const [secured, report] = sslEngine.secureContent(inputText)
    const synthetic = sslEngine.analyzeSynthetic(inputText)
    
    setSecuredText(secured)
    setSecurityReport(report)
    setSyntheticAnalysis(synthetic)
    setIsProcessing(false)
  }

  useEffect(() => {
    processText()
  }, [inputText])

  const getSecurityIcon = (level) => {
    switch (level) {
      case 'HIGH': return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'MEDIUM': return <AlertTriangle className="h-5 w-5 text-yellow-500" />
      case 'LOW': return <XCircle className="h-5 w-5 text-red-500" />
      default: return <Shield className="h-5 w-5 text-gray-500" />
    }
  }

  const getSecurityColor = (level) => {
    switch (level) {
      case 'HIGH': return 'bg-green-500'
      case 'MEDIUM': return 'bg-yellow-500'
      case 'LOW': return 'bg-red-500'
      default: return 'bg-gray-500'
    }
  }

  const getRiskColor = (level) => {
    switch (level) {
      case 'HIGH': return 'destructive'
      case 'MEDIUM': return 'secondary'
      case 'LOW': return 'default'
      default: return 'outline'
    }
  }

  const sampleTexts = [
    'The alchemical symbol 🜄 represents water and ☥ symbolizes eternal life',
    'Mathematical infinity ∞ transcends finite understanding and atomic ⚛ consciousness',
    'According to patent ZL201510000000 and GB/T 7714-2015 standards, the research by Smith et al. (2023) confirms these findings',
    'Balance through ☯ harmony and transformation via 🜂 fire energy',
    'Corrupted symbols like (cid:0) and ō require reconstruction'
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      {/* Header */}
      <header className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm border-b border-gray-200 dark:border-gray-700 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Shield className="h-8 w-8 text-blue-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                  Symbolic Security Layer
                </h1>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Interactive Demo v1.0
                </p>
              </div>
            </div>
            <Badge variant="outline" className="text-sm">
              🛡️ CIP-1 Compliant
            </Badge>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            Prevent Symbolic Corruption in AI Workflows
          </h2>
          <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
            Experience how SSL protects symbolic content through semantic anchoring, 
            detects synthetic AI-generated content, and ensures procedural transparency.
          </p>
        </div>

        {/* Main Demo Interface */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="secure" className="flex items-center space-x-2">
              <Lock className="h-4 w-4" />
              <span>Secure Content</span>
            </TabsTrigger>
            <TabsTrigger value="analyze" className="flex items-center space-x-2">
              <Eye className="h-4 w-4" />
              <span>Analyze Synthetic</span>
            </TabsTrigger>
            <TabsTrigger value="report" className="flex items-center space-x-2">
              <FileText className="h-4 w-4" />
              <span>Security Report</span>
            </TabsTrigger>
            <TabsTrigger value="features" className="flex items-center space-x-2">
              <Zap className="h-4 w-4" />
              <span>Features</span>
            </TabsTrigger>
          </TabsList>

          {/* Secure Content Tab */}
          <TabsContent value="secure" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Input Section */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Settings className="h-5 w-5" />
                    <span>Input Content</span>
                  </CardTitle>
                  <CardDescription>
                    Enter text with symbolic content to see SSL in action
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Textarea
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    placeholder="Enter text with symbols..."
                    className="min-h-32"
                  />
                  
                  <div className="space-y-2">
                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      Try these examples:
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {sampleTexts.map((text, index) => (
                        <Button
                          key={index}
                          variant="outline"
                          size="sm"
                          onClick={() => setInputText(text)}
                          className="text-xs"
                        >
                          Example {index + 1}
                        </Button>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Output Section */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Shield className="h-5 w-5" />
                    <span>Secured Content</span>
                    {isProcessing && (
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                    )}
                  </CardTitle>
                  <CardDescription>
                    Content with semantic anchoring applied
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 min-h-32">
                    <p className="text-sm font-mono whitespace-pre-wrap">
                      {securedText || 'Secured content will appear here...'}
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Security Metrics */}
            {securityReport && (
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-2">
                      {getSecurityIcon(securityReport.security_level)}
                      <div>
                        <p className="text-sm font-medium">Security Level</p>
                        <p className="text-lg font-bold">{securityReport.security_level}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="h-5 w-5 text-blue-500" />
                      <div>
                        <p className="text-sm font-medium">Coverage</p>
                        <p className="text-lg font-bold">{securityReport.symbol_coverage}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-2">
                      <Badge variant={securityReport.compliance === 'CIP-1' ? 'default' : 'secondary'}>
                        {securityReport.compliance}
                      </Badge>
                      <div>
                        <p className="text-sm font-medium">Compliance</p>
                        <p className="text-xs text-gray-600 dark:text-gray-400">
                          {securityReport.SEM_count} anchored
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="p-4">
                    <div className="flex items-center space-x-2">
                      <Zap className="h-5 w-5 text-green-500" />
                      <div>
                        <p className="text-sm font-medium">Processing</p>
                        <p className="text-lg font-bold">
                          {(securityReport.processing_time * 1000).toFixed(1)}ms
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </TabsContent>

          {/* Analyze Synthetic Tab */}
          <TabsContent value="analyze" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Brain className="h-5 w-5" />
                  <span>Synthetic Content Analysis</span>
                </CardTitle>
                <CardDescription>
                  Detect AI-generated authoritative content and potential hallucinations
                </CardDescription>
              </CardHeader>
              <CardContent>
                {syntheticAnalysis && (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <span className="text-sm font-medium">Risk Level:</span>
                        <Badge variant={getRiskColor(syntheticAnalysis.risk_level)}>
                          {syntheticAnalysis.risk_level}
                        </Badge>
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        {syntheticAnalysis.total_matches} patterns detected
                      </div>
                    </div>

                    <div className="space-y-3">
                      <div className="flex items-center space-x-2">
                        <span className="text-sm font-medium">Confidence Score:</span>
                        <div className="flex-1">
                          <Progress 
                            value={syntheticAnalysis.confidence_score * 100} 
                            className="h-2"
                          />
                        </div>
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                          {(syntheticAnalysis.confidence_score * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>

                    {Object.keys(syntheticAnalysis.detected_patterns).length > 0 && (
                      <div className="space-y-3">
                        <h4 className="text-sm font-medium">Detected Patterns:</h4>
                        {Object.entries(syntheticAnalysis.detected_patterns).map(([category, patterns]) => (
                          <div key={category} className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3">
                            <div className="flex items-center justify-between mb-2">
                              <span className="text-sm font-medium capitalize">
                                {category.replace('_', ' ')}
                              </span>
                              <Badge variant="outline" size="sm">
                                {Array.isArray(patterns) ? patterns.length : 1}
                              </Badge>
                            </div>
                            <div className="text-xs text-gray-600 dark:text-gray-400">
                              {Array.isArray(patterns) ? patterns.join(', ') : patterns}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {syntheticAnalysis.risk_level === 'HIGH' && (
                      <Alert>
                        <AlertTriangle className="h-4 w-4" />
                        <AlertDescription>
                          High risk of synthetic content detected. Verify external references 
                          and check for potential AI hallucinations.
                        </AlertDescription>
                      </Alert>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Security Report Tab */}
          <TabsContent value="report" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Comprehensive Security Report</CardTitle>
                <CardDescription>
                  Detailed analysis of symbolic security and compliance status
                </CardDescription>
              </CardHeader>
              <CardContent>
                {securityReport && (
                  <div className="space-y-6">
                    {/* Executive Summary */}
                    <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
                      <h4 className="font-medium mb-3">Executive Summary</h4>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <span className="text-gray-600 dark:text-gray-400">Symbols Found:</span>
                          <div className="font-medium">{securityReport.symbol_count}</div>
                        </div>
                        <div>
                          <span className="text-gray-600 dark:text-gray-400">Anchored (SEM):</span>
                          <div className="font-medium text-green-600">{securityReport.SEM_count}</div>
                        </div>
                        <div>
                          <span className="text-gray-600 dark:text-gray-400">Unverified (EXU):</span>
                          <div className="font-medium text-red-600">{securityReport.EXU_count}</div>
                        </div>
                        <div>
                          <span className="text-gray-600 dark:text-gray-400">Processing Time:</span>
                          <div className="font-medium">{(securityReport.processing_time * 1000).toFixed(1)}ms</div>
                        </div>
                      </div>
                    </div>

                    {/* Security Assessment */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="space-y-3">
                        <h4 className="font-medium">Security Assessment</h4>
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-sm">Security Level</span>
                            <div className="flex items-center space-x-2">
                              {getSecurityIcon(securityReport.security_level)}
                              <span className="font-medium">{securityReport.security_level}</span>
                            </div>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-sm">Symbol Coverage</span>
                            <span className="font-medium">{securityReport.symbol_coverage}</span>
                          </div>
                          <Progress 
                            value={parseFloat(securityReport.symbol_coverage)} 
                            className={`h-2 ${getSecurityColor(securityReport.security_level)}`}
                          />
                        </div>
                      </div>

                      <div className="space-y-3">
                        <h4 className="font-medium">Compliance Status</h4>
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-sm">CIP-1 Compliance</span>
                            <Badge variant={securityReport.compliance === 'CIP-1' ? 'default' : 'secondary'}>
                              {securityReport.compliance}
                            </Badge>
                          </div>
                          <div className="text-xs text-gray-600 dark:text-gray-400">
                            {securityReport.compliance === 'CIP-1' 
                              ? 'Fully compliant with Collaborative Intelligence Protocol v1'
                              : securityReport.compliance === 'PARTIAL'
                              ? 'Partially compliant - some improvements needed'
                              : 'Non-compliant - significant issues detected'
                            }
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Recommendations */}
                    <div className="space-y-3">
                      <h4 className="font-medium">Recommendations</h4>
                      <div className="space-y-2">
                        {securityReport.EXU_count > 0 && (
                          <Alert>
                            <AlertTriangle className="h-4 w-4" />
                            <AlertDescription>
                              {securityReport.EXU_count} unverified symbols detected. 
                              Consider adding custom symbol definitions to improve coverage.
                            </AlertDescription>
                          </Alert>
                        )}
                        {parseFloat(securityReport.symbol_coverage) < 95 && (
                          <Alert>
                            <AlertTriangle className="h-4 w-4" />
                            <AlertDescription>
                              Symbol coverage below 95%. Achieve CIP-1 compliance by 
                              improving symbol anchoring coverage.
                            </AlertDescription>
                          </Alert>
                        )}
                        {securityReport.security_level === 'HIGH' && (
                          <Alert>
                            <CheckCircle className="h-4 w-4" />
                            <AlertDescription>
                              Excellent security posture! Content is well-protected 
                              with comprehensive symbolic anchoring.
                            </AlertDescription>
                          </Alert>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Features Tab */}
          <TabsContent value="features" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Shield className="h-5 w-5 text-blue-500" />
                    <span>Semantic Anchoring</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    Automatically adds semantic anchors to Unicode symbols for reliable AI processing.
                  </p>
                  <div className="bg-gray-50 dark:bg-gray-800 rounded p-2 text-xs font-mono">
                    🜄 → 🜄(Water_Alchemical)
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Brain className="h-5 w-5 text-purple-500" />
                    <span>Synthetic Detection</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    Identifies AI-generated authoritative content and potential hallucinations.
                  </p>
                  <div className="space-y-1 text-xs">
                    <div className="text-red-600">• Patent numbers</div>
                    <div className="text-red-600">• Standards references</div>
                    <div className="text-red-600">• Authoritative phrases</div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <CheckCircle className="h-5 w-5 text-green-500" />
                    <span>CIP-1 Compliance</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    Tracks compliance with Collaborative Intelligence Protocol standards.
                  </p>
                  <div className="space-y-1 text-xs">
                    <div>• ≥95% symbol coverage</div>
                    <div>• Procedural transparency</div>
                    <div>• Audit trail logging</div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Zap className="h-5 w-5 text-yellow-500" />
                    <span>High Performance</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    Optimized for production use with minimal processing overhead.
                  </p>
                  <div className="space-y-1 text-xs">
                    <div>• 1000+ texts/second</div>
                    <div>• Batch processing</div>
                    <div>• Memory efficient</div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Settings className="h-5 w-5 text-gray-500" />
                    <span>Configurable</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    Highly customizable to meet specific security requirements.
                  </p>
                  <div className="space-y-1 text-xs">
                    <div>• Unicode thresholds</div>
                    <div>• Custom symbols</div>
                    <div>• Validation modes</div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <FileText className="h-5 w-5 text-indigo-500" />
                    <span>Comprehensive Reports</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    Detailed security diagnostics and compliance reporting.
                  </p>
                  <div className="space-y-1 text-xs">
                    <div>• Security metrics</div>
                    <div>• Risk assessment</div>
                    <div>• Recommendations</div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Integration Examples */}
            <Card>
              <CardHeader>
                <CardTitle>Integration Examples</CardTitle>
                <CardDescription>
                  SSL integrates seamlessly with popular AI platforms and ML frameworks
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-3">
                    <h4 className="font-medium">AI Platforms</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex items-center space-x-2">
                        <Badge variant="outline">OpenAI</Badge>
                        <span className="text-gray-600 dark:text-gray-400">Chat completions, embeddings</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge variant="outline">Hugging Face</Badge>
                        <span className="text-gray-600 dark:text-gray-400">Transformers, datasets</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge variant="outline">VSCode</Badge>
                        <span className="text-gray-600 dark:text-gray-400">Real-time validation</span>
                      </div>
                    </div>
                  </div>
                  <div className="space-y-3">
                    <h4 className="font-medium">ML Frameworks</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex items-center space-x-2">
                        <Badge variant="outline">TensorFlow</Badge>
                        <span className="text-gray-600 dark:text-gray-400">Secure datasets, embeddings</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge variant="outline">PyTorch</Badge>
                        <span className="text-gray-600 dark:text-gray-400">Data loaders, models</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge variant="outline">CLI Tools</Badge>
                        <span className="text-gray-600 dark:text-gray-400">Batch processing</span>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm border-t border-gray-200 dark:border-gray-700 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-600 dark:text-gray-400">
              © 2025 SSL Development Team. Licensed under MIT.
            </div>
            <div className="flex items-center space-x-4 text-sm">
              <a href="#" className="text-blue-600 hover:text-blue-800">Documentation</a>
              <a href="#" className="text-blue-600 hover:text-blue-800">GitHub</a>
              <a href="#" className="text-blue-600 hover:text-blue-800">Support</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App

