import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as path from 'path';

interface SSLConfig {
    autoSecureOnSave: boolean;
    unicodeThreshold: number;
    semanticAnchoring: boolean;
    strictValidation: boolean;
    showStatusBar: boolean;
    pythonPath: string;
}

interface SecurityReport {
    SEM_count: number;
    EXU_count: number;
    PROV_count: number;
    symbol_count: number;
    anchored_count: number;
    symbol_coverage: string;
    verification_status: string;
    security_level: string;
    compliance: string;
    processing_time?: number;
}

interface SSLResponse {
    secured_content: string;
    validation_report: SecurityReport;
    error?: string;
}

export class SSLExtension {
    private statusBarItem: vscode.StatusBarItem;
    private outputChannel: vscode.OutputChannel;
    private securityPanel: vscode.WebviewPanel | undefined;
    private currentReport: SecurityReport | undefined;

    constructor(private context: vscode.ExtensionContext) {
        this.statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
        this.outputChannel = vscode.window.createOutputChannel('SSL Security');
        this.updateStatusBar('Ready', 'ssl-ready');
    }

    public activate() {
        // Register commands
        this.context.subscriptions.push(
            vscode.commands.registerCommand('ssl.secureDocument', () => this.secureDocument()),
            vscode.commands.registerCommand('ssl.validateSymbols', () => this.validateSymbols()),
            vscode.commands.registerCommand('ssl.showSecurityReport', () => this.showSecurityReport()),
            vscode.commands.registerCommand('ssl.addCustomSymbol', () => this.addCustomSymbol()),
            vscode.commands.registerCommand('ssl.exportReport', () => this.exportReport())
        );

        // Register event handlers
        this.context.subscriptions.push(
            vscode.workspace.onDidSaveTextDocument((document) => this.onDocumentSave(document)),
            vscode.window.onDidChangeActiveTextEditor((editor) => this.onActiveEditorChange(editor))
        );

        // Show status bar if enabled
        const config = this.getConfig();
        if (config.showStatusBar) {
            this.statusBarItem.show();
        }

        this.outputChannel.appendLine('SSL Extension activated');
    }

    public deactivate() {
        this.statusBarItem.dispose();
        this.outputChannel.dispose();
        if (this.securityPanel) {
            this.securityPanel.dispose();
        }
    }

    private getConfig(): SSLConfig {
        const config = vscode.workspace.getConfiguration('ssl');
        return {
            autoSecureOnSave: config.get('autoSecureOnSave', true),
            unicodeThreshold: config.get('unicodeThreshold', 4096),
            semanticAnchoring: config.get('semanticAnchoring', true),
            strictValidation: config.get('strictValidation', false),
            showStatusBar: config.get('showStatusBar', true),
            pythonPath: config.get('pythonPath', 'python3')
        };
    }

    private async secureDocument() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor found');
            return;
        }

        const document = editor.document;
        const selection = editor.selection;
        
        let textToSecure: string;
        let range: vscode.Range;

        if (selection.isEmpty) {
            // Secure entire document
            textToSecure = document.getText();
            range = new vscode.Range(0, 0, document.lineCount - 1, document.lineAt(document.lineCount - 1).text.length);
        } else {
            // Secure selected text
            textToSecure = document.getText(selection);
            range = selection;
        }

        this.updateStatusBar('Securing...', 'ssl-processing');

        try {
            const result = await this.callSSLBackend(textToSecure);
            
            if (result.error) {
                vscode.window.showErrorMessage(`SSL Error: ${result.error}`);
                this.updateStatusBar('Error', 'ssl-error');
                return;
            }

            // Replace text with secured version
            await editor.edit(editBuilder => {
                editBuilder.replace(range, result.secured_content);
            });

            this.currentReport = result.validation_report;
            this.updateStatusBarWithReport(result.validation_report);
            
            vscode.window.showInformationMessage(
                `Document secured! Coverage: ${result.validation_report.symbol_coverage}, ` +
                `Security: ${result.validation_report.security_level}`
            );

        } catch (error) {
            vscode.window.showErrorMessage(`Failed to secure document: ${error}`);
            this.updateStatusBar('Error', 'ssl-error');
        }
    }

    private async validateSymbols() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor found');
            return;
        }

        const text = editor.document.getText();
        this.updateStatusBar('Validating...', 'ssl-processing');

        try {
            const result = await this.callSSLBackend(text, true); // validation only
            
            if (result.error) {
                vscode.window.showErrorMessage(`SSL Error: ${result.error}`);
                this.updateStatusBar('Error', 'ssl-error');
                return;
            }

            this.currentReport = result.validation_report;
            this.updateStatusBarWithReport(result.validation_report);
            
            // Show validation results
            const report = result.validation_report;
            const message = `Validation Complete!\n` +
                `Symbols Found: ${report.symbol_count}\n` +
                `Anchored: ${report.anchored_count}\n` +
                `Coverage: ${report.symbol_coverage}\n` +
                `Security Level: ${report.security_level}\n` +
                `Compliance: ${report.compliance}`;

            vscode.window.showInformationMessage(message, 'Show Details').then(selection => {
                if (selection === 'Show Details') {
                    this.showSecurityReport();
                }
            });

        } catch (error) {
            vscode.window.showErrorMessage(`Failed to validate symbols: ${error}`);
            this.updateStatusBar('Error', 'ssl-error');
        }
    }

    private async showSecurityReport() {
        if (!this.currentReport) {
            vscode.window.showWarningMessage('No security report available. Run validation first.');
            return;
        }

        if (this.securityPanel) {
            this.securityPanel.reveal();
        } else {
            this.securityPanel = vscode.window.createWebviewPanel(
                'sslSecurityReport',
                'SSL Security Report',
                vscode.ViewColumn.Beside,
                {
                    enableScripts: true,
                    retainContextWhenHidden: true
                }
            );

            this.securityPanel.onDidDispose(() => {
                this.securityPanel = undefined;
            });
        }

        this.securityPanel.webview.html = this.generateReportHTML(this.currentReport);
    }

    private async addCustomSymbol() {
        const symbol = await vscode.window.showInputBox({
            prompt: 'Enter the symbol to add',
            placeHolder: '🜄'
        });

        if (!symbol) {
            return;
        }

        const anchor = await vscode.window.showInputBox({
            prompt: 'Enter the semantic anchor',
            placeHolder: 'Water_Alchemical'
        });

        if (!anchor) {
            return;
        }

        const description = await vscode.window.showInputBox({
            prompt: 'Enter the description',
            placeHolder: 'Emotion: receptive field activation'
        });

        if (!description) {
            return;
        }

        try {
            // Call backend to add custom symbol
            await this.callSSLBackend('', false, 'add_symbol', { symbol, anchor, description });
            vscode.window.showInformationMessage(`Symbol ${symbol} added successfully!`);
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to add symbol: ${error}`);
        }
    }

    private async exportReport() {
        if (!this.currentReport) {
            vscode.window.showWarningMessage('No security report available. Run validation first.');
            return;
        }

        const uri = await vscode.window.showSaveDialog({
            defaultUri: vscode.Uri.file('ssl_security_report.json'),
            filters: {
                'JSON files': ['json'],
                'All files': ['*']
            }
        });

        if (uri) {
            const reportData = JSON.stringify(this.currentReport, null, 2);
            await vscode.workspace.fs.writeFile(uri, Buffer.from(reportData, 'utf8'));
            vscode.window.showInformationMessage(`Report exported to ${uri.fsPath}`);
        }
    }

    private async onDocumentSave(document: vscode.TextDocument) {
        const config = this.getConfig();
        if (!config.autoSecureOnSave) {
            return;
        }

        // Only process certain file types
        const supportedLanguages = ['markdown', 'python', 'latex', 'plaintext'];
        if (!supportedLanguages.includes(document.languageId)) {
            return;
        }

        // Auto-validate on save
        const text = document.getText();
        try {
            const result = await this.callSSLBackend(text, true);
            if (!result.error) {
                this.currentReport = result.validation_report;
                this.updateStatusBarWithReport(result.validation_report);
            }
        } catch (error) {
            this.outputChannel.appendLine(`Auto-validation failed: ${error}`);
        }
    }

    private onActiveEditorChange(editor: vscode.TextEditor | undefined) {
        if (!editor) {
            this.updateStatusBar('No Editor', 'ssl-inactive');
            return;
        }

        this.updateStatusBar('Ready', 'ssl-ready');
    }

    private async callSSLBackend(text: string, validateOnly: boolean = false, 
                                action: string = 'secure', params: any = {}): Promise<SSLResponse> {
        return new Promise((resolve, reject) => {
            const config = this.getConfig();
            
            // Prepare SSL backend script path
            const backendScript = path.join(this.context.extensionPath, '..', 'src', 'symbolic_security_layer', 'vscode_backend.py');
            
            const requestData = {
                action: action,
                text: text,
                validate_only: validateOnly,
                config: config,
                params: params
            };

            const pythonProcess = cp.spawn(config.pythonPath, [backendScript], {
                stdio: ['pipe', 'pipe', 'pipe']
            });

            let stdout = '';
            let stderr = '';

            pythonProcess.stdout.on('data', (data) => {
                stdout += data.toString();
            });

            pythonProcess.stderr.on('data', (data) => {
                stderr += data.toString();
            });

            pythonProcess.on('close', (code) => {
                if (code !== 0) {
                    reject(new Error(`Python process exited with code ${code}: ${stderr}`));
                    return;
                }

                try {
                    const result = JSON.parse(stdout);
                    resolve(result);
                } catch (error) {
                    reject(new Error(`Failed to parse SSL response: ${error}`));
                }
            });

            pythonProcess.on('error', (error) => {
                reject(new Error(`Failed to start Python process: ${error}`));
            });

            // Send request data to Python process
            pythonProcess.stdin.write(JSON.stringify(requestData));
            pythonProcess.stdin.end();
        });
    }

    private updateStatusBar(text: string, tooltip: string) {
        this.statusBarItem.text = `$(shield) SSL: ${text}`;
        this.statusBarItem.tooltip = tooltip;
        this.statusBarItem.command = 'ssl.showSecurityReport';
    }

    private updateStatusBarWithReport(report: SecurityReport) {
        const securityIcon = this.getSecurityIcon(report.security_level);
        this.statusBarItem.text = `${securityIcon} SSL: ${report.symbol_coverage} (${report.security_level})`;
        this.statusBarItem.tooltip = `Security: ${report.security_level}, Coverage: ${report.symbol_coverage}, Compliance: ${report.compliance}`;
    }

    private getSecurityIcon(level: string): string {
        switch (level) {
            case 'HIGH': return '$(shield-check)';
            case 'MEDIUM': return '$(shield)';
            case 'LOW': return '$(shield-x)';
            default: return '$(question)';
        }
    }

    private generateReportHTML(report: SecurityReport): string {
        return `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SSL Security Report</title>
            <style>
                body {
                    font-family: var(--vscode-font-family);
                    color: var(--vscode-foreground);
                    background-color: var(--vscode-editor-background);
                    padding: 20px;
                    line-height: 1.6;
                }
                .header {
                    border-bottom: 1px solid var(--vscode-panel-border);
                    padding-bottom: 20px;
                    margin-bottom: 20px;
                }
                .metric {
                    display: flex;
                    justify-content: space-between;
                    padding: 8px 0;
                    border-bottom: 1px solid var(--vscode-widget-border);
                }
                .metric-label {
                    font-weight: bold;
                }
                .security-high { color: var(--vscode-testing-iconPassed); }
                .security-medium { color: var(--vscode-testing-iconQueued); }
                .security-low { color: var(--vscode-testing-iconFailed); }
                .compliance-good { color: var(--vscode-testing-iconPassed); }
                .compliance-partial { color: var(--vscode-testing-iconQueued); }
                .section {
                    margin: 20px 0;
                    padding: 15px;
                    background-color: var(--vscode-editor-inactiveSelectionBackground);
                    border-radius: 4px;
                }
                .progress-bar {
                    width: 100%;
                    height: 20px;
                    background-color: var(--vscode-progressBar-background);
                    border-radius: 10px;
                    overflow: hidden;
                    margin: 10px 0;
                }
                .progress-fill {
                    height: 100%;
                    background-color: var(--vscode-progressBar-foreground);
                    transition: width 0.3s ease;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🛡️ SSL Security Report</h1>
                <p>Generated on ${new Date().toLocaleString()}</p>
            </div>

            <div class="section">
                <h2>Security Overview</h2>
                <div class="metric">
                    <span class="metric-label">Security Level:</span>
                    <span class="security-${report.security_level.toLowerCase()}">${report.security_level}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Symbol Coverage:</span>
                    <span>${report.symbol_coverage}</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${report.symbol_coverage}"></div>
                </div>
                <div class="metric">
                    <span class="metric-label">Compliance Status:</span>
                    <span class="compliance-${report.compliance === 'CIP-1' ? 'good' : 'partial'}">${report.compliance}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Verification Status:</span>
                    <span>${report.verification_status}</span>
                </div>
            </div>

            <div class="section">
                <h2>Symbol Analysis</h2>
                <div class="metric">
                    <span class="metric-label">Total Symbols Found:</span>
                    <span>${report.symbol_count}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Anchored Symbols:</span>
                    <span>${report.anchored_count}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Semantically Anchored (SEM):</span>
                    <span class="security-high">${report.SEM_count}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Existentially Unverified (EXU):</span>
                    <span class="security-low">${report.EXU_count}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Procedurally Valid (PROV):</span>
                    <span class="security-medium">${report.PROV_count}</span>
                </div>
            </div>

            ${report.processing_time ? `
            <div class="section">
                <h2>Performance</h2>
                <div class="metric">
                    <span class="metric-label">Processing Time:</span>
                    <span>${report.processing_time.toFixed(3)}s</span>
                </div>
            </div>
            ` : ''}

            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    ${report.EXU_count > 0 ? `<li>Add semantic anchors for ${report.EXU_count} unknown symbols</li>` : ''}
                    ${report.security_level === 'LOW' ? '<li>Improve security by adding more anchored symbols</li>' : ''}
                    ${report.compliance !== 'CIP-1' ? '<li>Achieve CIP-1 compliance by reaching 95%+ symbol coverage</li>' : ''}
                    ${report.security_level === 'HIGH' && report.compliance === 'CIP-1' ? '<li>✅ Excellent security posture maintained!</li>' : ''}
                </ul>
            </div>
        </body>
        </html>
        `;
    }
}

export function activate(context: vscode.ExtensionContext) {
    const extension = new SSLExtension(context);
    extension.activate();
    
    // Set context for when SSL is enabled
    vscode.commands.executeCommand('setContext', 'ssl.enabled', true);
}

export function deactivate() {
    // Cleanup handled by SSLExtension.deactivate()
}

