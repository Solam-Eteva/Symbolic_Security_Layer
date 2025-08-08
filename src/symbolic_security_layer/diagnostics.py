"""
Security Diagnostics and Reporting Module
Provides comprehensive analysis and visualization of security metrics
"""

import json
import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
from pathlib import Path
import base64
import io

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class SecurityDiagnosticsReport:
    """
    Comprehensive security diagnostics and reporting system
    Generates detailed analysis of symbolic security metrics
    """
    
    def __init__(self, config: Dict = None):
        self.config = self._load_config(config)
        self.metrics_history = []
        self.analysis_cache = {}
        
    def _load_config(self, config: Dict = None) -> Dict[str, Any]:
        """Load diagnostics configuration"""
        default_config = {
            "enable_visualizations": True,
            "output_format": "markdown",  # markdown, json, html
            "include_recommendations": True,
            "detail_level": "comprehensive",  # basic, detailed, comprehensive
            "export_charts": True,
            "chart_style": "seaborn",
            "color_palette": "viridis"
        }
        
        if config:
            default_config.update(config)
            
        return default_config
    
    def generate_report(self, security_data: Dict, 
                       output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive security diagnostics report
        """
        report_data = self._analyze_security_data(security_data)
        
        if self.config["output_format"] == "markdown":
            report_content = self._generate_markdown_report(report_data)
        elif self.config["output_format"] == "json":
            report_content = self._generate_json_report(report_data)
        elif self.config["output_format"] == "html":
            report_content = self._generate_html_report(report_data)
        else:
            raise ValueError(f"Unsupported output format: {self.config['output_format']}")
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
        
        return report_content
    
    def _analyze_security_data(self, data: Dict) -> Dict[str, Any]:
        """Perform comprehensive analysis of security data"""
        analysis = {
            "summary": self._generate_summary_stats(data),
            "symbol_analysis": self._analyze_symbols(data),
            "security_trends": self._analyze_security_trends(data),
            "compliance_analysis": self._analyze_compliance(data),
            "risk_assessment": self._assess_risks(data),
            "performance_metrics": self._analyze_performance(data),
            "recommendations": self._generate_recommendations(data),
            "metadata": {
                "analysis_timestamp": time.time(),
                "data_sources": list(data.keys()),
                "analysis_version": "1.0"
            }
        }
        
        return analysis
    
    def _generate_summary_stats(self, data: Dict) -> Dict[str, Any]:
        """Generate high-level summary statistics"""
        summary = {
            "total_items_processed": 0,
            "total_symbols_found": 0,
            "total_symbols_anchored": 0,
            "average_security_level": "UNKNOWN",
            "compliance_rate": 0.0,
            "processing_efficiency": 0.0
        }
        
        # Extract metrics from various data sources
        if "processing_summary" in data:
            proc_summary = data["processing_summary"]
            summary["total_items_processed"] = proc_summary.get("total_texts_processed", 0)
            summary["total_symbols_found"] = proc_summary.get("total_symbols_found", 0)
            summary["total_symbols_anchored"] = proc_summary.get("total_symbols_anchored", 0)
        
        if "security_distribution" in data:
            sec_dist = data["security_distribution"]
            total_items = sum(sec_dist.values())
            if total_items > 0:
                # Calculate weighted average security level
                weights = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
                weighted_sum = sum(weights.get(level, 0) * count for level, count in sec_dist.items())
                avg_weight = weighted_sum / total_items
                if avg_weight >= 2.5:
                    summary["average_security_level"] = "HIGH"
                elif avg_weight >= 1.5:
                    summary["average_security_level"] = "MEDIUM"
                else:
                    summary["average_security_level"] = "LOW"
        
        if "compliance_distribution" in data:
            comp_dist = data["compliance_distribution"]
            total_items = sum(comp_dist.values())
            compliant_items = comp_dist.get("CIP-1", 0)
            summary["compliance_rate"] = (compliant_items / total_items) if total_items > 0 else 0.0
        
        return summary
    
    def _analyze_symbols(self, data: Dict) -> Dict[str, Any]:
        """Analyze symbolic content and patterns"""
        analysis = {
            "symbol_distribution": {},
            "anchoring_effectiveness": {},
            "unknown_symbols": [],
            "top_secured_symbols": [],
            "symbol_categories": {}
        }
        
        if "vocabulary_analysis" in data:
            vocab_analysis = data["vocabulary_analysis"]
            analysis["symbol_distribution"] = {
                "secured": vocab_analysis.get("secured_tokens_count", 0),
                "unsecured": vocab_analysis.get("unsecured_tokens_count", 0),
                "total": vocab_analysis.get("total_vocabulary_size", 0)
            }
            analysis["anchoring_effectiveness"] = vocab_analysis.get("security_ratio", 0.0)
            analysis["top_secured_symbols"] = vocab_analysis.get("top_secured_tokens", [])
            analysis["unknown_symbols"] = vocab_analysis.get("top_unsecured_tokens", [])
        
        return analysis
    
    def _analyze_security_trends(self, data: Dict) -> Dict[str, Any]:
        """Analyze security trends and patterns"""
        trends = {
            "security_level_distribution": {},
            "coverage_statistics": {},
            "temporal_patterns": {},
            "improvement_opportunities": []
        }
        
        if "security_distribution" in data:
            trends["security_level_distribution"] = data["security_distribution"]
        
        # Analyze coverage patterns
        if "processing_summary" in data:
            proc_summary = data["processing_summary"]
            avg_coverage = proc_summary.get("average_coverage", "0%")
            coverage_value = float(avg_coverage.rstrip("%"))
            
            trends["coverage_statistics"] = {
                "average_coverage": coverage_value,
                "coverage_grade": self._grade_coverage(coverage_value),
                "target_coverage": 95.0,
                "gap_to_target": max(0, 95.0 - coverage_value)
            }
        
        return trends
    
    def _analyze_compliance(self, data: Dict) -> Dict[str, Any]:
        """Analyze compliance with security standards"""
        compliance = {
            "cip1_compliance": {},
            "standard_adherence": {},
            "compliance_gaps": [],
            "certification_readiness": "NOT_READY"
        }
        
        if "compliance_distribution" in data:
            comp_dist = data["compliance_distribution"]
            total_items = sum(comp_dist.values())
            compliant_items = comp_dist.get("CIP-1", 0)
            
            compliance["cip1_compliance"] = {
                "compliant_items": compliant_items,
                "total_items": total_items,
                "compliance_rate": (compliant_items / total_items) if total_items > 0 else 0.0,
                "compliance_percentage": f"{(compliant_items / total_items * 100):.1f}%" if total_items > 0 else "0%"
            }
            
            # Determine certification readiness
            compliance_rate = compliance["cip1_compliance"]["compliance_rate"]
            if compliance_rate >= 0.95:
                compliance["certification_readiness"] = "READY"
            elif compliance_rate >= 0.80:
                compliance["certification_readiness"] = "NEAR_READY"
            else:
                compliance["certification_readiness"] = "NOT_READY"
        
        return compliance
    
    def _assess_risks(self, data: Dict) -> Dict[str, Any]:
        """Assess security risks and vulnerabilities"""
        risks = {
            "risk_level": "UNKNOWN",
            "risk_factors": [],
            "vulnerability_count": 0,
            "mitigation_priority": [],
            "risk_score": 0.0
        }
        
        risk_score = 0.0
        risk_factors = []
        
        # Analyze security distribution for risks
        if "security_distribution" in data:
            sec_dist = data["security_distribution"]
            total_items = sum(sec_dist.values())
            low_security_items = sec_dist.get("LOW", 0)
            
            if total_items > 0:
                low_security_ratio = low_security_items / total_items
                if low_security_ratio > 0.3:
                    risk_factors.append("High proportion of low-security items")
                    risk_score += 0.4
                elif low_security_ratio > 0.1:
                    risk_factors.append("Moderate proportion of low-security items")
                    risk_score += 0.2
        
        # Analyze compliance risks
        if "compliance_distribution" in data:
            comp_dist = data["compliance_distribution"]
            total_items = sum(comp_dist.values())
            non_compliant = total_items - comp_dist.get("CIP-1", 0)
            
            if total_items > 0:
                non_compliant_ratio = non_compliant / total_items
                if non_compliant_ratio > 0.2:
                    risk_factors.append("High non-compliance rate")
                    risk_score += 0.3
                elif non_compliant_ratio > 0.05:
                    risk_factors.append("Moderate non-compliance rate")
                    risk_score += 0.1
        
        # Analyze unknown symbols
        if "vocabulary_analysis" in data:
            vocab_analysis = data["vocabulary_analysis"]
            unsecured_count = vocab_analysis.get("unsecured_tokens_count", 0)
            total_vocab = vocab_analysis.get("total_vocabulary_size", 1)
            
            unsecured_ratio = unsecured_count / total_vocab
            if unsecured_ratio > 0.1:
                risk_factors.append("High number of unsecured symbols")
                risk_score += 0.3
                
        # Determine overall risk level
        if risk_score >= 0.7:
            risks["risk_level"] = "HIGH"
        elif risk_score >= 0.4:
            risks["risk_level"] = "MEDIUM"
        else:
            risks["risk_level"] = "LOW"
        
        risks["risk_score"] = min(risk_score, 1.0)
        risks["risk_factors"] = risk_factors
        risks["vulnerability_count"] = len(risk_factors)
        
        return risks
    
    def _analyze_performance(self, data: Dict) -> Dict[str, Any]:
        """Analyze processing performance metrics"""
        performance = {
            "processing_speed": {},
            "efficiency_metrics": {},
            "resource_utilization": {},
            "optimization_suggestions": []
        }
        
        if "processing_summary" in data:
            proc_summary = data["processing_summary"]
            total_items = proc_summary.get("total_texts_processed", 0)
            processing_time = proc_summary.get("processing_time_seconds", 0)
            
            if processing_time > 0:
                items_per_second = total_items / processing_time
                performance["processing_speed"] = {
                    "items_per_second": items_per_second,
                    "total_processing_time": processing_time,
                    "average_time_per_item": processing_time / total_items if total_items > 0 else 0
                }
                
                # Generate optimization suggestions
                if items_per_second < 10:
                    performance["optimization_suggestions"].append("Consider batch processing optimization")
                if processing_time > 60:
                    performance["optimization_suggestions"].append("Consider parallel processing for large datasets")
        
        return performance
    
    def _generate_recommendations(self, data: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Security recommendations
        if "security_distribution" in data:
            sec_dist = data["security_distribution"]
            total_items = sum(sec_dist.values())
            low_security = sec_dist.get("LOW", 0)
            
            if total_items > 0 and low_security / total_items > 0.2:
                recommendations.append("🔒 Improve security by adding semantic anchors for symbolic content")
        
        # Compliance recommendations
        if "compliance_distribution" in data:
            comp_dist = data["compliance_distribution"]
            total_items = sum(comp_dist.values())
            compliant = comp_dist.get("CIP-1", 0)
            
            if total_items > 0 and compliant / total_items < 0.95:
                recommendations.append("📋 Achieve CIP-1 compliance by increasing symbol coverage to 95%+")
        
        # Symbol database recommendations
        if "vocabulary_analysis" in data:
            vocab_analysis = data["vocabulary_analysis"]
            unsecured_count = vocab_analysis.get("unsecured_tokens_count", 0)
            
            if unsecured_count > 0:
                recommendations.append(f"📚 Add {unsecured_count} unknown symbols to the symbol database")
        
        # Performance recommendations
        if "processing_summary" in data:
            proc_summary = data["processing_summary"]
            processing_time = proc_summary.get("processing_time_seconds", 0)
            
            if processing_time > 300:  # 5 minutes
                recommendations.append("⚡ Consider optimizing processing pipeline for better performance")
        
        return recommendations
    
    def _grade_coverage(self, coverage: float) -> str:
        """Grade coverage percentage"""
        if coverage >= 95:
            return "A+"
        elif coverage >= 90:
            return "A"
        elif coverage >= 85:
            return "B+"
        elif coverage >= 80:
            return "B"
        elif coverage >= 70:
            return "C"
        else:
            return "D"
    
    def _generate_markdown_report(self, analysis: Dict) -> str:
        """Generate markdown format report"""
        report = []
        
        # Header
        report.append("# Symbolic Security Layer - Diagnostics Report")
        report.append(f"*Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*")
        report.append("")
        
        # Executive Summary
        summary = analysis["summary"]
        report.append("## Executive Summary")
        report.append("")
        report.append(f"- **Items Processed**: {summary['total_items_processed']:,}")
        report.append(f"- **Symbols Found**: {summary['total_symbols_found']:,}")
        report.append(f"- **Symbols Anchored**: {summary['total_symbols_anchored']:,}")
        report.append(f"- **Average Security Level**: {summary['average_security_level']}")
        report.append(f"- **Compliance Rate**: {summary['compliance_rate']:.1%}")
        report.append("")
        
        # Security Analysis
        report.append("## Security Analysis")
        report.append("")
        
        trends = analysis["security_trends"]
        if "security_level_distribution" in trends:
            report.append("### Security Level Distribution")
            for level, count in trends["security_level_distribution"].items():
                report.append(f"- **{level}**: {count:,} items")
            report.append("")
        
        if "coverage_statistics" in trends:
            coverage = trends["coverage_statistics"]
            report.append("### Coverage Statistics")
            report.append(f"- **Average Coverage**: {coverage['average_coverage']:.1f}%")
            report.append(f"- **Coverage Grade**: {coverage['coverage_grade']}")
            report.append(f"- **Gap to Target**: {coverage['gap_to_target']:.1f}%")
            report.append("")
        
        # Symbol Analysis
        symbol_analysis = analysis["symbol_analysis"]
        report.append("## Symbol Analysis")
        report.append("")
        
        if "symbol_distribution" in symbol_analysis:
            dist = symbol_analysis["symbol_distribution"]
            report.append("### Symbol Distribution")
            report.append(f"- **Secured Symbols**: {dist.get('secured', 0):,}")
            report.append(f"- **Unsecured Symbols**: {dist.get('unsecured', 0):,}")
            report.append(f"- **Total Symbols**: {dist.get('total', 0):,}")
            report.append("")
        
        if symbol_analysis["top_secured_symbols"]:
            report.append("### Top Secured Symbols")
            for symbol in symbol_analysis["top_secured_symbols"][:5]:
                report.append(f"- `{symbol}`")
            report.append("")
        
        if symbol_analysis["unknown_symbols"]:
            report.append("### Unknown Symbols Requiring Attention")
            for symbol in symbol_analysis["unknown_symbols"][:5]:
                report.append(f"- `{symbol}`")
            report.append("")
        
        # Compliance Analysis
        compliance = analysis["compliance_analysis"]
        report.append("## Compliance Analysis")
        report.append("")
        
        if "cip1_compliance" in compliance:
            cip1 = compliance["cip1_compliance"]
            report.append("### CIP-1 Compliance")
            report.append(f"- **Compliance Rate**: {cip1['compliance_percentage']}")
            report.append(f"- **Compliant Items**: {cip1['compliant_items']:,}")
            report.append(f"- **Total Items**: {cip1['total_items']:,}")
            report.append(f"- **Certification Status**: {compliance['certification_readiness']}")
            report.append("")
        
        # Risk Assessment
        risks = analysis["risk_assessment"]
        report.append("## Risk Assessment")
        report.append("")
        report.append(f"- **Overall Risk Level**: {risks['risk_level']}")
        report.append(f"- **Risk Score**: {risks['risk_score']:.2f}/1.00")
        report.append(f"- **Vulnerabilities Found**: {risks['vulnerability_count']}")
        report.append("")
        
        if risks["risk_factors"]:
            report.append("### Risk Factors")
            for factor in risks["risk_factors"]:
                report.append(f"- {factor}")
            report.append("")
        
        # Performance Metrics
        performance = analysis["performance_metrics"]
        report.append("## Performance Metrics")
        report.append("")
        
        if "processing_speed" in performance:
            speed = performance["processing_speed"]
            report.append("### Processing Performance")
            report.append(f"- **Items per Second**: {speed['items_per_second']:.2f}")
            report.append(f"- **Total Processing Time**: {speed['total_processing_time']:.2f}s")
            report.append(f"- **Average Time per Item**: {speed['average_time_per_item']:.4f}s")
            report.append("")
        
        # Recommendations
        recommendations = analysis["recommendations"]
        if recommendations:
            report.append("## Recommendations")
            report.append("")
            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        # Metadata
        metadata = analysis["metadata"]
        report.append("## Report Metadata")
        report.append("")
        report.append(f"- **Analysis Version**: {metadata['analysis_version']}")
        report.append(f"- **Data Sources**: {', '.join(metadata['data_sources'])}")
        report.append(f"- **Generated**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metadata['analysis_timestamp']))}")
        
        return "\n".join(report)
    
    def _generate_json_report(self, analysis: Dict) -> str:
        """Generate JSON format report"""
        return json.dumps(analysis, indent=2, ensure_ascii=False)
    
    def _generate_html_report(self, analysis: Dict) -> str:
        """Generate HTML format report"""
        # Convert markdown to HTML (simplified implementation)
        markdown_content = self._generate_markdown_report(analysis)
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SSL Diagnostics Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                .metric {{ background: #f5f5f5; padding: 10px; margin: 10px 0; }}
                .risk-high {{ color: #d32f2f; }}
                .risk-medium {{ color: #f57c00; }}
                .risk-low {{ color: #388e3c; }}
            </style>
        </head>
        <body>
            <pre>{markdown_content}</pre>
        </body>
        </html>
        """
        
        return html_template
    
    def generate_visualization(self, data: Dict, chart_type: str = "summary") -> Optional[str]:
        """Generate visualization charts"""
        if not VISUALIZATION_AVAILABLE:
            print("Warning: Visualization libraries not available. Install with: pip install matplotlib seaborn pandas")
            return None
        
        plt.style.use(self.config["chart_style"] if self.config["chart_style"] in plt.style.available else "default")
        
        if chart_type == "summary":
            return self._create_summary_chart(data)
        elif chart_type == "security_distribution":
            return self._create_security_distribution_chart(data)
        elif chart_type == "compliance":
            return self._create_compliance_chart(data)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
    
    def _create_summary_chart(self, data: Dict) -> str:
        """Create summary dashboard chart"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('SSL Security Dashboard', fontsize=16)
        
        # Security level distribution
        if "security_distribution" in data:
            sec_dist = data["security_distribution"]
            ax1.pie(sec_dist.values(), labels=sec_dist.keys(), autopct='%1.1f%%')
            ax1.set_title('Security Level Distribution')
        
        # Compliance distribution
        if "compliance_distribution" in data:
            comp_dist = data["compliance_distribution"]
            ax2.bar(comp_dist.keys(), comp_dist.values())
            ax2.set_title('Compliance Distribution')
            ax2.tick_params(axis='x', rotation=45)
        
        # Symbol analysis
        if "vocabulary_analysis" in data:
            vocab = data["vocabulary_analysis"]
            categories = ['Secured', 'Unsecured']
            values = [vocab.get('secured_tokens_count', 0), vocab.get('unsecured_tokens_count', 0)]
            ax3.bar(categories, values)
            ax3.set_title('Symbol Security Status')
        
        # Coverage trend (placeholder)
        ax4.plot([1, 2, 3, 4, 5], [70, 75, 80, 85, 90])
        ax4.set_title('Coverage Trend (Sample)')
        ax4.set_ylabel('Coverage %')
        
        plt.tight_layout()
        
        # Save to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return chart_data
    
    def _create_security_distribution_chart(self, data: Dict) -> str:
        """Create detailed security distribution chart"""
        if "security_distribution" not in data:
            return None
        
        sec_dist = data["security_distribution"]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sec_dist.keys(), sec_dist.values())
        
        # Color bars based on security level
        colors = {'HIGH': 'green', 'MEDIUM': 'orange', 'LOW': 'red'}
        for bar, level in zip(bars, sec_dist.keys()):
            bar.set_color(colors.get(level, 'gray'))
        
        plt.title('Security Level Distribution')
        plt.xlabel('Security Level')
        plt.ylabel('Number of Items')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return chart_data
    
    def _create_compliance_chart(self, data: Dict) -> str:
        """Create compliance analysis chart"""
        if "compliance_distribution" not in data:
            return None
        
        comp_dist = data["compliance_distribution"]
        
        plt.figure(figsize=(8, 8))
        plt.pie(comp_dist.values(), labels=comp_dist.keys(), autopct='%1.1f%%', startangle=90)
        plt.title('Compliance Distribution')
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return chart_data

