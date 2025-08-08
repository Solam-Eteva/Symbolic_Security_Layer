#!/usr/bin/env python3
"""
Test suite for SSL Security Diagnostics
Tests the SecurityDiagnosticsReport class and reporting functionality
"""

import pytest
import tempfile
import json
import os
import sys

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from symbolic_security_layer.diagnostics import SecurityDiagnosticsReport


class TestSecurityDiagnosticsReport:
    """Test cases for SecurityDiagnosticsReport"""
    
    @pytest.fixture
    def diagnostics(self):
        """Create diagnostics report instance"""
        return SecurityDiagnosticsReport()
    
    @pytest.fixture
    def sample_security_data(self):
        """Sample security data for testing"""
        return {
            'processing_summary': {
                'total_texts_processed': 100,
                'total_symbols_found': 250,
                'total_symbols_anchored': 200,
                'processing_time_seconds': 5.5,
                'average_coverage': '80.0%'
            },
            'security_distribution': {
                'HIGH': 30,
                'MEDIUM': 50,
                'LOW': 20
            },
            'compliance_distribution': {
                'CIP-1': 75,
                'PARTIAL': 25
            },
            'vocabulary_analysis': {
                'total_vocabulary_size': 1000,
                'secured_tokens_count': 150,
                'unsecured_tokens_count': 50,
                'security_ratio': 0.75,
                'top_secured_tokens': ['🜄', '☥', '∞', '⚛', '☯'],
                'top_unsecured_tokens': ['🔮', '🌟', '🎭', '🎪', '🎨']
            }
        }
    
    def test_diagnostics_initialization(self, diagnostics):
        """Test diagnostics report initialization"""
        assert diagnostics is not None
        assert hasattr(diagnostics, 'config')
        assert hasattr(diagnostics, 'metrics_history')
        assert hasattr(diagnostics, 'analysis_cache')
        
        # Check default configuration
        assert diagnostics.config['enable_visualizations'] is True
        assert diagnostics.config['output_format'] == 'markdown'
        assert diagnostics.config['include_recommendations'] is True
    
    def test_custom_configuration(self):
        """Test diagnostics with custom configuration"""
        custom_config = {
            'output_format': 'json',
            'detail_level': 'basic',
            'enable_visualizations': False
        }
        
        diagnostics = SecurityDiagnosticsReport(custom_config)
        
        assert diagnostics.config['output_format'] == 'json'
        assert diagnostics.config['detail_level'] == 'basic'
        assert diagnostics.config['enable_visualizations'] is False
    
    def test_generate_summary_stats(self, diagnostics, sample_security_data):
        """Test summary statistics generation"""
        analysis = diagnostics._analyze_security_data(sample_security_data)
        summary = analysis['summary']
        
        # Check summary structure
        assert 'total_items_processed' in summary
        assert 'total_symbols_found' in summary
        assert 'total_symbols_anchored' in summary
        assert 'average_security_level' in summary
        assert 'compliance_rate' in summary
        
        # Check values
        assert summary['total_items_processed'] == 100
        assert summary['total_symbols_found'] == 250
        assert summary['total_symbols_anchored'] == 200
        assert summary['average_security_level'] in ['HIGH', 'MEDIUM', 'LOW']
        assert 0.0 <= summary['compliance_rate'] <= 1.0
    
    def test_analyze_symbols(self, diagnostics, sample_security_data):
        """Test symbol analysis"""
        analysis = diagnostics._analyze_security_data(sample_security_data)
        symbol_analysis = analysis['symbol_analysis']
        
        # Check symbol analysis structure
        assert 'symbol_distribution' in symbol_analysis
        assert 'anchoring_effectiveness' in symbol_analysis
        assert 'unknown_symbols' in symbol_analysis
        assert 'top_secured_symbols' in symbol_analysis
        
        # Check symbol distribution
        dist = symbol_analysis['symbol_distribution']
        assert 'secured' in dist
        assert 'unsecured' in dist
        assert 'total' in dist
        
        # Check values match input data
        assert dist['secured'] == 150
        assert dist['unsecured'] == 50
        assert dist['total'] == 1000
    
    def test_analyze_security_trends(self, diagnostics, sample_security_data):
        """Test security trends analysis"""
        analysis = diagnostics._analyze_security_data(sample_security_data)
        trends = analysis['security_trends']
        
        # Check trends structure
        assert 'security_level_distribution' in trends
        assert 'coverage_statistics' in trends
        
        # Check security level distribution
        sec_dist = trends['security_level_distribution']
        assert sec_dist == sample_security_data['security_distribution']
        
        # Check coverage statistics
        coverage_stats = trends['coverage_statistics']
        assert 'average_coverage' in coverage_stats
        assert 'coverage_grade' in coverage_stats
        assert 'target_coverage' in coverage_stats
        assert 'gap_to_target' in coverage_stats
        
        assert coverage_stats['average_coverage'] == 80.0
        assert coverage_stats['target_coverage'] == 95.0
        assert coverage_stats['gap_to_target'] == 15.0
    
    def test_analyze_compliance(self, diagnostics, sample_security_data):
        """Test compliance analysis"""
        analysis = diagnostics._analyze_security_data(sample_security_data)
        compliance = analysis['compliance_analysis']
        
        # Check compliance structure
        assert 'cip1_compliance' in compliance
        assert 'certification_readiness' in compliance
        
        # Check CIP-1 compliance
        cip1 = compliance['cip1_compliance']
        assert 'compliant_items' in cip1
        assert 'total_items' in cip1
        assert 'compliance_rate' in cip1
        assert 'compliance_percentage' in cip1
        
        # Check values
        assert cip1['compliant_items'] == 75
        assert cip1['total_items'] == 100
        assert cip1['compliance_rate'] == 0.75
        assert cip1['compliance_percentage'] == '75.0%'
        
        # Check certification readiness
        assert compliance['certification_readiness'] in ['READY', 'NEAR_READY', 'NOT_READY']
    
    def test_assess_risks(self, diagnostics, sample_security_data):
        """Test risk assessment"""
        analysis = diagnostics._analyze_security_data(sample_security_data)
        risks = analysis['risk_assessment']
        
        # Check risk assessment structure
        assert 'risk_level' in risks
        assert 'risk_score' in risks
        assert 'risk_factors' in risks
        assert 'vulnerability_count' in risks
        
        # Check risk level is valid
        assert risks['risk_level'] in ['HIGH', 'MEDIUM', 'LOW']
        
        # Check risk score is in valid range
        assert 0.0 <= risks['risk_score'] <= 1.0
        
        # Check risk factors
        assert isinstance(risks['risk_factors'], list)
        assert risks['vulnerability_count'] == len(risks['risk_factors'])
    
    def test_analyze_performance(self, diagnostics, sample_security_data):
        """Test performance analysis"""
        analysis = diagnostics._analyze_security_data(sample_security_data)
        performance = analysis['performance_metrics']
        
        # Check performance structure
        assert 'processing_speed' in performance
        
        # Check processing speed
        speed = performance['processing_speed']
        assert 'items_per_second' in speed
        assert 'total_processing_time' in speed
        assert 'average_time_per_item' in speed
        
        # Check calculated values
        assert speed['total_processing_time'] == 5.5
        assert speed['items_per_second'] == 100 / 5.5
        assert speed['average_time_per_item'] == 5.5 / 100
    
    def test_generate_recommendations(self, diagnostics, sample_security_data):
        """Test recommendations generation"""
        analysis = diagnostics._analyze_security_data(sample_security_data)
        recommendations = analysis['recommendations']
        
        # Should generate recommendations
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should include specific recommendations based on data
        rec_text = ' '.join(recommendations).lower()
        
        # Should recommend compliance improvement (75% < 95%)
        assert any('compliance' in rec.lower() for rec in recommendations)
        
        # Should recommend adding unknown symbols
        assert any('symbol' in rec.lower() for rec in recommendations)
    
    def test_grade_coverage(self, diagnostics):
        """Test coverage grading"""
        # Test different coverage levels
        assert diagnostics._grade_coverage(98.0) == 'A+'
        assert diagnostics._grade_coverage(92.0) == 'A'
        assert diagnostics._grade_coverage(87.0) == 'B+'
        assert diagnostics._grade_coverage(82.0) == 'B'
        assert diagnostics._grade_coverage(75.0) == 'C'
        assert diagnostics._grade_coverage(65.0) == 'D'
    
    def test_generate_markdown_report(self, diagnostics, sample_security_data):
        """Test markdown report generation"""
        report_content = diagnostics.generate_report(sample_security_data, output_path=None)
        
        # Check report is markdown format
        assert isinstance(report_content, str)
        assert len(report_content) > 0
        
        # Check markdown structure
        assert '# Symbolic Security Layer - Diagnostics Report' in report_content
        assert '## Executive Summary' in report_content
        assert '## Security Analysis' in report_content
        assert '## Compliance Analysis' in report_content
        assert '## Risk Assessment' in report_content
        
        # Check data is included
        assert '100' in report_content  # Total items processed
        assert '250' in report_content  # Total symbols found
        assert '80.0%' in report_content  # Average coverage
    
    def test_generate_json_report(self, diagnostics, sample_security_data):
        """Test JSON report generation"""
        diagnostics.config['output_format'] = 'json'
        report_content = diagnostics.generate_report(sample_security_data, output_path=None)
        
        # Check report is valid JSON
        assert isinstance(report_content, str)
        report_data = json.loads(report_content)
        
        # Check JSON structure
        assert isinstance(report_data, dict)
        assert 'summary' in report_data
        assert 'symbol_analysis' in report_data
        assert 'compliance_analysis' in report_data
        assert 'risk_assessment' in report_data
        assert 'metadata' in report_data
    
    def test_generate_html_report(self, diagnostics, sample_security_data):
        """Test HTML report generation"""
        diagnostics.config['output_format'] = 'html'
        report_content = diagnostics.generate_report(sample_security_data, output_path=None)
        
        # Check report is HTML format
        assert isinstance(report_content, str)
        assert '<!DOCTYPE html>' in report_content
        assert '<html>' in report_content
        assert '<head>' in report_content
        assert '<body>' in report_content
        assert 'SSL Diagnostics Report' in report_content
    
    def test_report_file_output(self, diagnostics, sample_security_data):
        """Test report output to file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_path = f.name
        
        try:
            # Generate report to file
            report_content = diagnostics.generate_report(sample_security_data, output_path)
            
            # Check file was created
            assert os.path.exists(output_path)
            
            # Check file content
            with open(output_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            assert file_content == report_content
            assert len(file_content) > 0
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_visualization_generation(self, diagnostics, sample_security_data):
        """Test visualization generation (if available)"""
        try:
            # Try to generate summary chart
            chart_data = diagnostics.generate_visualization(sample_security_data, 'summary')
            
            if chart_data is not None:
                # Should return base64 encoded image data
                assert isinstance(chart_data, str)
                assert len(chart_data) > 0
            else:
                # Visualization libraries not available
                assert True  # This is expected in test environment
                
        except ImportError:
            # Visualization libraries not available
            assert True  # This is expected
    
    def test_error_handling_invalid_data(self, diagnostics):
        """Test error handling with invalid data"""
        # Test with empty data
        empty_data = {}
        analysis = diagnostics._analyze_security_data(empty_data)
        
        # Should handle gracefully
        assert isinstance(analysis, dict)
        assert 'summary' in analysis
        assert 'metadata' in analysis
        
        # Test with malformed data
        malformed_data = {'invalid_key': 'invalid_value'}
        analysis = diagnostics._analyze_security_data(malformed_data)
        
        # Should handle gracefully
        assert isinstance(analysis, dict)
    
    def test_comprehensive_analysis_workflow(self, diagnostics, sample_security_data):
        """Test complete analysis workflow"""
        # Generate full analysis
        analysis = diagnostics._analyze_security_data(sample_security_data)
        
        # Check all analysis components are present
        expected_components = [
            'summary', 'symbol_analysis', 'security_trends',
            'compliance_analysis', 'risk_assessment', 'performance_metrics',
            'recommendations', 'metadata'
        ]
        
        for component in expected_components:
            assert component in analysis
            assert isinstance(analysis[component], dict) or isinstance(analysis[component], list)
        
        # Check metadata
        metadata = analysis['metadata']
        assert 'analysis_timestamp' in metadata
        assert 'data_sources' in metadata
        assert 'analysis_version' in metadata
        
        # Generate report from analysis
        report_content = diagnostics._generate_markdown_report(analysis)
        assert isinstance(report_content, str)
        assert len(report_content) > 1000  # Should be substantial report


class TestDiagnosticsIntegration:
    """Integration tests for diagnostics with other components"""
    
    def test_real_world_data_processing(self):
        """Test diagnostics with realistic security data"""
        # Simulate data from actual SSL processing
        realistic_data = {
            'processing_summary': {
                'total_texts_processed': 1000,
                'total_symbols_found': 1500,
                'total_symbols_anchored': 1200,
                'processing_time_seconds': 45.2,
                'average_coverage': '85.5%'
            },
            'security_distribution': {
                'HIGH': 400,
                'MEDIUM': 450,
                'LOW': 150
            },
            'compliance_distribution': {
                'CIP-1': 850,
                'PARTIAL': 150
            },
            'vocabulary_analysis': {
                'total_vocabulary_size': 5000,
                'secured_tokens_count': 300,
                'unsecured_tokens_count': 75,
                'security_ratio': 0.8,
                'top_secured_tokens': ['🜄', '☥', '∞', '⚛', '☯', '🜂', '🜃', '🜁'],
                'top_unsecured_tokens': ['🔮', '🌟', '🎭', '🎪', '🎨']
            }
        }
        
        diagnostics = SecurityDiagnosticsReport()
        
        # Generate comprehensive report
        report_content = diagnostics.generate_report(realistic_data)
        
        # Should generate substantial report
        assert len(report_content) > 2000
        
        # Should include all key metrics
        assert '1000' in report_content  # Total processed
        assert '85.5%' in report_content  # Coverage
        assert 'HIGH' in report_content  # Security level
        assert 'CIP-1' in report_content  # Compliance
    
    def test_edge_case_data(self):
        """Test diagnostics with edge case data"""
        # Test with minimal data
        minimal_data = {
            'processing_summary': {
                'total_texts_processed': 1,
                'total_symbols_found': 0,
                'total_symbols_anchored': 0,
                'processing_time_seconds': 0.1,
                'average_coverage': '0.0%'
            },
            'security_distribution': {'LOW': 1},
            'compliance_distribution': {'PARTIAL': 1}
        }
        
        diagnostics = SecurityDiagnosticsReport()
        report_content = diagnostics.generate_report(minimal_data)
        
        # Should handle gracefully
        assert isinstance(report_content, str)
        assert len(report_content) > 0
        
        # Test with maximum data
        maximum_data = {
            'processing_summary': {
                'total_texts_processed': 100000,
                'total_symbols_found': 500000,
                'total_symbols_anchored': 500000,
                'processing_time_seconds': 3600.0,
                'average_coverage': '100.0%'
            },
            'security_distribution': {'HIGH': 100000},
            'compliance_distribution': {'CIP-1': 100000}
        }
        
        report_content = diagnostics.generate_report(maximum_data)
        
        # Should handle large numbers gracefully
        assert isinstance(report_content, str)
        assert '100,000' in report_content or '100000' in report_content


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

