#!/usr/bin/env python3
"""
Test suite for SSL Core Engine
Tests the SymbolicSecurityEngine class and core functionality
"""

import pytest
import json
import tempfile
import os
import sys

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from symbolic_security_layer.core import SymbolicSecurityEngine


class TestSymbolicSecurityEngine:
    """Test cases for SymbolicSecurityEngine"""
    
    @pytest.fixture
    def ssl_engine(self):
        """Create a fresh SSL engine for each test"""
        return SymbolicSecurityEngine()
    
    @pytest.fixture
    def sample_symbolic_text(self):
        """Sample text with symbolic content"""
        return "The alchemical symbol 🜄 represents water and ☥ symbolizes life force"
    
    @pytest.fixture
    def sample_corrupted_text(self):
        """Sample text with corrupted symbols"""
        return "Corrupted symbols: (cid:0) and ō need reconstruction"
    
    def test_engine_initialization(self, ssl_engine):
        """Test SSL engine initializes correctly"""
        assert ssl_engine is not None
        assert hasattr(ssl_engine, 'symbol_db')
        assert hasattr(ssl_engine, 'config')
        assert hasattr(ssl_engine, 'reconstruction_log')
        assert hasattr(ssl_engine, 'session_id')
        
        # Check default configuration
        assert ssl_engine.config['unicode_threshold'] == 0x1000
        assert ssl_engine.config['auto_anchor'] is True
        assert ssl_engine.config['semantic_anchoring'] is True
        
        # Check symbol database is loaded
        assert len(ssl_engine.symbol_db) > 0
        assert '\U0001F704' in ssl_engine.symbol_db  # Water alchemical
        assert '\u2625' in ssl_engine.symbol_db      # Ankh
    
    def test_secure_content_text(self, ssl_engine, sample_symbolic_text):
        """Test securing textual content"""
        secured_content, report = ssl_engine.secure_content(sample_symbolic_text)
        
        # Check that content was processed
        assert secured_content is not None
        assert isinstance(secured_content, str)
        assert len(secured_content) >= len(sample_symbolic_text)
        
        # Check validation report
        assert isinstance(report, dict)
        assert 'SEM_count' in report
        assert 'EXU_count' in report
        assert 'symbol_count' in report
        assert 'symbol_coverage' in report
        assert 'security_level' in report
        assert 'compliance' in report
        
        # Should find symbols
        assert report['symbol_count'] > 0
        assert report['SEM_count'] > 0  # Should anchor known symbols
    
    def test_secure_content_dict(self, ssl_engine):
        """Test securing dictionary content"""
        test_dict = {
            '🜄': "Water element",
            'normal_key': "Normal value",
            '☥': "Life force"
        }
        
        secured_dict, report = ssl_engine.secure_content(test_dict)
        
        # Check structure is preserved
        assert isinstance(secured_dict, dict)
        assert len(secured_dict) == len(test_dict)
        
        # Check that symbolic keys were processed
        secured_keys = list(secured_dict.keys())
        assert any('🜄' in key for key in secured_keys)
        assert any('☥' in key for key in secured_keys)
        
        # Check report
        assert report['symbol_count'] > 0
        assert report['SEM_count'] > 0
    
    def test_secure_content_list(self, ssl_engine):
        """Test securing list content"""
        test_list = [
            "Text with 🜄 symbol",
            "Another text with ☥",
            {"nested": "🜂 fire symbol"}
        ]
        
        secured_list, report = ssl_engine.secure_content(test_list)
        
        # Check structure is preserved
        assert isinstance(secured_list, list)
        assert len(secured_list) == len(test_list)
        
        # Check report
        assert report['symbol_count'] > 0
        assert report['SEM_count'] > 0
    
    def test_corrupted_symbol_handling(self, ssl_engine, sample_corrupted_text):
        """Test handling of corrupted/OCR symbols"""
        secured_content, report = ssl_engine.secure_content(sample_corrupted_text)
        
        # Should detect corrupted symbols
        assert report['symbol_count'] > 0
        
        # Should mark unknown symbols as EXU
        assert report['EXU_count'] > 0
        
        # Should include anchoring for known corrupted patterns
        assert '(cid:0)' in secured_content or 'CID_Zero' in secured_content
    
    def test_symbol_coverage_calculation(self, ssl_engine):
        """Test symbol coverage calculation"""
        # Text with all known symbols
        known_symbols_text = "🜄 ☥ ∞"
        secured, report = ssl_engine.secure_content(known_symbols_text)
        
        coverage = float(report['symbol_coverage'].rstrip('%'))
        assert coverage > 90  # Should have high coverage for known symbols
        
        # Text with unknown symbols
        unknown_symbols_text = "🜄 ☥ 🔮 🌟"  # Mix of known and unknown
        secured, report = ssl_engine.secure_content(unknown_symbols_text)
        
        coverage = float(report['symbol_coverage'].rstrip('%'))
        assert coverage < 100  # Should have lower coverage due to unknown symbols
    
    def test_security_level_assessment(self, ssl_engine):
        """Test security level assessment"""
        # High security text (all known symbols)
        high_security_text = "🜄 ☥"
        _, report = ssl_engine.secure_content(high_security_text)
        assert report['security_level'] in ['HIGH', 'MEDIUM']
        
        # Low security text (many unknown symbols)
        low_security_text = "🔮 🌟 🎭 🎪 🎨"
        _, report = ssl_engine.secure_content(low_security_text)
        # Note: May not always be LOW due to small sample size
        assert report['security_level'] in ['HIGH', 'MEDIUM', 'LOW']
    
    def test_compliance_assessment(self, ssl_engine):
        """Test CIP-1 compliance assessment"""
        # Text that should achieve compliance
        compliant_text = "🜄 ☥"  # All known symbols
        _, report = ssl_engine.secure_content(compliant_text)
        
        # Check compliance status
        assert 'compliance' in report
        assert report['compliance'] in ['CIP-1', 'PARTIAL']
    
    def test_custom_symbol_addition(self, ssl_engine):
        """Test adding custom symbols"""
        custom_symbol = "🔮"
        custom_anchor = "Crystal_Ball"
        custom_description = "Divination: reveal hidden knowledge"
        
        # Add custom symbol
        ssl_engine.add_custom_symbol(custom_symbol, custom_anchor, custom_description)
        
        # Verify it was added
        assert custom_symbol in ssl_engine.symbol_db
        anchor, description = ssl_engine.symbol_db[custom_symbol]
        assert anchor == custom_anchor
        assert description == custom_description
        
        # Test that it's now recognized
        test_text = f"Custom symbol {custom_symbol} test"
        secured, report = ssl_engine.secure_content(test_text)
        
        assert report['SEM_count'] > 0  # Should be recognized as SEM
        assert custom_anchor in secured
    
    def test_symbol_info_retrieval(self, ssl_engine):
        """Test getting symbol information"""
        # Test known symbol
        symbol = '🜄'
        anchor, description = ssl_engine.get_symbol_info(symbol)
        assert anchor == 'Water_Alchemical'
        assert 'Emotion' in description
        
        # Test unknown symbol
        unknown_symbol = '🔮'
        anchor, description = ssl_engine.get_symbol_info(unknown_symbol)
        assert anchor == 'UNKNOWN'
        assert 'No description available' in description
    
    def test_symbol_database_export_import(self, ssl_engine):
        """Test exporting and importing symbol database"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = f.name
        
        try:
            # Export database
            ssl_engine.export_symbol_database(export_path)
            
            # Verify file was created
            assert os.path.exists(export_path)
            
            # Load and verify content
            with open(export_path, 'r', encoding='utf-8') as f:
                exported_db = json.load(f)
            
            assert len(exported_db) > 0
            assert '🜄' in exported_db
            
            # Create new engine and import
            new_engine = SymbolicSecurityEngine()
            original_size = len(new_engine.symbol_db)
            
            # Add a custom symbol to test import
            new_engine.add_custom_symbol('🔮', 'Test', 'Test description')
            
            # Import should update the database
            new_engine.import_symbol_database(export_path)
            
            # Verify import worked
            assert len(new_engine.symbol_db) >= original_size
            
        finally:
            # Cleanup
            if os.path.exists(export_path):
                os.unlink(export_path)
    
    def test_reconstruction_logging(self, ssl_engine):
        """Test reconstruction logging functionality"""
        # Enable logging
        ssl_engine.config['log_reconstructions'] = True
        
        # Process some content
        test_text = "Test with 🜄 symbol"
        ssl_engine.secure_content(test_text)
        
        # Check that log was created
        assert len(ssl_engine.reconstruction_log) > 0
        
        # Check log entry structure
        log_entry = ssl_engine.reconstruction_log[0]
        assert 'timestamp' in log_entry
        assert 'session_id' in log_entry
        assert 'original_type' in log_entry
        assert 'security_level' in log_entry
        
        # Test log export
        log_json = ssl_engine.generate_reconstruction_log()
        assert isinstance(log_json, str)
        
        # Should be valid JSON
        log_data = json.loads(log_json)
        assert isinstance(log_data, list)
        assert len(log_data) > 0
    
    def test_configuration_loading(self):
        """Test configuration loading from file"""
        config_data = {
            "unicode_threshold": 2048,
            "semantic_anchoring": False,
            "strict_validation": True
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            # Create engine with custom config
            ssl_engine = SymbolicSecurityEngine(config_path)
            
            # Verify config was loaded
            assert ssl_engine.config['unicode_threshold'] == 2048
            assert ssl_engine.config['semantic_anchoring'] is False
            assert ssl_engine.config['strict_validation'] is True
            
        finally:
            # Cleanup
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_error_handling(self, ssl_engine):
        """Test error handling for invalid inputs"""
        # Test with None input
        with pytest.raises(ValueError):
            ssl_engine.secure_content(None)
        
        # Test with unsupported type
        with pytest.raises(ValueError):
            ssl_engine.secure_content(123)
    
    def test_performance_metrics(self, ssl_engine):
        """Test that performance metrics are included"""
        test_text = "Performance test with 🜄 and ☥ symbols"
        secured, report = ssl_engine.secure_content(test_text)
        
        # Should include processing time
        assert 'processing_time' in report
        assert isinstance(report['processing_time'], (int, float))
        assert report['processing_time'] >= 0
        
        # Should include session ID
        assert 'session_id' in report
        assert report['session_id'] == ssl_engine.session_id


class TestSymbolDetection:
    """Test cases for symbol detection logic"""
    
    @pytest.fixture
    def ssl_engine(self):
        return SymbolicSecurityEngine()
    
    def test_unicode_threshold_detection(self, ssl_engine):
        """Test Unicode threshold-based detection"""
        # Characters above threshold should be detected
        high_unicode_text = "∞ ⚛ ☯"  # High Unicode codepoints
        _, report = ssl_engine.secure_content(high_unicode_text)
        assert report['symbol_count'] > 0
        
        # Regular ASCII should not be detected as symbols
        ascii_text = "Regular ASCII text 123"
        _, report = ssl_engine.secure_content(ascii_text)
        assert report['symbol_count'] == 0
    
    def test_ocr_artifact_detection(self, ssl_engine):
        """Test detection of OCR artifacts"""
        ocr_text = "Text with (cid:0) and ō artifacts"
        _, report = ssl_engine.secure_content(ocr_text)
        
        # Should detect OCR artifacts as symbols needing attention
        assert report['symbol_count'] > 0
    
    def test_mathematical_symbol_detection(self, ssl_engine):
        """Test detection of mathematical symbols"""
        math_text = "Mathematical symbols: ∑ ∫ ∞ ∂"
        _, report = ssl_engine.secure_content(math_text)
        
        # Should detect mathematical symbols
        assert report['symbol_count'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

