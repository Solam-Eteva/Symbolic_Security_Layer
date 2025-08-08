#!/usr/bin/env python3
"""
Test suite for SSL AI Integration Adapters
Tests the AIIntegrationAdapter and specialized adapters
"""

import pytest
import json
import os
import sys

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from symbolic_security_layer.core import SymbolicSecurityEngine
from symbolic_security_layer.adapters import AIIntegrationAdapter, OpenAIAdapter, HuggingFaceAdapter


class TestAIIntegrationAdapter:
    """Test cases for AIIntegrationAdapter"""
    
    @pytest.fixture
    def ssl_engine(self):
        """Create SSL engine for testing"""
        return SymbolicSecurityEngine()
    
    @pytest.fixture
    def ai_adapter(self, ssl_engine):
        """Create AI adapter for testing"""
        return AIIntegrationAdapter(ssl_engine)
    
    @pytest.fixture
    def sample_prompt(self):
        """Sample prompt with symbolic content"""
        return "Analyze the alchemical symbol 🜄 and its relationship to ☥ in consciousness studies"
    
    @pytest.fixture
    def sample_response(self):
        """Sample AI response with potential synthetic content"""
        return """According to authoritative sources, the patent ZL201510000000 demonstrates 
        the relationship between 🜄 (water) and ☥ (ankh) symbols. Based on GB/T 7714-2015 
        standards, this connection is well-documented in peer-reviewed research."""
    
    def test_adapter_initialization(self, ai_adapter):
        """Test adapter initializes correctly"""
        assert ai_adapter is not None
        assert hasattr(ai_adapter, 'security_engine')
        assert hasattr(ai_adapter, 'synthetic_patterns')
        
        # Check synthetic patterns are loaded
        assert len(ai_adapter.synthetic_patterns) > 0
        assert 'authoritative_phrases' in ai_adapter.synthetic_patterns
        assert 'patent_numbers' in ai_adapter.synthetic_patterns
        assert 'standards_references' in ai_adapter.synthetic_patterns
    
    def test_wrap_prompt(self, ai_adapter, sample_prompt):
        """Test prompt wrapping functionality"""
        wrapped = ai_adapter.wrap_prompt(sample_prompt)
        
        # Check wrapper structure
        assert isinstance(wrapped, dict)
        assert 'original' in wrapped
        assert 'secured' in wrapped
        assert 'validation' in wrapped
        assert 'compliance' in wrapped
        assert 'timestamp' in wrapped
        
        # Check content
        assert wrapped['original'] == sample_prompt
        assert len(wrapped['secured']) >= len(sample_prompt)
        
        # Check validation report
        validation = wrapped['validation']
        assert 'symbol_count' in validation
        assert 'SEM_count' in validation
        assert 'security_level' in validation
    
    def test_wrap_prompt_with_metadata(self, ai_adapter, sample_prompt):
        """Test prompt wrapping with metadata"""
        metadata = {'model': 'test-model', 'temperature': 0.7}
        wrapped = ai_adapter.wrap_prompt(sample_prompt, metadata)
        
        assert 'metadata' in wrapped
        assert wrapped['metadata'] == metadata
    
    def test_process_response(self, ai_adapter, sample_response):
        """Test response processing functionality"""
        processed = ai_adapter.process_response(sample_response)
        
        # Check response structure
        assert isinstance(processed, dict)
        assert 'original_response' in processed
        assert 'secured_response' in processed
        assert 'security_report' in processed
        assert 'synthetic_analysis' in processed
        assert 'risk_assessment' in processed
        assert 'recommendations' in processed
        
        # Check content
        assert processed['original_response'] == sample_response
        
        # Check synthetic analysis
        synthetic_analysis = processed['synthetic_analysis']
        assert 'detected_patterns' in synthetic_analysis
        assert 'confidence_scores' in synthetic_analysis
        assert 'total_matches' in synthetic_analysis
        
        # Should detect synthetic patterns in sample response
        assert synthetic_analysis['total_matches'] > 0
        assert len(synthetic_analysis['detected_patterns']) > 0
    
    def test_synthetic_content_detection(self, ai_adapter):
        """Test synthetic content detection patterns"""
        # Test authoritative phrases
        auth_text = "According to authoritative sources and verified data"
        analysis = ai_adapter._analyze_synthetic_content(auth_text)
        assert 'authoritative_phrases' in analysis['detected_patterns']
        
        # Test patent numbers
        patent_text = "Patent ZL201510000000 and US1234567 demonstrate"
        analysis = ai_adapter._analyze_synthetic_content(patent_text)
        assert 'patent_numbers' in analysis['detected_patterns']
        
        # Test standards references
        standards_text = "According to GB/T 7714-2015 and IEEE 802.11"
        analysis = ai_adapter._analyze_synthetic_content(standards_text)
        assert 'standards_references' in analysis['detected_patterns']
        
        # Test citation formats
        citation_text = "Research by Smith et al., 2023 and (Johnson 2022)"
        analysis = ai_adapter._analyze_synthetic_content(citation_text)
        assert 'citation_formats' in analysis['detected_patterns']
    
    def test_risk_assessment(self, ai_adapter, sample_response):
        """Test risk assessment calculation"""
        processed = ai_adapter.process_response(sample_response)
        risk_assessment = processed['risk_assessment']
        
        # Check risk assessment structure
        assert 'risk_level' in risk_assessment
        assert 'risk_score' in risk_assessment
        assert 'risk_factors' in risk_assessment
        assert 'requires_verification' in risk_assessment
        
        # Check risk level is valid
        assert risk_assessment['risk_level'] in ['HIGH', 'MEDIUM', 'LOW']
        
        # Check risk score is in valid range
        assert 0.0 <= risk_assessment['risk_score'] <= 1.0
        
        # Should identify risk factors for sample response
        assert len(risk_assessment['risk_factors']) > 0
    
    def test_recommendations_generation(self, ai_adapter, sample_response):
        """Test recommendations generation"""
        processed = ai_adapter.process_response(sample_response)
        recommendations = processed['recommendations']
        
        # Should generate recommendations
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should include verification recommendations for synthetic content
        verification_recs = [r for r in recommendations if 'verify' in r.lower() or 'check' in r.lower()]
        assert len(verification_recs) > 0
    
    def test_batch_processing(self, ai_adapter):
        """Test batch processing functionality"""
        items = [
            "Text with 🜄 symbol",
            {"prompt": "Analyze ☥ symbol", "metadata": {"model": "test"}},
            "Another text with synthetic content according to sources"
        ]
        
        results = ai_adapter.batch_process(items, batch_size=2)
        
        # Check results structure
        assert isinstance(results, list)
        assert len(results) == len(items)
        
        # Check each result
        for result in results:
            assert isinstance(result, dict)
            assert 'type' in result or 'data' in result
    
    def test_external_reference_verification(self, ai_adapter):
        """Test external reference verification"""
        # Test patent verification
        patent_ref = "ZL201510000000"
        verification = ai_adapter.verify_external_reference(patent_ref, "patent")
        
        assert 'reference' in verification
        assert 'type' in verification
        assert 'verified' in verification
        assert 'confidence' in verification
        assert 'notes' in verification
        
        assert verification['reference'] == patent_ref
        assert verification['type'] == 'patent'
        
        # Test standard verification
        standard_ref = "GB/T 7714-2015"
        verification = ai_adapter.verify_external_reference(standard_ref, "standard")
        
        assert verification['type'] == 'standard'
        
        # Test auto-detection
        auto_verification = ai_adapter.verify_external_reference(patent_ref, "auto")
        assert auto_verification['type'] == 'patent'
    
    def test_reference_type_detection(self, ai_adapter):
        """Test reference type detection"""
        # Test patent detection
        assert ai_adapter._detect_reference_type("ZL201510000000") == "patent"
        assert ai_adapter._detect_reference_type("US1234567") == "patent"
        
        # Test standard detection
        assert ai_adapter._detect_reference_type("GB/T 7714-2015") == "standard"
        assert ai_adapter._detect_reference_type("IEEE 802.11") == "ieee_standard"
        
        # Test DOI detection
        assert ai_adapter._detect_reference_type("doi:10.1000/123456") == "doi"
        
        # Test unknown
        assert ai_adapter._detect_reference_type("unknown_reference") == "unknown"
    
    def test_patent_pattern_verification(self, ai_adapter):
        """Test patent pattern verification"""
        # Valid patent format
        valid_patent = "ZL201510000000"
        result = ai_adapter._verify_patent_pattern(valid_patent)
        assert result['confidence'] > 0.5
        
        # Suspicious patent (all zeros)
        suspicious_patent = "ZL201510000000"
        result = ai_adapter._verify_patent_pattern(suspicious_patent)
        # Should detect suspicious pattern
        assert any('suspicious' in note.lower() for note in result['notes'])
        
        # Invalid format
        invalid_patent = "INVALID123"
        result = ai_adapter._verify_patent_pattern(invalid_patent)
        assert result['confidence'] < 0.5


class TestOpenAIAdapter:
    """Test cases for OpenAI-specific adapter"""
    
    @pytest.fixture
    def openai_adapter(self):
        """Create OpenAI adapter for testing"""
        return OpenAIAdapter(api_key="test-key")
    
    def test_openai_adapter_initialization(self, openai_adapter):
        """Test OpenAI adapter initialization"""
        assert openai_adapter is not None
        assert hasattr(openai_adapter, 'api_key')
        assert openai_adapter.api_key == "test-key"
    
    def test_secure_chat_completion(self, openai_adapter):
        """Test secure chat completion processing"""
        messages = [
            {"role": "user", "content": "Analyze 🜄 symbol"},
            {"role": "assistant", "content": "The symbol represents water"}
        ]
        
        result = openai_adapter.secure_chat_completion(messages)
        
        # Check result structure
        assert 'secured_messages' in result
        assert 'security_reports' in result
        assert 'original_messages' in result
        assert 'ssl_metadata' in result
        
        # Check secured messages
        secured_messages = result['secured_messages']
        assert len(secured_messages) == len(messages)
        
        # Check SSL metadata
        ssl_metadata = result['ssl_metadata']
        assert 'total_symbols' in ssl_metadata
        assert 'anchored_symbols' in ssl_metadata
        assert 'compliance_level' in ssl_metadata


class TestHuggingFaceAdapter:
    """Test cases for Hugging Face-specific adapter"""
    
    @pytest.fixture
    def hf_adapter(self):
        """Create Hugging Face adapter for testing"""
        return HuggingFaceAdapter(model_name="test-model")
    
    def test_hf_adapter_initialization(self, hf_adapter):
        """Test Hugging Face adapter initialization"""
        assert hf_adapter is not None
        assert hasattr(hf_adapter, 'model_name')
        assert hf_adapter.model_name == "test-model"
    
    def test_secure_text_generation(self, hf_adapter):
        """Test secure text generation processing"""
        prompt = "Generate text about 🜄 symbol"
        
        result = hf_adapter.secure_text_generation(prompt)
        
        # Check result structure
        assert 'secured_prompt' in result
        assert 'security_metadata' in result
        assert 'model_name' in result
        assert 'ssl_compliance' in result
        
        # Check content
        assert result['model_name'] == "test-model"
        assert len(result['secured_prompt']) >= len(prompt)


class TestSyntheticPatterns:
    """Test cases for synthetic content pattern detection"""
    
    @pytest.fixture
    def ai_adapter(self):
        return AIIntegrationAdapter()
    
    def test_authoritative_phrase_patterns(self, ai_adapter):
        """Test detection of authoritative phrases"""
        test_cases = [
            "According to authoritative sources",
            "Based on verified data",
            "Refer to official documentation",
            "As per established standards",
            "Confirmed by peer-reviewed research"
        ]
        
        for text in test_cases:
            analysis = ai_adapter._analyze_synthetic_content(text)
            assert 'authoritative_phrases' in analysis['detected_patterns']
    
    def test_patent_number_patterns(self, ai_adapter):
        """Test detection of patent number patterns"""
        test_cases = [
            "ZL201510000000",
            "US1234567",
            "EP1234567",
            "CN202010123456"
        ]
        
        for patent in test_cases:
            text = f"Patent {patent} demonstrates"
            analysis = ai_adapter._analyze_synthetic_content(text)
            assert 'patent_numbers' in analysis['detected_patterns']
    
    def test_standards_reference_patterns(self, ai_adapter):
        """Test detection of standards reference patterns"""
        test_cases = [
            "GB/T 7714-2015",
            "IEEE 802.11",
            "ISO 9001",
            "ANSI X3.4"
        ]
        
        for standard in test_cases:
            text = f"According to {standard} standard"
            analysis = ai_adapter._analyze_synthetic_content(text)
            assert 'standards_references' in analysis['detected_patterns']
    
    def test_citation_format_patterns(self, ai_adapter):
        """Test detection of citation format patterns"""
        test_cases = [
            "(Smith et al., 2023)",
            "[Johnson 2022]",
            "Brown, A. (2021)",
            "doi:10.1000/123456"
        ]
        
        for citation in test_cases:
            text = f"Research shows {citation} that"
            analysis = ai_adapter._analyze_synthetic_content(text)
            assert 'citation_formats' in analysis['detected_patterns']
    
    def test_confidence_scoring(self, ai_adapter):
        """Test confidence scoring for detected patterns"""
        # Text with multiple patterns
        multi_pattern_text = """According to authoritative sources and patent ZL201510000000, 
        the GB/T 7714-2015 standard demonstrates (Smith et al., 2023) findings."""
        
        analysis = ai_adapter._analyze_synthetic_content(multi_pattern_text)
        
        # Should have confidence scores for detected categories
        assert 'confidence_scores' in analysis
        
        for category in analysis['detected_patterns']:
            assert category in analysis['confidence_scores']
            score = analysis['confidence_scores'][category]
            assert 0.0 <= score <= 1.0
    
    def test_risk_indicator_generation(self, ai_adapter):
        """Test risk indicator generation"""
        # High-risk text with many patterns
        high_risk_text = """According to authoritative sources, patents ZL201510000000, 
        US1234567, and EP7654321 demonstrate compliance with GB/T 7714-2015, 
        IEEE 802.11, and ISO 9001 standards. Research by Smith et al. (2023), 
        Johnson (2022), and Brown et al. (2021) confirms these findings."""
        
        analysis = ai_adapter._analyze_synthetic_content(high_risk_text)
        
        # Should generate multiple risk indicators
        assert 'risk_indicators' in analysis
        assert len(analysis['risk_indicators']) > 0
        
        # Should detect high pattern density
        assert 'HIGH_PATTERN_DENSITY' in analysis['risk_indicators']
        assert 'UNVERIFIED_PATENTS' in analysis['risk_indicators']
        assert 'UNVERIFIED_STANDARDS' in analysis['risk_indicators']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

