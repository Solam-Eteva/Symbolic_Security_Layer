#!/usr/bin/env python3
"""
Test suite for SSL Secure Data Preprocessor
Tests the SecureDataPreprocessor and SecureEmbeddingLayer classes
"""

import pytest
import tempfile
import pickle
import os
import sys
import numpy as np

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from symbolic_security_layer.core import SymbolicSecurityEngine
from symbolic_security_layer.preprocessor import SecureDataPreprocessor, SecureEmbeddingLayer

# Mock TensorFlow and PyTorch for testing
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


class TestSecureDataPreprocessor:
    """Test cases for SecureDataPreprocessor"""
    
    @pytest.fixture
    def ssl_engine(self):
        """Create SSL engine for testing"""
        return SymbolicSecurityEngine()
    
    @pytest.fixture
    def preprocessor(self, ssl_engine):
        """Create preprocessor for testing"""
        return SecureDataPreprocessor(ssl_engine)
    
    @pytest.fixture
    def sample_dataset(self):
        """Sample dataset with symbolic content"""
        return [
            "The alchemical symbol 🜄 represents water",
            "Ancient wisdom in ☥ symbolizes life force",
            "Mathematical infinity ∞ transcends limits",
            "Corrupted symbol (cid:0) needs reconstruction",
            "Regular text without symbols",
            "Another text with 🜂 fire symbol"
        ]
    
    @pytest.fixture
    def sample_labels(self):
        """Sample labels for classification"""
        return [1, 1, 1, 0, 0, 1]  # 1 = symbolic, 0 = non-symbolic
    
    def test_preprocessor_initialization(self, preprocessor):
        """Test preprocessor initializes correctly"""
        assert preprocessor is not None
        assert hasattr(preprocessor, 'security_engine')
        assert hasattr(preprocessor, 'config')
        assert hasattr(preprocessor, 'vocabulary')
        assert hasattr(preprocessor, 'reverse_vocabulary')
        assert hasattr(preprocessor, 'security_metrics')
        
        # Check default configuration
        assert preprocessor.config['max_vocab_size'] == 50000
        assert preprocessor.config['tokenization_strategy'] == 'word'
        assert preprocessor.config['batch_size'] == 1000
    
    def test_custom_configuration(self):
        """Test preprocessor with custom configuration"""
        custom_config = {
            'max_vocab_size': 10000,
            'tokenization_strategy': 'char',
            'batch_size': 500
        }
        
        preprocessor = SecureDataPreprocessor(config=custom_config)
        
        assert preprocessor.config['max_vocab_size'] == 10000
        assert preprocessor.config['tokenization_strategy'] == 'char'
        assert preprocessor.config['batch_size'] == 500
    
    def test_process_dataset_basic(self, preprocessor, sample_dataset, sample_labels):
        """Test basic dataset processing"""
        processed_data, security_report = preprocessor.process_dataset(sample_dataset, sample_labels)
        
        # Check processed data structure
        sequences, labels = processed_data
        assert isinstance(sequences, list)
        assert len(sequences) == len(sample_dataset)
        assert labels == sample_labels
        
        # Check security report
        assert isinstance(security_report, dict)
        assert 'processing_summary' in security_report
        assert 'security_distribution' in security_report
        assert 'vocabulary_analysis' in security_report
        assert 'recommendations' in security_report
        
        # Check processing summary
        proc_summary = security_report['processing_summary']
        assert proc_summary['total_texts_processed'] == len(sample_dataset)
        assert proc_summary['total_symbols_found'] > 0
        assert 'processing_time_seconds' in proc_summary
    
    def test_process_dataset_without_labels(self, preprocessor, sample_dataset):
        """Test dataset processing without labels"""
        processed_data, security_report = preprocessor.process_dataset(sample_dataset)
        
        # Should return just sequences
        assert isinstance(processed_data, list)
        assert len(processed_data) == len(sample_dataset)
        
        # Security report should still be generated
        assert isinstance(security_report, dict)
        assert 'processing_summary' in security_report
    
    def test_vocabulary_building_word_tokenization(self, preprocessor, sample_dataset):
        """Test vocabulary building with word tokenization"""
        preprocessor.config['tokenization_strategy'] = 'word'
        processed_data, _ = preprocessor.process_dataset(sample_dataset)
        
        # Check vocabulary was built
        assert len(preprocessor.vocabulary) > 0
        assert len(preprocessor.reverse_vocabulary) > 0
        
        # Check special tokens
        assert '<PAD>' in preprocessor.vocabulary
        assert '<UNK>' in preprocessor.vocabulary
        assert '<START>' in preprocessor.vocabulary
        assert '<END>' in preprocessor.vocabulary
        
        # Check reverse mapping
        for token, idx in preprocessor.vocabulary.items():
            assert preprocessor.reverse_vocabulary[idx] == token
    
    def test_vocabulary_building_char_tokenization(self, preprocessor, sample_dataset):
        """Test vocabulary building with character tokenization"""
        preprocessor.config['tokenization_strategy'] = 'char'
        processed_data, _ = preprocessor.process_dataset(sample_dataset)
        
        # Check vocabulary was built
        assert len(preprocessor.vocabulary) > 0
        
        # Should contain individual characters
        assert 'a' in preprocessor.vocabulary or 'A' in preprocessor.vocabulary
        assert ' ' in preprocessor.vocabulary  # Space character
    
    def test_texts_to_sequences(self, preprocessor, sample_dataset):
        """Test text to sequence conversion"""
        # First process to build vocabulary
        preprocessor.process_dataset(sample_dataset)
        
        # Test conversion
        test_texts = ["The symbol 🜄 test", "Another test"]
        sequences = preprocessor._texts_to_sequences(test_texts)
        
        assert isinstance(sequences, list)
        assert len(sequences) == len(test_texts)
        
        # Each sequence should be a list of integers
        for seq in sequences:
            assert isinstance(seq, list)
            assert all(isinstance(idx, int) for idx in seq)
    
    def test_vocabulary_security_analysis(self, preprocessor, sample_dataset):
        """Test vocabulary security analysis"""
        processed_data, security_report = preprocessor.process_dataset(sample_dataset)
        
        vocab_analysis = security_report['vocabulary_analysis']
        
        # Check vocabulary analysis structure
        assert 'total_vocabulary_size' in vocab_analysis
        assert 'secured_tokens_count' in vocab_analysis
        assert 'unsecured_tokens_count' in vocab_analysis
        assert 'security_ratio' in vocab_analysis
        
        # Check values are reasonable
        assert vocab_analysis['total_vocabulary_size'] > 0
        assert 0.0 <= vocab_analysis['security_ratio'] <= 1.0
    
    def test_batch_processing(self, preprocessor):
        """Test batch processing functionality"""
        # Create larger dataset to test batching
        large_dataset = [f"Text {i} with 🜄 symbol" for i in range(50)]
        
        preprocessor.config['batch_size'] = 10  # Small batch size for testing
        
        processed_data, security_report = preprocessor.process_dataset(large_dataset)
        
        # Should process all items
        assert len(processed_data) == len(large_dataset)
        
        # Should have processing summary
        proc_summary = security_report['processing_summary']
        assert proc_summary['total_texts_processed'] == len(large_dataset)
    
    def test_save_and_load_preprocessor(self, preprocessor, sample_dataset):
        """Test saving and loading preprocessor state"""
        # Process dataset to build vocabulary
        preprocessor.process_dataset(sample_dataset)
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            save_path = f.name
        
        try:
            # Save preprocessor
            preprocessor.save_preprocessor(save_path)
            
            # Create new preprocessor and load
            new_preprocessor = SecureDataPreprocessor()
            new_preprocessor.load_preprocessor(save_path)
            
            # Check that state was loaded correctly
            assert new_preprocessor.vocabulary == preprocessor.vocabulary
            assert new_preprocessor.reverse_vocabulary == preprocessor.reverse_vocabulary
            assert new_preprocessor.config == preprocessor.config
            
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)
    
    def test_decode_sequence(self, preprocessor, sample_dataset):
        """Test sequence decoding"""
        # Process dataset to build vocabulary
        processed_data, _ = preprocessor.process_dataset(sample_dataset)
        sequences = processed_data if isinstance(processed_data, list) else processed_data[0]
        
        # Test decoding
        if sequences:
            original_text = sample_dataset[0]
            sequence = sequences[0]
            decoded_text = preprocessor.decode_sequence(sequence)
            
            # Decoded text should contain meaningful content
            assert isinstance(decoded_text, str)
            assert len(decoded_text) > 0
    
    def test_get_embedding_matrix(self, preprocessor, sample_dataset):
        """Test embedding matrix generation"""
        # Process dataset to build vocabulary
        preprocessor.process_dataset(sample_dataset)
        
        embedding_dim = 100
        embedding_matrix = preprocessor.get_embedding_matrix(embedding_dim)
        
        # Check matrix dimensions
        vocab_size = len(preprocessor.vocabulary)
        assert embedding_matrix.shape == (vocab_size, embedding_dim)
        
        # Check matrix contains reasonable values
        assert not np.isnan(embedding_matrix).any()
        assert not np.isinf(embedding_matrix).any()
    
    def test_get_embedding_matrix_with_pretrained(self, preprocessor, sample_dataset):
        """Test embedding matrix with pretrained embeddings"""
        # Process dataset to build vocabulary
        preprocessor.process_dataset(sample_dataset)
        
        # Create mock pretrained embeddings
        pretrained = {
            'the': np.array([0.1, 0.2, 0.3]),
            'symbol': np.array([0.4, 0.5, 0.6])
        }
        
        embedding_matrix = preprocessor.get_embedding_matrix(3, pretrained)
        
        # Check that pretrained embeddings were used
        vocab_size = len(preprocessor.vocabulary)
        assert embedding_matrix.shape == (vocab_size, 3)
        
        # Check specific embeddings if tokens exist in vocabulary
        if 'the' in preprocessor.vocabulary:
            idx = preprocessor.vocabulary['the']
            np.testing.assert_array_equal(embedding_matrix[idx], pretrained['the'])
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_create_tensorflow_dataset(self, preprocessor, sample_dataset, sample_labels):
        """Test TensorFlow dataset creation"""
        processed_data, _ = preprocessor.process_dataset(sample_dataset, sample_labels)
        sequences, labels = processed_data
        
        tf_dataset = preprocessor.create_tensorflow_dataset(sequences, labels, batch_size=2)
        
        # Check dataset type
        assert isinstance(tf_dataset, tf.data.Dataset)
        
        # Check dataset can be iterated
        for batch in tf_dataset.take(1):
            x_batch, y_batch = batch
            assert x_batch.shape[0] <= 2  # Batch size
            assert y_batch.shape[0] <= 2
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_create_pytorch_dataset(self, preprocessor, sample_dataset, sample_labels):
        """Test PyTorch dataset creation"""
        processed_data, _ = preprocessor.process_dataset(sample_dataset, sample_labels)
        sequences, labels = processed_data
        
        pt_dataset = preprocessor.create_pytorch_dataset(sequences, labels)
        
        # Check dataset type
        assert isinstance(pt_dataset, torch.utils.data.TensorDataset)
        
        # Check dataset length
        assert len(pt_dataset) == len(sequences)
        
        # Check dataset items
        x, y = pt_dataset[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)


class TestSecureEmbeddingLayer:
    """Test cases for SecureEmbeddingLayer"""
    
    @pytest.fixture
    def ssl_engine(self):
        return SymbolicSecurityEngine()
    
    @pytest.fixture
    def secure_embedding(self, ssl_engine):
        return SecureEmbeddingLayer(vocab_size=1000, embedding_dim=128, security_engine=ssl_engine)
    
    def test_secure_embedding_initialization(self, secure_embedding):
        """Test secure embedding layer initialization"""
        assert secure_embedding is not None
        assert secure_embedding.vocab_size == 1000
        assert secure_embedding.embedding_dim == 128
        assert hasattr(secure_embedding, 'security_engine')
        assert hasattr(secure_embedding, 'security_metrics')
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_create_tensorflow_layer(self, secure_embedding):
        """Test TensorFlow embedding layer creation"""
        tf_layer = secure_embedding.create_tensorflow_layer(mask_zero=True)
        
        # Check layer type
        assert isinstance(tf_layer, tf.keras.layers.Layer)
        
        # Test layer with sample input
        sample_input = tf.constant([[1, 2, 3, 0], [4, 5, 0, 0]])
        output = tf_layer(sample_input)
        
        # Check output shape
        expected_shape = (2, 4, 128)  # (batch_size, seq_len, embedding_dim)
        assert output.shape == expected_shape
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_create_pytorch_layer(self, secure_embedding):
        """Test PyTorch embedding layer creation"""
        pt_layer = secure_embedding.create_pytorch_layer(padding_idx=0)
        
        # Check layer type
        assert isinstance(pt_layer, torch.nn.Module)
        
        # Test layer with sample input
        sample_input = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
        output = pt_layer(sample_input)
        
        # Check output shape
        expected_shape = (2, 4, 128)  # (batch_size, seq_len, embedding_dim)
        assert output.shape == expected_shape


class TestPreprocessorIntegration:
    """Integration tests for preprocessor with different components"""
    
    @pytest.fixture
    def preprocessor(self):
        return SecureDataPreprocessor()
    
    def test_end_to_end_processing(self, preprocessor):
        """Test end-to-end processing workflow"""
        # Sample data with various symbolic content
        dataset = [
            "Alchemical symbols: 🜄 🜂 ☥",
            "Mathematical notation: ∞ ∑ ∫",
            "Corrupted text: (cid:0) ō 잇",
            "Regular text without symbols",
            "Mixed content with 🜄 and normal text"
        ]
        labels = [1, 1, 0, 0, 1]
        
        # Process dataset
        processed_data, security_report = preprocessor.process_dataset(dataset, labels)
        sequences, processed_labels = processed_data
        
        # Verify processing
        assert len(sequences) == len(dataset)
        assert processed_labels == labels
        
        # Check security metrics
        assert security_report['processing_summary']['total_symbols_found'] > 0
        assert security_report['processing_summary']['total_symbols_anchored'] >= 0
        
        # Test sequence decoding
        for i, seq in enumerate(sequences):
            decoded = preprocessor.decode_sequence(seq)
            assert isinstance(decoded, str)
            # Decoded text should contain some meaningful content
            assert len(decoded.strip()) > 0
    
    def test_security_metrics_tracking(self, preprocessor):
        """Test security metrics tracking throughout processing"""
        dataset = [
            "High security: 🜄 ☥ ∞",  # All known symbols
            "Medium security: 🜄 🔮",   # Mix of known/unknown
            "Low security: 🔮 🌟 🎭",   # Mostly unknown
            "No symbols: regular text"
        ]
        
        processed_data, security_report = preprocessor.process_dataset(dataset)
        
        # Check security distribution
        sec_dist = security_report.get('security_distribution', {})
        assert isinstance(sec_dist, dict)
        
        # Should have various security levels
        total_items = sum(sec_dist.values())
        assert total_items == len(dataset)
        
        # Check vocabulary analysis
        vocab_analysis = security_report['vocabulary_analysis']
        assert vocab_analysis['total_vocabulary_size'] > 0
    
    def test_recommendation_generation(self, preprocessor):
        """Test recommendation generation based on processing results"""
        # Dataset with security issues
        dataset = [
            "Unknown symbols: 🔮 🌟 🎭 🎪",
            "Corrupted: (cid:0) ō 잇",
            "Mixed: 🜄 with 🔮 unknown"
        ]
        
        processed_data, security_report = preprocessor.process_dataset(dataset)
        
        # Should generate recommendations
        recommendations = security_report.get('recommendations', [])
        assert isinstance(recommendations, list)
        
        # Should recommend adding unknown symbols
        symbol_recs = [r for r in recommendations if 'symbol' in r.lower()]
        assert len(symbol_recs) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

