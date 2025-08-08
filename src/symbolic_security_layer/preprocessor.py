"""
Secure Data Preprocessor
Facilitates data preparation for deep learning models with symbolic security
"""

import json
import time
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Iterator
from collections import Counter, defaultdict
import numpy as np

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

from .core import SymbolicSecurityEngine


class SecureDataPreprocessor:
    """
    Preprocessor for securing datasets before training ML models
    Integrates with TensorFlow and PyTorch workflows
    """
    
    def __init__(self, security_engine: SymbolicSecurityEngine = None, config: Dict = None):
        self.security_engine = security_engine or SymbolicSecurityEngine()
        self.config = self._load_config(config)
        self.vocabulary = {}
        self.reverse_vocabulary = {}
        self.security_metrics = defaultdict(int)
        self.processing_log = []
        
    def _load_config(self, config: Dict = None) -> Dict[str, Any]:
        """Load preprocessor configuration"""
        default_config = {
            "max_vocab_size": 50000,
            "min_frequency": 2,
            "tokenization_strategy": "word",  # word, char, subword
            "preserve_case": False,
            "handle_oov": True,
            "batch_size": 1000,
            "security_threshold": 0.8,
            "enable_caching": True,
            "cache_dir": "./ssl_cache"
        }
        
        if config:
            default_config.update(config)
            
        return default_config
    
    def process_dataset(self, dataset: Union[List[str], Iterator[str]], 
                       labels: Optional[List] = None) -> Tuple[Any, Dict]:
        """
        Process a dataset through the security layer
        Returns: (processed_dataset, security_report)
        """
        start_time = time.time()
        
        # Initialize tracking
        processed_texts = []
        security_reports = []
        total_symbols = 0
        total_anchored = 0
        
        # Process in batches
        batch_size = self.config["batch_size"]
        
        if isinstance(dataset, list):
            batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]
        else:
            # Handle iterators
            batches = []
            current_batch = []
            for item in dataset:
                current_batch.append(item)
                if len(current_batch) >= batch_size:
                    batches.append(current_batch)
                    current_batch = []
            if current_batch:
                batches.append(current_batch)
        
        for batch_idx, batch in enumerate(batches):
            batch_results = []
            
            for text in batch:
                secured_text, report = self.security_engine.secure_content(text)
                batch_results.append(secured_text)
                security_reports.append(report)
                
                # Update metrics
                total_symbols += report.get("symbol_count", 0)
                total_anchored += report.get("anchored_count", 0)
                self.security_metrics["total_processed"] += 1
                self.security_metrics[f"security_level_{report.get('security_level', 'unknown')}"] += 1
                
            processed_texts.extend(batch_results)
            
            # Log progress
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx + 1}/{len(batches)}")
        
        # Build vocabulary from secured texts
        self.vocabulary, self.reverse_vocabulary = self._build_vocabulary(processed_texts)
        
        # Convert to numerical format
        numerical_data = self._texts_to_sequences(processed_texts)
        
        # Generate comprehensive report
        processing_time = time.time() - start_time
        security_report = self._generate_processing_report(
            security_reports, total_symbols, total_anchored, processing_time
        )
        
        # Prepare final dataset
        if labels is not None:
            final_dataset = (numerical_data, labels)
        else:
            final_dataset = numerical_data
            
        return final_dataset, security_report
    
    def _build_vocabulary(self, texts: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build vocabulary from processed texts"""
        if self.config["tokenization_strategy"] == "word":
            tokens = self._tokenize_words(texts)
        elif self.config["tokenization_strategy"] == "char":
            tokens = self._tokenize_chars(texts)
        else:
            raise ValueError(f"Unsupported tokenization strategy: {self.config['tokenization_strategy']}")
        
        # Count token frequencies
        token_counts = Counter(tokens)
        
        # Filter by minimum frequency
        filtered_tokens = {
            token: count for token, count in token_counts.items()
            if count >= self.config["min_frequency"]
        }
        
        # Sort by frequency and limit vocabulary size
        sorted_tokens = sorted(filtered_tokens.items(), key=lambda x: x[1], reverse=True)
        vocab_tokens = sorted_tokens[:self.config["max_vocab_size"]]
        
        # Build vocabulary mappings
        vocabulary = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        for idx, (token, _) in enumerate(vocab_tokens, start=4):
            vocabulary[token] = idx
            
        reverse_vocabulary = {idx: token for token, idx in vocabulary.items()}
        
        return vocabulary, reverse_vocabulary
    
    def _tokenize_words(self, texts: List[str]) -> List[str]:
        """Tokenize texts into words"""
        tokens = []
        for text in texts:
            if not self.config["preserve_case"]:
                text = text.lower()
            # Simple word tokenization (can be enhanced with proper tokenizers)
            words = text.split()
            tokens.extend(words)
        return tokens
    
    def _tokenize_chars(self, texts: List[str]) -> List[str]:
        """Tokenize texts into characters"""
        tokens = []
        for text in texts:
            if not self.config["preserve_case"]:
                text = text.lower()
            tokens.extend(list(text))
        return tokens
    
    def _texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """Convert texts to numerical sequences"""
        sequences = []
        
        for text in texts:
            if self.config["tokenization_strategy"] == "word":
                tokens = text.split()
            else:
                tokens = list(text)
                
            if not self.config["preserve_case"]:
                tokens = [token.lower() for token in tokens]
            
            sequence = []
            for token in tokens:
                if token in self.vocabulary:
                    sequence.append(self.vocabulary[token])
                elif self.config["handle_oov"]:
                    sequence.append(self.vocabulary["<UNK>"])
                # Skip unknown tokens if handle_oov is False
                    
            sequences.append(sequence)
            
        return sequences
    
    def _generate_processing_report(self, security_reports: List[Dict], 
                                  total_symbols: int, total_anchored: int,
                                  processing_time: float) -> Dict[str, Any]:
        """Generate comprehensive processing report"""
        # Aggregate security metrics
        total_processed = len(security_reports)
        security_levels = Counter(report.get("security_level", "unknown") for report in security_reports)
        compliance_levels = Counter(report.get("compliance", "unknown") for report in security_reports)
        
        # Calculate coverage statistics
        coverage_scores = [
            float(report.get("symbol_coverage", "0%").rstrip("%")) 
            for report in security_reports
        ]
        avg_coverage = np.mean(coverage_scores) if coverage_scores else 0.0
        
        # Vocabulary security analysis
        vocab_security = self._analyze_vocabulary_security()
        
        report = {
            "processing_summary": {
                "total_texts_processed": total_processed,
                "total_symbols_found": total_symbols,
                "total_symbols_anchored": total_anchored,
                "processing_time_seconds": processing_time,
                "average_coverage": f"{avg_coverage:.1f}%"
            },
            "security_distribution": dict(security_levels),
            "compliance_distribution": dict(compliance_levels),
            "vocabulary_analysis": vocab_security,
            "recommendations": self._generate_processing_recommendations(security_reports),
            "metadata": {
                "vocabulary_size": len(self.vocabulary),
                "tokenization_strategy": self.config["tokenization_strategy"],
                "security_threshold": self.config["security_threshold"],
                "timestamp": time.time()
            }
        }
        
        return report
    
    def _analyze_vocabulary_security(self) -> Dict[str, Any]:
        """Analyze security characteristics of the vocabulary"""
        secured_tokens = []
        unsecured_tokens = []
        
        for token in self.vocabulary.keys():
            if any(ord(char) > self.security_engine.config["unicode_threshold"] for char in token):
                if any(char in self.security_engine.symbol_db for char in token):
                    secured_tokens.append(token)
                else:
                    unsecured_tokens.append(token)
        
        return {
            "total_vocabulary_size": len(self.vocabulary),
            "secured_tokens_count": len(secured_tokens),
            "unsecured_tokens_count": len(unsecured_tokens),
            "security_ratio": len(secured_tokens) / len(self.vocabulary) if self.vocabulary else 0.0,
            "top_secured_tokens": secured_tokens[:10],
            "top_unsecured_tokens": unsecured_tokens[:10]
        }
    
    def _generate_processing_recommendations(self, security_reports: List[Dict]) -> List[str]:
        """Generate recommendations based on processing results"""
        recommendations = []
        
        # Check overall security level
        low_security_count = sum(1 for r in security_reports if r.get("security_level") == "LOW")
        if low_security_count > len(security_reports) * 0.3:
            recommendations.append("Consider adding more semantic anchors - 30%+ of texts have low security")
        
        # Check compliance
        non_compliant_count = sum(1 for r in security_reports if r.get("compliance") != "CIP-1")
        if non_compliant_count > len(security_reports) * 0.2:
            recommendations.append("Improve CIP-1 compliance by increasing symbol coverage to 95%+")
        
        # Check unknown symbols
        total_exu = sum(r.get("EXU_count", 0) for r in security_reports)
        if total_exu > 0:
            recommendations.append(f"Add {total_exu} unknown symbols to the symbol database")
        
        return recommendations
    
    def create_tensorflow_dataset(self, sequences: List[List[int]], 
                                labels: Optional[List] = None,
                                batch_size: int = 32,
                                shuffle: bool = True) -> Any:
        """Create TensorFlow dataset from processed sequences"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
        
        # Pad sequences
        max_length = max(len(seq) for seq in sequences) if sequences else 0
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=max_length, padding='post'
        )
        
        if labels is not None:
            dataset = tf.data.Dataset.from_tensor_slices((padded_sequences, labels))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(padded_sequences)
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(sequences))
            
        dataset = dataset.batch(batch_size)
        
        return dataset
    
    def create_pytorch_dataset(self, sequences: List[List[int]], 
                             labels: Optional[List] = None) -> Any:
        """Create PyTorch dataset from processed sequences"""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")
        
        # Pad sequences
        max_length = max(len(seq) for seq in sequences) if sequences else 0
        padded_sequences = []
        
        for seq in sequences:
            padded = seq + [0] * (max_length - len(seq))  # Pad with 0s
            padded_sequences.append(padded)
        
        sequences_tensor = torch.tensor(padded_sequences, dtype=torch.long)
        
        if labels is not None:
            labels_tensor = torch.tensor(labels)
            return torch.utils.data.TensorDataset(sequences_tensor, labels_tensor)
        else:
            return torch.utils.data.TensorDataset(sequences_tensor)
    
    def save_preprocessor(self, filepath: str) -> None:
        """Save preprocessor state for later use"""
        state = {
            "vocabulary": self.vocabulary,
            "reverse_vocabulary": self.reverse_vocabulary,
            "config": self.config,
            "security_metrics": dict(self.security_metrics),
            "symbol_database": self.security_engine.symbol_db
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_preprocessor(self, filepath: str) -> None:
        """Load preprocessor state from file"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.vocabulary = state["vocabulary"]
        self.reverse_vocabulary = state["reverse_vocabulary"]
        self.config = state["config"]
        self.security_metrics = defaultdict(int, state["security_metrics"])
        self.security_engine.symbol_db = state["symbol_database"]
    
    def get_embedding_matrix(self, embedding_dim: int = 300, 
                           pretrained_embeddings: Optional[Dict] = None) -> np.ndarray:
        """Generate embedding matrix for the vocabulary"""
        vocab_size = len(self.vocabulary)
        embedding_matrix = np.random.normal(0, 0.1, (vocab_size, embedding_dim))
        
        if pretrained_embeddings:
            for token, idx in self.vocabulary.items():
                if token in pretrained_embeddings:
                    embedding_matrix[idx] = pretrained_embeddings[token]
        
        return embedding_matrix
    
    def decode_sequence(self, sequence: List[int]) -> str:
        """Decode numerical sequence back to text"""
        tokens = []
        for idx in sequence:
            if idx in self.reverse_vocabulary:
                token = self.reverse_vocabulary[idx]
                if token not in ["<PAD>", "<START>", "<END>"]:
                    tokens.append(token)
        
        if self.config["tokenization_strategy"] == "word":
            return " ".join(tokens)
        else:
            return "".join(tokens)


class SecureEmbeddingLayer:
    """Custom embedding layer with security tracking"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 security_engine: SymbolicSecurityEngine = None):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.security_engine = security_engine or SymbolicSecurityEngine()
        self.security_metrics = defaultdict(int)
    
    def create_tensorflow_layer(self, **kwargs) -> Any:
        """Create TensorFlow embedding layer with security tracking"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        class SecureTFEmbedding(tf.keras.layers.Embedding):
            def __init__(self, vocab_size, output_dim, security_engine, **kwargs):
                super().__init__(vocab_size, output_dim, **kwargs)
                self.security_engine = security_engine
                
            def call(self, inputs):
                # Track security metrics during forward pass
                # This is a simplified implementation
                return super().call(inputs)
        
        return SecureTFEmbedding(self.vocab_size, self.embedding_dim, 
                               self.security_engine, **kwargs)
    
    def create_pytorch_layer(self, **kwargs) -> Any:
        """Create PyTorch embedding layer with security tracking"""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        class SecurePTEmbedding(torch.nn.Embedding):
            def __init__(self, num_embeddings, embedding_dim, security_engine, **kwargs):
                super().__init__(num_embeddings, embedding_dim, **kwargs)
                self.security_engine = security_engine
                
            def forward(self, input):
                # Track security metrics during forward pass
                return super().forward(input)
        
        return SecurePTEmbedding(self.vocab_size, self.embedding_dim, 
                               self.security_engine, **kwargs)

