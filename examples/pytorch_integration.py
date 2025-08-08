#!/usr/bin/env python3
"""
PyTorch Integration Example for Symbolic Security Layer
Demonstrates how to use SSL with PyTorch models
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from symbolic_security_layer import SymbolicSecurityEngine, SecureDataPreprocessor, SecurityDiagnosticsReport

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available. Install with: pip install torch")
    PYTORCH_AVAILABLE = False


class SymbolicTextClassifier(nn.Module):
    """PyTorch model for symbolic text classification"""
    
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=32, num_classes=1):
        super(SymbolicTextClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Embedding layer
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM layer
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        
        # Dropout and classification
        dropped = self.dropout(last_hidden)
        logits = self.classifier(dropped)
        output = self.sigmoid(logits)
        
        return output


class SecurePyTorchTrainer:
    """Trainer class for PyTorch models with SSL integration"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output.squeeze(), target.float())
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            predicted = (output.squeeze() > 0.5).float()
            total += target.size(0)
            correct += (predicted == target.float()).sum().item()
            
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output.squeeze(), target.float())
                
                total_loss += loss.item()
                predicted = (output.squeeze() > 0.5).float()
                total += target.size(0)
                correct += (predicted == target.float()).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy


def create_sample_symbolic_dataset():
    """Create a sample dataset with symbolic content for demonstration"""
    symbolic_texts = [
        "The alchemical symbol 🜄 represents water and emotional receptivity",
        "Ancient wisdom encoded in ☥ (ankh) symbolizes life force",
        "Mathematical infinity ∞ transcends finite understanding",
        "The eye symbol 👁 watches over consciousness awakening",
        "Sacred geometry in ⬟ reveals hidden patterns",
        "Transformation through 🜂 (fire) initiates change",
        "Balance achieved via ☯ (yin-yang) harmonizes opposites",
        "The wheel ☸ guides cyclical wisdom and dharma",
        "Protection symbols ⛨ establish sacred boundaries",
        "Unity consciousness ॐ connects all existence",
        "Corrupted symbols like (cid:0) need reconstruction",
        "OCR artifacts ō require semantic anchoring",
        "Unknown glyphs 잇 challenge AI understanding",
        "Rare Unicode ⚛ demands special attention",
        "Chemical formulas H₂O contain subscript symbols",
        "Regular text without symbols is less complex",
        "Simple sentences have minimal symbolic content",
        "Plain language lacks esoteric meanings",
        "Basic communication uses common words",
        "Standard text follows conventional patterns"
    ]
    
    # Create labels (0 = low symbolic content, 1 = high symbolic content)
    labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    
    return symbolic_texts, labels


def demonstrate_secure_preprocessing():
    """Demonstrate secure data preprocessing for PyTorch"""
    print("=== Secure Data Preprocessing Demo ===")
    
    # Create sample dataset
    texts, labels = create_sample_symbolic_dataset()
    
    # Initialize secure preprocessor
    ssl_engine = SymbolicSecurityEngine()
    preprocessor = SecureDataPreprocessor(ssl_engine)
    
    print(f"Processing {len(texts)} texts with symbolic content...")
    
    # Process dataset through SSL
    processed_data, security_report = preprocessor.process_dataset(texts, labels)
    sequences, labels = processed_data
    
    print("\n--- Security Report ---")
    print(f"Total symbols found: {security_report['processing_summary']['total_symbols_found']}")
    print(f"Symbols anchored: {security_report['processing_summary']['total_symbols_anchored']}")
    print(f"Average coverage: {security_report['processing_summary']['average_coverage']}")
    print(f"Security distribution: {security_report['security_distribution']}")
    print(f"Vocabulary size: {security_report['vocabulary_analysis']['total_vocabulary_size']}")
    
    return preprocessor, sequences, labels, security_report


def demonstrate_pytorch_training():
    """Demonstrate training a PyTorch model with SSL-processed data"""
    if not PYTORCH_AVAILABLE:
        print("PyTorch not available for training demo")
        return
    
    print("\n=== PyTorch Training Demo ===")
    
    # Get preprocessed data
    preprocessor, sequences, labels, security_report = demonstrate_secure_preprocessing()
    
    # Create PyTorch dataset
    pytorch_dataset = preprocessor.create_pytorch_dataset(sequences, labels)
    
    # Split into train/test
    train_size = int(0.8 * len(pytorch_dataset))
    test_size = len(pytorch_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        pytorch_dataset, [train_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Create model
    vocab_size = len(preprocessor.vocabulary)
    model = SymbolicTextClassifier(vocab_size, embedding_dim=64, hidden_dim=32)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = SecurePyTorchTrainer(model, device)
    
    print(f"Training on device: {device}")
    
    # Train model
    print("\nTraining model with SSL-secured data...")
    num_epochs = 10
    
    for epoch in range(num_epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader)
        test_loss, test_acc = trainer.evaluate(test_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Test with new symbolic text
    test_text = "The sacred symbol ⚛ represents atomic consciousness 🜄"
    secured_test, test_report = preprocessor.security_engine.secure_content(test_text)
    
    print(f"\n--- Test Prediction ---")
    print(f"Original text: {test_text}")
    print(f"Secured text: {secured_test}")
    print(f"Security level: {test_report['security_level']}")
    
    # Convert to sequence and predict
    test_sequence = preprocessor._texts_to_sequences([secured_test])
    
    # Pad to match training data
    max_length = max(len(seq) for seq in sequences)
    padded_test = test_sequence[0] + [0] * (max_length - len(test_sequence[0]))
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor([padded_test], dtype=torch.long).to(device)
        prediction = model(test_tensor)
        
    print(f"Model prediction: {prediction.item():.4f} (symbolic content probability)")
    
    return model, preprocessor, security_report


def demonstrate_custom_embedding_layer():
    """Demonstrate custom embedding layer with security tracking"""
    if not PYTORCH_AVAILABLE:
        print("PyTorch not available for custom embedding demo")
        return
    
    print("\n=== Custom Secure Embedding Layer Demo ===")
    
    from symbolic_security_layer.preprocessor import SecureEmbeddingLayer
    
    # Create secure embedding layer
    ssl_engine = SymbolicSecurityEngine()
    secure_embedding = SecureEmbeddingLayer(
        vocab_size=1000,
        embedding_dim=128,
        security_engine=ssl_engine
    )
    
    # Create PyTorch layer
    pt_embedding = secure_embedding.create_pytorch_layer(padding_idx=0)
    
    print(f"Created secure embedding layer:")
    print(f"- Vocabulary size: {secure_embedding.vocab_size}")
    print(f"- Embedding dimension: {secure_embedding.embedding_dim}")
    print(f"- Security engine: {type(secure_embedding.security_engine).__name__}")
    
    # Test with sample input
    sample_input = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]], dtype=torch.long)
    output = pt_embedding(sample_input)
    
    print(f"Sample input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    
    return pt_embedding


def demonstrate_security_metrics_tracking():
    """Demonstrate security metrics tracking during training"""
    if not PYTORCH_AVAILABLE:
        print("PyTorch not available for metrics tracking demo")
        return
    
    print("\n=== Security Metrics Tracking Demo ===")
    
    # Get preprocessed data
    preprocessor, sequences, labels, security_report = demonstrate_secure_preprocessing()
    
    # Track security metrics over training
    security_metrics = {
        'epoch': [],
        'avg_security_level': [],
        'compliance_rate': [],
        'symbol_coverage': []
    }
    
    # Simulate training epochs with different data batches
    for epoch in range(5):
        # Simulate processing new data each epoch
        batch_texts = [
            f"Epoch {epoch} symbolic content with 🜄 and ☥",
            f"Training iteration {epoch} includes ∞ symbols",
            f"Batch {epoch} contains (cid:0) artifacts"
        ]
        
        batch_data, batch_report = preprocessor.process_dataset(batch_texts)
        
        # Extract metrics
        security_dist = batch_report.get('security_distribution', {})
        total_items = sum(security_dist.values()) if security_dist else 1
        
        # Calculate average security level (weighted)
        weights = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        weighted_sum = sum(weights.get(level, 0) * count for level, count in security_dist.items())
        avg_security = weighted_sum / total_items if total_items > 0 else 0
        
        # Calculate compliance rate
        compliance_dist = batch_report.get('compliance_distribution', {})
        compliant_items = compliance_dist.get('CIP-1', 0)
        compliance_rate = compliant_items / total_items if total_items > 0 else 0
        
        # Get symbol coverage
        coverage_str = batch_report.get('processing_summary', {}).get('average_coverage', '0%')
        coverage = float(coverage_str.rstrip('%'))
        
        # Store metrics
        security_metrics['epoch'].append(epoch + 1)
        security_metrics['avg_security_level'].append(avg_security)
        security_metrics['compliance_rate'].append(compliance_rate)
        security_metrics['symbol_coverage'].append(coverage)
        
        print(f"Epoch {epoch + 1}: Security={avg_security:.2f}, "
              f"Compliance={compliance_rate:.2f}, Coverage={coverage:.1f}%")
    
    return security_metrics


def generate_pytorch_report():
    """Generate PyTorch-specific security report"""
    print("\n=== PyTorch Security Report ===")
    
    # Get security data from preprocessing
    _, _, _, security_report = demonstrate_secure_preprocessing()
    
    # Add PyTorch-specific metrics
    pytorch_metrics = {
        'framework': 'PyTorch',
        'tensor_backend': 'CPU' if not torch.cuda.is_available() else 'CUDA',
        'torch_version': torch.__version__ if PYTORCH_AVAILABLE else 'Not Available'
    }
    
    security_report['pytorch_integration'] = pytorch_metrics
    
    # Generate detailed diagnostics
    diagnostics = SecurityDiagnosticsReport()
    report_content = diagnostics.generate_report(security_report)
    
    # Save report
    report_path = "ssl_pytorch_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        f.write("\n\n## PyTorch Integration Details\n")
        f.write(f"- Framework: {pytorch_metrics['framework']}\n")
        f.write(f"- Backend: {pytorch_metrics['tensor_backend']}\n")
        f.write(f"- Version: {pytorch_metrics['torch_version']}\n")
    
    print(f"PyTorch-specific report saved to: {report_path}")
    
    return report_content


def main():
    """Main demonstration function"""
    print("🛡️ Symbolic Security Layer - PyTorch Integration Demo")
    print("=" * 60)
    
    try:
        # Run demonstrations
        demonstrate_secure_preprocessing()
        
        if PYTORCH_AVAILABLE:
            demonstrate_pytorch_training()
            demonstrate_custom_embedding_layer()
            demonstrate_security_metrics_tracking()
        else:
            print("\nSkipping PyTorch-specific demos (PyTorch not installed)")
        
        generate_pytorch_report()
        
        print("\n✅ Demo completed successfully!")
        print("\nNext steps:")
        print("1. Install PyTorch: pip install torch")
        print("2. Run this script to see full PyTorch integration")
        print("3. Explore the generated security report")
        print("4. Integrate SSL into your own PyTorch workflows")
        print("5. Compare with TensorFlow integration results")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

