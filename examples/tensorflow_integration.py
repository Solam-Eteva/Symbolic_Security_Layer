#!/usr/bin/env python3
"""
TensorFlow Integration Example for Symbolic Security Layer
Demonstrates how to use SSL with TensorFlow models
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from symbolic_security_layer import SymbolicSecurityEngine, SecureDataPreprocessor, SecurityDiagnosticsReport

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Install with: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False


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
        "Chemical formulas H₂O contain subscript symbols"
    ]
    
    # Create labels (0 = low symbolic content, 1 = high symbolic content)
    labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]
    
    return symbolic_texts, labels


def demonstrate_secure_preprocessing():
    """Demonstrate secure data preprocessing for TensorFlow"""
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


def create_tensorflow_model(vocab_size, embedding_dim=64, max_length=50):
    """Create a simple TensorFlow model for symbolic text classification"""
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available for model creation")
        return None
    
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def demonstrate_tensorflow_training():
    """Demonstrate training a TensorFlow model with SSL-processed data"""
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available for training demo")
        return
    
    print("\n=== TensorFlow Training Demo ===")
    
    # Get preprocessed data
    preprocessor, sequences, labels, security_report = demonstrate_secure_preprocessing()
    
    # Create TensorFlow dataset
    tf_dataset = preprocessor.create_tensorflow_dataset(sequences, labels, batch_size=4, shuffle=True)
    
    # Create model
    vocab_size = len(preprocessor.vocabulary)
    model = create_tensorflow_model(vocab_size)
    
    print(f"\nModel architecture:")
    model.summary()
    
    # Train model
    print("\nTraining model with SSL-secured data...")
    history = model.fit(
        tf_dataset,
        epochs=5,
        verbose=1
    )
    
    # Evaluate model
    print("\n--- Training Results ---")
    final_loss = history.history['loss'][-1]
    final_accuracy = history.history['accuracy'][-1]
    print(f"Final loss: {final_loss:.4f}")
    print(f"Final accuracy: {final_accuracy:.4f}")
    
    # Test with new symbolic text
    test_text = "The sacred symbol ⚛ represents atomic consciousness 🜄"
    secured_test, test_report = preprocessor.security_engine.secure_content(test_text)
    
    print(f"\n--- Test Prediction ---")
    print(f"Original text: {test_text}")
    print(f"Secured text: {secured_test}")
    print(f"Security level: {test_report['security_level']}")
    
    # Convert to sequence and predict
    test_sequence = preprocessor._texts_to_sequences([secured_test])
    padded_test = tf.keras.preprocessing.sequence.pad_sequences(test_sequence, maxlen=50)
    prediction = model.predict(padded_test)
    
    print(f"Model prediction: {prediction[0][0]:.4f} (symbolic content probability)")
    
    return model, preprocessor, security_report


def demonstrate_custom_embedding_layer():
    """Demonstrate custom embedding layer with security tracking"""
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available for custom embedding demo")
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
    
    # Create TensorFlow layer
    tf_embedding = secure_embedding.create_tensorflow_layer(
        mask_zero=True,
        trainable=True
    )
    
    print(f"Created secure embedding layer:")
    print(f"- Vocabulary size: {secure_embedding.vocab_size}")
    print(f"- Embedding dimension: {secure_embedding.embedding_dim}")
    print(f"- Security engine: {type(secure_embedding.security_engine).__name__}")
    
    # Test with sample input
    sample_input = tf.constant([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]])
    output = tf_embedding(sample_input)
    
    print(f"Sample input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    
    return tf_embedding


def generate_comprehensive_report():
    """Generate comprehensive security diagnostics report"""
    print("\n=== Comprehensive Security Report ===")
    
    # Get security data from preprocessing
    _, _, _, security_report = demonstrate_secure_preprocessing()
    
    # Generate detailed diagnostics
    diagnostics = SecurityDiagnosticsReport()
    report_content = diagnostics.generate_report(security_report)
    
    # Save report
    report_path = "ssl_tensorflow_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Comprehensive report saved to: {report_path}")
    
    # Display key insights
    print("\n--- Key Security Insights ---")
    analysis = diagnostics._analyze_security_data(security_report)
    
    summary = analysis['summary']
    print(f"Items processed: {summary['total_items_processed']}")
    print(f"Average security level: {summary['average_security_level']}")
    print(f"Compliance rate: {summary['compliance_rate']:.1%}")
    
    risks = analysis['risk_assessment']
    print(f"Risk level: {risks['risk_level']}")
    print(f"Risk factors: {len(risks['risk_factors'])}")
    
    recommendations = analysis['recommendations']
    if recommendations:
        print("\nRecommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"{i}. {rec}")
    
    return report_content


def main():
    """Main demonstration function"""
    print("🛡️ Symbolic Security Layer - TensorFlow Integration Demo")
    print("=" * 60)
    
    try:
        # Run demonstrations
        demonstrate_secure_preprocessing()
        
        if TENSORFLOW_AVAILABLE:
            demonstrate_tensorflow_training()
            demonstrate_custom_embedding_layer()
        else:
            print("\nSkipping TensorFlow-specific demos (TensorFlow not installed)")
        
        generate_comprehensive_report()
        
        print("\n✅ Demo completed successfully!")
        print("\nNext steps:")
        print("1. Install TensorFlow: pip install tensorflow")
        print("2. Run this script to see full TensorFlow integration")
        print("3. Explore the generated security report")
        print("4. Integrate SSL into your own ML workflows")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

