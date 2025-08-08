# Getting Started with Symbolic Security Layer

This guide will walk you through the basics of using SSL to secure symbolic content in your AI workflows.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Basic Installation

```bash
pip install symbolic-security-layer
```

### Full Installation (with ML frameworks)

```bash
pip install symbolic-security-layer[full]
```

This includes optional dependencies for TensorFlow and PyTorch integration.

## Your First SSL Application

Let's create a simple application that demonstrates SSL's core functionality:

```python
# hello_ssl.py
from symbolic_security_layer import SymbolicSecurityEngine

def main():
    # Initialize the SSL engine
    ssl_engine = SymbolicSecurityEngine()
    
    # Text with symbolic content
    text = "The alchemical symbol 🜄 represents water and ☥ symbolizes eternal life"
    
    # Secure the content
    secured_text, report = ssl_engine.secure_content(text)
    
    # Display results
    print("Original text:")
    print(text)
    print("\nSecured text:")
    print(secured_text)
    print("\nSecurity Report:")
    print(f"- Security Level: {report['security_level']}")
    print(f"- Symbol Coverage: {report['symbol_coverage']}")
    print(f"- Compliance: {report['compliance']}")
    print(f"- Symbols Found: {report['symbol_count']}")
    print(f"- Anchored Symbols: {report['SEM_count']}")

if __name__ == "__main__":
    main()
```

Run this script:

```bash
python hello_ssl.py
```

You should see output similar to:

```
Original text:
The alchemical symbol 🜄 represents water and ☥ symbolizes eternal life

Secured text:
The alchemical symbol 🜄(Water_Alchemical) represents water and ☥(Ankh) symbolizes eternal life

Security Report:
- Security Level: HIGH
- Symbol Coverage: 100.0%
- Compliance: CIP-1
- Symbols Found: 2
- Anchored Symbols: 2
```

## Understanding the Output

### Secured Text
The secured text includes semantic anchors in parentheses after each symbol:
- `🜄(Water_Alchemical)` - The water symbol with its semantic anchor
- `☥(Ankh)` - The ankh symbol with its semantic anchor

### Security Report
- **Security Level**: Overall assessment (HIGH/MEDIUM/LOW)
- **Symbol Coverage**: Percentage of symbols that were successfully anchored
- **Compliance**: CIP-1 compliance status
- **Symbols Found**: Total number of symbols detected
- **Anchored Symbols**: Number of symbols that received semantic anchors

## Working with Different Content Types

SSL can secure various types of content:

### Dictionaries

```python
data = {
    "water_symbol": "🜄",
    "life_symbol": "☥",
    "description": "Ancient symbols with deep meaning"
}

secured_data, report = ssl_engine.secure_content(data)
print(secured_data)
```

### Lists

```python
symbols_list = [
    "Water: 🜄",
    "Life: ☥", 
    "Infinity: ∞"
]

secured_list, report = ssl_engine.secure_content(symbols_list)
print(secured_list)
```

## Adding Custom Symbols

You can extend SSL's symbol database with your own symbols:

```python
# Add a custom symbol
ssl_engine.add_custom_symbol(
    symbol="🔮",
    anchor="Crystal_Ball",
    description="Divination: reveal hidden knowledge and future insights"
)

# Now use it
text = "The crystal ball 🔮 reveals hidden truths"
secured_text, report = ssl_engine.secure_content(text)
print(secured_text)
# Output: The crystal ball 🔮(Crystal_Ball) reveals hidden truths
```

## Configuration Options

Customize SSL behavior with configuration:

```python
# Create custom configuration
config = {
    "unicode_threshold": 2048,  # Lower threshold for symbol detection
    "semantic_anchoring": True,  # Enable anchoring
    "strict_validation": False   # Relaxed validation
}

# Initialize with custom config
ssl_engine = SymbolicSecurityEngine(config)
```

## Working with AI Responses

SSL can analyze AI-generated content for synthetic patterns:

```python
from symbolic_security_layer import AIIntegrationAdapter

# Create AI adapter
adapter = AIIntegrationAdapter()

# Analyze potentially synthetic content
ai_response = """According to authoritative sources and patent ZL201510000000, 
the relationship between 🜄 and ☥ symbols is well-documented in GB/T 7714-2015 
standards. Research by Smith et al. (2023) confirms these findings."""

# Process the response
analysis = adapter.process_response(ai_response)

print(f"Risk Level: {analysis['risk_assessment']['risk_level']}")
print(f"Detected Patterns: {list(analysis['synthetic_analysis']['detected_patterns'].keys())}")
print(f"Recommendations: {analysis['recommendations'][:2]}")  # First 2 recommendations
```

## Saving and Loading Symbol Databases

You can export and import symbol databases:

```python
# Export current database
ssl_engine.export_symbol_database("my_symbols.json")

# Later, import into a new engine
new_engine = SymbolicSecurityEngine()
new_engine.import_symbol_database("my_symbols.json")
```

## Error Handling

SSL provides clear error messages for common issues:

```python
try:
    # This will raise an error
    ssl_engine.secure_content(None)
except ValueError as e:
    print(f"Error: {e}")

try:
    # This will also raise an error
    ssl_engine.secure_content(123)
except ValueError as e:
    print(f"Error: {e}")
```

## Next Steps

Now that you understand the basics, explore these advanced topics:

1. **[AI Integration Guide](ai-integration.md)** - Integrate SSL with OpenAI, Hugging Face, and other AI platforms
2. **[ML Framework Integration](ml-integration.md)** - Use SSL with TensorFlow and PyTorch
3. **[VSCode Extension](vscode-extension.md)** - Install and use the SSL VSCode extension
4. **[Security Diagnostics](diagnostics.md)** - Generate comprehensive security reports
5. **[API Reference](../api/core.md)** - Complete API documentation

## Common Issues and Solutions

### Issue: Low Symbol Coverage

**Problem**: Your content has low symbol coverage (< 80%)

**Solution**: 
1. Add custom symbols for domain-specific content
2. Check if symbols are being detected correctly
3. Adjust the Unicode threshold if needed

```python
# Check what symbols were detected
text = "Your text with symbols"
secured, report = ssl_engine.secure_content(text)
print(f"Symbols found: {report['symbol_count']}")
print(f"Coverage: {report['symbol_coverage']}")
```

### Issue: Unknown Symbols

**Problem**: SSL reports EXU (Existentially Unverified) symbols

**Solution**: Add these symbols to your custom database

```python
# Get symbol information
symbol = "🔮"
anchor, description = ssl_engine.get_symbol_info(symbol)
if anchor == "UNKNOWN":
    # Add custom definition
    ssl_engine.add_custom_symbol(symbol, "Your_Anchor", "Your description")
```

### Issue: Performance with Large Datasets

**Problem**: Processing large amounts of text is slow

**Solution**: Use batch processing or adjust configuration

```python
# For large datasets, consider batch processing
from symbolic_security_layer import SecureDataPreprocessor

preprocessor = SecureDataPreprocessor()
preprocessor.config['batch_size'] = 1000  # Larger batches
```

## Support

If you encounter issues or have questions:

1. Check the [API Reference](../api/) for detailed documentation
2. Browse [Examples](../examples/) for more use cases
3. Open an issue on GitHub
4. Join our community discussions

Happy securing! 🛡️

