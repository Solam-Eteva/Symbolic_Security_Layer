# Core API Reference

The `symbolic_security_layer.core` module provides the fundamental SSL functionality through the `SymbolicSecurityEngine` class.

## SymbolicSecurityEngine

The main class for symbolic security operations.

### Constructor

```python
SymbolicSecurityEngine(config_path: Optional[str] = None)
```

**Parameters:**
- `config_path` (str, optional): Path to JSON configuration file

**Example:**
```python
# Default configuration
ssl_engine = SymbolicSecurityEngine()

# Custom configuration file
ssl_engine = SymbolicSecurityEngine("config.json")
```

### Configuration

The engine accepts the following configuration options:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `unicode_threshold` | int | 4096 | Minimum Unicode codepoint for symbol detection |
| `semantic_anchoring` | bool | True | Enable semantic anchoring |
| `strict_validation` | bool | False | Enable strict validation mode |
| `auto_anchor` | bool | True | Automatically anchor detected symbols |
| `synthetic_detection` | bool | True | Enable synthetic content detection |
| `log_reconstructions` | bool | True | Log reconstruction operations |

**Example configuration file:**
```json
{
    "unicode_threshold": 2048,
    "semantic_anchoring": true,
    "strict_validation": false,
    "auto_anchor": true,
    "synthetic_detection": true,
    "log_reconstructions": true
}
```

### Methods

#### secure_content()

Secures symbolic content with semantic anchoring.

```python
secure_content(content: Union[str, Dict, List]) -> Tuple[Union[str, Dict, List], Dict]
```

**Parameters:**
- `content`: Content to secure (string, dictionary, or list)

**Returns:**
- Tuple of (secured_content, validation_report)

**Validation Report Structure:**
```python
{
    "SEM_count": int,           # Semantically anchored symbols
    "EXU_count": int,           # Existentially unverified symbols  
    "PROV_count": int,          # Procedurally valid symbols
    "symbol_count": int,        # Total symbols found
    "symbol_coverage": str,     # Coverage percentage (e.g., "95.2%")
    "security_level": str,      # "HIGH", "MEDIUM", or "LOW"
    "compliance": str,          # "CIP-1", "PARTIAL", or "NON-COMPLIANT"
    "processing_time": float,   # Processing time in seconds
    "session_id": str          # Unique session identifier
}
```

**Example:**
```python
text = "The symbol 🜄 represents water"
secured_text, report = ssl_engine.secure_content(text)

print(secured_text)  # "The symbol 🜄(Water_Alchemical) represents water"
print(report["security_level"])  # "HIGH"
print(report["symbol_coverage"])  # "100.0%"
```

#### add_custom_symbol()

Adds a custom symbol to the database.

```python
add_custom_symbol(symbol: str, anchor: str, description: str) -> None
```

**Parameters:**
- `symbol`: The Unicode symbol to add
- `anchor`: Semantic anchor text (alphanumeric and underscores only)
- `description`: Human-readable description

**Example:**
```python
ssl_engine.add_custom_symbol(
    symbol="🔮",
    anchor="Crystal_Ball", 
    description="Divination: reveal hidden knowledge"
)
```

#### get_symbol_info()

Retrieves information about a symbol.

```python
get_symbol_info(symbol: str) -> Tuple[str, str]
```

**Parameters:**
- `symbol`: The Unicode symbol to look up

**Returns:**
- Tuple of (anchor, description)

**Example:**
```python
anchor, description = ssl_engine.get_symbol_info("🜄")
print(anchor)      # "Water_Alchemical"
print(description) # "Emotion: receptivity, intuition, subconscious..."
```

#### export_symbol_database()

Exports the symbol database to a JSON file.

```python
export_symbol_database(file_path: str) -> None
```

**Parameters:**
- `file_path`: Path where to save the database

**Example:**
```python
ssl_engine.export_symbol_database("my_symbols.json")
```

#### import_symbol_database()

Imports symbols from a JSON file.

```python
import_symbol_database(file_path: str) -> None
```

**Parameters:**
- `file_path`: Path to the JSON file to import

**Example:**
```python
ssl_engine.import_symbol_database("my_symbols.json")
```

#### generate_reconstruction_log()

Generates a JSON log of all reconstruction operations.

```python
generate_reconstruction_log() -> str
```

**Returns:**
- JSON string containing reconstruction log

**Example:**
```python
log_json = ssl_engine.generate_reconstruction_log()
log_data = json.loads(log_json)
print(f"Total operations: {len(log_data)}")
```

### Properties

#### symbol_db

Dictionary containing the current symbol database.

```python
ssl_engine.symbol_db: Dict[str, Tuple[str, str]]
```

**Example:**
```python
print(f"Database size: {len(ssl_engine.symbol_db)}")
for symbol, (anchor, desc) in ssl_engine.symbol_db.items():
    print(f"{symbol} -> {anchor}")
```

#### config

Dictionary containing current configuration.

```python
ssl_engine.config: Dict[str, Any]
```

**Example:**
```python
print(f"Unicode threshold: {ssl_engine.config['unicode_threshold']}")
ssl_engine.config['strict_validation'] = True  # Modify configuration
```

#### reconstruction_log

List of reconstruction operations performed.

```python
ssl_engine.reconstruction_log: List[Dict]
```

**Example:**
```python
print(f"Operations logged: {len(ssl_engine.reconstruction_log)}")
if ssl_engine.reconstruction_log:
    latest = ssl_engine.reconstruction_log[-1]
    print(f"Latest operation: {latest['timestamp']}")
```

#### session_id

Unique identifier for the current session.

```python
ssl_engine.session_id: str
```

**Example:**
```python
print(f"Session ID: {ssl_engine.session_id}")
```

## Security Levels

SSL assigns security levels based on symbol coverage and anchoring success:

### HIGH Security
- Symbol coverage ≥ 90%
- All or most symbols successfully anchored
- CIP-1 compliant

### MEDIUM Security  
- Symbol coverage 70-89%
- Majority of symbols anchored
- Partial compliance

### LOW Security
- Symbol coverage < 70%
- Many unanchored symbols
- Non-compliant

## Compliance Standards

### CIP-1 (Collaborative Intelligence Protocol v1)
- Requires ≥ 95% symbol coverage
- All symbols must have semantic anchors
- Procedural transparency required

### PARTIAL
- 80-94% symbol coverage
- Most symbols anchored
- Some procedural gaps

### NON-COMPLIANT
- < 80% symbol coverage
- Significant anchoring failures
- Major procedural issues

## Error Handling

The SSL engine raises specific exceptions for different error conditions:

### ValueError
Raised for invalid input types or parameters:

```python
try:
    ssl_engine.secure_content(None)
except ValueError as e:
    print(f"Invalid input: {e}")

try:
    ssl_engine.secure_content(123)
except ValueError as e:
    print(f"Unsupported type: {e}")
```

### FileNotFoundError
Raised when configuration or database files are not found:

```python
try:
    ssl_engine = SymbolicSecurityEngine("nonexistent.json")
except FileNotFoundError as e:
    print(f"Config file not found: {e}")
```

### JSONDecodeError
Raised when configuration files contain invalid JSON:

```python
try:
    ssl_engine.import_symbol_database("invalid.json")
except json.JSONDecodeError as e:
    print(f"Invalid JSON: {e}")
```

## Performance Considerations

### Memory Usage
- Symbol database: ~50KB for default symbols
- Reconstruction log: ~1KB per operation
- Configuration: Negligible

### Processing Speed
- Text processing: ~1000 texts/second
- Symbol detection: ~10,000 symbols/second
- Database operations: ~100,000 lookups/second

### Optimization Tips

1. **Disable logging for production:**
```python
ssl_engine.config['log_reconstructions'] = False
```

2. **Adjust Unicode threshold for performance:**
```python
ssl_engine.config['unicode_threshold'] = 8192  # Higher = fewer symbols detected
```

3. **Use batch processing for large datasets:**
```python
from symbolic_security_layer import SecureDataPreprocessor
preprocessor = SecureDataPreprocessor(ssl_engine)
```

## Thread Safety

The `SymbolicSecurityEngine` is **not thread-safe**. For concurrent usage:

1. **Create separate instances per thread:**
```python
import threading

def worker():
    ssl_engine = SymbolicSecurityEngine()  # Thread-local instance
    # Use ssl_engine in this thread
```

2. **Use locks for shared instances:**
```python
import threading

ssl_engine = SymbolicSecurityEngine()
lock = threading.Lock()

def secure_with_lock(content):
    with lock:
        return ssl_engine.secure_content(content)
```

## Examples

### Basic Text Processing
```python
ssl_engine = SymbolicSecurityEngine()

texts = [
    "Mathematical infinity ∞ concept",
    "Alchemical water 🜄 symbol", 
    "Ancient ankh ☥ meaning"
]

for text in texts:
    secured, report = ssl_engine.secure_content(text)
    print(f"Original: {text}")
    print(f"Secured: {secured}")
    print(f"Security: {report['security_level']}")
    print()
```

### Custom Symbol Management
```python
ssl_engine = SymbolicSecurityEngine()

# Add domain-specific symbols
symbols_to_add = [
    ("🔮", "Crystal_Ball", "Divination and foresight"),
    ("🌟", "Star", "Guidance and aspiration"),
    ("🎭", "Theater_Mask", "Performance and identity")
]

for symbol, anchor, desc in symbols_to_add:
    ssl_engine.add_custom_symbol(symbol, anchor, desc)

# Test with custom symbols
text = "The crystal ball 🔮 shows a bright star 🌟"
secured, report = ssl_engine.secure_content(text)
print(secured)
# Output: The crystal ball 🔮(Crystal_Ball) shows a bright star 🌟(Star)
```

### Configuration Management
```python
# Save current configuration
config_data = ssl_engine.config.copy()
with open("ssl_config.json", "w") as f:
    json.dump(config_data, f, indent=2)

# Load configuration in new instance
new_engine = SymbolicSecurityEngine("ssl_config.json")
assert new_engine.config == config_data
```

## See Also

- [AI Integration API](adapters.md) - AI platform integration
- [Preprocessor API](preprocessor.md) - ML framework integration  
- [Diagnostics API](diagnostics.md) - Security reporting
- [Getting Started Guide](../guides/getting-started.md) - Basic usage tutorial

