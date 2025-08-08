#!/usr/bin/env python3
"""
VSCode Backend for Symbolic Security Layer
Handles requests from the VSCode extension
"""

import sys
import json
import traceback
from typing import Dict, Any

from .core import SymbolicSecurityEngine
from .adapters import AIIntegrationAdapter


class VSCodeBackend:
    """Backend service for VSCode extension integration"""
    
    def __init__(self):
        self.ssl_engine = SymbolicSecurityEngine()
        self.ai_adapter = AIIntegrationAdapter(self.ssl_engine)
        
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process request from VSCode extension"""
        try:
            action = request.get('action', 'secure')
            text = request.get('text', '')
            validate_only = request.get('validate_only', False)
            config = request.get('config', {})
            params = request.get('params', {})
            
            # Update SSL engine configuration
            self._update_config(config)
            
            if action == 'secure':
                return self._secure_text(text, validate_only)
            elif action == 'add_symbol':
                return self._add_custom_symbol(params)
            elif action == 'validate':
                return self._validate_text(text)
            elif action == 'get_report':
                return self._get_security_report()
            else:
                return {'error': f'Unknown action: {action}'}
                
        except Exception as e:
            return {
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _update_config(self, config: Dict[str, Any]) -> None:
        """Update SSL engine configuration from VSCode settings"""
        if 'unicodeThreshold' in config:
            self.ssl_engine.config['unicode_threshold'] = config['unicodeThreshold']
        if 'semanticAnchoring' in config:
            self.ssl_engine.config['semantic_anchoring'] = config['semanticAnchoring']
        if 'strictValidation' in config:
            self.ssl_engine.config['strict_validation'] = config['strictValidation']
    
    def _secure_text(self, text: str, validate_only: bool = False) -> Dict[str, Any]:
        """Secure text content with SSL processing"""
        secured_content, validation_report = self.ssl_engine.secure_content(text)
        
        result = {
            'secured_content': text if validate_only else secured_content,
            'validation_report': validation_report
        }
        
        return result
    
    def _validate_text(self, text: str) -> Dict[str, Any]:
        """Validate text without modification"""
        return self._secure_text(text, validate_only=True)
    
    def _add_custom_symbol(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add custom symbol to the database"""
        symbol = params.get('symbol')
        anchor = params.get('anchor')
        description = params.get('description')
        
        if not all([symbol, anchor, description]):
            return {'error': 'Missing required parameters: symbol, anchor, description'}
        
        self.ssl_engine.add_custom_symbol(symbol, anchor, description)
        
        return {
            'success': True,
            'message': f'Symbol {symbol} added successfully'
        }
    
    def _get_security_report(self) -> Dict[str, Any]:
        """Get current security report"""
        return {
            'reconstruction_log': self.ssl_engine.generate_reconstruction_log(),
            'symbol_database_size': len(self.ssl_engine.symbol_db),
            'session_id': self.ssl_engine.session_id
        }


def main():
    """Main entry point for VSCode backend"""
    try:
        # Read request from stdin
        request_data = sys.stdin.read()
        request = json.loads(request_data)
        
        # Process request
        backend = VSCodeBackend()
        response = backend.process_request(request)
        
        # Send response to stdout
        print(json.dumps(response, ensure_ascii=False))
        
    except Exception as e:
        error_response = {
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        print(json.dumps(error_response, ensure_ascii=False))
        sys.exit(1)


if __name__ == '__main__':
    main()

