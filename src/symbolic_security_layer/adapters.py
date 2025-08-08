"""
AI Integration Adapters
Provides interfaces for common AI platforms and services
"""

import re
import json
import time
import requests
from typing import Dict, List, Any, Optional, Union
from .core import SymbolicSecurityEngine


class AIIntegrationAdapter:
    """Adapter for common AI platforms"""
    
    def __init__(self, security_engine: SymbolicSecurityEngine = None):
        self.security_engine = security_engine or SymbolicSecurityEngine()
        self.synthetic_patterns = self._load_synthetic_patterns()
        
    def _load_synthetic_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for detecting synthetic/generated content"""
        return {
            'authoritative_phrases': [
                r'\b(?:according to|as per|based on|refer to|see also)\s+(?:sources?|studies?|data|research)\b',
                r'\b(?:authoritative|official|verified|confirmed)\s+(?:sources?|data|documentation)\b',
                r'\b(?:established|recognized|accepted)\s+(?:standards?|protocols?|procedures?)\b',
                r'\b(?:peer-reviewed|published|documented)\s+(?:research|studies?|findings)\b'
            ],
            'patent_numbers': [
                r'\b[A-Z]{2}\d{6,}\b',  # Pattern like ZL201510000000
                r'\b[A-Z]{2}\d{4}\d{6,}\b',  # Alternative patent formats
                r'\bUS\d{7,}\b',  # US patents
                r'\bEP\d{7,}\b'   # European patents
            ],
            'standards_references': [
                r'\b[A-Z]{2,}\/[A-Z]\s+\d{4}-\d+\b',  # Standards like GB/T 7714-2015
                r'\b[A-Z]{3,}\s+\d{3,}\b',  # ISO standards
                r'\bIEEE\s+\d{3,}\b',  # IEEE standards
                r'\bANSI\s+[A-Z]\d+\.\d+\b'  # ANSI standards
            ],
            'procedural_indicators': [
                r'\b(?:procedure|protocol|methodology|framework)\s+\d+\.\d+\b',
                r'\b(?:section|clause|article)\s+\d+\.\d+\b',
                r'\b(?:step|phase|stage)\s+\d+\b',
                r'\b(?:algorithm|method)\s+\d+\b'
            ],
            'citation_formats': [
                r'\([A-Za-z]+\s+et\s+al\.,?\s+\d{4}\)',  # Academic citations
                r'\[[A-Za-z]+\s+\d{4}\]',  # Bracket citations
                r'\b[A-Z][a-z]+,\s+[A-Z]\.\s+\(\d{4}\)',  # Author citations
                r'\bdoi:\s*10\.\d+\/[^\s]+\b'  # DOI references
            ],
            'technical_jargon': [
                r'\b(?:implementation|optimization|enhancement|methodology)\b',
                r'\b(?:framework|architecture|infrastructure|paradigm)\b',
                r'\b(?:scalability|reliability|efficiency|performance)\b',
                r'\b(?:integration|deployment|configuration|validation)\b'
            ]
        }
    
    def wrap_prompt(self, prompt: str, metadata: Dict = None) -> Dict[str, Any]:
        """Secure prompt before sending to AI"""
        secured_prompt, report = self.security_engine.secure_content(prompt)
        
        wrapper = {
            "original": prompt,
            "secured": secured_prompt,
            "validation": report,
            "compliance": report.get("compliance", "UNKNOWN"),
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # Add security warnings if needed
        if report.get("security_level") == "LOW":
            wrapper["warnings"] = ["Low security level detected - consider adding more semantic anchors"]
        if report.get("EXU_count", 0) > 0:
            wrapper["warnings"] = wrapper.get("warnings", []) + [f"{report['EXU_count']} unknown symbols detected"]
            
        return wrapper
    
    def process_response(self, response: str, context: Dict = None) -> Dict[str, Any]:
        """Process AI response through security layer"""
        # Detect synthetic patterns
        synthetic_analysis = self._analyze_synthetic_content(response)
        
        # Apply security processing
        secured_response, security_report = self.security_engine.secure_content(response)
        
        # Calculate risk assessment
        risk_assessment = self._calculate_risk_assessment(synthetic_analysis, security_report)
        
        result = {
            "original_response": response,
            "secured_response": secured_response,
            "security_report": security_report,
            "synthetic_analysis": synthetic_analysis,
            "risk_assessment": risk_assessment,
            "timestamp": time.time(),
            "context": context or {}
        }
        
        # Add recommendations
        result["recommendations"] = self._generate_recommendations(synthetic_analysis, security_report)
        
        return result
    
    def _analyze_synthetic_content(self, text: str) -> Dict[str, Any]:
        """Comprehensive analysis of potentially synthetic content"""
        analysis = {
            "detected_patterns": {},
            "confidence_scores": {},
            "total_matches": 0,
            "risk_indicators": []
        }
        
        total_matches = 0
        
        for category, pattern_list in self.synthetic_patterns.items():
            matches = []
            for pattern in pattern_list:
                found = re.findall(pattern, text, re.IGNORECASE)
                matches.extend(found)
                
            if matches:
                analysis["detected_patterns"][category] = matches
                total_matches += len(matches)
                
                # Calculate confidence score for this category
                confidence = min(len(matches) * 0.2, 1.0)  # Max 1.0
                analysis["confidence_scores"][category] = confidence
        
        analysis["total_matches"] = total_matches
        
        # Generate risk indicators
        if total_matches > 5:
            analysis["risk_indicators"].append("HIGH_PATTERN_DENSITY")
        if "patent_numbers" in analysis["detected_patterns"]:
            analysis["risk_indicators"].append("UNVERIFIED_PATENTS")
        if "standards_references" in analysis["detected_patterns"]:
            analysis["risk_indicators"].append("UNVERIFIED_STANDARDS")
        if len(analysis["detected_patterns"]) >= 3:
            analysis["risk_indicators"].append("MULTIPLE_SYNTHETIC_CATEGORIES")
            
        return analysis
    
    def _calculate_risk_assessment(self, synthetic_analysis: Dict, security_report: Dict) -> Dict[str, Any]:
        """Calculate comprehensive risk assessment"""
        risk_factors = []
        risk_score = 0.0
        
        # Synthetic content risk
        synthetic_risk = len(synthetic_analysis.get("risk_indicators", []))
        if synthetic_risk > 0:
            risk_factors.append(f"Synthetic content indicators: {synthetic_risk}")
            risk_score += synthetic_risk * 0.2
            
        # Security level risk
        security_level = security_report.get("security_level", "LOW")
        if security_level == "LOW":
            risk_factors.append("Low symbolic security level")
            risk_score += 0.3
        elif security_level == "MEDIUM":
            risk_score += 0.1
            
        # Unknown symbols risk
        exu_count = security_report.get("EXU_count", 0)
        if exu_count > 0:
            risk_factors.append(f"Unknown symbols: {exu_count}")
            risk_score += exu_count * 0.1
            
        # Determine overall risk level
        if risk_score >= 0.7:
            risk_level = "HIGH"
        elif risk_score >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
            
        return {
            "risk_level": risk_level,
            "risk_score": min(risk_score, 1.0),
            "risk_factors": risk_factors,
            "requires_verification": risk_level in ["HIGH", "MEDIUM"]
        }
    
    def _generate_recommendations(self, synthetic_analysis: Dict, security_report: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Synthetic content recommendations
        if "patent_numbers" in synthetic_analysis.get("detected_patterns", {}):
            recommendations.append("Verify patent numbers against official databases (USPTO, CNIPA, EPO)")
            
        if "standards_references" in synthetic_analysis.get("detected_patterns", {}):
            recommendations.append("Cross-check standards references with official standards bodies")
            
        if "citation_formats" in synthetic_analysis.get("detected_patterns", {}):
            recommendations.append("Validate academic citations against scholarly databases")
            
        # Security recommendations
        if security_report.get("EXU_count", 0) > 0:
            recommendations.append("Add semantic anchors for unknown symbols")
            
        if security_report.get("security_level") == "LOW":
            recommendations.append("Improve symbolic security by adding more anchored symbols")
            
        if security_report.get("compliance") != "CIP-1":
            recommendations.append("Achieve CIP-1 compliance by reaching 95%+ symbol coverage")
            
        return recommendations
    
    def batch_process(self, items: List[Union[str, Dict]], batch_size: int = 10) -> List[Dict]:
        """Process multiple items in batches"""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = []
            
            for item in batch:
                if isinstance(item, str):
                    result = self.process_response(item)
                    result["type"] = "response"
                elif isinstance(item, dict) and "prompt" in item:
                    wrapped = self.wrap_prompt(item["prompt"], item.get("metadata"))
                    result = {"type": "prompt", "data": wrapped}
                else:
                    result = {"type": "unknown", "data": item, "error": "Unsupported item type"}
                    
                batch_results.append(result)
                
            results.extend(batch_results)
            
        return results
    
    def verify_external_reference(self, reference: str, reference_type: str = "auto") -> Dict[str, Any]:
        """Attempt to verify external references (patents, standards, etc.)"""
        verification_result = {
            "reference": reference,
            "type": reference_type,
            "verified": False,
            "verification_method": "pattern_analysis",
            "confidence": 0.0,
            "notes": []
        }
        
        if reference_type == "auto":
            reference_type = self._detect_reference_type(reference)
            verification_result["type"] = reference_type
            
        # Pattern-based verification (placeholder for actual API calls)
        if reference_type == "patent":
            verification_result.update(self._verify_patent_pattern(reference))
        elif reference_type == "standard":
            verification_result.update(self._verify_standard_pattern(reference))
        elif reference_type == "doi":
            verification_result.update(self._verify_doi_pattern(reference))
            
        return verification_result
    
    def _detect_reference_type(self, reference: str) -> str:
        """Detect the type of reference"""
        if re.match(r'\b[A-Z]{2}\d{6,}\b', reference):
            return "patent"
        elif re.match(r'\b[A-Z]{2,}\/[A-Z]\s+\d{4}-\d+\b', reference):
            return "standard"
        elif re.match(r'\bdoi:\s*10\.\d+\/[^\s]+\b', reference):
            return "doi"
        elif re.match(r'\bIEEE\s+\d{3,}\b', reference):
            return "ieee_standard"
        else:
            return "unknown"
    
    def _verify_patent_pattern(self, patent: str) -> Dict[str, Any]:
        """Verify patent number pattern"""
        result = {"notes": []}
        
        # Check format validity
        if re.match(r'^[A-Z]{2}\d{4}\d{6}$', patent.replace(' ', '')):
            result["confidence"] = 0.8
            result["notes"].append("Valid patent number format")
            
            # Extract year and sequence
            year_part = patent[2:6]
            sequence = patent[6:]
            
            if year_part.isdigit():
                year = int(year_part)
                if 1990 <= year <= 2025:
                    result["confidence"] += 0.1
                    result["notes"].append(f"Plausible filing year: {year}")
                else:
                    result["notes"].append(f"Unusual filing year: {year}")
                    
            if sequence == "000000":
                result["confidence"] -= 0.3
                result["notes"].append("Suspicious sequence: 000000 (likely synthetic)")
                
        else:
            result["confidence"] = 0.2
            result["notes"].append("Invalid patent number format")
            
        return result
    
    def _verify_standard_pattern(self, standard: str) -> Dict[str, Any]:
        """Verify standard reference pattern"""
        result = {"notes": []}
        
        # Check common standard formats
        if re.match(r'^[A-Z]{2,}\/[A-Z]\s+\d{4}-\d+$', standard):
            result["confidence"] = 0.7
            result["notes"].append("Valid standard format")
        elif re.match(r'^IEEE\s+\d{3,}$', standard):
            result["confidence"] = 0.8
            result["notes"].append("Valid IEEE standard format")
        else:
            result["confidence"] = 0.3
            result["notes"].append("Non-standard format")
            
        return result
    
    def _verify_doi_pattern(self, doi: str) -> Dict[str, Any]:
        """Verify DOI pattern"""
        result = {"notes": []}
        
        if re.match(r'^doi:\s*10\.\d+\/[^\s]+$', doi):
            result["confidence"] = 0.9
            result["notes"].append("Valid DOI format")
        else:
            result["confidence"] = 0.2
            result["notes"].append("Invalid DOI format")
            
        return result


class OpenAIAdapter(AIIntegrationAdapter):
    """Specialized adapter for OpenAI API"""
    
    def __init__(self, api_key: str = None, security_engine: SymbolicSecurityEngine = None):
        super().__init__(security_engine)
        self.api_key = api_key
        
    def secure_chat_completion(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Secure chat completion with SSL processing"""
        # Process messages through security layer
        secured_messages = []
        security_reports = []
        
        for message in messages:
            if "content" in message:
                wrapped = self.wrap_prompt(message["content"])
                secured_message = message.copy()
                secured_message["content"] = wrapped["secured"]
                secured_messages.append(secured_message)
                security_reports.append(wrapped["validation"])
            else:
                secured_messages.append(message)
                
        return {
            "secured_messages": secured_messages,
            "security_reports": security_reports,
            "original_messages": messages,
            "ssl_metadata": {
                "total_symbols": sum(r.get("symbol_count", 0) for r in security_reports),
                "anchored_symbols": sum(r.get("anchored_count", 0) for r in security_reports),
                "compliance_level": "CIP-1" if all(r.get("compliance") == "CIP-1" for r in security_reports) else "PARTIAL"
            }
        }


class HuggingFaceAdapter(AIIntegrationAdapter):
    """Specialized adapter for Hugging Face models"""
    
    def __init__(self, model_name: str = None, security_engine: SymbolicSecurityEngine = None):
        super().__init__(security_engine)
        self.model_name = model_name
        
    def secure_text_generation(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Secure text generation with SSL processing"""
        wrapped_prompt = self.wrap_prompt(prompt)
        
        return {
            "secured_prompt": wrapped_prompt["secured"],
            "security_metadata": wrapped_prompt["validation"],
            "model_name": self.model_name,
            "ssl_compliance": wrapped_prompt["compliance"]
        }

