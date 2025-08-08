#!/usr/bin/env python3
"""
Command-line interface for Symbolic Security Layer
"""

import argparse
import json
import sys
import os
from typing import Optional, List

from .core import SymbolicSecurityEngine
from .adapters import AIIntegrationAdapter
from .diagnostics import SecurityDiagnosticsReport


def validate_command():
    """CLI command to validate symbolic content"""
    parser = argparse.ArgumentParser(
        description='Validate symbolic content with SSL',
        prog='ssl-validate'
    )
    parser.add_argument('input', help='Input text or file path')
    parser.add_argument('-f', '--file', action='store_true', 
                       help='Treat input as file path')
    parser.add_argument('-c', '--config', help='Configuration file path')
    parser.add_argument('-o', '--output', help='Output file for report')
    parser.add_argument('--json', action='store_true', 
                       help='Output report in JSON format')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        # Initialize SSL engine
        ssl_engine = SymbolicSecurityEngine(args.config)
        
        # Get input content
        if args.file:
            if not os.path.exists(args.input):
                print(f"Error: File '{args.input}' not found", file=sys.stderr)
                sys.exit(1)
            with open(args.input, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            content = args.input
        
        # Validate content
        secured_content, report = ssl_engine.secure_content(content)
        
        # Prepare output
        if args.json:
            output_data = {
                'original_content': content,
                'secured_content': secured_content,
                'validation_report': report
            }
            output = json.dumps(output_data, indent=2, ensure_ascii=False)
        else:
            output = f"""SSL Validation Report
{'=' * 50}
Security Level: {report['security_level']}
Symbol Coverage: {report['symbol_coverage']}
Compliance: {report['compliance']}
Symbols Found: {report['symbol_count']}
Anchored Symbols: {report['SEM_count']}
Processing Time: {report['processing_time']:.3f}s

Original Content:
{content}

Secured Content:
{secured_content}
"""
        
        # Output results
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"Report saved to: {args.output}")
        else:
            print(output)
            
        # Exit with appropriate code
        if report['compliance'] == 'CIP-1':
            sys.exit(0)
        elif report['compliance'] == 'PARTIAL':
            sys.exit(1)
        else:
            sys.exit(2)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(3)


def secure_command():
    """CLI command to secure symbolic content"""
    parser = argparse.ArgumentParser(
        description='Secure symbolic content with SSL',
        prog='ssl-secure'
    )
    parser.add_argument('input', help='Input text or file path')
    parser.add_argument('-f', '--file', action='store_true',
                       help='Treat input as file path')
    parser.add_argument('-c', '--config', help='Configuration file path')
    parser.add_argument('-o', '--output', help='Output file for secured content')
    parser.add_argument('--in-place', action='store_true',
                       help='Modify file in place (requires --file)')
    parser.add_argument('--report', help='Save validation report to file')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        # Validate arguments
        if args.in_place and not args.file:
            print("Error: --in-place requires --file", file=sys.stderr)
            sys.exit(1)
        
        # Initialize SSL engine
        ssl_engine = SymbolicSecurityEngine(args.config)
        
        # Get input content
        if args.file:
            if not os.path.exists(args.input):
                print(f"Error: File '{args.input}' not found", file=sys.stderr)
                sys.exit(1)
            with open(args.input, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            content = args.input
        
        # Secure content
        secured_content, report = ssl_engine.secure_content(content)
        
        # Output secured content
        if args.in_place:
            with open(args.input, 'w', encoding='utf-8') as f:
                f.write(secured_content)
            print(f"File '{args.input}' secured in place")
        elif args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(secured_content)
            print(f"Secured content saved to: {args.output}")
        else:
            print(secured_content)
        
        # Save report if requested
        if args.report:
            report_data = {
                'validation_report': report,
                'security_summary': {
                    'security_level': report['security_level'],
                    'compliance': report['compliance'],
                    'symbol_coverage': report['symbol_coverage']
                }
            }
            with open(args.report, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            print(f"Report saved to: {args.report}")
        
        # Verbose output
        if args.verbose:
            print(f"\nSecurity Level: {report['security_level']}")
            print(f"Symbol Coverage: {report['symbol_coverage']}")
            print(f"Compliance: {report['compliance']}")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def report_command():
    """CLI command to generate security diagnostics report"""
    parser = argparse.ArgumentParser(
        description='Generate SSL security diagnostics report',
        prog='ssl-report'
    )
    parser.add_argument('input', help='Input file or directory to analyze')
    parser.add_argument('-o', '--output', help='Output file for report')
    parser.add_argument('-f', '--format', choices=['markdown', 'json', 'html'],
                       default='markdown', help='Report format')
    parser.add_argument('-c', '--config', help='Configuration file path')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for processing')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        # Initialize components
        ssl_engine = SymbolicSecurityEngine(args.config)
        diagnostics = SecurityDiagnosticsReport({'output_format': args.format})
        
        # Collect files to process
        files_to_process = []
        if os.path.isfile(args.input):
            files_to_process.append(args.input)
        elif os.path.isdir(args.input):
            for root, dirs, files in os.walk(args.input):
                for file in files:
                    if file.endswith(('.txt', '.md', '.py', '.json')):
                        files_to_process.append(os.path.join(root, file))
        else:
            print(f"Error: '{args.input}' not found", file=sys.stderr)
            sys.exit(1)
        
        if not files_to_process:
            print("No files found to process", file=sys.stderr)
            sys.exit(1)
        
        # Process files
        all_reports = []
        total_files = len(files_to_process)
        
        for i, file_path in enumerate(files_to_process):
            if args.verbose:
                print(f"Processing {i+1}/{total_files}: {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                secured_content, report = ssl_engine.secure_content(content)
                report['file_path'] = file_path
                all_reports.append(report)
                
            except Exception as e:
                if args.verbose:
                    print(f"Warning: Failed to process {file_path}: {e}")
                continue
        
        # Aggregate data for diagnostics
        aggregated_data = {
            'processing_summary': {
                'total_texts_processed': len(all_reports),
                'total_symbols_found': sum(r['symbol_count'] for r in all_reports),
                'total_symbols_anchored': sum(r['SEM_count'] for r in all_reports),
                'processing_time_seconds': sum(r['processing_time'] for r in all_reports),
                'average_coverage': f"{sum(float(r['symbol_coverage'].rstrip('%')) for r in all_reports) / len(all_reports):.1f}%" if all_reports else "0.0%"
            },
            'security_distribution': {},
            'compliance_distribution': {},
            'file_reports': all_reports
        }
        
        # Calculate distributions
        for report in all_reports:
            sec_level = report['security_level']
            compliance = report['compliance']
            
            aggregated_data['security_distribution'][sec_level] = \
                aggregated_data['security_distribution'].get(sec_level, 0) + 1
            aggregated_data['compliance_distribution'][compliance] = \
                aggregated_data['compliance_distribution'].get(compliance, 0) + 1
        
        # Generate report
        report_content = diagnostics.generate_report(aggregated_data, args.output)
        
        if not args.output:
            print(report_content)
        else:
            print(f"Report generated: {args.output}")
        
        if args.verbose:
            print(f"\nProcessed {len(all_reports)} files")
            print(f"Total symbols found: {aggregated_data['processing_summary']['total_symbols_found']}")
            print(f"Average coverage: {aggregated_data['processing_summary']['average_coverage']}")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def export_command():
    """CLI command to export symbol database"""
    parser = argparse.ArgumentParser(
        description='Export SSL symbol database',
        prog='ssl-export'
    )
    parser.add_argument('output', help='Output file path')
    parser.add_argument('-c', '--config', help='Configuration file path')
    parser.add_argument('--format', choices=['json', 'csv'], default='json',
                       help='Export format')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        # Initialize SSL engine
        ssl_engine = SymbolicSecurityEngine(args.config)
        
        if args.format == 'json':
            # Export as JSON
            ssl_engine.export_symbol_database(args.output)
            print(f"Symbol database exported to: {args.output}")
            
        elif args.format == 'csv':
            # Export as CSV
            import csv
            with open(args.output, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Symbol', 'Anchor', 'Description'])
                
                for symbol, (anchor, description) in ssl_engine.symbol_db.items():
                    writer.writerow([symbol, anchor, description])
            
            print(f"Symbol database exported to CSV: {args.output}")
        
        if args.verbose:
            print(f"Exported {len(ssl_engine.symbol_db)} symbols")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        print("SSL Command Line Interface")
        print("Available commands:")
        print("  ssl-validate  - Validate symbolic content")
        print("  ssl-secure    - Secure symbolic content")
        print("  ssl-report    - Generate security report")
        print("  ssl-export    - Export symbol database")
        print("\nUse --help with any command for more information")
        sys.exit(1)


if __name__ == '__main__':
    main()

