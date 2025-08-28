"""
Security management CLI for API keys and configuration validation.

This tool helps users:
- Validate API key configuration
- Check for potential security issues
- Test API key functionality
- Review security settings
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

from generate_glossary.utils.secure_config import (
    validate_api_keys, key_manager, mask_sensitive_data, APIKeyManager
)
from generate_glossary.utils.logger import setup_logger
from generate_glossary.utils.llm import LLMFactory, Provider, OPENAI_MODELS, GEMINI_MODELS

logger = setup_logger("security_cli")

def validate_keys() -> bool:
    """Validate all API keys and show status"""
    print("\n=== API Key Validation ===")
    
    try:
        is_valid = validate_api_keys()
        status = key_manager.get_key_status()
        
        print(f"✅ Validation complete")
        print(f"📊 Summary: {status['total_loaded']} loaded, {status['total_failed']} failed\n")
        
        # Show loaded keys
        if status['loaded_keys']:
            print("🔑 Loaded API Keys:")
            for name, info in status['loaded_keys'].items():
                required_str = "Required" if info['required'] else "Optional"
                status_icon = "✅" if info['masked_value'] != "[NOT_PROVIDED]" else "⚠️"
                print(f"  {status_icon} {name}: {info['masked_value']} ({required_str})")
        
        # Show failed keys
        if status['failed_keys']:
            print("\n❌ Failed Keys:")
            for key in status['failed_keys']:
                print(f"  - {key}")
                
        return is_valid
        
    except Exception as e:
        print(f"❌ Validation failed: {mask_sensitive_data(str(e))}")
        return False

def test_llm_connectivity() -> bool:
    """Test connectivity to LLM providers"""
    print("\n=== LLM Connectivity Test ===")
    
    providers_to_test = [
        (Provider.OPENAI, OPENAI_MODELS["mini"]),
        (Provider.GEMINI, GEMINI_MODELS["default"])
    ]
    
    all_working = True
    
    for provider, model in providers_to_test:
        print(f"\n🧪 Testing {provider.upper()} with {model}...")
        
        try:
            llm = LLMFactory.create_llm(
                provider=provider,
                model=model,
                temperature=0.1
            )
            
            # Simple test prompt
            test_prompt = "Respond with exactly: 'Connection test successful'"
            response = llm.infer(
                prompt=test_prompt,
                system_prompt="You are a test system. Follow instructions exactly."
            )
            
            if response and response.text:
                print(f"  ✅ {provider.upper()}: Connected successfully")
                print(f"     Response: {response.text[:50]}...")
            else:
                print(f"  ❌ {provider.upper()}: Empty response")
                all_working = False
                
        except Exception as e:
            print(f"  ❌ {provider.upper()}: {mask_sensitive_data(str(e))}")
            all_working = False
            
    return all_working

def scan_for_exposed_keys(directory: Path) -> None:
    """Scan directory for potentially exposed API keys"""
    print(f"\n=== Security Scan: {directory} ===")
    
    suspicious_files = []
    key_exposures = []
    
    # Common file types to check
    extensions = ['.py', '.sh', '.env', '.txt', '.md', '.yaml', '.yml', '.json']
    
    for ext in extensions:
        for file_path in directory.rglob(f"*{ext}"):
            try:
                # Skip large files and directories we don't want to scan
                if file_path.stat().st_size > 1024 * 1024:  # Skip files > 1MB
                    continue
                    
                if any(skip in str(file_path) for skip in ['.git', '__pycache__', 'node_modules', '.venv']):
                    continue
                    
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                # Check for potential API keys
                manager = APIKeyManager()
                masked_content = manager.detect_and_mask_keys(content)
                
                if masked_content != content:
                    key_exposures.append({
                        'file': str(file_path),
                        'relative_path': str(file_path.relative_to(directory))
                    })
                    
                # Check for suspicious patterns
                suspicious_patterns = [
                    'api_key =',
                    'API_KEY =',
                    'secret =',
                    'SECRET =',
                    'token =',
                    'TOKEN =',
                    'password =',
                    'PASSWORD ='
                ]
                
                content_lower = content.lower()
                if any(pattern.lower() in content_lower for pattern in suspicious_patterns):
                    suspicious_files.append(str(file_path.relative_to(directory)))
                    
            except (PermissionError, UnicodeDecodeError, OSError):
                continue
                
    # Report findings
    if key_exposures:
        print(f"⚠️ Found {len(key_exposures)} files with potential API key exposure:")
        for exposure in key_exposures[:10]:  # Show first 10
            print(f"  - {exposure['relative_path']}")
        if len(key_exposures) > 10:
            print(f"  ... and {len(key_exposures) - 10} more")
    else:
        print("✅ No exposed API keys detected")
        
    if suspicious_files:
        print(f"\n🔍 Found {len(suspicious_files)} files with suspicious patterns:")
        for file in suspicious_files[:5]:  # Show first 5
            print(f"  - {file}")
        if len(suspicious_files) > 5:
            print(f"  ... and {len(suspicious_files) - 5} more")
    else:
        print("✅ No suspicious credential patterns found")

def show_security_recommendations() -> None:
    """Show security best practices"""
    print("\n=== Security Recommendations ===")
    
    recommendations = [
        "🔐 Store API keys in environment variables, never in code",
        "📝 Use .env files for development, .env.example for templates", 
        "🚫 Add .env to .gitignore to prevent accidental commits",
        "🔄 Rotate API keys regularly (quarterly recommended)",
        "📊 Monitor API key usage for unusual patterns",
        "🛡️ Use minimum required permissions for API keys",
        "📱 Use separate API keys for development vs production",
        "🔍 Regularly scan codebase for accidentally committed secrets",
        "⚠️ Never log API keys or include them in error messages",
        "🏠 Consider using cloud secret management for production"
    ]
    
    for rec in recommendations:
        print(f"  {rec}")
        
    print(f"\n📚 Additional Resources:")
    print(f"  - GitHub Secret Scanning: https://docs.github.com/en/code-security/secret-scanning")
    print(f"  - Git-secrets: https://github.com/awslabs/git-secrets")
    print(f"  - API Key Security Guide: https://cloud.google.com/docs/authentication/api-keys")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Security management for glossary generation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s validate                    # Validate all API keys
  %(prog)s test                       # Test LLM connectivity
  %(prog)s scan                       # Scan current directory for exposed keys
  %(prog)s scan --directory /path     # Scan specific directory
  %(prog)s recommendations            # Show security recommendations
  %(prog)s all                        # Run all checks
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validate command
    subparsers.add_parser('validate', help='Validate API key configuration')
    
    # Test command  
    subparsers.add_parser('test', help='Test LLM provider connectivity')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan for exposed API keys')
    scan_parser.add_argument('--directory', type=Path, default=Path.cwd(),
                            help='Directory to scan (default: current directory)')
    
    # Recommendations command
    subparsers.add_parser('recommendations', help='Show security recommendations')
    
    # All command
    all_parser = subparsers.add_parser('all', help='Run all security checks')
    all_parser.add_argument('--directory', type=Path, default=Path.cwd(),
                           help='Directory to scan (default: current directory)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
        
    exit_code = 0
    
    try:
        if args.command == 'validate':
            if not validate_keys():
                exit_code = 1
                
        elif args.command == 'test':
            if not test_llm_connectivity():
                exit_code = 1
                
        elif args.command == 'scan':
            scan_for_exposed_keys(args.directory)
            
        elif args.command == 'recommendations':
            show_security_recommendations()
            
        elif args.command == 'all':
            print("🔒 Running comprehensive security check...\n")
            
            # Run all checks
            keys_valid = validate_keys()
            llm_working = test_llm_connectivity()
            scan_for_exposed_keys(args.directory)
            show_security_recommendations()
            
            # Summary
            print(f"\n=== Security Check Summary ===")
            print(f"API Keys: {'✅ Valid' if keys_valid else '❌ Issues found'}")
            print(f"LLM Connectivity: {'✅ Working' if llm_working else '❌ Issues found'}")
            print(f"Code Scan: ✅ Completed")
            
            if not keys_valid or not llm_working:
                exit_code = 1
                
    except KeyboardInterrupt:
        print("\n🛑 Security check interrupted")
        exit_code = 1
    except Exception as e:
        print(f"\n❌ Security check failed: {mask_sensitive_data(str(e))}")
        exit_code = 1
        
    return exit_code

if __name__ == "__main__":
    sys.exit(main())