#!/usr/bin/env python
import sys
import os
import traceback

try:
    with open('debug_output.txt', 'w') as f:
        f.write(f"Python version: {sys.version}\n")
        f.write(f"Current working directory: {os.getcwd()}\n")
        f.write(f"Files in current directory: {os.listdir('.')}\n")
        
        f.write("\nTrying to import modules...\n")
        
        try:
            import generate_glossary
            f.write("generate_glossary imported successfully\n")
        except ImportError as e:
            f.write(f"Error importing generate_glossary: {e}\n")
            f.write(traceback.format_exc())
            
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
    
print("Debug script completed") 