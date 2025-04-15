#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check and validate the CNN MNIST project structure.
This script verifies that all required files and directories exist.
"""

import os
import sys
import json

# Define the expected project structure
EXPECTED_STRUCTURE = {
    "dirs": [
        "data",
        "data/mnist",
        "data/mnist_samples",
        "figures",
        "models",
        "notebooks",
        "scripts",
        "utils"
    ],
    "files": [
        "Dockerfile",
        "docker-compose.yml",
        "requirements.txt",
        "start.sh",
        "run_pipeline.sh",
        "README.md",
        "data/README.md",
        "figures/README.md",
        "models/README.md",
        "notebooks/README.md",
        "scripts/README.md",
        "utils/README.md",
        "utils/check_project_structure.py",
        "utils/project_summary.py",
        "utils/model_comparison.py",
        "scripts/data_prep.py",
        "scripts/extract_sample_images.py",
        "scripts/train_cnn.py",
        "scripts/visualize_features.py"
    ],
    "executables": [
        "start.sh",
        "run_pipeline.sh",
        "scripts/data_prep.py",
        "scripts/extract_sample_images.py",
        "scripts/train_cnn.py",
        "scripts/visualize_features.py",
        "utils/check_project_structure.py",
        "utils/project_summary.py",
        "utils/model_comparison.py"
    ]
}

def check_structure():
    """Check if all expected directories and files exist."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    print("Checking CNN MNIST project structure...")
    print(f"Project root: {project_root}")
    
    missing_dirs = []
    missing_files = []
    non_executable_files = []
    
    # Check directories
    for dir_path in EXPECTED_STRUCTURE["dirs"]:
        if not os.path.isdir(dir_path):
            missing_dirs.append(dir_path)
    
    # Check files
    for file_path in EXPECTED_STRUCTURE["files"]:
        if not os.path.isfile(file_path):
            missing_files.append(file_path)
    
    # Check executables
    for exec_path in EXPECTED_STRUCTURE["executables"]:
        if os.path.isfile(exec_path) and not os.access(exec_path, os.X_OK):
            non_executable_files.append(exec_path)
    
    # Print results
    if missing_dirs:
        print("\nMissing directories:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
    
    if missing_files:
        print("\nMissing files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
    
    if non_executable_files:
        print("\nNon-executable files (should be executable):")
        for exec_path in non_executable_files:
            print(f"  - {exec_path}")
    
    # Summary
    if not missing_dirs and not missing_files and not non_executable_files:
        print("\n✅ Project structure is complete and valid.")
        return True
    else:
        print("\n❌ Project structure has issues.")
        print(f"  Missing directories: {len(missing_dirs)}")
        print(f"  Missing files: {len(missing_files)}")
        print(f"  Non-executable files: {len(non_executable_files)}")
        return False

def fix_executables():
    """Make required files executable."""
    for exec_path in EXPECTED_STRUCTURE["executables"]:
        if os.path.isfile(exec_path) and not os.access(exec_path, os.X_OK):
            try:
                os.chmod(exec_path, 0o755)
                print(f"Made executable: {exec_path}")
            except Exception as e:
                print(f"Error making {exec_path} executable: {e}")

def main():
    """Main function to check and optionally fix project structure."""
    if len(sys.argv) > 1 and sys.argv[1] == "--fix":
        print("Checking and fixing project structure...")
        check_structure()
        print("\nFixing executable permissions...")
        fix_executables()
        print("\nRechecking structure...")
        check_structure()
    else:
        check_structure()

if __name__ == "__main__":
    main() 