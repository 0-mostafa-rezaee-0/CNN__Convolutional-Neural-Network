<div style="font-size:2em; font-weight:bold; text-align:center; margin-top:20px;">Utilities Directory</div>

## Table of Contents 
<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#1-overview"><i><b>1. Overview</b></i></a>
</div>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#2-directory-contents"><i><b>2. Directory Contents</b></i></a>
</div>
&nbsp;

<details>
  <summary><a href="#3-utility-descriptions"><i><b>3. Utility Descriptions</b></i></a></summary>
  <div>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#31-check_project_structurepy">3.1. check_project_structure.py</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#32-project_summarypy">3.2. project_summary.py</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#33-model_comparisonpy">3.3. model_comparison.py</a><br>
  </div>
</details>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#4-development"><i><b>4. Development</b></i></a>
</div>
&nbsp;

## 1. Overview

This directory contains utility scripts for maintaining and checking the CNN MNIST project.

## 2. Directory Contents

```
utils/
├── check_project_structure.py    # Script to validate project structure
├── model_comparison.py           # Script to compare CNN and ANN models
└── project_summary.py            # Script to generate project status summary
```

## 3. Utility Descriptions

### 3.1 check_project_structure.py

This script validates that all required files and directories exist in the project:
- Checks for all expected directories
- Verifies that required files are present
- Ensures executable files have proper permissions

The script can also fix issues with executable permissions.

Usage:
```bash
# Check project structure
python utils/check_project_structure.py

# Check and fix executable permissions
python utils/check_project_structure.py --fix
```

### 3.2 project_summary.py

This script generates a comprehensive summary of the project's current state:
- Data availability and statistics
- Model status, size, and performance metrics
- Available visualizations
- Timestamp information for all components

The summary is displayed in the terminal and also saved as a JSON file.

Usage:
```bash
# Generate and display project summary
python utils/project_summary.py
```

The script will create a `project_summary.json` file in the project root directory.

### 3.3 model_comparison.py

This script performs a comprehensive comparison between CNNs and traditional ANNs on the MNIST dataset:
- Trains both a CNN and a traditional ANN model with the same dataset
- Measures training time, accuracy, and parameter counts
- Generates visualizations comparing performance metrics
- Outputs a detailed comparison table

The comparison helps understand why CNNs are more effective for image processing tasks.

Usage:
```bash
# Run the comparison with default settings
python utils/model_comparison.py

# Customize training parameters
python utils/model_comparison.py --epochs 15 --batch_size 64
```

The script will create a comparison visualization in the figures directory.

## 4. Development

When adding new files or directories to the project, consider updating the `check_project_structure.py` script to include those new components in the validation process. 