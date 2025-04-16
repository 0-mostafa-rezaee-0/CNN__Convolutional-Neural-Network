# LSTM-Based Time Series Forecasting Project Prompt

<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">LSTM-Based Time Series Forecasting Project</div>

<div style="text-align:center; margin-bottom:30px;">A comprehensive project for time series forecasting using Long Short-Term Memory networks in Docker</div>

## Table of Contents

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#task-overview"><i><b>1. Task Overview</b></i></a>
</div>
&nbsp;

<details>
  <summary><a href="#project-requirements"><i><b>2. Project Requirements</b></i></a></summary>
  <div>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#core-functionality">2.1. Core Functionality</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#technical-specifications">2.2. Technical Specifications</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#project-structure">2.3. Project Structure</a><br>
  </div>
</details>
&nbsp;

<details>
  <summary><a href="#specific-implementation-details"><i><b>3. Specific Implementation Details</b></i></a></summary>
  <div>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#1-docker-environment">3.1. Docker Environment</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#2-lstm-architecture">3.2. LSTM Architecture</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#3-training-approach">3.3. Training Approach</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#4-visualization-requirements">3.4. Visualization Requirements</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#5-documentation-standards">3.5. Documentation Standards</a><br>
  </div>
</details>
&nbsp;

<details>
  <summary><a href="#expected-project-structure-for-readme-files"><i><b>4. Expected Project Structure for README Files</b></i></a></summary>
  <div>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#main-readmemd">4.1. Main README.md</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#directory-specific-readmes">4.2. Directory-Specific READMEs</a><br>
  </div>
</details>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#educational-focus"><i><b>5. Educational Focus</b></i></a>
</div>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#deliverables"><i><b>6. Deliverables</b></i></a>
</div>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#comparison-with-traditional-models"><i><b>7. Comparison with Traditional Models</b></i></a>
</div>
&nbsp;

<details>
  <summary><a href="#toc-generation-rules"><i><b>8. TOC Generation Rules</b></i></a></summary>
  <div>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#template-rules">8.1. Template Rules</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#example">8.2. Example</a><br>
  </div>
</details>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#tree-structure-documentation"><i><b>9. Tree Structure Documentation</b></i></a>
</div>
&nbsp;

<details>
  <summary><a href="#docker-setup-instructions"><i><b>10. Docker Setup Instructions</b></i></a></summary>
  <div>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#step-1-build-and-run-your-container">10.1. Step 1: Build and Run Your Container</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#step-2-verify-the-container">10.2. Step 2: Verify the Container</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#step-3-attach-vs-code-to-the-container">10.3. Step 3: Attach VS Code to the Container</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#step-4-run-the-python-script">10.4. Step 4: Run the Python Script</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#step-5-work-with-jupyter-notebooks-in-vs-code">10.5. Step 5: Work with Jupyter Notebooks in VS Code</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#step-6-stop-and-remove-the-container">10.6. Step 6: Stop and Remove the Container</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#note-1-jupyter-on-browser">10.7. Note 1: Jupyter on Browser</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#note-2-keeping-your-environment-up-to-date">10.8. Note 2: Keeping Your Environment Up-to-Date</a><br>
  </div>
</details>
&nbsp;

<details>
  <summary><a href="#essential-docker-commands"><i><b>11. Essential Docker Commands</b></i></a></summary>
  <div>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#managing-images">11.1. Managing Images</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#managing-containers">11.2. Managing Containers</a><br>
  </div>
</details>
&nbsp;

## Task Overview

Create a comprehensive, well-documented project for time series forecasting using Long Short-Term Memory (LSTM) neural networks within a Docker container environment. The project should follow a clear, educational structure suitable for students learning about sequence modeling, deep learning, and containerization.

## Project Requirements

### Core Functionality
1. Download and preprocess benchmark time series datasets (e.g., stock prices, weather data, energy consumption)
2. Create LSTM model architectures suitable for time series forecasting
3. Train the models with appropriate techniques (early stopping, model checkpoints)
4. Visualize training progress and results
5. Save and evaluate the trained models
6. Provide interactive notebooks for exploration

### Technical Specifications
1. Create a Docker-based development environment with all necessary dependencies
2. Use TensorFlow/Keras for implementing the LSTM models
3. Develop reusable Python scripts for each major function
4. Generate comprehensive visualizations for model understanding
5. Document all components thoroughly with README files

### Project Structure

Follow this specific directory structure exactly. All the README.md files should follow the documentation standards outlined in the "Tree Structure Documentation" section below.

## Tree Structure Documentation

TASK: 
Turn any folder/file list I give you into a Markdown‑compatible tree diagram that follows the rules below.

TEMPLATE RULES: 
1. Output one fenced code block using triple back‑ticks ``` (no language tag).  
2. First line inside the block must be exactly:  Folder PATH listing
3. Inside every directory:  
   • List sub‑directories first (alphabetical).  
   • Then list files (alphabetical).  
4. Connectors & indentation  
   • Directory on the current level → prefix +---  
   • File (any depth)               → for each ancestor directory, write │     
                                      then four spaces (␣␣␣␣) and the filename  
                                      (no "+---").  
   • Maintain the vertical spine by repeating │    for each ancestor level.  
5. Alignment  
   • Pad with spaces so all descriptions start in the same column.  
   • After the name, add two spaces, then <-- and a concise description.  
6. Insert a line containing only │ after the last child of every directory to visually close that block.  
7. Output nothing outside the code block—no commentary, no extra Markdown.

```
Folder PATH listing
+---data                        <-- Data directory for time series datasets
│   +---processed               <-- Preprocessed time series data
│   +---raw                     <-- Raw time series datasets
│   │       README.md           <-- Documentation for the dataset
│
+---figures                     <-- Visualizations and plots
│       forecast_plots.png      <-- Time series forecast visualizations
│       training_history.png    <-- Training/validation metrics over time
│       lstm_states.png         <-- LSTM internal state visualizations
│       correlation_matrix.png  <-- Feature correlation heatmap
│       prediction_samples.png  <-- Examples of predictions
│       README.md               <-- Documentation for visualizations
│
+---models                      <-- Model implementations and saved files
│   +---architectures           <-- LSTM architecture implementations
│   +---configs                 <-- Model configuration files
│   +---evaluation              <-- Model evaluation utilities
│   +---training                <-- Model training utilities
│   │       README.md           <-- Documentation for models
│   +---__init__.py             <-- Package initialization
│   +---lstm_model_best.h5      <-- Best model based on validation loss
│   +---lstm_model_final.h5     <-- Final trained model
│   +---model_factory.py        <-- Factory for creating model instances
│   +---model_registry.py       <-- Registry of available models
│
+---notebooks                   <-- Jupyter notebooks for interactive learning
│       time_series_lstm.ipynb  <-- Main notebook for the project
│       exploratory_analysis.ipynb  <-- Basic data exploration notebook
│       README.md               <-- Documentation for notebooks
│
+---scripts                     <-- Python scripts
│       data_prep.py            <-- Download and preprocess time series data
│       train_lstm.py           <-- Train the LSTM model
│       evaluate_models.py      <-- Evaluate model performance
│       forecast.py             <-- Generate forecasts with trained models
│       README.md               <-- Documentation for scripts
│
+---utils                       <-- Utility functions and helpers
│       metrics.py              <-- Custom evaluation metrics
│       visualization.py        <-- Visualization utilities
│       preprocessing.py        <-- Preprocessing utilities
│       README.md               <-- Documentation for utilities
│
│       .dockerignore           <-- Docker ignore file
│       .gitignore              <-- Git ignore file
│       docker-compose.yml      <-- Docker Compose configuration
│       Dockerfile              <-- Docker configuration for environment setup
│       LICENSE                 <-- License information
│       README.md               <-- Project overview
│       requirements.txt        <-- Python dependencies for the project
│       start.sh                <-- Startup script for Docker container
│       prompt_LSTM.md          <-- Original project specifications
```

Make sure to follow this exact structure when implementing the project, including all files and directories as shown.

## Specific Implementation Details

### 1. Docker Environment
- Base on Python 3.9 or higher
- Include TensorFlow, Keras, NumPy, Pandas, Matplotlib, Scikit-learn, and Seaborn
- Configure Jupyter Notebook access on port 8888
- Include a startup script to simplify container launch

### 2. LSTM Architecture
- Create models with:
  - Input layer accepting variable-length time series data
  - Multiple LSTM layers with appropriate units
  - Dropout for regularization
  - Dense layers for forecasting
  - Options for univariate and multivariate forecasting
- Use appropriate activation functions throughout

### 3. Training Approach
- Implement data windowing for sequence modeling
- Use Mean Squared Error (MSE) or Mean Absolute Error (MAE) loss
- Apply Adam optimizer
- Implement early stopping and model checkpointing
- Monitor validation loss
- Save the best model during training

### 4. Visualization Requirements
- Generate time series plots with actual vs. predicted values
- Plot training and validation loss
- Visualize LSTM internal states and feature importance
- Create correlation heatmaps for multivariate data
- Show examples of successful and failed predictions

### 5. Documentation Standards
- Each directory must have a comprehensive README.md file
- READMEs should include:
  - Title and overview
  - Detailed table of contents with collapsible sections when appropriate
  - File descriptions and purposes
  - Usage instructions
  - Implementation details
- Document code with detailed comments
- Maintain consistent styling across all documentation

## Expected Project Structure for README Files

### Main README.md
- Title displayed as `<h1>` in a center-aligned div
- Table of contents with collapsible sections for items with subheadings
- Sections including Project Overview, Educational Objectives, Prerequisites, Project Structure, Getting Started, Project Components, Learning Exercises, Common Issues, Resources for Further Learning, and License

### Directory-Specific READMEs
- Follow the same styling as the main README
- Include specific details about the files in that directory
- Provide usage examples relevant to the directory's purpose
- Feature tables of contents appropriate to each file's length and complexity

## Educational Focus
The project should serve as a learning tool for:
- Understanding LSTM architectures and their advantages for sequence modeling
- Visualizing and interpreting LSTM internal states
- Comparing performance between different forecasting approaches
- Best practices for model training, validation, and testing
- Docker containerization for reproducible machine learning environments

## Deliverables
1. Complete project structure with all files and directories
2. Working Docker container with all dependencies
3. Executable Python scripts for all core functionality
4. Interactive Jupyter notebooks with explanations
5. Comprehensive documentation in README files
6. Visualization outputs for model understanding
7. Trained LSTM models with strong forecasting performance

## Comparison with Traditional Models
The project should explicitly highlight:
- Performance differences between LSTM and traditional time series forecasting methods (ARIMA, exponential smoothing)
- Why LSTMs work better for complex time series with long-term dependencies
- Computational trade-offs between the approaches
- Memory capabilities of LSTM vs traditional methods
- Visualization of LSTM memory cell states to show how they capture patterns

## TOC Generation Rules
TASK:
Build a Table of Contents that works in a Markdown (.md) file, following the exact format below.

### Template Rules
1. For every level‑1 heading (H1) that has at least one level‑2 child (H2), create a collapsible block:
   <details>
     <summary><a href="#{slug}"><i><b>{num}. {title}</b></i></a></summary>
     <div>
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{children}
     </div>
   </details>
   &nbsp;

2. Inside each collapsible block, list the H2 children:
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#{slug}">{num}.{subnum}. {title}</a><br>

3. For an H1 without H2 children, output a simple, non‑collapsible div:
   <div>
     &nbsp;&nbsp;&nbsp;&nbsp;<a href="#{slug}"><i><b>{num}. {title}</b></i></a>
   </div>
   &nbsp;

4. Convert every heading text to a lowercase‑hyphen "slug" for the href targets
   (e.g., "1.2. My Heading" → #12-my-heading).

5. Preserve all non‑breaking spaces (&nbsp;) and <br> exactly as shown.

6. Underlining: rely on the browser's default link styling; do not add extra CSS.

7. IMPORTANT: A Heading is collapsible only if it has sub heading.

8. The order of headings in the Table of Contents is based on the order of headings in text.

### Example
<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#heading-1"><i><b>1. Heading 1</b></i></a>
</div>
&nbsp;

<details>
  <summary><a href="#heading-1-1"><i><b>2. Heading 1</b></i></a></summary>
  <div>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#heading-2">2.1. Heading 2</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#heading-2-1">2.2. Heading 2</a><br>
  </div>
</details>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#heading-1-2"><i><b>3. Heading 1</b></i></a>
</div>
&nbsp;

## Docker Setup Instructions

### Step 1: Build and Run Your Container

On your host machine (in the project folder), you have two options:

* **First (recommended):**
  This method extracts the project name to use as the image and container names.

  To make start.sh executable if it is not:

  ```bash
  chmod +x start.sh 
  ```

  To extract the project name and then build the image and run the container:

  ```bash
  ./start.sh 
  ```
* **Second:**    
  In this method, the image and container names default to `lstm-time-series`.

  ```bash
  docker-compose up --build -d
  ```

**Note:**

* `--build`: We could omit "--build", but then changes to Dockerfile or dependencies would not be applied.
* `-d`: The "-d" flag runs the container in detached mode, allowing you to continue using the terminal for other tasks.

### Step 2: Verify the Container

Run:

```
docker-compose ps
```

Make sure the container status is "Up" and port 8888 is mapped.

### Step 3: Attach VS Code to the Container

Follow these steps carefully:

1. Press `Ctrl+Shift+P` to open the command palette.
2. Type and select `Dev Containers: Attach to Running Container…`.
3. Choose the container named `lstm-time-series`. A second VS Code window will open.
4. In the second VS Code window, click `Open Folder`. In the top box, you will see `/root`. Delete `root` to reveal `app`. Select `app` and click `OK`. You will then see all your project's folders and files.
5. In the second VS Code window, install the following extensions: `Docker`, `Dev Containers`, `Python`, and `Jupyter`. If you see a `Reload the window` button after installing each extension, make sure to click it every time.
6. You are all set and can continue.

**Note**: If you cannot select the kernel, close the second VS Code window and repeat steps 1, 2, 3, and 4. The correct kernel will then be automatically attached to the notebooks.

### Step 4: Run the Python Script

In the VS Code terminal, open the terminal. You will see a bash which means you are inside the container. Run:

```
python scripts/data_prep.py
```

You should see the expected output (downloading and preprocessing datasets).

### Step 5: Work with Jupyter Notebooks in VS Code

- Open exploratory_analysis.ipynb in VS Code.
- In the top-right corner of the notebook, you should see a kernel with the same name as your project. If not, click the `Select Kernel` button and choose the `Jupyter kernel` option. This will display a kernel with your project's name and the Python kernel specified in the Dockerfile. The libraries from the `requirements.txt` file, installed in the Docker container, will be automatically available for use.
- You can now run and edit cells within the container.

### Step 6: Stop and Remove the Container

```
docker-compose down
```

### Note 1: Jupyter on Browser

Access the Jupyter interface directly in your browser at localhost:8888

### Note 2: Keeping Your Environment Up-to-Date

- To rebuild your container with any changes, run on your host:

  ```
  docker-compose up --build
  ```
- After installing a new package, update requirements.txt inside the container by running:

  ```
  pip freeze > requirements.txt
  ```
- For pulling the latest base image, run:

  ```
  docker-compose build --pull
  ```

## Essential Docker Commands

### Managing Images

```
# Pull images from Docker Hub
docker pull python:3.9-slim
docker pull jupyter/datascience-notebook

# List all images
docker images

# Remove images
docker rmi <image1> <image2> ...
```

### Managing Containers

```
# List running containers
docker ps

# List all containers (including stopped ones)
docker ps -a

# List only container IDs
docker ps -aq

# Remove containers
docker rm <CONTAINER1> <CONTAINER2> ...

# Remove all containers
docker rm $(docker ps -aq)

# Run a container in detached mode
docker run -d <IMAGE name or ID>

# Start/stop containers
docker start <CONTAINER name or ID>
docker stop <CONTAINER name or ID>

# Start/stop all containers at once
docker start $(docker ps -aq)
docker stop $(docker ps -aq)
```

**Note**: You can use just the first few characters of a container ID for identification. For example: `docker stop 2f`











