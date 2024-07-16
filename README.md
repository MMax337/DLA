# Diffusion-Limited Aggregation (DLA) with Surface Tension

## Project Overview

This project was developed as part of a research group with the primary goal of incorporating surface tension into the Diffusion-Limited Aggregation (DLA) process. DLA is a process that describes the growth pattern of clusters formed by particles adhering to a substrate in a random manner. By including surface tension, we aim to simulate more realistic physical phenomena and enhance the understanding of pattern formation in natural systems.

## Objectives

1. **Incorporate Surface Tension:** Introduce surface tension effects into the traditional DLA model to better simulate real-world phenomena.
2. **Implement Multiple Models:** Develop and compare three different models incorporating surface tension to study their impact on the aggregation patterns.
3. **Analyze Results:** Conduct experiments and analyze the resulting patterns to understand the influence of surface tension on DLA.

## Features

- **Grid-Based Simulation:** The grid is used to simulate particle movement and aggregation.
- **Probability-Based Movement:** Particle aggregation are controlled by probability functions, incorporating parameters for surface tension.
- **Dynamic Grid Expansion:** The grid dynamically expands to accommodate growing clusters.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/dla-surface-tension.git
   cd dla-surface-tension
   ```
2. **Install the dependencies**:
  ```bash
      pip install -r requirements.txt
  ```

## Usage
Import the DLA function to your script and run the simulation.
```python
  from DLA import DLA
```
