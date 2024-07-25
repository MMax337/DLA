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
Specify the settings of the simulation. An example is provided in `DLA/example.txt`.

After specifying the settings, you can run the simulation using the following command:
```bash
python simulation.py path_to_file
```

### Rules
   1. Lines starting with '#' are considered comments
   2. Empty lines are skipped.

### Parameters
You should provide 2 parameters. The simulation will generate an image containing the grid with pictures for every combination.
Example:
```
X = [0, 2, 4]; D1
Y = [0, 2, 4, 8]; D2
```
After ';' you can provide your own names that would be used in the image generation.
The example settings:
```
particles = 20_000; Add = (X, 0.5); Move = (32, 0, X, 0.5); ratio = (1, 2, Y); offset = 5_000

```
1. `parctiles` - the number of particles to be placed on the grid.
2. `Add` - Parameters that determine the probability of crystallization of particles from the upstream and the 3D flux stream.
3. `Move`
      * the first 2 params determine the role of curvature on the deattachment of the particles
      * the last 2 parameters determine the role of curvature in the attachment of these particles
4. `ratio` - The ratio of upstream : moving : 3D flux particles. The values are integers.
5. `offset` The number of particles that attach before the moving and 3D flux processes start.start.
