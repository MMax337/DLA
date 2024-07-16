from __future__ import annotations


import numpy as np
from numba import jit
from scipy.signal import convolve2d
from typing import List

from ParticleGrid import ParticleGrid

      
    

@jit(nopython=True)
def check_neighbors(x: int, y: int, play_ground: np.ndarray) -> bool:
    """
    Check if there are any neighboring cells with particles. 
    The periodic boundary conditions are imposed on the y axis.
    
    Args:
        x (int): The x-coordinate of the cell.
        y (int): The y-coordinate of the cell.
        play_ground (np.ndarray): The grid.
    
    Returns:
        bool: True if there is a neighboring cell with a value of 1, False otherwise.
    """
    if (x >= play_ground.shape[0] - 1 or x <= 0):  
        return play_ground[x, (y+1)%play_ground.shape[1]] == 1 or play_ground[x, (y-1)%play_ground.shape[1]] == 1
    
    return play_ground[x+1 , y] == 1 or play_ground[x-1, y] == 1 \
        or play_ground[x, (y+1)%play_ground.shape[1]] == 1 or play_ground[x, (y-1)%play_ground.shape[1]] == 1


@jit(nopython=True)
def check_for_exit(x : int, y : int, pg : np.ndarray):
    """
    Check if a particle can exit, i.e. is not surrounded by 4 particles.
    
    Args:
        x (int): The x-coordinate of the cell.
        y (int): The y-coordinate of the cell.
        pg (np.ndarray): The playground grid.
    
    Returns:
        bool: True if the cell can exit, False otherwise.
    """
    if (x >= pg.shape[0] - 1  or x <= 0):  
        return pg[x, (y+1)%pg.shape[1]] != 1 or pg[x, (y-1)%pg.shape[1]] != 1
    
    return pg[x+1, y] != 1 or pg[x-1, y] != 1 \
        or pg[x, (y+1)%pg.shape[1]] != 1 or pg[x, (y-1)%pg.shape[1]] != 1


@jit(nopython=True)
def prob(x: int, y: int, playground: np.ndarray, A: float, B: float) -> float:
    """
    Calculate the probability of attaching based on the local curvature.
    The probability is computed according to Vicsek's work "Pattern Formation in Diffusion-Limited Aggregation" (1984). 
    
    Args:
        x (int): The x-coordinate of the cell.
        y (int): The y-coordinate of the cell.
        playground (np.ndarray): The playground grid.
        A (float): Coefficient for the probability calculation.
        B (float): Coefficient for the probability calculation.
    
    Returns:
        float: The calculated probability.
    """
    l = 9
    half_l = l // 2
    n0 = (l - 1) / (2 * l)
    n = np.sum(playground[x - half_l: x+half_l+1, y - half_l: y+half_l+1]) / (l * l)
    prob  = A * (n - n0) + B
    p = prob if prob > 0.01 else 0.01
    return p


def prob_grid_fall(pg: np.ndarray, kernel: np.ndarray, A: float, B: float) -> np.ndarray:
    # TODO: rewrite convolve2d and include it into ParticleGrid
    """
    Calculate the probability grid for particle fall based on the local curvature.
    
    
    Args:
        pg (np.ndarray): The playground grid.
        kernel (np.ndarray): The convolution kernel.
        A (float): Coefficient for the probability calculation.
        B (float): Coefficient for the probability calculation.
    
    Returns:
        np.ndarray: The grid of the values that can be translated to probability by the 
        inverse transform sampling.
        (In the grid values can be negative and larger than one.)
    """
    l = kernel.shape[0]
    n0 = (l - 1) / (2 * l)
    out = convolve2d(pg, kernel, mode='same', boundary='fill', fillvalue=0) / kernel.size

    prob = np.exp(A * (n0 - out) + B)
    prob *= pg

    prob -= np.min(prob)  
    prob *= pg
    prob[-1, :] = 0

    return prob


def get_index(prob_grid: np.ndarray, pg : np.ndarray):
    """
    Using inverse transform sampling generates an index where particle will fall.
    This particle is ensured to be able to exit.
    
    Args:
        prob_grid (np.ndarray): The probability grid.
        pg (np.ndarray): The playground grid.
    
    Returns:
        tuple: The coordinates of the chosen cell.
    """
    
    flat_grid = prob_grid.ravel() 
    
    cumulative_probs = np.cumsum(flat_grid)

    while True:
        random_num = np.random.uniform() * cumulative_probs[-1]
        chosen_index = np.searchsorted(cumulative_probs, random_num)
        
        chosen_coords = np.unravel_index(chosen_index, prob_grid.shape)

        if check_for_exit(chosen_coords[0], chosen_coords[1], pg):
            break
        
    return chosen_coords

def update_prob_grid(grid : ParticleGrid, kernel: np.ndarray, A_fall: float, B_fall: float):
    # TODO: rewrite convolve2d and include it into ParticleGrid
    """
    Update the probability grid for particle fall.
    
    Args:
        grid (ParticleGrid): The ParticleGrid object.
        kernel (np.ndarray): The convolution kernel.
        A_fall (float): Coefficient for the probability calculation.
        B_fall (float): Coefficient for the probability calculation.
    """
    k_x, k_y = kernel.shape
    for x,y in zip(grid.to_modify_x, grid.to_modify_y):
      prob = prob_grid_fall(grid.pg[max(x-k_x + 1, 0) :x + k_x,  max(y - k_y + 1,0) :y + k_y], kernel, A_fall, B_fall)
      grid.prob_fall_grid[max(x-k_x + 1, 0) :x + k_x,  max(y - k_y + 1,0) :y + k_y] = prob

    grid.clear()
    


def particle_fall(grid : ParticleGrid, particle_number: int, kernel: np.ndarray, A: float, B: float, A_add: float, B_add: float) -> None:
    """
    Simulate the fall of a specified number of particles.
    
    Args:
        grid (ParticleGrid): The ParticleGrid object.
        particle_number (int): The number of particles to simulate.
        kernel (np.ndarray): The convolution kernel.
        A (float): Coefficient for the probability calculation.
        B (float): Coefficient for the probability calculation.
        A_add (float): Additional coefficient for the probability calculation.
        B_add (float): Additional coefficient for the probability calculation.
    """
    num = 0
    while num < particle_number:
        if(grid.prob_fall_grid is None): 
            grid.prob_fall(prob_grid_fall(grid.pg, kernel, A, B))
            grid.clear()
        else: update_prob_grid(grid, kernel, A, B)
        index = get_index(grid.prob_fall_grid, grid.pg)
        
        grid.pg[index] = 0
        grid.append_x_y(index[0], index[1])
        move_particle(grid, pos=index, A=A_add, B=B_add)
        num += 1
    
    return grid.pg


@jit(nopython=True)
def add_particles(particles: int, grid : ParticleGrid, A, B):
    """
    Simulate adding particles from the top border.
    
    Args:
        particles (int): The number of particles to add.
        grid (ParticleGrid): The ParticleGrid object.
        A (float): Coefficient for the probability calculation.
        B (float): Coefficient for the probability calculation.
    """
    num_particles = 0
    while num_particles < particles:
        generate_add_particle(grid, A, B)
        num_particles += 1

    return


@jit(nopython=True)
def generate_add_particle(grid : ParticleGrid, A: float, B: float) -> bool:
    """
    Generate and add a single particle to the grid.
    
    Args:
        grid (ParticleGrid): The ParticleGrid object.
        A (float): Coefficient for the probability calculation.
        B (float): Coefficient for the probability calculation.
    """

    pos = (grid.max_height - 50, np.random.randint(1, grid.pg.shape[1] - 1))
    move_particle(grid, pos, A, B)


@jit(nopython=True)
def move_particle(grid : ParticleGrid, pos: tuple[int, int], A: float, B: float) -> bool:
    """
    Helper function that perfoms the random walk for the specified particle
    
    Args:
        grid (ParticleGrid): The ParticleGrid object.
        pos (tuple[int, int]): The starting position of the particle.
        A (float): Coefficient for the inclusion of the surface tension in attachment
        B (float): Coefficient for the inclusion of the surface tension in attachment
    
    Returns:
        bool: True if the particle was successfully moved, False otherwise.
    """
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    play_ground = grid.pg
    size_x, size_y = play_ground.shape
    
    x_position, y_position = pos

    count = 0
    while True:
        count += 1
        if count > 1_000_000:
            return
        
        index = np.random.randint(0, 4)
        dx, dy = directions[index]
        x_position = (x_position + dx) % size_x
        y_position = (y_position + dy) % size_y

        if play_ground[x_position, y_position] == 1 or x_position < grid.max_height - 60:
            x_position -= dx
            y_position -= dy
            continue

        if check_neighbors(x_position, y_position, play_ground):
            if prob(x_position, y_position, play_ground, A, B) >= np.random.uniform(): 
                play_ground[x_position, y_position] = 1
                grid.append_x_y(x_position, y_position)
                if x_position < grid.max_height:
                    grid.max_height = x_position
                    if(x_position - 50 <= 0):
                        grid.increaseGrid()
                
                return


def flux3d(grid : ParticleGrid, particle_number : int, A_add : float, B_add : float):
    """
    Simulates the 3D flux of particles within the grid.

    Args:
        grid (ParticleGrid): The ParticleGrid object representing the grid.
        particle_number (int): The number of particles to simulate.
        A_add (float): Parameter A for the probability function used in particle movement.
        B_add (float): Parameter B for the probability function used in particle movement.
    """
    playground = grid.pg
    num = 0
    while num < particle_number:
        low = max(0, grid.max_height - 20) *playground.shape[0]
        index = np.unravel_index(np.random.randint(low=low, high=playground.size), shape=playground.shape)


        while playground[index] != 0:
            index = np.unravel_index(np.random.randint(low=low, high=playground.size), shape=playground.shape)
        move_particle(grid, pos=index, A=A_add, B=B_add) 
        num += 1
    
    return


def DLA(playground, particles, add_fall_ratio: List[int], add_param: tuple[float, float], fall_param: tuple[float, float], offset: int):
    """
    Simulates the Diffusion-Limited Aggregation (DLA) process.

    Args:
        playground (np.ndarray): The initial playground grid.
        particles (int): The total number of particles to simulate.
        add_fall_ratio (List[int]): List containing the ratio of particles to add, particles to fall, and flux3d particles.
        add_param (tuple[float, float]): Parameters (A_add, B_add) for the probability function used in particle addition.
        fall_param (tuple[float, float]): Parameters (A_fall, B_fall, A_fall_add, B_fall_add) for the probability function used in particle falling.
        offset (int): Initial number of particles to add.

    Returns:
        np.ndarray: The final playground grid after the DLA process.
    """

    A_add, B_add = add_param
    A_fall, B_fall, A_fall_add, B_fall_add = fall_param

    kernel = np.ones((9, 9))
    grid = ParticleGrid(playground)

    num_particles = 0
    add_particles(particles=offset, grid=grid, A=A_add, B=B_add)
    num_particles += offset

    
    to_add, to_fall, flux3d_num = add_fall_ratio

    while num_particles < particles:
        add_particles(particles=to_add, grid=grid, A=A_add, B=B_add)
        particle_fall(grid, to_fall, kernel, A_fall, B_fall, A_add=A_fall_add, B_add=B_fall_add)
        flux3d(grid, flux3d_num, A_add, B_add)
        num_particles += to_add + flux3d_num

    return grid.pg


