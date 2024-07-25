from __future__ import annotations
from typing import Optional

import numpy as np
import numba

from numba.experimental import jitclass




spec = [
    ('prob_fall_grid', numba.optional(numba.float64[:, :])),
    ('pg', numba.float64[:, :]),
    ('to_modify_x', numba.int64[:]),
    ('to_modify_y', numba.int64[:]),
    ('max_height', numba.int64)
]



@jitclass(spec)
class ParticleGrid:
    """
    A class to manage the grid for a Diffusion-Limited Aggregation (DLA) simulation involving particles.
    This class surves as a conviniet wrapper of the grid and related to it elements.
    
    Attributes:
        prob_fall_grid (np.ndarray or None): The probability grid for particle falling.
        pg (np.ndarray): The playground grid where particles are added.
        to_modify_x (List[int]): Array of x-coordinates for which the falling probability should be recalculated
        to_modify_y (List[int]): Array of y-coordinates for which the falling probability should be recalculated
        max_height (int): The maximal height of the aggregate.
    
    """
    def __init__(self, pg: np.ndarray):
        self.prob_fall_grid = None
        self.pg = pg
        self.to_modify_x = np.zeros(0, dtype=np.int64)
        self.to_modify_y = np.zeros(0, dtype=np.int64)
        self.max_height = pg.shape[0]

    def prob_fall(self, grid : np.ndarray):
        self.prob_fall_grid = grid

    def append_x_y(self, x : int, y : int):
        self.to_modify_x = np.append(self.to_modify_x, x)
        self.to_modify_y = np.append(self.to_modify_y, y)

    def clear(self):
        self.to_modify_x = np.zeros(0, dtype=np.int64)
        self.to_modify_y = np.zeros(0, dtype=np.int64)


    def increaseGrid(self, by : Optional[int] = 100):
      # TODO: In future releases implement convole2d yourself to make it compatible 
      # and get rid of the akwardness. So the update grid is called automatically
      # Using some basic loops and so on.
      """
        Increases the size of the grid by adding "by" rows at the top.
        After this function execution the prob_fall_grid is None and has to be reset!
        The prob_fall calculation relies on the convolve2d which is incompatible with numba.
        
        Args:
            by (Optional[int] = 100) : number of rows to be added
      """
      new = np.zeros((self.pg.shape[0] + by, self.pg.shape[1]))
      new[-self.pg.shape[0] :, :] = self.pg
      self.max_height += 100
      self.prob_fall_grid = None
      
      self.clear()

      self.pg = new
