from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np



def generate_image(grids : Dict[Tuple[any, any], np.ndarray], 
                   dimensions : tuple[int,int], subtitle_name : tuple[str, str],
                   title : str):
    #rows columns
    r, c = dimensions
  

    params = list(grids.keys())
      
    params.sort()
    fig, axs = plt.subplots(r, c, figsize=(11,11) , squeeze=True)
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8, wspace=0.15, hspace=0.1)
    max_val = max([val.shape[0] for val in grids.values()])

    for i, elem in enumerate(reversed(params)):
        if r != 1 and c != 1:
            ax = axs[i//c][c - 1 - i%c]
        elif  r != 1 or c != 1:
            ax = axs[i]
        else: # r == 1, c == 1
            ax = axs

        padded_arr = np.zeros((max_val, grids[elem].shape[1]))
        padded_arr[(max_val - grids[elem].shape[0]):] = grids[elem]
        ax.imshow(padded_arr, aspect='auto')
        ax.axis('off')
        ax.set_title(f"{subtitle_name[0]}={elem[0]}, {subtitle_name[1]}={elem[1]}")

    fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(title + '.png', dpi = 1000)