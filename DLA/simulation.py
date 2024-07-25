import sys
from pathlib import Path
import ast
import numpy as np
from dla import DLA
from imgen import generate_image

def line2params(line):
    def split_tuple(string : str, converter ):
        return tuple(map(converter, map(lambda x: x.strip(), string.strip('()').split(','))))

    conversion = {
        'particles' : int,
        'offset' : int,
        'Add' : lambda x: split_tuple(x, float),
        'Move' : lambda x: split_tuple(x, float),
        'ratio' : lambda x: split_tuple(x, int),
    }

    params = dict(map(lambda x: [token.strip() for token in x.split('=')], line.split(';')))

    for key, val in params.items():
        params[key] = conversion[key](val)
    
    return params

if len(sys.argv) != 2:
    print("Please provide the path to settings")
    exit()

path = Path(sys.argv[1])

with open(path, 'r') as file:
    lines = [line.strip() for line in file if line != '' and line !='\n' and line[0] != '#']


X_split = lines[0].split(';')
Y_split = lines[1].split(';')

names = ('X' if len(X_split) != 2 else X_split[1].strip(),
         'Y' if len(Y_split) != 2 else Y_split[1].strip())


X = ast.literal_eval(X_split[0].split('=')[1].strip())
Y = ast.literal_eval(Y_split[0].split('=')[1].strip())
last = lines[-1]


shape = (100, 400)
grids = {}
for x in X:
    for y in Y:
        line = last.replace('X', str(x))
        line = line.replace('Y', str(y))
        
        params = line2params(line)

        pg = np.zeros(shape=shape)
        pg[-1, :] = 1
         
        grids[(x,y)] = DLA(pg, particles=params['particles'], add_param=params['Add'],
                         fall_param=params['Move'], add_fall_ratio=params['ratio'],
                         offset=params['offset'])
        
        print("Done: ", x, y)



generate_image(grids, (len(X), len(Y)), names, last.replace('X', names[0]).replace('Y', names[1]))





    

