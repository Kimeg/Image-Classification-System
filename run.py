from static import *
from model import *
from lib import *
from PIL import Image

import multiprocessing as mp
import pandas as pd
import numpy as np
import itertools
import random
import torch
import glob
import math
import sys
import os

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

def parallel_proc(args):
    i, o = args

    shape, func = random.choice(shapes)

    obj = func()

    for task in tasks:
        obj = task(obj)

    img = Image.fromarray(np.uint8(obj)).convert('RGB') 
    img.save(os.path.join(o, f"{shape}_{i+1}.png"))
    return

def generate_data():
    #cvs = ['fill', 'scale', 'rotate', 'translate']
    #for cv in cvs:

    if os.path.isdir(output_dir):
        return

    for _set, nSample in zip(['train', 'valid', 'test'], [1000, 300, 200]):
        o = os.path.join(output_dir, _set)

        try: os.makedirs(o)
        except: pass
        
        pool = mp.Pool(nCPU)
        r = pool.map_async(parallel_proc, [(i, o) for i in range(nSample)])
        r.wait()

        pool.close()
        pool.join()
    return

def test(func, name):
    obj = func() 
    translated = translate(obj)
    scaled = scale(obj)
    rotated = rotate(obj)
    filled = fill(obj)

    img = Image.fromarray(np.uint8(obj)).convert('RGB') 
    img.save(f'./{name}.png')

    img = Image.fromarray(np.uint8(translated)).convert('RGB') 
    img.save(f'./translated_{name}.png')

    img = Image.fromarray(np.uint8(scaled)).convert('RGB') 
    img.save(f'./scaled_{name}.png')

    img = Image.fromarray(np.uint8(rotated)).convert('RGB') 
    img.save(f'./rotated_{name}.png')

    img = Image.fromarray(np.uint8(filled)).convert('RGB') 
    img.save(f'./filled_{name}.png')
    return

def main():
    ''' Generate image data, skips if data exists '''
    generate_data()

    print('Image Data Generation Complete.')

    ''' tf for tensorflow backend, else use pytorch '''
    if sys.argv[1] == 'tf':
        train_tf(output_dir)
    else:
        use_torch(DEVICE, output_dir)

    return

if __name__=="__main__":
    shapes = [('square', createS), ('triangle', createT), ('circle', createC)]
    tasks = [fill, scale, rotate, translate]
    output_dir = f'./image'
    main()
