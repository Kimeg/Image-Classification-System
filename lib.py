from static import *
import numpy as np
import itertools
import random
import math

def calc_rmsd(u, v):
    return np.sqrt(np.sum([(i-j)**2 for i, j in zip(u, v)]))

def calc_r2(true, pred):
    y_mean = np.mean(true)

    ss_tot = np.sum([(v-y_mean)**2 for v in true])
    ss_res = np.sum([(p-y)**2 for p,y in zip(pred, true)])
    return 1 - (ss_res/ss_tot)

def isValidIndex(i, j):
    if (i<0 or i>=DIM or j<0 or j>=DIM):
        return False
    return True

''' Recursively apply color to neighboring cells within specified range, i.e. THICKNESS '''
def recursiveNeighborSearch(mat, i, j, depth):
    if depth==THICKNESS:
        return

    for _dir in DIRS:
        x = _dir[0]+i
        y = _dir[1]+j

        if not isValidIndex(x, y):
            continue

        mat[x][y] = 255
        recursiveNeighborSearch(mat, x, y, depth+1)
    return

def fill(mat):
    new = np.zeros((DIM, DIM))
    for i in range(DIM):
        for j in range(DIM):
            if mat[i][j]==0:
                continue
            
            recursiveNeighborSearch(new, i, j, 0)
    return new

def getRotationMatrix(angle):
    return np.array([[math.cos(angle), math.sin(angle)],[-math.sin(angle), math.cos(angle)]])

def rotate(mat):
    new = np.zeros((DIM, DIM))

    indices = np.array([[p[0]-half, p[1]-half] for p in itertools.product(range(DIM), repeat=2) if mat[p]==255])
    
    angle = degToRad(random.choice(ANGLES)*random.choice(SIGNS))

    rotated_indices = [[int(i[0]+half), int(i[1]+half)] for i in np.dot(indices, getRotationMatrix(angle))]

    for i, j in rotated_indices:
        new[i][j] = 255 
    return new

def scale(mat):
    x_scaler = random.choice(SCALERS)
    y_scaler = random.choice(SCALERS)
    
    new = np.zeros((DIM, DIM))
    for i in range(DIM):
        for j in range(DIM):
            if mat[i][j]==0:
                continue

            x = int((i-half*x_scaler)+half)
            y = int((j-half*y_scaler)+half)

            #x = int(i*x_scaler)
            #y = int(j*y_scaler)

            new[x][y] = 255
    return new

def translate(mat):
    x_offset = np.random.randint(0, TRANSLATION)*random.choice(SIGNS)
    y_offset = np.random.randint(0, TRANSLATION)*random.choice(SIGNS)

    new = np.zeros((DIM, DIM))
    #indices = {i+x_offset:j+y_offset for i, j in itertools.permutations(range(DIM), 2) if mat[i][j]==255}
    #new = np.zeros((DIM, DIM))
    #new[list(indices.keys()), list(indices.values())] = 255

    for i in range(DIM):
        for j in range(DIM):
            if mat[i][j]==0:
                continue
            new[i+x_offset][j+y_offset] = 255
    return new

def degToRad(angle):
    return angle*math.pi/180

''' Generate matrix of circular data '''
def createC():
    mat = np.zeros((DIM, DIM))

    angle = 0

    for i in range(360):
        rad = degToRad(angle)

        x = int(math.cos(rad)*RADIUS + half)
        y = int(math.sin(rad)*RADIUS + half)
        
        mat[x][y] = 255

        angle += 1

    return mat

''' Generate matrix of triangular data '''
def createT():
    mat = np.zeros((DIM, DIM))
    
    n = len(TRIANGLE_CORNERS)
    for i in range(n):
        is_last = i==n-1
        to_index = 0 if is_last else i+1

        _from = TRIANGLE_CORNERS[i]
        _to = TRIANGLE_CORNERS[to_index]

        dx = _to[0]-_from[0]
        dy = _to[1]-_from[1]
    
        xs = 1 if dx>0 else -1
        ys = 1 if dy>0 else -1
          
        xstep = abs(dx)/STEP_SIZE
        ystep = abs(dy)/STEP_SIZE

        mat[TRIANGLE_CORNERS[i][0], TRIANGLE_CORNERS[i][1]] = 255
        mat[[TRIANGLE_CORNERS[i][0]+xs*int(j*xstep) for j in range(STEP_SIZE)], [TRIANGLE_CORNERS[i][1]+ys*int(j*ystep) for j in range(STEP_SIZE)]] = 255

    return mat

''' Generate matrix of rectangular data '''
def createS():
    mat = np.zeros((DIM, DIM))
    
    n = len(SQUARE_CORNERS)
    for i in range(n):
        is_last = i==n-1
        to_index = 0 if is_last else i+1

        _from = SQUARE_CORNERS[i]
        _to = SQUARE_CORNERS[to_index]

        dx = _to[0]-_from[0]
        dy = _to[1]-_from[1]
    
        xs = 1 if dx>0 else -1
        ys = 1 if dy>0 else -1

        mat[SQUARE_CORNERS[i][0], SQUARE_CORNERS[i][1]] = 255

        mat[[SQUARE_CORNERS[i][0]+xs*j for j in range(abs(dx))], SQUARE_CORNERS[i][1]] = 255
        mat[SQUARE_CORNERS[i][0], [SQUARE_CORNERS[i][1]+ys*j for j in range(abs(dy))]] = 255
        
        #for j in range(abs(dx)):
        #    mat[SQUARE_CORNERS[i][0]+xs*j, SQUARE_CORNERS[i][1]] = 255

        #for j in range(abs(dy)):
        #    mat[SQUARE_CORNERS[i][0], SQUARE_CORNERS[i][1]+ys*j] = 255

    return mat
