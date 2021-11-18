DIM = 50
q1 = int(5*DIM/7)
q3 = DIM-q1
half = int(DIM/2)
CENTER = (half, half)
STEP_SIZE = 40
RADIUS = int(DIM/4)
TRANSLATION = int(DIM/12)
SCALERS = [val/10. for val in range(7, 13)]
THICKNESS = 1
SIGNS = [1, -1]
ANGLES = range(0, 10, 2)
nCPU = 10
LABEL_MAP = {'square': 0, 'triangle': 1, 'circle': 2}
EPOCHS = 30

TF_MODEL_FILE = '/lwork01/yjkim/tf.h5'
TORCH_MODEL_FILE = '/lwork01/yjkim/torch.h5'
TF_PRED_FILE = '/lwork01/yjkim/tf_pred.h5'
TORCH_PRED_FILE = '/lwork01/yjkim/torch_pred.h5'

SQUARE_CORNERS = [
    (q1, q1),
    (q3, q1),
    (q3, q3),
    (q1, q3)
]

TRIANGLE_CORNERS = [
    (q3, half),
    (q1, q1),
    (q1, q3),
]

DIRS = [
    (1, 0),
    (-1, 0),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
    (0, 1),
    (0, -1),
]

