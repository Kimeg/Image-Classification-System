from keras.preprocessing.image import img_to_array, load_img, array_to_img
from torch.utils.data import TensorDataset, DataLoader, Dataset
from collections import defaultdict
from sklearn import preprocessing
from static import *
from lib import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import os

''' Customized CNN class for pytorch model training '''
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = 3,
            out_channels = 32,
            kernel_size = 3,
            padding = 'same'  
        )
        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = 3,
            padding = 'same'  
        )
        self.conv3 = nn.Conv2d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = 3,
            padding = 'same'  
        )
        self.fc1 = nn.Linear(in_features=50*50*128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=3)
        return

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        #x = x.view(-1, 128)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = nn.Dropout(p=0.5)(x)
        x = nn.BatchNorm1d(num_features=64)(x)

        x = F.relu(self.fc2(x))
        x = nn.Dropout(p=0.5)(x)
        x = nn.BatchNorm1d(num_features=16)(x)

        x = F.log_softmax(self.fc3(x))
        return x

class CustomData(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        return
    
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return (x, y)

    def __repr__(self):
        return str(self.x_data[:10])

''' CNN structure defined by keras API '''
def create_model(shape):
    #strategy = tf.distribute.MirroredStrategy()

    #with strategy.scope():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, 3, 1, input_shape=shape[1:], padding="same"))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Conv2D(64, 3, 1, padding="same"))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Conv2D(128, 3, 1, padding="same"))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.01), ))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dropout(rate=0))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(50, kernel_regularizer=tf.keras.regularizers.l2(0.01), ))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dropout(rate=0))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(shape[-1], kernel_regularizer=tf.keras.regularizers.l2(0.01), ))
    model.add(tf.keras.layers.Activation("softmax"))

    #sgd = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-6, clipvalue=1)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model

''' General method for preprocessing image data '''
def preprocess_data(output_dir):
    dataset = defaultdict(list)

    for data in ['train', 'valid', 'test']:

        images = glob.glob(f'{output_dir}/{data}/*.png')

        for i, _file in enumerate(images):
            #print(i, data, _file)

            img = load_img(_file)
            img_arr = img_to_array(img)
            label = LABEL_MAP[_file.split('/')[-1].split('_')[0]]

            dataset[f'X_{data}'].append(img_arr)
            dataset[f'Y_{data}'].append(label)

    return dataset  

''' Tensorflow pipeline '''
def train_tf(output_dir):
    dataset = preprocess_data(output_dir)

    X_train = np.array(dataset['X_train'])
    X_valid = np.array(dataset['X_valid'])
    X_test =  np.array(dataset['X_test'])

    Y_train = np.array(dataset['Y_train'])
    Y_valid = np.array(dataset['Y_valid'])
    Y_test =  np.array(dataset['Y_test'])

    scaler = preprocessing.StandardScaler()

    X_train_std = X_train/255. 
    X_valid_std = X_valid/255. 

    stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, verbose=1, mode='auto')
    logger = tf.keras.callbacks.CSVLogger('./log', separator=',', append=False)

    if not os.path.isfile(TF_MODEL_FILE):
        bestmodel = tf.keras.callbacks.ModelCheckpoint(filepath=TF_MODEL_FILE, verbose=1, save_best_only=True)

        model = create_model(X_train_std.shape)

        history = model.fit(
            X_train_std, Y_train,
            validation_data = (X_valid_std, Y_valid),
            epochs = 20,
            batch_size = 64,
            verbose = 1,
            callbacks = [stop, logger, bestmodel]
        )
    else:
        model = tf.keras.models.load_model(TF_MODEL_FILE)

    X_test_std = X_test/255.

    #scores = model.evaluate(X_test_std, Y_test)
    predict_tf(model, X_test_std, Y_test)

    #pd.DataFrame(pred).to_csv(TF_PRED_FILE, float_format="%.3f", index=False)
    return

def predict_tf(model, X_test_std, Y_test):
    ''' Deprecated since ver 2.6 '''
    #pred = model.predict_classes(X_test_std) #.ravel()

    pred = model.predict(X_test_std) 
    pred = np.argmax(pred, axis=1)
    
    print(calc_rmsd(Y_test, pred))
    print(calc_r2(Y_test, pred))
    return

''' Pytorch pipeline '''
def create_model_torch(DEVICE):
    model = CNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    epochs = EPOCHS
    return (model, optimizer, criterion, epochs)

def reformat_mat(arr):
    total = []
    for img in arr:
        t1 = defaultdict(list)
        for row in img:
            t2 = defaultdict(list)
            for col in row:
                t2['r'].append(col[0])
                t2['g'].append(col[1])
                t2['b'].append(col[2])
            t1['r'].append(t2['r'])
            t1['g'].append(t2['g'])
            t1['b'].append(t2['b'])
        total.append([t1['r'], t1['g'], t1['b']])
    return np.array(total)

def use_torch(DEVICE, output_dir):
    dataset = preprocess_data(output_dir)

    if not os.path.isfile(TORCH_MODEL_FILE):
        model, optimizer, criterion, epochs = create_model_torch(DEVICE)

        dataset['Y_train'] = np.array([[v] for v in dataset['Y_train']])
        dataset['Y_valid'] = np.array([[v] for v in dataset['Y_valid']])

        dataset['X_train'] = reformat_mat(dataset['X_train'])
        dataset['X_valid'] = reformat_mat(dataset['X_valid'])

        train_loader = DataLoader(CustomData(dataset['X_train'], dataset['Y_train']), batch_size=32, shuffle=True)
        valid_loader = DataLoader(CustomData(dataset['X_valid'], dataset['Y_valid']), batch_size=32, shuffle=True)

        for epoch in range(1, epochs+1):
            train_torch(model, train_loader, criterion, optimizer, epoch, DEVICE)
            valid_loss, valid_acc = evaluate_torch(model, valid_loader, criterion, DEVICE)
            print(f"Valid Epoch : {epoch}\nValid Loss : (valid_loss)\nValid Acc : {valid_acc}")

        torch.save(model.state_dict(), TORCH_MODEL_FILE)
    else:
        dataset['Y_test'] =  np.array([[v] for v in dataset['Y_test']])
        dataset['X_test'] =  reformat_mat(dataset['X_test'])

        test_loader =  DataLoader(CustomData(dataset['X_test'],  dataset['Y_test']), batch_size=dataset['X_test'].shape[0])

        model = CNN() 
        model.load_state_dict(torch.load(TORCH_MODEL_FILE))
        model.eval()

        pred, correct = predict_torch(model, test_loader, DEVICE)

        pred = pred.numpy()
        print(sum([i==j for i,j in zip(pred, dataset['Y_test'])])/float(len(pred)))
    return

def train_torch(*args):
    model, train_loader, criterion, optimizer, epoch, DEVICE = args
    model.train()

    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()
        output = model(image)

        label = torch.tensor(label, dtype=torch.long)

        loss = criterion(output, label[:,0])

        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print(f"Train Epoch : {epoch}\nTrain batch idx : (batch_idx)\nTrain Loss : {loss.item()}")
    return

def evaluate_torch(*args):
    model, valid_loader, criterion, DEVICE = args
    model.eval()

    valid_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in valid_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)

            output = model(image)

            label = torch.tensor(label, dtype=torch.long)

            valid_loss += criterion(output, label[:,0]).item()

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item() 

    valid_loss /= len(valid_loader.dataset)
    valid_acc = 100. * correct / len(valid_loader.dataset)
    return valid_loss, valid_acc

def predict_torch(model, test_loader, DEVICE):
    correct = 0
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(test_loader):
            #image = image.to(DEVICE)
            #label = label.to(DEVICE)

            output = model(image)

            label = torch.tensor(label, dtype=torch.long)

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item() 

    return (pred, correct)
