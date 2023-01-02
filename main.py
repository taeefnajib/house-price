"""
Original code is taken from:
https://github.com/Nanfengzhijia/Pytorch-boston-house-price/blob/master/homework-house-pytorch.py
This tutorial uses load_boston method with is removed from sklearn recently because of a legal issue. 
So we used a csv file downloaded from here: https://www.kaggle.com/datasets/arunjathari/bostonhousepricedata
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from dataclasses_json import dataclass_json
from dataclasses import dataclass

@dataclass_json
@dataclass
class Hyperparameters(object):
    filepath = "data.csv"
    test_size: float = 0.25
    lr: float = 0.001
    max_epoch: int = 10000

# Instantiating Hyperparameters() class
hp = Hyperparameters()

# Creating dataset
def create_dataset(filepath):
    boston = pd.read_csv(filepath)
    x = boston.drop(["MEDV"], axis=1)
    y = boston["MEDV"]
    return x, y

# Normalizing features
def normalize_feature(x):
    ss_input = MinMaxScaler()
    x = ss_input.fit_transform(x)
    return x 

# Turning x and y into tensors
def create_tensors(x, y):
    feat = torch.from_numpy(np.asarray(x)).type(torch.FloatTensor)
    tar = torch.from_numpy(np.asarray(y)).type(torch.FloatTensor)
    return feat, tar

# Splitting train and test dataset
def split_dataset(x, y, test_size):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size)
    return train_x, test_x, train_y, test_y

# Creating the model
model = nn.Sequential(
    nn.Linear(13,16),
    nn.ReLU(),
    nn.Linear(16,13),
    nn.ReLU(),
    nn.Linear(13,1)
)
# Specifying loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=hp.lr)

# Training the model on train dataset
def train_model(max_epoch, model, train_x, train_y):
    max_epoch = max_epoch
    iter_loss = []
    for i in range(max_epoch):
    # Predicting on train_x
        y_pred = model(train_x)
    # Creating loss and adding to list iter_loss
        loss = criterion(y_pred, train_y)
        iter_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%(max_epoch/10)==1:
            print(f"Epoch: {i} - Loss: {loss}")
    # return model

# Predicting on the test dataset
def predict(model, test_x):
    output = model(test_x)
    return output.detach().numpy()

# Running workflow
def run_wf(filepath, test_size, max_epoch, model):
    x, y = create_dataset(filepath)
    x = normalize_feature(x)
    feat, tar = create_tensors(x, y)
    train_x, test_x, train_y, test_y = split_dataset(feat, tar, test_size=test_size)
    train_model(max_epoch, model, train_x, train_y)
    y_pred = predict(model, test_x)
    y_pred = torch.from_numpy(np.asarray(y_pred)).type(torch.FloatTensor)
    test_y = test_y.view(127,1)
    pred_loss = criterion(y_pred, test_y)
    print(f"Prediction Loss: {pred_loss}")
    return y_pred

if __name__ == "__main__":
    run_wf(hp.filepath, hp.test_size, hp.max_epoch, model=model)


