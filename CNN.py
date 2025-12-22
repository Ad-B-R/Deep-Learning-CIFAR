import cifar_import as cifar
import matplotlib.pyplot as plt
import torch 
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import copy

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")

X_train_ = cifar.X_tensor.to(device)
X_test_ = cifar.X_test.to(device)

Y_train = cifar.Y_tensor.to(device)
Y_test = cifar.Y_test.to(device)

train_dataset = TensorDataset(X_train_/255, Y_train)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)