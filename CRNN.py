import torch
import torch.nn as nn
import copy
import cifar_import as cifar
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class CRNN(nn.Module):
    def __init__(self, device: torch.device, hidden_size: int, 
                batch_size: int, num_layer: int):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size=3, stride=1, padding=1) # 32x32
        self.batch1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2) # 16x16
        
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1, padding=1) # 16x16 
        self.batch2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2) # 8x8
        
        self.conv3 = nn.Conv2d(64,128,kernel_size=3, stride = 1,padding=1) # 8x8
        self.batch3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2) # 4x4

        self.drop = nn.Dropout2d()
        self.RELU = nn.ReLU()
        self.fc1 = nn.Linear(128*4*4,10)
        self.batch_size = batch_size
        self.device = device 
        self.num_layer = num_layer
        self.hidden_size = hidden_size

        self.input_size_per_time = 512   
        self.sequential_length = 4
        self.LSTM = nn.LSTM(hidden_size=hidden_size, 
                        input_size=self.input_size_per_time, 
                        num_layers=num_layer, batch_first=True, device=device)
        self.fc1 = nn.Linear(hidden_size, 10)
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.RELU(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.RELU(x)
        x = self.pool3(x)

        x = x.permute(0,2,3,1)
        x = x.reshape(x.size(0), self.sequential_length, self.input_size_per_time)
        c0 = torch.zeros(self.num_layer, x.size(0), self.hidden_size)
        h0 = torch.zeros(self.num_layer, x.size(0), self.hidden_size)
        x, (h0,c0) = self.LSTM(x, (h0, c0))

        x = x[:,-1,:]
        return self.fc1(x)
    
class CIFARCRNN():
    def __init__(self, model: CRNN, 
                X_train, Y_train: torch.Tensor, X_test: torch.Tensor, Y_test:torch.Tensor, 
                batch_size: int, X_epoch: list):
        self.dataset = TensorDataset(X_train, Y_train.long()) 
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.model = model
        self.intial_state = copy.deepcopy(self.model.state_dict())
        self.epoch_list = X_epoch
        self.accuracy = []
        self.X_test = X_test
        self.Y_test = Y_test
    
    def reset_state(self):
        self.model.load_state_dict(self.intial_state)
    
    def epoch_loop(self):
        for epoch in self.epoch_list:
            self.reset_state()
            self.train_for_one_epoch(epoch)

    def train_for_one_epoch(self, epoch):
        correct = torch.tensor(0.0)
        loss_opti = nn.CrossEntropyLoss()
        SGD_opti = optim.SGD(self.model.parameters())
        for _ in range(epoch):
            for x, y in self.dataloader:
                SGD_opti.zero_grad()
                y_pred = self.model.forward(x)
                loss = loss_opti(y_pred, y)
                loss.backward()
                SGD_opti.step()
        self.model.eval()
        for i in range(self.X_test.size(0)):
            target = self.Y_test[i]
            y_pred = torch.argmax(self.model.forward(
                self.X_test[i].unsqueeze(0)), dim=1)
            correct += (y_pred==target).sum()
        acc = correct/self.X_test.size(0)*100
        self.accuracy.append(acc)
        print(acc.item())

    def plot_figure(self):
        plt.figure(figsize=(10, 5))          
        plt.plot(self.epoch_list, self.accuracy, marker='o', 
        linestyle='-', color='b', label='Training Accuracy')

        plt.title('Training Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True)            
        plt.legend()                        
        plt.show()

device = torch.device("cpu")

X_train = cifar.X_tensor  
Y_train = cifar.Y_tensor  
X_test = cifar.X_test  
Y_test = cifar.Y_test

CRNN_object = CRNN(device,64,100,2)
X_epoch = [1,5,10,20,25,30,40,45,50]

Cifar_obj = CIFARCRNN(CRNN_object, X_train, Y_train, X_test,
                      Y_test, 100, X_epoch)
Cifar_obj.epoch_loop()
Cifar_obj.plot_figure()
