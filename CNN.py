import cifar_import as cifar
import matplotlib.pyplot as plt
import torch 
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import copy

class CNN(nn.Module):
    def __init__(self, device: torch.device):
        super(CNN, self).__init__()
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
        self.fc1 = nn.Linear(128*4*4,10)
        self.RELU = nn.ReLU()

    def forward_compute(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.RELU(x)
        x = self.batch1(x)
        x = self.pool1(x)

        x = self.drop(x)

        x = self.conv2(x)
        x = self.RELU(x)
        x = self.batch2(x)
        x = self.pool2(x)

        x = self.drop(x)

        x = self.conv3(x)
        x = self.RELU(x)
        x = self.batch3(x)
        x = self.pool3(x)

        x = torch.flatten(x,1)
        x = self.fc1(x)
        return x

class CIFAR_NN():
    def __init__(self, model: CNN, X_train_: torch.Tensor, Y_train: torch.Tensor, 
                 X_test_: torch.Tensor, Y_test: torch.Tensor, 
                 X_epoch: list, device: torch.device):
        self.train_dataset = TensorDataset(X_train_, Y_train)
        self.train_loader = DataLoader(self.train_dataset, batch_size=100, 
                                    shuffle=True)
        self.X_test_ = X_test_
        self.Y_test = Y_test
        self.epoch_list = X_epoch
        self.model = model
        self.accuracy = []
        self.device = device
        # self.meta = cifar.unpickle(os.path.join('cifar-10-batches-py','batches.meta'))
        self.initial_state = copy.deepcopy(self.model.state_dict())
    def reset_state(self):
        self.model.load_state_dict(self.initial_state)

    def train_for_one_epoch(self, epoch: int):
        correct = torch.tensor(0.0, device=self.device)
        loss_opti = nn.CrossEntropyLoss()
        SGD_opti = optim.SGD(self.model.parameters())
        # print(self.model.network.state_dict())

        for _ in range(epoch):
            self.model.train()

            for x,y in self.train_loader:
                SGD_opti.zero_grad()
                y_pred = self.model.forward_compute(x)
                Loss = loss_opti(y_pred,y)

                Loss.backward()
                SGD_opti.step()
        self.model.eval()
        for i in range(self.X_test_.size(0)):
            target = Y_test[i]
            y_pred = torch.argmax(self.model.forward_compute(
                X_test_[i].unsqueeze(0)), dim=1)
            correct+= (target==y_pred).sum()

        acc = correct.item()/X_test_.size(0)*100

        print(f"epoches: {epoch}, accuracy: {acc}%")

        self.accuracy.append(acc)
    
    def epoch_loop(self):
        for epoch in self.epoch_list:
            self.reset_state()
            self.train_for_one_epoch(epoch)
    
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


device = torch.device(
    # "cuda" if torch.cuda.is_available() else 
    "cpu")

X_train_ = cifar.X_tensor.to(device)
X_test_ = cifar.X_test.to(device)

Y_train = cifar.Y_tensor.to(device)
Y_test = cifar.Y_test.to(device)
  
X_epoch = [1,5,10,20,40,67]

CIFAR_model = CNN(device)

ANN = CIFAR_NN(CIFAR_model, X_train_, Y_train, X_test_, Y_test, X_epoch, device)
ANN.epoch_loop()
ANN.plot_figure()
