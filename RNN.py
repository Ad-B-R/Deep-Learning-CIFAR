import cifar_import as cifar
import matplotlib.pyplot as plt
import torch 
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import copy

class RNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layer: int, 
                batch_size: int, device: torch.device):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.device = device
        self.num_layer = num_layer
        self.batch_size = batch_size
        self.LSTM = nn.LSTM(hidden_size=hidden_size, input_size=input_size,
                            num_layers=self.num_layer, batch_first=True, 
                            device=self.device)
        
        self.fc = nn.Linear(hidden_size, 10)
    
    def forward(self, x: torch.Tensor):
        x = x.permute(0,2,3,1)
        x = x.reshape(x.size(0), -1, self.input_size)
        h0 = torch.zeros(self.num_layer, x.size(0), 
                        self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layer, x.size(0), 
                        self.hidden_size).to(self.device)
        out, (h0, c0) = self.LSTM(x, (h0,c0))
        return self.fc(out[:,-1,:])
    
class CIFARRNN():
    def __init__(self, model: RNN, device: torch.device,
                X_train_: torch.Tensor, Y_train: torch.Tensor, 
                X_test_: torch.Tensor, Y_test: torch.Tensor, 
                X_epoch: list):
        self.model = model
        self.X_train = X_train_
        self.Y_train = Y_train
        self.X_test = X_test_
        self.Y_test = Y_test
        self.epoch_list = X_epoch
        self.device = device
        self.accuracy = []
        self.initial_state = copy.deepcopy(self.model.state_dict())
        self.train_dataset = TensorDataset(X_train_, Y_train)
        self.train_loader = DataLoader(self.train_dataset, batch_size=100, 
                                    shuffle=True)

    def reset_state(self):
        self.model.load_state_dict(self.initial_state)
    
    def train_for_one_epoch(self, epoch:int):
        correct = torch.tensor(0.0, device=self.device)
        loss_opti = nn.CrossEntropyLoss()
        SGD_opti = optim.SGD(self.model.parameters())
        for _ in range(epoch):
            for x, y in self.train_loader:
                SGD_opti.zero_grad()
                y_pred = self.model.forward(x)
                Loss = loss_opti(y_pred, y)
                Loss.backward()
                SGD_opti.step()
        self.model.eval()

        for i in range(self.X_test.size(0)):
            target = self.Y_test[i]
            pred = torch.argmax(self.model.forward(self.X_test[i].unsqueeze(0)), dim = 1)
            correct += (target==pred).sum()
        acc = (correct/self.X_test.size(0)*100).item()
        
        self.accuracy.append(acc)

        print(f"epoches: {epoch}, accuracy: {acc}%")

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

X_tensor = cifar.X_tensor.to(device)
Y_tensor = cifar.Y_tensor.to(device)
X_test = cifar.X_test.to(device)
Y_test = cifar.Y_test.to(device)

X_epoch = [1,5,10,20,25,30,40,45,50]

RNN_model = RNN(256, 128, 2, 100, device)
cifar_rnn = CIFARRNN(RNN_model, device, X_tensor, Y_tensor, X_test, Y_test, X_epoch)
cifar_rnn.epoch_loop()
cifar_rnn.plot_figure()