import cifar_import as cifar
import matplotlib.pyplot as plt
import torch 
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import copy

class SimpleMLP(nn.Module):
    def __init__(self, device: torch.device, input=3072, hidden1= 1024, 
                hidden2 = 256, output = 10):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
                nn.Linear(input,hidden1),
                nn.ReLU(),
                nn.Linear(hidden1,hidden2),
                nn.ReLU(),
                nn.Linear(hidden2,output)
        ).to(device)

    def forward_compute(self, x: torch.Tensor):
        return self.network(x)

class CIFAR_NN():
    def __init__(self, model: SimpleMLP, X_train_: torch.Tensor, Y_train: torch.Tensor, 
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
        self.initial_state = copy.deepcopy(self.model.network.state_dict())
    def reset_state(self):
        self.model.network.load_state_dict(self.initial_state)

    def train_for_one_epoch(self, epoch: int):
        correct = torch.tensor(0.0, device=self.device)
        loss_opti = nn.CrossEntropyLoss()
        SGD_opti = optim.SGD(self.model.network.parameters())
        # print(self.model.network.state_dict())

        for _ in range(epoch):
            self.model.network.train()

            for x,y in self.train_loader:
                SGD_opti.zero_grad()
                y_pred = self.model.forward_compute(x)
                Loss = loss_opti(y_pred,y)

                Loss.backward()
                SGD_opti.step()
        self.model.network.eval()

        for i in range(self.X_test_.size(0)):
            target = Y_test[i]
            y_pred = torch.argmax(self.model.forward_compute(X_test_[i]), dim=1)
            correct+= (target==y_pred)

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
    "cuda" if torch.cuda.is_available() else "cpu")

X_train_ = cifar.X_tensor.reshape(cifar.X_tensor.shape[0], -1).to(device)
X_test_ = cifar.X_test.reshape(cifar.X_test.shape[0], -1).to(device)

Y_train = cifar.Y_tensor.to(device)
Y_test = cifar.Y_test.to(device)

X_epoch = [1,5,10,20,25,30,40,45,50]
CIFAR_model = SimpleMLP(device)

ANN = CIFAR_NN(CIFAR_model, X_train_, Y_train, X_test_, Y_test, X_epoch, device)
ANN.epoch_loop()
ANN.plot_figure()
