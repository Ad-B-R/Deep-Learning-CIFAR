import cifar_import as cifar
import matplotlib.pyplot as plt
import torch 
import torch.optim as optim
from torch import nn
import os
from torch.utils.data import DataLoader, TensorDataset

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")

X_train_ = cifar.X_tensor.reshape(cifar.X_tensor.shape[0], -1).to(device)
X_test_ = cifar.X_test.reshape(cifar.X_test.shape[0], -1).to(device)

Y_train = cifar.Y_tensor.to(device)
Y_test = cifar.Y_test.to(device)

model = nn.Sequential(
    nn.Linear(3072,1024),
    nn.ReLU(),
    nn.Linear(1024,256),
    nn.ReLU(),
    nn.Linear(256,10)
).to(device)

train_dataset = TensorDataset(X_train_, Y_train)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
X_epoch = [1,5,10,20,50,100]
Y_acc = []

for epoch_val in X_epoch:
    loss_opti = nn.CrossEntropyLoss()
    SGD_opti = optim.SGD(model.parameters())
    correct = torch.tensor(0.0, device=device)
    
    batch_size = 100

    for _ in range(epoch_val):
        model.train()

        for x,y in train_loader:

            SGD_opti.zero_grad()
            y_train = model(x)
            loss = loss_opti(y_train,y)

            loss.backward()
            SGD_opti.step()

    model.eval()
    meta = cifar.unpickle(os.path.join('cifar-10-batches-py','batches.meta'))

    for index in range(X_test_.size(0)):
        y_pred_index = torch.argmax(model(X_test_[index]))
        y_target_index = cifar.Y_test[index]
        correct+= (meta[b'label_names'][y_pred_index] == 
                meta[b'label_names'][y_target_index])
    acc = correct.item()/X_test_.size(0)*100
    print(acc)
    Y_acc.append(acc)

plt.figure(figsize=(10, 5))          
plt.plot(X_epoch, Y_acc, marker='o', 
        linestyle='-', color='b', label='Training Accuracy')


plt.title('Training Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)            
plt.legend()                        
plt.show()