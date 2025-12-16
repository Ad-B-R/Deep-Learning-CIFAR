import cifar_import as cifar
import matplotlib.pyplot as plt
import torch 
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_flatten = cifar.X_tensor.reshape(cifar.X_tensor.shape[0], -1).to(device)
X_test_flatten = cifar.X_test.reshape(cifar.X_test.shape[0], -1).to(device)
X_img = cifar.X_test[0]
Y_img = cifar.Y_test[0]

Y_train = cifar.Y_tensor.to(device)
Y_test = cifar.Y_test.to(device)
correct = torch.tensor(0.0, device=device)

for j in range(X_test_flatten.shape[0]):
    lowest = None
    '''
    for i,x in zip(range(X_train_flatten.shape[0]), X_train_flatten):
        dist = torch.sum(torch.abs(x-X_test_flatten[j])).item()
        if(dist<lowest_L1):
            lowest_L1 = dist
            lowest = Y_train[i]
    if(lowest.item()==Y_test[j].item()): correct_pred+=1
    '''
    dists = torch.sum(
    torch.abs(X_train_flatten - X_test_flatten[j]),
    dim=1
    )
    index = torch.argmin(dists)
    correct+=(Y_test[j]==Y_train[index]).float()
print(correct*100/X_test_flatten.shape[0])

meta = cifar.unpickle(os.path.join('cifar-10-batches-py','batches.meta'))
print(meta[b'label_names'][cifar.Y_test[0]])
img = cifar.X_test[0].permute(1,2,0)
img = img.numpy()/255.0
plt.imshow(img)
plt.axis("off")
plt.show()
