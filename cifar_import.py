import torch
import os
import pickle 
import numpy
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file,'rb') as f:
        unpickled_list = pickle.load(f, encoding='bytes')
    return unpickled_list

X_tensor = []
Y_tensor = []
for i in range(5):
    batch = unpickle(os.path.join('cifar-10-batches-py',f'data_batch_{i+1}'))

    X = torch.tensor(batch[b'data'], dtype=torch.float32)
    Y = torch.tensor(batch[b'labels'], dtype=torch.long)
    
    X_tensor.append(X)
    Y_tensor.append(Y)
    
X_tensor = torch.cat(X_tensor, dim=0).view(-1,3,32,32)
Y_tensor = torch.cat(Y_tensor, dim=0)

batch = unpickle(os.path.join('cifar-10-batches-py', 'test_batch'))
X_test = torch.tensor(batch[b'data'], dtype=torch.float32).view(-1,3,32,32)
Y_test = torch.tensor(batch[b'labels'], dtype=torch.long)

'''
print(X_tensor.shape)

img = X_tensor[0].permute(1,2,0)
img = img.numpy()/255.0

plt.imshow(img)
plt.axis("off")
plt.show()
'''