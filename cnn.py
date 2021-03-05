import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from utils import *

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten,self).__init__()

    def forward(self,input):
        return input.view(input.size(0),-1)

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.flatten = Flatten()
        # TODO initialize model layers here
        self.model = nn.Sequential(nn.Conv2d(1,32,3),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d((2,2)),
                                   nn.Linear(32*13*13,10))

    def forward(self, x):


        # TODO use model layers to predict the two digits
        xf = self.model(x)
        return xf

def compute_accuracy(predictions, y):
    """Computes the accuracy of predictions against the gold labels, y."""
    return np.mean(np.equal(predictions.numpy(), y.numpy()))

def batchify_data(x,y,batch_size):
    N = int(x.shape[0]/batch_size)
    batched_x = []
    batched_y = []
    for i in range(N):
        batched_x.append(x[i*batch_size:(i+1)*batch_size])
        batched_y.append(y[i*batch_size:(i+1)*batch_size])
    return np.array(batched_x),np.array(batched_y)
# # Convert label from 1D to 10D
# def convert(y):
#     label = torch.zeros(y.shape[0],10)
#     for i in range(y.shape[0]):
#         num = y[i]
#         label[i][num] = 1
#     return label

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model.parameters(),lr = 0.01, momentum=0.9)

# Load MNIST data
train_x, train_y, test_x, test_y = get_MNIST_data()
batch_size = 5
train_x,train_y = batchify_data(train_x,train_y,batch_size)
test_x,test_y = batchify_data(test_x,test_y,batch_size)

# Reshape to 2D image
train_x = np.reshape(train_x,(train_x.shape[0],1,28,28))
train_y = np.reshape(train_y,(train_y.shape[0],1,28,28))
test_x = np.reshape(test_x,(test_x.shape[0],1,28,28))
test_y = np.reshape(test_y,(test_y.shape[0],1,28,28))

train_xt = torch.from_numpy(train_x)
train_yt = torch.from_numpy(train_y)
test_xt = torch.from_numpy(test_x)
test_yt = torch.from_numpy(test_y)

# Train model
n_epochs = 10
for epoch in range(n_epochs):
    train_loss = []
    batch_accuracies = []
    for i in range(train_y.shape[0]):
        optimizer.zero_grad()
        output = model(train_xt[i])
        loss = criterion(output,train_yt[i])
        loss.backward()
        optimizer.step()

        # Predict and store accuracy
        predictions = torch.argmax(output,dim=1)
        batch_accuracies.append(compute_accuracy(predictions,train_yt[i]))
        # Compute loss
        train_loss.append(loss.item())

    print('avg_loss{}: {:.6f}'.format(epoch+1,np.mean(train_loss)))
    print('avg_accuracy{}: {:.6f}'.format(epoch+1,np.mean(batch_accuracies)))

# Test the model
model.eval()
test_loss = []
batch_accuracies = []
for i in range(test_yt.shape[0]):
    output = model(test_xt[i])
    loss = criterion(output,test_yt[i])
    # Predict and store accuracy
    predictions = torch.argmax(output,dim=1)
    batch_accuracies.append(compute_accuracy(predictions,test_yt[i]))
    # Compute loss
    test_loss.append(loss.item())

print('test_loss: {:.6f}'.format(np.mean(test_loss)))
print('test_accuracy: {:.6f}'.format(np.mean(batch_accuracies)))