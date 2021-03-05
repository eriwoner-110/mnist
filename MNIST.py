import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
from matplotlib import pyplot as plt

batch_size = 32
n_epochs = 10

# If cuda is available, use GPU
# device = torch.device('cuda:0')

# Load data
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
train_data = datasets.MNIST('../data',train=True,transform=transform)
test_data = datasets.MNIST('../data',train=False,transform=transform)

train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=True)

# print(train_data.data.shape)
# print(train_data.targets.shape)
# print(test_data.data.size())
# print(test_data.targets.size())

class Flatten(nn.Module):
    def forward(self,input):
        return input.view(input.size(0),-1)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.flatten = Flatten()
        self.model = nn.Sequential(nn.Conv2d(1,64,3,padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64,128,3,padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d((2,2)),
                                   nn.Conv2d(128,256,3,padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d((2,2)),
                                   self.flatten,
                                   nn.Linear(256*7*7,1024),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(0.2),
                                   nn.Linear(1024,256),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(0.2),
                                   nn.Linear(256,10),
                                   )

    def forward(self, x):
        x = self.model(x)
        return x

def plot_graph(loss,accuracy):
    """Plot the graph of loss and accuracy versus epochs"""
    n_epochs = len(loss)
    x_axis = np.linspace(1,n_epochs,n_epochs)
    fig, axes = plt.subplots(1,2)
    axes[0].plot(x_axis,loss,label='loss')
    axes[0].legend()
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[1].plot(x_axis,accuracy,label='accuracy')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('accuracy')
    axes[1].legend()
    plt.show()

def compute_accuracy(predictions, y):
    """Computes the accuracy of predictions against the gold labels, y."""
    return np.mean(np.equal(predictions.numpy(), y.numpy()))


def run_epoch(data, model, optimizer):
    """Train model for one pass of train data, and return loss, acccuracy"""
    # Gather losses
    losses = []
    batch_accuracies = []

    # If model is in train mode, use optimizer.
    is_training = model.training

    # Iterate through batches
    for batch in tqdm(data):
        # Grab x,y
        x,y = batch

        #use gpu
        # x = x.to(device)
        # y = y.to(device)

        # Get output prediction
        out = model(x)

        # Predict and store accuracy
        predictions = torch.argmax(out, dim=1)
        batch_accuracies.append(compute_accuracy(predictions, y))

        # Compute losses
        loss = F.cross_entropy(out, y)
        losses.append(loss.data.item())

        # If training, do an update.
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    avg_accuracy = np.mean(batch_accuracies)
    return avg_loss, avg_accuracy


def train_model(train_data, dev_data, model, lr=0.01, momentum=0.9, nesterov=False, n_epochs=10):
    """Train a model for N epochs given data and hyper-params."""
    # We optimize with SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)
    losses = []
    accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(1, n_epochs + 1):
        print("-------------\nEpoch {}:\n".format(epoch))

        # Run **training***
        loss, acc = run_epoch(train_data, model.train(), optimizer)
        print('Train | loss: {:.6f}  accuracy: {:.6f}'.format(loss, acc))
        losses.append(loss)
        accuracies.append(acc)

        # Run **validation**
        val_loss, val_acc = run_epoch(dev_data, model.eval(), optimizer)
        print('Valid | loss: {:.6f}  accuracy: {:.6f}'.format(val_loss, val_acc))
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Save model
        torch.save(model, 'mnist_model_fully_connected.pt')

    return losses,accuracies,val_losses,val_accuracies

def main():
    # Train the model
    model = CNN()
    # Use gpu
    # model = CNN().to(device)
    losses,accuracies,val_losses,val_accuracies = train_model(train_loader,test_loader,model)
    plot_graph(losses,accuracies)
    plot_graph(val_losses,val_accuracies)


if __name__=='__main__':
    # For reproductivity
    np.random.seed(12321)
    torch.manual_seed(12321)
    main()



