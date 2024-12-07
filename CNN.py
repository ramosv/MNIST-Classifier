import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
import numpy as np

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Fully Connected Layer
        self.fc1 = nn.Linear(256 * 1 * 1, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, epoch, output_lines):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            message =(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            print(message)
            output_lines.append(message)
def test(model, device, test_loader, output_lines):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() 
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    message = (f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')
    
    print(message)
    output_lines.append(message)

def Run_CNN(epochs=14, lr=1.0, percent=1.0):    
    output_lines = []
    # setting a manual seed for reproducibility
    torch.manual_seed(1)

    #Testing on mac: well use mps otherwise uncomment cuda or cpu
    device = torch.device('mps')
    #device = torch.device('cuda')
    #device = torch.device('cpu')

    #Take the tain and test argscumetns
    train_kwargs = {'batch_size': 64}
    test_kwargs = {'batch_size': 1000}

    # Transform from torch vision library uses Compose to chain multiple transforms together
    # ToTensor() converts the image to a tensor: a tensor is just a multi-dimensional array
    # Normalize() normalizes the image by the mean and standard deviation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the MNIST dataset which is already part of torch vision
    # dataset1 is the training dataset and dataset2 is the test dataset
    # we only need to download dataset1 as dataset2 is already available
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    
    # Adjust the training dataset based on percent argument
    # We can use the Subset class to create a subset of the dataset
    # We can then use the DataLoader class to load the subset
    if percent < 1.0:
        # setting the number of training samples based on the percent argument
        num_train = int(len(dataset1) * percent)
        indices = np.random.choice(len(dataset1), num_train, replace=False)
        subset = Subset(dataset1, indices)
        train_loader = torch.utils.data.DataLoader(subset, **train_kwargs)
        message = (f'Using {num_train} training samples out of {len(dataset1)}')
        print(message)
        output_lines.append(message)
    else:
        train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        message= (f'Using all {len(dataset1)} training samples')
        output_lines.append(message)

    # regardless of the size of the training dataset, we will use all the test samples
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    
    # Create the model and optimizer
    model = NeuralNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(1, epochs + 1):
        # train and test functions do not return anything
        # we capture the result in our shell script by combinng grep and the print
        train(model, device, train_loader, optimizer, epoch,output_lines)
        test(model, device, test_loader, output_lines)
        scheduler.step()

    torch.save(model.state_dict(), "mnist_cnn.pt")
    message = ("Model saved as mnist_cnn.pt")
    output_lines.append(message)

    return output_lines