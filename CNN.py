import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
import numpy as np

# CNN
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
    
# FNN
class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)  # Input size: 28x28 pixels
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)  # Output size: 10 classes

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

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
            message = (f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
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
    accuracy = 100. * correct / len(test_loader.dataset)

    message = (f'\nTest set: Average loss: {test_loss:.4f}, '
               f'Accuracy: {correct}/{len(test_loader.dataset)} '
               f'({accuracy:.0f}%)\n')
    print(message)
    output_lines.append(message)


def Run_Model(model_type="CNN", epochs=5, lr=1.0, percent=1.0):
    output_lines = []
    torch.manual_seed(1)

    device = torch.device('mps' if torch.mps.is_available() else 'cpu')

    train_kwargs = {'batch_size': 64}
    test_kwargs = {'batch_size': 1000}

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)

    if percent < 1.0:
        num_train = int(len(dataset1) * percent)
        indices = np.random.choice(len(dataset1), num_train, replace=False)
        subset = Subset(dataset1, indices)
        train_loader = torch.utils.data.DataLoader(subset, **train_kwargs)
        output_lines.append(f'Using {num_train} training samples out of {len(dataset1)}')
    else:
        train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        output_lines.append(f'Using all {len(dataset1)} training samples')

    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Initialize model
    if model_type == "CNN":
        model = NeuralNet().to(device)
    elif model_type == "FNN":
        model = FNN().to(device)
    else:
        raise ValueError("Invalid model_type. Choose 'CNN' or 'FNN'.")

    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, output_lines)
        test(model, device, test_loader, output_lines)
        scheduler.step()

    torch.save(model.state_dict(), f"mnist_{model_type.lower()}.pt")
    output_lines.append(f"Model saved as mnist_{model_type.lower()}.pt")

    return output_lines