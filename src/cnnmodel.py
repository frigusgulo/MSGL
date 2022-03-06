import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print("CUDA is not available.  Training on CPU ...")
else:
    print("CUDA is available!  Training on GPU ...")


class Net(nn.Module):
    def __init__(self, inputsize: tuple):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(inputsize[0], inputsize[1], 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(
    args,
    model,
    criterion,
    device,
    train_loader,
    optimizer,
    epoch,
    num_epochs,
    verbose=True,
):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).unsqueeze(1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0 and verbose:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    if epoch == num_epochs:
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device).unsqueeze(1)
                output = model(data)
                pred = torch.round(output)
                correct += torch.sum(pred == target)
            accuracy = np.rint(
                (correct.detach().item() / len(train_loader.dataset)) * 100
            )
            print(
                "\n\nTrain Set Confusion Matrix: \n\n", confusion_matrix(pred, target)
            )
            print(
                "\nTrain Epoch: {} \t Accuracy {}% \tLoss: {:.6f}".format(
                    epoch, accuracy, loss.item()
                )
            )


def test(model, criterion, device, epoch, num_epochs, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).unsqueeze(1)
            output = model(data)
            test_loss += criterion(output, target)  # sum up batch loss
            pred = torch.round(output)
            correct += torch.sum(pred == target)

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    if epoch == num_epochs:
        print("\n\nTest Set Confusion Matrix: \n\n", confusion_matrix(pred, target))
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )


class Dataset(Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, features, labels):
        "Initialization"
        self.labels = labels
        self.features = features

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.labels)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample

        # Load data and get label
        X = self.features[index, :]
        Y = self.labels[index]

        return X, Y

    def scale(self, scaler, train):
        if train:
            self.features = scaler.fit_transform(self.features)
        else:
            self.features = scaler.transform(self.features)
