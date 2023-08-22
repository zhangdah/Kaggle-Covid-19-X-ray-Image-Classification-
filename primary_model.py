import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

torch.manual_seed(1)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, red_dim):
        super(ResidualBlock, self).__init__()
        self.red_dim = red_dim
        self.residual_block = nn.Sequential(
            nn.Conv2d(256, red_dim, 1),
            nn.BatchNorm2d(red_dim),
            nn.ReLU(),
            nn.Conv2d(red_dim, red_dim, 3, padding=1),
            nn.BatchNorm2d(red_dim),
            nn.ReLU(),
            nn.Conv2d(red_dim, 256, 1),
            nn.BatchNorm2d(256)
        )
        
    def forward(self, x):
        residual = x

        out = self.residual_block(x)

        # residual connection
        out += residual
        out = F.relu(out)

        return out

class Covid19PredictionNetwork(nn.Module):
    def __init__(self, block, num_res_blocks, red_dim):
        super(Covid19PredictionNetwork, self).__init__()
        self.name = "CPN"+"_num_res"+str(num_res_blocks)+"_red_dim"+str(red_dim)
        self.num_res_blocks = num_res_blocks
        self.red_dim = red_dim

        # define model architecture
        residual_net = OrderedDict()
        for i in range(num_res_blocks):
            residual_net[str(i)] = block(red_dim)
        self.residual_net = nn.Sequential(residual_net)

        self.fc = nn.Linear(256*6*6, 3)

    def forward(self, x):
        x = self.residual_net(x)
        x = x.view(-1, 256*6*6)
        x = self.fc(x)
        return x

def get_model_name(name, batch_size, learning_rate, epoch):
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,batch_size,learning_rate,epoch)
    return path

def get_accuracy(model, data_loader):
    correct = 0
    total = 0
    class_acc = [0, 0, 0]
    class_total = [0, 0, 0]

    for imgs, labels in data_loader:
        
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()

        output = model(imgs)
        pred = output.max(1, keepdim=True)[1]
        corrected = pred.eq(labels.view_as(pred)).sum().item()

        correct += corrected
        class_acc[labels[0].item()] += corrected

        total += imgs.shape[0]
        class_total[labels[0].item()] += imgs.shape[0]

    print("Covid acc:", class_acc[0]/class_total[0])
    print("Healthy acc:", class_acc[1]/class_total[1])
    print("Virual acc:", class_acc[2]/class_total[2])

    return correct / total

def train(model, x_train, x_val, batch_size=64, num_epochs=50, lr=0.001):

    train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(x_val, batch_size=batch_size)

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses, train_acc, val_acc = [], [], []

    # training
    for epoch in range(num_epochs):
        iterations = 0
        loss_avg = 0

        for imgs, labels in iter(train_loader):

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()

            out = model(imgs)

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            iterations += 1
            loss_avg += loss.item()

        losses.append(loss_avg/iterations)

        train_acc.append(get_accuracy(model, train_loader))
        val_acc.append(get_accuracy(model, val_loader))
        print("Epoch:", epoch, "Training Accuracy:", train_acc[-1], "Validation Accuarcy:", val_acc[-1])
        model_path = get_model_name(model.name, batch_size, lr, epoch)
        torch.save(model.state_dict(), model_path)

    # plotting
    plt.title("Loss")
    plt.plot(range(len(losses)), losses, label="Train")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Train&Validation Accuarcy")
    plt.plot(range(len(train_acc)), train_acc, label="Train")
    plt.plot(range(len(val_acc)), val_acc, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))

if __name__ == '__main__':

    # hyperparameters
    num_res_blocks = 4
    red_dim = 128

    batch_size = 64
    lr = 0.001
    num_epochs = 30

    model = Covid19PredictionNetwork(ResidualBlock, num_res_blocks, red_dim)

    x_train = torchvision.datasets.DatasetFolder("Alexnet_emb/Training", loader=torch.load, extensions=(''))
    x_test = torchvision.datasets.DatasetFolder("Alexnet_emb/Testing", loader=torch.load, extensions=(''))
    x_val = torchvision.datasets.DatasetFolder("Alexnet_emb/Validation", loader=torch.load, extensions=(''))

    train(model, x_train, x_val, batch_size=batch_size, num_epochs=num_epochs, lr=lr)
    
    # test_loader = torch.utils.data.DataLoader(x_test, batch_size=batch_size)
    # print("Testing acc:", get_accuracy(model, test_loader))

