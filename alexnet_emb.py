import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

if __name__ == '__main__':
    
    alexnet = torchvision.models.alexnet(pretrained=True)

    data_dir = "Data/"
    save_dir = "Alexnet_emb/"

    covid_counter = 0
    healthy_counter = 0
    viral_counter = 0

    for i in ("Training", "Validation", "Testing"):

        dataset = torchvision.datasets.ImageFolder(
            root=data_dir+i,
            transform=torchvision.transforms.ToTensor())
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        for img, label in iter(loader):

            emb = alexnet.features(img)

            if label == 0:
                img_save_dir = save_dir+i+"/covid/"+"covid"+str(covid_counter)
                covid_counter += 1
            elif label == 1:
                img_save_dir = save_dir+i+"/healthy/"+"healthy"+str(healthy_counter)
                healthy_counter += 1
            else:
                img_save_dir = save_dir+i+"/viral/"+"viral"+str(viral_counter)
                viral_counter += 1

            torch.save(torch.from_numpy(emb.squeeze().detach().numpy()), img_save_dir)