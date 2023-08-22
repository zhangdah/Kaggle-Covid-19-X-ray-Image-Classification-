import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

torch.manual_seed(1)

def load_dataset(dir="COVID-19_Radiography_Database/", val_size=0.2, test_size=0.1):

    # all the dataset
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    	torchvision.transforms.Resize((256,256)),
    	torchvision.transforms.Grayscale(num_output_channels=1)])

    dataset = torchvision.datasets.ImageFolder(
        root=dir, 
        transform=transform)
    
    # use train_test_split to split the dataset to 60% train, 20% validation, 20% test
    # stratified split
    train_idx, test_idx = train_test_split(np.arange(len(dataset)),
                        test_size=test_size,
                        shuffle=True,
                        stratify=dataset.targets,
                        random_state=1)
    
    train_idx, val_idx = train_test_split(train_idx,
                        test_size=val_size/(1-test_size),
                        shuffle=True,
                        stratify=[dataset.targets[i] for i in train_idx],
                        random_state=1)
    
    return train_idx, val_idx, test_idx, dataset

if __name__ == '__main__':
    # split data with 0.7 for training, 0.2 fo validation, 0.1 for testing
    train_idx, val_idx, test_idx, dataset = load_dataset()

    # set sampler for dataloader
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

    # set up dataloader for data preprocessing
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=test_sampler)

    # label:
    # 0: covid case
    # 1: healthy
    # 2: viral pneumonia

    # image counter
    covid_counter = 0
    healthy_counter = 0
    viral_counter = 0

    # process training data
    for img, label in iter(train_loader):

        image_augmentation = torch.nn.Sequential(
            torchvision.transforms.RandomCrop((torch.randint(low=224, high=256, size=(2,)))),
            torchvision.transforms.Resize((224, 224)),
        )

        image = image_augmentation(img)

        plt.imshow(image.squeeze(), cmap='gray', vmin=0, vmax=1)
        plt.show()

        image = image * 0.8

        plt.imshow(image.squeeze(), cmap='gray', vmin=0, vmax=1)
        plt.show()

        break