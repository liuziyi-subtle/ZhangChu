from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
# from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from utils import accuracy, init_logger
import datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                data_loader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                data_loader = valid_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            # for inputs, labels in dataloaders[phase]:
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            # epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print("len(data_loader.dataset): {}.".format(
                len(data_loader.dataset)))
            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


dataset_train, dataset_valid = datasets.get_dataset(
    "/data/data/with-zhangchu/results/df_records_cotton.csv")
dataset_train.data = dataset_train.data[:256, :]
dataset_valid.data = dataset_valid.data[:128, :]
train_loader = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=32,
                                           shuffle=True,
                                           num_workers=4,
                                           pin_memory=True)
valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                           batch_size=32,
                                           shuffle=False,
                                           num_workers=4,
                                           pin_memory=True)


model_ft = torch.load(
    "/data/workspace/with-zhangchu/soybean-and-cotton/model_20200803.model")
# num_features = model_ft.linear.in_features
# model_ft.linear = nn.Linear(num_features, 7)
model_ft.linear.out_features = 7
model_ft.linear.reset_parameters()
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


model_ft = train_model(model_ft, train_loader, valid_loader, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=5)

print()
