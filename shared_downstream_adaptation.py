from minimol import Minimol

import os
import math
from copy import deepcopy
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from tdc.benchmark_group import admet_group

from contextlib import redirect_stdout, redirect_stderr


class MultiTaskModel(nn.Module):
    def __init__(self, hidden_dim=512, input_dim=512, head_hidden_dim=256, dropout=0.1, task_names=None):
        super(MultiTaskModel, self).__init__()
        
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.heads = nn.ModuleDict({
            task_name: nn.Sequential(
                nn.Linear(hidden_dim, head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, 1)
            ) for task_name in task_names
        })

        self.trunk_frozen = False

    def forward(self, x, task_name):
        x = self.dense1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.dense2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.heads[task_name](x)
        return x

    def freeze_trunk(self):
        self.trunk_frozen = True
        for param in self.dense1.parameters():
            param.requires_grad = False
        for param in self.dense2.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False
        for param in self.bn2.parameters():
            param.requires_grad = False

    def unfreeze_trunk(self):
        self.trunk_frozen = False
        for param in self.dense1.parameters():
            param.requires_grad = True
        for param in self.dense2.parameters():
            param.requires_grad = True
        for param in self.bn1.parameters():
            param.requires_grad = True
        for param in self.bn2.parameters():
            param.requires_grad = True



def model_factory(lr=3e-3, epochs=25, warmup=5, weight_decay=1e-4):
    model = MultiTaskModel()
    optimiser = optim.adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_fn(epoch):
        if epoch < warmup: return epoch / warmup
        else: return (1 + math.cos(math.pi * (epoch - warmup) / (epochs - warmup))) / 2

    lr_scheduler = LambdaLR(optimiser, lr_lambda=lr_fn)
    return model, optimiser, lr_scheduler


def evaluate(predictor, task, eval_type='val'):
    predictor.eval()
    total_loss = 0

    dataloader = task.val_dataloader if eval_type == 'val' else task.test_dataloader

    with torch.no_grad():
        for inputs, targets in dataloader:
            logits = predictor(inputs, task_name=task.name).squeeze()
            loss = task.get_loss(logits, targets)
            total_loss += loss.item()

    loss = total_loss / len(dataloader)
    
    return loss


def evaluate_ensemble(predictors, dataloader, task):
    predictions = []
    with torch.no_grad():
        
        for inputs, _ in dataloader:
            ensemble_logits = [predictor(inputs).squeeze() for predictor in predictors]
            averaged_logits = torch.mean(torch.stack(ensemble_logits), dim=0)
            if task == 'classification':
                predictions += torch.sigmoid(averaged_logits)
            else:
                predictions += averaged_logits

    return predictions


def train_one_epoch(predictor, task, optimiser):
    train_loss = 0
        
    for inputs, targets in task.train_loader:
        optimiser.zero_grad()
        logits = predictor(inputs, task_name=task.name).squeeze()
        loss = task.get_loss(logits, targets)
        loss.backward()
        optimiser.step()
        train_loss += loss.item()

    return predictor, train_loss / len(task.train_loader)


class AdmetDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples['Embedding'].tolist()
        self.targets = [float(target) for target in samples['Y'].tolist()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = torch.tensor(self.samples[idx])
        target = torch.tensor(self.targets[idx])
        return sample, target


class Task:
    def __init__(self, dataset_name, featuriser):
        benchmark = group.get(dataset_name)
        with open(os.devnull, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull): # suppress output
            mols_test               = benchmark['test']
            mols_train, mols_valid  = group.get_train_valid_split(benchmark=dataset_name, seed=42)
            mols_test['Embedding']  = featuriser(list(mols_test['Drug']))
            mols_train['Embedding'] = featuriser(list(mols_train['Drug']))
            mols_valid['Embedding'] = featuriser(list(mols_valid['Drug']))
        self.name         = dataset_name
        self.test_loader  = DataLoader(AdmetDataset(mols_test), batch_size=128, shuffle=False)
        self.val_loader   = DataLoader(AdmetDataset(mols_valid), batch_size=128, shuffle=False)
        self.train_loader = DataLoader(AdmetDataset(mols_train), batch_size=32, shuffle=True)
        self.task         = 'classification' if len(benchmark['test']['Y'].unique()) == 2 else 'regression'
        self.loss_fn      = nn.BCELoss() if self.task == 'classification' else nn.MSELoss()        

    def get_loss(self, logits, targets):
        if self.task == 'classification':
            return self.loss_fn(torch.sigmoid(logits), targets)
        else:
            return self.loss_fn(logits, targets)



EPOCHS = 25

group = admet_group(path='admet_data/')
featuriser = Minimol()
tasks = {dataset_name: Task(dataset_name, featuriser) for dataset_name in group.dataset_names}
del featuriser
model, optimiser, lr_scheduler = model_factory()

model.unfreeze_trunk()
for epoch in range(EPOCHS):
    for task_i, (task_name, task) in enumerate(tasks.items()):
        #lr_scheduler.step(epoch)
        model, train_loss = train_one_epoch(model, task, optimiser, lr_scheduler)
        val_loss = evaluate(model, task, eval_type='val')
        print(f'epoch={epoch+1} / {EPOCHS} | {task_name=} | {train_loss:.4f=} | {val_loss:.4f=}')