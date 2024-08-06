import builtins

original_print = print

def print(*args, **kwargs):
    original_print(*args, flush=True, **kwargs)


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


class TaskHead(nn.Module):
    def __init__(self, hidden_dim=512, input_dim=512, dropout=0.1, depth=3, combine=True):
        super(TaskHead, self).__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, hidden_dim)
        self.final_dense = nn.Linear(input_dim + hidden_dim, 1) if combine else nn.Linear(hidden_dim, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.combine = combine
        self.depth = depth

    def forward(self, x):
        original_x = x

        x = self.dense1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.dense2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        if self.depth == 4:
            x = self.dense3(x)
            x = self.bn3(x)
            x = F.relu(x)
            x = self.dropout(x)

        x = torch.cat((x, original_x), dim=1) if self.combine else x
        x = self.final_dense(x)
        
        return x


def model_factory(hidden_dim, depth, combine, task, lr, epochs=25, warmup=5, weight_decay=0.0001):
    model = TaskHead(hidden_dim=hidden_dim, depth=depth, combine=combine)
    optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCELoss() if task == 'classification' else nn.MSELoss()        

    def lr_fn(epoch):
        if epoch < warmup: return epoch / warmup
        else: return (1 + math.cos(math.pi * (epoch - warmup) / (epochs - warmup))) / 2

    lr_scheduler = LambdaLR(optimiser, lr_lambda=lr_fn)
    return model, optimiser, lr_scheduler, loss_fn


def cantor_pairing(a, b):
    """
    We have two loops one with repetitions and one with folds;
    To ensure that each innermost execution is seeded with a unique seed,
    we use Cantor Pairing function to combine two seeds into a unique number.
    """
    return (a + b) * (a + b + 1) // 2 + b


def evaluate(predictor, dataloader, loss_fn, task):
    predictor.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            logits = predictor(inputs).squeeze()
            loss = loss_fn(torch.sigmoid(logits), targets) if task == 'classification' else loss_fn(logits, targets)
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


def train_one_epoch(predictor, train_loader, optimiser, lr_scheduler, loss_fn, epoch):
    predictor.train()        
    train_loss = 0
    
    lr_scheduler.step(epoch)
    
    for inputs, targets in train_loader:
        optimiser.zero_grad()
        logits = predictor(inputs).squeeze()
        loss = loss_fn(torch.sigmoid(logits), targets) if task == 'classification' else loss_fn(logits, targets)
        loss.backward()
        optimiser.step()
        train_loss += loss.item()

    return predictor


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


import sys
ckpt_name = sys.argv[1]
base  = '/home/blazejb/minimol/minimol/ckpts/'

EPOCHS = 25
REPETITIONS = 3
ENSEMBLE_SIZE = 5
RESULTS_FILE_PATH = f'results_best_{ckpt_name}.pkl'
DEFAULT_HPARAMS = {'hidden_dim': 512,  'depth': 3, 'combine': True, 'lr': 0.0003}
# SWEEP_RESULTS = {
#     'caco2_wang':                       {'hidden_dim': 2048, 'depth': 4, 'combine': True, 'lr': 0.0005},
#     'hia_hou':                          {'hidden_dim': 2048, 'depth': 4, 'combine': True, 'lr': 0.0003},
#     'pgp_broccatelli':                  {'hidden_dim': 512,  'depth': 4, 'combine': True, 'lr': 0.0003},
#     'bioavailability_ma':               {'hidden_dim': 512,  'depth': 3, 'combine': True, 'lr': 0.0003},
#     'lipophilicity_astrazeneca':        {'hidden_dim': 2048, 'depth': 4, 'combine': True, 'lr': 0.0005},
#     'solubility_aqsoldb':               {'hidden_dim': 1024, 'depth': 4, 'combine': True, 'lr': 0.0005},
#     'bbb_martins':                      {'hidden_dim': 2048, 'depth': 3, 'combine': True, 'lr': 0.0001},
#     'ppbr_az':                          {'hidden_dim': 2048, 'depth': 4, 'combine': True, 'lr': 0.0003},
#     'vdss_lombardo':                    {'hidden_dim': 1024, 'depth': 4, 'combine': True, 'lr': 0.0001},
#     'cyp2d6_veith':                     {'hidden_dim': 512,  'depth': 3, 'combine': True, 'lr': 0.0001},
#     'cyp3a4_veith':                     {'hidden_dim': 512,  'depth': 3, 'combine': True, 'lr': 0.0001},
#     'cyp2c9_veith':                     {'hidden_dim': 512,  'depth': 3, 'combine': True, 'lr': 0.0001},
#     'cyp2d6_substrate_carbonmangels':   {'hidden_dim': 512,  'depth': 3, 'combine': True, 'lr': 0.0001},
#     'cyp3a4_substrate_carbonmangels':   {'hidden_dim': 512,  'depth': 3, 'combine': True, 'lr': 0.0001},
#     'cyp2c9_substrate_carbonmangels':   {'hidden_dim': 1024, 'depth': 3, 'combine': True, 'lr': 0.0005},
#     'half_life_obach':                  {'hidden_dim': 1024, 'depth': 3, 'combine': True, 'lr': 0.0003},
#     'clearance_microsome_az':           {'hidden_dim': 1024, 'depth': 4, 'combine': True, 'lr': 0.0005},
#     'clearance_hepatocyte_az':          {'hidden_dim': 2048, 'depth': 4, 'combine': True, 'lr': 0.0005},
#     'herg':                             {'hidden_dim': 512,  'depth': 3, 'combine': True, 'lr': 0.0003},
#     'ames':                             {'hidden_dim': 512,  'depth': 3, 'combine': True, 'lr': 0.0001},
#     'dili':                             {'hidden_dim': 512,  'depth': 4, 'combine': True, 'lr': 0.0005},
#     'ld50_zhu':                         {'hidden_dim': 1024, 'depth': 4, 'combine': True, 'lr': 0.0001},
# }

if os.path.exists(RESULTS_FILE_PATH):
    with open(RESULTS_FILE_PATH, 'rb') as f:
        predictions_list = pickle.load(f)
else:
    predictions_list = []

group = admet_group(path='admet_data/')
featuriser = Minimol(ckpt_folder=os.path.join(base, ckpt_name))

# LOOP 1: repetitions
for rep_i, seed1 in enumerate(range(1, REPETITIONS+1)):
    print(f"Repetition {rep_i + 1} / 5")
    predictions = {}

    # LOOP 2: datasets
    for dataset_i, dataset_name in enumerate(group.dataset_names):
        print(f"\tDataset {dataset_name}, {dataset_i + 1} / {len(group.dataset_names)}")

        benchmark = group.get(dataset_name)
        name = benchmark['name']
        mols_test = benchmark['test']
        with open(os.devnull, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull): # suppress output
            mols_test['Embedding'] = featuriser(list(mols_test['Drug']))
        test_loader = DataLoader(AdmetDataset(mols_test), batch_size=128, shuffle=False)

        task = 'classification' if len(benchmark['test']['Y'].unique()) == 2 else 'regression'
        
        best_models = []
        # LOOP3: ensemble on folds
        for fold_i, seed2 in enumerate(range(REPETITIONS+1, REPETITIONS+ENSEMBLE_SIZE+1)):
            print(f"\t\tFold {fold_i + 1} / 5")
            seed = cantor_pairing(seed1, seed2)

            with open(os.devnull, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull): # suppress output
                mols_train, mols_valid = group.get_train_valid_split(benchmark = name, split_type = 'default', seed = seed)
                mols_train['Embedding'] = featuriser(list(mols_train['Drug']))
                mols_valid['Embedding'] = featuriser(list(mols_valid['Drug']))
            val_loader   = DataLoader(AdmetDataset(mols_valid), batch_size=128, shuffle=False)
            train_loader = DataLoader(AdmetDataset(mols_train), batch_size=32, shuffle=True)

            # hparams = SWEEP_RESULTS[dataset_name]
            hparams = DEFAULT_HPARAMS
            model, optimiser, lr_scheduler, loss_fn = model_factory(**hparams, task=task)

            best_epoch = {"model": None, "result": None}
            
            # LOOP4: training loop
            for epoch in range(EPOCHS):
                model = train_one_epoch(model, train_loader, optimiser, lr_scheduler, loss_fn, epoch)
                val_loss = evaluate(model, val_loader, loss_fn, task=task)

                if best_epoch['model'] is None:
                    best_epoch['model'] = deepcopy(model)
                    best_epoch['result'] = deepcopy(val_loss)
                else:
                    best_epoch['model'] = best_epoch['model'] if best_epoch['result'] <= val_loss else deepcopy(model)
                    best_epoch['result'] = best_epoch['result'] if best_epoch['result'] <= val_loss else deepcopy(val_loss)

            best_models.append(deepcopy(best_epoch['model']))

        y_pred_test = evaluate_ensemble(best_models, test_loader, task)
        
        predictions[name] = y_pred_test

    predictions_list.append(predictions)
    with open(RESULTS_FILE_PATH, 'wb') as f: pickle.dump(predictions_list, f)

results = group.evaluate_many(predictions_list)
print(results)

"""
>> {
    'caco2_wang': [0.35, 0.018],
    'hia_hou': [0.993, 0.005],
    'pgp_broccatelli': [0.942, 0.002],
    'bioavailability_ma': [0.689, 0.02],
    'lipophilicity_astrazeneca': [0.456, 0.008],
    'solubility_aqsoldb': [0.741, 0.013],
    'bbb_martins': [0.924, 0.003],
    'ppbr_az': [7.696, 0.125],
    'vdss_lombardo': [0.535, 0.027],
    'cyp2d6_veith': [0.719, 0.004],
    'cyp3a4_veith': [0.877, 0.001],
    'cyp2c9_veith': [0.823, 0.006],
    'cyp2d6_substrate_carbonmangels': [0.695, 0.032],
    'cyp3a4_substrate_carbonmangels': [0.663, 0.008],
    'cyp2c9_substrate_carbonmangels': [0.474, 0.025],
    'half_life_obach': [0.495, 0.042],
    'clearance_microsome_az': [0.628, 0.005],
    'clearance_hepatocyte_az': [0.446, 0.029],
    'herg': [0.846, 0.016],
    'ames': [0.849, 0.004],
    'dili': [0.956, 0.006],
    'ld50_zhu': [0.585, 0.008]
}
"""