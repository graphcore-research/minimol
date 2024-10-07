from typing import Union

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import matplotlib.pyplot as plt


def run(variance             : float,
        data_generator       : callable, 
        model_class          : classmethod,
        epochs               : int                   = 10, 
        n_samples            : int                   = 1000,
        batch_size           : int                   = 128,
        lr                   : Union[float, tuple]   = 1e-1,
        uncertainty_weighing : bool                  = False,
        repeats              : int                   = 1,
        verbose              : bool                  = False,
        plot_dynamics        : bool                  = False):

    learned_variances = np.zeros(shape=(repeats))
    device = torch.device('cuda' if torch.is_available() and batch_size > 64 else 'cpu')
    
    if plot_dynamics:
        all_lrs              = []
        all_uw_losses        = []
        all_criterion_losses = []
        all_variances        = []
        all_log_vars         = []

    for i, seed in enumerate(range(repeats)):
        x, y = data_generator(n_samples=n_samples, variance=variance, seed=seed, as_tensors=True)
        model = model_class(uw=uncertainty_weighing).to(device)

        if isinstance(lr, tuple) or isinstance(lr, list):
            lr1 = lr[0]
            lr2 = lr[1]
        else:
            lr1 = lr2 = lr

        params = [
            {'params': model.linear.parameters(), 'lr': lr1},
            {'params': [model.log_variance],      'lr': lr2}
        ]

        optimizer        = optim.Adam(params)
        # n_warmup_samples = n_samples // 10
        # warmup_scheduler = LinearLR(optimizer, start_factor=lr / 1e1, total_iters=n_warmup_samples)
        # cosine_scheduler = CosineAnnealingLR(optimizer, T_max=n_samples - n_warmup_samples)
        # scheduler        = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[n_warmup_samples])

        dataloader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for x_batch, y_true in dataloader:
                x_batch = x_batch.to(device)
                y_true = y_true.to(device)
                optimizer.zero_grad()
                outputs = model(x_batch)
                uw_loss, criterion_loss = model.compute_loss(outputs, y_true)
                uw_loss.backward()
                optimizer.step()
                # scheduler.step()

                if plot_dynamics:
                    all_lrs.append(optimizer.param_groups[0]['lr'])
                    all_uw_losses.append(uw_loss.item())
                    all_criterion_losses.append(criterion_loss.item())
                    all_log_vars.append(model.log_variance.detach().cpu().numpy())
                    all_variances.append(torch.exp(model.log_variance).detach().cpu().numpy())

        learned_variances[i] = torch.exp(model.log_variance).detach().cpu().numpy()

    if plot_dynamics:
        alpha = 0.7
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.title("Loss vs Step")
        plt.plot(all_uw_losses, label="UW loss", alpha=alpha, color='blue')
        plt.plot(all_criterion_losses, label="Criterion loss", alpha=alpha, color='red')
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.title("Learned variance vs. steps")
        plt.plot(all_variances, alpha=alpha, label="Learned var")
        plt.plot(np.ones_like(all_variances)*variance, color='grey', alpha=alpha, label="True var")
        plt.xlabel("Step")
        plt.ylabel("Learned variance")
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.title("Learned log_variance vs. steps")
        plt.plot(all_log_vars, alpha=alpha, label="Learned log var")
        plt.plot(np.ones_like(all_variances)*np.log(variance), color='grey', alpha=alpha, label="True log var")
        plt.xlabel("Step")
        plt.ylabel("Learned log_variance")
        plt.legend()

        plt.tight_layout()
        plt.show()

    return learned_variances