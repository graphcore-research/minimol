import torch
import random
import numpy as np
import matplotlib.pyplot as plt


def generate_1d_regression_data(n_samples    : int   = 1000,
                                domain_range : int   = 2.0,
                                variance     : float = 0.1,
                                seed         : int   = 42,
                                as_tensors   : bool  = False):
    np.random.seed(seed)
    
    a = np.random.uniform(-10, 10)
    b = np.random.uniform(-10, 10)
    
    f = lambda x: (a * x) + b
    
    x = np.random.uniform(-domain_range, domain_range, n_samples).reshape(-1, 1)
    y = f(x).flatten() + np.random.normal(0, np.sqrt(variance), n_samples)
    
    if as_tensors:
        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.from_numpy(y).float().view(-1, 1)
        return x_tensor, y_tensor

    return x, y


def generate_classification_data(n_samples   : int,
                                 variance    : float, 
                                 n_features  : int = 2, 
                                 mean_range  : tuple = (-3, 3), 
                                 seed        : int = 42, 
                                 as_tensors  : bool = False,
                                 one_hot     : bool = False):
    """
    The variance is defined as `(var0 + var1) / (||mu1 - mu0||^2)`
    """
    np.random.seed(seed)

    mu0 = np.random.uniform(mean_range[0], mean_range[1], n_features)
    mu1 = np.random.uniform(mean_range[0], mean_range[1], n_features)
    mean_distance = np.linalg.norm(mu1 - mu0) ** 2
    var0 = var1 = variance * mean_distance / 2

    cov0 = np.diag([var0] * n_features)
    cov1 = np.diag([var1] * n_features)
    
    X0 = np.random.multivariate_normal(mu0, cov0, n_samples//2)
    X1 = np.random.multivariate_normal(mu1, cov1, n_samples//2)
    
    x = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    
    if one_hot:
        y = np.eye(2)[y.astype(int)]

    if as_tensors:
        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.from_numpy(y).float() if one_hot else torch.from_numpy(y).long()
        return x_tensor, y_tensor
    
    return x, y


def generate_boltzmann_classification_data(n_samples   : int,
                                           variance    : float,
                                           n_features  : int   = 2,
                                           mean_range  : tuple = (-3, 3),
                                           seed        : int   = 42,
                                           one_hot     : bool  = False,
                                           as_tensors  : bool  = False):

    np.random.seed(seed)
    
    def boltzmann_distribution(x, mu, cov, T):
        inv_cov = np.linalg.inv(cov)
        energy = 0.5 * np.dot((x - mu), np.dot(inv_cov, (x - mu).T))
        return np.exp(-energy / T)

    def metropolis_hastings(n_samples, mu, cov, T):
        samples = []
        current_sample = np.random.multivariate_normal(mu, cov)
        samples.append(current_sample)

        for _ in range(n_samples - 1):
            proposal = np.random.multivariate_normal(current_sample, cov)
            acceptance_ratio = (boltzmann_distribution(proposal, mu, cov, T) / 
                                boltzmann_distribution(current_sample, mu, cov, T))
            if np.random.rand() < acceptance_ratio:
                current_sample = proposal
            samples.append(current_sample)
        
        return np.array(samples)

    mu0 = np.random.uniform(mean_range[0], mean_range[1], n_features)
    mu1 = np.random.uniform(mean_range[0], mean_range[1], n_features)
    
    cov0 = np.eye(n_features)
    cov1 = np.eye(n_features)

    X0 = metropolis_hastings(n_samples // 2, mu0, cov0, variance)
    X1 = metropolis_hastings(n_samples // 2, mu1, cov1, variance)
    
    x = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

    if one_hot:
        y = np.eye(2)[y.astype(int)]

    if as_tensors:
        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.from_numpy(y).float() if one_hot else torch.from_numpy(y).long()
        return x_tensor, y_tensor

    return x, y
