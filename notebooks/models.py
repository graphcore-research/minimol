import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


class LinearRegressionModel(nn.Module):

    def __init__(self,
                 uw             : bool  = False,
                 init_var_value : float = 0.001):

        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        self.log_variance = nn.Parameter(torch.tensor(init_var_value), requires_grad=True) if uw else None
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        return self.linear(x)
        
    def _get_log_sqrt_var(self):
        return 0.5 * self.log_variance if self.log_variance else None
        
    def _get_loss_weight(self):
        return 0.5 * torch.exp(-self.log_variance) if self.log_variance else None

    def compute_loss(self, y, y_true):
        criterion_loss = self.criterion(y, y_true) 
        if self.log_variance:
            uw_loss = self._get_loss_weight() * criterion_loss + self._get_log_sqrt_var()
            return uw_loss, criterion_loss
        return criterion_loss


class LogisticRegressionModel(nn.Module):
    
    def __init__(self,
                 uw             : bool  = False,
                 n_features     : int   = 1,
                 init_var_value : float = 0.001,
                 exact          : bool  = False):
    
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(n_features, 2)
        self.log_variance = nn.Parameter(torch.log(torch.tensor(init_var_value)), requires_grad=True) if uw else None
        self.criterion = nn.CrossEntropyLoss()
        self.exact = exact

    def forward(self, x):
        return self.linear(x)
    
    def _exact_reg_term(self, logits):
        numerator   = torch.sum(torch.exp(self._get_loss_weight() * logits), axis=1)
        denominator = torch.sum(torch.exp(logits), axis=1)**(self._get_loss_weight()) 
        return torch.mean(torch.log(numerator / denominator))

    def _get_log_sigma(self):
        return 0.5 * self.log_variance if self.log_variance else None
        
    def _get_loss_weight(self):
        return torch.exp(-self.log_variance) if self.log_variance else None

    def compute_loss(self, y, y_true):
        criterion_loss = self.criterion(y, y_true) 

        if self.log_variance:
            uw_loss = self._get_loss_weight() * criterion_loss
            uw += self._exact_reg_term(y) if self.exact else self._get_log_sigma()
            return uw_loss, criterion_loss
        
        return criterion_loss
