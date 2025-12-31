import torch
import torch.nn as nn

class ParetoSetModel(torch.nn.Module):
    def __init__(self, n_dim, n_obj):
        super(ParetoSetModel, self).__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj
       
        self.fc1 = nn.Linear(self.n_obj, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.n_dim)
        
        self.alpha = nn.Parameter(data=torch.rand((1, n_dim - 1), dtype=torch.float), requires_grad=True)
        self.beta = nn.Parameter(data=torch.rand((1, n_dim - 1), dtype=torch.float), requires_grad=True)
        self.gamma = nn.Parameter(data=torch.rand((1, n_dim - 1), dtype=torch.float), requires_grad=True)
        
       
    def forward(self, pref):

        x = torch.relu(self.fc1(pref))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        x[:,0] = torch.sigmoid(x[:,0]) 
        x[:,2] = torch.sigmoid(x[:,2]) 
        

        alpha = torch.sigmoid(self.alpha[0]) 
        beta =  self.beta[0]
        
        x[:,1] = 1 - alpha[0] * (x[:,0] - beta[0]) ** 2
        x[:,3] = 1 - alpha[1] * (x[:,0] - beta[1] ) ** 2
        
        x = torch.clamp(x, min = 0, max = 1)
      
        return x.to(torch.float64)