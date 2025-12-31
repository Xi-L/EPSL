import torch
import torch.nn as nn

class ParetoSetModel(torch.nn.Module):
    def __init__(self, n_dim, n_obj):
        super(ParetoSetModel, self).__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.n_node = 400 
        
        self.fc1 = nn.Linear(self.n_obj, self.n_node)
        self.fc2 = nn.Linear(self.n_node, self.n_node)
        self.fc4 = nn.Linear(self.n_node, self.n_node)
        self.fc3 = nn.Linear(self.n_node, 1)
      
        self.keypoints = torch.nn.parameter.Parameter(torch.rand(4, self.n_dim), requires_grad=True)
        self.normalized_keypoints = None
        self.t =None
        
    def forward(self, pref):

        t = torch.relu(self.fc1(pref))
        t = torch.relu(self.fc2(t))
        t = torch.relu(self.fc4(t))
        t = self.fc3(t) + pref[:,0][:,None]
        
        t = torch.sigmoid(t) 
        
        self.t = t
        
        n_keypoints = self.keypoints.shape[0]
        t = t * (n_keypoints-1)
        

        b_list = []
        for i in range(n_keypoints-1):
            bi = (t>i) & (t<=i+1)
            b_list.append(bi)
            
        
        a_list = []
        
        self.normalized_keypoints = torch.sigmoid(self.keypoints)
        
        if self.training == False:
            self.normalized_keypoints[self.normalized_keypoints < 0.01] = 0
            self.normalized_keypoints[self.normalized_keypoints > 0.99] = 1
        
        for i in range(n_keypoints-1):
            ai = self.normalized_keypoints[i] + (self.normalized_keypoints[i+1] - self.normalized_keypoints[i]) * (t-i)
            a_list.append(ai)
       
        res = sum([ai*bi for ai,bi in zip(a_list,b_list)])
       
        return res.to(torch.float64)