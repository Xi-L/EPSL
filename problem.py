import torch
import numpy as np

device = 'cpu'

def get_problem(name, *args, **kwargs):
    name = name.lower()
    
    PROBLEM = {
        'syn': SYN,
        're21': RE21,
        're22': RE22,
        're23': RE23,
        're24': RE24,
        're25': RE25,
        're31': RE31,
        're32': RE32,
        're33': RE33,
        're34': RE34,
        're35': RE35,
        're36': RE36,
        're37': RE37,
        're41': RE41,
        're42': RE42,
        're61': RE61,
        're91': RE91
 }

    if name not in PROBLEM:
        raise Exception("Problem not found.")
    
    return PROBLEM[name](*args, **kwargs)


def closest_value(arr, val):
    '''
    Get closest value to val in arr
    '''
    return arr[torch.argmin(torch.abs(arr[:, None] - val), axis=0)]

def div(x1, x2):
    '''
    Divide x1 / x2, return 0 where x2 == 0
    
    '''
    results = x1 * 0.0
    results[x2 != 0.0] = x1[x2 != 0.0] / x2[x2 != 0.0]
   
    return results


class SYN():
    def __init__(self, n_dim = 3):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]
        
    def evaluate(self, x):
        n = x.shape[1]
        
        sum1 = (1.0/(n - 1)) * torch.sum(torch.stack([(x[:,i+1] - (torch.sin(10 * (x[:,0] - 0.5)) + 1) / 2 ) ** 2 for i in range(n - 1)]), axis = 0)
        
        f1 = x[:,0]
        f2 = (1 + sum1) * (1 - torch.sqrt(f1/(1 + sum1)))
            
        objs = torch.stack([f1,f2]).T
        
        return objs

    
class RE21():
    def __init__(self, n_dim = 4):
        
        F = 10.0
        sigma = 10.0
        tmp_val = F / sigma
        
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([tmp_val, np.sqrt(2.0) * tmp_val, np.sqrt(2.0) * tmp_val, tmp_val]).float()
        self.ubound = torch.ones(n_dim).float() * 3 * tmp_val
        self.nadir_point = [2886.3695604236013, 0.039999999999998245]
        
    def evaluate(self, x):
        
        F = 10.0
        E = 2.0 * 1e5
        L = 200.0
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.to(device)
            self.ubound = self.ubound.to(device)
        
        x = x * (self.ubound - self.lbound) + self.lbound
        
        f1 =  L * ((2 * x[:,0]) + np.sqrt(2.0) * x[:,1] + torch.sqrt(x[:,2]) + x[:,3])
        f2 =  ((F * L) / E) * ((2.0 / x[:,0]) + (2.0 * np.sqrt(2.0) / x[:,1]) - (2.0 * np.sqrt(2.0) / x[:,2]) + (2.0 /  x[:,3]))
        
        f1 = f1 
        f2 = f2 
        
        objs = torch.stack([f1,f2]).T
        
        return objs
    
class RE22():
    def __init__(self, n_dim = 3):
        
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([0.2, 0.0, 0.0]).float()
        self.ubound = torch.tensor([15, 20, 40]).float()
        self.nadir_point = [361.262944647, 180.01547]
        
    def evaluate(self, x):
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.to(device)
            self.ubound = self.ubound.to(device)
        
        x = x * (self.ubound - self.lbound) + self.lbound
        
        feasible_vals = torch.tensor([0.20, 0.31, 0.40, 0.44, 0.60, 0.62, 0.79, 0.80, 0.88, 0.93, 1.0, 1.20, 1.24, 1.32, 1.40, 1.55, 1.58, 1.60, 1.76, 1.80, 1.86, 2.0, 2.17, 2.20, 2.37, 2.40, 2.48, 2.60, 2.64, 2.79, 2.80, 3.0, 3.08, 3,10, 3.16, 3.41, 3.52, 3.60, 3.72, 3.95, 3.96, 4.0, 4.03, 4.20, 4.34, 4.40, 4.65, 4.74, 4.80, 4.84, 5.0, 5.28, 5.40, 5.53, 5.72, 6.0, 6.16, 6.32, 6.60, 7.11, 7.20, 7.80, 7.90, 8.0, 8.40, 8.69, 9.0, 9.48, 10.27, 11.0, 11.06, 11.85, 12.0, 13.0, 14.0, 15.0])
        if x.device.type == 'cuda':
            feasible_vals = feasible_vals.to(device)
            
        x1 = closest_value(feasible_vals, x[:,0]).to(torch.float64)
        x2 = x[:,1]
        x3 = x[:,2]
        
        #First original objective function
        f1 = (29.4 * x1) + (0.6 * x2 * x3)
      
        g1 = (x1 * x3) - 7.735 * div(x1 * x1, x2) - 180.0
        g2 = 4.0 - div(x3, x2)
        
        
        
        g = torch.stack([g1,g2])
        z = torch.zeros(g.shape).float().to(device).to(torch.float64)
        g = torch.where(g < 0, -g, z)
        
        f2 = torch.sum(g, axis = 0).float() 
        
        objs = torch.stack([f1,f2]).T
        
        return objs
    
    
class RE23():
    def __init__(self, n_dim = 4):
        
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([1, 1, 10,10]).float()
        self.ubound = torch.tensor([100, 100, 200, 240]).float()
        self.nadir_point = [5852.05896876, 1288669.78054]
        
    def evaluate(self, x):
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.to(device)
            self.ubound = self.ubound.to(device)
        
        x = x * (self.ubound - self.lbound) + self.lbound
      
        x1 = 0.0625 * torch.round(x[:,0])  
        x2 = 0.0625 * torch.round(x[:,1])  
        x3 = x[:,2]
        x4 = x[:,3]
        
        #First original objective function
        f1 = (0.6224 * x1 * x3* x4) + (1.7781 * x2 * x3 * x3) + (3.1661 * x1 * x1 * x4) + (19.84 * x1 * x1 * x3)
        f1 = f1.float()
        
        # Original constraint functions 	
        g1 = x1 - (0.0193 * x3)
        g2 = x2 - (0.00954 * x3)
        g3 = (np.pi * x3 * x3 * x4) + ((4.0/3.0) * (np.pi * x3 * x3 * x3)) - 1296000
        
        
        g = torch.stack([g1,g2,g3])
        z = torch.zeros(g.shape).float().to(device).to(torch.float64)
        g = torch.where(g < 0, -g, z)
         
        f2 = torch.sum(g, axis = 0).to(torch.float64)
        
        
        objs = torch.stack([f1,f2]).T
        
        return objs
    
class RE24():
    def __init__(self, n_dim = 2):
        
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.ones(n_dim).float() * 0.5
        self.ubound = torch.tensor([4, 50]).float()
        self.nadir_point = [481.608088535, 44.2819047619]
        
    def evaluate(self, x):
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.to(device)
            self.ubound = self.ubound.to(device)
        
        x = x * (self.ubound - self.lbound) + self.lbound
        
        x1 = x[:,0]
        x2 = x[:,1]

        #First original objective function
        f1 = x1 + (120 * x2)

        E = 700000
        sigma_b_max = 700
        tau_max = 450
        delta_max = 1.5
        sigma_k = (E * x1 * x1) / 100
        sigma_b = 4500 / (x1 * x2)
        tau = 1800 / x2
        delta = (56.2 * 10000) / (E * x1 * x2 * x2)
	
        g1 = 1 - (sigma_b / sigma_b_max)
        g2 = 1 - (tau / tau_max)
        g3 = 1 - (delta / delta_max)
        g4 = 1 - (sigma_b / sigma_k)
        
       
        g = torch.stack([g1,g2,g3,g4])
        z = torch.zeros(g.shape).float().to(device).to(torch.float64)
        g = torch.where(g < 0, -g, z)
         
        f2 = torch.sum(g, axis = 0).float() 
        
        objs = torch.stack([f1,f2]).T
        
        return objs
    
class RE25():
    def __init__(self, n_dim = 3):
        
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([1, 0.6, 0.09]).float()
        self.ubound = torch.tensor([70, 30, 0.5]).float()
        self.nadir_point = [0.40397042546, 2224669.22419]
        
    def evaluate(self, x):
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.to(device)
            self.ubound = self.ubound.to(device)
        
        x = x * (self.ubound - self.lbound) + self.lbound
        
        feasible_vals = torch.tensor([0.009, 0.0095, 0.0104, 0.0118, 0.0128, 0.0132, 0.014, 0.015, 0.0162, 0.0173, 0.018, 0.02, 0.023, 0.025, 0.028, 0.032, 0.035, 0.041, 0.047, 0.054, 0.063, 0.072, 0.08, 0.092, 0.105, 0.12, 0.135, 0.148, 0.162, 0.177, 0.192, 0.207, 0.225, 0.244, 0.263, 0.283, 0.307, 0.331, 0.362, 0.394, 0.4375, 0.5])
       
        if x.device.type == 'cuda':
            feasible_vals = feasible_vals.to(device)
        feasible_vals_mat = feasible_vals.repeat([x.shape[0],1])
        

        x1 = torch.round(x[:,0]) # x[:,0] 
        x2 = x[:,1]
        idx = torch.abs(feasible_vals_mat - x[:,2].reshape(x.shape[0],1)).argmin(dim = 1)
        x3 = feasible_vals[idx] # x[:,2] 
        
        # first original objective function
        f1 = (np.pi * np.pi * x2 * x3 * x3 * (x1 + 2)) / 4.0
        f1 = f1.float()
	    
        # constraint functions
        Cf = ((4.0 * (x2 / x3) - 1) / (4.0 * (x2 / x3) - 4)) + (0.615 * x3 / x2)
        Fmax = 1000.0
        S = 189000.0
        G = 11.5 * 1e+6
        K  = (G * x3 * x3 * x3 * x3) / (8 * x1 * x2 * x2 * x2)
        lmax = 14.0
        lf = (Fmax / K) + 1.05 *  (x1 + 2) * x3
        dmin = 0.2
        Dmax = 3
        Fp = 300.0
        sigmaP = Fp / K
        sigmaPM = 6
        sigmaW = 1.25

        g1 = -((8 * Cf * Fmax * x2) / (np.pi * x3 * x3 * x3)) + S
        g2 = -lf + lmax
        g3 = -3 + (x2 / x3)
        g4 = -sigmaP + sigmaPM
        g5 = -sigmaP - ((Fmax - Fp) / K) - 1.05 * (x1 + 2) * x3 + lf
        g6 =  sigmaW- ((Fmax - Fp) / K) 
        
        g = torch.stack([g1,g2,g3,g4,g5,g6])
        z = torch.zeros(g.shape).float().to(device).to(torch.float64)
        g = torch.where(g < 0, -g, z)
         
        f2 = torch.sum(g, axis = 0).float() 
        
        objs = torch.stack([f1,f2]).T
        
        return objs
    
class RE31():
    def __init__(self, n_dim = 3):
        
      
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([1e-5, 1e-5, 1]).float()
        self.ubound = torch.tensor([100, 100, 3]).float()
        self.nadir_point = [500.002668442, 8246211.25124, 19359919.7502]
        
    def evaluate(self, x):
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.to(device)
            self.ubound = self.ubound.to(device)
        
        x = x * (self.ubound - self.lbound) + self.lbound
      
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]

        # First original objective function
        f1 = x1 * torch.sqrt(16.0 + (x3 * x3)) + x2 * torch.sqrt(1.0 + x3 * x3)
        # Second original objective function
        f2 = div((20.0 * torch.sqrt(16.0 + (x3 * x3))),(x3 * x1))

        # Constraint functions 
        g1 = 0.1 - f1
        g2 = 100000.0 - f2
        g3 = 100000 - div( (80.0 * torch.sqrt(1.0 + x3 * x3)) , (x3 * x2))
        
        g = torch.stack([g1,g2,g3])
        z = torch.zeros(g.shape).float().to(device).to(torch.float64)
        g = torch.where(g < 0, -g, z)

        f3 = torch.sum(g, axis = 0).float() 
        
        
        objs = torch.stack([f1,f2,f3]).T
        
        return objs    
    
class RE32():
    def __init__(self, n_dim = 4):
        
      
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([0.125, 0.1, 0.1, 0.125]).float()
        self.ubound = torch.tensor([5, 10, 10, 5]).float()
        self.nadir_point = [37.7831517014, 17561.6, 425062976.628]
        
    def evaluate(self, x):
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.to(device)
            self.ubound = self.ubound.to(device)
        
        x = x * (self.ubound - self.lbound) + self.lbound
      
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        x4 = x[:,3]

        P = 6000
        L = 14
        E = 30 * 1e6

        G = 12 * 1e6
        tauMax = 13600
        sigmaMax = 30000
    
        # First original objective function
        f1 = (1.10471 * x1 * x1 * x2) + (0.04811 * x3 * x4) * (14.0 + x2)
        # Second original objective function
        f2 = (4 * P * L * L * L) / (E * x4 * x3 * x3 * x3)

        # Constraint functions
        M = P * (L + (x2 / 2))
        tmpVar = ((x2 * x2) / 4.0) + torch.pow((x1 + x3) / 2.0, 2)
        R = torch.sqrt(tmpVar)
        tmpVar = ((x2 * x2) / 12.0) + torch.pow((x1 + x3) / 2.0, 2)
        J = 2 * np.sqrt(2) * x1 * x2 * tmpVar

        tauDashDash = (M * R) / J    
        tauDash = P / (np.sqrt(2) * x1 * x2)
        tmpVar = tauDash * tauDash + ((2 * tauDash * tauDashDash * x2) / (2 * R)) + (tauDashDash * tauDashDash)
        tau = torch.sqrt(tmpVar)
        sigma = (6 * P * L) / (x4 * x3 * x3)
        tmpVar = 4.013 * E * torch.sqrt((x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4) / 36.0) / (L * L)
        tmpVar2 = (x3 / (2 * L)) * np.sqrt(E / (4 * G))
        PC = tmpVar * (1 - tmpVar2)

        g1 = tauMax - tau
        g2 = sigmaMax - sigma
        g3 = x4 - x1
        g4 = PC - P

        g = torch.stack([g1,g2,g3,g4])
        z = torch.zeros(g.shape).float().to(device).to(torch.float64)
        g = torch.where(g < 0, -g, z)

        f3 = torch.sum(g, axis = 0).float() 
        
        objs = torch.stack([f1,f2,f3]).T
        
        return objs    
    
class RE33():
    def __init__(self, n_dim = 4):
        
      
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([55, 75, 1000, 11]).float()
        self.ubound = torch.tensor([80, 110, 3000, 20]).float()
        self.nadir_point = [5.3067, 3.12833430979, 25.0]
        
    def evaluate(self, x):
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.to(device)
            self.ubound = self.ubound.to(device)
        
        x = x * (self.ubound - self.lbound) + self.lbound
      
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        x4 = x[:,3]
        
        # First original objective function
        f1 = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
        # Second original objective function
        f2 = ((9.82 * 1e6) * (x2 * x2 - x1 * x1)) / (x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1))
    
        # Reformulated objective functions
        g1 = (x2 - x1) - 20.0
        g2 = 0.4 - (x3 / (3.14 * (x2 * x2 - x1 * x1)))
        g3 = 1.0 - (2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1)) / torch.pow((x2 * x2 - x1 * x1), 2)
        g4 = (2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)) / (x2 * x2 - x1 * x1) - 900.0
                        
        g = torch.stack([g1,g2,g3,g4])
        z = torch.zeros(g.shape).float().to(device).to(device).to(torch.float64)
        g = torch.where(g < 0, -g, z)

        f3 = torch.sum(g, axis = 0).float() 
        
        
        objs = torch.stack([f1,f2,f3]).T
        
        return objs
    
class RE34():
    def __init__(self, n_dim = 5):
        
      
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([1, 1, 1, 1, 1]).float()
        self.ubound = torch.tensor([3, 3, 3, 3, 3]).float()
        self.nadir_point = [1695.2002035, 10.7454, 0.26399999965]
        
    def evaluate(self, x):
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.to(device)
            self.ubound = self.ubound.to(device)
        
        x = x * (self.ubound - self.lbound) + self.lbound
      
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        x4 = x[:,3]
        x5 = x[:,4]

        f1 = 1640.2823 + (2.3573285 * x1) + (2.3220035 * x2) + (4.5688768 * x3) + (7.7213633 * x4) + (4.4559504 * x5)
        f2 = 6.5856 + (1.15 * x1) - (1.0427 * x2) + (0.9738 * x3) + (0.8364 * x4) - (0.3695 * x1 * x4) + (0.0861 * x1 * x5) + (0.3628 * x2 * x4)  - (0.1106 * x1 * x1)  - (0.3437 * x3 * x3) + (0.1764 * x4 * x4)
        f3 = -0.0551 + (0.0181 * x1) + (0.1024 * x2) + (0.0421 * x3) - (0.0073 * x1 * x2) + (0.024 * x2 * x3) - (0.0118 * x2 * x4) - (0.0204 * x3 * x4) - (0.008 * x3 * x5) - (0.0241 * x2 * x2) + (0.0109 * x4 * x4)
        
        
        objs = torch.stack([f1,f2,f3]).T
        
        return objs    
    

class RE35():
    def __init__(self, n_dim = 7):
        
      
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5.0 ]).float()
        self.ubound = torch.tensor([3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5]).float()
        self.nadir_point = [6634.56208, 1695.96387746, 397.358927317]
        
    def evaluate(self, x):
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.to(device)
            self.ubound = self.ubound.to(device)
        
        x = x * (self.ubound - self.lbound) + self.lbound
      
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = torch.round(x[:,2])
        x4 = x[:,3]
        x5 = x[:,4]
        x6 = x[:,5]
        x7 = x[:,6]
        

        # First original objective function (weight)
        f1 = 0.7854 * x1 * (x2 * x2) * (((10.0 * x3 * x3) / 3.0) + (14.933 * x3) - 43.0934) - 1.508 * x1 * (x6 * x6 + x7 * x7) + 7.477 * (x6 * x6 * x6 + x7 * x7 * x7) + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7)
    
        # Second original objective function (stress)
        tmpVar = torch.pow((745.0 * x4) / (x2 * x3), 2.0)  + 1.69 * 1e7
        f2 =  torch.sqrt(tmpVar) / (0.1 * x6 * x6 * x6)

        # Constraint functions 	
        g1 = -(1.0 / (x1 * x2 * x2 * x3)) + 1.0 / 27.0
        g2 = -(1.0 / (x1 * x2 * x2 * x3 * x3)) + 1.0 / 397.5
        g3 = -(x4 * x4 * x4) / (x2 * x3 * x6 * x6 * x6 * x6) + 1.0 / 1.93
        g4 = -(x5 * x5 * x5) / (x2 * x3 * x7 * x7 * x7 * x7) + 1.0 / 1.93
        g5 = -(x2 * x3) + 40.0
        g6 = -(x1 / x2) + 12.0
        g7 = -5.0 + (x1 / x2)
        g8 = -1.9 + x4 - 1.5 * x6
        g9 = -1.9 + x5 - 1.1 * x7
        g10 =  -f2 + 1300.0
        tmpVar = torch.pow((745.0 * x5) / (x2 * x3), 2.0) + 1.575 * 1e8
        g11 = -torch.sqrt(tmpVar) / (0.1 * x7 * x7 * x7) + 1100.0
        
        
        g = torch.stack([g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11])
        z = torch.zeros(g.shape).float().to(device).to(torch.float64)
        g = torch.where(g < 0, -g, z)

        f3 = torch.sum(g, axis = 0).float() 
        
        
        objs = torch.stack([f1,f2,f3]).T
        
        return objs    
    
    
class RE36():
    def __init__(self, n_dim = 4):
        
      
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([12, 12, 12, 12]).float()
        self.ubound = torch.tensor([60, 60, 60, 60]).float()
        self.nadir_point = [5.931, 56.0, 0.355720675227]
        
    def evaluate(self, x):
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.to(device)
            self.ubound = self.ubound.to(device)
        
        x = x * (self.ubound - self.lbound) + self.lbound
      
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        x4 = x[:,3]

        # First original objective function
        f1 = torch.abs(6.931 - ((x3 / x1) * (x4 / x2)))
        # Second original objective function (the maximum value among the four variables)
        l = torch.stack([x1, x2, x3, x4])
        f2 = torch.max(l, dim = 0)[0]
        
        g1 = 0.5 - (f1 / 6.931)   
        
        g = torch.stack([g1])
        z = torch.zeros(g.shape).float().to(device).to(torch.float64)
        g = torch.where(g < 0, -g, z)                
        f3 = g[0]
        
        objs = torch.stack([f1,f2,f3]).T
        
        return objs
    
class RE37():
    def __init__(self, n_dim = 4):
        
      
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([0, 0, 0, 0]).float()
        self.ubound = torch.tensor([1, 1, 1, 1]).float()
        self.nadir_point = [0.98949120096, 0.956587924661, 0.987530948586]
        
    def evaluate(self, x):
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.to(device)
            self.ubound = self.ubound.to(device)
        
        x = x * (self.ubound - self.lbound) + self.lbound
      
        xAlpha = x[:,0]
        xHA = x[:,1]
        xOA = x[:,2]
        xOPTT = x[:,3]
 
        # f1 (TF_max)
        f1 = 0.692 + (0.477 * xAlpha) - (0.687 * xHA) - (0.080 * xOA) - (0.0650 * xOPTT) - (0.167 * xAlpha * xAlpha) - (0.0129 * xHA * xAlpha) + (0.0796 * xHA * xHA) - (0.0634 * xOA * xAlpha) - (0.0257 * xOA * xHA) + (0.0877 * xOA * xOA) - (0.0521 * xOPTT * xAlpha) + (0.00156 * xOPTT * xHA) + (0.00198 * xOPTT * xOA) + (0.0184 * xOPTT * xOPTT)
        # f2 (X_cc)
        f2 = 0.153 - (0.322 * xAlpha) + (0.396 * xHA) + (0.424 * xOA) + (0.0226 * xOPTT) + (0.175 * xAlpha * xAlpha) + (0.0185 * xHA * xAlpha) - (0.0701 * xHA * xHA) - (0.251 * xOA * xAlpha) + (0.179 * xOA * xHA) + (0.0150 * xOA * xOA) + (0.0134 * xOPTT * xAlpha) + (0.0296 * xOPTT * xHA) + (0.0752 * xOPTT * xOA) + (0.0192 * xOPTT * xOPTT)
        # f3 (TT_max)
        f3 = 0.370 - (0.205 * xAlpha) + (0.0307 * xHA) + (0.108 * xOA) + (1.019 * xOPTT) - (0.135 * xAlpha * xAlpha) + (0.0141 * xHA * xAlpha) + (0.0998 * xHA * xHA) + (0.208 * xOA * xAlpha) - (0.0301 * xOA * xHA) - (0.226 * xOA * xOA) + (0.353 * xOPTT * xAlpha) - (0.0497 * xOPTT * xOA) - (0.423 * xOPTT * xOPTT) + (0.202 * xHA * xAlpha * xAlpha) - (0.281 * xOA * xAlpha * xAlpha) - (0.342 * xHA * xHA * xAlpha) - (0.245 * xHA * xHA * xOA) + (0.281 * xOA * xOA * xHA) - (0.184 * xOPTT * xOPTT * xAlpha) - (0.281 * xHA * xAlpha * xOA)
 
         
        objs = torch.stack([f1,f2,f3]).T
        
        return objs
    
class RE41():
    def __init__(self, n_dim = 7):
        
      
        self.n_dim = n_dim
        self.n_obj = 4
        self.lbound = torch.tensor([0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4]).float()
        self.ubound = torch.tensor([1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2]).float()
        self.nadir_point = [39.2905121788, 4.42725, 13.09138125, 9.49401929991]
        
    def evaluate(self, x):
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.to(device)
            self.ubound = self.ubound.to(device)
        
        x = x * (self.ubound - self.lbound) + self.lbound
      
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        x4 = x[:,3]
        x5 = x[:,4]
        x6 = x[:,5]
        x7 = x[:,6]

        # First original objective function
        f1 = 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 0.00001 * x6 + 2.73 * x7
        # Second original objective function
        f2 = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
        # Third original objective function
        Vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
        Vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
        f3 = 0.5 * (Vmbp + Vfd)

        # Constraint functions
        g1 = 1 -(1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3)
        g2 = 0.32 -(0.261 - 0.0159 * x1 * x2 - 0.06486 * x1 -  0.019 * x2 * x7 + 0.0144 * x3 * x5 + 0.0154464 * x6)
        g3 = 0.32 -(0.214 + 0.00817 * x5 - 0.045195 * x1 - 0.0135168 * x1 + 0.03099 * x2 * x6 - 0.018 * x2 * x7  + 0.007176 * x3 + 0.023232 * x3 - 0.00364 * x5 * x6 - 0.018 * x2 * x2)
        g4 = 0.32 -(0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2 * x2)
        g5 = 32 -(28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7)
        g6 = 32 -(33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728)
        g7 =  32 -(46.36 - 9.9 * x2 - 4.4505 * x1)
        g8 =  4 - f2
        g9 =  9.9 - Vmbp
        g10 =  15.7 - Vfd


        g = torch.stack([g1,g2,g3,g4,g5,g6,g7,g8,g9,g10])
        z = torch.zeros(g.shape).float().to(device).to(torch.float64)
        g = torch.where(g < 0, -g, z)

        f4 = torch.sum(g, axis = 0).float() 
         
        objs = torch.stack([f1,f2,f3,f4]).T
        
        return objs
    
class RE42():
    def __init__(self, n_dim = 6):
        
      
        self.n_dim = n_dim
        self.n_obj = 4
        self.lbound = torch.tensor([150.0, 20.0 , 13.0 , 10.0, 14.0, 0.63]).float()
        self.ubound = torch.tensor([274.32, 32.31, 25.0, 11.71, 18.0, 0.75]).float()
        self.nadir_point = [-663.57644, 15690.493, 5179.22898, 3207.04559]
        
    def evaluate(self, x):
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.to(device)
            self.ubound = self.ubound.to(device)
        
        x = x * (self.ubound - self.lbound) + self.lbound
      
        constraintFuncs = torch.zeros(9).to(device)

        x_L = x[:,0]
        x_B = x[:,1]
        x_D = x[:,2]
        x_T = x[:,3]
        x_Vk = x[:,4]
        x_CB = x[:,5]
   
        displacement = 1.025 * x_L * x_B * x_T * x_CB
        V = 0.5144 * x_Vk
        g = 9.8065
        Fn = V / torch.pow(g * x_L, 0.5)
        a = (4977.06 * x_CB * x_CB) - (8105.61 * x_CB) + 4456.51
        b = (-10847.2 * x_CB * x_CB) + (12817.0 * x_CB) - 6960.32

        power = (torch.pow(displacement, 2.0/3.0) * torch.pow(x_Vk, 3.0)) / (a + (b * Fn))
        outfit_weight = 1.0 * torch.pow(x_L , 0.8) * torch.pow(x_B , 0.6) * torch.pow(x_D, 0.3) * torch.pow(x_CB, 0.1)
        steel_weight = 0.034 * torch.pow(x_L ,1.7) * torch.pow(x_B ,0.7) * torch.pow(x_D ,0.4) * torch.pow(x_CB ,0.5)
        machinery_weight = 0.17 * torch.pow(power, 0.9)
        light_ship_weight = steel_weight + outfit_weight + machinery_weight

        ship_cost = 1.3 * ((2000.0 * torch.pow(steel_weight, 0.85))  + (3500.0 * outfit_weight) + (2400.0 * torch.pow(power, 0.8)))
        capital_costs = 0.2 * ship_cost

        DWT = displacement - light_ship_weight

        running_costs = 40000.0 * torch.pow(DWT, 0.3)

        round_trip_miles = 5000.0
        sea_days = (round_trip_miles / 24.0) * x_Vk
        handling_rate = 8000.0

        daily_consumption = ((0.19 * power * 24.0) / 1000.0) + 0.2
        fuel_price = 100.0
        fuel_cost = 1.05 * daily_consumption * sea_days * fuel_price
        port_cost = 6.3 * torch.pow(DWT, 0.8)

        fuel_carried = daily_consumption * (sea_days + 5.0)
        miscellaneous_DWT = 2.0 * torch.pow(DWT, 0.5)
        
        cargo_DWT = DWT - fuel_carried - miscellaneous_DWT
        port_days = 2.0 * ((cargo_DWT / handling_rate) + 0.5)
        RTPA = 350.0 / (sea_days + port_days)

        voyage_costs = (fuel_cost + port_cost) * RTPA
        annual_costs = capital_costs + running_costs + voyage_costs
        annual_cargo = cargo_DWT * RTPA

        f1 = annual_costs / annual_cargo
        f2 = light_ship_weight
        # f_2 is dealt as a minimization problem
        f3 = -annual_cargo

        # Reformulated objective functions
        g1 = (x_L / x_B) - 6.0
        g2 = -(x_L / x_D) + 15.0
        g3 = -(x_L / x_T) + 19.0
        g4 = 0.45 * torch.pow(DWT, 0.31) - x_T
        g5 = 0.7 * x_D + 0.7 - x_T
        g6 = 500000.0 - DWT
        g7 = DWT - 3000.0
        g8 = 0.32 - Fn

        KB = 0.53 * x_T
        BMT = ((0.085 * x_CB - 0.002) * x_B * x_B) / (x_T * x_CB)
        KG = 1.0 + 0.52 * x_D
        g9 = (KB + BMT - KG) - (0.07 * x_B)
        
        g = torch.stack([g1,g2,g3,g4,g5,g6,g7,g8,g9])
        z = torch.zeros(g.shape).float().to(device).to(torch.float64)
        g = torch.where(g < 0, -g, z)

        f4 = torch.sum(g, axis = 0).float() 
         
        objs = torch.stack([f1,f2,f3,f4]).T
        
        return objs    
    
class RE61():
    def __init__(self, n_dim = 3):
        
      
        self.n_dim = n_dim
        self.n_obj = 6
        self.lbound = torch.tensor([0.01, 0.01, 0.01]).float()
        self.ubound = torch.tensor([0.45, 0.1, 0.1]).float()
        self.nadir_point = [80896.9128355, 1350.0, 2853468.96494, 7076861.67064, 87748.6339553, 2.50994535821]
        
    def evaluate(self, x):
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.to(device)
            self.ubound = self.ubound.to(device)
        
        x = x * (self.ubound - self.lbound) + self.lbound
      
        f = torch.zeros(6).to(device)
        g = torch.zeros(7).to(device)
        
        # First original objective function
        f1 = 106780.37 * (x[:,1] + x[:,2]) + 61704.67
        #Second original objective function
        f2 = 3000 * x[:,0]
        # Third original objective function
        f3 = 305700 * 2289 * x[:,1] / np.power(0.06*2289, 0.65)
        # Fourth original objective function
        f4 = 250 * 2289 * torch.exp(-39.75*x[:,1]+9.9*x[:,2]+2.74)
        # Fifth original objective function
        f5 = 25 * (1.39 /(x[:,0]*x[:,1]) + 4940*x[:,2] -80)

        # Constraint functions          
        g1 = 1 - (0.00139/(x[:,0]*x[:,1])+4.94*x[:,2]-0.08)
        g2 = 1 - (0.000306/(x[:,0]*x[:,1])+1.082*x[:,2]-0.0986)       
        g3 = 50000 - (12.307/(x[:,0]*x[:,1]) + 49408.24*x[:,2]+4051.02)
        g4 = 16000 - (2.098/(x[:,0]*x[:,1])+8046.33*x[:,2]-696.71)     
        g5 = 10000 - (2.138/(x[:,0]*x[:,1])+7883.39*x[:,2]-705.04)     
        g6 = 2000 - (0.417*x[:,0]*x[:,1] + 1721.26*x[:,2]-136.54)       
        g7 = 550 - (0.164/(x[:,0]*x[:,1])+631.13*x[:,2]-54.48) 

        g = torch.stack([g1,g2,g3,g4,g5,g6,g7])
        z = torch.zeros(g.shape).float().to(device).to(torch.float64)
        g = torch.where(g < 0, -g, z)

        f6 = torch.sum(g, axis = 0).float() 
         
        objs = torch.stack([f1,f2,f3,f4,f5,f6]).T
        
        return objs    
    
class RE91():
    def __init__(self, n_dim = 7):
        
      
        self.n_dim = n_dim
        self.n_obj = 9
        self.lbound = torch.tensor([0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4]).float()
        self.ubound = torch.tensor([1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2]).float()
        self.nadir_point = [39.6023250742, 1.00188125422, 112.487885728, 0.79017024474, 1.42304666576, 1.08576907833, 1.1215181762, 0.993535488298, 1.01068662645]
        
    def evaluate(self, x):
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.to(device)
            self.ubound = self.ubound.to(device)
        
        x = x * (self.ubound - self.lbound) + self.lbound
      
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        x4 = x[:,3]
        x5 = x[:,4]
        x6 = x[:,5]
        x7 = x[:,6]
        # stochastic variables
        x8 = 0.006 * (np.random.normal(0, 1)) + 0.345
        x9 = 0.006 * (np.random.normal(0, 1)) + 0.192
        x10 = 10 * (np.random.normal(0, 1)) + 0.0
        x11 = 10 * (np.random.normal(0, 1)) + 0.0

        # First function
        f1 = 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 +  4.01 * x4 +  1.75 * x5 +  0.00001 * x6  +  2.73 * x7
        # Second function
        f2 = (1.16 - 0.3717* x2 * x4 - 0.00931 * x2 * x10 - 0.484 * x3 * x9 + 0.01343 * x6 * x10 )/1.0
        f2[f2<0] = 0
        # Third function
        f3 = (0.261 - 0.0159 * x1 * x2 - 0.188 * x1 * x8 - 0.019 * x2 * x7 + 0.0144 * x3 * x5 + 0.87570001 * x5 * x10 + 0.08045 * x6 * x9 + 0.00139 * x8 * x11 + 0.00001575 * x10 * x11)/0.32
        f3[f3<0] = 0
        # Fourth function  
        f4 = (0.214 + 0.00817 * x5 - 0.131 * x1 * x8 - 0.0704 * x1 * x9 + 0.03099 * x2 * x6 - 0.018 * x2 * x7 + 0.0208 * x3 * x8 + 0.121 * x3 * x9 - 0.00364 * x5 * x6 + 0.0007715 * x5 * x10 - 0.0005354 * x6 * x10 + 0.00121 * x8 * x11 + 0.00184 * x9 * x10 - 0.018 * x2 * x2)/0.32
        f4[f4<0] = 0
        # Fifth function  
        f5 = (0.74 - 0.61* x2 - 0.163 * x3 * x8 + 0.001232 * x3 * x10 - 0.166 * x7 * x9 + 0.227 * x2 * x2)/0.32
        f5[f5<0] = 0
        # Sixth function       
        tmp = (( 28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 0.0207 * x5 * x10 + 6.63 * x6 * x9 - 7.77 * x7 * x8 + 0.32 * x9 * x10) + (33.86 + 2.95 * x3 + 0.1792 * x10 - 5.057 * x1 * x2 - 11 * x2 * x8 - 0.0215 * x5 * x10 - 9.98 * x7 * x8 + 22 * x8 * x9) + (46.36 - 9.9 * x2 - 12.9 * x1 * x8 + 0.1107 * x3 * x10) )/3     
        f6 =  tmp/32
        f6[f6<0] = 0
        # Seventh function           
        f7 =  (4.72 - 0.5 * x4 - 0.19 * x2 * x3 - 0.0122 * x4 * x10 + 0.009325 * x6 * x10 + 0.000191 * x11 * x11)/4.0
        f7[f7 < 0] = 0
        # EighthEighth function               
        f8 =  (10.58 - 0.674 * x1 * x2 - 1.95  * x2 * x8  + 0.02054  * x3 * x10 - 0.0198  * x4 * x10  + 0.028  * x6 * x10)/9.9
        f8[f8 <0] = 0
        # Ninth function               
        f9 =  (16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6 + 0.0432 * x9 * x10 - 0.0556 * x9 * x11 - 0.000786 * x11 * x11)/15.7
        f9[f9<0] = 0 		    
         
        objs = torch.stack([f1,f2,f3,f4,f5,f6,f7,f8,f9]).T
        
        return objs    