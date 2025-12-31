"""
Runing the proposed Evolutionary Paret Set Learning (EPSL) method on 16 test problems.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 

import numpy as np
import torch

import schedulefree

from matplotlib import pyplot as plt

from pymoo.indicators.hv import HV
from pymoo.util.ref_dirs import get_reference_directions

from problem import get_problem

def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)
            
def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []
        ref_dir = np.full(n_dim, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)
    
    
hv_value_true_dict = {'re21':0.888555388212816, 're22':0.762745852861501, 're23':1.1635525932784616, 're24':1.171256434424709, 're25':1.0811754149670405,
                      're31':1.330998828712509, 're32':1.3306416000738288, 're33':1.014313740464521, 're34': 1.0505616850845179,
                      're35':1.3032487679703135, 're36':1.0059624890382002, 're37':0.8471959081902014,
                      're41':0.8213476947317269, 're42':0.9223397435426084, 're61':1.183, 're91':0.0662,
                      'zdt1': 0.87661645, 'zdt2': 0.5432833299998329, 'zdt3': 1.331738735248482,
                      'f1':0.87661645, 'f2':0.87661645, 'f3':0.87661645, 'f4':0.87661645, 'f5':0.87661645, 
                      'syn':0.87661645}


ins_list = ['re21', 're22', 're23','re24','re25', 're31', 're32', 're33','re34','re35', 're36','re37','re41', 're42', 're61', 're91']


# number of independent runs
n_run = 1 


# PSL 
# number of learning steps
n_steps = 2000 
# number of sampled solutions for gradient estimation
n_sample = 5 
n_pref_update = 10
sampling_method = 'Bernoulli-Shrinkage'

# device
device = 'cpu'
# -----------------------------------------------------------------------------

model_type = 'normal' 

# basic EPSL model
if model_type == 'normal':
    from model import ParetoSetModel


elif model_type == 'variable_shared_component_syn':  # relation model for SYN (e.g., 'syn' in ins_list)
    from model_shared_component_syn import ParetoSetModel 
elif model_type == 'variable_shared_component_re':  # relation model for RE21 (e.g., 're21' in ins_list)
    from model_shared_component import ParetoSetModel 
    
elif model_type == 'variable_relation':  # relation model for RE21 (e.g., 're21' in ins_list)
    from model_variable_relation import ParetoSetModel 
    
elif model_type == 'key_point':
    from model_keypoint import ParetoSetModel

# testing ref
r = np.linspace(start=0, stop=1,num=5)
ref_vec_test = torch.tensor(np.array([1-r, r])).T.to(device).float()

hv_gap_list = {}
hv_large_set_gap_list = {}

for test_ins in ins_list:
    print(test_ins)
    hv_gap_list[test_ins] = []
    hv_large_set_gap_list[test_ins] = []
    
    if test_ins in ['re21', 're22', 're23','re24','re25', 're31', 're32', 're33','re34','re35', 're36','re37','re41', 're42', 're61', 're91']:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'data/RE/ParetoFront/{test_ins}.dat')
        pf = np.loadtxt(file_path)
        ideal_point = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'data/RE/ideal_nadir_points/ideal_point_{test_ins}.dat'))
        nadir_point = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'data/RE/ideal_nadir_points/nadir_point_{test_ins}.dat'))
    else:
        ideal_point = np.zeros(2)
        nadir_point = np.ones(2)

    # get problem info
    hv_value_true = hv_value_true_dict[test_ins]
    hv_all_value = np.zeros([n_run, n_steps])
    problem = get_problem(test_ins)
    n_dim = problem.n_dim
    n_obj = problem.n_obj
    

    ref_point = problem.nadir_point
    ref_point = [1.1*x for x in ref_point]
    ref_point = torch.Tensor(ref_point).to(device)
    
    for run_iter in range(n_run):
        
        # intitialize the model and optimizer 
        psmodel = ParetoSetModel(n_dim, n_obj)
        psmodel.to(device)
            
        # optimizer
        optimizer = schedulefree.AdamWScheduleFree(psmodel.parameters(), lr=0.0025, warmup_steps = 10)
       
        z = torch.ones(n_obj).to(device) 
        
        # EPSL steps
        for t_step in range(n_steps):
            psmodel.train()
            optimizer.train()
            
            sigma = 0.01 
            
            # sample n_pref_update preferences
            alpha = np.ones(n_obj)
            pref = np.random.dirichlet(alpha,n_pref_update)
            pref_vec  = torch.tensor(pref).to(device).float() 
            
            # get the current coressponding solutions
            x = psmodel(pref_vec)
           
            grad_es_list = []
            

            for k in range(pref_vec.shape[0]):
                
                # Sampling
                if sampling_method == 'Gaussian':
                    delta = torch.randn(n_sample, n_dim).to(device).double()
                    
                if sampling_method == 'Bernoulli':
                    delta = (torch.bernoulli(0.5 * torch.ones(n_sample, n_dim)) - 0.5) / 0.5
                    delta = delta.to(device).double()
                    
                if sampling_method == 'Bernoulli-Shrinkage':
                    m = np.sqrt((n_sample + n_dim - 1) / (4 * n_sample))
                    delta = (torch.bernoulli(0.5 * torch.ones(n_sample, n_dim)) - 0.5) / m
                    delta = delta.to(device).double()
                
                
                x_plus_delta = x[k] + sigma * delta
                delta_plus_fixed = delta
              
              
                x_plus_delta[x_plus_delta > 1] = 1
                x_plus_delta[x_plus_delta < 0] = 0
                
                value_plus_delta = problem.evaluate(x_plus_delta) 
                
                ideal_point_tensor = torch.tensor(ideal_point).to(device)
                value_plus_delta = (value_plus_delta - ideal_point_tensor) / (ref_point - ideal_point_tensor)
              
                z =  torch.min(torch.cat((z.reshape(1,n_obj), value_plus_delta - 0.1)), axis = 0).values.data
                
                # STCH
                u = 0.1 
                tch_value = u * torch.logsumexp( (1/pref_vec[k]) * torch.abs(value_plus_delta  - z) / u , axis = 1)
                tch_value = tch_value.detach()
              
               
                rank_idx = torch.argsort(tch_value)
                tch_value_rank = torch.ones(len(tch_value)).to(device)
                tch_value_rank[rank_idx] = torch.linspace(-0.5, 0.5, len(tch_value)).to(device)
            
                grad_es_k =  1.0 / (n_sample * sigma) * torch.sum(tch_value_rank.reshape(len(tch_value),1)  * delta_plus_fixed , axis = 0)
                grad_es_list.append(grad_es_k)
            
            grad_es = torch.stack(grad_es_list)
            
          
            # gradient-based pareto set model update 
            optimizer.zero_grad()
            psmodel(pref_vec).backward(grad_es)
            optimizer.step()  
           

        psmodel.eval()
        optimizer.eval()
            
        with torch.no_grad():
            
            if n_obj == 2:
                pref_size = 100
                pref = np.stack([np.linspace(0,1,100), 1 - np.linspace(0,1,100)]).T
                pref = torch.tensor(pref).to(device).float()
            
            if n_obj == 3:
                
                pref_size = 105 
                pref = torch.tensor(das_dennis(13,3)).to(device).float()  # 105
                #pref = torch.tensor(das_dennis(43,3)).to(device).float()  # 990
                #pref = torch.tensor(das_dennis(44,3)).to(device).float()   # 1035
                #pref  = torch.tensor(das_dennis(140,3)).to(device).float()   # 10011
                
            if n_obj == 4:
                pref_size = 120 
                pref = torch.tensor(das_dennis(7,4)).to(device).float()   # 120
                #pref = torch.tensor(das_dennis(16,4)).to(device).float()   # 969
               
                
            if n_obj == 6:
                pref_size = 182
                pref = torch.tensor(get_reference_directions("multi-layer",
                            get_reference_directions("das-dennis", 6, n_partitions=4, scaling=1.0),
                            get_reference_directions("das-dennis", 6, n_partitions=3, scaling=0.5))).to(device).float() 
                # # # 504
                # pref = torch.tensor(get_reference_directions("multi-layer",
                #             get_reference_directions("das-dennis", 6, n_partitions=5, scaling=1.0),
                #             get_reference_directions("das-dennis", 6, n_partitions=5, scaling=0.5))).to(device).float() 
                # # 923
                # pref = torch.tensor(get_reference_directions("multi-layer",
                #             get_reference_directions("das-dennis", 6, n_partitions=6, scaling=1.0),
                #             get_reference_directions("das-dennis", 6, n_partitions=6, scaling=0.5))).to(device).float() 
                
            
            if n_obj == 9:
                pref_size = 90
                
                # 90
                pref = torch.tensor(get_reference_directions("multi-layer",
                    get_reference_directions("das-dennis", 9, n_partitions=2, scaling=1.0),
                    get_reference_directions("das-dennis", 9, n_partitions=2, scaling=0.5))).to(device).float() 
                # # 210
                # pref = torch.tensor(get_reference_directions("multi-layer",
                #     get_reference_directions("das-dennis", 9, n_partitions=3, scaling=1.0),
                #     get_reference_directions("das-dennis", 9, n_partitions=2, scaling=0.5))).to(device).float() 
        
            sol = psmodel(pref)
            obj = problem.evaluate(sol)
            generated_ps = sol.cpu().numpy()
            generated_pf = obj.cpu().numpy()
            
            ### HV calculation ###
            results_F_norm = (generated_pf - ideal_point) / (nadir_point - ideal_point) 
            
            hv = HV(ref_point=np.array([1.1] * n_obj))
            
            hv_value = hv(results_F_norm)
            hv_gap_value = hv_value_true - hv_value
            hv_gap_list[test_ins].append(hv_gap_value)
        
            print("hv_gap", "{:.2e}".format(hv_gap_value))
           
            if run_iter == (n_run - 1):
                print("hv_gap_mean", "{:.2e}".format(np.mean(hv_gap_list[test_ins])))
            
            if n_obj == 2:
            
                fig = plt.figure()
                
                if test_ins in ['re21', 're22', 're23', 're24', 're25', 're31','re32','re33', 're34','re35','re36','re37']:
                    plt.scatter(pf[:,0],pf[:,1],c = 'k',  marker = '.', s = 2, alpha = 1, label ='Pareto Front', zorder = 1)
              
                plt.plot(generated_pf[:,0],generated_pf[:,1], c = 'tomato',  alpha = 1, lw = 5, label='EPSL', zorder = 2)
                
                plt.xlabel(r'$f_1(x)$',size = 16)
                plt.ylabel(r'$f_2(x)$',size = 16)
            
             
                handles = []
                pareto_front_label = plt.Line2D((0,1),(0,0), color='k', marker='o', linestyle='', label = 'Pareto Front')
                epsl_label = plt.Line2D((0,1),(0,0), color='tomato', marker='o', lw = 2, label = 'EPSL')
                
                handles.extend([pareto_front_label])
                handles.extend([epsl_label])
                
                
                order = [0,1]
                plt.legend(fontsize = 14)
                plt.legend(handles=handles,fontsize = 14, scatterpoints=3,
                        bbox_to_anchor = (1, 1))
                plt.grid()
               
                
            if n_obj == 3:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                
            
                ax.scatter(generated_pf[:,0],generated_pf[:,1],generated_pf[:,2], c = 'tomato', s =10, label = 'STCH')
                max_lim = np.max(generated_pf, axis = 0)
                min_lim = np.min(generated_pf, axis = 0)
                
                ax.set_xlim(min_lim[0], max_lim[0])
                ax.set_ylim(max_lim[1],min_lim[1])
                ax.set_zlim(min_lim[2], max_lim[2])
                
            
                ax.set_xlabel(r'$f_1(x)$',size = 12)
                ax.set_ylabel(r'$f_2(x)$',size = 12)
                ax.set_zlabel(r'$f_3(x)$',size = 12)
                
                plt.legend(loc=1, bbox_to_anchor=(1,1))