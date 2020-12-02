# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:57:08 2019

@author: Office
"""

import csv
from six.moves import cPickle as pickle
import numpy as np

def csv2pickle1():
    x = []
    with open('/home/hossam/work/AIRL/inverse_rl/data/Traj_df_FBF_interaction2_to_python_complete_training.csv','r') as f:
        reader = csv.reader(f)
        for line in reader: x.append(line)
        
    for i in range(1,len(x)):
        for j in range(4,14):    
            x[i][j] = float(x[i][j])
    
    Obj_ID = ['']       
    for i in range(1,len(x)):
        Obj_ID = np.vstack((Obj_ID, x[i][3])) 
        
    Obj_ID_unique = np.unique(Obj_ID)
    indexes = np.unique(Obj_ID, return_index=True)[1]
    Obj_ID_unique= [Obj_ID[index] for index in sorted(indexes)]
    
    len_obj = len(Obj_ID_unique)
    
    Obj_ID_unique = Obj_ID_unique[1:len_obj]
    
    expert_training = []
    example = {'observations': x[1][4:12], 'actions':x[1][12:14]}
    #example = {'observations': x[1][4:12], 'actions':x[1][13:14], 'agent_infos': {'log_std': [0,0] , 'mean': [0,0]}, 'a_logprobs' : [0]}
    
    
    j = 0
    
    for i in range(2,len(x)):
        obj = Obj_ID_unique[j][0]
        if x[i][3] == obj:
    
            
            example['observations'] = np.vstack((example['observations'], x[i][4:12]))
            example['actions'] = np.vstack((example['actions'], x[i][12:14]))
        elif j < len(Obj_ID_unique)-1:
            j = j+1
            expert_training.append(example)
            example = {'observations': x[i][4:12], 'actions':x[i][12:14]}
        else:
            break
        
    
        
    expert_training.pop(93)
    expert_training.pop(117)
    expert_training.pop(139)
    expert_training.pop(158)
    expert_training.pop(222)
    expert_training.pop(296)
    expert_training.pop(328)
    expert_training.pop(356)
    expert_training.pop(431)
    expert_training.pop(457)
    expert_training.pop(490)
    expert_training.pop(580)
    expert_training.pop(116)
    expert_training.pop(136)
    expert_training.pop(153)
    expert_training.pop(215)
    expert_training.pop(287)
    expert_training.pop(317)
    
    expert_training.pop(343)
    
    expert_training.pop(416)
    expert_training.pop(440)
    expert_training.pop(471)
    expert_training.pop(559)
    
    return expert_training
        
        
        

#with open('D:/github_clones/data pickle/Traj_df_FBF_interaction2_to_python_complete_training_ungroup.p','wb') as f:
#    pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)
#    
#    
#    
#    
#x = []
#with open('D:/github_clones/data pickle/Traj_df_FBF_interaction2_to_python_complete_test_ungroup.csv','r') as f:
#    reader = csv.reader(f)
#    for line in reader: x.append(line)
#
#with open('D:/github_clones/data pickle/Traj_df_FBF_interaction2_to_python_complete_test_ungroup.p','wb') as f:
#    pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)