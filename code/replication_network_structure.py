from generate import gen_univ
from dnn import DNN
#from lossfunctions import CauchyLoss,TukeyLoss
import torch
#import numpy as np

from DRR import DRR
import gc

#%% Input the sample size, simulated model, error distribution and other parameters
SIZE=2**9; model='block';error='cauchy';replication=100

# Network strutures
S=(256**2)*4  # total number of parameters of the neural network
L=8           # Hidden layer for the deepest network is 2L
width_vec_list=[[int((S/(2*x+1))**(1/2))]*(2*x+2) for x in range(int(L))] # List of network structures


# Loss functions
#loss_ls = torch.nn.MSELoss(reduction='mean')  
loss_L1 = torch.nn.L1Loss(reduction='mean')  

# Statistics
#stat_ls=torch.zeros([replication,L],requires_grad=False); # Collecting square losses on the testing data
stat_L1=torch.zeros([replication,L],requires_grad=False);

#%% Replications of Training 

for width_vec in width_vec_list:
    width_vec.insert(0,1);width_vec.append(1)

for r in range(replication):
    print('The %d replication.\n'%r)
    data = gen_univ(model=model,size=SIZE,error=error,df=2,sigma=1); # Training data
    data_test = gen_univ(model=model,size=10000,error=error,df=2,sigma=1); # Testing data
    x=data[:][0];y=data[:][1];
    x_test=data_test[:][0];y_test=data_test[:][1];
    fx_test=y_test-data_test[:][2].data.numpy();
    for (l,width_vec) in enumerate(width_vec_list):
            net=DNN(width_vec)
            net=DRR(net=net,data=data,loss='L1',tunepara=None,epoch=1500,batch_size=None,stopping='early',min_epoch=400,patience=100,lr=1e-3)
            y_drr=net(data_test[:][0]);
            del net
            gc.collect();
            stat_L1[r,l]=(loss_L1(y_test,y_drr)-loss_L1(y_test,fx_test)).detach();
            del y_drr
            gc.collect();
    
#%% Print the results
torch.set_printoptions(precision=2,sci_mode=False)

print(stat_ls)
        
print((stat_ls.mean(0)))

print((stat_ls.std(0)))
