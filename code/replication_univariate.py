from generate import gen_univ
from dnn import DNN
from lossfunctions import CauchyLoss,TukeyLoss
import torch
from qreg import QRegressor
from pyearth import Earth
from DRR import DRR
import gc
#%% Input the sample size, simulated model, error distribution and other parameters
SIZE=2**9; model='heavisine';error='cauchy';replication=100
# Methods of regression
methods=['ls','L1','huber','cauchy','tukey','KQR','MARS']

# Loss functions
loss_ls = torch.nn.MSELoss(reduction='mean')  
loss_L1 = torch.nn.L1Loss(reduction='mean')   
loss_huber = torch.nn.SmoothL1Loss(reduction='mean',beta=1.345) 
loss_cauchy = CauchyLoss(k=1,reduction='mean')
loss_tukey = TukeyLoss(t=4.685,reduction='mean')

# Statistics
stat_ls=torch.zeros([replication,len(methods)],requires_grad=False); # Collecting square losses on the testing data
stat_LAD=torch.zeros([replication,len(methods)],requires_grad=False); # Collecting LAD losses on the testing data
stat_huber=torch.zeros([replication,len(methods)],requires_grad=False); # Collecting Huber losses on the testing data
stat_cauchy=torch.zeros([replication,len(methods)],requires_grad=False); # Collecting Cauchy losses on the testing data
stat_tukey=torch.zeros([replication,len(methods)],requires_grad=False); # Collecting Tukey losses on the testing data
stat_L1=torch.zeros([replication,len(methods)],requires_grad=False);    # Collecting L1 distances (to the target) on the testing data
stat_L2=torch.zeros([replication,len(methods)],requires_grad=False);    # Collecting L2 distances (to the target) on the testing data

#%% Replications of Training 
for r in range(replication):
    print('The %d replication.\n'%r)
    data = gen_univ(model=model,size=SIZE,error=error,df=2,sigma=1); # Training data
    data_test = gen_univ(model=model,size=10000,error=error,df=2,sigma=1); # Testing data
    x=data[:][0];y=data[:][1];
    x_test=data_test[:][0];y_test=data_test[:][1];
    fx_test=y_test-data_test[:][2].data.numpy();
    for (k,method) in enumerate(methods):
        if (method!='KQR') and (method!='MARS'):
            net=DNN(width_vec=[1,256,256,256,256,256,1]);
            net=DRR(net=net,data=data,loss=method,tunepara=None,epoch=1500,batch_size=None,stopping='early',min_epoch=400,patience=50,lr=1e-3)
            y_drr=net(data_test[:][0]);
            del net
            gc.collect();
            stat_ls[r,k]=(loss_ls(y_test,y_drr)-loss_ls(y_test,fx_test)).detach();
            stat_LAD[r,k]=(loss_L1(y_test,y_drr)-loss_L1(y_test,fx_test)).detach();
            stat_huber[r,k]=(loss_huber(y_test,y_drr)-loss_huber(y_test,fx_test)).detach();
            stat_cauchy[r,k]=(loss_cauchy(y_test,y_drr)-loss_cauchy(y_test,fx_test)).detach();
            stat_tukey[r,k]=(loss_tukey(y_test,y_drr)-loss_tukey(y_test,fx_test)).detach();
            stat_L1[r,k]=(torch.mean(torch.abs(y_drr-fx_test))).detach();
            stat_L2[r,k]=(torch.mean(torch.pow(y_drr-fx_test,2))).detach();
            del y_drr
            gc.collect();
        elif method=='KQR':
            reg = QRegressor(C=1e4,  # Trade-off parameter
                             probs=[0.5],  # Quantile levels
                             gamma_out=0.,  # Inner kernel parameter
                             #gamma_in=1.,
                             eps=0.,  # Epsilon-loss level
                             stepsize_factor=10.,
                             kernel='rbf', #nput kernel ('rbf' or 'linear')
                             alg='sdca',  # Algorithm (can change to 'qp')
                             tol=1e-7,
                             max_iter=1e5,  # Maximal number of iteration
                             active_set=True,  # Active set strategy
                             verbose=True)
            reg.fit(x.numpy().astype(float), y.numpy().astype(float));
            y_kqr=torch.from_numpy(reg.predict(data_test[:][0].squeeze()).T)
            del reg
            gc.collect();
            stat_ls[r,k]=(loss_ls(y_test,y_kqr)-loss_ls(y_test,fx_test)).detach();
            stat_LAD[r,k]=(loss_L1(y_test,y_kqr)-loss_L1(y_test,fx_test)).detach();
            stat_huber[r,k]=(loss_huber(y_test,y_kqr)-loss_huber(y_test,fx_test)).detach();
            stat_cauchy[r,k]=(loss_cauchy(y_test,y_kqr)-loss_cauchy(y_test,fx_test)).detach();
            stat_tukey[r,k]=(loss_tukey(y_test,y_kqr)-loss_tukey(y_test,fx_test)).detach();
            stat_L1[r,k]=(torch.mean(torch.abs(y_kqr-fx_test))).detach();
            stat_L2[r,k]=(torch.mean(torch.pow(y_kqr-fx_test,2))).detach();
            del y_kqr
            gc.collect()
        elif method=='MARS':
            spline = Earth();spline.fit(x.numpy().astype(float), y.numpy().astype(float));
            y_mars = torch.from_numpy(spline.predict(data_test[:][0]).reshape(-1,1));
            del spline
            gc.collect();
            stat_ls[r,k]=(loss_ls(y_test,y_mars)-loss_ls(y_test,fx_test)).detach();
            stat_LAD[r,k]=(loss_L1(y_test,y_mars)-loss_L1(y_test,fx_test)).detach();
            stat_huber[r,k]=(loss_huber(y_test,y_mars)-loss_huber(y_test,fx_test)).detach();
            stat_cauchy[r,k]=(loss_cauchy(y_test,y_mars)-loss_cauchy(y_test,fx_test)).detach();
            stat_tukey[r,k]=(loss_tukey(y_test,y_mars)-loss_tukey(y_test,fx_test)).detach();
            stat_L1[r,k]=(torch.mean(torch.abs(y_mars-fx_test))).detach();
            stat_L2[r,k]=(torch.mean(torch.pow(y_mars-fx_test,2))).detach();
            del y_mars
            gc.collect();
    del data,data_test,x,y,x_test,y_test,fx_test
    gc.collect()
            

#%% Print the statistics
torch.set_printoptions(precision=2,sci_mode=False)

print(stat_ls.detach()) 
print(stat_LAD.detach()) 
print(stat_huber.detach())
print(stat_cauchy.detach()) 
print(stat_tukey.detach()) 
print(stat_L1.detach()) 
print(stat_L2.detach()) 

