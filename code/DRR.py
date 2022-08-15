#%% Loading packages
import torch
import torch.utils.data as Data
from lossfunctions import CauchyLoss,TukeyLoss
from torch.autograd import Variable
from tqdm import tqdm
import gc
#%%
def DRR(net,data,loss='huber',tunepara=None,epoch=1000,batch_size=None,stopping='early',min_epoch=400,patience=100,lr=1e-3):
    """
    # net: network to be trained
    # data: the traiding dataset including x & y, in the format that packed by function 'torch.utils.data.Data.TensorDataset'
    # loss: the loss function used to train the network, including 'ls', 'L1','huber','cauchy','tukey
    # tunepara: tuning parameter in the indicated robust loss function, only valid for those loss functions with a tuning paramter
    # epoch: please input an integer to indicate how many epochs you would like to train the network
    # batch_size: the batch size during training using variant minibatch SGD based optimization methods 
    # stopping: the stopping rule of the training, including 'normal':train the network for 'epoch' times and stop, 'early': early stopping based on loss values
    # min_epoch: only valid for 'early stopping', the minimal times of ieteration of the training
    # patience: only valid for 'early stopping', if the training loss in previous 'patience' epochs (start from min_epoch) did not achieve a new minimum, then stop the training
    # lr: learning rate in the training using 'Adam' algorithm with learning rate 0.001 as default
    """
    # Default parameter settings
    if tunepara==None:
        if loss=='huber':
            tunepara=1.345;
        elif loss=='cauchy':
            tunepara=1;
        elif loss=='tukey':
            tunepara=4.685;
    if batch_size==None:
        batch_size=int(data[:][0].size()[0]/4) # Default batch size is n/4, where n is sample size
    loader = Data.DataLoader(dataset=data, batch_size=batch_size,shuffle=True, num_workers=0,)
    #Choosing Loss function according to loss
    losses={'ls':torch.nn.MSELoss(reduction='mean'), # this is for regression mean squared loss
                'L1':torch.nn.L1Loss(reduction='mean'),   # this is for regression least absolute deviation loss
                'huber':torch.nn.SmoothL1Loss(beta=tunepara,reduction='mean'), # this is for regression Huber loss
                'cauchy': CauchyLoss(k=tunepara,reduction='mean'), # this is for regression CauchyLoss loss
                'tukey': TukeyLoss(t=tunepara,reduction='mean'), # this is for regression TukeyLoss loss
            }
    lossfunction=losses[loss]
    optimizer = torch.optim.Adam(net.parameters(), lr=lr) #Using Adam to train the network
    loss_values=10000*torch.ones([1,1]);trigger_list=torch.zeros([1,1]);trigger_times = 0
    for t in tqdm(range(epoch),total=epoch):
        for step, (x, y,_) in enumerate(loader):
            net.train()
            x, y = Variable(x.float()), Variable(y.float())
            prediction = net(x)     # input x and predict based on x
            loss_value= lossfunction(prediction, y)     # must be (1. nn output, 2. target)
            optimizer.zero_grad()   # clear gradients for next train
            loss_value.backward()   # backpropagation, compute gradients
            optimizer.step()        # apply gradients
            net.eval()
            # Early stopping    
        if stopping=='early':
            loss_current=lossfunction(net(data[:][0]),data[:][1])
            loss_values=torch.cat((loss_values,loss_current.unsqueeze(0).unsqueeze(1)),0)
            loss_min=loss_values.min()
            if t > min_epoch:
                if loss_current >loss_min:
                    trigger_list = torch.cat((trigger_list,torch.ones([1,1])),0)
                    trigger_times = trigger_list[-patience:].sum()
                    if trigger_times >= patience:
                        print('Early stopping!')
                        break
                        return (net,loss_values)
                else:
                    trigger_list=torch.cat((trigger_list,torch.zeros([1,1])),0)
                    trigger_times = trigger_list[-patience:].sum()
    del data,tunepara,loss,batch_size,lossfunction,t,loss_values,x,y,optimizer,prediction,trigger_list,loader,losses,stopping,epoch,min_epoch,patience,lr
    gc.collect()
    return net