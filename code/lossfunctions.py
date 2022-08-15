import torch

class CauchyLoss(torch.nn.Module):
    def __init__(self,k=1,reduction='mean'):
        super(CauchyLoss,self).__init__()
        self.k = k
        self.reduction = reduction 
    def derive(self,x,y):
        diff = torch.sub(x,y)
        totloss = (2*(self.k**2)*diff)/(1+(self.k**2)*torch.pow(diff,2))
        return totloss
    def forward(self,x,y):
        size = y.size()[0]
        diff = torch.sub(x,y)
        totloss = torch.log(1+(self.k**2)*torch.pow(diff,2))
        if self.reduction=='mean':
            totloss= torch.sum(totloss/size)
        return totloss
    
    
class TukeyLoss(torch.nn.Module):
    def __init__(self,t=4.685,reduction='mean'):
        super(TukeyLoss,self).__init__()
        self.t = t
        self.reduction = reduction 
    def derive(self,x,y):
        diff = torch.sub(x,y)
        index = (torch.abs(torch.sub(x,y))<self.t).float()
        totloss = index*torch.mul(x,torch.pow(1-torch.pow(diff,2)/(self.t**2),2))+(1-index)*0
        return totloss
    def forward(self,x,y):
        size = y.size()[0]
        index = (torch.abs(torch.sub(x,y))<self.t).float()
        totloss = (1-torch.pow(1-torch.pow(torch.abs(torch.sub(x,y))/self.t,2),3))*(self.t)**2/6*index+(1-index)*(self.t)**2/6
        if self.reduction=='mean':
            totloss=torch.sum(totloss/size)
        return totloss
    
    
class LAD(torch.nn.Module):
    def __init__(self,reduction='mean'):
        super(LAD,self).__init__()
        self.reduction = reduction 
    def derive(self,x,y):
        diff = torch.sub(x,y)
        totloss = torch.sgn(diff)
        return totloss
    def forward(self,x,y):
        size = y.size()[0]
        diff = torch.sub(x,y)
        totloss = torch.abs(diff)
        if self.reduction=='mean':
            totloss=torch.sum(totloss/size)
        return totloss
    
class Huber(torch.nn.Module):
    def __init__(self,tau=1,reduction='mean'):
        super(Huber,self).__init__()
        self.reduction = reduction 
        self.tau = tau
    def derive(self,x,y):
        diff = torch.sub(x,y)
        index = (torch.abs(diff)<self.tau).float()
        totloss = torch.mul(index,diff)*(self.tau**2)+(1-index)*(self.tau*torch.sign(diff))
        return totloss
    def forward(self,x,y):
        size = y.size()[0]
        diff = torch.sub(x,y)
        index = (torch.abs(diff)<self.tau).float()
        totloss = index*torch.pow(diff,2)*(self.tau**2)/2+(1-index)*(self.tau*torch.abs(diff)-(self.tau**2)/2)
        if self.reduction=='mean':
            totloss=torch.sum(totloss/size)
        return totloss
    
    