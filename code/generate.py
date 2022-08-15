import numpy as np
import torch
import torch.utils.data as Data

# Univariate functions
# Self define basic functions for data generating
def block(input):
    t=torch.FloatTensor([0.1,0.15,0.23,0.28,0.40,0.44,0.65,0.76,0.78,0.81])
    h=torch.FloatTensor([4,-5,-2.5,4,-3,2.1,4.3,-1.1,-2.1,-4.2])
    fx=torch.sum(torch.mul((torch.sgn(torch.sub(input,t))+1)/2,h),1)
    output=torch.FloatTensor(fx.reshape(-1,1))
    return output

def bump(input):
    t=torch.FloatTensor([0.1,0.15,0.23,0.28,0.40,0.44,0.65,0.76,0.78,0.81])
    h=torch.FloatTensor([4,5,2.5,4,3,2.1,4.3,1.1,2.1,4.2])
    w=2*torch.FloatTensor([0.005,0.005,0.006,0.01,0.01,0.03,0.01,0.01,0.005,0.008])
    fx=torch.sum(torch.mul(torch.pow(torch.abs(torch.sub(input,t))/w+1,-4),h),1)
    output=torch.FloatTensor(fx.reshape(-1,1))
    return output

def heavisine(input):
    fx=4*torch.sin(4*np.pi*input)-torch.sgn(input-0.3)-torch.sgn(0.72-input)
    output=torch.FloatTensor(fx.reshape(-1,1))
    return output


def dopller(input):
    fx=10*torch.mul(torch.pow(torch.mul(input,1-input),0.5),torch.sin(2*np.pi*(1+0.1)/(input+0.15)))
    output=torch.FloatTensor(fx.reshape(-1,1))
    return output


def gen_univ(model='block',size=2**10,error='t',df=2,sigma=1):
    """
    # model: univariate models including 'block','bump','heavisine','dopller'
    # size: sample size
    # error: error type including 't','normal','cauchy','mix'
    # df: degree of freedoom of the error distribution (only valid for 't' distribution)
    # sigma: a scalar multipling the error term, used to control the error variance
    """
    ind=torch.bernoulli(0.8*torch.ones([size,1]))
    errors={'t':torch.from_numpy(np.random.standard_t(df,[size,1])),
            'normal':torch.randn([size,1]),
            'cauchy':torch.from_numpy(np.random.standard_cauchy([size,1])),
            'mix':ind*torch.randn([size,1])+100*(1-ind)*torch.randn([size,1])
        }
    eps=errors[error].float()
    x = torch.rand([size,1]).float()
    models={'block':block(x),'bump':bump(x),'heavisine':heavisine(x),'dopller':dopller(x)}
    fx=5*models[model].float()
    y=fx+sigma*eps
    return Data.TensorDataset(x, y, sigma*eps)
    

#%% Mutivariate functions
# Self define basic functions for data generating
def my_linear(input,a=-2.2,b=0.3):
    output=a*torch.FloatTensor(input)+b
    return output

def my_cubic(input,a=0.7,b=-0.2,c=0.3,d=-0.3):
    input=torch.FloatTensor(input)
    output=a*(input**3)+b*(input**2)+c*(input)+d
    return output

def my_root(input,a=0.3):
    sign=torch.sign(input)
    output=(sign*a)*(sign*input)**(1/2)
    return output

def my_log(input,a=0.8,b=0.01):
    output=a*torch.log(torch.abs(input)+b)
    return output

def my_exp(input,a=0.2,b=-0.1):
    output=a*torch.exp(torch.min(input+b,torch.Tensor([4])))
    return output

def my_sin(input,a=6.28):
    output=torch.sin(a*input)
    return output

def my_inv(input,a=0.5,b=0.05):
    output=torch.pow(a*torch.abs(input)+b,-1)
    return output




def gen_multi(model='KA',d=2**3,size=2**10,error='t',df=2,sigma=1):
    """
    # model: multivariate models,including 'KA','linear','addtive2'
    # size: sample size
    # error: error type including 't','normal','cauchy','mix'
    # df: degree of freedoom of the error distribution (only valid for 't' distribution)
    # sigma: a scalar multipling the error term, used to control the error variance
    """
    ind=torch.bernoulli(0.8*torch.ones([size,1]))
    errors={'t':torch.from_numpy(np.random.standard_t(df,[size,1])),
            'normal':torch.from_numpy(np.random.standard_normal([size,1])),
            'cauchy':torch.from_numpy(np.random.standard_cauchy([size,1])),
            'mix':ind*torch.randn([size,1])+100*(1-ind)*torch.randn([size,1])
        }
    eps=errors[error].float()
    x = torch.rand([size,d]).float()
    if model=='KA':
        func_list=[my_linear,my_inv,my_exp,my_sin,my_log,my_cubic,my_root,]
        torch.manual_seed(2021)     # reproducible
        index_psi = torch.multinomial(torch.ones([2*d+1,7]),d,replacement=True);
        index_g=torch.multinomial(torch.ones([7]),2*d+1,replacement=True);
        y_1=torch.zeros([size,2*d+1])
        y=torch.zeros([size,1])
        for i in range(2*d+1):
            for j in range(d):
                y_1[:,i]=y_1[:,i]+func_list[index_psi[i,j]](x[:,j])
                y=y+func_list[index_g[i]](y_1[:,i]).reshape([size,1])
        y=y+sigma*eps
    if model=='linear':
        torch.manual_seed(2021)     # reproducible
        A=torch.randn([d,1])
        y=torch.mm(x,A)+sigma*eps  
    if model=='additive2':
        x[:,5:]=0.5;
        x_eps=0.01*torch.rand([size,d]);
        x=x+x_eps;
        func_list=[my_linear,my_inv,my_exp,my_sin,my_log,my_cubic,my_root,]
        torch.manual_seed(2021)     # reproducible
        index1= torch.multinomial(torch.ones([7]),d,replacement=True);
        index2=torch.multinomial(torch.ones([7]),d,replacement=True);
        y1=torch.zeros([size,1]);
        y2=torch.zeros([size,1]);
        for i in range(d):
            y1=y1+func_list[index1[i]](x[:,i].reshape([size,1]))
        for i in range(d):
            y2=y2+func_list[index2[i]](y1).reshape([size,1]);
        y=y2+sigma*eps   
    return Data.TensorDataset(x, y, sigma*eps)


