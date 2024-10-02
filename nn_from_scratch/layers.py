import numpy as np
from .module import Module

class Linear(Module):
    """
    A module which applies a linear transformation 
    A common name is fully-connected layer, InnerProductLayer in caffe. 
    
    The module should work with 2D input of shape (n_samples, n_feature).
    """
    def __init__(self, n_in, n_out, W=None, b=None):
        super(Linear, self).__init__()
       
        # This is a nice initialization
        stdv = 1./np.sqrt(n_in)
        if W is None:
            self.W = np.random.uniform(-stdv, stdv, size = (n_out, n_in)) #why not the opposite sizes
        else: 
            self.W = W
        if b is None:
            self.b = np.random.uniform(-stdv, stdv, size = n_out)
        else:
            self.b = b 
        
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)
        
    def updateOutput(self, input):

        self.output = np.add(input @ self.W.T, self.b)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput @ self.W
        return self.gradInput
    
    def accGradParameters(self, input, gradOutput):
        self.gradW =  gradOutput.T @ input
        self.gradb = np.ones(input.shape[0]) @ gradOutput
        return self.gradW, self.gradb
    
    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)
        
    def getParameters(self):
        return [self.W, self.b]
    
    def getGradParameters(self):
        return [self.gradW, self.gradb]
    
    def __repr__(self):
        s = self.W.shape
        q = 'Linear %d -> %d' %(s[1],s[0])
        return q


class SoftMax(Module):
    def __init__(self):
         super(SoftMax, self).__init__()
    
    def updateOutput(self, input):
        # start with normalization for numerical stability
        input = np.subtract(input, input.max(axis=1, keepdims=True))
        exponents = np.exp(input)
        self.output = (exponents.T / np.sum(exponents, axis=1)).T

        return self.output
    
    def updateGradInput(self, input, gradOutput):

        input = np.subtract(input, input.max(axis=1, keepdims=True))
        exponents = np.exp(input)
        softmax_forw = (exponents.T / np.sum(exponents, axis=1)).T   
        soft_new_dim = softmax_forw[:,:, np.newaxis]
        derivative = np.vectorize(np.diag, signature='(n)->(n,n)')(softmax_forw) \
                    - np.einsum('ijk, ikc-> ijk', soft_new_dim, soft_new_dim) 
        self.gradInput = np.einsum("ik, ika -> ia", gradOutput, derivative)

        return self.gradInput
    
    def __repr__(self):
        return "SoftMax"


class LogSoftMax(Module):
    def __init__(self):
         super(LogSoftMax, self).__init__()
    
    def updateOutput(self, input):
        input = np.subtract(input, input.max(axis=1, keepdims=True))
        exponents = np.exp(input)
        self.output = np.log((exponents.T / np.sum(exponents, axis=1)).T)
                
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        input = np.subtract(input, input.max(axis=1, keepdims=True))
        exponents = np.exp(input)
        softmax_forw = (exponents.T / np.sum(exponents, axis=1)).T 
        soft_new_dim = softmax_forw[:,:, np.newaxis]
        derivative = np.vectorize(np.diag, signature='(n)->(n,n)')(np.ones(softmax_forw.shape)) \
                            - np.einsum('ijk, ikc-> ijk', np.ones(soft_new_dim.shape), soft_new_dim)                   
        self.gradInput = np.einsum("ik, ika -> ia", gradOutput, derivative)      

        return self.gradInput
    
    def __repr__(self):
        return "LogSoftMax"


class BatchNormalization(Module):
    EPS = 1e-3
    def __init__(self, alpha = 0.9):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = None 
        self.moving_variance = None
        
    def updateOutput(self, input):
        if not self.training:
            mean = self.moving_mean
            var = self.moving_variance
            self.output = (input - mean) / (np.sqrt(var) + self.EPS)
            return self.output
        
        mean = np.mean(input, axis=0)
        var = np.var(input, axis=0)
        self.output = (input - mean) / (np.sqrt(var) + self.EPS)
        
        if self.moving_mean is None:
            self.moving_mean = np.array([mean])
        else:
            self.moving_mean = self.moving_mean* self.alpha + mean * (1 - self.alpha)
        if self.moving_variance is None:
            self.moving_variance = np.array([var])
        else:
            self.moving_variance = self.moving_variance * self.alpha + var * (1 - self.alpha)

        return self.output
    
    def updateGradInput(self, input, gradOutput):
        mean = np.mean(input, axis=0)
        var = np.var(input, axis=0)
        N = gradOutput.shape[0]
        dxhat = gradOutput
        dVar =  np.sum(dxhat * (input - mean), 0) * ((-1/2)*((var + self.EPS))**(-3/2))
        dMean = np.sum(dxhat * -1/np.sqrt(var + self.EPS), 0) + dVar * (np.sum(-2*(input - mean), 0)) / N
        x1 = dxhat * (1/np.sqrt(var + self.EPS)) 
        x2 = dVar * (2 * (input - mean)/ N)
        x3 = dMean * (1 / N)
        dx = x1 + x2 + x3
        self.gradInput = dx
        return self.gradInput
    
    def __repr__(self):
        return "BatchNormalization"

        
class ChannelwiseScaling(Module):
    # addition to the batch_normalization part
    """
       Implements linear transform of input y = \gamma * x + \beta
       where \gamma, \beta - learnable vectors of length x.shape[-1]
    """
    def __init__(self, n_out):
        super(ChannelwiseScaling, self).__init__()

        stdv = 1./np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta = np.random.uniform(-stdv, stdv, size=n_out)
        
        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def updateOutput(self, input):
        # print(input)
        self.output = input * self.gamma + self.beta
        # print(self.output)
        return self.output
        
    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * self.gamma
        return self.gradInput
    
    def accGradParameters(self, input, gradOutput):
        self.gradBeta = np.sum(gradOutput, axis=0)
        self.gradGamma = np.sum(gradOutput*input, axis=0)
    
    def zeroGradParameters(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)
        
    def getParameters(self):
        return [self.gamma, self.beta]
    
    def getGradParameters(self):
        return [self.gradGamma, self.gradBeta]
    
    def __repr__(self):
        return "ChannelwiseScaling"

class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p
        self.mask = None
        
    def updateOutput(self, input):
        if not self.training:
            self.output = input
            return self.output
        if self.p < 1: self.mask = np.random.binomial(n=1, p=1-self.p, size=input.shape) / (1-self.p)
        else: self.mask = np.zeros(input.shape)

        self.output = input * self.mask 
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * self.mask 
        return self.gradInput 
        
    def __repr__(self):
        return "Dropout"