import numpy as np
from .module import Module

class ReLU(Module):
    def __init__(self):
         super(ReLU, self).__init__()
    
    def updateOutput(self, input):
        self.output = np.maximum(input, 0)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput, input > 0)
        return self.gradInput
    
    def __repr__(self):
        return "ReLU"


class LeakyReLU(Module):
    def __init__(self, slope = 0.03):
        super(LeakyReLU, self).__init__()
            
        self.slope = slope
        
    def updateOutput(self, input):
        self.output = np.maximum(0, input) + self.slope * np.minimum(0, input)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput, input > 0) + self.slope * np.multiply(gradOutput, input < 0)
        return self.gradInput
    
    def __repr__(self):
        return "LeakyReLU"


class ELU(Module):
    def __init__(self, alpha = 1.0):
        super(ELU, self).__init__()
        self.alpha = alpha
        
    def updateOutput(self, input):
        self.output = np.maximum(input, 0) + self.alpha * (np.exp(np.minimum(0, input))-1)
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput, input > 0) + self.alpha * np.multiply(np.multiply(gradOutput, input <= 0), np.exp(input))
        return self.gradInput
    
    def __repr__(self):
        return "ELU"


class SoftPlus(Module):
    def __init__(self):
        super(SoftPlus, self).__init__()
    
    def updateOutput(self, input):
        self.output = np.log(1+ np.exp(input))
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput / (1 + np.exp(-input)) 
        return self.gradInput
    
    def __repr__(self):
        return "SoftPlus"