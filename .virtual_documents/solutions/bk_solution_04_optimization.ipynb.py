import torch
import matplotlib.pyplot as plt
import numpy as np

from utils import visualize_optimizer, test_optimizer


class Optimizer:
    """Base class for each optimizer"""
    
    def __init__(self, initial_params):
        # store initial model weights
        self.params = initial_params
    
    def step(self):
        """Updates the weights stored in self.params"""
        raise NotImplementedError()
         
    def zero_grad(self):
        """Torch accumulates gradients, so we need to clear them after every update"""
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()


from typing import List

class GradientDescent(Optimizer):
    
    def __init__(self, initial_params: List[torch.tensor], learning_rate):
        super().__init__(initial_params)
        self.learning_rate = learning_rate
    
    @torch.no_grad()
    def step(self):
        for param in self.params:
            param -= self.learning_rate * param.grad
            # Please note that it's important to change the parameters in-place (-=) so the original tensors are modified


test_optimizer(GradientDescent)


visualize_optimizer(GradientDescent, n_steps=20, learning_rate=0.1, title='Small LR')
visualize_optimizer(GradientDescent, n_steps=15, learning_rate=0.5, title='Large LR')


class Momentum(Optimizer):
    
    def __init__(self, initial_params, learning_rate, gamma):
        super().__init__(initial_params)
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.v = [torch.zeros_like(tensor.detach()) for tensor in initial_params]
    
    @torch.no_grad()
    def step(self):
        for param, v in zip(self.params, self.v):
            v *= self.gamma
            v += self.learning_rate * param.grad
            param -= v


test_optimizer(Momentum)


for lr in np.linspace(0.001, 0.01, 3):
    visualize_optimizer(Momentum, n_steps=100, learning_rate=lr, title=f'LR={lr:.4f}', gamma=0.8)



class Adagrad(Optimizer):
    
    def __init__(self, initial_params, learning_rate, epsilon):
        super().__init__(initial_params)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.r = [torch.zeros_like(tensor.detach()) for tensor in initial_params]
    
    @torch.no_grad()
    def step(self):
        for param, r in zip(self.params, self.r):
            r += param.grad ** 2
            param -= self.learning_rate / torch.sqrt(r + self.epsilon) * param.grad


test_optimizer(Adagrad)


for lr in np.linspace(0.001, 0.01, 3)*100:
    visualize_optimizer(Adagrad, n_steps=50, learning_rate=lr, epsilon=1e-8, title=f'LR={lr:.2f}')


class RMSProp(Optimizer):
    
    def __init__(self, initial_params, learning_rate, gamma, epsilon):
        super().__init__(initial_params)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.r = [torch.zeros_like(tensor.detach()) for tensor in initial_params]
    
    @torch.no_grad()
    def step(self):
        for param, r in zip(self.params, self.r):
            r *= self.gamma
            r += (1 - self.gamma) * param.grad ** 2
            param -= self.learning_rate / torch.sqrt(r + self.epsilon) * param.grad


test_optimizer(RMSProp)


for lr in np.linspace(0.001, 0.01, 3)*10:
    visualize_optimizer(RMSProp, n_steps=100, learning_rate=lr, gamma=0.9, epsilon=1e-8, title=f'LR={lr:.3f}')


class Adam(Optimizer):
   
    def __init__(self, initial_params, learning_rate, beta1, beta2, epsilon):
        super().__init__(initial_params)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = [torch.zeros_like(t.detach()) for t in initial_params]
        self.v = [torch.zeros_like(t.detach()) for t in initial_params]
        self.t = 1
        
    @torch.no_grad()
    def step(self):
        for param, m, v in zip(self.params, self.m, self.v):
            m *= self.beta1
            m += (1 - self.beta1) * param.grad
            m_hat = m / (1 - self.beta1 ** self.t)
            v *= self.beta2
            v += (1 - self.beta2) * param.grad ** 2
            v_hat = v / (1 - self.beta2 ** self.t)
            param -= (self.learning_rate / (torch.sqrt(v_hat) + self.epsilon)) * m_hat
        
        self.t += 1


test_optimizer(Adam)


for lr in np.linspace(0.001, 0.01, 3)*50:
    visualize_optimizer(Adam, n_steps=60, learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8, title=f'LR={lr:.3f}')



