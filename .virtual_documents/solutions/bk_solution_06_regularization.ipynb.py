get_ipython().run_line_magic("load_ext", " autoreload")
get_ipython().run_line_magic("autoreload", " 2")


import matplotlib.pyplot as plt
get_ipython().run_line_magic("matplotlib", " inline")

import torch

from utils import load_fashionmnist, ModelTrainer, show_results, test_dropout, test_bn


torch.manual_seed(44)

train_dataset = load_fashionmnist(train=True, shrinkage=0.01)
test_dataset = load_fashionmnist(train=False, shrinkage=0.1)


n_epochs = 300
learning_rate = 0.05
batch_size = 128

trainer = ModelTrainer(train_dataset, test_dataset, batch_size=batch_size)


model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10)
)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
history = trainer.train(model, optimizer, n_epochs=n_epochs)
show_results(model=history)


class Dropout(torch.nn.Module):
    
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        assert 0 <= p and p < 1, 'p out of range'
        self.bernoulli = torch.distributions.bernoulli.Bernoulli(probs=1-p)
        self.multipler = 1. / (1. - p)
        self.p = p
        
    def forward(self, x):
        if self.training:
#             mask = self.bernoulli.sample(x.size())
            mask = torch.rand_like(x) > self.p
            x = x * mask * self.multipler
        return x


test_dropout(Dropout)


model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    Dropout(0.5),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    Dropout(0.5),
    torch.nn.Linear(256, 10)
)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
dropout_history = trainer.train(model, optimizer, n_epochs=n_epochs)
show_results(model=dropout_history)


model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(256, 10)
)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
dropout_history = trainer.train(model, optimizer, n_epochs=n_epochs)
show_results(model=dropout_history)


class BatchNorm(torch.nn.Module):
    
    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.m = momentum
        self.beta = torch.nn.Parameter(torch.zeros(1,num_features), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.ones(1,num_features), requires_grad=True)
        self.register_buffer(name='mu', tensor=torch.zeros(1,num_features))
        self.register_buffer(name='sigma', tensor=torch.ones(1,num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, keepdim=True, unbiased=False)
            self.mu = (1 - self.m) * self.mu + self.m * mean
            self.sigma = (1 - self.m) * self.sigma + self.m * var
        else:
            mean = self.mu
            var = self.sigma
            
        z = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * z + self.beta


test_bn(BatchNorm)


model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    BatchNorm(256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 256),
    BatchNorm(256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10)
)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
bn_history = trainer.train(model, optimizer, n_epochs=n_epochs)
show_results(model=bn_history)


show_results(vanilla=history, dropout=dropout_history, bn=bn_history, 
             orientation='vertical', accuracy_bottom=0.5, loss_top=1.75)



