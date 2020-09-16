import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from collections import OrderedDict

class MRNN(nn.Module):

    def __init__(self, dim_in=6, dim_act=5):
        super(MRNN, self).__init__()

        self.dim_in = dim_in
        self.dim_act = dim_act
        self.dim_h = 32

        self.init_params()


    def init_params(self):

        self.g = nn.Sequential(OrderedDict([\
                ("g", nn.Linear(self.dim_h+self.dim_in, self.dim_h)),\
                ("act_g", nn.Sigmoid())]))

        self.j = nn.Sequential(OrderedDict([\
                ("j", nn.Linear(self.dim_h+self.dim_in, self.dim_h)),\
                ("act_j", nn.Tanh())]))

        self.w_h2y = nn.Sequential(OrderedDict([\
                ("w_h2y", nn.Linear(self.dim_h, self.dim_act))]))

        self.cell_state = torch.zeros((1,self.dim_h))

        self.num_params = self.get_params().shape[0]
    
    def forward(self, x):
        
        x = torch.Tensor(x)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        x = torch.cat((self.cell_state, x), axis=-1)

        g_out = self.g(x) 

        j_out = (1.0 - g_out) * self.j(x)

        self.cell_state = g_out * self.cell_state + j_out

        y = self.w_h2y(self.cell_state) 

        return y
        
    def get_action(self, x):

        act = self.forward(x)
        return act.detach().cpu().numpy()

    def get_params(self):
        params = np.array([])

        for param in self.g.named_parameters():
            params = np.append(params, param[1].detach().numpy().ravel())

        for param in self.j.named_parameters():
            params = np.append(params, param[1].detach().numpy().ravel())

        for param in self.w_h2y.named_parameters():
            params = np.append(params, param[1].detach().numpy().ravel())

        return params

    def set_params(self, my_params):

        if my_params is None:
            my_params = self.init_mean + torch.randn(self.num_params) * torch.sqrt(torch.tensor(self.var))

        param_start = 0
        for name, param in self.g.named_parameters():

            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.Tensor(\
                    my_params[param_start:param_stop].reshape(param.shape)))

        for name, param in self.j.named_parameters():

            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.Tensor(\
                    my_params[param_start:param_stop].reshape(param.shape)))

        for name, param in self.w_h2y.named_parameters():

            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.Tensor(\
                    my_params[param_start:param_stop].reshape(param.shape)))

    def reset(self):
        self.cell_state *= 0. 


if __name__ == "__main__":

    mrnn = MRNN()

    temp = mrnn.forward(np.random.randn(1,6))
    print(temp)
