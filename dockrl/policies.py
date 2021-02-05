import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from collections import OrderedDict
from functools import reduce

class Params():

    def __init__(self, dim_in=7, dim_act=6):
        
        self.dim_act = dim_act

        self.init_params()

    def init_params(self):

        self.params = np.random.randn(self.dim_act)
        self.num_params = self.dim_act

    def forward(self, obs):
        return self.get_params()

    def get_params(self):
        return self.params

    def set_params(self, params):
        assert params.shape == self.params.shape

        self.params = params 

    def reset(self):
        pass


class GraphNN(nn.Module):

    def __init__(self, dim_in=7, dim_act=6):
        super(GraphNN, self).__init__()
        
        self.ligand_dim = dim_in
        self.dim_h = 8
        self.dim_act = dim_act
        # This is a guesstimate based on: 
        # https://pymolwiki.org/index.php/Displaying_Biochemical_Properties
        self.bond_cutoff = 3.6
        self.number_updates = 8
        self.dropout_rate = 1. / self.dim_h

        self.initialize_gnn()
        self.reset()

        my_params = self.get_params()
        self.num_params = my_params.shape[0]

    def initialize_gnn(self):

        # vertices MLP, with 8 element key and query vectors for self-attention
        self.model = nn.Sequential(\
                nn.Linear(self.ligand_dim, self.dim_h),\
                nn.LeakyReLU(),\
                nn.Linear(self.dim_h, self.dim_h),\
                nn.LeakyReLU(),\
                nn.Linear(self.dim_h, self.ligand_dim + 8 + 8)
                )

        self.encoder = nn.Sequential(\
                nn.Linear(2*self.ligand_dim, self.dim_h),\
                nn.LeakyReLU(),
                nn.Dropout(p=self.dropout_rate)\
                )

        self.decoder = nn.Sequential(\
                nn.Linear(self.dim_h, self.ligand_dim),\
                )

        self.action_layer = nn.Sequential(\
                nn.Linear(self.ligand_dim, self.dim_h),\
                nn.LeakyReLU(),\
                nn.Linear(self.dim_h, self.dim_act),\
                nn.Sigmoid()
                )
        
    def get_distance(self, node_0, node_1):

        return torch.sum(torch.sqrt(torch.abs(node_0 - node_1)**2))

    def build_graph(self, x):

        self.graph = torch.zeros(x.shape[0],x.shape[0])

        for ii in range(x.shape[0]):
            node_ii = x[ii, 0:3]
            for jj in range(x.shape[0]):
                node_jj = x[jj, 0:3]

                distance = self.get_distance(node_ii, node_jj)
                if distance <= self.bond_cutoff:
                    self.graph[ii, jj] = 1.0
                
        self.graph = self.graph * (1 - torch.eye(self.graph.shape[0]))

    def forward(self, x, return_codes=False, template=None):

        if type(x) != torch.Tensor:
            x = torch.Tensor(x)

        if template is not None:
            self.build_graph(template.detach())
        else:
            self.build_graph(x.detach())
        
        new_graph = torch.Tensor() #torch.zeros_like(x)
        codes = torch.Tensor() #torch.zeros(x.shape[0], self.dim_h)
        temp_input = [torch.Tensor()] 
        #orch.Tensor() #torch.zeros(x.shape[0], self.dim_h+8+8)

        for kk in range(x.shape[0]):
            # loop through nodes for each node
            for ll in range(x.shape[0]):
                if self.graph[kk,ll]:
                    temp_input[-1] = torch.cat([temp_input[-1],\
                            self.model(x[ll]).unsqueeze(0)])

            keys = temp_input[-1][:,-16:-8]
            queries = temp_input[-1][:,-8:]

            attention = torch.zeros(1, keys.shape[0])

            for mm in range(keys.shape[0]):
                attention[:, mm] = torch.matmul(queries[mm], keys[mm].T)

            attention = torch.softmax(attention, dim=1)

            my_input = torch.sum(attention.T \
                    * temp_input[-1][:,:self.ligand_dim],dim=0)
            my_input = torch.cat([x[kk], my_input])

            #this is where the cell gating would happen (TODO)
            codes = torch.cat([codes, self.encoder(my_input).unsqueeze(0)])

            new_graph = torch.cat([new_graph, self.decoder(codes[-1]).unsqueeze(0)])


        if return_codes:
            return codes, new_graph
        else:
            return new_graph


    def get_actions(self, x):

        if type(x) != torch.Tensor:
            x = torch.Tensor(x)

        my_template = x

        for ii in range(self.number_updates):
            x = self.forward(x, template=my_template)

        x = torch.mean(x, dim=0)

        x = self.action_layer(x)

        return x

    def get_params(self):
        params = np.array([])

        for param in self.model.named_parameters():
            params = np.append(params, param[1].detach().numpy().ravel())

        for param in self.encoder.named_parameters():
            params = np.append(params, param[1].detach().numpy().ravel())

        for param in self.decoder.named_parameters():
            params = np.append(params, param[1].detach().numpy().ravel())

        for param in self.action_layer.named_parameters():
            params = np.append(params, param[1].detach().numpy().ravel())

        return params

    def set_params(self, my_params):

        if my_params is None:
            my_params = self.init_mean + torch.randn(self.num_params) * torch.sqrt(torch.tensor(self.var))

        param_start = 0
        for name, param in self.model.named_parameters():

            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.Tensor(\
                    my_params[param_start:param_stop].reshape(param.shape)))

        for name, param in self.encoder.named_parameters():

            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.Tensor(\
                    my_params[param_start:param_stop].reshape(param.shape)))

        for name, param in self.decoder.named_parameters():

            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.Tensor(\
                    my_params[param_start:param_stop].reshape(param.shape)))

        for name, param in self.action_layer.named_parameters():

            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.Tensor(\
                    my_params[param_start:param_stop].reshape(param.shape)))

    def reset(self):
        # initialize using gated cell states here later (maybe)
        pass

class MRNN(nn.Module):
    def __init__(self, dim_in=6, dim_act=5):
        super(MRNN, self).__init__()

        self.dim_in = dim_in
        self.dim_act = dim_act
        self.dim_h = 8

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
