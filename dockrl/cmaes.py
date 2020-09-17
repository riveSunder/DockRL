import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from collections import OrderedDict
from copy import deepcopy
import time

from dockrl.policies import MRNN
from dockrl.dock_env import DockEnv

class CMAES():

    def __init__(self, policy_fn, env_fn=DockEnv, dim_in=7, dim_act=6):

        self.env = env_fn()

        self.population_size = 16
        self.elite_keep = int(self.population_size/4)

        self.dim_in = dim_in
        self.dim_act = dim_act
    
        self.population = [policy_fn(dim_in, dim_act) \
                for ii in range(self.population_size)]

        self.distribution = [np.zeros((self.population[0].num_params)),\
                np.eye(self.population[0].num_params)]
    
        self.total_env_interacts = 0

    def get_agent_action(self, obs, agent_idx):

        obs = obs.reshape(1, self.dim_in)
        action = self.population[agent_idx].forward(obs)
        action = action.detach().numpy().squeeze()

        return action

    
    def get_fitness(self, agent_idx):

        fitness = []
        sum_rewards = []
        total_steps = 0

        self.population[agent_idx].reset()

        obs = self.env.reset()
        done = False
        sum_reward = 0.0
        while not done:
            action = self.get_agent_action(obs, agent_idx)

            if len(action.shape) > 1:
                action = action[0]

            obs, reward, done, info = self.env.step(action)

            sum_reward += reward
            total_steps += 1

        sum_rewards.append(sum_reward)

        fitness = np.sum(sum_rewards)

        return fitness, total_steps

    def update_pop(self, fitness_list):

        sorted_indices = list(np.argsort(fitness_list))
        sorted_indices.reverse()
        sorted_fitness = np.array(fitness_list)[sorted_indices]

        self.elite_pop = []

        elite_params = None
        for jj in range(self.elite_keep):

            self.elite_pop.append(self.population[sorted_indices[jj]])

            if elite_params is None:
                elite_params = self.population[sorted_indices[jj]].get_params()[np.newaxis,:]
            else:
                elite_params = np.append(elite_params,\
                        self.population[sorted_indices[jj]].get_params()[np.newaxis,:],\
                        axis=0)


        temp_params =  np.copy(elite_params)
        
        for kk in range(self.elite_keep, self.population_size):

            temp_params = np.append(temp_params,\
                    self.population[kk].get_params()[np.newaxis,:],\
                    axis=0)

        params_mean = np.mean(elite_params, axis=0)

        covar = np.matmul((elite_params - self.distribution[0]).T,\
                (elite_params - self.distribution[0]))

        covar = np.clip(covar, -1e1, 1e1)

        var = np.mean( (elite_params - self.distribution[0])**2, axis=0)

        covar_matrix = covar # + np.diag(var)

        self.distribution = [params_mean, \
                covar_matrix]

        for ll in range(self.population_size):
            params = np.random.multivariate_normal(self.distribution[0],\
                    self.distribution[1])

            self.population[ll].set_params(params)

    def train(self, max_generations=10):
        
        exp_id = "./logs/exp_log{}.npy".format(int(time.time())) 

        fitness_log = {"max_fitness": [],\
                "mean_fitness": [],\
                "sd_fitness": [],\
                "total_env_interacts": []}

        t0 = time.time()
        for gen in range(max_generations):
            fitness_list = []
            for agent_idx in range(self.population_size):

                fitness, steps = self.get_fitness(agent_idx)
                fitness_list.append(fitness)

                self.total_env_interacts += steps

            max_fit = np.max(fitness_list)
            mean_fit = np.mean(fitness_list)
            sd_fit = np.std(fitness_list)
            t1 = time.time()
            elapsed = t1 - t0
            print("gen {}, fitness mean: {:.2e} +/- {:.2e} s.d. max {:.2e}"\
                    .format(gen, mean_fit, sd_fit, max_fit) )
            print("{:.2f} elapsed, total env. interactions: {}"\
                    .format(elapsed, self.total_env_interacts))


            fitness_log["max_fitness"].append(max_fit)
            fitness_log["mean_fitness"].append(mean_fit)
            fitness_log["sd_fitness"].append(sd_fit)
            fitness_log["total_env_interacts"].append(self.total_env_interacts)

            np.save(exp_id, fitness_log)

            self.update_pop(fitness_list)

if __name__ == "__main__":

    cmaes = CMAES(policy_fn=MRNN, env_fn=DockEnv)

    cmaes.train(max_generations=100)

