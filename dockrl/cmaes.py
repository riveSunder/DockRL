import time
import argparse

import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from collections import OrderedDict
from copy import deepcopy

from dockrl.policies import MRNN, Params, GraphNN
from dockrl.dock_env import DockEnv

import pdb
import os
import sys
import subprocess
from mpi4py import MPI
comm = MPI.COMM_WORLD

class CMAES():

    def __init__(self, policy_fn, env_fn=DockEnv, num_workers=0, \
            pop_size=64, dim_in=7, dim_act=6):

        self.env = env_fn()

        self.num_workers = num_workers
        self.population_size = pop_size
        self.elite_keep = int(self.population_size/8)

        self.dim_in = dim_in
        self.dim_act = dim_act
    
        self.population = [policy_fn(dim_in, dim_act) \
                for ii in range(self.population_size)]

        self.lr = 1.e0
        self.distribution = [np.zeros((self.population[0].num_params)),\
                0.1*np.eye(self.population[0].num_params)]
    
        self.total_env_interacts = 0

        self.champions = [None] * self.elite_keep
        self.champion_fitness = [-float("Inf")] * self.elite_keep

    def get_agent_action(self, obs, agent_idx, elite=False):

        if elite:
            action = self.champions[agent_idx].get_actions(obs)
        else:
            action = self.population[agent_idx].get_actions(obs)

        action = action.detach().numpy().squeeze()

        return action

    
    def get_fitness(self, agent_idx, worker_idx=None, epds=11):

        fitness = []
        sum_rewards = []
        rmsd = []
        total_steps = 0

        for epd in range(epds):
            self.population[agent_idx].reset()

            obs = self.env.reset(sample_idx = epd % 11)
            done = False
            sum_reward = 0.0
            while not done:
                action = self.get_agent_action(obs, agent_idx)

                if len(action.shape) > 1:
                    action = action[0]

                obs, reward, done, info = self.env.step(action, worker_idx)

                sum_reward += reward
                total_steps += 1
                rmsd.append(info["rmsd"])

            sum_rewards.append(sum_reward)

        fitness = np.mean(sum_rewards)

        if np.isnan(fitness):
            print("warning, nan detected (!)")
            pass

        info["rmsd"] = np.mean(rmsd)

        return fitness, total_steps, info

    def update_pop(self, fitness_list):

        sorted_indices = list(np.argsort(fitness_list))
        sorted_indices.reverse()
        sorted_fitness = np.array(fitness_list)[sorted_indices]

        self.elite_pop = []

        elite_params = None
        for jj in range(self.elite_keep):

            self.elite_pop.append(self.population[sorted_indices[jj]])

            if sorted_fitness[jj] > self.champion_fitness[jj]:

                #self.champions.insert(jj, deepcopy(self.elite_pop[-1]))
                self.champions.insert(jj, self.population[sorted_indices[jj]])
                self.champion_fitness.insert(jj, sorted_fitness[jj])

                _ = self.champion_fitness.pop()
                _ = self.champions.pop()

            if elite_params is None:
                elite_params = self.population[sorted_indices[jj]].get_params()[np.newaxis,:]
            else:
                elite_params = np.append(elite_params,\
                        self.population[sorted_indices[jj]].get_params()[np.newaxis,:],\
                        axis=0)


        temp_params =  np.copy(elite_params)
        
        for kk in range(self.population_size):

            temp_params = np.append(temp_params,\
                    self.population[kk].get_params()[np.newaxis,:],\
                    axis=0)

        params_mean = np.mean(elite_params, axis=0)
        pop_mean = np.mean(temp_params, axis=0)

        covar = (1./self.elite_keep) *  np.matmul((elite_params - self.distribution[0]).T,\
                (elite_params - self.distribution[0]))

        if (0): #self.check_pd(covar):
            print("covariance is not positive semi-definite, finding nearest pd matrix")
            covar = self.get_nearest_pd(covar)

        var =  np.mean((elite_params - self.distribution[0])**2, axis=0)

        covar_matrix = covar  

        params_mean = self.lr * params_mean + (1-self.lr) * self.distribution[0]
        covar = self.lr * covar + (1-self.lr) * self.distribution[1]

        params_mean = np.clip(params_mean, -5e0, 5e0)
        covar = np.clip(covar, -5e0, 5e0)

        #covar = np.eye(covar.shape[0]) * 1e-1
        self.distribution = [params_mean, \
                covar_matrix]

        use_elitism = True

        for ll in range(self.population_size):

            if use_elitism and ll < self.elite_keep:
                self.population[ll].set_params(self.champions[ll].get_params())
            else:

                params = np.random.multivariate_normal(self.distribution[0],\
                        self.distribution[1])

                self.population[ll].set_params(params)


    def mpi_fork(self, n):
        """
        relaunches the current script with workers
        Returns "parent" for original parent, "child" for MPI children
        (from https://github.com/garymcintire/mpi_util/)
        via https://github.com/google/brain-tokyo-workshop/tree/master/WANNRelease
        """
        global num_worker, rank
        if n<=1:
            print("if n<=1")
            num_worker = 0
            rank = 0
            return "child"

        if os.getenv("IN_MPI") is None:
            env = os.environ.copy()
            env.update(\
                    MKL_NUM_THREADS="1", \
                    OMP_NUM_THREAdS="1",\
                    IN_MPI="1",\
                    )
            print( ["mpirun", "-np", str(n), sys.executable] + sys.argv)
            subprocess.check_call(["mpirun", "-np", str(n), sys.executable] \
                +['-u']+ sys.argv, env=env)

            return "parent"
            rank = 0
        else:
            num_worker = comm.Get_size()
            rank = comm.Get_rank()
            return "child"


    def train(self, max_generations=10):

        my_num_workers = self.num_workers

        if self.mpi_fork(my_num_workers) == "parent":
            os._exit(0)

        if rank == 0:
            self.mantle(max_generations)
        else:
            self.arm(max_generations)

    def check_pd(self, my_matrix):

        try:
            temp = np.linalg.cholesky(my_matrix)
            return True
        except:
            return False

    def get_nearest_pd(self, my_matrix):

        temp_matrix = (my_matrix + my_matrix.T) / 2
        _, s, v = np.linalg.svd(temp_matrix)

        hermitian = np.dot(v.T, np.dot(np.diag(s), v))

        matrix_2 = (temp_matrix + hermitian) / 2

        matrix_3 = (matrix_2 + matrix_2.T) / 2

        if self.check_pd(matrix_3):
            return matrix_3

        eye_matrix = np.eye(my_matrix.shape[0])

        count = 1
        spacing = np.spacing(np.linalg.norm(my_matrix))


        while not(self.check_pd(matrix_3)):
            min_eig = np.min(np.real(np.linalg.eigvals(matrix_3)))
            matrix_3 += eye_matrix * (-min_eig * count**2 + spacing )
            count += 1

        return matrix_3

    def mantle(self, max_generations=10):
        
        self.exp_id = "./logs/exp_log{}.npy".format(int(time.time())) 

        fitness_log = {"max_fitness": [],\
                "mean_fitness": [],\
                "sd_fitness": [],\
                "total_env_interacts": [],\
                "rmsd_mean": [],\
                "rmsd_min": [],\
                "rmsd_sd": [],\
                "generation": []}


        t0 = time.time()
        rmsd = []

        for gen in range(max_generations+1):

            np.save("distribution0_{}.npy".format(self.exp_id[7:-4]),\
                    self.distribution[0])#  , self.champions, self.champion_fitness)
            
            np.save("distribution1_{}.npy".format(self.exp_id[7:-4]),\
                    self.distribution[1])#  , self.champions, self.champion_fitness)



            # send parameters to workers
            # but if num_workers == 0, just do it all in the mantle process

            if self.num_workers > 0:

                # break pop into pieces and send to workers
                subpop_size = int(self.population_size / (self.num_workers-1))
                pop_remainder = self.population_size % (self.num_workers-1)
                pop_left = self.population_size


                batch_end = 0 
                extras = 0
                for cc in range(1, self.num_workers):
                    batch_size = min(subpop_size, pop_left)

                    if pop_remainder:
                        batch_size += 1
                        pop_remainder -= 1
                        extras += 1

                    batch_start = batch_end
                    batch_end = batch_start + batch_size

                    params_list = [my_agent.get_params() \
                            for my_agent in self.population[batch_start:batch_end]]

                    comm.send(params_list, dest=cc)

            if gen > 0:
                # receive parameters from workers (skip on first pass)

                # update population (skip on first pass) 
                self.update_pop(fitness_list)

            if self.num_workers > 0:
                # receive fitness scores from workers
                fitness_list = []
                info_list = []
                rmsd = []
                total_steps = 0
                pop_left = self.population_size

                for cc in range(1, num_worker):

                    fit = comm.recv(source=cc)
                    fitness_list.extend(fit[0])

                    total_steps += fit[1]
                    info_list.extend(fit[2])
                    rmsd.extend([elem["rmsd"] for elem in fit[2]])

                self.total_env_interacts += total_steps


            else:

                fitness_list = []
                for agent_idx in range(self.population_size):

                    fitness, steps, info = self.get_fitness(agent_idx)
                    fitness_list.append(fitness)

                    rmsd.append(info["rmsd"])

                    self.total_env_interacts += steps

            t1 = time.time()
            elapsed = t1 - t0

            if True in [np.isnan(elem) for elem in fitness_list]:
                print("warning, nan in fitness_list")
                fitness_list = [-float("Inf") if np.isnan(elem) else elem for elem in fitness_list]

            max_fit = np.max(fitness_list)
            mean_fit = np.mean(fitness_list)
            sd_fit = np.std(fitness_list)

            print("gen {}, fitness mean: {:.2e} +/- {:.2e} s.d. max {:.2e}"\
                    .format(gen, mean_fit, sd_fit, max_fit) )
            print("{:.2f} elapsed, total env. interactions: {}"\
                    .format(elapsed, self.total_env_interacts))


            fitness_log["max_fitness"].append(max_fit)
            fitness_log["mean_fitness"].append(mean_fit)
            fitness_log["sd_fitness"].append(sd_fit)
            fitness_log["total_env_interacts"].append(self.total_env_interacts)
            fitness_log["rmsd_min"].append(np.min(rmsd))
            fitness_log["rmsd_mean"].append(np.mean(rmsd))
            fitness_log["rmsd_sd"].append(np.std(rmsd))
            fitness_log["generation"].append(gen)

            np.save(self.exp_id, fitness_log)

        for cc in range(self.num_workers):
            comm.send(0, dest=cc)


    def arm(self, max_generations):

        while True:
            params_list = comm.recv(source=0)

            if params_list == 0:
                print("worker {} shutting down".format(rank))
                break
            
            if self.population_size < len(params_list):
                print("This should be unreachable (pop_size)")

                self.population = [policy_fn(dim_in, dim_act) \
                        for ii in range(self.population_size)]

            self.population_size = len(params_list)

            self.population = self.population[:self.population_size] 

            fitness_sublist = []
            info_sublist = []
            total_substeps = 0
            for dd in range(self.population_size):
                self.population[dd].set_params(params_list[dd])

                fitness, steps, info = self.get_fitness(dd, worker_idx=rank)

                fitness_sublist.append(fitness)
                info_sublist.append(info)

                total_substeps += steps

            comm.send([fitness_sublist, total_substeps, info_sublist], dest=0)
            

    def evaluate_rmsd(self, mode="test", idx=0):
        
        rmsd = 0.0
        num_docks = 16

        num_samples = len(self.env.ligands_test_dir) \
                if mode=="test" else len(self.env.ligands_dir)

        for ii in range(num_docks):

            obs = self.env.reset(test=(mode == "test"), sample_idx=ii % num_samples)
            action = self.get_agent_action(obs, idx, elite=True)
            obs, reward, done, info = self.env.step(action)
            #action = self.get_agent_action(obs, 0, elite=True)

            self.env.run_docking(action)

            rmsd_temp = self.env.get_rmsd()
            rmsd += rmsd_temp 

        rmsd /= num_docks

        print("Average rmsd over {} runs with cmaes optimized weights = {:.3f}"\
                .format(num_docks, rmsd))
        print("Scoring function weights:  \n", action)
        

        return rmsd

class DirectCMAES(CMAES):

    def __init__(self, policy_fn=Params, env_fn=DockEnv, \
            num_workers=0, pop_size=10, dim_in=7, dim_act=6):
        super(DirectCMAES, self).__init__(policy_fn, env_fn, num_workers,\
                pop_size, dim_in, dim_act)


    def get_agent_action(self, obs, agent_idx, elite=False):

        if elite:
            action = self.champions[0].forward(obs)
            #self.elite_pop[agent_idx].forward(obs)
        else:
            action = self.population[agent_idx].forward(obs)

        return action

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Optimization Parameters")
    parser.add_argument("-c", "--cpu", type=int, default=0,\
            help="number of cpu workers")
    parser.add_argument("-p", "--population_size", type=int, default=10,\
            help="size of evo population")
    parser.add_argument("-g", "--max_generations", type=int, default=1,\
            help="total number of generations to train")
    parser.add_argument("-pi", "--policy", type=str, default="Params",\
            help="which policy to use")

    

    args = parser.parse_args()

    num_workers = args.cpu
    pop_size = args.population_size
    max_generations= args.max_generations

    #my_policy_fn = Params
    if "graphnn" in args.policy.lower():
        my_policy_fn = GraphNN
        my_cmaes = CMAES
    elif "params" in args.policy.lower():
        my_policy_fn = Params
        my_cmaes = DirectCMAES

    cmaes = my_cmaes(policy_fn=my_policy_fn, env_fn=DockEnv, num_workers=num_workers,\
            pop_size=pop_size, dim_in=7, dim_act=6)
    cmaes.train(max_generations=max_generations)
    
    
    if(1):
        if rank == 0:
            rmsd_default_train = cmaes.env.get_default_rmsd(mode="train")
            rmsd_esben_train = cmaes.env.get_bjerrum_rmsd(mode="train")
            optim0_rmsd_train = cmaes.evaluate_rmsd(mode="train", idx=0)
            optim1_rmsd_train = cmaes.evaluate_rmsd(mode="train", idx=1)
            optim2_rmsd_train = cmaes.evaluate_rmsd(mode="train", idx=2)
            optim3_rmsd_train = cmaes.evaluate_rmsd(mode="train", idx=3)

            print("test")
            rmsd_default_test = cmaes.env.get_default_rmsd(mode="test")
            rmsd_esben_test = cmaes.env.get_bjerrum_rmsd(mode="test")
            optim0_rmsd_test = cmaes.evaluate_rmsd(mode="test", idx=0)
            optim1_rmsd_test = cmaes.evaluate_rmsd(mode="test", idx=1)
            optim2_rmsd_test = cmaes.evaluate_rmsd(mode="test", idx=2)
            optim3_rmsd_test = cmaes.evaluate_rmsd(mode="test", idx=3)

            print("(train) rmsd default, Bjerrum's, and cma-es = {:.2e}, {:.2e}, {:.2e}"\
                    .format(rmsd_default_train, rmsd_esben_train, optim0_rmsd_train))
            print("(test) rmsd default, Bjerrum's, and cma-es = {:.2e}, {:.2e}, {:.2e}"\
                    .format(rmsd_default_test, rmsd_esben_test, optim0_rmsd_test))

            arg_str = "cmaes exp. pop size: {}, generations: {}, cpu workers: {} \n\n"\
                    .format(pop_size, max_generations, num_workers)



            with open(cmaes.exp_id[:-4]+"_results.txt",'w') as f:
                f.write(arg_str)

                f.write("(train) rmsd default, Bjerrum's = {:.2f}, {:.2f}\n"\
                    .format(rmsd_default_train, rmsd_esben_train))
                f.write("(test) rmsd default, Bjerrum's = {:.2f}, {:.2f}\n"\
                    .format(rmsd_default_test, rmsd_esben_test))

                f.write("(train) champions 0-3 scores: {:.2f}, {:.2f}, {:.2f}, {:.2f}\n\n"\
                    .format(optim0_rmsd_train, optim1_rmsd_train, \
                            optim2_rmsd_train, optim3_rmsd_train))
                f.write("(test) champions 0-3 scores: {:.2f}, {:.2f}, {:.2f}, {:.2f}\n\n"\
                    .format(optim0_rmsd_test, optim1_rmsd_test, \
                            optim2_rmsd_test, optim3_rmsd_test))
                for ii in range(4):
                    f.write("\nchampion {} params".format(ii))
                    f.write(str(cmaes.population[ii].get_params()))
                    f.write("\n")


    else: 
        print("computer words not working right")
        pass

    
    print("end for rank ", rank)

    #cmaes = CMAES(policy_fn=MRNN, env_fn=DockEnv)
    #cmaes.max_steps = 4
    #cmaes.train(max_generations=200)

