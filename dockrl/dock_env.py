import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import os 

class DockEnv():

    def __init__(self):

        self.receptors_dir = os.listdir("./data/receptors/")
        self.ligands_dir = os.listdir("./data/ligands/")
        
        self.ligand = None
        self.receptor = None
        self.exhaustiveness = 1
        self.max_steps = 1


    def run_docking(self, action=None, idx=None):

        assert self.ligand is not None, "should be unreachable, call env.reset() first"

        if idx is None:
            obj_filename = "dock.score"
            output_filename =  "./output/{}-redocking.pdbqt".format(self.ligand[0:4]) 
        else:
            obj_filename = "dock{}.score".format(idx)
            output_filename =  "./output/{}-redocking{}.pdbqt"\
                    .format(self.ligand[0:4], idx) 

        if action is None:
            my_command = "./smina.static"\
                    +" -r ./data/receptors/{}".format(self.receptor) \
                    + " -l ./data/ligands/{}".format(self.ligand) \
                    + " --autobox_ligand {}".format(self.ligand) \
                    + " --autobox_add 4 --exhaustiveness {}".format(self.exhaustiveness) \
                    + " -o {}".format(output_filename)\
                    + " --cpu 3 -q" 
        else:
            f = open(obj_filename,'w')

            my_scoring_weights = "{:.8f} gauss(o=0,_w=0.5,_c=8)\n".format(action[0])\
                    + "{:.8f} gauss(o=3,_w=2,_c=8)\n".format(action[1])\
                    +"{:.8f} hydrophobic(g=0.5,_b=1.5,_c=8)\n".format(action[2])\
                    +"{:.8f} non_dir_h_bond(g=-0.7,_b=0,_c=8)\n".format(action[3])\
                    + "{:.8f} repulsion(o=0,_c=8)\n".format(action[4])\
                    + "{:.8f} num_tors_div".format(action[5])

            
            f.write(my_scoring_weights)
            f.close()

            my_command = "./smina.static --custom_scoring {}".format(obj_filename)\
                    + " -r ./data/receptors/{}".format(self.receptor) \
                    + " -l ./data/ligands/{}".format(self.ligand) \
                    + " --autobox_ligand {}".format(self.ligand) \
                    + " --autobox_add 4 --exhaustiveness {}".format(self.exhaustiveness) \
                    + " -o {}".format(output_filename)\
                    + " --cpu 3 -q" 

        #import pdb; pdb.set_trace()
        os.system(my_command)

    
    def get_rmsd(self, worker_idx=None):

        if worker_idx is None:
            redock_fn = "./output/{}-redocking.pdbqt".format(self.ligand[0:4])
        else:
            redock_fn = "./output/{}-redocking{}.pdbqt".format(self.ligand[0:4], worker_idx)

        f = open(redock_fn, 'r')
        f_gt = open("./data/ligands/{}".format(self.ligand), 'r')
    
        stop = False
        rsd = 0.0
        count = 0

        gt = f_gt.readline().split()
        
        comp = f.readline().split() 

        while ("ATOM" not in gt) or ('1' not in gt):
            gt = f_gt.readline().split()
        while ("ATOM" not in comp) or ('1' not in comp):
            comp = f.readline().split()


        while not stop:

            gt = f_gt.readline().split()
            comp = f.readline().split()

            if len(gt) >= 12:
                count += 1
                coords_gt = np.array([float(elem) for elem in gt[5:8]])
                coords_comp = np.array([float(elem) for elem in comp[5:8]])

                rsd += np.sum(np.sqrt((coords_gt - coords_comp)**2))

            if count > 0 and "TORSDOF" in gt:
                stop = True

        rmsd = rsd / count

        f.close()
        f_gt.close()

        return rmsd

    def get_default_rmsd(self):

        rmsd = 0.0
        num_docks = 50
        for ii in range(num_docks):
            
            _ = self.reset()
            self.run_docking(action=None)

            rmsd += self.get_rmsd()

        rmsd /= num_docks

        print("Average rmsd over {} runs with default score weighting  = {:.3f}"\
                .format(num_docks, rmsd))

        return rmsd

    def get_esben_rmsd(self):


        rmsd = 0.0
        num_docks = 50

        action = np.array([-0.0460161,\
                -0.000384274,\
                -0.00812176,\
                -0.431416,\
                0.366584,\
                0.0])
        for ii in range(num_docks):
            _ = self.reset()
            self.run_docking(action)

            rmsd += self.get_rmsd()

        rmsd /= num_docks

        print("Average rmsd over {} runs with weighting from Esben et al. 2016 = {:.3f}"\
                .format(num_docks, rmsd))

        return rmsd


    def step(self, action, worker_idx=None):

        assert self.ligand is not None, "Must call env.reset() before env.step(action)"

        self.run_docking(action, idx=worker_idx)

        rmsd = self.get_rmsd(worker_idx=worker_idx)

        
        reward = - rmsd
        # regularization
        l1_reg = 1e-3
        l2_reg = 1e-3
        reward -= l1_reg * np.sum(np.abs(action)) + l2_reg * np.sum(np.abs(action**2))


        obs = np.append(action, reward)

        self.steps += 1

        if self.steps < self.max_steps:
            done = False
        else:
            done = True

        info = {"rmsd": rmsd}

        return obs, reward, done, info 
        
    def reset(self):

        self.steps = 0

        self.ligand = np.random.choice(self.ligands_dir, \
                p=[1/len(self.ligands_dir) for elem in self.ligands_dir])
        
        for receptor in self.receptors_dir:
            if self.ligand[0:4] in receptor:
                self.receptor = receptor
                break


        if (0):
            action = np.array([-0.0460161,\
                    -0.000384274,\
                    -0.00812176,\
                    -0.431416,\
                    0.366584,\
                    0.0])

            self.run_docking(action)
            rmsd = self.get_rmsd() 

            obs = np.append(rmsd, action)
        else:
            obs = np.zeros((7))

        return obs


if __name__ == "__main__":

    env = DockEnv()

    obs = env.reset()
    
    done = False
    reward_sum = 0.0

    env.get_esben_rmsd()
    env.get_default_rmsd()

    import pdb; pdb.set_trace()
    while not done:
        obs, reward, done, info = env.step(np.random.randn(6))

        reward_sum += reward

    print("cumulative reward: {}".format(reward_sum))

