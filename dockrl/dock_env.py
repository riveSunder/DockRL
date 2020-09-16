import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import os 

class DockingEnv():

    def __init__(self):

        self.receptors_dir = os.listdir("./data/receptors/")
        self.ligands_dir = os.listdir("./data/ligands/")
        
        self.ligand = None
        self.receptor = None
        self.exhaustiveness = 1
        self.max_steps = 16


    def run_docking(self, action=None):

        assert self.ligand is not None, "should be unreachable, call env.reset() first"

        if action is None:
            my_command = "./smina.static"\
                    +" -r ./data/receptors/{}".format(self.receptor) \
                    + " -l ./data/ligands/{}".format(self.ligand) \
                    + " --autobox_ligand {}".format(self.ligand) \
                    + " --autobox_add 4 --exhaustiveness {}".format(self.exhaustiveness) \
                    + " -o ./output/{}-redocking.pdbqt".format(self.ligand[0:4])
        else:
            with open("dock.score",'w') as f:
                my_scoring_weights = "{:.8f} gauss(o=0,_w=0.5,_c=8)\n".format(action[0])\
                        + "{:.8f} gauss(o=3,_w=2,_c=8)\n".format(action[1])\
                        +"{:.8f} hydrophobic(g=0.5,_b=1.5,_c=8)\n".format(action[2])\
                        +"{:.8f} non_dir_h_bond(g=-0.7,_b=0,_c=8)\n".format(action[3])\
                        + "{:.8f} repulsion(o=0,_c=8)\n".format(action[4])\
                        + "{:.8f} num_tors_div".format(action[5])

                f.write(my_scoring_weights)
                f.close()

            my_command = "./smina.static --custom_scoring dock.score"\
                    + " -r ./data/receptors/{}".format(self.receptor) \
                    + " -l ./data/ligands/{}".format(self.ligand) \
                    + " --autobox_ligand {}".format(self.ligand) \
                    + " --autobox_add 4 --exhaustiveness {}".format(self.exhaustiveness) \
                    + " -o ./output/{}-redocking.pdbqt".format(self.ligand[0:4])

        os.system(my_command)

    
    def get_rmsd(self):

        f = open("./output/{}-redocking.pdbqt".format(self.ligand[0:4]), 'r')
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

            if len(gt) == 12:
                count += 1
                coords_gt = np.array([float(elem) for elem in gt[5:8]])
                coords_comp = np.array([float(elem) for elem in comp[5:8]])

                rsd += np.sum(np.sqrt((coords_gt - coords_comp)**2))

            if count > 0 and len(gt) < 12:
                stop = True

        rmsd = rsd / count

        return rmsd

    def step(self, action):

        assert self.ligand is not None, "Must call env.reset() before env.step(action)"

        self.run_docking(action)

        rmsd = self.get_rmsd()

        reward = 2.0 - rmsd

        obs = np.append(action, reward)

        self.steps += 1

        if self.steps >= self.max_steps:
            done = True
        else:
            done = False

        info = {"msg": "thanks for looking at info"}

        return obs, reward, done, info 
        
    def reset(self):

        self.steps = 0

        self.ligand = np.random.choice(self.ligands_dir, \
                p=[1/len(self.ligands_dir) for elem in self.ligands_dir])
        
        for receptor in self.receptors_dir:
            if self.ligand[0:4] in receptor:
                self.receptor = receptor
                break

        self.run_docking()

        rmsd = self.get_rmsd() 
        obs = np.append(rmsd, np.zeros((5)))

        return obs


if __name__ == "__main__":

    env = DockingEnv()

    obs = env.reset()
    
    done = False
    reward_sum = 0.0
    while not done:
        obs, reward, done, info = env.step(np.random.randn(6))

        reward_sum += reward

    print("cumulative reward: {}".format(reward_sum))



